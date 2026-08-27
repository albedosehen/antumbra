//! `antumbra-mcp`: the Model Context Protocol server that exposes Antumbra's
//! Penumbra memory to an agent. The runtime surface that lets a client (Claude
//! Code, any MCP host) talk to Antumbra instead of a separate memory service.
//!
//! Two transports. **stdio** (default) serves a single `(tenant, user)` passed
//! on the command line: connect as owner, provision the principal, sign the
//! session in so every tool is engine-isolated. **HTTP** (`--http <addr>`) is
//! the networked, per-request multi-tenant surface: each request carries a
//! signed JWT whose `tenant`/`user` claims become `$auth`, and the server keeps
//! one signed-in session per identity (see [`http`]).

use std::sync::Arc;

use anyhow::Result;
use chrono::Utc;
use clap::Parser;
use rmcp::ServiceExt;

use antumbra_core::ports::Embedder;
use antumbra_core::{Compartment, CompartmentId, TenantId, UserId};
use antumbra_store::repo::{compartment, principal};
use antumbra_store::{ConnectionConfig, Store, EMBED_DIM};

/// The JWT token contract now lives in the shared `antumbra-auth` crate (so the
/// hosted control plane can mint what this server verifies); re-exported here so
/// `crate::auth::…` keeps resolving.
mod auth {
    pub use antumbra_auth::*;
}
mod copal;
mod embed;
mod http;
mod notify;
mod server;
use server::McpServer;

#[derive(Parser)]
#[command(
    name = "antumbra-mcp",
    about = "Antumbra MCP server over the Penumbra memory store",
    version
)]
struct Cli {
    /// SurrealDB url: `surrealkv://./data/antumbra.skv` (persistent),
    /// `mem://` (ephemeral), or `ws://host:8000/rpc`.
    #[arg(long, default_value = "surrealkv://./data/antumbra.skv")]
    url: String,
    /// The workspace (tenant) this stdio server serves. Ignored with `--http`,
    /// where the tenant comes from each request's verified JWT.
    #[arg(long, default_value = "ws:default")]
    tenant: String,
    /// The user this stdio session acts as (the compartment-ownership / sharing
    /// actor). Ignored with `--http`.
    #[arg(long, default_value = "user:default")]
    user: String,
    /// The host/device this session runs on (stamped as memory provenance).
    /// Defaults to the machine name.
    #[arg(long)]
    host: Option<String>,
    /// Serve the networked multi-tenant HTTP surface on this address
    /// (e.g. `0.0.0.0:8081`) instead of stdio. Requires a JWT key.
    #[arg(long)]
    http: Option<String>,
    /// HS256 shared secret for verifying request JWTs (symmetric).
    #[arg(long, env = "ANTUMBRA_JWT_SECRET")]
    jwt_secret: Option<String>,
    /// Path to a PEM RSA public key for verifying request JWTs (RS256).
    #[arg(long)]
    jwt_public_key: Option<std::path::PathBuf>,
    /// Required JWT audience claim (this server's identifier), if set.
    #[arg(long)]
    jwt_audience: Option<String>,
    /// Root username for an authenticated remote SurrealDB (`ws://`). Omit for an
    /// embedded store or an unauthenticated server. The DB credential is the
    /// server's *own* login (it then signs in per request as each tenant); it is
    /// distinct from the JWTs that authenticate clients.
    #[arg(long, env = "ANTUMBRA_DB_USER")]
    db_user: Option<String>,
    /// Root password for the remote SurrealDB.
    #[arg(long, env = "ANTUMBRA_DB_PASS")]
    db_pass: Option<String>,
    /// Enable the autonomous propose trigger: once a user's unorganized inbox
    /// reaches this many memories, a write auto-clusters it into proposed
    /// compartments (reversible; the user curates). Off when unset.
    #[arg(long)]
    auto_propose: Option<usize>,
    /// Enable the autonomous consolidation trigger: when a reinforced memory's
    /// compartment clears the consolidation gate, it graduates into a private
    /// expert on the GPU in the background. Needs `--features models` + a GPU to
    /// actually train. Off when unset.
    #[arg(long, default_value_t = false)]
    auto_consolidate: bool,
    /// Print a long-lived hook token for `--tenant`/`--user`, signed with
    /// `--jwt-secret` (HS256), and exit -- the credential a non-interactive
    /// lifecycle hook presents on the offline / self-hosted tier (the server
    /// otherwise only verifies, never mints).
    #[arg(long)]
    mint_token: bool,
    /// Validity, in days, of a `--mint-token` token.
    #[arg(long, default_value_t = 365)]
    token_ttl_days: u64,
    /// Embed via an OpenAI-compatible `/embeddings` endpoint (e.g. a local
    /// text-embeddings-inference / Ollama server) instead of the built-in
    /// embedder. It must return `EMBED_DIM`-wide vectors. Off when unset.
    #[arg(long, env = "ANTUMBRA_EMBEDDER_URL")]
    embedder_url: Option<String>,
    /// Model name sent to `--embedder-url` (must produce the index dimension).
    #[arg(long, default_value = "all-MiniLM-L6-v2")]
    embedder_model: String,
    /// Optional bearer key for `--embedder-url`.
    #[arg(long, env = "ANTUMBRA_EMBEDDER_KEY")]
    embedder_key: Option<String>,
    /// Rerank hybrid-recall candidates with a cross-encoder via a TEI/Cohere-style
    /// `/rerank` endpoint (the precision stage after RRF). Off when unset.
    #[arg(long, env = "ANTUMBRA_RERANK_URL")]
    rerank_url: Option<String>,
    /// Model name sent to `--rerank-url` (omit for text-embeddings-inference,
    /// which ignores it; Cohere/Jina require it).
    #[arg(long)]
    rerank_model: Option<String>,
    /// Optional bearer key for `--rerank-url`.
    #[arg(long, env = "ANTUMBRA_RERANK_KEY")]
    rerank_key: Option<String>,
    /// Archive each ingested document's ORIGINAL content to a copal file
    /// service (the document of record): bare `host:port` or a full URL base.
    /// The upload happens before any chunk is stored (a dead copal fails the
    /// ingest), and every chunk carries the copal file id + digest. Off when
    /// unset: ingest keeps only the chunks, exactly as before.
    #[arg(long, env = "ANTUMBRA_COPAL_ADDR")]
    copal_addr: Option<String>,
    /// The tenant presented to copal (`x-copal-tenant`, its header auth mode).
    #[arg(long, env = "ANTUMBRA_COPAL_TENANT", default_value = "antumbra")]
    copal_tenant: String,
}

async fn connect(url: &str, db_user: Option<&str>, db_pass: Option<&str>) -> Result<Store> {
    let mut builder = ConnectionConfig::builder()
        .url(url)
        .namespace("antumbra")
        .database("main");
    // Root login for an authenticated remote (`ws://`); embedded/unauthenticated
    // stores need none. The server signs in per request as each tenant on top.
    if let (Some(user), Some(pass)) = (db_user, db_pass) {
        builder = builder.username(user).password(pass);
    }
    let config = builder.build()?;
    Ok(Store::connect(config, EMBED_DIM).await?)
}

/// A credential-less **serving** connection to an already-provisioned remote: it
/// connects without applying the schema and only ever holds a per-request record
/// session, so the engine ACL is enforced (a root connection would bypass it,
/// R-6). Used by the networked HTTP surface for the actual request work.
pub(crate) async fn connect_serving(url: &str) -> Result<Store> {
    let config = ConnectionConfig::builder()
        .url(url)
        .namespace("antumbra")
        .database("main")
        .build()?;
    Ok(Store::connect_without_schema(config, EMBED_DIM).await?)
}

/// The real candle BERT embedder under `--features models`, else the
/// byte-histogram fake. Both produce `EMBED_DIM`-wide vectors.
#[cfg(feature = "models")]
fn make_embedder() -> Result<Box<dyn Embedder>> {
    Ok(Box::new(antumbra_serve::BertEmbedder::load()?))
}

#[cfg(not(feature = "models"))]
fn make_embedder() -> Result<Box<dyn Embedder>> {
    Ok(Box::new(antumbra_core::testing::FixedEmbedder::new(
        EMBED_DIM,
    )))
}

/// Owner-side provisioning for an identity: ensure the principal exists and the
/// user's default (inbox) compartment is present. Idempotent (safe across
/// restarts). The caller must be in **owner mode** (not signed in as a tenant),
/// since it writes the principal/compartment tables. Returns the default
/// compartment id. Shared by both transports.
pub(crate) async fn provision_identity(
    store: &Store,
    tenant: &TenantId,
    user: &UserId,
) -> Result<CompartmentId> {
    principal::provision(store, tenant, user).await?;
    let default_compartment = CompartmentId::new(format!(
        "comp:{}:{}:default",
        tenant.as_str(),
        user.as_str()
    ));
    let exists = compartment::list_owned(store, tenant, user)
        .await?
        .iter()
        .any(|c| c.id == default_compartment);
    if !exists {
        compartment::create(
            store,
            &Compartment::new(
                default_compartment.clone(),
                tenant.clone(),
                user.clone(),
                "default",
                Utc::now(),
            ),
        )
        .await?;
    }
    Ok(default_compartment)
}

/// Build a signed-in stdio session: one connection, provisioned and bound as
/// `(tenant, user)` for the life of the process. (The HTTP transport instead
/// shares one connection across identities (see [`http`]) because an embedded
/// engine is single-writer.)
async fn build_session(
    url: &str,
    db_user: Option<&str>,
    db_pass: Option<&str>,
    tenant: TenantId,
    user: UserId,
    host: String,
    embedder: Arc<dyn Embedder>,
) -> Result<McpServer> {
    let store = connect(url, db_user, db_pass).await?;
    let default_compartment = provision_identity(&store, &tenant, &user).await?;
    store.signin(&tenant, &user).await?;
    // The `answer` tool serves through the routed expert. Build the engine from
    // the session's visible population (one connection; embedded is single-writer).
    let serve = build_serve(&store).await?;
    Ok(McpServer::new(
        store,
        embedder,
        tenant,
        user,
        host,
        default_compartment,
        serve,
    ))
}

/// Build the serving engine the `answer` tool drives: a resident
/// [`MultiAdapterServe`](antumbra_serve::MultiAdapterServe) over the shared base,
/// registered with every expert the `store` can currently see (routing scopes
/// which a session may actually pick). `None` when the population is empty or the
/// build has no model backend. Built once at startup; restart to pick up experts
/// minted afterward.
#[cfg(feature = "models")]
pub(crate) async fn build_serve(
    store: &Store,
) -> Result<Option<Arc<dyn antumbra_core::ports::Serve>>> {
    use antumbra_serve::{MultiAdapterServe, RaftConfig};

    let experts = antumbra_store::repo::expert::list(store).await?;
    if experts.is_empty() {
        return Ok(None);
    }
    let base = experts[0].base_model.clone();
    // Serve the learned mode (greedy + repetition penalty + n-gram block +
    // nucleus), not the training-time exploration draw.
    let cfg = RaftConfig::for_serving(RaftConfig::default().max_new_tokens, 0.0);
    let mut engine = MultiAdapterServe::new(base, cfg);
    for e in &experts {
        engine.register(e.id.clone(), e.artifact_uri.clone());
    }
    Ok(Some(Arc::new(engine)))
}

#[cfg(not(feature = "models"))]
pub(crate) async fn build_serve(
    _store: &Store,
) -> Result<Option<Arc<dyn antumbra_core::ports::Serve>>> {
    Ok(None)
}

fn default_host(explicit: Option<String>) -> String {
    explicit.unwrap_or_else(|| {
        std::env::var("COMPUTERNAME")
            .or_else(|_| std::env::var("HOSTNAME"))
            .unwrap_or_else(|_| "local".into())
    })
}

fn main() -> Result<()> {
    // SurrealDB's engine-enforced ACL subqueries for tenant isolation and memory compartments recurse deep;
    // host the runtime on a large-stack thread so the 1 MB Windows main-thread
    // stack does not overflow (see the matching note in the CLI).
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(|| {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?
                .block_on(run())
        })?
        .join()
        .map_err(|_| anyhow::anyhow!("antumbra-mcp worker thread panicked"))?
}

async fn run() -> Result<()> {
    let cli = Cli::parse();

    // Mint a hook token and exit -- no DB or embedder needed. HS256 only: the
    // server holds the symmetric secret; an RS256 deployment mints via its auth
    // service's private key.
    if cli.mint_token {
        let secret = cli.jwt_secret.as_deref().ok_or_else(|| {
            anyhow::anyhow!(
                "--mint-token needs --jwt-secret (HS256); RS256 tokens are minted by your auth service"
            )
        })?;
        let ttl = std::time::Duration::from_secs(cli.token_ttl_days * 24 * 60 * 60);
        println!(
            "{}",
            auth::mint_hs256(secret.as_bytes(), &cli.tenant, &cli.user, ttl)?
        );
        return Ok(());
    }

    let host = default_host(cli.host);
    // A configured endpoint embeds on the tenant's side (P-1c); otherwise the
    // build-time embedder (candle BERT under `models`, else the byte-histogram
    // fake). Either way the vectors are EMBED_DIM-wide.
    let embedder: Arc<dyn Embedder> = match cli.embedder_url {
        Some(url) => Arc::new(embed::HttpEmbedder::new(
            url,
            cli.embedder_model,
            cli.embedder_key,
        )),
        None => Arc::from(make_embedder()?),
    };

    // Optional cross-encoder rerank stage (P-2). Operator-configured endpoint; the
    // precision stage runs after hybrid recall and degrades to RRF order on error.
    let reranker: Option<Arc<dyn antumbra_core::ports::Reranker>> = cli.rerank_url.map(|url| {
        Arc::new(antumbra_rerank::HttpReranker::new(
            url,
            cli.rerank_model,
            cli.rerank_key,
        )) as Arc<dyn antumbra_core::ports::Reranker>
    });

    // Optional copal document-of-record archive. Operator-configured; absent,
    // ingest keeps only the chunks (the v0 behavior, unchanged).
    let copal: Option<Arc<copal::CopalArchive>> = cli
        .copal_addr
        .as_deref()
        .map(|addr| Arc::new(copal::CopalArchive::new(addr, cli.copal_tenant.clone())));

    if let Some(addr) = cli.http {
        // Networked multi-tenant surface: identity per request from a verified JWT.
        let verifier = build_verifier(&cli.jwt_secret, &cli.jwt_public_key, &cli.jwt_audience)?;
        return http::serve(
            addr,
            cli.url,
            cli.db_user,
            cli.db_pass,
            host,
            embedder,
            verifier,
            cli.auto_propose,
            cli.auto_consolidate,
            reranker,
            copal,
        )
        .await;
    }

    // stdio: one fixed identity for the life of the process.
    let mut service = build_session(
        &cli.url,
        cli.db_user.as_deref(),
        cli.db_pass.as_deref(),
        TenantId::new(cli.tenant),
        UserId::new(cli.user),
        host,
        embedder,
    )
    .await?;
    if let Some(threshold) = cli.auto_propose {
        service = service.with_auto_propose(threshold);
    }
    if cli.auto_consolidate {
        service = service.with_auto_consolidate();
    }
    if let Some(r) = reranker {
        service = service.with_reranker(r);
    }
    if let Some(c) = copal {
        service = service.with_copal_archive(c);
    }
    let running = service
        .serve((tokio::io::stdin(), tokio::io::stdout()))
        .await?;
    running.waiting().await?;
    Ok(())
}

/// Resolve the JWT verifier from the configured key material. RS256 (a PEM
/// public key) takes precedence over an HS256 secret; one is required.
fn build_verifier(
    secret: &Option<String>,
    public_key: &Option<std::path::PathBuf>,
    audience: &Option<String>,
) -> Result<auth::JwtVerifier> {
    let mut v = match (public_key, secret) {
        (Some(pem_path), _) => {
            let pem = std::fs::read(pem_path)?;
            auth::JwtVerifier::rs256_pem(&pem)?
        }
        (None, Some(secret)) => auth::JwtVerifier::hs256(secret.as_bytes()),
        (None, None) => anyhow::bail!(
            "--http needs a JWT key: pass --jwt-secret (HS256) or --jwt-public-key <pem> (RS256)"
        ),
    };
    if let Some(aud) = audience {
        v = v.with_audience(aud);
    }
    Ok(v)
}
