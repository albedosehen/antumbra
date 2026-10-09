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
//!
//! Either way the server logs to stderr, and with `OTEL_EXPORTER_OTLP_ENDPOINT`
//! set it also exports a span per HTTP request and per tool call to that
//! collector over OTLP/HTTP (see [`telemetry`]).

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
mod dependencies;
mod embed;
mod hardware;
mod http;
mod notify;
mod profile;
mod secrets;
mod server;
mod session;
mod telemetry;
mod warmup;
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
    /// The host/device this session runs on: stamped as memory provenance, and
    /// the name this machine registers under in its user's fabric (ADR-0017).
    /// Defaults to the machine name, which inside a container is the container
    /// ID -- ephemeral, so a recreate would register the same machine as a new
    /// node every time. Set it to the real host there.
    #[arg(long, env = "ANTUMBRA_HOST")]
    host: Option<String>,
    /// Serve the networked multi-tenant HTTP surface on this address
    /// (e.g. `0.0.0.0:8081`) instead of stdio. Requires a JWT key.
    #[arg(long)]
    http: Option<String>,
    /// HS256 shared secret for verifying request JWTs (symmetric).
    #[arg(long, env = "ANTUMBRA_JWT_SECRET", hide_env_values = true)]
    jwt_secret: Option<String>,
    /// Read the HS256 secret from this file instead (a Docker secret, a
    /// Kubernetes secret, a Key Vault mount): it then appears in neither the
    /// process arguments nor the environment.
    #[arg(long, env = "ANTUMBRA_JWT_SECRET_FILE", conflicts_with = "jwt_secret")]
    jwt_secret_file: Option<std::path::PathBuf>,
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
    #[arg(long, env = "ANTUMBRA_DB_PASS", hide_env_values = true)]
    db_pass: Option<String>,
    /// Read the root password from this file instead (see --jwt-secret-file).
    #[arg(long, env = "ANTUMBRA_DB_PASS_FILE", conflicts_with = "db_pass")]
    db_pass_file: Option<std::path::PathBuf>,
    /// Enable the autonomous propose trigger: once a user's unorganized inbox
    /// reaches this many memories, a write auto-clusters it into proposed
    /// compartments (reversible; the user curates). Off when unset.
    #[arg(long)]
    auto_propose: Option<usize>,
    /// Retired (ADR-0027): a reinforced memory no longer trains its compartment
    /// into an expert, which taught experts to echo memories. Standing experts
    /// are trained from accepted behaviors by the keeper, on any node that can
    /// train. Still accepted, so a deployment that passes it keeps starting.
    #[arg(long, default_value_t = false, hide = true)]
    auto_consolidate: bool,
    /// The chunk index (ADR-0025), on the networked surface: how many pieces
    /// of a memory to embed at once while cutting memories into the pieces
    /// recall searches. Zero leaves the index as it is; recall then reads whole
    /// memories for anything not in it.
    #[arg(long, env = "ANTUMBRA_CHUNK_IN_FLIGHT", default_value_t = 4)]
    chunk_in_flight: usize,
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
    #[arg(long, env = "ANTUMBRA_EMBEDDER_KEY", hide_env_values = true)]
    embedder_key: Option<String>,
    /// Use the deterministic byte-histogram stand-in instead of a real embedder.
    /// Recall is then NOT semantic (it matches character statistics), so this is
    /// for smoke tests and demos only. Without `--features models`, the server
    /// refuses to start with neither this nor `--embedder-url` rather than
    /// silently degrading recall.
    #[arg(long, env = "ANTUMBRA_FAKE_EMBEDDER", default_value_t = false)]
    fake_embedder: bool,
    /// Rerank hybrid-recall candidates with a cross-encoder via a TEI/Cohere-style
    /// `/rerank` endpoint (the precision stage after RRF). Off when unset.
    #[arg(long, env = "ANTUMBRA_RERANK_URL")]
    rerank_url: Option<String>,
    /// Model name sent to `--rerank-url` (omit for text-embeddings-inference,
    /// which ignores it; Cohere/Jina require it).
    #[arg(long, env = "ANTUMBRA_RERANK_MODEL")]
    rerank_model: Option<String>,
    /// The relevance floor's calibration as `a,b` in
    /// `P(relevant) = sigmoid(a * log10(score) + b)`, for a reranker fitted by
    /// hand. Without it the floor uses the fit for the model the endpoint
    /// serves, or the configured `--rerank-model`, and runs without a floor for
    /// a model nobody has fitted.
    #[arg(long, env = "ANTUMBRA_FLOOR_CALIBRATION")]
    floor_calibration: Option<String>,
    /// Optional bearer key for `--rerank-url`.
    #[arg(long, env = "ANTUMBRA_RERANK_KEY", hide_env_values = true)]
    rerank_key: Option<String>,
    /// Archive each ingested document's ORIGINAL content to a copal file
    /// service (the document of record): bare `host:port` or a full URL base.
    /// The upload happens before any chunk is stored (a dead copal fails the
    /// ingest), and every chunk carries the copal file id + digest. Off when
    /// unset: ingest keeps only the chunks, exactly as before.
    #[arg(long, env = "ANTUMBRA_COPAL_ADDR")]
    copal_addr: Option<String>,
    /// Land every workspace's documents under this ONE copal tenant, named by
    /// the `x-copal-tenant` header (copal `header` auth mode). Omit all three
    /// tenancy args for the default: per-workspace tenancy over header auth,
    /// where each workspace presents ITSELF as the copal tenant (its own
    /// quotas, listings, and search scope). Either way the archive identity is
    /// (workspace, title). At most one of --copal-tenant / --copal-key /
    /// --copal-keys.
    #[arg(long, env = "ANTUMBRA_COPAL_TENANT")]
    copal_tenant: Option<String>,
    /// One `ck1` copal API key for every workspace: shared tenancy under
    /// copal's deployed `keys` auth mode, where the tenant is bound to the
    /// credential (the key's tenant is THE tenant).
    #[arg(long, env = "ANTUMBRA_COPAL_KEY", hide_env_values = true)]
    copal_key: Option<String>,
    /// Path to a JSON file mapping workspace tenant -> `ck1` copal API key
    /// (e.g. `{"ws:acme": "ck1..."}`): per-workspace tenancy under copal's
    /// `keys` auth mode. Loaded once at startup (restart to pick up newly
    /// minted keys); a workspace absent from the map fails its ingest rather
    /// than landing in another tenant.
    #[arg(long, env = "ANTUMBRA_COPAL_KEYS")]
    copal_keys: Option<std::path::PathBuf>,
    /// Which tools this server advertises and serves: `all` (default), `agent`
    /// (the developer-agent profile: recall/store/reinforce/penalize memories,
    /// recall/ingest documents, route, answer), or a comma-separated list of
    /// tool names, in which `agent` expands (`agent,population`). Anything
    /// outside the profile is neither listed nor callable, over JSON-RPC or
    /// the REST shim: less description text in every session, and no operator
    /// action (sharing, revoking, forgetting) one agent call away.
    #[arg(long, env = "ANTUMBRA_TOOLS", default_value = "all")]
    tools: String,
    /// Serve GitHub App webhooks at `POST /github/webhook` (HTTP transport
    /// only), each delivery verified against this HMAC secret (the App's
    /// webhook secret). A merged pull request re-anchors the merged branch's
    /// memories to the merge commit and becomes a memory itself; a deleted
    /// branch marks its memories orphaned. Needs --github-tenant or
    /// --github-repos to say which workspace a repository's memories live in.
    #[arg(long, env = "ANTUMBRA_GITHUB_WEBHOOK_SECRET", hide_env_values = true)]
    github_webhook_secret: Option<String>,
    /// Read the webhook secret from this file instead (see --jwt-secret-file).
    #[arg(
        long,
        env = "ANTUMBRA_GITHUB_WEBHOOK_SECRET_FILE",
        conflicts_with = "github_webhook_secret"
    )]
    github_webhook_secret_file: Option<std::path::PathBuf>,
    /// Every repository the App delivers events for lives in this ONE
    /// workspace (tenant).
    #[arg(long, env = "ANTUMBRA_GITHUB_TENANT", conflicts_with = "github_repos")]
    github_tenant: Option<String>,
    /// Path to a JSON file mapping repository slug -> workspace tenant (e.g.
    /// `{"github.com/acme/orders": "ws:acme"}`). A repository absent from the
    /// map is acknowledged and ignored rather than landing somewhere else.
    #[arg(long, env = "ANTUMBRA_GITHUB_REPOS")]
    github_repos: Option<std::path::PathBuf>,
    /// The GitHub App's id, with --github-app-key-file: lets the receiver read
    /// repository contents as the App (installation tokens), which is what
    /// ingesting a merged pull request's documents and cold-starting a newly
    /// installed repository need. Without it the receiver still re-anchors,
    /// orphans, and remembers pull requests.
    #[arg(long, env = "ANTUMBRA_GITHUB_APP_ID")]
    github_app_id: Option<String>,
    /// Path to the App's private key (PEM), read once at startup.
    #[arg(long, env = "ANTUMBRA_GITHUB_APP_KEY_FILE")]
    github_app_key_file: Option<std::path::PathBuf>,
    /// The GitHub API base (`https://<host>/api/v3` for Enterprise Server).
    #[arg(
        long,
        env = "ANTUMBRA_GITHUB_API_URL",
        default_value = "https://api.github.com"
    )]
    github_api_url: String,
    /// Post a knowledge diff on every pull request in a mapped repository: a
    /// neutral check run listing the memories and documents anchored to the
    /// paths it changes and to its branch. Needs the App (--github-app-id)
    /// with the Checks permission (write). Off by default.
    #[arg(long, env = "ANTUMBRA_GITHUB_KNOWLEDGE_DIFF")]
    github_knowledge_diff: bool,
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

/// The built-in embedder when no `--embedder-url` is configured: the real candle
/// BERT under `--features models`; `--fake-embedder` still selects the
/// byte-histogram stand-in for smoke tests. Both produce `EMBED_DIM`-wide vectors.
#[cfg(feature = "models")]
fn make_embedder(fake: bool) -> Result<Box<dyn Embedder>> {
    if fake {
        return Ok(Box::new(antumbra_core::testing::FixedEmbedder::new(
            EMBED_DIM,
        )));
    }
    Ok(Box::new(antumbra_serve::BertEmbedder::load()?))
}

/// Without `models` there is no built-in embedder: the operator brings one
/// (`--embedder-url`) or opts into the byte-histogram stand-in explicitly.
/// Refusing to start beats a server whose recall silently matches character
/// statistics instead of meaning.
#[cfg(not(feature = "models"))]
fn make_embedder(fake: bool) -> Result<Box<dyn Embedder>> {
    if fake {
        return Ok(Box::new(antumbra_core::testing::FixedEmbedder::new(
            EMBED_DIM,
        )));
    }
    anyhow::bail!(
        "no embedder configured: pass --embedder-url <OpenAI-compatible /embeddings endpoint \n         returning {EMBED_DIM}-d vectors> (for example Ollama serving all-minilm), or \n         --fake-embedder to accept the non-semantic byte-histogram stand-in (demos only)"
    )
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

/// Register this machine in the user's fabric (ADR-0017 A2): what backend the
/// build can drive, how much video memory the device it would train on has, and
/// the role that follows. Re-registering on every start is the point -- the
/// answer changes when the binary is rebuilt with a GPU backend, or the card is
/// pulled -- and the row is keyed per (tenant, user, host), so it updates.
///
/// A failed registration is reported and not fatal. The fabric is not why the
/// operator started a memory server, and a node that cannot say where it is
/// still recalls and stores perfectly well; it just will not be dispatched to.
/// Reported rather than swallowed, because a fabric that quietly has no genesis
/// node looks exactly like a fabric whose genesis node is a laptop.
pub(crate) async fn register_node(store: &Store, tenant: &TenantId, user: &UserId, host: &str) {
    let profile = antumbra_core::DeviceProfile::detected(
        tenant.clone(),
        user.clone(),
        host,
        hardware::backend(),
        hardware::vram_mib(),
        Utc::now(),
    );
    match antumbra_store::repo::device::upsert(store, &profile).await {
        Ok(()) => eprintln!(
            "antumbra-mcp: registered {host} as a {} node ({}{})",
            profile.role.as_str(),
            profile.backend,
            match profile.vram_mib {
                Some(mib) => format!(", {mib} MiB"),
                None => String::new(),
            }
        ),
        Err(e) => eprintln!("antumbra-mcp: could not register {host} in the fabric: {e}"),
    }
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
    register_node(&store, &tenant, &user, &host).await;
    // The `answer` tool serves through the routed expert. Build the engine from
    // the session's visible population (one connection; embedded is single-writer).
    let serve = build_serve(&store, &host).await?;
    // The process outlives the record session, so a keeper re-signs the
    // connection in before it expires (checked at every tool call).
    let keeper = session::SessionKeeper::new(store.clone(), tenant.clone(), user.clone());
    Ok(McpServer::new(
        store,
        embedder,
        tenant,
        user,
        host,
        default_compartment,
        serve,
    )
    .with_session_keeper(keeper))
}

/// Build the serving engine the `answer` tool drives: a resident
/// [`MultiAdapterServe`](antumbra_serve::MultiAdapterServe) over the shared base,
/// registered with every expert the `store` can currently see (routing scopes
/// which a session may actually pick). `None` only when the build has no model
/// backend.
///
/// An empty population still gets an engine. It used to get `None`, so a fresh
/// node had nothing for the consolidation trigger to hot-register its first
/// expert into: the train succeeded, the log said "now servable", and `answer`
/// reported that serving was not configured until someone restarted the server
/// (EXP-022). The engine is lazy, so an empty one costs no VRAM: the base loads on
/// the first `answer` that has an adapter to serve.
#[cfg(feature = "models")]
pub(crate) async fn build_serve(
    store: &Store,
    host: &str,
) -> Result<Option<Arc<dyn antumbra_core::ports::Serve>>> {
    use antumbra_serve::{MultiAdapterServe, RaftConfig};

    // Dormant experts stay registered, since they are served when named;
    // archived ones are not (ADR-0022 S-5).
    let experts = antumbra_store::repo::lifecycle::servable(store).await?;
    // With no expert to name a base, use the one consolidation trains on, so the
    // first minted adapter fits the resident model.
    let base = serving_base(
        experts.first().map(|e| e.base_model.as_str()),
        &RaftConfig::default().base_model,
    );
    // Serve the learned mode (greedy + repetition penalty + n-gram block +
    // nucleus), not the training-time exploration draw.
    let cfg = RaftConfig::for_serving(RaftConfig::default().max_new_tokens, 0.0);
    let mut engine = MultiAdapterServe::new(base, cfg);
    // Only the adapters this machine can actually open (ADR-0017 A2). The row
    // travels and the weights do not, so once a user's nodes reconcile, every
    // node learns about every expert while exactly one holds each file.
    // Registering them all would route to an adapter that is not here and fail
    // at serve time, on a path that looks perfectly valid in the row.
    let (mine, elsewhere): (Vec<_>, Vec<_>) = experts.iter().partition(|e| e.is_placed_on(host));
    for e in &mine {
        engine.register(e.id.clone(), e.artifact_uri.clone());
    }
    if !elsewhere.is_empty() {
        eprintln!(
            "antumbra-mcp: {} of {} experts live on another node and are not served here",
            elsewhere.len(),
            experts.len()
        );
    }
    Ok(Some(Arc::new(engine)))
}

#[cfg(not(feature = "models"))]
pub(crate) async fn build_serve(
    _store: &Store,
    _host: &str,
) -> Result<Option<Arc<dyn antumbra_core::ports::Serve>>> {
    Ok(None)
}

/// The base model the serving engine loads: the population's, or `default` when
/// there is no population yet.
#[cfg_attr(not(feature = "models"), allow(dead_code))]
fn serving_base(population_base: Option<&str>, default: &str) -> String {
    population_base.unwrap_or(default).to_string()
}

/// The machine name, or whatever the operator named it.
///
/// Inside a container the hostname is the container ID, which changes on every
/// recreate. Since a node's row is keyed on (tenant, user, host), taking that
/// as the name would register the same machine as a new node each time and
/// leave the old rows behind still claiming to be trainers. `--host` /
/// `ANTUMBRA_HOST` is how a containerised deployment says what it really is.
fn default_host(explicit: Option<String>) -> String {
    explicit.unwrap_or_else(|| {
        std::env::var("COMPUTERNAME")
            .or_else(|_| std::env::var("HOSTNAME"))
            .unwrap_or_else(|_| "local".into())
    })
}

fn main() -> Result<()> {
    // Before the runtime and held until it is gone, so the spans still queued
    // are flushed on the way out, by an exporter whose blocking HTTP client
    // must not be dropped inside the runtime.
    let _telemetry = telemetry::init();
    // SurrealDB's engine-enforced ACL subqueries for tenant isolation and memory compartments recurse deep;
    // host the runtime on a large-stack thread so the 1 MB Windows main-thread
    // stack does not overflow (see the matching note in the CLI).
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(|| {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            let outcome = runtime.block_on(run());
            // Whatever still runs on the blocking pool (an embed call, a
            // train) gets a moment and is then left behind: waiting on a train
            // would outlast the grace period, and the flush comes after this.
            runtime.shutdown_timeout(std::time::Duration::from_secs(2));
            outcome
        })?
        .join()
        .map_err(|_| anyhow::anyhow!("antumbra-mcp worker thread panicked"))?
}

impl Cli {
    /// The optional endpoints as the operator meant them. docker compose
    /// passes an unset variable through as an empty one (`${VAR:-}`), and
    /// clap reads an empty variable as a value, so a stack without a reranker
    /// or copal configured one at the address "": every recall paid a failing
    /// rerank call, and every document ingest would fail on the archive. An
    /// empty setting is an unset one.
    fn without_empty_settings(mut self) -> Self {
        let set = |value: Option<String>| value.filter(|v| !v.trim().is_empty());
        self.rerank_url = set(self.rerank_url);
        self.rerank_key = set(self.rerank_key);
        self.copal_addr = set(self.copal_addr);
        self.copal_tenant = set(self.copal_tenant);
        self.copal_key = set(self.copal_key);
        self.copal_keys = self.copal_keys.filter(|p| !p.as_os_str().is_empty());
        self
    }
}

async fn run() -> Result<()> {
    let cli = Cli::parse().without_empty_settings();
    // Secrets resolve once, from the inline flag or its file, and only the
    // resolved value is used from here on.
    let jwt_secret = secrets::resolve(
        cli.jwt_secret.clone(),
        cli.jwt_secret_file.as_deref(),
        "JWT secret",
    )?;
    let db_pass = secrets::resolve(
        cli.db_pass.clone(),
        cli.db_pass_file.as_deref(),
        "database password",
    )?;

    // Mint a hook token and exit -- no DB or embedder needed. HS256 only: the
    // server holds the symmetric secret; an RS256 deployment mints via its auth
    // service's private key.
    if cli.mint_token {
        let secret = jwt_secret.as_deref().ok_or_else(|| {
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

    if cli.auto_consolidate {
        eprintln!(
            "antumbra-mcp: --auto-consolidate is retired and does nothing; standing experts \
             are trained from accepted behaviors (ADR-0027)"
        );
    }
    let host = default_host(cli.host);
    // A configured endpoint embeds on the tenant's side (P-1c); otherwise the
    // built-in embedder (candle BERT under `models`; the byte-histogram stand-in
    // only when asked for explicitly). Either way the vectors are EMBED_DIM-wide.
    let embedder: Arc<dyn Embedder> = match cli.embedder_url {
        Some(url) => Arc::new(embed::HttpEmbedder::new(
            url,
            cli.embedder_model,
            cli.embedder_key,
        )),
        None => Arc::from(make_embedder(cli.fake_embedder)?),
    };

    // Optional cross-encoder rerank stage (P-2). Operator-configured endpoint; the
    // precision stage runs after hybrid recall and degrades to RRF order on error.
    // Built once as the concrete type and then viewed two ways. The same
    // endpoint answers both questions, and calling it twice to get an order and
    // then a magnitude would double the latency of every recall.
    let rerank_model = cli.rerank_model.filter(|m| !m.trim().is_empty());
    let explicit_calibration =
        match cli.floor_calibration.as_deref().map(str::trim) {
            None | Some("") => None,
            Some(text) => Some(antumbra_rerank::floor::parse_calibration(text).ok_or_else(
                || anyhow::anyhow!("--floor-calibration wants `a,b` with a > 0, got `{text}`"),
            )?),
        };
    // The model the endpoint serves, asked once: it decides the floor's
    // calibration, since a fit belongs to one model's scores.
    let rerank_url_for_recheck = cli.rerank_url.clone();
    let rerank_key_for_recheck = cli.rerank_key.clone();
    let served = match cli.rerank_url.clone() {
        Some(url) => {
            let key = cli.rerank_key.clone();
            tokio::task::spawn_blocking(move || antumbra_rerank::served_model(&url, key.as_deref()))
                .await
                .ok()
                .flatten()
        }
        None => None,
    };
    let http_reranker: Option<Arc<antumbra_rerank::HttpReranker>> = cli.rerank_url.map(|url| {
        Arc::new(antumbra_rerank::HttpReranker::new(
            url,
            rerank_model.clone(),
            cli.rerank_key,
        ))
    });
    let reranker: Option<Arc<dyn antumbra_core::ports::Reranker>> = http_reranker
        .clone()
        .map(|r| r as Arc<dyn antumbra_core::ports::Reranker>);
    let scorer: Option<Arc<dyn antumbra_core::ports::RelevanceScorer>> =
        http_reranker.map(|r| r as Arc<dyn antumbra_core::ports::RelevanceScorer>);
    // The relevance floor (ADR-0023 B-2). The same cross-encoder that orders the
    // pool also answers "does this answer the query" once its score is mapped
    // through the fitted calibration, so the floor costs no extra model and
    // appears exactly when a reranker is configured.
    let mut running_calibration = None;
    let warm_scorer = scorer.clone();
    let decider: Option<Arc<dyn antumbra_core::ports::TypedDecider>> = scorer.and_then(|s| {
        use antumbra_rerank::floor::{choose, CalibratedFloor, FloorChoice};
        match choose(
            explicit_calibration,
            served.as_deref(),
            rerank_model.as_deref(),
        ) {
            FloorChoice::Calibrated {
                calibration,
                because,
            } => {
                eprintln!("antumbra-mcp: relevance floor on, calibrated: {because}");
                running_calibration = Some(calibration);
                Some(Arc::new(CalibratedFloor::with_calibration(s, calibration))
                    as Arc<dyn antumbra_core::ports::TypedDecider>)
            }
            FloorChoice::Off { because } => {
                eprintln!("antumbra-mcp: relevance floor off: {because}");
                None
            }
        }
    });
    // Pay for the models' first use now rather than in the first recall, and
    // check the floor's calibration again once the reranker answers, when the
    // endpoint did not say at startup what it serves.
    warmup::spawn(
        embedder.clone(),
        warm_scorer,
        rerank_url_for_recheck
            .filter(|_| served.is_none())
            .map(|url| warmup::Recheck {
                url,
                key: rerank_key_for_recheck,
                calibration: running_calibration,
            }),
    );

    // Optional copal document-of-record archive. Operator-configured; absent,
    // ingest keeps only the chunks (the v0 behavior, unchanged).
    let copal = antumbra_copal::CopalArchive::from_flags(
        cli.copal_addr.as_deref(),
        cli.copal_tenant,
        cli.copal_key,
        cli.copal_keys.as_deref(),
    )?;
    // The tool profile, validated against the real tool list so a typo
    // refuses at startup instead of silently hiding a tool.
    let profile = profile::ToolProfile::parse(&cli.tools, &server::McpServer::all_tool_names())?
        .map(Arc::new);
    // The GitHub App webhook receiver: its secret resolves like every other
    // secret, and a repository must map to a workspace before an event can
    // touch anything. Webhooks arrive over the network, so HTTP only.
    let github_secret = secrets::resolve(
        cli.github_webhook_secret.clone(),
        cli.github_webhook_secret_file.as_deref(),
        "GitHub webhook secret",
    )?;
    let github = http::GithubConfig::from_flags(
        github_secret,
        cli.github_tenant.clone(),
        cli.github_repos.as_deref(),
        cli.github_app_id.clone(),
        cli.github_app_key_file.as_deref(),
        &cli.github_api_url,
    )?;
    if cli.github_knowledge_diff && !github.as_ref().is_some_and(|g| g.reads_contents()) {
        anyhow::bail!(
            "--github-knowledge-diff posts check runs as the App: it needs the webhook receiver              and --github-app-id with --github-app-key-file"
        );
    }
    let github = github.map(|g| g.with_knowledge_diff(cli.github_knowledge_diff));
    if github.is_some() && cli.http.is_none() {
        anyhow::bail!("the GitHub webhook receiver needs --http: deliveries arrive over the networked surface");
    }

    if let Some(addr) = cli.http {
        // Networked multi-tenant surface: identity per request from a verified JWT.
        let verifier = build_verifier(&jwt_secret, &cli.jwt_public_key, &cli.jwt_audience)?;
        return http::serve(
            addr,
            cli.url,
            cli.db_user,
            db_pass,
            host,
            embedder,
            verifier,
            cli.auto_propose,
            cli.chunk_in_flight,
            reranker,
            decider,
            copal,
            profile,
            github,
        )
        .await;
    }

    // stdio: one fixed identity for the life of the process.
    let mut service = build_session(
        &cli.url,
        cli.db_user.as_deref(),
        db_pass.as_deref(),
        TenantId::new(cli.tenant),
        UserId::new(cli.user),
        host,
        embedder,
    )
    .await?;
    if let Some(threshold) = cli.auto_propose {
        service = service.with_auto_propose(threshold);
    }
    if let Some(r) = reranker {
        service = service.with_reranker(r);
    }
    if let Some(d) = decider {
        service = service.with_decider(d);
    }
    if let Some(c) = copal {
        service = service.with_copal_archive(c);
    }
    if let Some(p) = profile {
        service = service.with_tool_profile(p);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// What compose hands over for an unset endpoint is an empty string, and
    /// clap keeps it; the server treats it as no endpoint at all.
    #[test]
    fn an_empty_endpoint_setting_is_an_unset_one() {
        let cli = Cli::try_parse_from([
            "antumbra-mcp",
            "--rerank-url",
            "",
            "--copal-addr",
            " ",
            "--copal-tenant",
            "",
        ])
        .expect("empty values parse")
        .without_empty_settings();
        assert_eq!(cli.rerank_url, None);
        assert_eq!(cli.copal_addr, None);
        assert_eq!(cli.copal_tenant, None);

        let cli = Cli::try_parse_from([
            "antumbra-mcp",
            "--rerank-url",
            "http://rerank/rerank",
            "--copal-addr",
            "http://copal:8080",
        ])
        .expect("set values parse")
        .without_empty_settings();
        assert_eq!(cli.rerank_url.as_deref(), Some("http://rerank/rerank"));
        assert_eq!(cli.copal_addr.as_deref(), Some("http://copal:8080"));
    }

    #[test]
    fn an_empty_population_still_names_a_base() {
        assert_eq!(serving_base(None, "org/base"), "org/base");
        assert_eq!(
            serving_base(Some("org/trained-on"), "org/base"),
            "org/trained-on"
        );
    }

    /// A node puts itself in the user's fabric on start, under its own record
    /// session -- which is the whole point of ADR-0017's write rule: the row a
    /// node writes is its own user's, and it writes it as that user rather than
    /// as the owner.
    #[tokio::test]
    async fn a_node_registers_itself_as_the_user_it_serves() -> Result<()> {
        let store = Store::connect_memory(antumbra_store::EMBED_DIM).await?;
        let tenant = TenantId::new("ws:fabric");
        let user = UserId::new("user:lily");
        provision_identity(&store, &tenant, &user).await?;
        store.signin(&tenant, &user).await?;

        register_node(&store, &tenant, &user, "her-laptop").await;

        let fabric = antumbra_store::repo::device::list_for_user(&store, &tenant, &user).await?;
        let [node] = fabric.as_slice() else {
            anyhow::bail!("one machine, one row, got {}", fabric.len());
        };
        assert_eq!(node.host, "her-laptop");
        assert_eq!(node.backend, hardware::backend());
        // The role is the one the hardware earns, not one this test asserts
        // into being: on a CPU build that is Memory, on a CUDA build with a big
        // enough card it is Genesis, and either way it is derived.
        assert_eq!(
            node.role,
            antumbra_core::role_for(hardware::backend(), hardware::vram_mib())
        );

        // Starting again is not a second machine.
        register_node(&store, &tenant, &user, "her-laptop").await;
        assert_eq!(
            antumbra_store::repo::device::list_for_user(&store, &tenant, &user)
                .await?
                .len(),
            1
        );
        Ok(())
    }

    /// The cold start: a node with no experts yet must still come up with a
    /// serving engine, or the first expert it mints has nowhere to be registered.
    #[cfg(feature = "models")]
    #[tokio::test]
    async fn a_fresh_node_gets_a_serving_engine() -> Result<()> {
        let store = Store::connect_memory(antumbra_store::EMBED_DIM).await?;
        let Some(serve) = build_serve(&store, "test-host").await? else {
            anyhow::bail!("an empty population must still get a serving engine");
        };
        let minted = antumbra_core::ExpertId::new("expert:user:test:comp:fresh");
        assert!(!serve.can_serve(&minted));
        serve.register_expert(&minted, "adapters/fresh_g0.safetensors");
        assert!(serve.can_serve(&minted), "hot-registration has a target");
        Ok(())
    }
}
