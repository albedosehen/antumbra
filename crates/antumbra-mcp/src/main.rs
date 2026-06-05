//! `antumbra-mcp` — the Model Context Protocol server that exposes Antumbra's
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

mod auth;
mod http;
mod server;
use server::McpServer;

#[derive(Parser)]
#[command(name = "antumbra-mcp", about = "Antumbra MCP server over the Penumbra memory store", version)]
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
    /// Enable the autonomous propose trigger: once a user's unorganized inbox
    /// reaches this many memories, a write auto-clusters it into proposed
    /// compartments (reversible; the user curates). Off when unset.
    #[arg(long)]
    auto_propose: Option<usize>,
}

async fn connect(url: &str) -> Result<Store> {
    let config = ConnectionConfig::builder()
        .url(url)
        .namespace("antumbra")
        .database("main")
        .build()?;
    Ok(Store::connect(config, EMBED_DIM).await?)
}

/// The real candle BERT embedder under `--features models`, else the
/// byte-histogram fake. Both produce `EMBED_DIM`-wide vectors.
#[cfg(feature = "models")]
fn make_embedder() -> Result<Box<dyn Embedder>> {
    Ok(Box::new(antumbra_serve::BertEmbedder::load()?))
}

#[cfg(not(feature = "models"))]
fn make_embedder() -> Result<Box<dyn Embedder>> {
    Ok(Box::new(antumbra_core::testing::FixedEmbedder::new(EMBED_DIM)))
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
    let default_compartment =
        CompartmentId::new(format!("comp:{}:{}:default", tenant.as_str(), user.as_str()));
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
/// shares one connection across identities — see [`http`] — because an embedded
/// engine is single-writer.)
async fn build_session(
    url: &str,
    tenant: TenantId,
    user: UserId,
    host: String,
    embedder: Arc<dyn Embedder>,
) -> Result<McpServer> {
    let store = connect(url).await?;
    let default_compartment = provision_identity(&store, &tenant, &user).await?;
    store.signin(&tenant, &user).await?;
    // Serving (the `answer` tool) is a route-only seam here; wiring the real
    // MultiAdapterServe is GPU work tracked on the roadmap (R-4).
    Ok(McpServer::new(
        store,
        embedder,
        tenant,
        user,
        host,
        default_compartment,
        None,
    ))
}

fn default_host(explicit: Option<String>) -> String {
    explicit.unwrap_or_else(|| {
        std::env::var("COMPUTERNAME")
            .or_else(|_| std::env::var("HOSTNAME"))
            .unwrap_or_else(|_| "local".into())
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let host = default_host(cli.host);
    let embedder: Arc<dyn Embedder> = Arc::from(make_embedder()?);

    if let Some(addr) = cli.http {
        // Networked multi-tenant surface: identity per request from a verified JWT.
        let verifier = build_verifier(&cli.jwt_secret, &cli.jwt_public_key, &cli.jwt_audience)?;
        return http::serve(addr, cli.url, host, embedder, verifier, cli.auto_propose).await;
    }

    // stdio: one fixed identity for the life of the process.
    let mut service = build_session(
        &cli.url,
        TenantId::new(cli.tenant),
        UserId::new(cli.user),
        host,
        embedder,
    )
    .await?;
    if let Some(threshold) = cli.auto_propose {
        service = service.with_auto_propose(threshold);
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
