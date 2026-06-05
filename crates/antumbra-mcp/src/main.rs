//! `antumbra-mcp` — the Model Context Protocol server that exposes Antumbra's
//! Penumbra memory to an agent. The runtime surface that lets a client (Claude
//! Code, any MCP host) talk to Antumbra instead of a separate memory service.
//!
//! v0 serves a single tenant (the workspace passed via `--tenant`) over stdio:
//! it connects as owner, provisions the tenant's principal, then signs the
//! session in as that tenant so every tool is engine-isolated. The networked,
//! per-request multi-tenant surface (HTTP, identity-resolved tenant) is next.

use std::sync::Arc;

use anyhow::Result;
use clap::Parser;
use rmcp::ServiceExt;

use antumbra_core::ports::Embedder;
use antumbra_core::{TenantId, UserId};
use antumbra_store::repo::principal;
use antumbra_store::{ConnectionConfig, Store, EMBED_DIM};

mod server;
use server::McpServer;

#[derive(Parser)]
#[command(name = "antumbra-mcp", about = "Antumbra MCP server over the Penumbra memory store", version)]
struct Cli {
    /// SurrealDB url: `surrealkv://./data/antumbra.skv` (persistent),
    /// `mem://` (ephemeral), or `ws://host:8000/rpc`.
    #[arg(long, default_value = "surrealkv://./data/antumbra.skv")]
    url: String,
    /// The workspace (tenant) this server serves. Every tool is engine-isolated
    /// to it.
    #[arg(long)]
    tenant: String,
    /// The user this session acts as (the compartment-ownership / sharing
    /// actor). Defaults to a per-tenant default user.
    #[arg(long, default_value = "user:default")]
    user: String,
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

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Connect as owner (applies the schema), provision the (tenant, user)
    // principal, then bind the session so the engine enforces isolation and
    // compartment access for this user.
    let store = connect(&cli.url).await?;
    let tenant = TenantId::new(cli.tenant);
    let user = UserId::new(cli.user);
    principal::provision(&store, &tenant, &user).await?;
    store.signin(&tenant, &user).await?;

    let embedder: Arc<dyn Embedder> = Arc::from(make_embedder()?);
    let service = McpServer::new(store, embedder, tenant);

    let running = service
        .serve((tokio::io::stdin(), tokio::io::stdout()))
        .await?;
    running.waiting().await?;
    Ok(())
}
