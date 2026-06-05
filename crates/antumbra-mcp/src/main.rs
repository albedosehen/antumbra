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
use antumbra_core::TenantId;
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
    /// The workspace this server serves. Every tool is engine-isolated to it.
    #[arg(long)]
    tenant: String,
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

    // Connect as owner (applies the schema), provision the tenant's principal,
    // then bind the session as that tenant so the engine enforces isolation.
    let store = connect(&cli.url).await?;
    let tenant = TenantId::new(cli.tenant);
    principal::provision(&store, &tenant).await?;
    store.signin_tenant(&tenant).await?;

    let embedder: Arc<dyn Embedder> = Arc::from(make_embedder()?);
    let service = McpServer::new(store, embedder, tenant);

    let running = service
        .serve((tokio::io::stdin(), tokio::io::stdout()))
        .await?;
    running.waiting().await?;
    Ok(())
}
