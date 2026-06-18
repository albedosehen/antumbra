//! Per-workspace embedder configuration: a hosted multi-tenant deployment lets
//! each workspace bring its own OpenAI-compatible `/embeddings` endpoint (its own
//! Ollama / TEI / provider), instead of one endpoint baked into the server.
//!
//! One row per tenant (`tenant_id` is unique). By default the endpoint MUST
//! return `EMBED_DIM`-wide vectors -- the HNSW index is fixed-dimension. The
//! optional `source_dim` opens the **Matryoshka** path: a generalist model
//! (BGE-M3, multilingual-e5, …) returns a longer vector whose leading
//! `EMBED_DIM` prefix is stored (re-normalized) into the same fixed index, so a
//! richer embedder is a *configuration*, not an index change. Either way the
//! stored vector is exactly `EMBED_DIM`-wide; changing the model means
//! re-embedding the workspace's memories (the `reembed` CLI). All access is via
//! surql-rs builders; the table is tenant-scoped by engine PERMISSIONS.

use surql::query::builder::Query;
use surql::query::crud::{delete_records, first, upsert_record};
use surql::types::operators::eq;
use surql::types::RecordID;

use antumbra_core::{Result, TenantId};

use crate::error::map;
use crate::store::Store;

const TABLE: &str = "embedder_config";

/// A workspace's embedder endpoint. `api_key` is an optional bearer token for the
/// endpoint; stored in the tenant-scoped row (engine-ACL'd to the workspace).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EmbedderConfig {
    pub tenant_id: String,
    /// Full OpenAI-compatible embeddings endpoint (e.g. `http://host:11434/v1/embeddings`).
    pub url: String,
    /// Model name sent to the endpoint.
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// The Matryoshka source dimension. `None` (the default, and how every
    /// pre-existing row deserializes) is the strict path: the endpoint must
    /// return exactly `EMBED_DIM`-wide vectors. `Some(n)` (with `n > EMBED_DIM`)
    /// is the Matryoshka path: the endpoint returns `n`-dim vectors and the
    /// re-normalized leading `EMBED_DIM` prefix is stored — e.g. `Some(1024)`
    /// means "expect 1024 from the model, store the renormalized 384 prefix".
    /// The stored vector is always `EMBED_DIM`-wide either way.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_dim: Option<u32>,
}

/// One record id per tenant, so a set replaces the prior config (idempotent).
fn key(tenant: &str) -> String {
    tenant.replace([':', '/', '\\', '|'], "_")
}

/// Set (or replace) the workspace's embedder. Idempotent.
pub async fn upsert(store: &Store, cfg: &EmbedderConfig) -> Result<()> {
    let rid = RecordID::<()>::new(TABLE, key(&cfg.tenant_id).as_str()).map_err(map)?;
    upsert_record(store.client(), &rid, serde_json::to_value(cfg)?)
        .await
        .map_err(map)?;
    Ok(())
}

/// The workspace's embedder, or `None` to fall back to the server default.
pub async fn get(store: &Store, tenant: &TenantId) -> Result<Option<EmbedderConfig>> {
    let query = Query::new()
        .select(None)
        .from_table(TABLE)
        .map_err(map)?
        .where_(eq("tenant_id", tenant.as_str()));
    first(store.client(), &query).await.map_err(map)
}

/// Clear the workspace's embedder (revert to the server default). No-op if absent.
pub async fn delete(store: &Store, tenant: &TenantId) -> Result<()> {
    delete_records(
        store.client(),
        TABLE,
        Some(&eq("tenant_id", tenant.as_str())),
    )
    .await
    .map_err(map)?;
    Ok(())
}
