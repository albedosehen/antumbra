//! Per-workspace embedder configuration: a hosted multi-tenant deployment lets
//! each workspace bring its own OpenAI-compatible `/embeddings` endpoint (its own
//! Ollama / TEI / provider), instead of one endpoint baked into the server.
//!
//! One row per tenant (`tenant_id` is unique). The endpoint MUST return
//! `EMBED_DIM`-wide vectors -- the HNSW index is fixed-dimension, so this is a
//! model/endpoint choice, not a dimension one; changing the model means
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
    /// Model name sent to the endpoint (must produce the index dimension).
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
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
    delete_records(store.client(), TABLE, Some(&eq("tenant_id", tenant.as_str())))
        .await
        .map_err(map)?;
    Ok(())
}
