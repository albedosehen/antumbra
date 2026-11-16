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
use surql::query::crud::{delete_records, first};
use surql::types::operators::eq;

use antumbra_core::{Result, TenantId};

use crate::error::map;
use crate::store::Store;

const TABLE: &str = "embedder_config";

/// A workspace's embedder endpoint. `api_key` is an optional bearer token for the
/// endpoint, stored in a row only the owner view reads or writes
/// (`OWNER_VIEW_PERMS`); `Debug` shows whether it is set, never what it is.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
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

impl std::fmt::Debug for EmbedderConfig {
    /// The key is a secret: a log line or a panic message gets whether one is
    /// set, not the key.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbedderConfig")
            .field("tenant_id", &self.tenant_id)
            .field("url", &self.url)
            .field("model", &self.model)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("source_dim", &self.source_dim)
            .finish()
    }
}

/// One record id per tenant, so a set replaces the prior config (idempotent).
fn key(tenant: &str) -> String {
    tenant.replace([':', '/', '\\', '|'], "_")
}

/// Set (or replace) the workspace's embedder. Idempotent.
pub async fn upsert(store: &Store, cfg: &EmbedderConfig) -> Result<()> {
    store.put(TABLE, key(&cfg.tenant_id).as_str(), cfg).await
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repo::principal;
    use antumbra_core::{AntumbraError, UserId};

    fn config(url: &str) -> EmbedderConfig {
        EmbedderConfig {
            tenant_id: "ws:org".into(),
            url: url.into(),
            model: "all-MiniLM-L6-v2".into(),
            api_key: Some("sk-the-workspace-key".into()),
            source_dim: None,
        }
    }

    /// The row as the owner sees it, or an error naming what went wrong.
    async fn owner_row(store: &Store) -> Result<EmbedderConfig> {
        store.invalidate().await?;
        get(store, &TenantId::new("ws:org"))
            .await?
            .ok_or_else(|| AntumbraError::other("the owner sees no row"))
    }

    // The row holds the endpoint's key, so a member's record session neither
    // reads it nor changes it: a read answers nothing, and a write or a delete
    // persists nothing, while the owner view reads and writes as before.
    #[tokio::test]
    async fn a_member_session_neither_reads_nor_writes_the_row() -> Result<()> {
        let store = crate::testing::mem_store_of(4).await?;
        let tenant = TenantId::new("ws:org");
        let member = UserId::new("user:m");
        principal::provision(&store, &tenant, &member).await?;
        upsert(&store, &config("http://embed.example/v1/embeddings")).await?;

        store.signin(&tenant, &member).await?;
        if get(&store, &tenant).await?.is_some() {
            return Err(AntumbraError::other("a member read the row"));
        }
        // The engine refuses a member's writes by persisting nothing or with
        // an error; the row is what is checked.
        upsert(&store, &config("http://attacker.example/v1/embeddings"))
            .await
            .ok();
        delete(&store, &tenant).await.ok();

        let row = owner_row(&store).await?;
        if row.url != "http://embed.example/v1/embeddings" {
            return Err(AntumbraError::other(format!(
                "a member changed the endpoint to {}",
                row.url
            )));
        }
        if row.api_key.as_deref() != Some("sk-the-workspace-key") {
            return Err(AntumbraError::other("the key changed or went"));
        }
        // The owner still writes and clears it.
        upsert(&store, &config("http://other.example/v1/embeddings")).await?;
        if owner_row(&store).await?.url != "http://other.example/v1/embeddings" {
            return Err(AntumbraError::other("the owner's write did not land"));
        }
        delete(&store, &tenant).await?;
        if get(&store, &tenant).await?.is_some() {
            return Err(AntumbraError::other("the owner's delete did not land"));
        }
        Ok(())
    }

    #[test]
    fn debug_output_says_whether_a_key_is_set_and_never_what_it_is() -> Result<()> {
        let shown = format!("{:?}", config("http://embed.example/v1/embeddings"));
        if shown.contains("sk-the-workspace-key") {
            return Err(AntumbraError::other(format!("the key is in: {shown}")));
        }
        if !shown.contains("<redacted>") || !shown.contains("embed.example") {
            return Err(AntumbraError::other(format!(
                "not the shape expected: {shown}"
            )));
        }
        let mut without = config("http://embed.example/v1/embeddings");
        without.api_key = None;
        if !format!("{without:?}").contains("api_key: None") {
            return Err(AntumbraError::other("an absent key should read as None"));
        }
        Ok(())
    }
}
