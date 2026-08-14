//! Knowledge-document chunk repository (P-3). Tenant-isolated exactly like
//! `memory`: the engine `PERMISSIONS ... WHERE tenant_id = $auth.tenant` scopes
//! every row, and the repo also filters explicitly (defense-in-depth). HNSW
//! vector recall via surql-rs's `vector_search_indexed` builder, through
//! `document_chunk_embedding_hnsw`; no raw SurrealQL.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use surql::query::builder::Query;
use surql::query::crud::{query_records, upsert_record};
use surql::query::helpers::fulltext_search_query;
use surql::types::operators::eq;
use surql::types::RecordID;

use antumbra_core::{DocumentChunk, DocumentChunkId, Result, TenantId};

use crate::dto::parse_dt;
use crate::error::map;
use crate::fusion::{rrf_fuse, DEFAULT_RRF_K};
use crate::knn::{candidate_pool, search_effort};
use crate::store::Store;

const TABLE: &str = "document_chunk";

/// Candidate-pool sizing for [`recall_hybrid`] (see the `memory` repo for the
/// rationale): each leg fetches `k * POOL_MULTIPLIER`, clamped, so fusion can
/// reorder before truncating to `k`.
const POOL_MULTIPLIER: usize = 5;
const MIN_POOL: usize = 20;
const MAX_POOL: usize = 200;

#[derive(Serialize, Deserialize)]
struct ChunkRow {
    key: String,
    tenant_id: String,
    title: String,
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    ordinal: u32,
    content: String,
    #[serde(default)]
    embedding: Option<Vec<f32>>,
    created_at: String,
}

impl ChunkRow {
    fn from_domain(c: &DocumentChunk) -> Self {
        ChunkRow {
            key: c.id.as_str().to_string(),
            tenant_id: c.tenant.as_str().to_string(),
            title: c.title.clone(),
            source: c.source.clone(),
            ordinal: c.ordinal,
            content: c.content.clone(),
            embedding: c.embedding.clone(),
            created_at: c.created_at.to_rfc3339(),
        }
    }

    fn into_domain(self) -> Result<DocumentChunk> {
        Ok(DocumentChunk {
            id: DocumentChunkId::new(self.key),
            tenant: TenantId::new(self.tenant_id),
            title: self.title,
            source: self.source,
            ordinal: self.ordinal,
            content: self.content,
            embedding: self.embedding,
            created_at: parse_dt(&self.created_at)?,
        })
    }
}

/// Insert a document's chunks (addressed by their globally-unique ids).
pub async fn insert_chunks(store: &Store, chunks: &[DocumentChunk]) -> Result<()> {
    for c in chunks {
        let id = RecordID::<()>::new(TABLE, c.id.as_str()).map_err(map)?;
        let data = serde_json::to_value(ChunkRow::from_domain(c))?;
        upsert_record(store.client(), &id, data)
            .await
            .map_err(map)?;
    }
    Ok(())
}

/// Semantic recall: the `k` document chunks nearest `query` *within* `tenant`,
/// through `document_chunk_embedding_hnsw`.
///
/// The tenant filter is ANDed onto the KNN clause so the candidate set never
/// crosses a tenant boundary (and the engine ACL scopes it again) — but against
/// an index-backed KNN that filter is a RESIDUAL: the graph walk returns its
/// nearest neighbours across the whole table and the equality thins them
/// afterwards. So the walk is asked for a wider pool and the answer is
/// truncated to `k` once the thinning is done; asking for `k` directly would
/// hand a tenant with a small share of the chunks a short answer.
pub async fn recall(
    store: &Store,
    tenant: &TenantId,
    query: &[f32],
    k: usize,
) -> Result<Vec<DocumentChunk>> {
    let vector: Vec<f64> = query.iter().map(|&x| f64::from(x)).collect();
    let pool = candidate_pool(k);
    let q = Query::new()
        .select(None)
        .from_table(TABLE)
        .map_err(map)?
        .where_(eq("tenant_id", tenant.as_str()))
        .vector_search_indexed("embedding", vector, pool as i64, search_effort(pool))
        .map_err(map)?;
    let rows: Vec<ChunkRow> = query_records(store.client(), &q).await.map_err(map)?;
    rows.into_iter()
        .take(k)
        .map(ChunkRow::into_domain)
        .collect()
}

/// Hybrid recall over document chunks: fuse the dense (HNSW) and sparse (BM25
/// full-text) legs over `query_text` + `query_vec` via Reciprocal Rank Fusion,
/// returning the top `k` chunks for `tenant`. The sparse leg is best-effort (a
/// full-text error degrades to dense-only); a blank query is dense-only.
pub async fn recall_hybrid(
    store: &Store,
    tenant: &TenantId,
    query_text: &str,
    query_vec: &[f32],
    k: usize,
) -> Result<Vec<DocumentChunk>> {
    let pool = k.saturating_mul(POOL_MULTIPLIER).clamp(MIN_POOL, MAX_POOL);

    let dense = recall(store, tenant, query_vec, pool).await?;
    let sparse = sparse_recall(store, tenant, query_text, pool)
        .await
        .unwrap_or_default();

    if sparse.is_empty() {
        return Ok(dense.into_iter().take(k).collect());
    }

    let dense_ids: Vec<String> = dense.iter().map(|c| c.id.as_str().to_string()).collect();
    let sparse_ids: Vec<String> = sparse.iter().map(|c| c.id.as_str().to_string()).collect();
    let fused = rrf_fuse(&[dense_ids, sparse_ids], DEFAULT_RRF_K);

    let mut by_id: HashMap<String, DocumentChunk> = HashMap::new();
    for c in dense.into_iter().chain(sparse) {
        by_id.entry(c.id.as_str().to_string()).or_insert(c);
    }
    Ok(fused
        .into_iter()
        .take(k)
        .filter_map(|id| by_id.remove(&id))
        .collect())
}

/// The BM25 full-text (sparse) leg of [`recall_hybrid`]: the `k` chunks whose
/// `content` best matches `query_text`, tenant-scoped, in BM25 relevance order.
/// An empty/blank query returns nothing.
async fn sparse_recall(
    store: &Store,
    tenant: &TenantId,
    query_text: &str,
    k: usize,
) -> Result<Vec<DocumentChunk>> {
    if query_text.trim().is_empty() {
        return Ok(Vec::new());
    }
    let q = fulltext_search_query(TABLE, "content", 1, query_text, None, "score")
        .map_err(map)?
        .where_(eq("tenant_id", tenant.as_str()))
        .limit(k as i64)
        .map_err(map)?;
    let rows: Vec<ChunkRow> = query_records(store.client(), &q).await.map_err(map)?;
    rows.into_iter().map(ChunkRow::into_domain).collect()
}

/// Distinct document titles the tenant has ingested (the document list).
pub async fn list_titles(store: &Store, tenant: &TenantId) -> Result<Vec<String>> {
    // Paged by id (see `Store::read_paged`): a tenant's whole chunk corpus, one
    // row per chunk, is large enough to overflow a frame. The titles are sorted
    // and deduped below, so the row order off the wire is irrelevant.
    let filter = eq("tenant_id", tenant.as_str());
    let rows: Vec<ChunkRow> = store.read_paged(TABLE, None, Some(&filter)).await?;
    let mut titles: Vec<String> = rows.into_iter().map(|r| r.title).collect();
    titles.sort();
    titles.dedup();
    Ok(titles)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::EMBED_DIM;
    use chrono::Utc;

    fn chunk(id: &str, ordinal: u32, embedding: Vec<f32>) -> DocumentChunk {
        DocumentChunk {
            id: DocumentChunkId::new(id),
            tenant: TenantId::new("t"),
            title: "onboarding guide".into(),
            source: Some("guide.md".into()),
            ordinal,
            content: format!("chunk {ordinal}"),
            embedding: Some(embedding),
            created_at: Utc::now(),
        }
    }

    #[tokio::test]
    async fn chunks_round_trip_and_recall_by_vector() {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        let tenant = TenantId::new("t");
        let mut a = vec![0.0f32; EMBED_DIM];
        a[0] = 1.0;
        let mut b = vec![0.0f32; EMBED_DIM];
        b[1] = 1.0;
        insert_chunks(&store, &[chunk("dc:1", 0, a.clone()), chunk("dc:2", 1, b)])
            .await
            .unwrap();

        // The chunk nearest the `a` query is dc:1.
        let near = recall(&store, &tenant, &a, 2).await.unwrap();
        assert_eq!(near.len(), 2);
        assert_eq!(near[0].id.as_str(), "dc:1");
        assert_eq!(near[0].title, "onboarding guide");

        // The document list dedups the title across its chunks.
        assert_eq!(
            list_titles(&store, &tenant).await.unwrap(),
            vec!["onboarding guide".to_string()]
        );

        // Another tenant sees none of it (defense-in-depth filter; the engine
        // ACL scopes it too under a record session).
        let other = recall(&store, &TenantId::new("other"), &a, 2)
            .await
            .unwrap();
        assert!(other.is_empty());
    }
}
