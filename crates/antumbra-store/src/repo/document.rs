//! Knowledge-document chunk repository (P-3). Tenant-isolated exactly like
//! `memory`: the engine `PERMISSIONS ... WHERE tenant_id = $auth.tenant` scopes
//! every row, and the repo also filters explicitly (defense-in-depth). HNSW
//! vector recall via surql-rs's `vector_search` builder; no raw SurrealQL.

use serde::{Deserialize, Serialize};

use surql::query::builder::Query;
use surql::query::crud::{query_records, upsert_record};
use surql::query::helpers::VectorDistanceType;
use surql::types::operators::eq;
use surql::types::RecordID;

use antumbra_core::{DocumentChunk, DocumentChunkId, Result, TenantId};

use crate::dto::parse_dt;
use crate::error::map;
use crate::store::Store;

const TABLE: &str = "document_chunk";

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

/// Semantic recall: the `k` document chunks nearest `query` *within* `tenant`.
/// The tenant filter is ANDed onto the KNN clause so the candidate set never
/// crosses a tenant boundary (and the engine ACL scopes it again).
pub async fn recall(
    store: &Store,
    tenant: &TenantId,
    query: &[f32],
    k: usize,
) -> Result<Vec<DocumentChunk>> {
    let vector: Vec<f64> = query.iter().map(|&x| f64::from(x)).collect();
    let q = Query::new()
        .select(None)
        .from_table(TABLE)
        .map_err(map)?
        .where_(eq("tenant_id", tenant.as_str()))
        .vector_search(
            "embedding",
            vector,
            k as i64,
            VectorDistanceType::Cosine,
            None,
        )
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
