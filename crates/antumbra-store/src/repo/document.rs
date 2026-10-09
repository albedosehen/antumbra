//! Knowledge-document chunk repository (P-3). Isolated exactly like `memory`:
//! the engine's compartment rule scopes every row (tenant, then the shared pool
//! or a compartment the session owns or was granted), and the repo also filters
//! by tenant explicitly (defense-in-depth). HNSW
//! vector recall via surql-rs's `vector_search_indexed` builder, through
//! `document_chunk_embedding_hnsw`; no raw SurrealQL.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use surql::query::builder::Query;
use surql::query::crud::{delete_records, query_records, upsert_record};
use surql::query::expressions::{as_, count_all, count_if, field};
use surql::types::operators::{and_, eq, is_none};
use surql::types::RecordID;

use antumbra_core::{CompartmentId, DocumentChunk, DocumentChunkId, Result, TenantId};

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
    // Copal document-of-record provenance: the archived file's id and its
    // content digest. Absent (not null) when the document was ingested without
    // an archive, so old rows and archive-less rows are indistinguishable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    copal_file: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    copal_digest: Option<String>,
    // Absent (not null) when None so the engine sees `compartment = NONE` for
    // the shared pool, which is also what every row written before documents
    // had compartments looks like.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    compartment: Option<String>,
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
            copal_file: c.copal_file.clone(),
            copal_digest: c.copal_digest.clone(),
            compartment: c.compartment.as_ref().map(|c| c.as_str().to_string()),
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
            copal_file: self.copal_file,
            copal_digest: self.copal_digest,
            compartment: self.compartment.map(CompartmentId::new),
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

/// Drop every chunk of `title` within `tenant` and `compartment` (`None` = the
/// shared pool). Ingest calls this before writing a document's new generation,
/// which is what makes a re-ingest an in-place replacement even when the
/// document shrank.
///
/// A document's identity is (tenant, compartment, title), so the compartment is
/// part of the predicate: two members may each keep a private document under the
/// same title, and replacing one must never reach the other. The engine's write
/// rule guards the same line from the other side, since a scoped session cannot
/// delete a row in a compartment it may not write to.
pub async fn delete_title(
    store: &Store,
    tenant: &TenantId,
    title: &str,
    compartment: Option<&CompartmentId>,
) -> Result<()> {
    let placed = match compartment {
        Some(c) => eq("compartment", c.as_str()),
        None => is_none("compartment"),
    };
    let condition = and_(
        and_(eq("tenant_id", tenant.as_str()), eq("title", title)),
        placed,
    );
    delete_records(store.client(), TABLE, Some(&condition))
        .await
        .map_err(map)?;
    Ok(())
}

/// Semantic recall: the `k` document chunks nearest `query` *within* `tenant`,
/// through `document_chunk_embedding_hnsw`.
///
/// The tenant filter is ANDed onto the KNN clause so the candidate set never
/// crosses a tenant boundary (and the engine ACL scopes it again) — but against
/// an index-backed KNN that filter is a RESIDUAL: the graph walk returns its
/// nearest neighbors across the whole table and the equality thins them
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
/// `content` best matches any word of `query_text`, tenant-scoped, in BM25
/// relevance order (see `crate::lexical`). An empty/blank query returns nothing.
async fn sparse_recall(
    store: &Store,
    tenant: &TenantId,
    query_text: &str,
    k: usize,
) -> Result<Vec<DocumentChunk>> {
    let rows: Vec<ChunkRow> =
        crate::lexical::any_word(store, TABLE, "*", tenant, query_text, k, None).await?;
    rows.into_iter().map(ChunkRow::into_domain).collect()
}

/// Whether any chunk of `title` is readable in `tenant` and `compartment`
/// (`None` = the shared pool) on this session. Ingest asks after it writes: under
/// a record session a write the engine refuses is silent, so without this an
/// ingest into a compartment the session cannot write to would report its chunks
/// as stored with none of them there.
pub async fn title_exists(
    store: &Store,
    tenant: &TenantId,
    title: &str,
    compartment: Option<&CompartmentId>,
) -> Result<bool> {
    let placed = match compartment {
        Some(c) => eq("compartment", c.as_str()),
        None => is_none("compartment"),
    };
    let q = Query::new()
        .select(None)
        .from_table(TABLE)
        .map_err(map)?
        .where_(and_(
            and_(eq("tenant_id", tenant.as_str()), eq("title", title)),
            placed,
        ))
        .limit(1)
        .map_err(map)?;
    let rows: Vec<ChunkRow> = query_records(store.client(), &q).await.map_err(map)?;
    Ok(!rows.is_empty())
}

/// Distinct document titles the tenant has ingested (the document list).
pub async fn list_titles(store: &Store, tenant: &TenantId) -> Result<Vec<String>> {
    Ok(summaries(store, tenant)
        .await?
        .into_iter()
        .map(|d| d.title)
        .collect())
}

/// One document as its chunks describe it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocumentSummary {
    pub title: String,
    /// How many chunks it was cut into.
    pub chunks: u32,
    /// Whether any of its chunks names an archived file of record (copal).
    pub archived: bool,
}

/// Every document the tenant holds, by title, with its chunk count, grouped
/// by the ENGINE.
///
/// Reading the chunks to count them is the trap [`super::memory::count`]
/// records: every chunk row carries its 384-float embedding, so listing a
/// large corpus that way drags millions of floats across the wire to produce
/// a few titles. Grouped where the rows live, only one row per title comes
/// back. Sorted by title.
pub async fn summaries(store: &Store, tenant: &TenantId) -> Result<Vec<DocumentSummary>> {
    #[derive(Deserialize)]
    struct Group {
        title: String,
        chunks: u64,
        #[serde(default)]
        archived: u64,
    }
    let query = Query::new()
        .select_expr(vec![
            field("title"),
            as_(&count_all(), "chunks"),
            as_(&count_if("copal_file != NONE"), "archived"),
        ])
        .from_table(TABLE)
        .map_err(map)?
        .where_(eq("tenant_id", tenant.as_str()))
        .group_by(["title"]);
    let groups: Vec<Group> = query_records(store.client(), &query).await.map_err(map)?;
    let mut documents: Vec<DocumentSummary> = groups
        .into_iter()
        .map(|g| DocumentSummary {
            title: g.title,
            chunks: g.chunks as u32,
            archived: g.archived > 0,
        })
        .collect();
    documents.sort_by(|a, b| a.title.cmp(&b.title));
    Ok(documents)
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
            copal_file: None,
            copal_digest: None,
            compartment: None,
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

        // The document list dedups the title across its chunks, and counts
        // them.
        assert_eq!(
            list_titles(&store, &tenant).await.unwrap(),
            vec!["onboarding guide".to_string()]
        );
        assert_eq!(
            summaries(&store, &tenant).await.unwrap(),
            vec![DocumentSummary {
                title: "onboarding guide".into(),
                chunks: 2,
                archived: false,
            }]
        );

        // Another tenant sees none of it (defense-in-depth filter; the engine
        // ACL scopes it too under a record session).
        let other = recall(&store, &TenantId::new("other"), &a, 2)
            .await
            .unwrap();
        assert!(other.is_empty());
    }

    #[tokio::test]
    async fn copal_provenance_round_trips_and_absent_stays_absent() {
        // A chunk stamped with document-of-record provenance keeps it across
        // the store; one ingested without an archive comes back with both
        // fields absent (never null), indistinguishable from a pre-copal row.
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        let tenant = TenantId::new("t");
        let mut a = vec![0.0f32; EMBED_DIM];
        a[0] = 1.0;
        let mut b = vec![0.0f32; EMBED_DIM];
        b[1] = 1.0;
        let archived = chunk("dc:arch", 0, a.clone()).with_copal("file:01J", "sha256:abc");
        insert_chunks(&store, &[archived, chunk("dc:bare", 1, b)])
            .await
            .unwrap();

        let got = recall(&store, &tenant, &a, 2).await.unwrap();
        assert_eq!(got.len(), 2);
        let by_id: HashMap<&str, &DocumentChunk> = got.iter().map(|c| (c.id.as_str(), c)).collect();
        assert_eq!(by_id["dc:arch"].copal_file.as_deref(), Some("file:01J"));
        assert_eq!(by_id["dc:arch"].copal_digest.as_deref(), Some("sha256:abc"));
        assert_eq!(by_id["dc:bare"].copal_file, None);
        assert_eq!(by_id["dc:bare"].copal_digest, None);
    }
}

#[cfg(test)]
mod delete_title_tests {
    use super::*;
    use crate::schema::EMBED_DIM;
    use chrono::Utc;

    fn chunk(id: &str, tenant: &str, title: &str) -> DocumentChunk {
        DocumentChunk {
            id: DocumentChunkId::new(id),
            tenant: TenantId::new(tenant),
            title: title.into(),
            source: None,
            ordinal: 0,
            content: title.into(),
            embedding: Some(vec![0.0; EMBED_DIM]),
            created_at: Utc::now(),
            copal_file: None,
            copal_digest: None,
            compartment: None,
        }
    }

    /// Each title is one row, counted per tenant, and archived when any of
    /// its chunks names a file of record.
    #[tokio::test]
    async fn summaries_count_each_titles_chunks_in_its_tenant() {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        let mut archived = chunk("docchunk:b0", "ws:t", "b");
        archived.copal_file = Some("file:b".into());
        insert_chunks(
            &store,
            &[
                chunk("docchunk:a0", "ws:t", "a"),
                chunk("docchunk:a1", "ws:t", "a"),
                chunk("docchunk:a2", "ws:t", "a"),
                archived,
                chunk("docchunk:b1", "ws:t", "b"),
                chunk("docchunk:ua", "ws:u", "a"),
            ],
        )
        .await
        .unwrap();
        let summary = |title: &str, chunks: u32, archived: bool| DocumentSummary {
            title: title.into(),
            chunks,
            archived,
        };
        assert_eq!(
            summaries(&store, &TenantId::new("ws:t")).await.unwrap(),
            vec![summary("a", 3, false), summary("b", 2, true)]
        );
        assert_eq!(
            summaries(&store, &TenantId::new("ws:u")).await.unwrap(),
            vec![summary("a", 1, false)]
        );
        assert!(summaries(&store, &TenantId::new("ws:none"))
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn delete_title_drops_only_that_title_in_that_tenant() {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        insert_chunks(
            &store,
            &[
                chunk("docchunk:a0", "ws:t", "a"),
                chunk("docchunk:a1", "ws:t", "a"),
                chunk("docchunk:b0", "ws:t", "b"),
                chunk("docchunk:ua", "ws:u", "a"),
            ],
        )
        .await
        .unwrap();
        delete_title(&store, &TenantId::new("ws:t"), "a", None)
            .await
            .unwrap();
        assert_eq!(
            list_titles(&store, &TenantId::new("ws:t")).await.unwrap(),
            vec!["b".to_string()]
        );
        assert_eq!(
            list_titles(&store, &TenantId::new("ws:u")).await.unwrap(),
            vec!["a".to_string()],
            "another tenant's document of the same title is untouched"
        );
    }
}

#[cfg(test)]
mod existing_database_tests {
    use super::*;
    use crate::repo::{compartment, principal};
    use antumbra_core::{Compartment, UserId};
    use chrono::Utc;
    use surql::schema::table::{table_schema, TableMode};

    const DIM: usize = 4;

    fn chunk(id: &str, title: &str, content: &str, compartment: Option<&str>) -> DocumentChunk {
        let c = DocumentChunk {
            id: DocumentChunkId::new(id),
            tenant: TenantId::new("ws:org"),
            title: title.into(),
            source: None,
            ordinal: 0,
            content: content.into(),
            embedding: Some(vec![1.0, 0.0, 0.0, 0.0]),
            created_at: Utc::now(),
            copal_file: None,
            copal_digest: None,
            compartment: None,
        };
        match compartment {
            Some(comp) => c.in_compartment(comp),
            None => c,
        }
    }

    async fn recallable(store: &Store, tenant: &TenantId) -> Result<Vec<String>> {
        let mut seen: Vec<String> = recall(store, tenant, &[1.0, 0.0, 0.0, 0.0], 10)
            .await?
            .into_iter()
            .map(|c| c.content)
            .collect();
        seen.sort();
        Ok(seen)
    }

    /// The case a fresh-store test cannot see. A database that predates private
    /// documents already has `document_chunk`, defined with the tenant-wide rule,
    /// and the schema is applied `IF NOT EXISTS`: left at that, the new rule would
    /// never reach it, however green the suite. Reconnecting has to bring the
    /// rule, and must leave the rows and the vector index as they were.
    #[tokio::test]
    async fn an_existing_database_picks_up_the_rule_and_keeps_its_data() -> Result<()> {
        let store = Store::connect_memory(DIM).await?;
        let tenant = TenantId::new("ws:org");
        let lily = UserId::new("user:lily");
        let oslo = UserId::new("user:oslo");
        principal::provision(&store, &tenant, &lily).await?;
        principal::provision(&store, &tenant, &oslo).await?;
        compartment::create(
            &store,
            &Compartment::new(
                CompartmentId::new("comp:lily-private"),
                tenant.clone(),
                lily.clone(),
                "lily private",
                Utc::now(),
            ),
        )
        .await?;

        // Put the table back the way an older build defined it.
        let as_it_was = table_schema(TABLE)
            .with_mode(TableMode::Schemaless)
            .with_permissions([
                ("select", "tenant_id = $auth.tenant"),
                ("create", "tenant_id = $auth.tenant"),
                ("update", "tenant_id = $auth.tenant"),
                ("delete", "tenant_id = $auth.tenant"),
            ])
            .to_surql_overwrite();
        store.client().query(&as_it_was).await.map_err(map)?;
        insert_chunks(
            &store,
            &[
                chunk(
                    "docchunk:private",
                    "review",
                    "lily's salary review",
                    Some("comp:lily-private"),
                ),
                chunk("docchunk:pool", "handbook", "the team handbook", None),
            ],
        )
        .await?;

        // Under the old rule oslo reads lily's document: this is the hole.
        store.signin(&tenant, &oslo).await?;
        assert_eq!(recallable(&store, &tenant).await?.len(), 2);

        // A newer build connects and applies its schema.
        store.invalidate().await?;
        store.ensure_schema().await?;

        store.signin(&tenant, &oslo).await?;
        assert_eq!(
            recallable(&store, &tenant).await?,
            vec!["the team handbook".to_string()],
            "the rule reached the existing table"
        );
        store.invalidate().await?;
        assert_eq!(
            recallable(&store, &tenant).await?.len(),
            2,
            "both rows are still there and still recallable through the vector index"
        );
        Ok(())
    }
}
