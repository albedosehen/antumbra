//! Penumbra memory repository: tenant-isolated traces (the consolidation
//! source for the counterfactual boundary and the learned gate).
//!
//! Isolation is engine-enforced: the `memory` table carries a row-level
//! `PERMISSIONS ... WHERE tenant_id = $auth.tenant` clause, so once a per-tenant
//! `ScopeCredentials` session binds `$auth.tenant`, the engine refuses any read
//! whose `tenant_id` does not match; a forgotten filter cannot leak. This repo
//! *also* applies an explicit `WHERE tenant_id = ...` as the documented second
//! layer (defense-in-depth), and re-checks `tenant_id` on a point read by id (so
//! another tenant's trace reads as absent). Built on surql-rs builders + `crud`;
//! no raw SurrealQL.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::Instrument as _;

use surql::query::builder::Query;
use surql::query::crud::{delete_records, get_record, merge_record, query_records, upsert_record};
use surql::query::expressions::{field, value};
use surql::types::operators::{and_, contains_any, eq, is_none, is_not_none, lt};
use surql::types::RecordID;

use antumbra_core::calibrate::calibrated_score;
use antumbra_core::{
    CompartmentId, ExpertId, Memory, MemoryId, MemoryNetwork, MemoryStatus, Result, TenantId,
    UserId,
};

use crate::dto::parse_dt;
use crate::error::map;
use crate::fusion::{rrf_fuse, DEFAULT_RRF_K};
use crate::knn::{candidate_pool, search_effort};
use crate::repo::sync::{ChangeAction, ChangeEvent};
use crate::store::Store;

const TABLE: &str = "memory";

#[derive(Serialize, Deserialize)]
pub(crate) struct MemoryRow {
    key: String,
    tenant_id: String,
    network: MemoryNetwork,
    content: String,
    #[serde(default)]
    embedding: Option<Vec<f32>>,
    #[serde(default)]
    confidence: f32,
    #[serde(default)]
    reinforcement: u32,
    #[serde(default)]
    evidence: Vec<String>,
    #[serde(default)]
    volatile: bool,
    #[serde(default)]
    consolidated_expert: Option<String>,
    // Absent (not null) when None so the engine sees `compartment = NONE` for
    // un-compartmentalized memories (the shared tenant pool).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    compartment: Option<String>,
    #[serde(default)]
    author: Option<String>,
    #[serde(default)]
    author_host: Option<String>,
    #[serde(default)]
    status: MemoryStatus,
    created_at: String,
    updated_at: String,
    // Tombstone marker. Absent (NONE) for a live trace so read paths can test it
    // cheaply; an RFC3339 timestamp once forgotten.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) deleted_at: Option<String>,
}

impl MemoryRow {
    fn from_domain(m: &Memory) -> Self {
        MemoryRow {
            key: m.id.as_str().to_string(),
            tenant_id: m.tenant.as_str().to_string(),
            network: m.network,
            content: m.content.clone(),
            embedding: m.embedding.clone(),
            confidence: m.confidence,
            reinforcement: m.reinforcement,
            evidence: m.evidence.clone(),
            volatile: m.volatile,
            consolidated_expert: m
                .consolidated_expert
                .as_ref()
                .map(|e| e.as_str().to_string()),
            compartment: m.compartment.as_ref().map(|c| c.as_str().to_string()),
            author: m.author.as_ref().map(|u| u.as_str().to_string()),
            author_host: m.author_host.clone(),
            status: m.status,
            created_at: m.created_at.to_rfc3339(),
            updated_at: m.updated_at.to_rfc3339(),
            deleted_at: m.deleted_at.map(|t| t.to_rfc3339()),
        }
    }

    pub(crate) fn into_domain(self) -> Result<Memory> {
        Ok(Memory {
            id: MemoryId::new(self.key),
            tenant: TenantId::new(self.tenant_id),
            network: self.network,
            content: self.content,
            embedding: self.embedding,
            confidence: self.confidence,
            reinforcement: self.reinforcement,
            evidence: self.evidence,
            volatile: self.volatile,
            consolidated_expert: self.consolidated_expert.map(ExpertId::new),
            compartment: self.compartment.map(CompartmentId::new),
            author: self.author.map(UserId::new),
            author_host: self.author_host,
            status: self.status,
            created_at: parse_dt(&self.created_at)?,
            updated_at: parse_dt(&self.updated_at)?,
            deleted_at: self.deleted_at.as_deref().map(parse_dt).transpose()?,
        })
    }
}

/// Insert or replace a memory (addressed by its globally-unique id).
pub async fn upsert(store: &Store, memory: &Memory) -> Result<()> {
    let id = RecordID::<()>::new(TABLE, memory.id.as_str()).map_err(map)?;
    // Whether this write creates the memory, read only to announce it.
    let created =
        store.announcing() && matches!(get(store, &memory.tenant, &memory.id).await, Ok(None));
    let data: Value = serde_json::to_value(MemoryRow::from_domain(memory))?;
    upsert_record(store.client(), &id, data)
        .await
        .map_err(map)?;
    let action = if created {
        ChangeAction::Create
    } else {
        ChangeAction::Update
    };
    announce(store, action, Some(memory));
    Ok(())
}

/// Fetch one memory by id, but only if it belongs to `tenant`; a trace owned by
/// another tenant reads as absent (the defensive isolation re-check).
pub async fn get(store: &Store, tenant: &TenantId, id: &MemoryId) -> Result<Option<Memory>> {
    let rid = RecordID::<()>::new(TABLE, id.as_str()).map_err(map)?;
    match get_record(store.client(), &rid).await.map_err(map)? {
        Some(value) => {
            let row: MemoryRow = serde_json::from_value(value)?;
            // Another tenant's trace, or a tombstone, reads as absent.
            if row.tenant_id != tenant.as_str() || row.deleted_at.is_some() {
                return Ok(None);
            }
            Ok(Some(row.into_domain()?))
        }
        None => Ok(None),
    }
}

/// All of a tenant's memories (tenant-filtered).
pub async fn list(store: &Store, tenant: &TenantId) -> Result<Vec<Memory>> {
    // Paged by id (see `Store::read_paged`): a tenant's whole population can be
    // thousands of embedding-carrying rows. Order is unspecified, so id-paging
    // needs no re-sort.
    let filter = eq("tenant_id", tenant.as_str());
    let rows: Vec<MemoryRow> = store.read_paged(TABLE, None, Some(&filter)).await?;
    rows.into_iter()
        .filter(|r| r.deleted_at.is_none()) // hide tombstones (forgotten traces)
        .map(MemoryRow::into_domain)
        .collect()
}

/// How many live memories `tenant` holds, counted by the ENGINE.
///
/// `list(store, tenant).await?.len()` is the obvious way to get this and is a
/// trap: it materializes every row, and a `MemoryRow` carries its 384-float
/// `embedding`. Counting a 5538-memory tenant that way drags roughly 2.1 million
/// floats across `ws://` to produce one integer, which is how `workspace_stats`
/// came to fail with "connection error: Connection reset" on a store where every
/// other tool worked -- it had been fine at 9 memories and could not survive the
/// migrated corpus. The same hazard is already recorded on
/// [`all_unscoped_lite`]; this is the counting case of it.
///
/// Tombstones are excluded here rather than in Rust, because the whole point is
/// that no row crosses the wire.
pub async fn count(store: &Store, tenant: &TenantId) -> Result<u32> {
    #[derive(Deserialize)]
    struct CountRow {
        count: u64,
    }
    let query = Query::new()
        .select(Some(vec!["count()".to_string()]))
        .from_table(TABLE)
        .map_err(map)?
        .where_(and_(
            eq("tenant_id", tenant.as_str()),
            is_none("deleted_at"),
        ))
        .group_all();
    let rows: Vec<CountRow> = query_records(store.client(), &query).await.map_err(map)?;
    // `GROUP ALL` over an empty match set returns no row at all, not a zero.
    Ok(rows.first().map_or(0, |r| r.count as u32))
}

/// Where a memory is anchored, and nothing else of it: enough to decide which
/// memories a provenance event touches (a merge, a deleted branch) without
/// reading every embedding and every body to find the few it does.
#[derive(Debug, Clone, PartialEq)]
pub struct Anchored {
    pub id: MemoryId,
    pub evidence: Vec<String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Deserialize)]
struct AnchoredRow {
    key: String,
    #[serde(default)]
    evidence: Vec<String>,
    created_at: String,
    #[serde(default)]
    deleted_at: Option<String>,
}

/// Rows per page in [`list_anchored`]. An anchored row is a few hundred bytes,
/// so a page this size stays within a few megabytes while holding a whole
/// workspace: on kuskokwim, 5,672 memories read in pages of 500 took twelve
/// queries and 3.7 seconds, nearly as long as reading them in full.
const ANCHORED_PAGE_ROWS: i64 = 10_000;

/// A tenant's live memories as [`Anchored`]. A full row carries a 384-float
/// embedding and the whole content, so a workspace of 5,672 memories is tens of
/// megabytes to read; this projection is a few hundred bytes a row.
pub async fn list_anchored(store: &Store, tenant: &TenantId) -> Result<Vec<Anchored>> {
    let fields: Vec<String> = ["key", "evidence", "created_at", "deleted_at"]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    let filter = eq("tenant_id", tenant.as_str());
    let rows: Vec<AnchoredRow> = store
        .read_in_pages_of(ANCHORED_PAGE_ROWS, TABLE, Some(fields), Some(&filter))
        .await?;
    rows.into_iter()
        .filter(|r| r.deleted_at.is_none())
        .map(|r| {
            Ok(Anchored {
                id: MemoryId::new(r.key),
                evidence: r.evidence,
                created_at: parse_dt(&r.created_at)?,
            })
        })
        .collect()
}

/// Every memory across all tenants, with NO tenant filter. As an owner/root
/// session this is the cross-tenant view (profiling / training across tenants);
/// as a tenant-authenticated session the engine's row-level PERMISSIONS still
/// scope the result to that tenant, which is precisely the engine-enforcement
/// guarantee (isolation holds even with no app-side WHERE).
pub async fn all_unscoped(store: &Store) -> Result<Vec<Memory>> {
    // Paged by id (see `Store::read_paged`): the cross-tenant population is the
    // largest read in the store -- thousands of embedding-carrying rows that
    // would overflow one `ws://` frame. No app-side filter (the engine ACL still
    // scopes a tenant session); order is unspecified, so no re-sort.
    let rows: Vec<MemoryRow> = store.read_paged(TABLE, None, None).await?;
    rows.into_iter()
        .filter(|r| r.deleted_at.is_none()) // hide tombstones (forgotten traces)
        .map(MemoryRow::into_domain)
        .collect()
}

/// Like [`all_unscoped`] but WITHOUT each memory's embedding vector. The operator
/// console loads the whole population to show it, yet never displays the vectors;
/// a full store's worth of 384-float embeddings is megabytes that stalls a
/// `ws://` client, so omit them here. Recall still uses the indexed vectors.
pub async fn all_unscoped_lite(store: &Store) -> Result<Vec<Memory>> {
    let fields = without_embedding();
    // Paged by id (see `Store::read_paged`): the whole population, minus the
    // vectors. Dropping the embeddings shrinks each row, but the row *count* is
    // still the full store, so the frame can still overflow without paging.
    let rows: Vec<MemoryRow> = store.read_paged(TABLE, Some(fields), None).await?;
    rows.into_iter()
        .filter(|r| r.deleted_at.is_none())
        .map(MemoryRow::into_domain)
        .collect()
}

/// Every [`MemoryRow`] field but `embedding`, for a read that never uses the
/// vector: the row's serde default leaves it `None`.
pub(crate) fn without_embedding() -> Vec<String> {
    [
        "key",
        "tenant_id",
        "network",
        "content",
        "confidence",
        "reinforcement",
        "evidence",
        "volatile",
        "consolidated_expert",
        "compartment",
        "author",
        "author_host",
        "status",
        "created_at",
        "updated_at",
        "deleted_at",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect()
}

/// A page of `tenant`'s live memories, most recently updated first: `limit`
/// of them after the first `offset`, in one network when one is given, and
/// written from one machine when `host` names it.
///
/// Ordered and cut by the engine, and read without the embeddings, so a page
/// costs its own rows however large the store is. [`list`] reads every row,
/// vector and all, to answer the same question about the first few.
pub async fn recent(
    store: &Store,
    tenant: &TenantId,
    network: Option<MemoryNetwork>,
    host: Option<&str>,
    limit: u32,
    offset: u32,
) -> Result<Vec<Memory>> {
    let live = and_(eq("tenant_id", tenant.as_str()), is_none("deleted_at"));
    let filter = match network {
        Some(n) => and_(live, eq("network", n.as_str())),
        None => live,
    };
    let filter = match host {
        Some(h) => and_(filter, written_from(h)),
        None => filter,
    };
    let query = Query::new()
        .select(Some(without_embedding()))
        .from_table(TABLE)
        .map_err(map)?
        .where_(filter)
        .order_by("updated_at", "DESC")
        .map_err(map)?
        .limit(i64::from(limit))
        .map_err(map)?
        .offset(i64::from(offset))
        .map_err(map)?;
    let rows: Vec<MemoryRow> = query_records(store.client(), &query).await.map_err(map)?;
    rows.into_iter().map(MemoryRow::into_domain).collect()
}

/// The memories written from `host`, compared as host names are everywhere
/// else (`antumbra_core::handoff::normalize_host`): trimmed and lowercased on
/// both sides. A server stamps its own name as the operating system gives it,
/// `KUSKOKWIM` on Windows, so an exact match would miss the very rows a
/// machine running its own server wrote. A row with no host is not written
/// from anywhere, and `?? ''` keeps the function from failing on it.
fn written_from(host: &str) -> surql::types::operators::Operator {
    surql::types::operators::eq_expr(
        "string::lowercase(string::trim(author_host ?? ''))",
        value(host.trim().to_lowercase()),
    )
}

/// `tenant`'s live memories carrying any of `entries` as an evidence entry,
/// matched whole and filtered by the engine, read without the embeddings.
///
/// The evidence graph (ADR-0019) finds its edges this way: every edge carries
/// exactly one `dep-source:<source>` entry, so asking for the four of them
/// returns the edges and nothing else, without reading every memory of a
/// large store to find the few that are edges.
pub async fn with_any_evidence(
    store: &Store,
    tenant: &TenantId,
    entries: &[String],
) -> Result<Vec<Memory>> {
    if entries.is_empty() {
        return Ok(Vec::new());
    }
    let filter = and_(
        and_(eq("tenant_id", tenant.as_str()), is_none("deleted_at")),
        contains_any("evidence", entries.iter().map(|e| Value::String(e.clone()))),
    );
    let rows: Vec<MemoryRow> = store
        .read_paged(TABLE, Some(without_embedding()), Some(&filter))
        .await?;
    rows.into_iter().map(MemoryRow::into_domain).collect()
}

/// A compartment's memories (the corpus for per-compartment consolidation,
/// across compartments, the latent-spaces of the memory store). Tenant + compartment filtered; the engine ACL also applies
/// under a tenant session.
pub async fn list_by_compartment(
    store: &Store,
    tenant: &TenantId,
    compartment: &CompartmentId,
) -> Result<Vec<Memory>> {
    // Paged by id (see `Store::read_paged`); the existing tenant+compartment
    // filter is preserved. Order is unspecified, so no re-sort.
    let filter = and_(
        eq("tenant_id", tenant.as_str()),
        eq("compartment", compartment.as_str()),
    );
    let rows: Vec<MemoryRow> = store.read_paged(TABLE, None, Some(&filter)).await?;
    rows.into_iter()
        .filter(|r| r.deleted_at.is_none()) // hide tombstones (forgotten traces)
        .map(MemoryRow::into_domain)
        .collect()
}

/// A tenant's memories in one network (tenant + network filtered).
pub async fn list_by_network(
    store: &Store,
    tenant: &TenantId,
    network: MemoryNetwork,
) -> Result<Vec<Memory>> {
    // Paged by id (see `Store::read_paged`); the existing tenant+network filter
    // is preserved. Order is unspecified, so no re-sort.
    let filter = and_(
        eq("tenant_id", tenant.as_str()),
        eq("network", network.as_str()),
    );
    let rows: Vec<MemoryRow> = store.read_paged(TABLE, None, Some(&filter)).await?;
    rows.into_iter()
        .filter(|r| r.deleted_at.is_none()) // hide tombstones (forgotten traces)
        .map(MemoryRow::into_domain)
        .collect()
}

/// Semantic recall: the `k` nearest memories to `query` *within* `tenant` (and
/// optionally one network), through `memory_embedding_hnsw`.
///
/// The tenant filter is ANDed onto the KNN clause so the candidate set never
/// crosses a tenant boundary — but against an index-backed KNN both that filter
/// and the tombstone check below it are RESIDUAL: the graph walk returns its
/// nearest neighbours across the whole table and everything else thins them
/// afterwards. So the walk is asked for a wider pool and the answer is
/// truncated to `k` once the thinning is done. Asking for `k` directly would
/// hand a tenant with a small share of the table a short answer, and a tenant
/// whose nearest neighbours are all tombstones an empty one.
pub async fn recall(
    store: &Store,
    tenant: &TenantId,
    query: &[f32],
    k: usize,
    network: Option<MemoryNetwork>,
) -> Result<Vec<Memory>> {
    let vector: Vec<f64> = query.iter().map(|&x| f64::from(x)).collect();
    let condition = match network {
        Some(net) => and_(
            eq("tenant_id", tenant.as_str()),
            eq("network", net.as_str()),
        ),
        None => eq("tenant_id", tenant.as_str()),
    };
    let pool = candidate_pool(k);
    let q = Query::new()
        .select(None)
        .from_table(TABLE)
        .map_err(map)?
        .where_(condition)
        .vector_search_indexed("embedding", vector, pool as i64, search_effort(pool))
        .map_err(map)?;
    let rows: Vec<MemoryRow> = query_records(store.client(), &q).await.map_err(map)?;
    rows.into_iter()
        .filter(|r| r.deleted_at.is_none()) // hide tombstones (forgotten traces)
        .take(k)
        .map(MemoryRow::into_domain)
        .collect()
}

/// The target of the spans around [`recall_hybrid`]'s legs. antumbra-mcp's
/// trace filter lets it through and nothing else from this crate, so these
/// spans are what it exports of the store.
const RECALL_SPAN_TARGET: &str = "antumbra_store::recall";

/// Hybrid recall for several queries at once, fused by rank: the whole of a
/// prompt and each part of one that asks for more than one thing
/// (`antumbra_core::query::parts`). Each query runs as [`recall_hybrid`] does,
/// and the lists are fused by Reciprocal Rank Fusion, so each part's best
/// matches stand beside the whole prompt's before the top `k` are kept. With
/// one query it is [`recall_hybrid`].
pub async fn recall_hybrid_many(
    store: &Store,
    tenant: &TenantId,
    queries: &[(String, Vec<f32>)],
    k: usize,
    network: Option<MemoryNetwork>,
    probes: &[Vec<f32>],
) -> Result<Vec<Memory>> {
    // Each query's recall runs at once with the others', as its legs do. They
    // rank keys, so the rows read whole are the k kept, once for every query.
    let mut rankings = futures::future::try_join_all(
        queries
            .iter()
            .map(|(text, vector)| hybrid_keys(store, tenant, text, vector, k, network, probes)),
    )
    .await?;
    let keys = if rankings.len() == 1 {
        rankings.swap_remove(0)
    } else {
        rrf_fuse(&rankings, DEFAULT_RRF_K)
            .into_iter()
            .take(k)
            .collect()
    };
    rows_in_order(store, tenant, &keys).await
}

/// Hybrid recall: fuse the dense (HNSW vector) and sparse (BM25 full-text) legs
/// over `query_text` and its `query_vec` embedding via Reciprocal Rank Fusion,
/// returning the top `k` memories for `tenant` (optionally one network).
///
/// The dense leg finds semantically-near traces; the sparse leg catches the
/// exact tokens (identifiers, error codes, tickers) a 384-d vector silently
/// drops. The sparse leg is best-effort: if the full-text query errors (e.g. the
/// index is still building, or an older store predates it), recall degrades to
/// dense-only rather than failing. A blank `query_text` is dense-only by design.
/// `probes` calibrates the DENSE leg: see [`antumbra_core::calibrate`]. Pass an
/// empty slice to rank it by raw cosine, which is what every caller did before
/// the calibration existed and what a caller with no embedder still does.
pub async fn recall_hybrid(
    store: &Store,
    tenant: &TenantId,
    query_text: &str,
    query_vec: &[f32],
    k: usize,
    network: Option<MemoryNetwork>,
    probes: &[Vec<f32>],
) -> Result<Vec<Memory>> {
    let keys = hybrid_keys(store, tenant, query_text, query_vec, k, network, probes).await?;
    rows_in_order(store, tenant, &keys).await
}

/// [`recall_hybrid`]'s ranking as memory keys, best first. Each leg ranks keys
/// with their scores and leaves the rows in the engine, and only the `k` kept
/// are read whole, by the caller. On kuskokwim a recall read about 6 MB of
/// rows, mostly embeddings, to keep 3 to 12 memories, and decoding them, not
/// the engine, was where its time went: the dense leg took 30 ms in the engine
/// and 200 ms in the leg.
async fn hybrid_keys(
    store: &Store,
    tenant: &TenantId,
    query_text: &str,
    query_vec: &[f32],
    k: usize,
    network: Option<MemoryNetwork>,
    probes: &[Vec<f32>],
) -> Result<Vec<String>> {
    // Pull a wider candidate pool from each leg than the final k, so fusion has
    // room to reorder before truncating. The same sizing the dense leg uses
    // against its own residual filters, for the same reason.
    let pool = candidate_pool(k);
    // Each read runs in a span of its own under the caller's, so a slow recall
    // says which leg took the time. The leg and the pool size, never the query.
    let leg = |name: &'static str| {
        tracing::info_span!(
            target: RECALL_SPAN_TARGET,
            "recall leg",
            otel.name = name,
            antumbra.recall.pool = pool,
        )
    };

    // The three legs read independently of one another, so they run at once:
    // a recall costs its slowest leg rather than the sum of them.
    let (dense, chunk_leg, sparse) = futures::join!(
        dense_scored(store, tenant, query_vec, pool, network, probes)
            .instrument(leg("recall dense")),
        chunk_scored(store, tenant, query_vec, pool, network, probes)
            .instrument(leg("recall chunks")),
        sparse_keys(store, tenant, query_text, pool, network).instrument(leg("recall lexical")),
    );
    let dense = dense?;
    // The chunk index (ADR-0025): each memory scores by its best vector, its
    // whole one or one of its pieces', through the same calibration. A memory
    // with no chunks yet scores by its whole vector as before, so the index can
    // be empty, partial or rebuilt; a failing chunk leg leaves the whole-memory
    // list as it was.
    let dense: Vec<String> = match chunk_leg {
        Ok(pieces) if !pieces.is_empty() => {
            best_of(store, tenant, dense, pieces, query_vec, probes, pool)
                .instrument(leg("recall chunk scores"))
                .await?
        }
        Ok(_) => dense.into_iter().map(|s| s.key).collect(),
        Err(e) => {
            eprintln!(
                "antumbra-store: chunk leg failed, the dense leg reads whole memories only: {e}"
            );
            dense.into_iter().map(|s| s.key).collect()
        }
    };
    // Best-effort, but never SILENTLY: a failing lexical leg degrades recall to
    // dense-only, which looks exactly like a ranking quirk from the outside. That
    // is not hypothetical -- a missing `ORDER BY` in this very function went
    // unnoticed for months because the symptom was indistinguishable from "the
    // embedding is bad at this query". Say so, the way the rerank stage already
    // says so when it falls back to RRF order.
    let sparse = match sparse {
        Ok(keys) => keys,
        Err(e) => {
            eprintln!("antumbra-store: lexical leg failed, recall is dense-only: {e}");
            Vec::new()
        }
    };

    // Nothing lexical to fuse: dense already is the answer (and `rrf_fuse` over a
    // single list is order-preserving, but skip the allocation).
    if sparse.is_empty() {
        return Ok(dense.into_iter().take(k).collect());
    }
    Ok(rrf_fuse(&[dense, sparse], DEFAULT_RRF_K)
        .into_iter()
        .take(k)
        .collect())
}

/// A memory a vector leg found, by key, and how near the query it is.
struct Scored {
    key: String,
    score: f32,
}

/// The dense leg as keys with scores: the `k` live memories nearest
/// `query_vec`, nearest first. It asks the index for the pool [`recall`] asks
/// for, widened against tombstones, but reads for each row only its key, its
/// tombstone and its cosine, which is one minus the distance the engine has
/// from walking the COSINE index. With `probes` the score is calibrated
/// instead (re-ranking the pool here, before fusion: sorting the FUSED list by
/// a dense score would throw away the lexical leg), and that reads the vectors.
/// The server passes none (ADR-0026: on the user's store the calibration cost
/// more recall than it saved); the bench can.
async fn dense_scored(
    store: &Store,
    tenant: &TenantId,
    query_vec: &[f32],
    k: usize,
    network: Option<MemoryNetwork>,
    probes: &[Vec<f32>],
) -> Result<Vec<Scored>> {
    let calibrating = !probes.is_empty();
    let fields = if calibrating {
        "key, deleted_at, embedding"
    } else {
        "key, deleted_at, vector::distance::knn() AS distance"
    };
    let pool = candidate_pool(k);
    let surql = format!(
        "SELECT {fields} FROM {TABLE} WHERE tenant_id = $tenant{} \
         AND embedding <|{pool},{}|> $vector",
        if network.is_some() {
            " AND network = $network"
        } else {
            ""
        },
        search_effort(pool)
    );
    #[derive(Deserialize)]
    struct Row {
        key: String,
        #[serde(default)]
        deleted_at: Option<String>,
        #[serde(default)]
        distance: Option<f64>,
        #[serde(default)]
        embedding: Option<Vec<f32>>,
    }
    let rows: Vec<Row> = store
        .query_rows(&surql, leg_vars(tenant, query_vec, network))
        .await?;
    let mut hits: Vec<Scored> = rows
        .into_iter()
        .filter(|r| r.deleted_at.is_none()) // hide tombstones (forgotten traces)
        .take(k)
        .map(|r| {
            let score = match &r.embedding {
                Some(e) => calibrated_score(query_vec, e, probes),
                None => r.distance.map_or(f32::MIN, |d| (1.0 - d) as f32),
            };
            Scored { key: r.key, score }
        })
        .collect();
    if calibrating {
        hits.sort_by(|a, b| b.score.total_cmp(&a.score));
    }
    Ok(hits)
}

/// The chunk leg as `(memory key, score)` pieces, nearest first: the cosine the
/// engine has from the index, or, with `probes`, the calibrated score, which
/// reads each piece's vector.
async fn chunk_scored(
    store: &Store,
    tenant: &TenantId,
    query_vec: &[f32],
    k: usize,
    network: Option<MemoryNetwork>,
    probes: &[Vec<f32>],
) -> Result<Vec<(String, f32)>> {
    if probes.is_empty() {
        return crate::repo::memory_chunk::nearest_scored(store, tenant, query_vec, k, network)
            .await;
    }
    let pieces = crate::repo::memory_chunk::nearest(store, tenant, query_vec, k, network).await?;
    Ok(pieces
        .into_iter()
        .map(|(memory, v)| (memory, calibrated_score(query_vec, &v, probes)))
        .collect())
}

/// The lexical leg as keys, best first: [`sparse_recall`]'s ranking, reading
/// for each row only its key and its tombstone.
async fn sparse_keys(
    store: &Store,
    tenant: &TenantId,
    query_text: &str,
    k: usize,
    network: Option<MemoryNetwork>,
) -> Result<Vec<String>> {
    #[derive(Deserialize)]
    struct Row {
        key: String,
        #[serde(default)]
        deleted_at: Option<String>,
    }
    let network = network.map(|n| n.as_str());
    let rows: Vec<Row> = crate::lexical::any_word(
        store,
        TABLE,
        "key, deleted_at",
        tenant,
        query_text,
        k,
        network,
    )
    .await?;
    Ok(rows
        .into_iter()
        .filter(|r| r.deleted_at.is_none())
        .map(|r| r.key)
        .collect())
}

/// The variables a vector leg binds: the tenant, the query vector and, when
/// given, the network.
fn leg_vars(
    tenant: &TenantId,
    query_vec: &[f32],
    network: Option<MemoryNetwork>,
) -> std::collections::BTreeMap<String, Value> {
    let mut vars = std::collections::BTreeMap::from([
        ("tenant".to_string(), serde_json::json!(tenant.as_str())),
        ("vector".to_string(), serde_json::json!(query_vec)),
    ]);
    if let Some(net) = network {
        vars.insert("network".to_string(), serde_json::json!(net.as_str()));
    }
    vars
}

/// Rank the dense leg's memories and the memories the chunk leg found together,
/// each by its best score: its whole vector's or its nearest piece's. Scores,
/// not ranks, so a piece that is not near the query cannot tie a whole memory
/// that is. A memory only the chunk leg found gets its whole score from
/// [`whole_scores`], which leaves out one forgotten since its chunks were cut,
/// so it drops out here.
async fn best_of(
    store: &Store,
    tenant: &TenantId,
    dense: Vec<Scored>,
    pieces: Vec<(String, f32)>,
    query_vec: &[f32],
    probes: &[Vec<f32>],
    k: usize,
) -> Result<Vec<String>> {
    let mut best_piece: HashMap<String, f32> = HashMap::new();
    for (memory, s) in pieces {
        best_piece
            .entry(memory)
            .and_modify(|b| *b = b.max(s))
            .or_insert(s);
    }
    let mut whole: HashMap<String, f32> = dense.into_iter().map(|s| (s.key, s.score)).collect();
    let missing: Vec<String> = best_piece
        .keys()
        .filter(|key| !whole.contains_key(*key))
        .cloned()
        .collect();
    whole.extend(whole_scores(store, tenant, &missing, query_vec, probes).await?);
    let mut scored: Vec<(f32, String)> = whole
        .into_iter()
        .map(|(key, w)| {
            let best = best_piece.get(&key).map_or(w, |p| p.max(w));
            (best, key)
        })
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    Ok(scored.into_iter().take(k).map(|(_, key)| key).collect())
}

/// The whole-vector score of each live memory among `keys`, by key: its cosine
/// to `query_vec`, taken in the engine, or with `probes` its calibrated score,
/// which reads the vectors. A forgotten memory is left out; one with no vector
/// scores lowest, as it would unscored.
async fn whole_scores(
    store: &Store,
    tenant: &TenantId,
    keys: &[String],
    query_vec: &[f32],
    probes: &[Vec<f32>],
) -> Result<HashMap<String, f32>> {
    if keys.is_empty() {
        return Ok(HashMap::new());
    }
    let fields = if probes.is_empty() {
        "key, IF embedding != NONE THEN vector::similarity::cosine(embedding, $vector) END AS score"
    } else {
        "key, embedding"
    };
    let surql = format!(
        "SELECT {fields} FROM $keys.map(|$k| type::record('{TABLE}', $k)) \
         WHERE tenant_id = $tenant AND deleted_at = NONE"
    );
    let mut vars = leg_vars(tenant, query_vec, None);
    vars.insert("keys".to_string(), serde_json::json!(keys));
    #[derive(Deserialize)]
    struct Row {
        key: String,
        #[serde(default)]
        score: Option<f64>,
        #[serde(default)]
        embedding: Option<Vec<f32>>,
    }
    let rows: Vec<Row> = store.query_rows(&surql, vars).await?;
    Ok(rows
        .into_iter()
        .map(|r| {
            let score = match &r.embedding {
                Some(e) => calibrated_score(query_vec, e, probes),
                None => r.score.map_or(f32::MIN, |s| s as f32),
            };
            (r.key, score)
        })
        .collect())
}

/// The live memories under `keys`, in that order, read whole: what recall
/// returns, embeddings included, since its caller reports each one's
/// similarity.
async fn rows_in_order(store: &Store, tenant: &TenantId, keys: &[String]) -> Result<Vec<Memory>> {
    let mut by_key: HashMap<String, Memory> = get_many(store, tenant, keys)
        .instrument(tracing::info_span!(
            target: RECALL_SPAN_TARGET,
            "recall leg",
            otel.name = "recall fetch",
            antumbra.recall.pool = keys.len(),
        ))
        .await?
        .into_iter()
        .map(|m| (m.id.as_str().to_string(), m))
        .collect();
    Ok(keys.iter().filter_map(|key| by_key.remove(key)).collect())
}

/// The live memories of `tenant` among `ids`, in no particular order.
pub async fn get_many(store: &Store, tenant: &TenantId, ids: &[String]) -> Result<Vec<Memory>> {
    let rows: Vec<MemoryRow> = store
        .rows_by_key(TABLE, ids, tenant, "deleted_at = NONE")
        .await?;
    rows.into_iter().map(MemoryRow::into_domain).collect()
}

/// The BM25 full-text (sparse) leg of [`recall_hybrid`]: the `k` memories whose
/// `content` best matches any word of `query_text`, tenant- (and optionally
/// network-)scoped, in the engine's BM25 relevance order (see
/// `crate::lexical`). An empty/blank query returns nothing. Public for the
/// bench, whose no-model baseline it is.
pub async fn sparse_recall(
    store: &Store,
    tenant: &TenantId,
    query_text: &str,
    k: usize,
    network: Option<MemoryNetwork>,
) -> Result<Vec<Memory>> {
    let network = network.map(|n| n.as_str());
    let rows: Vec<MemoryRow> =
        crate::lexical::any_word(store, TABLE, "*", tenant, query_text, k, network).await?;
    rows.into_iter()
        .filter(|r| r.deleted_at.is_none())
        .map(MemoryRow::into_domain)
        .collect()
}

/// Merge only the named fields of a memory row (a partial `UPDATE ... MERGE`),
/// leaving every other field as the engine already holds it. The read-modify-
/// write mutators below write *just* the fields they change this way rather than
/// rewriting the whole row, so a concurrent forget's `deleted_at` (set between
/// the read and this write) is never clobbered back to live: the resurrection-
/// safe alternative to a full-row upsert. The engine PERMISSIONS still gate the
/// update, exactly as the upsert did.
async fn merge_fields(store: &Store, id: &MemoryId, patch: Value) -> Result<()> {
    let rid = RecordID::<()>::new(TABLE, id.as_str()).map_err(map)?;
    merge_record(store.client(), &rid, patch)
        .await
        .map_err(map)?;
    Ok(())
}

/// Reinforce a trace (recurrence + confidence bump), tenant-checked. Returns the
/// updated memory, or `None` if it does not exist for this tenant or has been
/// forgotten.
///
/// One atomic, tombstone-guarded statement rather than a read-modify-write: the
/// increment and the confidence bump are computed server-side from the row's
/// *current* values (`reinforcement = reinforcement + 1`), so concurrent
/// reinforcements cannot lose an increment, and the `deleted_at IS NONE` guard
/// means a trace forgotten between any read and this write simply matches no row
/// -- it can never be resurrected. Mirrors `Memory::reinforce`; the confidence
/// formula is self-bounding for confidence in `[0, 1]`, so no clamp is needed.
pub async fn reinforce(
    store: &Store,
    tenant: &TenantId,
    id: &MemoryId,
    now: chrono::DateTime<chrono::Utc>,
) -> Result<Option<Memory>> {
    let target = RecordID::<()>::new(TABLE, id.as_str())
        .map_err(map)?
        .to_string();
    // confidence + (1 - confidence) * 0.25
    let confidence_bump = field("confidence") + (value(1) - field("confidence")) * 0.25;
    let q = Query::new()
        .update_set(target)
        .map_err(map)?
        .set_expr("reinforcement", field("reinforcement") + 1)
        .map_err(map)?
        .set_expr("confidence", confidence_bump)
        .map_err(map)?
        .set("updated_at", now.to_rfc3339())
        .map_err(map)?
        .where_(and_(
            eq("tenant_id", tenant.as_str()),
            is_none("deleted_at"),
        ))
        .return_after();
    let rows: Vec<MemoryRow> = query_records(store.client(), &q).await.map_err(map)?;
    let updated = rows
        .into_iter()
        .next()
        .map(MemoryRow::into_domain)
        .transpose()?;
    announce(store, ChangeAction::Update, updated.as_ref());
    Ok(updated)
}

/// Penalize a memory whose thesis was FALSIFIED: decay its confidence toward 0 -- the mirror
/// of [`reinforce`]. A falsified exemplar (e.g. a trade whose realized P&L was a loss) must not
/// clear the consolidation gate and graduate into an expert, so a penalty lowers CONFIDENCE
/// while leaving `reinforcement` (how often the pattern recurred) untouched. Same in-engine
/// atomicity + tenant + tombstone guards as [`reinforce`], so a cross-tenant or forgotten
/// memory matches no row (a penalty never resurrects a tombstone). `confidence * 0.75` stays
/// within `[0, 1]`, so no clamp is needed.
pub async fn penalize(
    store: &Store,
    tenant: &TenantId,
    id: &MemoryId,
    now: chrono::DateTime<chrono::Utc>,
) -> Result<Option<Memory>> {
    let target = RecordID::<()>::new(TABLE, id.as_str())
        .map_err(map)?
        .to_string();
    // confidence * 0.75 -- move 25% of the way toward 0 (the inverse of reinforce's 25% to 1).
    let confidence_decay = field("confidence") * 0.75;
    let q = Query::new()
        .update_set(target)
        .map_err(map)?
        .set_expr("confidence", confidence_decay)
        .map_err(map)?
        .set("updated_at", now.to_rfc3339())
        .map_err(map)?
        .where_(and_(
            eq("tenant_id", tenant.as_str()),
            is_none("deleted_at"),
        ))
        .return_after();
    let rows: Vec<MemoryRow> = query_records(store.client(), &q).await.map_err(map)?;
    let updated = rows
        .into_iter()
        .next()
        .map(MemoryRow::into_domain)
        .transpose()?;
    announce(store, ChangeAction::Update, updated.as_ref());
    Ok(updated)
}

/// Record that a trace graduated into the umbra as `expert`, tenant-checked.
pub async fn mark_consolidated(
    store: &Store,
    tenant: &TenantId,
    id: &MemoryId,
    expert: ExpertId,
    now: chrono::DateTime<chrono::Utc>,
) -> Result<Option<Memory>> {
    match get(store, tenant, id).await? {
        Some(mut m) => {
            m.mark_consolidated(expert, now);
            merge_fields(
                store,
                id,
                serde_json::json!({
                    "consolidated_expert": m.consolidated_expert.as_ref().map(ExpertId::as_str),
                    "updated_at": m.updated_at.to_rfc3339(),
                }),
            )
            .await?;
            announce(store, ChangeAction::Update, Some(&m));
            Ok(Some(m))
        }
        None => Ok(None),
    }
}

/// Forget a trace as a **tombstone** (the deletion that propagates and routes):
/// mark it `deleted_at = now` and persist, so read paths hide it while sync (R-1)
/// and live propagation (R-2) carry the deletion to other replicas/grantees
/// instead of it resurfacing. Tenant-checked via `get`; a no-op (returns `None`)
/// if the trace is absent or already a tombstone. Use [`purge`] to hard-remove
/// tombstones once the propagation grace window has passed.
pub async fn soft_delete(
    store: &Store,
    tenant: &TenantId,
    id: &MemoryId,
    now: chrono::DateTime<chrono::Utc>,
) -> Result<Option<Memory>> {
    match get(store, tenant, id).await? {
        Some(mut m) => {
            m.soft_delete(now);
            merge_fields(
                store,
                id,
                serde_json::json!({
                    "deleted_at": m.deleted_at.map(|t| t.to_rfc3339()),
                    "updated_at": m.updated_at.to_rfc3339(),
                }),
            )
            .await?;
            announce(store, ChangeAction::Update, Some(&m));
            Ok(Some(m))
        }
        None => Ok(None),
    }
}

/// Hard-remove tombstones forgotten before `older_than` (the grace window), so
/// they do not accumulate forever. Run it on a cadence with a window wider than
/// the sync interval, so every replica has seen the tombstone before it is
/// purged (resurrection-safe garbage collection). Returns how many were purged.
pub async fn purge(store: &Store, older_than: chrono::DateTime<chrono::Utc>) -> Result<usize> {
    // Select only actual tombstones older than the grace window -- a range scan
    // on the `deleted_at` index, not a full-table read. Every timestamp is UTC
    // RFC3339, so the lexicographic `<` matches chronological order.
    let cutoff = older_than.to_rfc3339();
    let query = Query::new()
        .select(None)
        .from_table(TABLE)
        .map_err(map)?
        .where_(and_(
            is_not_none("deleted_at"),
            lt("deleted_at", cutoff.as_str()),
        ));
    let rows: Vec<MemoryRow> = query_records(store.client(), &query).await.map_err(map)?;
    let mut purged = 0;
    for row in rows {
        let condition = and_(
            eq("key", row.key.as_str()),
            eq("tenant_id", row.tenant_id.as_str()),
        );
        delete_records(store.client(), TABLE, Some(&condition))
            .await
            .map_err(map)?;
        purged += 1;
    }
    Ok(purged)
}

/// Hard-delete a trace, but only within the caller's tenant (the `tenant_id`
/// predicate is ANDed onto the key, so no cross-tenant delete). Bypasses the
/// tombstone path -- prefer [`soft_delete`] for user-facing forgets so the
/// deletion propagates; this is for purges and internal cleanup.
pub async fn delete(store: &Store, tenant: &TenantId, id: &MemoryId) -> Result<()> {
    // What is removed, read only to announce it.
    let removed = if store.announcing() {
        get(store, tenant, id).await.ok().flatten()
    } else {
        None
    };
    let condition = and_(eq("key", id.as_str()), eq("tenant_id", tenant.as_str()));
    delete_records(store.client(), TABLE, Some(&condition))
        .await
        .map_err(map)?;
    announce(store, ChangeAction::Delete, removed.as_ref());
    Ok(())
}

/// Announce a memory write in-process, when the store announces its writes
/// ([`Store::announce_changes`]), as the row a LIVE watch would deliver.
fn announce(store: &Store, action: ChangeAction, memory: Option<&Memory>) {
    let Some(memory) = memory.filter(|_| store.announcing()) else {
        return;
    };
    if let Ok(row) = serde_json::to_value(MemoryRow::from_domain(memory)) {
        store.announce(ChangeEvent { action, row });
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod acl_cost;
