//! The chunk index (ADR-0025): each memory's text in overlapping pieces, each
//! with its own vector, so a passage from the middle of a long memory can find
//! it. Derived from the memory and kept by the server's chunker; never synced.
//!
//! A chunk carries its memory's workspace, compartment and network, so the
//! table's select rule is the memory table's own and a chunk is visible exactly
//! when its memory is. Only the server writes it, in owner mode.

use std::collections::{BTreeMap, HashMap};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use surql::query::builder::Query;
use surql::query::crud::{delete_records, query_records, upsert_record};
use surql::types::operators::{and_, eq, gt};
use surql::types::RecordID;

use antumbra_core::{cosine_similarity, Memory, MemoryNetwork, Result, TenantId};

use crate::error::map;
use crate::knn::search_effort;
use crate::repo::memory;
use crate::store::Store;

const TABLE: &str = "memory_chunk";
const MEMORY: &str = "memory";

/// How many chunk rows to read per memory wanted: a memory's neighboring
/// pieces crowd the nearest rows, so the search reads wider than it returns.
const ROWS_PER_MEMORY: usize = 4;
/// The most chunk rows one search reads.
const MAX_ROWS: usize = 800;

#[derive(Serialize, Deserialize)]
struct ChunkRow {
    key: String,
    /// The memory's id, as its `key` (`memory:...`).
    memory: String,
    tenant_id: String,
    network: MemoryNetwork,
    // Absent, not null, for the shared pool, so the select rule sees
    // `compartment = NONE` exactly as it does on the memory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    compartment: Option<String>,
    ordinal: u32,
    content_hash: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    embedding: Option<Vec<f32>>,
}

impl ChunkRow {
    /// File this chunk under `memory`'s current workspace, compartment and
    /// network.
    fn filed_as(mut self, memory: &Memory) -> Self {
        self.tenant_id = memory.tenant.as_str().to_string();
        self.network = memory.network;
        self.compartment = memory.compartment.as_ref().map(|c| c.as_str().to_string());
        self
    }
}

/// What the index holds for one memory: the content it was cut from and where
/// it was filed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Indexed {
    pub content_hash: String,
    pub compartment: Option<String>,
    pub network: MemoryNetwork,
}

impl Indexed {
    /// Whether the chunks are filed where `memory` is now.
    pub fn filed_as(&self, memory: &Memory) -> bool {
        self.network == memory.network
            && self.compartment.as_deref() == memory.compartment.as_ref().map(|c| c.as_str())
    }
}

fn record(memory: &str, ordinal: u32) -> Result<RecordID<()>> {
    RecordID::<()>::new(TABLE, format!("{memory}~{ordinal}").as_str()).map_err(map)
}

async fn write(store: &Store, row: ChunkRow) -> Result<()> {
    let id = record(&row.memory, row.ordinal)?;
    let data: Value = serde_json::to_value(row)?;
    upsert_record(store.client(), &id, data)
        .await
        .map_err(map)?;
    Ok(())
}

/// Replace a memory's chunks with `vectors`, one per piece in order, cut from
/// content whose hash is `content_hash`.
pub async fn replace(
    store: &Store,
    memory: &Memory,
    content_hash: &str,
    vectors: Vec<Vec<f32>>,
) -> Result<()> {
    remove(store, &memory.tenant, memory.id.as_str()).await?;
    for (ordinal, vector) in vectors.into_iter().enumerate() {
        let ordinal = ordinal as u32;
        let row = ChunkRow {
            key: format!("{}~{ordinal}", memory.id.as_str()),
            memory: memory.id.as_str().to_string(),
            tenant_id: String::new(),
            network: memory.network,
            compartment: None,
            ordinal,
            content_hash: content_hash.to_string(),
            embedding: Some(vector),
        };
        write(store, row.filed_as(memory)).await?;
    }
    Ok(())
}

/// File a memory's chunks where the memory is now, keeping their vectors: the
/// memory moved to another compartment or network, its text did not.
pub async fn refile(store: &Store, memory: &Memory) -> Result<()> {
    let query = Query::new()
        .select(None)
        .from_table(TABLE)
        .map_err(map)?
        .where_(and_(
            eq("tenant_id", memory.tenant.as_str()),
            eq("memory", memory.id.as_str()),
        ));
    let rows: Vec<ChunkRow> = query_records(store.client(), &query).await.map_err(map)?;
    for row in rows {
        write(store, row.filed_as(memory)).await?;
    }
    Ok(())
}

/// Drop a memory's chunks: it was forgotten, or is about to be cut again.
pub async fn remove(store: &Store, tenant: &TenantId, memory: &str) -> Result<()> {
    let condition = and_(eq("tenant_id", tenant.as_str()), eq("memory", memory));
    delete_records(store.client(), TABLE, Some(&condition))
        .await
        .map_err(map)?;
    Ok(())
}

/// A memory's first chunk, as [`indexed`] reads it.
#[derive(Deserialize)]
struct Head {
    memory: String,
    content_hash: String,
    #[serde(default)]
    compartment: Option<String>,
    network: MemoryNetwork,
}

fn by_memory(rows: Vec<Head>) -> HashMap<String, Indexed> {
    rows.into_iter()
        .map(|h| {
            (
                h.memory,
                Indexed {
                    content_hash: h.content_hash,
                    compartment: h.compartment,
                    network: h.network,
                },
            )
        })
        .collect()
}

/// What the index holds for every memory of `tenant` it has chunks for, by
/// memory id. Read from each memory's first chunk.
pub async fn indexed(store: &Store, tenant: &TenantId) -> Result<HashMap<String, Indexed>> {
    let query = Query::new()
        .select(Some(vec![
            "memory".to_string(),
            "content_hash".to_string(),
            "compartment".to_string(),
            "network".to_string(),
        ]))
        .from_table(TABLE)
        .map_err(map)?
        .where_(and_(eq("tenant_id", tenant.as_str()), eq("ordinal", 0)));
    let rows: Vec<Head> = query_records(store.client(), &query).await.map_err(map)?;
    Ok(by_memory(rows))
}

/// [`indexed`] for `memories` alone, each read by its first chunk's key: what
/// a pass over a few changed memories needs, without reading the first chunk
/// of every memory in the workspace.
pub async fn indexed_among(
    store: &Store,
    tenant: &TenantId,
    memories: &[String],
) -> Result<HashMap<String, Indexed>> {
    let keys: Vec<String> = memories.iter().map(|m| format!("{m}~0")).collect();
    let rows: Vec<Head> = store.rows_by_key(TABLE, &keys, tenant, "").await?;
    Ok(by_memory(rows))
}

/// The pieces nearest `vector`, as `(memory id, piece vector)`, nearest
/// first: enough rows for about `k` memories, since a memory's pieces crowd
/// each other. Tenant- and, when given, network-scoped; the table's select
/// rule scopes it further to what the session may read. The vectors come back
/// so the caller can score them the way it scores whole memories.
pub async fn nearest(
    store: &Store,
    tenant: &TenantId,
    vector: &[f32],
    k: usize,
    network: Option<MemoryNetwork>,
) -> Result<Vec<(String, Vec<f32>)>> {
    if k == 0 || vector.iter().all(|x| *x == 0.0) {
        return Ok(Vec::new());
    }
    let rows = k.saturating_mul(ROWS_PER_MEMORY).clamp(k, MAX_ROWS);
    let values: Vec<f64> = vector.iter().map(|&x| f64::from(x)).collect();
    let condition = match network {
        Some(net) => and_(
            eq("tenant_id", tenant.as_str()),
            eq("network", net.as_str()),
        ),
        None => eq("tenant_id", tenant.as_str()),
    };
    let query = Query::new()
        .select(Some(vec!["memory".to_string(), "embedding".to_string()]))
        .from_table(TABLE)
        .map_err(map)?
        .where_(condition)
        .vector_search_indexed("embedding", values, rows as i64, search_effort(rows))
        .map_err(map)?;
    #[derive(Deserialize)]
    struct Hit {
        memory: String,
        #[serde(default)]
        embedding: Option<Vec<f32>>,
    }
    let hits: Vec<Hit> = query_records(store.client(), &query).await.map_err(map)?;
    // Ordered here rather than by the engine: an ORDER BY over a computed
    // similarity returned each row twice under a record session.
    let mut pieces: Vec<(f32, String, Vec<f32>)> = hits
        .into_iter()
        .filter_map(|h| {
            let v = h.embedding?;
            Some((cosine_similarity(vector, &v), h.memory, v))
        })
        .collect();
    pieces.sort_by(|a, b| b.0.total_cmp(&a.0));
    Ok(pieces.into_iter().map(|(_, m, v)| (m, v)).collect())
}

/// [`nearest`] without the vectors: each piece's memory and its cosine to
/// `vector`, nearest first. The engine has the distance from walking the index
/// (`vector::distance::knn()`, one minus the cosine on this COSINE index), so a
/// piece comes back as a key and a number rather than 384 numbers: on
/// kuskokwim the leg's 150 pieces were 1.2 MB with their vectors and 6 KB
/// without, for the same 30 ms in the engine.
pub async fn nearest_scored(
    store: &Store,
    tenant: &TenantId,
    vector: &[f32],
    k: usize,
    network: Option<MemoryNetwork>,
) -> Result<Vec<(String, f32)>> {
    if k == 0 || vector.iter().all(|x| *x == 0.0) {
        return Ok(Vec::new());
    }
    let rows = k.saturating_mul(ROWS_PER_MEMORY).clamp(k, MAX_ROWS);
    let also = if network.is_some() {
        " AND network = $network"
    } else {
        ""
    };
    let surql = format!(
        "SELECT memory, vector::distance::knn() AS distance FROM {TABLE} \
         WHERE tenant_id = $tenant{also} AND embedding <|{rows},{}|> $vector",
        search_effort(rows)
    );
    let mut vars = BTreeMap::from([
        ("tenant".to_string(), json!(tenant.as_str())),
        ("vector".to_string(), json!(vector)),
    ]);
    if let Some(net) = network {
        vars.insert("network".to_string(), json!(net.as_str()));
    }
    #[derive(Deserialize)]
    struct Hit {
        memory: String,
        #[serde(default)]
        distance: Option<f64>,
    }
    let hits: Vec<Hit> = store.query_rows(&surql, vars).await?;
    // Ordered here, as `nearest` orders, rather than by the engine.
    let mut pieces: Vec<(String, f32)> = hits
        .into_iter()
        .filter_map(|h| Some((h.memory, (1.0 - h.distance?) as f32)))
        .collect();
    pieces.sort_by(|a, b| b.1.total_cmp(&a.1));
    Ok(pieces)
}

/// A memory as the chunker reads it: no embedding, and its tombstone if it
/// has one.
#[derive(Debug, Clone)]
pub struct ChunkSource {
    pub memory: Memory,
    pub forgotten: bool,
}

/// The memories of `tenant` the chunker should look at: all of them, or those
/// changed since `since`, forgotten ones included so their chunks can go.
pub async fn sources(
    store: &Store,
    tenant: &TenantId,
    since: Option<DateTime<Utc>>,
) -> Result<Vec<ChunkSource>> {
    let condition = match since {
        Some(t) => and_(
            eq("tenant_id", tenant.as_str()),
            gt("updated_at", t.to_rfc3339()),
        ),
        None => eq("tenant_id", tenant.as_str()),
    };
    let rows: Vec<memory::MemoryRow> = store
        .read_paged(MEMORY, Some(memory::without_embedding()), Some(&condition))
        .await?;
    rows.into_iter()
        .map(|r| {
            let forgotten = r.deleted_at.is_some();
            Ok(ChunkSource {
                memory: r.into_domain()?,
                forgotten,
            })
        })
        .collect()
}

/// Every workspace that holds a memory.
pub async fn tenants(store: &Store) -> Result<Vec<TenantId>> {
    let query = Query::new()
        .select(Some(vec!["tenant_id".to_string()]))
        .from_table(MEMORY)
        .map_err(map)?
        .group_by(["tenant_id"]);
    #[derive(Deserialize)]
    struct Row {
        tenant_id: String,
    }
    let rows: Vec<Row> = query_records(store.client(), &query).await.map_err(map)?;
    Ok(rows
        .into_iter()
        .map(|r| TenantId::new(r.tenant_id))
        .collect())
}

#[cfg(test)]
mod tests;
