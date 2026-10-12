//! The chunk index: each memory's text in overlapping pieces, each
//! with its own vector, so a passage from the middle of a long memory can find
//! it. Derived from the memory and kept by the server's chunker; never synced.
//!
//! A chunk carries its memory's workspace, compartment and network, so the
//! table's select rule is the memory table's own and a chunk is visible exactly
//! when its memory is. Only the server writes it, in owner mode.

use std::collections::{BTreeMap, HashMap};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use surql::query::builder::Query;
use surql::query::crud::{delete_records, query_records, upsert_record};
use surql::types::operators::{and_, eq, gt};
use surql::types::RecordID;

use antumbra_core::{cosine_similarity, Memory, MemoryNetwork, Result, TenantId, UserId};

use crate::error::map;
use crate::knn::search_effort;
use crate::repo::memory;
use crate::schema;
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
    nearest_scored_within(store, tenant, vector, k, network, SEARCH_TIMEOUT).await
}

/// How long the chunk search may walk before the engine stops it.
///
/// Under a record session the engine checks the table's read rule during the
/// index walk and keeps walking until it has the pieces it was asked for, so
/// a session that can read fewer pieces than that walks the whole index,
/// running the read rule on every one. Measured on the embedded engine (a
/// debug build): 9.2 s for a member who could read 80 of 12,080 pieces, all
/// 80 returned, against 0.96 s for the owner who read every piece. On
/// kuskokwim the owner's chunk leg took at most 0.51 s in a day, and an
/// identity that could read almost nothing took 2.9 to 4.6 s on every recall.
///
/// Past this the leg finds nothing and recall goes on with its dense and
/// lexical legs, which the same session runs in milliseconds.
pub const SEARCH_TIMEOUT: Duration = Duration::from_millis(1000);

/// The most pieces the exact search reads for a session. A session that can
/// read more is searched by the index walk, which finds enough readable
/// pieces early when the session reads that much. Measured on the embedded
/// engine (a release build): reading and scoring a piece costs about 0.09 ms,
/// so the exact search stays near 0.2 s at this size, and 8 ms over 80.
pub const EXACT_MAX: usize = 2000;

/// How long a session found to read more than [`EXACT_MAX`] pieces is
/// searched by the walk before the exact search is tried for it again.
const WALK_FOR: Duration = Duration::from_secs(300);

/// [`nearest_scored`] with the engine told to stop after `timeout`. A search
/// stopped there finds nothing rather than failing the recall.
///
/// A record session that can read few pieces is searched exactly: the shared
/// pool and the compartments it can read, each through the index on the
/// chunk's compartment, scored here. The walk would run the read rule on
/// every piece of the workspace to find them. The engine still applies the
/// read rule to every piece either search reads, so the list of compartments
/// decides what is cheap to read, never what is visible.
pub(crate) async fn nearest_scored_within(
    store: &Store,
    tenant: &TenantId,
    vector: &[f32],
    k: usize,
    network: Option<MemoryNetwork>,
    timeout: Duration,
) -> Result<Vec<(String, f32)>> {
    search(store, tenant, vector, k, network, timeout, EXACT_MAX).await
}

async fn search(
    store: &Store,
    tenant: &TenantId,
    vector: &[f32],
    k: usize,
    network: Option<MemoryNetwork>,
    timeout: Duration,
    exact_max: usize,
) -> Result<Vec<(String, f32)>> {
    if k == 0 || vector.iter().all(|x| *x == 0.0) {
        return Ok(Vec::new());
    }
    let rows = k.saturating_mul(ROWS_PER_MEMORY).clamp(k, MAX_ROWS);
    if let Some(user) = store.session_user() {
        let readable = readable_compartments(store, tenant, &user).await?;
        let key = walk_key(tenant, &user, &readable);
        if !walks(&key) {
            match exact(
                store, tenant, vector, rows, network, &readable, timeout, exact_max,
            )
            .await?
            {
                Some(pieces) => return Ok(pieces),
                None => remember_walk(key),
            }
        }
    }
    walk(store, tenant, vector, rows, network, timeout).await
}

/// The variables every chunk search binds.
fn search_vars(
    tenant: &TenantId,
    vector: &[f32],
    network: Option<MemoryNetwork>,
) -> BTreeMap<String, Value> {
    let mut vars = BTreeMap::from([
        ("tenant".to_string(), json!(tenant.as_str())),
        ("vector".to_string(), json!(vector)),
    ]);
    if let Some(net) = network {
        vars.insert("network".to_string(), json!(net.as_str()));
    }
    vars
}

/// `surql`'s rows, or `None` when the engine stopped it at its `TIMEOUT`.
async fn bounded<T: serde::de::DeserializeOwned>(
    store: &Store,
    surql: &str,
    vars: BTreeMap<String, Value>,
    timeout: Duration,
) -> Result<Option<Vec<T>>> {
    match store.query_rows(surql, vars).await {
        Ok(rows) => Ok(Some(rows)),
        Err(e) if timed_out(&e.to_string()) => {
            tracing::warn!(
                target: "antumbra_store::recall",
                timeout_ms = timeout.as_millis() as u64,
                "the chunk search stopped at its timeout; recall goes on without it"
            );
            Ok(None)
        }
        Err(e) => Err(e),
    }
}

/// The index walk: the `rows` readable pieces nearest `vector`, as the engine
/// finds them, stopped at `timeout`.
async fn walk(
    store: &Store,
    tenant: &TenantId,
    vector: &[f32],
    rows: usize,
    network: Option<MemoryNetwork>,
    timeout: Duration,
) -> Result<Vec<(String, f32)>> {
    let also = if network.is_some() {
        " AND network = $network"
    } else {
        ""
    };
    let surql = format!(
        "SELECT memory, vector::distance::knn() AS distance FROM {TABLE} \
         WHERE tenant_id = $tenant{also} AND embedding <|{rows},{}|> $vector \
         TIMEOUT {}ns",
        search_effort(rows),
        timeout.as_nanos().max(1)
    );
    #[derive(Deserialize)]
    struct Hit {
        memory: String,
        #[serde(default)]
        distance: Option<f64>,
    }
    let Some(hits) =
        bounded::<Hit>(store, &surql, search_vars(tenant, vector, network), timeout).await?
    else {
        return Ok(Vec::new());
    };
    // Ordered here, as `nearest` orders, rather than by the engine.
    let mut pieces: Vec<(String, f32)> = hits
        .into_iter()
        .filter_map(|h| Some((h.memory, (1.0 - h.distance?) as f32)))
        .collect();
    pieces.sort_by(|a, b| b.1.total_cmp(&a.1));
    Ok(pieces)
}

/// The exact search: every piece of the shared pool and of `readable`, scored
/// by cosine here and cut to the `rows` nearest. `None` when there are more
/// than `max` of them, for the walk to search instead.
///
/// The shared pool and the compartments are read by two queries: OR'd
/// together the planner reads the whole workspace and filters it, and an
/// empty list read alone does the same, so it is not read at all.
#[allow(clippy::too_many_arguments)]
async fn exact(
    store: &Store,
    tenant: &TenantId,
    vector: &[f32],
    rows: usize,
    network: Option<MemoryNetwork>,
    readable: &[String],
    timeout: Duration,
    max: usize,
) -> Result<Option<Vec<(String, f32)>>> {
    let also = if network.is_some() {
        " AND network = $network"
    } else {
        ""
    };
    let select = |place: &str, limit: usize| {
        format!(
            "SELECT memory, vector::similarity::cosine(embedding, $vector) AS score \
             FROM {TABLE} WHERE tenant_id = $tenant AND {place}{also} \
             LIMIT {limit} TIMEOUT {}ns",
            timeout.as_nanos().max(1)
        )
    };
    #[derive(Deserialize)]
    struct Scored {
        memory: String,
        #[serde(default)]
        score: Option<f64>,
    }
    let mut vars = search_vars(tenant, vector, network);
    let Some(mut hits) = bounded::<Scored>(
        store,
        &select("compartment = NONE", max + 1),
        vars.clone(),
        timeout,
    )
    .await?
    else {
        return Ok(Some(Vec::new()));
    };
    if hits.len() > max {
        return Ok(None);
    }
    if !readable.is_empty() {
        vars.insert("readable".to_string(), json!(readable));
        let left = max + 1 - hits.len();
        let Some(more) = bounded::<Scored>(
            store,
            &select("compartment INSIDE $readable", left),
            vars,
            timeout,
        )
        .await?
        else {
            return Ok(Some(Vec::new()));
        };
        if hits.len() + more.len() > max {
            return Ok(None);
        }
        hits.extend(more);
    }
    let mut pieces: Vec<(String, f32)> = hits
        .into_iter()
        .filter_map(|h| Some((h.memory, h.score? as f32)))
        .collect();
    pieces.sort_by(|a, b| b.1.total_cmp(&a.1));
    pieces.truncate(rows);
    Ok(Some(pieces))
}

/// The compartments `user` can read in `tenant` beside the shared pool: those
/// it owns, those granted to it, and those offered to an enabled hive. The
/// read rule's own branches, as lists, from the same text
/// (`schema::{owned,granted,hive}_compartments`). The user is bound, not read
/// from `$auth`: outside a permission clause a record session cannot read its
/// own principal.
pub(crate) async fn readable_compartments(
    store: &Store,
    tenant: &TenantId,
    user: &UserId,
) -> Result<Vec<String>> {
    let surql = format!(
        "RETURN array::union(array::union(({}), ({})), ({}))",
        schema::owned_compartments("$tenant", "$user"),
        schema::granted_compartments("$user"),
        schema::hive_compartments("$tenant"),
    );
    let vars = BTreeMap::from([
        ("tenant".to_string(), json!(tenant.as_str())),
        ("user".to_string(), json!(user.as_str())),
    ]);
    let mut list: Vec<String> = store.query_rows(&surql, vars).await?;
    list.sort();
    list.dedup();
    Ok(list)
}

type WalkKey = (String, String, u64);

/// The sessions found to read more than the exact search takes, by tenant,
/// user and the list of compartments they read, and when.
fn walkers() -> &'static std::sync::Mutex<HashMap<WalkKey, Instant>> {
    static WALKERS: OnceLock<std::sync::Mutex<HashMap<WalkKey, Instant>>> = OnceLock::new();
    WALKERS.get_or_init(Default::default)
}

fn walk_key(tenant: &TenantId, user: &UserId, readable: &[String]) -> WalkKey {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    readable.hash(&mut h);
    (
        tenant.as_str().to_string(),
        user.as_str().to_string(),
        h.finish(),
    )
}

/// Whether this session is searched by the walk for now. A changed list of
/// compartments is a new key, so a grant or a revoke is seen at once.
fn walks(key: &WalkKey) -> bool {
    walkers()
        .lock()
        .ok()
        .and_then(|m| m.get(key).copied())
        .is_some_and(|at| at.elapsed() < WALK_FOR)
}

fn remember_walk(key: WalkKey) {
    if let Ok(mut m) = walkers().lock() {
        if m.len() > 10_000 {
            m.clear();
        }
        m.insert(key, Instant::now());
    }
}

/// Whether an error is the engine stopping a query at its `TIMEOUT`.
fn timed_out(message: &str) -> bool {
    let m = message.to_lowercase();
    m.contains("timeout") || m.contains("timed out")
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

#[cfg(test)]
mod acl_cost;
