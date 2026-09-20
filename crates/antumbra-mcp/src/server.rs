//! The Antumbra MCP tool surface over the Penumbra memory store.
//!
//! Every tool operates on a single bound tenant (the workspace this server was
//! started for); the Store session is signed in as that tenant, so reads are
//! engine-enforced: the server cannot serve another tenant's memory even if a
//! tool's filter were wrong. This mirrors the `ai_memory` MCP surface: store,
//! recall (semantic), reinforce, forget, list.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use chrono::Utc;
use rmcp::handler::server::wrapper::{Json, Parameters};
use rmcp::schemars;
use rmcp::{tool, tool_handler, tool_router, ErrorData, ServerHandler};
use serde::{Deserialize, Serialize};

use antumbra_core::ports::{ActRequest, Embedder};
use antumbra_core::{
    demote_out_of_scope, orphan_of, scope_from_str, scope_of_evidence, GitContext, GitProvenance,
    Scope,
};
use antumbra_core::{
    Capability, ClusterConfig, Compartment, CompartmentId, DocumentChunk, EdgeType, ExpertId,
    Grant, Memory, MemoryEdge, MemoryId, MemoryNetwork, Origin, TenantId, UserId,
};
use antumbra_store::repo::{boundary, compartment, document, edge, expert, memory, router};
use antumbra_store::Store;

/// One (tenant, user) MCP session over its Penumbra. `#[tool_handler]` resolves
/// the tools via `Self::tool_router()`, so no router field is stored.
#[derive(Clone)]
pub struct McpServer {
    store: Store,
    /// A stable OWNER connection the autonomous consolidation background task
    /// runs on (gather + provision + mint execute as root). In the multi-tenant
    /// HTTP server `store` is the per-request *scoped* connection, which a
    /// detached task cannot rely on; `None` falls back to `store` (stdio /
    /// embedded, where it is already the owner connection).
    #[cfg_attr(not(feature = "models"), allow(dead_code))]
    consolidation_store: Option<Store>,
    embedder: Arc<dyn Embedder>,
    tenant: TenantId,
    /// The user this session acts as (compartment owner / grantor).
    user: UserId,
    /// The host/device, stamped as provenance on every memory written.
    host: String,
    /// The compartment new memories land in when none is named (the session's
    /// fresh space; engine-isolated to this user until shared).
    default_compartment: CompartmentId,
    /// When set, the antumbra auto-organizes the inbox once it grows past the
    /// threshold (the autonomous propose trigger). `None` = on-demand only.
    auto_propose: Option<AutoProposeConfig>,
    /// When set, a reinforced memory whose compartment clears the consolidation
    /// gate auto-graduates into a private expert on the GPU (the autonomous sleep
    /// trigger). `None` = on-demand only (the `consolidate-compartment` CLI).
    #[cfg_attr(not(feature = "models"), allow(dead_code))]
    auto_consolidate: Option<AutoConsolidateConfig>,
    /// Per-compartment consolidation state: what is in flight (so a burst of
    /// reinforces coalesces into one train instead of stacking GPU jobs), what was
    /// written to meanwhile, and what the gate last said.
    #[cfg_attr(not(feature = "models"), allow(dead_code))]
    consolidating: consolidation::SharedConsolidation,
    /// The serving engine the `answer` tool drives (route → serve through the
    /// expert's adapter). `None` = serving not configured (route-only surface).
    serve: Option<Arc<dyn antumbra_core::ports::Serve>>,
    /// Where this session registers its peer on initialize, so live propagation
    /// (R-2) can push shared-memory changes to it. `None` = no live delivery
    /// (stdio, route-only, or tests).
    registry: Option<crate::notify::PeerRegistry>,
    /// Optional cross-encoder precision stage (P-2) applied after hybrid recall:
    /// re-scores the wide RRF candidate pool over (query, content) and reorders.
    /// `None` = RRF order is returned as-is (rerank endpoint not configured).
    reranker: Option<Arc<dyn antumbra_core::ports::Reranker>>,
    /// Bounded per-server cache of rerank orders, keyed by (query, candidate-id
    /// set), so a repeated recall of the same pool skips the endpoint round-trip.
    reranker_cache: Arc<tokio::sync::Mutex<RerankCache>>,
    /// The copal document-of-record archive: when set, `ingest_document`
    /// uploads the ORIGINAL content to copal FIRST (failing the ingest if
    /// copal is unreachable) and stamps every stored chunk with the file id +
    /// digest. `None` = no archive; ingest behaves exactly as before.
    copal: Option<Arc<antumbra_copal::CopalArchive>>,
    /// Which tools this session advertises and serves (`--tools`). `None` =
    /// every tool. Enforced at `tools/list`, at JSON-RPC `tools/call`, and at
    /// the REST shim, so a tool outside the profile is neither seen nor run.
    profile: Option<Arc<crate::profile::ToolProfile>>,
    /// Keeps this server's record session signed in past the store's session
    /// duration (see `session`); checked at every tool call. `None` where the
    /// transport signs in per request (the embedded networked surface).
    session: Option<Arc<crate::session::SessionKeeper>>,
}

/// The cross-encoder candidate pool: rerank re-scores a wide RRF pool, then
/// truncates to the caller's k. ~100 candidates is the precision/latency knee for
/// a cross-encoder (one batched POST).
const RERANK_POOL_MAX: usize = 100;

/// Bounded cache of `(query, sorted candidate ids) -> reranked id order`. A plain
/// insertion-ordered map capped at `CAP`; on overflow the oldest entry is
/// evicted. A miss merely recomputes, so eviction is always safe.
struct RerankCache {
    map: std::collections::HashMap<String, Vec<String>>,
    order: std::collections::VecDeque<String>,
}

impl RerankCache {
    const CAP: usize = 1024;

    fn new() -> Self {
        Self {
            map: std::collections::HashMap::new(),
            order: std::collections::VecDeque::new(),
        }
    }

    fn get(&self, key: &str) -> Option<Vec<String>> {
        self.map.get(key).cloned()
    }

    fn put(&mut self, key: String, value: Vec<String>) {
        if self.map.insert(key.clone(), value).is_none() {
            self.order.push_back(key);
            while self.order.len() > Self::CAP {
                if let Some(old) = self.order.pop_front() {
                    self.map.remove(&old);
                }
            }
        }
    }
}

/// Build the cache key for a rerank over `query` + `candidates`. The candidate
/// ids are sorted so the key is independent of the pool's internal order (the
/// same recall set always hits the same entry); ids cannot contain a newline, so
/// `\n` is an unambiguous separator.
fn rerank_cache_key(query: &str, candidates: &[(String, String)]) -> String {
    let mut ids: Vec<&str> = candidates.iter().map(|(id, _)| id.as_str()).collect();
    ids.sort_unstable();
    let mut key =
        String::with_capacity(query.len() + 1 + ids.iter().map(|s| s.len() + 1).sum::<usize>());
    key.push_str(query);
    key.push('\n');
    for id in ids {
        key.push_str(id);
        key.push('\n');
    }
    key
}

/// Reorder `hits` to follow the reranker's id `order` and take the top `k`. Ids
/// in `order` not present in `hits` are skipped; any hit whose id is missing from
/// `order` is dropped (the reranker contract returns a full permutation, so this
/// only guards a misbehaving reranker — and `take(k)` bounds the result either
/// way).
fn reorder_by_ids<T>(
    hits: Vec<T>,
    order: &[String],
    id_of: impl Fn(&T) -> String,
    k: usize,
) -> Vec<T> {
    let mut by_id: std::collections::HashMap<String, T> =
        hits.into_iter().map(|h| (id_of(&h), h)).collect();
    order
        .iter()
        .filter_map(|id| by_id.remove(id))
        .take(k)
        .collect()
}

/// Tuning for the autonomous propose trigger.
#[derive(Clone)]
struct AutoProposeConfig {
    /// Inbox size at/above which a proposal pass fires after a write.
    threshold: usize,
    min_size: usize,
    similarity_threshold: f32,
}

/// Tuning for the autonomous consolidation trigger. Plain primitives so the
/// server struct stays free of the models-gated trainer types; the gate policy
/// and `RaftConfig` are built from these inside the models-gated trigger.
#[cfg_attr(not(feature = "models"), allow(dead_code))]
#[derive(Clone)]
struct AutoConsolidateConfig {
    min_recurrence: u32,
    min_confidence: f32,
    rounds: usize,
    samples: usize,
    max_new_tokens: usize,
    lr: f64,
    replay_ratio: f64,
}

fn err(e: impl std::fmt::Display) -> ErrorData {
    ErrorData::internal_error(e.to_string(), None)
}

fn parse_network(s: &str) -> MemoryNetwork {
    match s.trim().to_lowercase().as_str() {
        "bank" => MemoryNetwork::Bank,
        "opinion" => MemoryNetwork::Opinion,
        _ => MemoryNetwork::World,
    }
}

/// A process-unique, monotonic id with a table prefix: a nanosecond timestamp
/// plus a single process-global counter, so two sessions (even different
/// tenants) minting their first id in the same nanosecond never collide; no
/// extra deps.
/// A caller-supplied anchor, validated: a repo slug without `@` (host/org/name),
/// a 7-40 hex digit commit, a branch without `:`.
fn anchor_from(prov: ProvenanceParams) -> Result<GitProvenance, ErrorData> {
    let anchor = GitProvenance {
        repo: prov.repo,
        commit: prov.commit,
        branch: prov.branch,
        path: prov.path,
    };
    if !anchor.is_valid() {
        return Err(ErrorData::invalid_params(
            "provenance must have a repo slug without '@' (host/org/name), a 7-40 hex digit \
             commit, and a branch without ':'"
                .to_string(),
            None,
        ));
    }
    Ok(anchor)
}

fn next_id(prefix: &str) -> String {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{prefix}:{t:x}-{n:x}")
}

fn default_network() -> String {
    "world".into()
}

fn parse_edge_type(s: &str) -> EdgeType {
    match s.trim().to_lowercase().as_str() {
        "supersedes" => EdgeType::Supersedes,
        "contradicts" => EdgeType::Contradicts,
        "follows" => EdgeType::Follows,
        "caused" => EdgeType::Caused,
        _ => EdgeType::References,
    }
}

fn default_edge_type() -> String {
    "references".into()
}

fn parse_capability(s: &str) -> Capability {
    match s.trim().to_lowercase().as_str() {
        "link" => Capability::Link,
        _ => Capability::Reference,
    }
}

fn default_capability() -> String {
    "reference".into()
}

pub(crate) mod consolidation;
mod engine;
mod params;
use self::params::*;

#[tool_router]
impl McpServer {
    /// Store a memory in this tenant's Penumbra.
    #[tool(
        description = "Store a memory (content + network) in your workspace's memory. Returns the new memory id."
    )]
    async fn store_memory(
        &self,
        Parameters(p): Parameters<StoreParams>,
    ) -> Result<Json<StoredOut>, ErrorData> {
        let embedding = self.embedder.embed(&p.content).await.map_err(err)?;
        let id = next_id("memory");
        let compartment = p
            .compartment
            .map(CompartmentId::new)
            .unwrap_or_else(|| self.default_compartment.clone());
        let mut m = Memory::new(
            id.clone(),
            self.tenant.clone(),
            parse_network(&p.network),
            p.content,
            p.confidence.unwrap_or(0.6),
            Utc::now(),
        )
        .with_embedding(embedding)
        .in_compartment(compartment)
        .by(self.user.clone(), self.host.clone());
        let mut evidence = p.evidence.unwrap_or_default();
        if let Some(prov) = p.provenance {
            evidence.push(anchor_from(prov)?.to_evidence());
        }
        if !evidence.is_empty() {
            m = m.with_evidence(evidence);
        }
        if p.volatile.unwrap_or(false) {
            m = m.volatile(true);
        }
        memory::upsert(&self.store, &m).await.map_err(err)?;
        // The compartment write-ACL silently drops a write into a compartment the
        // caller may not write to (an UPSERT under a record session succeeds but
        // persists nothing). Verify the row landed, so the caller gets a real
        // error instead of a phantom id -- and we never consolidate a vanished write.
        if memory::get(&self.store, &self.tenant, &MemoryId::new(id.clone()))
            .await
            .map_err(err)?
            .is_none()
        {
            return Err(ErrorData::invalid_params(
                format!(
                    "could not store the memory into compartment '{}': it is not one you can \
                     write to (create it with create_compartment first, or omit it for your default)",
                    m.compartment.as_ref().map(|c| c.as_str()).unwrap_or("(default)")
                ),
                None,
            ));
        }
        // Autonomous triggers (both no-ops when disabled / below threshold): the
        // antumbra organizes the inbox once it grows, and a write that itself
        // clears the consolidation gate graduates its compartment now.
        self.maybe_consolidate(&m).await;
        let auto_proposed = self.maybe_auto_propose().await.map_err(err)?;
        Ok(Json(StoredOut { id, auto_proposed }))
    }

    /// How many candidates to pull from hybrid recall before reranking: a wide
    /// pool when the cross-encoder is configured (so it has room to reorder),
    /// else just the caller's `k`.
    fn recall_pool(&self, k: usize) -> usize {
        if self.reranker.is_some() {
            k.max(1).saturating_mul(10).clamp(20, RERANK_POOL_MAX)
        } else {
            k
        }
    }

    /// Apply the optional cross-encoder precision stage to a wide candidate pool
    /// and truncate to `k`. Best-effort: with no reranker (or a trivial pool, or
    /// any reranker error) the input order is preserved. `id_of`/`text_of` adapt
    /// the row type to the `(id, content)` the reranker scores.
    async fn rerank_to_k<T>(
        &self,
        query: &str,
        hits: Vec<T>,
        k: usize,
        id_of: impl Fn(&T) -> String,
        text_of: impl Fn(&T) -> String,
    ) -> Vec<T> {
        let Some(reranker) = self.reranker.as_ref() else {
            return hits.into_iter().take(k).collect();
        };
        // Nothing to reorder (or no query): skip the round-trip.
        if hits.len() <= 1 || query.trim().is_empty() {
            return hits.into_iter().take(k).collect();
        }

        let candidates: Vec<(String, String)> =
            hits.iter().map(|h| (id_of(h), text_of(h))).collect();
        let cache_key = rerank_cache_key(query, &candidates);

        // Cache hit: reuse the previously-computed order.
        if let Some(order) = self.reranker_cache.lock().await.get(&cache_key) {
            return reorder_by_ids(hits, &order, &id_of, k);
        }

        match reranker.rerank(query, &candidates).await {
            Ok(order) => {
                self.reranker_cache
                    .lock()
                    .await
                    .put(cache_key, order.clone());
                reorder_by_ids(hits, &order, &id_of, k)
            }
            Err(e) => {
                // Degrade to the pre-rerank (RRF) order — a reranker fault must
                // never turn a successful recall into a failure.
                eprintln!("antumbra-mcp: rerank failed, using RRF order: {e}");
                hits.into_iter().take(k).collect()
            }
        }
    }

    /// Semantic recall over this tenant's Penumbra.
    #[tool(
        description = "Recall the memories most relevant to a query from your workspace's memory (semantic search)."
    )]
    async fn recall_memories(
        &self,
        Parameters(p): Parameters<RecallParams>,
    ) -> Result<Json<MemoriesOut>, ErrorData> {
        let q = self.embedder.embed(&p.query).await.map_err(err)?;
        let net = p.network.as_deref().map(parse_network);
        let k = p.top_k.unwrap_or(5) as usize;
        // Hybrid recall: dense (HNSW) + sparse (BM25 full-text) fused by RRF, so
        // exact tokens the embedding drops still surface. Pull a wide pool when a
        // reranker is configured, then re-score + truncate to k.
        let hits = memory::recall_hybrid(
            &self.store,
            &self.tenant,
            &p.query,
            &q,
            self.recall_pool(k),
            net,
        )
        .await
        .map_err(err)?;
        let hits = self
            .rerank_to_k(
                &p.query,
                hits,
                k,
                |m| m.id.as_str().to_string(),
                |m| m.content.clone(),
            )
            .await;
        // Scope each hit against where the caller is (repo + branch): a memory
        // learned on another branch or in another repo is demoted below the
        // in-scope ones, never hidden. The hook that boots a session goes
        // further with git in hand (is the commit on HEAD, does the branch
        // still exist); the server only knows what the caller told it.
        // Say how close each one is while the hit is still in hand. Recall
        // returns `top_k` whether or not anything was relevant, and the order
        // alone cannot tell a near match from the best of a bad lot; the cosine
        // can. Attached here rather than afterwards because the demotion below
        // reorders the views, and a similarity paired with the wrong memory
        // would be worse than none.
        let with_similarity = |m: &Memory, mut view: MemoryView| {
            view.similarity = m
                .embedding
                .as_deref()
                .map(|e| antumbra_core::cosine_similarity(&q, e));
            view
        };
        let memories = if p.repo.is_some() || p.branch.is_some() {
            let ctx = GitContext {
                repo: p.repo,
                branch: p.branch,
            };
            let views: Vec<MemoryView> = hits
                .iter()
                .map(|m| with_similarity(m, MemoryView::scoped(m, &ctx)))
                .collect();
            demote_out_of_scope(views, |v| {
                v.scope.as_deref().map_or(Scope::Unknown, scope_from_str)
            })
        } else {
            hits.iter()
                .map(|m| with_similarity(m, MemoryView::from(m)))
                .collect()
        };
        Ok(Json(MemoriesOut { memories }))
    }

    /// Ingest a knowledge document: chunk, embed, and store it for recall. A
    /// document is reference material the agent was *given*, kept distinct from
    /// the episodic memory it *earned* so neither drowns the other.
    #[tool(
        description = "Ingest a knowledge document into your workspace: it is split into overlapping chunks, each embedded for semantic recall (kept distinct from episodic memory). Returns the title and the number of chunks stored."
    )]
    async fn ingest_document(
        &self,
        Parameters(p): Parameters<IngestDocumentParams>,
    ) -> Result<Json<IngestedOut>, ErrorData> {
        let provenance = p.provenance.map(anchor_from).transpose()?;
        let compartment = p
            .compartment
            .filter(|c| !c.trim().is_empty())
            .map(CompartmentId::new);
        // Ask before anything is archived. The ingest uploads the original first
        // and the engine refuses a chunk it may not write by silently persisting
        // nothing, so a refusal found afterwards would already have left the
        // original in the archive under a compartment its author cannot write to.
        if let Some(compartment) = &compartment {
            let writable =
                compartment::can_write(&self.store, &self.tenant, &self.user, compartment)
                    .await
                    .map_err(err)?;
            if !writable {
                return Err(ErrorData::invalid_params(
                    format!(
                        "cannot ingest into compartment '{}': it is not one you can write to \
                         (create it with create_compartment first, or omit it for the shared pool)",
                        compartment.as_str()
                    ),
                    None,
                ));
            }
        }
        let doc = antumbra_ingest::Document {
            title: p.title.clone(),
            source: p.source,
            content: p.content,
            provenance,
            compartment,
        };
        // One ingest path for every door (`antumbra-ingest`): the original to
        // copal first, fail closed; the title's chunks replaced in place; the
        // anchor folded into every chunk's source.
        let out = antumbra_ingest::ingest_text(
            &self.store,
            self.embedder.as_ref(),
            &self.tenant,
            self.copal.as_deref(),
            &doc,
        )
        .await
        .map_err(|e| err(format!("{e:#}")))?;
        Ok(Json(IngestedOut {
            title: p.title,
            chunks: out.chunks,
        }))
    }

    /// Semantic recall over the ingested knowledge documents.
    #[tool(
        description = "Recall the document chunks most relevant to a query from your ingested knowledge documents (semantic search, separate from episodic memory recall)."
    )]
    async fn recall_documents(
        &self,
        Parameters(p): Parameters<RecallDocumentsParams>,
    ) -> Result<Json<DocumentChunksOut>, ErrorData> {
        let q = self.embedder.embed(&p.query).await.map_err(err)?;
        let k = p.top_k.unwrap_or(5) as usize;
        // Hybrid recall (dense HNSW + sparse BM25, RRF-fused), like memory recall;
        // wide pool + cross-encoder rerank when configured.
        let hits =
            document::recall_hybrid(&self.store, &self.tenant, &p.query, &q, self.recall_pool(k))
                .await
                .map_err(err)?;
        let hits = self
            .rerank_to_k(
                &p.query,
                hits,
                k,
                |c| c.id.as_str().to_string(),
                |c| c.content.clone(),
            )
            .await;
        Ok(Json(DocumentChunksOut {
            chunks: hits.iter().map(DocumentChunkView::from).collect(),
        }))
    }

    /// The expert population visible to this session (read-only observability for
    /// a dashboard / status view, P-2). The expert ACL already scopes the list to
    /// shared experts plus this user's own private ones.
    #[tool(
        description = "List the expert population visible to you (shared experts plus your own private ones), each with its generation and fitness."
    )]
    async fn population(&self) -> Result<Json<PopulationOut>, ErrorData> {
        let experts = expert::list(&self.store).await.map_err(err)?;
        Ok(Json(PopulationOut {
            experts: experts
                .iter()
                .map(|e| ExpertView {
                    id: e.id.as_str().to_string(),
                    name: e.name.clone(),
                    generation: e.generation.0,
                    fitness: e.fitness,
                    private: e.owner.as_ref() == Some(&self.user),
                })
                .collect(),
        }))
    }

    /// At-a-glance counts for this workspace (read-only, ACL-scoped).
    #[tool(
        description = "At-a-glance counts for your workspace: memories, knowledge documents, visible experts, boundaries, and your compartments."
    )]
    async fn workspace_stats(&self) -> Result<Json<StatsOut>, ErrorData> {
        let memories = memory::list(&self.store, &self.tenant)
            .await
            .map_err(err)?
            .len() as u32;
        let documents = document::list_titles(&self.store, &self.tenant)
            .await
            .map_err(err)?
            .len() as u32;
        let experts = expert::list(&self.store).await.map_err(err)?.len() as u32;
        let boundaries = boundary::list(&self.store).await.map_err(err)?.len() as u32;
        let compartments = compartment::list_owned(&self.store, &self.tenant, &self.user)
            .await
            .map_err(err)?
            .len() as u32;
        Ok(Json(StatsOut {
            memories,
            documents,
            experts,
            boundaries,
            compartments,
        }))
    }

    /// Reinforce a memory that proved useful.
    #[tool(
        description = "Reinforce a memory (raise its strength and recurrence) when it proves useful."
    )]
    async fn reinforce_memory(
        &self,
        Parameters(p): Parameters<IdParams>,
    ) -> Result<Json<ReinforceOut>, ErrorData> {
        match memory::reinforce(
            &self.store,
            &self.tenant,
            &MemoryId::new(p.memory_id),
            Utc::now(),
        )
        .await
        .map_err(err)?
        {
            Some(m) => {
                // A reinforcement may push this memory's compartment over the
                // consolidation gate; graduate it into a private expert if so.
                self.maybe_consolidate(&m).await;
                Ok(Json(ReinforceOut {
                    found: true,
                    reinforcement: m.reinforcement,
                    confidence: m.confidence,
                }))
            }
            None => Ok(Json(ReinforceOut {
                found: false,
                reinforcement: 0,
                confidence: 0.0,
            })),
        }
    }

    /// Penalize a memory whose thesis was falsified (the inverse of reinforce).
    #[tool(
        description = "Penalize a memory (a falsified thesis, e.g. a losing trade): decay its confidence so it does not clear the consolidation gate and graduate into an expert. The inverse of reinforce_memory."
    )]
    async fn penalize_memory(
        &self,
        Parameters(p): Parameters<IdParams>,
    ) -> Result<Json<ReinforceOut>, ErrorData> {
        // No consolidation trigger here: a penalty can only LOWER confidence, so unlike
        // reinforce it never pushes a compartment over the graduation gate.
        match memory::penalize(
            &self.store,
            &self.tenant,
            &MemoryId::new(p.memory_id),
            Utc::now(),
        )
        .await
        .map_err(err)?
        {
            Some(m) => Ok(Json(ReinforceOut {
                found: true,
                reinforcement: m.reinforcement,
                confidence: m.confidence,
            })),
            None => Ok(Json(ReinforceOut {
                found: false,
                reinforcement: 0,
                confidence: 0.0,
            })),
        }
    }

    /// Forget (delete) a memory.
    #[tool(description = "Forget (delete) a memory from your workspace's memory.")]
    async fn forget_memory(
        &self,
        Parameters(p): Parameters<IdParams>,
    ) -> Result<Json<ForgetOut>, ErrorData> {
        // Soft-delete (tombstone): hidden from reads here, and the deletion
        // propagates across the fleet (R-1) and routes to grantees (R-2) instead
        // of resurfacing from another replica.
        let forgotten = memory::soft_delete(
            &self.store,
            &self.tenant,
            &MemoryId::new(p.memory_id),
            Utc::now(),
        )
        .await
        .map_err(err)?
        .is_some();
        Ok(Json(ForgetOut { forgotten }))
    }

    /// List this tenant's memories (optionally one network).
    #[tool(
        description = "List your workspace's memories, optionally filtered to one network (world/bank/opinion)."
    )]
    async fn list_memories(
        &self,
        Parameters(p): Parameters<ListParams>,
    ) -> Result<Json<MemoriesOut>, ErrorData> {
        let mems = match p.network.as_deref().map(parse_network) {
            Some(net) => memory::list_by_network(&self.store, &self.tenant, net)
                .await
                .map_err(err)?,
            None => memory::list(&self.store, &self.tenant).await.map_err(err)?,
        };
        Ok(Json(MemoriesOut {
            memories: mems.iter().map(MemoryView::from).collect(),
        }))
    }

    /// Relate two memories with a typed edge (the Penumbra graph).
    #[tool(
        description = "Relate two memories with a typed edge: references/supersedes/contradicts/follows/caused."
    )]
    async fn relate_memories(
        &self,
        Parameters(p): Parameters<RelateParams>,
    ) -> Result<Json<RelateOut>, ErrorData> {
        let e = MemoryEdge::new(
            self.tenant.clone(),
            p.from_id,
            p.to_id,
            parse_edge_type(&p.edge_type),
            p.weight.unwrap_or(1.0),
            Utc::now(),
        );
        edge::relate(&self.store, &e).await.map_err(err)?;
        Ok(Json(RelateOut { related: true }))
    }

    /// The memories connected from a memory (optionally one edge type).
    #[tool(
        description = "Get the memories connected from a memory (optionally filtered to one edge type)."
    )]
    async fn get_neighbors(
        &self,
        Parameters(p): Parameters<NeighborsParams>,
    ) -> Result<Json<NeighborsOut>, ErrorData> {
        let et = p.edge_type.as_deref().map(parse_edge_type);
        let from = MemoryId::new(p.memory_id);
        // Only list a memory's edges if the caller can actually see that memory:
        // edge rows are tenant-scoped, so without this a caller who knows another
        // user's private memory id could enumerate its edge structure (types,
        // weights). Target content is already gated below via memory::get.
        if memory::get(&self.store, &self.tenant, &from)
            .await
            .map_err(err)?
            .is_none()
        {
            return Ok(Json(NeighborsOut {
                neighbors: Vec::new(),
            }));
        }
        let edges = edge::neighbors(&self.store, &self.tenant, &from, et)
            .await
            .map_err(err)?;
        let mut neighbors = Vec::new();
        for e in &edges {
            if let Some(m) = memory::get(&self.store, &self.tenant, &e.to_id)
                .await
                .map_err(err)?
            {
                neighbors.push(NeighborView {
                    edge_type: e.edge_type.as_str().to_string(),
                    weight: e.weight,
                    memory: MemoryView::from(&m),
                });
            }
        }
        Ok(Json(NeighborsOut { neighbors }))
    }

    /// Route a task across the shared expert population (the brain). Returns the
    /// covering expert(s) ranked, or escalate when the task is out of
    /// distribution. Pure-arithmetic gate inference (no model load).
    #[tool(
        description = "Route a task across the shared population AND your private experts: which expert(s) cover it, ranked, or escalate if none."
    )]
    async fn route(
        &self,
        Parameters(p): Parameters<RouteParams>,
    ) -> Result<Json<RouteOut>, ErrorData> {
        let v = self.embedder.embed(&p.task).await.map_err(err)?;
        let routes = self
            .ranked_routes(&v, p.top_k.unwrap_or(3) as usize)
            .await
            .map_err(err)?;
        let covered = !routes.is_empty();
        Ok(Json(RouteOut {
            covered,
            escalate: !covered,
            routes,
        }))
    }

    /// Route a task and serve the answer through the covering expert's adapter
    /// (the full recall→route→serve surface). Escalates when nothing covers it
    /// or when no serving engine is configured.
    #[tool(
        description = "Answer a task: route it across the shared population and your private experts, then generate a response through the covering expert's adapter. Escalates if nothing covers it."
    )]
    async fn answer(
        &self,
        Parameters(p): Parameters<AnswerParams>,
    ) -> Result<Json<AnswerOut>, ErrorData> {
        let Some(serve) = self.serve.clone() else {
            return Ok(Json(AnswerOut {
                answer: String::new(),
                expert_id: None,
                escalate: true,
                note: Some("serving not configured (run the server with a serving engine / --features models)".into()),
            }));
        };
        let v = self.embedder.embed(&p.task).await.map_err(err)?;
        let routes = self.ranked_routes(&v, 1).await.map_err(err)?;
        let Some(top) = routes.first() else {
            return Ok(Json(AnswerOut {
                answer: String::new(),
                expert_id: None,
                escalate: true,
                note: Some("no in-scope expert; escalate".into()),
            }));
        };
        let expert_id = top.expert_id.clone();
        let expert = ExpertId::new(expert_id.clone());
        // The serving engine snapshots its adapter population at startup; a route
        // to an expert it can't serve (e.g. one minted afterward) escalates cleanly
        // rather than surfacing a "no adapter registered" error.
        if !serve.can_serve(&expert) {
            return Ok(Json(AnswerOut {
                answer: String::new(),
                expert_id: Some(expert_id),
                escalate: true,
                note: Some("covering expert not resident in the serving engine; escalate".into()),
            }));
        }
        // Generation is synchronous compute inside an `async fn`, like a train: it
        // runs off the runtime's workers so other sessions' calls keep moving.
        let request = ActRequest {
            task_id: next_id("answer"),
            prompt: p.task,
            adapters: vec![expert],
        };
        let out = consolidation::spawn_heavy(async move { serve.act(request).await })
            .await
            .map_err(err)?
            .map_err(err)?;
        Ok(Json(AnswerOut {
            answer: out.final_output,
            expert_id: Some(expert_id),
            escalate: false,
            note: None,
        }))
    }

    /// Create a private compartment (latent-space) owned by you.
    #[tool(
        description = "Create a new compartment (a private latent-space of memory) you own. Store into it via store_memory's compartment arg. Returns its id."
    )]
    async fn create_compartment(
        &self,
        Parameters(p): Parameters<CreateCompartmentParams>,
    ) -> Result<Json<CompartmentView>, ErrorData> {
        let id = next_id("comp");
        let c = Compartment::new(
            id.clone(),
            self.tenant.clone(),
            self.user.clone(),
            p.name.clone(),
            Utc::now(),
        );
        compartment::create(&self.store, &c).await.map_err(err)?;
        Ok(Json(CompartmentView {
            id,
            name: p.name,
            origin: Origin::User.as_str().to_string(),
        }))
    }

    /// Propose compartments by clustering your uncompartmented memories.
    #[tool(
        description = "Have the antumbra propose compartments by clustering your uncompartmented memories into competence-coherent regions. Returns proposals (label, member ids, cohesion). With apply=true it also creates each as a proposed compartment you own and moves its members in (reversible by deleting the compartment)."
    )]
    async fn propose_compartments(
        &self,
        Parameters(p): Parameters<ProposeCompartmentsParams>,
    ) -> Result<Json<ProposalsOut>, ErrorData> {
        let candidates = self.inbox_pool().await.map_err(err)?;
        let cfg = ClusterConfig {
            similarity_threshold: p.similarity_threshold,
            min_size: p.min_size,
            ..ClusterConfig::default()
        };
        // Fully-qualified: the crate fn shares this tool's name.
        let proposals = antumbra_core::propose_compartments(&candidates, &cfg);
        let mut views = Vec::with_capacity(proposals.len());
        for prop in proposals {
            let compartment_id = if p.apply {
                Some(self.apply_proposal(&prop).await.map_err(err)?)
            } else {
                None
            };
            views.push(ProposalView {
                label: prop.label,
                members: prop
                    .members
                    .iter()
                    .map(|m| m.as_str().to_string())
                    .collect(),
                cohesion: prop.cohesion,
                compartment_id,
            });
        }
        Ok(Json(ProposalsOut { proposals: views }))
    }

    /// List the compartments you own.
    #[tool(description = "List the compartments you own (including any the antumbra proposed).")]
    async fn list_compartments(&self) -> Result<Json<CompartmentsOut>, ErrorData> {
        let comps = compartment::list_owned(&self.store, &self.tenant, &self.user)
            .await
            .map_err(err)?;
        Ok(Json(CompartmentsOut {
            compartments: comps
                .iter()
                .map(|c| CompartmentView {
                    id: c.id.as_str().to_string(),
                    name: c.name.clone(),
                    origin: c.origin.as_str().to_string(),
                })
                .collect(),
        }))
    }

    /// Share a compartment you own with another user.
    #[tool(
        description = "Share one of your compartments with another user: reference (they can recall it) or link (they can also connect to it)."
    )]
    async fn share_compartment(
        &self,
        Parameters(p): Parameters<ShareParams>,
    ) -> Result<Json<ShareOut>, ErrorData> {
        let g = Grant::new(
            self.tenant.clone(),
            CompartmentId::new(p.compartment_id),
            UserId::new(p.grantee),
            parse_capability(&p.capability),
            self.user.clone(),
            Utc::now(),
        );
        compartment::grant(&self.store, &g).await.map_err(err)?;
        Ok(Json(ShareOut { shared: true }))
    }

    /// Revoke a user's access to one of your compartments.
    #[tool(
        description = "Revoke a user's access to one of your compartments (takes effect immediately)."
    )]
    async fn revoke_compartment(
        &self,
        Parameters(p): Parameters<RevokeParams>,
    ) -> Result<Json<RevokeOut>, ErrorData> {
        compartment::revoke(
            &self.store,
            &self.tenant,
            &CompartmentId::new(p.compartment_id),
            &UserId::new(p.grantee),
            Utc::now(),
        )
        .await
        .map_err(err)?;
        Ok(Json(RevokeOut { revoked: true }))
    }
}

#[cfg(test)]
mod tests;
