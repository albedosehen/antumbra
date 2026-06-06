//! The Antumbra MCP tool surface over the Penumbra memory store.
//!
//! Every tool operates on a single bound tenant (the workspace this server was
//! started for); the Store session is signed in as that tenant, so reads are
//! engine-enforced — the server cannot serve another tenant's memory even if a
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
    Capability, ClusterConfig, Compartment, CompartmentId, EdgeType, ExpertId, Grant, Memory,
    MemoryEdge, MemoryId, MemoryNetwork, Origin, TenantId, UserId,
};
use antumbra_store::repo::{compartment, edge, expert, memory, router};
use antumbra_store::Store;

/// One (tenant, user) MCP session over its Penumbra. `#[tool_handler]` resolves
/// the tools via `Self::tool_router()`, so no router field is stored.
#[derive(Clone)]
pub struct McpServer {
    store: Store,
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
    /// The serving engine the `answer` tool drives (route → serve through the
    /// expert's adapter). `None` = serving not configured (route-only surface).
    serve: Option<Arc<dyn antumbra_core::ports::Serve>>,
    /// Where this session registers its peer on initialize, so live propagation
    /// (R-2) can push shared-memory changes to it. `None` = no live delivery
    /// (stdio, route-only, or tests).
    registry: Option<crate::notify::PeerRegistry>,
    counter: Arc<AtomicU64>,
}

/// Tuning for the autonomous propose trigger.
#[derive(Clone)]
struct AutoProposeConfig {
    /// Inbox size at/above which a proposal pass fires after a write.
    threshold: usize,
    min_size: usize,
    similarity_threshold: f32,
}

impl McpServer {
    /// `serve` is the engine the `answer` tool drives (a real `MultiAdapterServe`
    /// under `--features models`, a fake in tests, or `None` for a route-only
    /// surface where `answer` reports serving is not configured).
    pub fn new(
        store: Store,
        embedder: Arc<dyn Embedder>,
        tenant: TenantId,
        user: UserId,
        host: String,
        default_compartment: CompartmentId,
        serve: Option<Arc<dyn antumbra_core::ports::Serve>>,
    ) -> Self {
        Self {
            store,
            embedder,
            tenant,
            user,
            host,
            default_compartment,
            auto_propose: None,
            serve,
            registry: None,
            counter: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Register this session's peer into `registry` on initialize, so live
    /// shared-memory changes (R-2) are pushed to it over its SSE stream.
    #[must_use]
    pub fn with_registry(mut self, registry: crate::notify::PeerRegistry) -> Self {
        self.registry = Some(registry);
        self
    }

    /// Enable the autonomous propose trigger: once the unorganized inbox reaches
    /// `threshold` memories, a write auto-clusters it into `Origin::Proposed`
    /// compartments (reversible — the user curates). Off by default.
    #[must_use]
    pub fn with_auto_propose(mut self, threshold: usize) -> Self {
        self.auto_propose = Some(AutoProposeConfig {
            threshold,
            min_size: 3,
            similarity_threshold: 0.6,
        });
        self
    }

    /// The *unorganized* memory pool: the inbox (default compartment) plus
    /// anything uncompartmented. Deliberately-filed compartments are left alone —
    /// the antumbra proposes structure only over what the user has not organized.
    async fn inbox_pool(&self) -> antumbra_core::Result<Vec<Memory>> {
        Ok(memory::list(&self.store, &self.tenant)
            .await?
            .into_iter()
            .filter(|m| {
                m.compartment.is_none() || m.compartment.as_ref() == Some(&self.default_compartment)
            })
            .collect())
    }

    /// Persist one proposal as an `Origin::Proposed` compartment you own and move
    /// its members in. Reversible: deleting the compartment undoes it. Returns the
    /// new compartment id.
    async fn apply_proposal(
        &self,
        prop: &antumbra_core::ProposedCompartment,
    ) -> antumbra_core::Result<String> {
        let id = next_id(&self.counter, "comp");
        let c = Compartment::new(
            id.clone(),
            self.tenant.clone(),
            self.user.clone(),
            prop.label.clone(),
            Utc::now(),
        )
        .proposed();
        compartment::create(&self.store, &c).await?;
        for mid in &prop.members {
            if let Some(mut m) = memory::get(&self.store, &self.tenant, mid).await? {
                m.compartment = Some(CompartmentId::new(id.clone()));
                m.updated_at = Utc::now();
                memory::upsert(&self.store, &m).await?;
            }
        }
        Ok(id)
    }

    /// The autonomous propose trigger: once the inbox reaches the configured
    /// threshold, cluster it and auto-create the proposals (so the inbox shrinks
    /// below the threshold and the trigger quiets until it grows again). Returns
    /// the created compartment ids; empty when disabled or below threshold.
    async fn maybe_auto_propose(&self) -> antumbra_core::Result<Vec<String>> {
        let Some(cfg) = self.auto_propose.clone() else {
            return Ok(Vec::new());
        };
        let pool = self.inbox_pool().await?;
        if pool.len() < cfg.threshold {
            return Ok(Vec::new());
        }
        let cluster_cfg = ClusterConfig {
            similarity_threshold: cfg.similarity_threshold,
            min_size: cfg.min_size,
            ..ClusterConfig::default()
        };
        let proposals = antumbra_core::propose_compartments(&pool, &cluster_cfg);
        let mut created = Vec::with_capacity(proposals.len());
        for prop in &proposals {
            created.push(self.apply_proposal(prop).await?);
        }
        Ok(created)
    }

    /// Rank the experts covering an embedded task: shared experts via the learned
    /// router, plus the user's own private experts by centroid. Top-`k`, best
    /// first. The expert ACL already scopes `expert::list` to shared + own-private.
    async fn ranked_routes(&self, v: &[f32], k: usize) -> antumbra_core::Result<Vec<RouteHit>> {
        let mut routes: Vec<RouteHit> = Vec::new();
        if let Some(router) = router::load(&self.store).await? {
            if router.covers(v) {
                for (id, probability) in router.route(v) {
                    routes.push(RouteHit {
                        expert_id: id.as_str().to_string(),
                        probability,
                        private: false,
                    });
                }
            }
        }
        for e in expert::list(&self.store).await? {
            if e.owner.as_ref() == Some(&self.user) {
                if let Some(sim) = e.capability_similarity(v) {
                    if sim >= PRIVATE_ROUTE_FLOOR {
                        routes.push(RouteHit {
                            expert_id: e.id.as_str().to_string(),
                            probability: sim,
                            private: true,
                        });
                    }
                }
            }
        }
        routes.sort_by(|a, b| {
            b.probability
                .partial_cmp(&a.probability)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        routes.truncate(k);
        Ok(routes)
    }
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

/// A process-unique, monotonic id with a table prefix (timestamp + counter; no
/// extra deps).
fn next_id(counter: &AtomicU64, prefix: &str) -> String {
    let n = counter.fetch_add(1, Ordering::Relaxed);
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

#[derive(Deserialize, schemars::JsonSchema)]
struct StoreParams {
    /// The text to remember.
    content: String,
    /// Network: `world` (facts), `bank` (experiences), `opinion` (judgments).
    #[serde(default = "default_network")]
    network: String,
    /// Initial confidence in `[0,1]` (default 0.6).
    confidence: Option<f32>,
    /// Provenance sources for the memory.
    evidence: Option<Vec<String>>,
    /// `true` if the fact changes over time (kept in store, never consolidated).
    volatile: Option<bool>,
    /// The compartment to store into. Omit to use this session's default space.
    compartment: Option<String>,
}

#[derive(Serialize, schemars::JsonSchema)]
struct StoredOut {
    id: String,
    /// Compartment ids the antumbra auto-created from the inbox on this write
    /// (only when the autonomous propose trigger is enabled and fired).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    auto_proposed: Vec<String>,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct RecallParams {
    /// What to recall — embedded and matched by semantic similarity.
    query: String,
    /// How many to return (default 5).
    top_k: Option<u32>,
    /// Optional network filter (`world`/`bank`/`opinion`).
    network: Option<String>,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct ListParams {
    /// Optional network filter (`world`/`bank`/`opinion`).
    network: Option<String>,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct IdParams {
    memory_id: String,
}

#[derive(Serialize, schemars::JsonSchema)]
struct MemoryView {
    id: String,
    content: String,
    network: String,
    confidence: f32,
    reinforcement: u32,
}

impl From<&Memory> for MemoryView {
    fn from(m: &Memory) -> Self {
        Self {
            id: m.id.as_str().to_string(),
            content: m.content.clone(),
            network: m.network.as_str().to_string(),
            confidence: m.confidence,
            reinforcement: m.reinforcement,
        }
    }
}

#[derive(Serialize, schemars::JsonSchema)]
struct MemoriesOut {
    memories: Vec<MemoryView>,
}

#[derive(Serialize, schemars::JsonSchema)]
struct ReinforceOut {
    found: bool,
    reinforcement: u32,
    confidence: f32,
}

#[derive(Serialize, schemars::JsonSchema)]
struct ForgetOut {
    forgotten: bool,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct RelateParams {
    from_id: String,
    to_id: String,
    /// `references` / `supersedes` / `contradicts` / `follows` / `caused`.
    #[serde(default = "default_edge_type")]
    edge_type: String,
    /// Edge strength (default 1.0).
    weight: Option<f32>,
}

#[derive(Serialize, schemars::JsonSchema)]
struct RelateOut {
    related: bool,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct NeighborsParams {
    memory_id: String,
    /// Optional edge-type filter.
    edge_type: Option<String>,
}

#[derive(Serialize, schemars::JsonSchema)]
struct NeighborView {
    edge_type: String,
    weight: f32,
    memory: MemoryView,
}

#[derive(Serialize, schemars::JsonSchema)]
struct NeighborsOut {
    neighbors: Vec<NeighborView>,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct RouteParams {
    /// The task to route across the shared expert population.
    task: String,
    /// How many candidate experts to return (default 3).
    top_k: Option<u32>,
}

/// Minimum cosine similarity for one of the user's *private* experts to be
/// offered as a route candidate (a heuristic floor — private experts are not in
/// the shared learned router, so they are matched directly by centroid; a
/// per-private-expert learned boundary is the eventual refinement).
const PRIVATE_ROUTE_FLOOR: f32 = 0.3;

#[derive(Serialize, schemars::JsonSchema)]
struct RouteHit {
    expert_id: String,
    probability: f32,
    /// `true` if this is one of *your* private experts (consolidated from your
    /// compartment), matched by centroid; `false` for a shared expert.
    private: bool,
}

#[derive(Serialize, schemars::JsonSchema)]
struct RouteOut {
    /// Whether the population covers this task (vs out-of-distribution).
    covered: bool,
    /// `true` when no expert covers it — defer to the generalist.
    escalate: bool,
    routes: Vec<RouteHit>,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct AnswerParams {
    /// The task to route and answer through the covering expert.
    task: String,
}

#[derive(Serialize, schemars::JsonSchema)]
struct AnswerOut {
    /// The generated answer (empty when escalating).
    answer: String,
    /// The expert that served it, when one covered the task.
    #[serde(skip_serializing_if = "Option::is_none")]
    expert_id: Option<String>,
    /// `true` when no expert covered it, or serving is not configured.
    escalate: bool,
    /// Why it escalated, when it did.
    #[serde(skip_serializing_if = "Option::is_none")]
    note: Option<String>,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct CreateCompartmentParams {
    /// A display name for the new compartment.
    name: String,
}

#[derive(Serialize, schemars::JsonSchema)]
struct CompartmentView {
    id: String,
    name: String,
    origin: String,
}

#[derive(Serialize, schemars::JsonSchema)]
struct CompartmentsOut {
    compartments: Vec<CompartmentView>,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct ShareParams {
    compartment_id: String,
    /// The user to share with (a user id in this tenant).
    grantee: String,
    /// `reference` (recall) or `link` (also connect). Defaults to reference.
    #[serde(default = "default_capability")]
    capability: String,
}

#[derive(Serialize, schemars::JsonSchema)]
struct ShareOut {
    shared: bool,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct RevokeParams {
    compartment_id: String,
    grantee: String,
}

#[derive(Serialize, schemars::JsonSchema)]
struct RevokeOut {
    revoked: bool,
}

fn default_threshold() -> f32 {
    0.6
}

fn default_min_size() -> usize {
    3
}

#[derive(Deserialize, schemars::JsonSchema)]
struct ProposeCompartmentsParams {
    /// Cosine at/above which two memories cluster together (default 0.6).
    #[serde(default = "default_threshold")]
    similarity_threshold: f32,
    /// Smallest cluster worth proposing; singletons and pairs are noise (default 3).
    #[serde(default = "default_min_size")]
    min_size: usize,
    /// Persist each proposal as an `Origin::Proposed` compartment you own and
    /// move its members into it. Default false (suggest only; reversible by
    /// deleting the compartment).
    #[serde(default)]
    apply: bool,
}

#[derive(Serialize, schemars::JsonSchema)]
struct ProposalView {
    /// Heuristic label from the cluster's most central memory; rename on accept.
    label: String,
    /// The memory ids grouped into this proposed region.
    members: Vec<String>,
    /// Mean cosine of members to the centroid — rank proposals by this.
    cohesion: f32,
    /// Set when `apply` was true: the id of the created proposed compartment.
    #[serde(skip_serializing_if = "Option::is_none")]
    compartment_id: Option<String>,
}

#[derive(Serialize, schemars::JsonSchema)]
struct ProposalsOut {
    proposals: Vec<ProposalView>,
}

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
        let id = next_id(&self.counter, "memory");
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
        if let Some(ev) = p.evidence {
            m = m.with_evidence(ev);
        }
        if p.volatile.unwrap_or(false) {
            m = m.volatile(true);
        }
        memory::upsert(&self.store, &m).await.map_err(err)?;
        // Autonomous trigger: if the inbox has grown enough, the antumbra
        // organizes it now (no-op when disabled or below threshold).
        let auto_proposed = self.maybe_auto_propose().await.map_err(err)?;
        Ok(Json(StoredOut { id, auto_proposed }))
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
        let hits = memory::recall(
            &self.store,
            &self.tenant,
            &q,
            p.top_k.unwrap_or(5) as usize,
            net,
        )
        .await
        .map_err(err)?;
        Ok(Json(MemoriesOut {
            memories: hits.iter().map(MemoryView::from).collect(),
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
        let edges = edge::neighbors(&self.store, &self.tenant, &MemoryId::new(p.memory_id), et)
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
        let out = serve
            .act(ActRequest {
                task_id: next_id(&self.counter, "answer"),
                prompt: p.task,
                adapters: vec![ExpertId::new(expert_id.clone())],
            })
            .await
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
        let id = next_id(&self.counter, "comp");
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

#[tool_handler]
impl ServerHandler for McpServer {
    /// On initialize, record this session's peer under its (tenant, user) identity
    /// so the live-propagation watcher can push shared-memory changes to it (R-2).
    /// A no-op when no registry is wired (stdio / route-only / tests).
    async fn on_initialized(
        &self,
        context: rmcp::service::NotificationContext<rmcp::service::RoleServer>,
    ) {
        // Bind THIS session's connection to its identity. rmcp builds one server
        // (and, on a remote, one DB connection) per session, so the binding must
        // happen here -- the HTTP layer's signin runs on a different handle and a
        // cloned remote connection does not share it. Signing in as the record
        // scopes the engine ACL for every tool call in this session (R-6).
        if let Err(e) = self.store.signin(&self.tenant, &self.user).await {
            eprintln!(
                "antumbra-mcp: session signin failed for {}/{}: {e}",
                self.tenant.as_str(),
                self.user.as_str()
            );
        }
        if let Some(registry) = &self.registry {
            let identity = crate::auth::Identity {
                tenant: self.tenant.as_str().to_string(),
                user: self.user.as_str().to_string(),
            };
            registry.register(identity, context.peer.clone()).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antumbra_core::router::{LearnedRouter, RouterExpert};
    use antumbra_core::testing::FixedEmbedder;
    use antumbra_core::{Expert, ExpertId, Generation};
    use antumbra_store::repo::{expert, router};
    use antumbra_store::EMBED_DIM;
    use chrono::Utc;

    async fn server() -> McpServer {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        McpServer::new(
            store,
            Arc::new(FixedEmbedder::new(EMBED_DIM)),
            TenantId::new("ws:test"),
            UserId::new("user:test"),
            "test-host".into(),
            CompartmentId::new("comp:test:default"),
            None,
        )
    }

    #[tokio::test]
    async fn store_recall_reinforce_list_forget_roundtrip() {
        let s = server().await;

        let stored = s
            .store_memory(Parameters(StoreParams {
                content: "In acme-api use deno install, not npm".into(),
                network: "opinion".into(),
                confidence: Some(0.9),
                evidence: None,
                volatile: None,
                compartment: None,
            }))
            .await
            .unwrap();
        let id = stored.0.id.clone();
        assert!(id.starts_with("memory:"));

        let recalled = s
            .recall_memories(Parameters(RecallParams {
                query: "how do I add a dependency in acme-api".into(),
                top_k: Some(5),
                network: None,
            }))
            .await
            .unwrap();
        assert_eq!(recalled.0.memories.len(), 1);
        assert!(recalled.0.memories[0].content.contains("deno"));

        let r = s
            .reinforce_memory(Parameters(IdParams {
                memory_id: id.clone(),
            }))
            .await
            .unwrap();
        assert!(r.0.found);
        assert_eq!(r.0.reinforcement, 1);

        let listed = s
            .list_memories(Parameters(ListParams { network: None }))
            .await
            .unwrap();
        assert_eq!(listed.0.memories.len(), 1);

        s.forget_memory(Parameters(IdParams { memory_id: id }))
            .await
            .unwrap();
        assert!(s
            .list_memories(Parameters(ListParams { network: None }))
            .await
            .unwrap()
            .0
            .memories
            .is_empty());
    }

    #[tokio::test]
    async fn relate_and_get_neighbors() {
        let s = server().await;
        let store = |content: &str| StoreParams {
            content: content.into(),
            network: "world".into(),
            confidence: None,
            evidence: None,
            volatile: None,
            compartment: None,
        };
        let a = s
            .store_memory(Parameters(store("a deno project")))
            .await
            .unwrap()
            .0
            .id;
        let b = s
            .store_memory(Parameters(store("use deno install")))
            .await
            .unwrap()
            .0
            .id;

        s.relate_memories(Parameters(RelateParams {
            from_id: a.clone(),
            to_id: b.clone(),
            edge_type: "supersedes".into(),
            weight: Some(0.8),
        }))
        .await
        .unwrap();

        let neighbors = s
            .get_neighbors(Parameters(NeighborsParams {
                memory_id: a,
                edge_type: None,
            }))
            .await
            .unwrap();
        assert_eq!(neighbors.0.neighbors.len(), 1);
        assert_eq!(neighbors.0.neighbors[0].edge_type, "supersedes");
        assert_eq!(neighbors.0.neighbors[0].memory.id, b);
    }

    #[tokio::test]
    async fn route_escalates_without_router_then_routes_with_one() {
        let s = server().await;

        // No router trained yet -> escalate (out of distribution).
        let r = s
            .route(Parameters(RouteParams {
                task: "add two numbers".into(),
                top_k: None,
            }))
            .await
            .unwrap();
        assert!(r.0.escalate && !r.0.covered && r.0.routes.is_empty());

        // Save a permissive router with one expert.
        router::save(
            &s.store,
            &LearnedRouter {
                weights: vec![1.0; EMBED_DIM],
                experts: vec![RouterExpert {
                    id: ExpertId::new("expert:adder"),
                    centroid: vec![0.0; EMBED_DIM],
                }],
                temperature: 0.1,
                floor: -1.0,
            },
        )
        .await
        .unwrap();

        let r = s
            .route(Parameters(RouteParams {
                task: "add two numbers".into(),
                top_k: Some(3),
            }))
            .await
            .unwrap();
        assert!(r.0.covered && !r.0.escalate);
        assert_eq!(r.0.routes.len(), 1);
        assert_eq!(r.0.routes[0].expert_id, "expert:adder");

        // A PRIVATE expert owned by this session's user is matched by centroid
        // and offered alongside the shared route (flagged private).
        let cap = FixedEmbedder::new(EMBED_DIM)
            .embed("my private skill")
            .await
            .unwrap();
        let now = Utc::now();
        expert::insert(
            &s.store,
            &Expert {
                id: ExpertId::new("expert:mine"),
                name: "mine".into(),
                base_model: "base".into(),
                artifact_uri: "mem://mine".into(),
                capability_card: serde_json::Value::Null,
                capability_vec: Some(cap),
                fitness: 1.0,
                frozen_at: Some(now),
                generation: Generation::ZERO,
                owner: Some(UserId::new("user:test")),
                compartment: Some(CompartmentId::new("comp:test")),
                created_at: now,
            },
        )
        .await
        .unwrap();
        let r = s
            .route(Parameters(RouteParams {
                task: "my private skill".into(),
                top_k: Some(5),
            }))
            .await
            .unwrap();
        assert!(
            r.0.routes
                .iter()
                .any(|h| h.private && h.expert_id == "expert:mine"),
            "the user's private expert must be routable"
        );
    }

    #[tokio::test]
    async fn compartment_tools_create_list_store_share() {
        let s = server().await;

        let c = s
            .create_compartment(Parameters(CreateCompartmentParams {
                name: "deno work".into(),
            }))
            .await
            .unwrap();
        assert!(c.0.id.starts_with("comp:"));
        assert_eq!(c.0.origin, "user");

        let listed = s.list_compartments().await.unwrap();
        assert!(listed.0.compartments.iter().any(|x| x.id == c.0.id));

        // Store a memory explicitly into the new compartment.
        let m = s
            .store_memory(Parameters(StoreParams {
                content: "use deno install".into(),
                network: "opinion".into(),
                confidence: None,
                evidence: None,
                volatile: None,
                compartment: Some(c.0.id.clone()),
            }))
            .await
            .unwrap();
        assert!(m.0.id.starts_with("memory:"));

        // Share + revoke succeed (engine enforcement is proven in the store
        // crate's penumbra_compartment test; here we exercise the tool plumbing).
        assert!(
            s.share_compartment(Parameters(ShareParams {
                compartment_id: c.0.id.clone(),
                grantee: "user:other".into(),
                capability: "reference".into(),
            }))
            .await
            .unwrap()
            .0
            .shared
        );
        assert!(
            s.revoke_compartment(Parameters(RevokeParams {
                compartment_id: c.0.id,
                grantee: "user:other".into(),
            }))
            .await
            .unwrap()
            .0
            .revoked
        );
    }

    #[tokio::test]
    async fn propose_compartments_clusters_inbox_then_applies() {
        let s = server().await;
        // Memories written with no compartment land in the inbox (default).
        for content in [
            "deno run typescript module",
            "deno test typescript suite",
            "deno bundle typescript output",
            "deno fmt typescript files",
        ] {
            s.store_memory(Parameters(StoreParams {
                content: content.into(),
                network: "world".into(),
                confidence: None,
                evidence: None,
                volatile: None,
                compartment: None,
            }))
            .await
            .unwrap();
        }

        // Suggest-only: proposals returned, nothing persisted. Threshold 0.0
        // groups the whole inbox into one cluster regardless of the embedder's
        // geometry, so the wiring (list -> cluster -> view) is deterministic.
        let suggested = s
            .propose_compartments(Parameters(ProposeCompartmentsParams {
                similarity_threshold: 0.0,
                min_size: 3,
                apply: false,
            }))
            .await
            .unwrap();
        assert!(!suggested.0.proposals.is_empty(), "a region is proposed");
        assert!(
            suggested
                .0
                .proposals
                .iter()
                .all(|p| p.compartment_id.is_none()),
            "suggest-only must not persist"
        );

        // Apply: the antumbra creates a proposed compartment and moves members in.
        let applied = s
            .propose_compartments(Parameters(ProposeCompartmentsParams {
                similarity_threshold: 0.0,
                min_size: 3,
                apply: true,
            }))
            .await
            .unwrap();
        let prop = applied.0.proposals.first().expect("a proposal");
        let new_id = prop.compartment_id.clone().expect("apply persisted an id");

        // It is owned by the user and marked as a proposal awaiting curation.
        let comps = s.list_compartments().await.unwrap();
        assert!(comps
            .0
            .compartments
            .iter()
            .any(|c| c.id == new_id && c.origin == "proposed"));

        // Its members were moved out of the inbox into it (engine round-trip).
        let moved = memory::list_by_compartment(&s.store, &s.tenant, &CompartmentId::new(new_id))
            .await
            .unwrap();
        assert_eq!(moved.len(), prop.members.len());
        assert!(moved.len() >= 3, "the whole inbox region moved");
    }

    #[tokio::test]
    async fn shared_connection_isolates_tenants_under_signin() {
        // The HTTP transport's model on an embedded (single-writer) engine: ONE
        // shared connection, signed in per request. Two tenants' servers share
        // the store; serialized signin must isolate them through the real MCP
        // tools, not just at the store layer.
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        let embedder: Arc<dyn Embedder> = Arc::new(FixedEmbedder::new(EMBED_DIM));
        let (ta, ua) = (TenantId::new("ws:a"), UserId::new("user:a"));
        let (tb, ub) = (TenantId::new("ws:b"), UserId::new("user:b"));

        // Provision both identities owner-side (principal + default compartment).
        let comp_a = crate::provision_identity(&store, &ta, &ua).await.unwrap();
        let comp_b = crate::provision_identity(&store, &tb, &ub).await.unwrap();
        let server_a = McpServer::new(
            store.clone(),
            embedder.clone(),
            ta.clone(),
            ua.clone(),
            "h".into(),
            comp_a,
            None,
        );
        let server_b = McpServer::new(
            store.clone(),
            embedder.clone(),
            tb.clone(),
            ub.clone(),
            "h".into(),
            comp_b,
            None,
        );

        // Request 1: bind tenant a, store a memory via a's server.
        store.signin(&ta, &ua).await.unwrap();
        server_a
            .store_memory(Parameters(StoreParams {
                content: "alpha-only secret".into(),
                network: "world".into(),
                confidence: None,
                evidence: None,
                volatile: None,
                compartment: None,
            }))
            .await
            .unwrap();

        // Request 2: re-bind the SAME connection as tenant b; b must not see it.
        store.signin(&tb, &ub).await.unwrap();
        let b_view = server_b
            .list_memories(Parameters(ListParams { network: None }))
            .await
            .unwrap();
        assert!(
            b_view
                .0
                .memories
                .iter()
                .all(|m| !m.content.contains("alpha")),
            "tenant b must not see tenant a's memory over the shared connection"
        );

        // Request 3: a re-binds and DOES see its own memory.
        store.signin(&ta, &ua).await.unwrap();
        let a_view = server_a
            .list_memories(Parameters(ListParams { network: None }))
            .await
            .unwrap();
        assert!(
            a_view
                .0
                .memories
                .iter()
                .any(|m| m.content.contains("alpha")),
            "tenant a must see its own memory"
        );
    }

    #[tokio::test]
    async fn auto_propose_trigger_fires_once_the_inbox_grows() {
        // With the autonomous trigger armed at 4, the inbox is left alone until it
        // reaches 4 memories, at which point a write self-organizes it into an
        // Origin::Proposed compartment (and the inbox shrinks, quieting the trigger).
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        let s = McpServer::new(
            store,
            Arc::new(FixedEmbedder::new(EMBED_DIM)),
            TenantId::new("ws:test"),
            UserId::new("user:test"),
            "h".into(),
            CompartmentId::new("comp:ws:test:user:test:default"),
            None,
        )
        .with_auto_propose(4);

        let put = |n: usize| StoreParams {
            content: format!("deno typescript task {n}"),
            network: "world".into(),
            confidence: None,
            evidence: None,
            volatile: None,
            compartment: None,
        };

        // The first three writes stay below the threshold: no auto-proposal.
        for n in 0..3 {
            let out = s.store_memory(Parameters(put(n))).await.unwrap();
            assert!(
                out.0.auto_proposed.is_empty(),
                "below threshold: inbox left alone"
            );
        }
        // The fourth write reaches the threshold: the antumbra organizes the inbox.
        let out = s.store_memory(Parameters(put(3))).await.unwrap();
        assert!(
            !out.0.auto_proposed.is_empty(),
            "at threshold the antumbra auto-creates proposed compartment(s)"
        );

        // The created compartment is a proposal the user owns, awaiting curation.
        let comps = s.list_compartments().await.unwrap();
        assert!(comps.0.compartments.iter().any(|c| c.origin == "proposed"));

        // The inbox shrank below the threshold, so the next write does not re-fire.
        let again = s.store_memory(Parameters(put(99))).await.unwrap();
        assert!(
            again.0.auto_proposed.is_empty(),
            "inbox no longer over threshold"
        );
    }

    #[tokio::test]
    async fn answer_routes_then_serves_through_the_expert() {
        use antumbra_core::testing::EchoServe;
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        // A permissive router with one expert so routing covers any task.
        router::save(
            &store,
            &LearnedRouter {
                weights: vec![1.0; EMBED_DIM],
                experts: vec![RouterExpert {
                    id: ExpertId::new("expert:adder"),
                    centroid: vec![0.0; EMBED_DIM],
                }],
                temperature: 0.1,
                floor: -1.0,
            },
        )
        .await
        .unwrap();
        let s = McpServer::new(
            store,
            Arc::new(FixedEmbedder::new(EMBED_DIM)),
            TenantId::new("ws:test"),
            UserId::new("user:test"),
            "h".into(),
            CompartmentId::new("comp:test:default"),
            Some(Arc::new(EchoServe)),
        );
        let out = s
            .answer(Parameters(AnswerParams {
                task: "add two numbers".into(),
            }))
            .await
            .unwrap();
        assert!(!out.0.escalate, "the population covers the task");
        assert_eq!(out.0.expert_id.as_deref(), Some("expert:adder"));
        // EchoServe serves the prompt straight back — proves route -> serve wiring.
        assert_eq!(out.0.answer, "add two numbers");
    }

    #[tokio::test]
    async fn answer_without_a_serving_engine_reports_not_configured() {
        let s = server().await; // serve = None
        let out = s
            .answer(Parameters(AnswerParams { task: "x".into() }))
            .await
            .unwrap();
        assert!(out.0.escalate && out.0.note.is_some());
    }
}
