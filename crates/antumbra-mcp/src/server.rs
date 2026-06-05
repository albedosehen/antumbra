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

use antumbra_core::ports::Embedder;
use antumbra_core::{
    Capability, Compartment, CompartmentId, EdgeType, Grant, Memory, MemoryEdge, MemoryId,
    MemoryNetwork, Origin, TenantId, UserId,
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
    counter: Arc<AtomicU64>,
}

impl McpServer {
    pub fn new(
        store: Store,
        embedder: Arc<dyn Embedder>,
        tenant: TenantId,
        user: UserId,
        host: String,
        default_compartment: CompartmentId,
    ) -> Self {
        Self {
            store,
            embedder,
            tenant,
            user,
            host,
            default_compartment,
            counter: Arc::new(AtomicU64::new(0)),
        }
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

#[tool_router]
impl McpServer {
    /// Store a memory in this tenant's Penumbra.
    #[tool(description = "Store a memory (content + network) in your workspace's memory. Returns the new memory id.")]
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
        Ok(Json(StoredOut { id }))
    }

    /// Semantic recall over this tenant's Penumbra.
    #[tool(description = "Recall the memories most relevant to a query from your workspace's memory (semantic search).")]
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
    #[tool(description = "Reinforce a memory (raise its strength and recurrence) when it proves useful.")]
    async fn reinforce_memory(
        &self,
        Parameters(p): Parameters<IdParams>,
    ) -> Result<Json<ReinforceOut>, ErrorData> {
        match memory::reinforce(&self.store, &self.tenant, &MemoryId::new(p.memory_id), Utc::now())
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
        memory::delete(&self.store, &self.tenant, &MemoryId::new(p.memory_id))
            .await
            .map_err(err)?;
        Ok(Json(ForgetOut { forgotten: true }))
    }

    /// List this tenant's memories (optionally one network).
    #[tool(description = "List your workspace's memories, optionally filtered to one network (world/bank/opinion).")]
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
    #[tool(description = "Relate two memories with a typed edge: references/supersedes/contradicts/follows/caused.")]
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
    #[tool(description = "Get the memories connected from a memory (optionally filtered to one edge type).")]
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
    #[tool(description = "Route a task across the shared population AND your private experts: which expert(s) cover it, ranked, or escalate if none.")]
    async fn route(&self, Parameters(p): Parameters<RouteParams>) -> Result<Json<RouteOut>, ErrorData> {
        let v = self.embedder.embed(&p.task).await.map_err(err)?;
        let k = p.top_k.unwrap_or(3) as usize;
        let mut routes: Vec<RouteHit> = Vec::new();

        // Shared experts via the learned router (pure-arithmetic gate).
        if let Some(router) = router::load(&self.store).await.map_err(err)? {
            if router.covers(&v) {
                for (id, probability) in router.route(&v) {
                    routes.push(RouteHit {
                        expert_id: id.as_str().to_string(),
                        probability,
                        private: false,
                    });
                }
            }
        }

        // The user's own private experts (not in the shared router) — matched by
        // centroid. The session's expert ACL already scopes the list to shared +
        // own-private; the explicit owner filter keeps only the private ones.
        for e in expert::list(&self.store).await.map_err(err)? {
            if e.owner.as_ref() == Some(&self.user) {
                if let Some(sim) = e.capability_similarity(&v) {
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
        let covered = !routes.is_empty();
        Ok(Json(RouteOut {
            covered,
            escalate: !covered,
            routes,
        }))
    }

    /// Create a private compartment (latent-space) owned by you.
    #[tool(description = "Create a new compartment (a private latent-space of memory) you own. Store into it via store_memory's compartment arg. Returns its id.")]
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
    #[tool(description = "Share one of your compartments with another user: reference (they can recall it) or link (they can also connect to it).")]
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
    #[tool(description = "Revoke a user's access to one of your compartments (takes effect immediately).")]
    async fn revoke_compartment(
        &self,
        Parameters(p): Parameters<RevokeParams>,
    ) -> Result<Json<RevokeOut>, ErrorData> {
        compartment::revoke(
            &self.store,
            &self.tenant,
            &CompartmentId::new(p.compartment_id),
            &UserId::new(p.grantee),
        )
        .await
        .map_err(err)?;
        Ok(Json(RevokeOut { revoked: true }))
    }
}

#[tool_handler]
impl ServerHandler for McpServer {}

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
        let a = s.store_memory(Parameters(store("a deno project"))).await.unwrap().0.id;
        let b = s.store_memory(Parameters(store("use deno install"))).await.unwrap().0.id;

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
            r.0.routes.iter().any(|h| h.private && h.expert_id == "expert:mine"),
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
}
