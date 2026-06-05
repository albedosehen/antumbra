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
use antumbra_core::{Memory, MemoryId, MemoryNetwork, TenantId};
use antumbra_store::repo::memory;
use antumbra_store::Store;

/// One tenant's MCP session over its Penumbra. `#[tool_handler]` resolves the
/// tools via `Self::tool_router()`, so no router field is stored.
#[derive(Clone)]
pub struct McpServer {
    store: Store,
    embedder: Arc<dyn Embedder>,
    tenant: TenantId,
    counter: Arc<AtomicU64>,
}

impl McpServer {
    pub fn new(store: Store, embedder: Arc<dyn Embedder>, tenant: TenantId) -> Self {
        Self {
            store,
            embedder,
            tenant,
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

/// A process-unique, monotonic memory id (timestamp + counter; no extra deps).
fn next_id(counter: &AtomicU64) -> String {
    let n = counter.fetch_add(1, Ordering::Relaxed);
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("memory:{t:x}-{n:x}")
}

fn default_network() -> String {
    "world".into()
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

#[tool_router]
impl McpServer {
    /// Store a memory in this tenant's Penumbra.
    #[tool(description = "Store a memory (content + network) in your workspace's memory. Returns the new memory id.")]
    async fn store_memory(
        &self,
        Parameters(p): Parameters<StoreParams>,
    ) -> Result<Json<StoredOut>, ErrorData> {
        let embedding = self.embedder.embed(&p.content).await.map_err(err)?;
        let id = next_id(&self.counter);
        let mut m = Memory::new(
            id.clone(),
            self.tenant.clone(),
            parse_network(&p.network),
            p.content,
            p.confidence.unwrap_or(0.6),
            Utc::now(),
        )
        .with_embedding(embedding);
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
}

#[tool_handler]
impl ServerHandler for McpServer {}

#[cfg(test)]
mod tests {
    use super::*;
    use antumbra_core::testing::FixedEmbedder;
    use antumbra_store::EMBED_DIM;

    async fn server() -> McpServer {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        McpServer::new(
            store,
            Arc::new(FixedEmbedder::new(EMBED_DIM)),
            TenantId::new("ws:test"),
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
}
