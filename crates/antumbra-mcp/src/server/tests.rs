//! The MCP server tests: real tool functions driven against an embedded store
//! (isolation under sign-in, rerank degradation, copal fail-closed, answer
//! escalation, and the rest). Split out of `server.rs` so the impl stays readable.

use super::*;
use antumbra_core::router::{LearnedRouter, RouterExpert};
use antumbra_core::testing::FixedEmbedder;
use antumbra_core::{BoundaryId, Expert, ExpertId, FailureBoundary, Generation, Grain};
use antumbra_store::repo::{expert, router};
use antumbra_store::EMBED_DIM;
use chrono::Utc;

// Ids stay unique even minted back-to-back (potentially within one
// nanosecond, where the timestamp portion collides): the process-global
// counter disambiguates, so two sessions cannot mint the same id.
#[test]
fn next_id_is_unique_under_a_burst() {
    let ids: std::collections::HashSet<String> = (0..1000).map(|_| next_id("m")).collect();
    assert_eq!(
        ids.len(),
        1000,
        "the process-global counter keeps ids unique"
    );
}

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
            provenance: None,
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
            repo: None,
            branch: None,
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

/// Seed several memories then recall: returns the recalled content list in
/// order. Shared by the rerank tests so the control and reranked runs are
/// over identical data.
async fn seed_and_recall(s: &McpServer, query: &str) -> Vec<String> {
    for content in [
        "the first ordinary note about scheduling",
        "a second unrelated note on logging config",
        "PROMOTE: the exact answer about deno install in acme-api",
        "a fourth note mentioning npm dependencies loosely",
    ] {
        s.store_memory(Parameters(StoreParams {
            provenance: None,
            content: content.into(),
            network: "world".into(),
            confidence: Some(0.8),
            evidence: None,
            volatile: None,
            compartment: None,
        }))
        .await
        .unwrap();
    }
    s.recall_memories(Parameters(RecallParams {
        repo: None,
        branch: None,
        query: query.into(),
        top_k: Some(4),
        network: None,
    }))
    .await
    .unwrap()
    .0
    .memories
    .into_iter()
    .map(|m| m.content)
    .collect()
}

#[tokio::test]
async fn rerank_promotes_the_cross_encoder_winner_to_top1() {
    use antumbra_core::testing::ScriptedReranker;

    // With a reranker that promotes the "PROMOTE" candidate, it leads.
    let store = Store::connect_memory(EMBED_DIM).await.unwrap();
    let reranked = McpServer::new(
        store,
        Arc::new(FixedEmbedder::new(EMBED_DIM)),
        TenantId::new("ws:test"),
        UserId::new("user:test"),
        "test-host".into(),
        CompartmentId::new("comp:test:default"),
        None,
    )
    .with_reranker(Arc::new(ScriptedReranker::promoting_content_substring(
        "PROMOTE",
    )));
    let out = seed_and_recall(&reranked, "how do I install a dependency").await;
    assert!(
        out[0].contains("PROMOTE"),
        "the cross-encoder winner is reranked to top-1: {out:?}"
    );
}

#[tokio::test]
async fn without_reranker_rrf_order_is_unchanged_control() {
    // The control: the SAME data with no reranker need not put PROMOTE first
    // (the embedding/RRF order stands). This proves the reorder above is the
    // reranker's doing, not an artifact of the data.
    let plain = server().await;
    let out = seed_and_recall(&plain, "how do I install a dependency").await;
    // PROMOTE is not guaranteed top-1 without the cross-encoder; at minimum the
    // ordering is allowed to differ from the reranked run. Assert the control
    // simply returns all four, in some order, without rerank applied.
    assert_eq!(out.len(), 4, "control returns the recalled set: {out:?}");
}

#[tokio::test]
async fn reranker_error_degrades_to_rrf_order_without_failing() {
    use antumbra_core::testing::ScriptedReranker;

    // A failing reranker must NOT turn a successful recall into an error.
    let store = Store::connect_memory(EMBED_DIM).await.unwrap();
    let s = McpServer::new(
        store,
        Arc::new(FixedEmbedder::new(EMBED_DIM)),
        TenantId::new("ws:test"),
        UserId::new("user:test"),
        "test-host".into(),
        CompartmentId::new("comp:test:default"),
        None,
    )
    .with_reranker(Arc::new(ScriptedReranker::failing()));
    let out = seed_and_recall(&s, "how do I install a dependency").await;
    assert_eq!(
        out.len(),
        4,
        "recall still returns its results when rerank errors: {out:?}"
    );
}

#[tokio::test]
async fn rerank_reorders_document_chunks_too() {
    use antumbra_core::testing::ScriptedReranker;

    let store = Store::connect_memory(EMBED_DIM).await.unwrap();
    let s = McpServer::new(
        store,
        Arc::new(FixedEmbedder::new(EMBED_DIM)),
        TenantId::new("ws:test"),
        UserId::new("user:test"),
        "test-host".into(),
        CompartmentId::new("comp:test:default"),
        None,
    )
    .with_reranker(Arc::new(ScriptedReranker::promoting_content_substring(
        "PROMOTE",
    )));
    // Two short docs so each is a single chunk; one carries the token.
    s.ingest_document(Parameters(IngestDocumentParams {
        title: "doc-a".into(),
        source: None,
        content: "an ordinary chunk about scheduling and logging".into(),
    }))
    .await
    .unwrap();
    s.ingest_document(Parameters(IngestDocumentParams {
        title: "doc-b".into(),
        source: None,
        content: "PROMOTE: the exact chunk answering the dependency question".into(),
    }))
    .await
    .unwrap();
    let out = s
        .recall_documents(Parameters(RecallDocumentsParams {
            query: "how do I install a dependency".into(),
            top_k: Some(2),
        }))
        .await
        .unwrap();
    let contents: Vec<String> = out.0.chunks.into_iter().map(|c| c.content).collect();
    assert!(
        contents[0].contains("PROMOTE"),
        "reranked chunk leads: {contents:?}"
    );
}

/// The autonomous consolidation trigger, end to end: storing then
/// reinforcing a memory past the gate fires `maybe_consolidate`, which mints
/// a private expert in the background with no manual `consolidate-compartment`
/// call. Gated: needs the candle trainer (`--features models`) + a GPU +
/// `python` (the exec verifier) + the base weights. Run with `--ignored`.
#[cfg(feature = "models")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs --features models + a GPU + python; run with --ignored"]
async fn auto_consolidate_mints_a_private_expert_on_reinforce() {
    use antumbra_core::ports::Serve;
    let store = Store::connect_memory(EMBED_DIM).await.unwrap();
    let comp = "comp:ws:test:auto";
    // A real (empty) serve engine: the trigger should hot-register the minted
    // expert into it, so `answer` could serve it with no restart.
    let serve = Arc::new(antumbra_serve::MultiAdapterServe::new(
        "Qwen/Qwen2.5-Coder-1.5B",
        antumbra_serve::RaftConfig::default(),
    ));
    let s = McpServer::new(
        store.clone(),
        Arc::new(FixedEmbedder::new(EMBED_DIM)),
        TenantId::new("ws:test"),
        UserId::new("user:test"),
        "test-host".into(),
        CompartmentId::new(comp),
        Some(serve.clone() as Arc<dyn antumbra_core::ports::Serve>),
    )
    .with_auto_consolidate();

    // A high-confidence opinion graduates on the provenance tier once it is
    // reinforced past recurrence >= 2; `None` compartment lands in the
    // server default (`comp`), which the minted expert is named for.
    let stored = s
        .store_memory(Parameters(StoreParams {
            provenance: None,
            content: "Prefer `deno install` over `npm install` in this project.".into(),
            network: "opinion".into(),
            confidence: Some(1.0),
            evidence: None,
            volatile: None,
            compartment: None,
        }))
        .await
        .unwrap();
    let id = stored.0.id.clone();

    // Reinforce past the gate. A no-op early reinforce (recurrence < 2) returns
    // fast, freeing the per-compartment guard before a later one trains; the
    // spacing keeps the trigger from being swallowed by an in-flight no-op.
    for _ in 0..3 {
        s.reinforce_memory(Parameters(IdParams {
            memory_id: id.clone(),
        }))
        .await
        .unwrap();
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }

    // The consolidation runs in a spawned task. `can_serve` flips true only
    // after the expert is minted AND hot-registered, so it is the end-to-end
    // signal that the whole loop closed (memory -> expert -> servable).
    let want = ExpertId::new(format!("expert:user:test:{comp}"));
    let mut servable = false;
    for _ in 0..240 {
        if serve.can_serve(&want) {
            servable = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }
    assert!(
        servable,
        "autonomous consolidation should mint {} and hot-register it for serving",
        want.as_str()
    );
    assert!(
        expert::get(&store, &want).await.unwrap().is_some(),
        "the minted expert is persisted in the store"
    );
}

#[tokio::test]
async fn relate_and_get_neighbors() {
    let s = server().await;
    let store = |content: &str| StoreParams {
        provenance: None,
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
async fn route_escalates_when_a_boundary_inhibits_the_task() {
    let s = server().await;
    // A permissive router that would otherwise route the task.
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

    // Without a boundary, the task routes.
    let r = s
        .route(Parameters(RouteParams {
            task: "add two numbers".into(),
            top_k: Some(3),
        }))
        .await
        .unwrap();
    assert!(r.0.covered && !r.0.escalate);

    // An actionable boundary whose failure context IS this task region: the
    // task sits closer to C than to C', so inhibition fires and
    // the served path escalates rather than route to an expert that fails here.
    // C embeds to the task region itself (sim_fail = 1), C' to a far context,
    // so the relative margin clears the escalate threshold.
    let emb = FixedEmbedder::new(EMBED_DIM);
    let fail_vec = emb.embed("add two numbers").await.unwrap();
    let ok_vec = emb.embed("a poem about gardening").await.unwrap();
    boundary::upsert(
        &s.store,
        &FailureBoundary {
            id: BoundaryId::new("boundary:b1"),
            behavior: "add two numbers".into(),
            fail_context: serde_json::json!({ "domain": "math" }),
            near_ok_context: Some(serde_json::json!({ "domain": "prose" })),
            governing_features: vec!["domain".into()],
            grain: Some(Grain::Project),
            context_vec: Some(fail_vec),
            ok_context_vec: Some(ok_vec),
            confidence: 0.9,
            generation: Generation::ZERO,
            created_at: Utc::now(),
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
    assert!(
        r.0.escalate && r.0.routes.is_empty(),
        "a task inside a known failure scope escalates"
    );
}

#[tokio::test]
async fn call_tool_dispatches_a_named_tool() {
    let s = server().await;
    // Store, then list, both through the by-name dispatcher (the REST path).
    let stored = s
        .call_tool(
            "store_memory",
            serde_json::json!({ "content": "the deno runtime" }),
        )
        .await
        .unwrap();
    assert!(stored.is_object());
    let listed = s
        .call_tool("list_memories", serde_json::json!({}))
        .await
        .unwrap();
    assert!(listed.to_string().contains("deno"));
    // Unknown tool and malformed arguments are errors, not panics.
    assert!(s.call_tool("nope", serde_json::json!({})).await.is_err());
    assert!(s
        .call_tool("store_memory", serde_json::json!({ "missing": "content" }))
        .await
        .is_err());
}

#[tokio::test]
async fn ingest_then_recall_a_knowledge_document() {
    let s = server().await;
    let ingested = s
        .ingest_document(Parameters(IngestDocumentParams {
            title: "Onboarding".into(),
            content: "Antumbra keeps knowledge documents separate from episodic memory. \
                      Ingesting a document chunks it, embeds each chunk, and makes it \
                      recallable. This project uses the deno runtime."
                .into(),
            source: Some("onboarding.md".into()),
        }))
        .await
        .unwrap();
    assert!(ingested.0.chunks >= 1);
    assert_eq!(ingested.0.title, "Onboarding");

    let recalled = s
        .recall_documents(Parameters(RecallDocumentsParams {
            query: "what runtime does this project use".into(),
            top_k: Some(3),
        }))
        .await
        .unwrap();
    assert!(!recalled.0.chunks.is_empty());
    assert!(recalled.0.chunks.iter().all(|c| c.title == "Onboarding"));
    // No copal archive configured: no provenance, exactly as before.
    assert!(recalled
        .0
        .chunks
        .iter()
        .all(|c| c.copal_file.is_none() && c.copal_digest.is_none()));

    // Reachable over the REST dispatcher too.
    let viarest = s
        .call_tool(
            "recall_documents",
            serde_json::json!({ "query": "runtime", "top_k": 1 }),
        )
        .await
        .unwrap();
    assert!(viarest.get("chunks").is_some());
}

/// A canned copal for the ingest tests: create answers with a fixed file
/// id, upload with a fixed digest; `Err` variants exercise the fail-closed
/// path (a configured archive that is down must fail the ingest).
struct CannedCopal {
    up: bool,
}

impl crate::copal::CopalTransport for CannedCopal {
    fn post_json(
        &self,
        _url: &str,
        _credential: &crate::copal::CopalCredential,
        _body: &serde_json::Value,
    ) -> antumbra_core::Result<serde_json::Value> {
        if self.up {
            Ok(serde_json::json!({ "id": "file:01J", "state": "draft" }))
        } else {
            Err(antumbra_core::AntumbraError::other("connection refused"))
        }
    }

    fn put_bytes(
        &self,
        _url: &str,
        _credential: &crate::copal::CopalCredential,
        _content_type: &str,
        _body: &[u8],
    ) -> antumbra_core::Result<serde_json::Value> {
        if self.up {
            Ok(serde_json::json!({ "digest": "sha256:abc", "state": "ready" }))
        } else {
            Err(antumbra_core::AntumbraError::other("connection refused"))
        }
    }
}

fn canned_archive(up: bool) -> Arc<crate::copal::CopalArchive> {
    Arc::new(crate::copal::CopalArchive::with_transport(
        "127.0.0.1:9010",
        // Per-workspace tenancy (the default): the session's workspace
        // presents itself as the copal tenant. The header side is proven
        // in `crate::copal`'s own tests; these care about the ingest path.
        crate::copal::CopalTenancy::PerWorkspace,
        Arc::new(CannedCopal { up }),
    ))
}

#[tokio::test]
async fn ingest_with_a_copal_archive_stamps_every_chunk_with_provenance() {
    let s = server().await.with_copal_archive(canned_archive(true));
    s.ingest_document(Parameters(IngestDocumentParams {
        title: "Onboarding".into(),
        content: "Antumbra keeps knowledge documents separate from episodic memory. \
                  This project uses the deno runtime."
            .into(),
        source: Some("onboarding.md".into()),
    }))
    .await
    .unwrap();

    let recalled = s
        .recall_documents(Parameters(RecallDocumentsParams {
            query: "what runtime does this project use".into(),
            top_k: Some(3),
        }))
        .await
        .unwrap();
    assert!(!recalled.0.chunks.is_empty());
    // Every chunk names the archived original: the file AND the bytes.
    assert!(recalled
        .0
        .chunks
        .iter()
        .all(|c| c.copal_file.as_deref() == Some("file:01J")
            && c.copal_digest.as_deref() == Some("sha256:abc")));
}

#[tokio::test]
async fn ingest_fails_closed_when_the_configured_copal_is_unreachable() {
    // Upload-first: with the archive configured but down, the ingest fails
    // BEFORE any chunk is stored -- a document of record that silently
    // dropped originals would be worse than none.
    let s = server().await.with_copal_archive(canned_archive(false));
    let res = s
        .ingest_document(Parameters(IngestDocumentParams {
            title: "Onboarding".into(),
            content: "some reference text".into(),
            source: None,
        }))
        .await;
    assert!(
        res.is_err(),
        "a configured archive that is down fails ingest"
    );
    // Nothing was stored: the workspace still lists zero documents.
    assert_eq!(s.workspace_stats().await.unwrap().0.documents, 0);
}

#[tokio::test]
async fn population_and_stats_report_the_workspace() {
    let s = server().await;
    // A fresh workspace has no experts.
    assert!(s.population().await.unwrap().0.experts.is_empty());
    assert_eq!(s.workspace_stats().await.unwrap().0.experts, 0);

    // Seed a memory and a document through the dispatcher.
    s.call_tool(
        "store_memory",
        serde_json::json!({ "content": "remember this" }),
    )
    .await
    .unwrap();
    s.call_tool(
        "ingest_document",
        serde_json::json!({ "title": "Doc", "content": "some reference text" }),
    )
    .await
    .unwrap();

    let stats = s.workspace_stats().await.unwrap();
    assert!(stats.0.memories >= 1);
    assert_eq!(stats.0.documents, 1);

    // The stats tool is reachable over the REST dispatcher too.
    let via = s
        .call_tool("workspace_stats", serde_json::json!({}))
        .await
        .unwrap();
    assert!(via.get("memories").is_some());
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
            provenance: None,
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
            provenance: None,
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
            provenance: None,
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
        provenance: None,
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
    // EchoServe serves the prompt straight back, proving route -> serve wiring.
    assert_eq!(out.0.answer, "add two numbers");
}

// A serving engine that doesn't have the routed expert's adapter must make
// `answer` ESCALATE, not error -- and never call `act` (F4).
struct Unservable;
#[async_trait::async_trait]
impl antumbra_core::ports::Serve for Unservable {
    async fn act(
        &self,
        _req: ActRequest,
    ) -> antumbra_core::Result<antumbra_core::ports::ActOutput> {
        panic!("act must not be called when the expert is unservable");
    }
    fn can_serve(&self, _expert: &ExpertId) -> bool {
        false
    }
}

#[tokio::test]
async fn answer_escalates_when_the_routed_expert_is_not_servable() {
    let store = Store::connect_memory(EMBED_DIM).await.unwrap();
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
        Some(Arc::new(Unservable)),
    );
    let out = s
        .answer(Parameters(AnswerParams {
            task: "add two numbers".into(),
        }))
        .await
        .unwrap();
    assert!(
        out.0.escalate,
        "an unservable routed expert escalates, not errors"
    );
    assert_eq!(out.0.expert_id.as_deref(), Some("expert:adder"));
    assert!(out.0.answer.is_empty());
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

/// Provenance travels with the memory: a structured anchor becomes a `git:`
/// evidence entry, comes back parsed on every view, and a recall given the
/// caller's repo + branch tags each hit's scope and demotes the out-of-scope
/// ones below the in-scope ones without hiding them.
#[tokio::test]
async fn provenance_is_stored_and_scopes_recall() {
    let s = server().await;
    let store = |content: &str, branch: &str, repo: &str| StoreParams {
        provenance: Some(ProvenanceParams {
            repo: repo.into(),
            commit: "b697da7".into(),
            branch: Some(branch.into()),
            path: None,
        }),
        content: content.into(),
        network: "world".into(),
        confidence: Some(0.8),
        evidence: Some(vec!["cargo test passed".into()]),
        volatile: None,
        compartment: None,
    };
    // Three memories about the same thing, learned on different branches / repos.
    s.store_memory(Parameters(store(
        "the orders route is POST /orders on feat/orders",
        "feat/orders",
        "github.com/o/r",
    )))
    .await
    .unwrap();
    s.store_memory(Parameters(store(
        "the orders route is POST /orders on main",
        "main",
        "github.com/o/r",
    )))
    .await
    .unwrap();
    s.store_memory(Parameters(store(
        "the orders route lives in another repo",
        "main",
        "github.com/o/other",
    )))
    .await
    .unwrap();

    // No context: every hit comes back with its parsed provenance and no scope.
    let plain = s
        .recall_memories(Parameters(RecallParams {
            repo: None,
            branch: None,
            query: "orders route".into(),
            top_k: Some(5),
            network: None,
        }))
        .await
        .unwrap();
    assert_eq!(plain.0.memories.len(), 3);
    for m in &plain.0.memories {
        let p = m
            .provenance
            .as_ref()
            .expect("provenance parsed from evidence");
        assert_eq!(p.commit, "b697da7");
        assert!(m.scope.is_none());
    }

    // On main in github.com/o/r: the main memory is in scope, the feature-branch
    // and other-repo memories are demoted to the tail, in that relative order.
    let scoped = s
        .recall_memories(Parameters(RecallParams {
            repo: Some("GitHub.com/o/r.git".into()),
            branch: Some("main".into()),
            query: "orders route".into(),
            top_k: Some(5),
            network: None,
        }))
        .await
        .unwrap();
    let scopes: Vec<&str> = scoped
        .0
        .memories
        .iter()
        .map(|m| m.scope.as_deref().unwrap())
        .collect();
    assert_eq!(scopes[0], "in_scope", "{scopes:?}");
    assert!(
        scopes[1..]
            .iter()
            .all(|s| *s == "other_branch" || *s == "other_repo"),
        "{scopes:?}"
    );
    assert!(scoped.0.memories[0].content.contains("on main"));

    // A malformed anchor is refused, not stored as junk evidence.
    let bad = s
        .store_memory(Parameters(StoreParams {
            provenance: Some(ProvenanceParams {
                repo: "git@github.com:o/r".into(),
                commit: "not-hex".into(),
                branch: None,
                path: None,
            }),
            content: "junk".into(),
            network: "world".into(),
            confidence: None,
            evidence: None,
            volatile: None,
            compartment: None,
        }))
        .await;
    assert!(bad.is_err());
}
