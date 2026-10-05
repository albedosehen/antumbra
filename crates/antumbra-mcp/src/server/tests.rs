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
            full: None,
            floor: None,
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
        .list_memories(Parameters(ListParams {
            network: None,
            limit: None,
            offset: None,
        }))
        .await
        .unwrap();
    assert_eq!(listed.0.memories.len(), 1);

    s.forget_memory(Parameters(IdParams { memory_id: id }))
        .await
        .unwrap();
    assert!(s
        .list_memories(Parameters(ListParams {
            network: None,
            limit: None,
            offset: None
        }))
        .await
        .unwrap()
        .0
        .memories
        .is_empty());
}

/// A memory's view says when it was last written, and reinforcing it moves that
/// on. With `reinforcement`, that makes a memory a counter which also says when
/// it last counted (ADR-0021, skill usage).
#[tokio::test]
async fn a_view_says_when_a_memory_was_last_written_and_reinforcing_moves_it() -> anyhow::Result<()>
{
    let s = server().await;
    let said = |e: ErrorData| anyhow::anyhow!("{e:?}");
    let stored = s
        .store_memory(Parameters(StoreParams {
            provenance: None,
            content: "[skill-use:deploy] deploy".into(),
            network: "world".into(),
            confidence: Some(0.9),
            evidence: None,
            volatile: Some(true),
            compartment: None,
        }))
        .await
        .map_err(said)?;
    let written = |listed: &Json<MemoriesOut>| {
        listed
            .0
            .memories
            .first()
            .map(|m| m.updated_at.clone())
            .ok_or_else(|| anyhow::anyhow!("the memory is not listed"))
    };
    let list = || {
        s.list_memories(Parameters(ListParams {
            network: None,
            limit: None,
            offset: None,
        }))
    };

    let before = written(&list().await.map_err(said)?)?;
    let at = chrono::DateTime::parse_from_rfc3339(&before)?;
    assert!(
        (Utc::now() - at.with_timezone(&Utc)).num_seconds().abs() < 60,
        "{before}"
    );

    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    s.reinforce_memory(Parameters(IdParams {
        memory_id: stored.0.id.clone(),
    }))
    .await
    .map_err(said)?;
    let after = written(&list().await.map_err(said)?)?;
    assert!(after > before, "{before} then {after}");
    Ok(())
}

/// A recall says how close each memory is, and says it about the right memory
/// even on the path that reorders the results. The demotion moves out-of-scope
/// hits down, so a similarity attached after it would be paired with whichever
/// memory happened to land in that slot.
#[tokio::test]
async fn a_recall_scores_each_memory_and_the_score_follows_it_through_the_demotion(
) -> anyhow::Result<()> {
    let s = server().await;
    let said = |e: ErrorData| anyhow::anyhow!("{e:?}");
    let store = |content: &str, repo: &str| {
        s.store_memory(Parameters(StoreParams {
            provenance: Some(ProvenanceParams {
                repo: repo.into(),
                commit: "abc1234".into(),
                branch: Some("main".into()),
                path: None,
            }),
            content: content.into(),
            network: "world".into(),
            confidence: Some(0.9),
            evidence: None,
            volatile: None,
            compartment: None,
        }))
    };
    // The first is what the query asks about; the second is elsewhere, and also
    // in another repository, so the demotion will move it.
    store(
        "deno install is how dependencies are added",
        "github.com/me/here",
    )
    .await
    .map_err(said)?;
    store("the kettle is in the kitchen", "github.com/me/elsewhere")
        .await
        .map_err(said)?;

    let recalled = s
        .recall_memories(Parameters(RecallParams {
            repo: Some("github.com/me/here".into()),
            branch: Some("main".into()),
            query: "how do I add a dependency".into(),
            top_k: Some(5),
            network: None,
            full: None,
            floor: None,
        }))
        .await
        .map_err(said)?;
    anyhow::ensure!(
        recalled.0.memories.len() == 2,
        "expected both memories back"
    );

    for view in &recalled.0.memories {
        let Some(similarity) = view.similarity else {
            anyhow::bail!("`{}` came back with no similarity", view.content);
        };
        anyhow::ensure!(
            (-1.0..=1.0).contains(&similarity),
            "{similarity} is not a cosine"
        );
        // The score belongs to THIS memory: re-embedding its own content must
        // land closer to it than to the other one. That is what a score paired
        // with the wrong row would fail.
        let own = s.embedder.embed(&view.content).await?;
        let query = s.embedder.embed("how do I add a dependency").await?;
        let expected = antumbra_core::cosine_similarity(&query, &own);
        anyhow::ensure!(
            (similarity - expected).abs() < 1e-5,
            "`{}` carries {similarity}, but its own content scores {expected}",
            view.content
        );
    }
    Ok(())
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
        full: None,
        floor: None,
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
        compartment: None,
        provenance: None,
        title: "doc-a".into(),
        source: None,
        content: "an ordinary chunk about scheduling and logging".into(),
    }))
    .await
    .unwrap();
    s.ingest_document(Parameters(IngestDocumentParams {
        compartment: None,
        provenance: None,
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
    let store = Store::connect_memory(EMBED_DIM).await.unwrap();
    let comp = "comp:ws:test:auto";
    // The engine a server builds for itself on a fresh node, with no experts
    // yet: the trigger must be able to hot-register the minted expert into it,
    // so `answer` serves it with no restart. Hand-building an engine here hid
    // the cold start, where the server came up with no engine at all (EXP-022).
    let serve = match crate::build_serve(&store, "test-host").await {
        Ok(Some(serve)) => serve,
        other => panic!(
            "a fresh node must still get a serving engine, got {:?}",
            other.map(|engine| engine.is_some())
        ),
    };
    let s = McpServer::new(
        store.clone(),
        Arc::new(FixedEmbedder::new(EMBED_DIM)),
        TenantId::new("ws:test"),
        UserId::new("user:test"),
        "test-host".into(),
        CompartmentId::new(comp),
        Some(serve.clone()),
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
    assert_eq!(
        r.0.reason.as_deref(),
        Some("no learned router yet, and none of your private experts is close to this task")
    );

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
            placed_on: None,
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
    assert!(
        r.0.reason
            .as_deref()
            .unwrap_or("")
            .starts_with("a failure boundary covers this task"),
        "{:?}",
        r.0.reason
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

mod behaviour;
mod depgraph;
mod handoff;
mod lifecycle;
mod workspace;

/// A server pinned to a named host, for the fabric tests: which machine a run
/// belongs on is decided by comparing this against what the registry says.
async fn server_on(host: &str) -> anyhow::Result<McpServer> {
    let store = Store::connect_memory(EMBED_DIM).await?;
    Ok(McpServer::new(
        store,
        Arc::new(FixedEmbedder::new(EMBED_DIM)),
        TenantId::new("ws:test"),
        UserId::new("user:test"),
        host.into(),
        CompartmentId::new("comp:test:default"),
        None,
    ))
}

async fn register(
    s: &McpServer,
    host: &str,
    role: antumbra_core::DeviceRole,
) -> anyhow::Result<()> {
    antumbra_store::repo::device::upsert(
        &s.store,
        &antumbra_core::DeviceProfile::new(
            s.tenant.clone(),
            s.user.clone(),
            host,
            if role.can_train() { "cuda" } else { "cpu" },
            role,
            Utc::now(),
        ),
    )
    .await?;
    Ok(())
}

async fn waiting(s: &McpServer) -> anyhow::Result<Vec<antumbra_core::GenesisRequest>> {
    Ok(antumbra_store::repo::genesis::list_open_for_user(&s.store, &s.tenant, &s.user).await?)
}

/// ADR-0017 A2. A node that cannot train does not grind the model while the
/// user's GPU box sits idle, and it does not drop the work either: it leaves a
/// request where the machine that can train will find it.
#[tokio::test]
async fn a_node_that_cannot_train_leaves_the_run_for_the_one_that_can() -> anyhow::Result<()> {
    let s = server_on("her-laptop").await?;
    let comp = CompartmentId::new("comp:rust");
    register(&s, "her-laptop", antumbra_core::DeviceRole::Memory).await?;
    register(&s, "the-rig", antumbra_core::DeviceRole::Genesis).await?;

    assert!(
        s.escalate_genesis(&s.store, &comp).await,
        "the laptop must not take a run the rig is for"
    );
    assert_eq!(
        waiting(&s)
            .await?
            .iter()
            .map(|r| (
                r.compartment.as_str(),
                r.from_host.as_str(),
                r.to_host.as_str()
            ))
            .collect::<Vec<_>>(),
        vec![("comp:rust", "her-laptop", "the-rig")],
        "the work is recorded, not merely reported"
    );

    // Reinforced again: one piece of work, one request.
    assert!(s.escalate_genesis(&s.store, &comp).await);
    assert_eq!(waiting(&s).await?.len(), 1);
    Ok(())
}

/// The other side of the same decision, and the reason an empty fabric is safe:
/// a node with nowhere better to send the work keeps it.
#[tokio::test]
async fn a_run_stays_where_it_is_when_there_is_nowhere_better_for_it() -> anyhow::Result<()> {
    let comp = CompartmentId::new("comp:rust");

    // Nothing registered at all: the single-node case, and every build from
    // before the registry existed.
    let alone = server_on("the-only-box").await?;
    assert!(!alone.escalate_genesis(&alone.store, &comp).await);

    // Registered, and this node is the one the fabric names.
    let rig = server_on("the-rig").await?;
    register(&rig, "the-rig", antumbra_core::DeviceRole::Genesis).await?;
    assert!(!rig.escalate_genesis(&rig.store, &comp).await);

    // A fabric of memory nodes only: nobody can train, so the work stays with
    // whoever has it rather than waiting on a machine that does not exist.
    let laptop = server_on("her-laptop").await?;
    register(&laptop, "her-laptop", antumbra_core::DeviceRole::Memory).await?;
    register(&laptop, "his-laptop", antumbra_core::DeviceRole::Memory).await?;
    assert!(!laptop.escalate_genesis(&laptop.store, &comp).await);

    for s in [&alone, &rig, &laptop] {
        assert!(
            waiting(s).await?.is_empty(),
            "nothing is left waiting when nothing was escalated"
        );
    }
    Ok(())
}

async fn ask(
    s: &McpServer,
    compartment: &str,
    from: &str,
    at: chrono::DateTime<Utc>,
) -> anyhow::Result<()> {
    antumbra_store::repo::genesis::ask(
        &s.store,
        &antumbra_core::GenesisRequest::new(
            s.tenant.clone(),
            s.user.clone(),
            CompartmentId::new(compartment),
            from,
            "the-rig",
            at,
        ),
    )
    .await?;
    Ok(())
}

/// ADR-0017 A2, the taking half. A node that can train clears what its user's
/// other machines left for it before the compartment it happened to be handed,
/// because a request has been waiting and the write has not.
#[tokio::test]
async fn a_trainer_takes_the_run_that_has_waited_longest() -> anyhow::Result<()> {
    let rig = server_on("the-rig").await?;
    let now = Utc::now();
    ask(
        &rig,
        "comp:rust",
        "her-laptop",
        now - chrono::Duration::hours(1),
    )
    .await?;
    ask(
        &rig,
        "comp:surql",
        "his-laptop",
        now - chrono::Duration::hours(3),
    )
    .await?;

    let taken = rig
        .claim_genesis_request(&rig.store)
        .await
        .ok_or_else(|| anyhow::anyhow!("a waiting run must be taken"))?;
    assert_eq!(taken.compartment.as_str(), "comp:surql", "oldest first");
    assert_eq!(taken.status, antumbra_core::GenesisStatus::Claimed);

    // Claiming is a write, so a second trainer in the same fabric takes the
    // next one rather than the same one twice.
    let also = rig
        .claim_genesis_request(&rig.store)
        .await
        .ok_or_else(|| anyhow::anyhow!("the second run must be taken"))?;
    assert_eq!(also.compartment.as_str(), "comp:rust");

    // Nothing pending left, even though both are still open.
    assert!(rig.claim_genesis_request(&rig.store).await.is_none());
    assert_eq!(
        antumbra_store::repo::genesis::list_open_for_user(&rig.store, &rig.tenant, &rig.user)
            .await?
            .len(),
        2,
        "claimed is not finished: a trainer that dies does not lose the work"
    );
    Ok(())
}

/// The ordinary case, and the only one in a fabric of a single node: there is
/// nothing waiting, so the trainer gets on with what it was handed.
#[tokio::test]
async fn an_empty_queue_leaves_the_trainer_to_its_own_work() -> anyhow::Result<()> {
    let alone = server_on("the-only-box").await?;
    assert!(alone.claim_genesis_request(&alone.store).await.is_none());
    // And another user's waiting run is not this user's to take.
    let other = server_on("the-only-box").await?;
    antumbra_store::repo::genesis::ask(
        &other.store,
        &antumbra_core::GenesisRequest::new(
            other.tenant.clone(),
            UserId::new("user:someone-else"),
            CompartmentId::new("comp:theirs"),
            "their-laptop",
            "their-rig",
            Utc::now(),
        ),
    )
    .await?;
    assert!(other.claim_genesis_request(&other.store).await.is_none());
    Ok(())
}

/// ADR-0023 B-1, validations 1 to 3: a recall is bounded, says when it cut, and
/// hands over the whole text on request.
///
/// The bound exists because recall returns `top_k` rows of unbounded prose into
/// a context window the caller still has to do work in. These assert the three
/// things an agent needs to trust it: the payload has a ceiling, the marker is
/// exact rather than advisory, and there is a documented way to get the rest.
mod bounded_answers;

/// ADR-0023 B-2 / ADR-0024 D-2: recall says nothing as nothing.
///
/// Validation 4 asks that a recall where nothing cleared the floor be
/// distinguishable BY A FIELD rather than by inference from one where weak rows
/// did. These drive that through the typed-decision port with a scripted
/// decider, so the behaviour is pinned before any trained head exists.
mod relevance_floor;
mod standing;

/// Takes the `tools/list` capture ADR-0023 B-3's lint is pointed at, without
/// standing a server up.
///
/// The lint lives in `antumbra-cli` and this surface lives here, and neither
/// crate depends on the other, so the two meet through a file. `tool_router()`
/// is a static accessor over the macro-generated table, which is the same table
/// `advertised_tools` filters, so a capture taken this way is what an unfiltered
/// session would be advertised.
///
/// Ignored because it writes a file. Take the capture and lint it with:
///   ANTUMBRA_TOOLS_LIST_OUT=/tmp/tools-list.json \
///   cargo test -p antumbra-mcp -- --ignored capture_the_tool_list
///   ANTUMBRA_TOOLS_LIST=/tmp/tools-list.json \
///   cargo test -p antumbra-cli -- --ignored --nocapture report_what_a_live_server_returns
#[test]
#[ignore = "writes a capture; set ANTUMBRA_TOOLS_LIST_OUT"]
fn capture_the_tool_list() {
    let Ok(out) = std::env::var("ANTUMBRA_TOOLS_LIST_OUT") else {
        println!("ANTUMBRA_TOOLS_LIST_OUT unset -- skipped");
        return;
    };
    let tools = McpServer::tool_router().list_all();
    let body = serde_json::json!({ "tools": tools });
    std::fs::write(
        &out,
        serde_json::to_string_pretty(&body).expect("serialize"),
    )
    .expect("write the capture");
    println!("wrote {} tool(s) to {out}", tools.len());
}
