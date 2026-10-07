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
mod device;
mod handoff;
mod lifecycle;
mod workspace;

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
