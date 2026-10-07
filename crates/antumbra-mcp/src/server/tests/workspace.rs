//! What a workspace holds and who may see it: knowledge documents and the
//! compartments they live in, the proposals that group an inbox, tenant
//! isolation over a shared connection, answering from a routed expert, and
//! the tool profile that narrows what is advertised.
//!
//! A child of the server tests, so `super::*` is their fixtures.

use super::*;

#[tokio::test]
async fn ingest_then_recall_a_knowledge_document() {
    let s = server().await;
    let ingested = s
        .ingest_document(Parameters(IngestDocumentParams {
            compartment: None,
            provenance: None,
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
/// A copal that counts what reaches it, so a test can show that a refused
/// ingest never got as far as the archive.
struct CountingCopal {
    calls: std::sync::atomic::AtomicUsize,
}

impl antumbra_copal::CopalTransport for CountingCopal {
    fn post_json(
        &self,
        _url: &str,
        _credential: &antumbra_copal::CopalCredential,
        _body: &serde_json::Value,
    ) -> antumbra_core::Result<serde_json::Value> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(serde_json::json!({ "id": "01k6f2x9q3w8e5r7t1y4z6a8b0", "state": "draft" }))
    }

    fn put_bytes(
        &self,
        _url: &str,
        _credential: &antumbra_copal::CopalCredential,
        _content_type: &str,
        _digest: &str,
        _body: &[u8],
    ) -> antumbra_core::Result<serde_json::Value> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(
            serde_json::json!({ "digest": "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08", "state": "ready" }),
        )
    }
}

/// A document goes into a compartment only when its author can write there, and
/// the question is asked before the original is archived: the ingest uploads
/// first, and the engine's refusal of a chunk is silent, so a refusal discovered
/// afterwards would already have left the original in the archive.
#[tokio::test]
async fn ingest_into_a_compartment_is_private_and_refused_before_archiving_when_not_yours(
) -> anyhow::Result<()> {
    let copal = Arc::new(CountingCopal {
        calls: std::sync::atomic::AtomicUsize::new(0),
    });
    let archive = Arc::new(antumbra_copal::CopalArchive::with_transport(
        "127.0.0.1:9010",
        antumbra_copal::CopalTenancy::PerWorkspace,
        copal.clone(),
    ));
    let s = server().await.with_copal_archive(archive);
    let mine = s
        .create_compartment(Parameters(CreateCompartmentParams {
            name: "reviews".into(),
        }))
        .await?
        .0
        .id;
    let doc = |compartment: &str| IngestDocumentParams {
        compartment: Some(compartment.into()),
        provenance: None,
        title: "review".into(),
        content: "A private salary review.".into(),
        source: None,
    };

    let refused = match s
        .ingest_document(Parameters(doc("comp:someone-elses")))
        .await
    {
        Ok(_) => anyhow::bail!("a compartment that is not yours must be refused"),
        Err(e) => e.message.to_string(),
    };
    assert!(refused.contains("not one you can write to"), "{refused}");
    assert_eq!(
        copal.calls.load(std::sync::atomic::Ordering::SeqCst),
        0,
        "nothing reached the archive"
    );

    s.ingest_document(Parameters(doc(&mine))).await?;
    assert!(copal.calls.load(std::sync::atomic::Ordering::SeqCst) > 0);
    let recalled = s
        .recall_documents(Parameters(RecallDocumentsParams {
            query: "salary review".into(),
            top_k: Some(3),
        }))
        .await?;
    assert_eq!(recalled.0.chunks.len(), 1);
    assert_eq!(
        recalled.0.chunks[0].compartment.as_deref(),
        Some(mine.as_str()),
        "the recalled chunk says where it is kept"
    );
    Ok(())
}

struct CannedCopal {
    up: bool,
}

impl antumbra_copal::CopalTransport for CannedCopal {
    fn post_json(
        &self,
        _url: &str,
        _credential: &antumbra_copal::CopalCredential,
        _body: &serde_json::Value,
    ) -> antumbra_core::Result<serde_json::Value> {
        if self.up {
            Ok(serde_json::json!({ "id": "01k6f2x9q3w8e5r7t1y4z6a8b0", "state": "draft" }))
        } else {
            Err(antumbra_core::AntumbraError::other("connection refused"))
        }
    }

    fn put_bytes(
        &self,
        _url: &str,
        _credential: &antumbra_copal::CopalCredential,
        _content_type: &str,
        _digest: &str,
        _body: &[u8],
    ) -> antumbra_core::Result<serde_json::Value> {
        if self.up {
            Ok(
                serde_json::json!({ "digest": "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08", "state": "ready" }),
            )
        } else {
            Err(antumbra_core::AntumbraError::other("connection refused"))
        }
    }
}

fn canned_archive(up: bool) -> Arc<antumbra_copal::CopalArchive> {
    Arc::new(antumbra_copal::CopalArchive::with_transport(
        "127.0.0.1:9010",
        // Per-workspace tenancy (the default): the session's workspace
        // presents itself as the copal tenant. The header side is proven
        // in `antumbra_copal`'s own tests; these care about the ingest path.
        antumbra_copal::CopalTenancy::PerWorkspace,
        Arc::new(CannedCopal { up }),
    ))
}

#[tokio::test]
async fn ingest_with_a_copal_archive_stamps_every_chunk_with_provenance() {
    let s = server().await.with_copal_archive(canned_archive(true));
    s.ingest_document(Parameters(IngestDocumentParams {
        compartment: None,
        provenance: None,
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
    assert!(recalled.0.chunks.iter().all(|c| c.copal_file.as_deref()
        == Some("01k6f2x9q3w8e5r7t1y4z6a8b0")
        && c.copal_digest.as_deref()
            == Some("9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08")));
}

#[tokio::test]
async fn ingest_fails_closed_when_the_configured_copal_is_unreachable() {
    // Upload-first: with the archive configured but down, the ingest fails
    // BEFORE any chunk is stored -- a document of record that silently
    // dropped originals would be worse than none.
    let s = server().await.with_copal_archive(canned_archive(false));
    let res = s
        .ingest_document(Parameters(IngestDocumentParams {
            compartment: None,
            provenance: None,
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

    // The document list names it, with its chunks, and over the REST
    // dispatcher too.
    let listed = s.list_documents().await.unwrap();
    assert_eq!(listed.0.documents.len(), 1);
    assert!(listed.0.documents[0].chunks >= 1);
    let via = s
        .call_tool("list_documents", serde_json::json!({}))
        .await
        .unwrap();
    assert_eq!(via["documents"].as_array().map(Vec::len), Some(1));

    // The stats tool is reachable over the REST dispatcher too.
    let via = s
        .call_tool("workspace_stats", serde_json::json!({}))
        .await
        .unwrap();
    assert!(via.get("memories").is_some());
}

/// A page of `list_memories` is the most recently updated first, says
/// whether another follows, and continues from `offset`, over the REST
/// dispatcher as over JSON-RPC.
#[tokio::test]
async fn list_memories_pages_newest_first() {
    let s = server().await;
    for content in ["first", "second", "third"] {
        s.call_tool("store_memory", serde_json::json!({ "content": content }))
            .await
            .unwrap();
        // Distinct update times, so the order is the order of storing.
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    let page = |offset: u32| {
        s.call_tool(
            "list_memories",
            serde_json::json!({ "limit": 2, "offset": offset }),
        )
    };
    let contents = |v: &serde_json::Value| -> Vec<String> {
        v["memories"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["content"].as_str().unwrap().to_string())
            .collect()
    };
    let first = page(0).await.unwrap();
    assert_eq!(contents(&first), ["third", "second"]);
    assert_eq!(first["more"], true);
    let last = page(2).await.unwrap();
    assert_eq!(contents(&last), ["first"]);
    assert!(
        last.get("more").is_none(),
        "the last page says nothing follows"
    );
    let unpaged = s
        .call_tool("list_memories", serde_json::json!({}))
        .await
        .unwrap();
    assert_eq!(contents(&unpaged).len(), 3);
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
        .list_memories(Parameters(ListParams {
            network: None,
            limit: None,
            offset: None,
        }))
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
    let b_page = server_b
        .list_memories(Parameters(ListParams {
            network: None,
            limit: Some(50),
            offset: None,
        }))
        .await
        .unwrap();
    assert!(
        b_page
            .0
            .memories
            .iter()
            .all(|m| !m.content.contains("alpha")),
        "nor in a page, which the engine orders and cuts under b's session"
    );

    // Request 3: a re-binds and DOES see its own memory.
    store.signin(&ta, &ua).await.unwrap();
    let a_view = server_a
        .list_memories(Parameters(ListParams {
            network: None,
            limit: None,
            offset: None,
        }))
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
            repo: None,
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
            repo: None,
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
        .answer(Parameters(AnswerParams {
            task: "x".into(),
            repo: None,
        }))
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
            full: None,
            floor: None,
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
            full: None,
            floor: None,
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

/// A tool profile is enforced where the agent first sees the tools and where
/// it calls them: `agent` advertises its seventeen, the REST dispatch serves those
/// and refuses the rest by name, and no profile advertises everything.
#[tokio::test]
async fn a_tool_profile_narrows_what_is_advertised_and_served() {
    let all = McpServer::all_tool_names();
    assert_eq!(all.len(), 33, "{all:?}");
    let profile = crate::profile::ToolProfile::parse("agent", &all)
        .unwrap()
        .expect("agent is a profile, not `all`");
    let s = server().await.with_tool_profile(Arc::new(profile));
    let advertised: Vec<String> = s
        .advertised_tools()
        .into_iter()
        .map(|t| t.name.to_string())
        .collect();
    assert_eq!(advertised.len(), 17, "{advertised:?}");
    assert!(advertised.iter().any(|n| n == "recall_memories"));
    assert!(!advertised.iter().any(|n| n == "share_compartment"));

    let stored = s
        .call_tool(
            "store_memory",
            serde_json::json!({ "content": "deno, not npm, in acme-api", "network": "opinion" }),
        )
        .await
        .unwrap();
    assert!(stored["id"].as_str().unwrap().starts_with("memory:"));

    let err = s
        .call_tool("list_memories", serde_json::json!({}))
        .await
        .unwrap_err();
    assert!(
        err.message.contains("not in this server's tool profile")
            && err.message.contains("recall_memories"),
        "{}",
        err.message
    );

    assert_eq!(server().await.advertised_tools().len(), 33);
}
