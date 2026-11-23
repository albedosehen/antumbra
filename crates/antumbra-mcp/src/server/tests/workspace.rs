//! What a workspace holds and who may see it: knowledge documents and the
//! compartments they live in, the proposals that group an inbox, tenant
//! isolation over a shared connection, answering from a routed expert, and
//! the tool profile that narrows what is advertised.
//!
//! A child of the server tests, so `super::*` is their fixtures.

use super::*;

#[tokio::test]
async fn ingest_then_recall_a_knowledge_document() -> anyhow::Result<()> {
    let s = server().await?;
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
        .await?;
    assert!(ingested.0.chunks >= 1);
    assert_eq!(ingested.0.title, "Onboarding");

    let recalled = s
        .recall_documents(Parameters(RecallDocumentsParams {
            query: "what runtime does this project use".into(),
            top_k: Some(3),
        }))
        .await?;
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
        .await?;
    assert!(viarest.get("chunks").is_some());
    Ok(())
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
    let s = server().await?.with_copal_archive(archive);
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
async fn ingest_with_a_copal_archive_stamps_every_chunk_with_provenance() -> anyhow::Result<()> {
    let s = server().await?.with_copal_archive(canned_archive(true));
    s.ingest_document(Parameters(IngestDocumentParams {
        compartment: None,
        provenance: None,
        title: "Onboarding".into(),
        content: "Antumbra keeps knowledge documents separate from episodic memory. \
                  This project uses the deno runtime."
            .into(),
        source: Some("onboarding.md".into()),
    }))
    .await?;

    let recalled = s
        .recall_documents(Parameters(RecallDocumentsParams {
            query: "what runtime does this project use".into(),
            top_k: Some(3),
        }))
        .await?;
    assert!(!recalled.0.chunks.is_empty());
    // Every chunk names the archived original: the file AND the bytes.
    assert!(recalled.0.chunks.iter().all(|c| c.copal_file.as_deref()
        == Some("01k6f2x9q3w8e5r7t1y4z6a8b0")
        && c.copal_digest.as_deref()
            == Some("9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08")));
    Ok(())
}

#[tokio::test]
async fn ingest_fails_closed_when_the_configured_copal_is_unreachable() -> anyhow::Result<()> {
    // Upload-first: with the archive configured but down, the ingest fails
    // BEFORE any chunk is stored -- a document of record that silently
    // dropped originals would be worse than none.
    let s = server().await?.with_copal_archive(canned_archive(false));
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
    assert_eq!(s.workspace_stats().await?.0.documents, 0);
    Ok(())
}

#[tokio::test]
async fn population_and_stats_report_the_workspace() -> anyhow::Result<()> {
    let s = server().await?;
    // A fresh workspace has no experts.
    assert!(s.population().await?.0.experts.is_empty());
    assert_eq!(s.workspace_stats().await?.0.experts, 0);

    // Seed a memory and a document through the dispatcher.
    s.call_tool(
        "store_memory",
        serde_json::json!({ "content": "remember this" }),
    )
    .await?;
    s.call_tool(
        "ingest_document",
        serde_json::json!({ "title": "Doc", "content": "some reference text" }),
    )
    .await?;

    let stats = s.workspace_stats().await?;
    assert!(stats.0.memories >= 1);
    assert_eq!(stats.0.documents, 1);

    // The document list names it, with its chunks, and over the REST
    // dispatcher too.
    let listed = s.list_documents().await?;
    assert_eq!(listed.0.documents.len(), 1);
    assert!(listed.0.documents[0].chunks >= 1);
    let via = s.call_tool("list_documents", serde_json::json!({})).await?;
    assert_eq!(via["documents"].as_array().map(Vec::len), Some(1));

    // The stats tool is reachable over the REST dispatcher too.
    let via = s
        .call_tool("workspace_stats", serde_json::json!({}))
        .await?;
    assert!(via.get("memories").is_some());
    Ok(())
}

/// An engine that says how it serves, as a model-backed one does.
struct Described;
#[async_trait::async_trait]
impl antumbra_core::ports::Serve for Described {
    async fn act(
        &self,
        _req: ActRequest,
    ) -> antumbra_core::Result<antumbra_core::ports::ActOutput> {
        Err(antumbra_core::AntumbraError::other(
            "asking how an engine serves answers nothing",
        ))
    }
    fn serving(&self) -> Option<antumbra_core::ports::Serving> {
        Some(antumbra_core::ports::Serving {
            device: "cpu".into(),
            base: "q6_k".into(),
            kernels: Some("avx2".into()),
        })
    }
}

/// `workspace_stats` says whether the server answers through experts and on
/// what, which is what `antumbra setup check` reads: `null` with no engine (a
/// build without the model backend) and for an engine that does not read a
/// model, and the engine's own account where it gives one.
#[tokio::test]
async fn stats_say_how_the_server_serves_experts() -> anyhow::Result<()> {
    let none = server().await?;
    let via = none
        .call_tool("workspace_stats", serde_json::json!({}))
        .await?;
    assert_eq!(via.get("serving"), Some(&serde_json::Value::Null));

    let build = |serve: Arc<dyn antumbra_core::ports::Serve>| async move {
        anyhow::Ok(McpServer::new(
            mem_store().await?,
            Arc::new(FixedEmbedder::new(EMBED_DIM)),
            TenantId::new("ws:test"),
            UserId::new("user:test"),
            "h".into(),
            CompartmentId::new("comp:test:default"),
            Some(serve),
        ))
    };
    let echo = build(Arc::new(antumbra_core::testing::EchoServe)).await?;
    let via = echo
        .call_tool("workspace_stats", serde_json::json!({}))
        .await?;
    assert_eq!(via.get("serving"), Some(&serde_json::Value::Null));

    let described = build(Arc::new(Described)).await?;
    let via = described
        .call_tool("workspace_stats", serde_json::json!({}))
        .await?;
    assert_eq!(
        via.get("serving"),
        Some(&serde_json::json!({ "device": "cpu", "base": "q6_k", "kernels": "avx2" }))
    );
    Ok(())
}

/// A page of `list_memories` is the most recently updated first, says
/// whether another follows, and continues from `offset`, over the REST
/// dispatcher as over JSON-RPC.
#[tokio::test]
async fn list_memories_pages_newest_first() -> anyhow::Result<()> {
    let s = server().await?;
    for content in ["first", "second", "third"] {
        s.call_tool("store_memory", serde_json::json!({ "content": content }))
            .await?;
        // Distinct update times, so the order is the order of storing.
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    let page = |offset: u32| {
        s.call_tool(
            "list_memories",
            serde_json::json!({ "limit": 2, "offset": offset }),
        )
    };
    let contents = |v: &serde_json::Value| -> anyhow::Result<Vec<String>> {
        let Some(memories) = v["memories"].as_array() else {
            anyhow::bail!("the page lists no memories: {v}");
        };
        memories
            .iter()
            .map(|m| {
                m["content"]
                    .as_str()
                    .map(str::to_string)
                    .ok_or_else(|| anyhow::anyhow!("a listed memory has no content: {m}"))
            })
            .collect()
    };
    let first = page(0).await?;
    assert_eq!(contents(&first)?, ["third", "second"]);
    assert_eq!(first["more"], true);
    let last = page(2).await?;
    assert_eq!(contents(&last)?, ["first"]);
    assert!(
        last.get("more").is_none(),
        "the last page says nothing follows"
    );
    let unpaged = s.call_tool("list_memories", serde_json::json!({})).await?;
    assert_eq!(contents(&unpaged)?.len(), 3);
    Ok(())
}

#[tokio::test]
async fn compartment_tools_create_list_store_share() -> anyhow::Result<()> {
    let s = server().await?;

    let c = s
        .create_compartment(Parameters(CreateCompartmentParams {
            name: "deno work".into(),
        }))
        .await?;
    assert!(c.0.id.starts_with("comp:"));
    assert_eq!(c.0.origin, "user");

    let listed = s.list_compartments().await?;
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
        .await?;
    assert!(m.0.id.starts_with("memory:"));

    // Share + revoke succeed (engine enforcement is proven in the store
    // crate's penumbra_compartment test; here we exercise the tool plumbing).
    assert!(
        s.share_compartment(Parameters(ShareParams {
            compartment_id: c.0.id.clone(),
            grantee: "user:other".into(),
            capability: "reference".into(),
        }))
        .await?
        .0
        .shared
    );
    assert!(
        s.revoke_compartment(Parameters(RevokeParams {
            compartment_id: c.0.id,
            grantee: "user:other".into(),
        }))
        .await?
        .0
        .revoked
    );
    Ok(())
}

// Sharing and revoking are the owner's: another member of the workspace who
// names your compartment is refused in words, not told `shared: true` while
// the engine persisted nothing; a share that landed reads back as a grant,
// and a revoke that landed leaves none. Two users' servers over one store.
#[tokio::test]
async fn sharing_someone_elses_compartment_is_refused_not_reported_shared() -> anyhow::Result<()> {
    let store = mem_store().await?;
    let tenant = TenantId::new("ws:test");
    let server_for = |user: &str| McpServer::for_tests_as(store.clone(), "ws:test", user);
    let a = server_for("user:a");
    let b = server_for("user:b");
    let owned = a
        .create_compartment(Parameters(CreateCompartmentParams { name: "a's".into() }))
        .await?;
    let id = CompartmentId::new(owned.0.id.clone());
    let share = |grantee: &str| ShareParams {
        compartment_id: owned.0.id.clone(),
        grantee: grantee.into(),
        capability: "reference".into(),
    };
    let grants = || antumbra_store::repo::compartment::list_grants(&store, &tenant, &id);

    let Err(e) = b.share_compartment(Parameters(share("user:c"))).await else {
        anyhow::bail!("B shared A's compartment");
    };
    anyhow::ensure!(
        e.code == rmcp::model::ErrorCode::INVALID_PARAMS && e.message.contains("not one you own"),
        "{e:?}"
    );
    let Err(e) = b
        .share_compartment(Parameters(ShareParams {
            compartment_id: "comp:nowhere".into(),
            ..share("user:c")
        }))
        .await
    else {
        anyhow::bail!("B shared a compartment that does not exist");
    };
    anyhow::ensure!(e.code == rmcp::model::ErrorCode::INVALID_PARAMS, "{e:?}");
    let Err(e) = a.share_compartment(Parameters(share("user:a"))).await else {
        anyhow::bail!("A shared a compartment with its own owner");
    };
    anyhow::ensure!(e.code == rmcp::model::ErrorCode::INVALID_PARAMS, "{e:?}");
    anyhow::ensure!(grants().await?.is_empty(), "a refused share left a grant");

    anyhow::ensure!(
        a.share_compartment(Parameters(share("user:c")))
            .await?
            .0
            .shared
    );
    let after = grants().await?;
    anyhow::ensure!(
        after.len() == 1
            && after
                .first()
                .is_some_and(|g| g.grantee == UserId::new("user:c")),
        "the owner's share did not read back as one grant"
    );

    let revoke = || RevokeParams {
        compartment_id: owned.0.id.clone(),
        grantee: "user:c".into(),
    };
    let Err(e) = b.revoke_compartment(Parameters(revoke())).await else {
        anyhow::bail!("B revoked a grant on A's compartment");
    };
    anyhow::ensure!(e.code == rmcp::model::ErrorCode::INVALID_PARAMS, "{e:?}");
    anyhow::ensure!(
        grants().await?.len() == 1,
        "B's refused revoke changed the grants"
    );
    anyhow::ensure!(a.revoke_compartment(Parameters(revoke())).await?.0.revoked);
    anyhow::ensure!(
        grants().await?.is_empty(),
        "the owner's revoke did not land"
    );
    Ok(())
}

#[tokio::test]
async fn propose_compartments_clusters_inbox_then_applies() -> anyhow::Result<()> {
    let s = server().await?;
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
        .await?;
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
        .await?;
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
        .await?;
    let Some(prop) = applied.0.proposals.first() else {
        anyhow::bail!("apply returned no proposal");
    };
    let Some(new_id) = prop.compartment_id.clone() else {
        anyhow::bail!("apply persisted no compartment id");
    };

    // It is owned by the user and marked as a proposal awaiting curation.
    let comps = s.list_compartments().await?;
    assert!(comps
        .0
        .compartments
        .iter()
        .any(|c| c.id == new_id && c.origin == "proposed"));

    // Its members were moved out of the inbox into it (engine round-trip).
    let moved =
        memory::list_by_compartment(&s.store, &s.tenant, &CompartmentId::new(new_id)).await?;
    assert_eq!(moved.len(), prop.members.len());
    assert!(moved.len() >= 3, "the whole inbox region moved");
    Ok(())
}

#[tokio::test]
async fn shared_connection_isolates_tenants_under_signin() -> anyhow::Result<()> {
    // The HTTP transport's model on an embedded (single-writer) engine: ONE
    // shared connection, signed in per request. Two tenants' servers share
    // the store; serialized signin must isolate them through the real MCP
    // tools, not just at the store layer.
    let store = mem_store().await?;
    let embedder: Arc<dyn Embedder> = Arc::new(FixedEmbedder::new(EMBED_DIM));
    let (ta, ua) = (TenantId::new("ws:a"), UserId::new("user:a"));
    let (tb, ub) = (TenantId::new("ws:b"), UserId::new("user:b"));

    // Provision both identities owner-side (principal + default compartment).
    let comp_a = crate::provision_identity(&store, &ta, &ua).await?;
    let comp_b = crate::provision_identity(&store, &tb, &ub).await?;
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
    store.signin(&ta, &ua).await?;
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
        .await?;

    // Request 2: re-bind the SAME connection as tenant b; b must not see it.
    store.signin(&tb, &ub).await?;
    let b_view = server_b
        .list_memories(Parameters(ListParams {
            host: None,
            network: None,
            limit: None,
            offset: None,
        }))
        .await?;
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
            host: None,
            network: None,
            limit: Some(50),
            offset: None,
        }))
        .await?;
    assert!(
        b_page
            .0
            .memories
            .iter()
            .all(|m| !m.content.contains("alpha")),
        "nor in a page, which the engine orders and cuts under b's session"
    );

    // Request 3: a re-binds and DOES see its own memory.
    store.signin(&ta, &ua).await?;
    let a_view = server_a
        .list_memories(Parameters(ListParams {
            host: None,
            network: None,
            limit: None,
            offset: None,
        }))
        .await?;
    assert!(
        a_view
            .0
            .memories
            .iter()
            .any(|m| m.content.contains("alpha")),
        "tenant a must see its own memory"
    );
    Ok(())
}

#[tokio::test]
async fn auto_propose_trigger_fires_once_the_inbox_grows() -> anyhow::Result<()> {
    // With the autonomous trigger armed at 4, the inbox is left alone until it
    // reaches 4 memories, at which point a write self-organizes it into an
    // Origin::Proposed compartment (and the inbox shrinks, quieting the trigger).
    let store = mem_store().await?;
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
        let out = s.store_memory(Parameters(put(n))).await?;
        assert!(
            out.0.auto_proposed.is_empty(),
            "below threshold: inbox left alone"
        );
    }
    // The fourth write reaches the threshold: the antumbra organizes the inbox.
    let out = s.store_memory(Parameters(put(3))).await?;
    assert!(
        !out.0.auto_proposed.is_empty(),
        "at threshold the antumbra auto-creates proposed compartment(s)"
    );

    // The created compartment is a proposal the user owns, awaiting curation.
    let comps = s.list_compartments().await?;
    assert!(comps.0.compartments.iter().any(|c| c.origin == "proposed"));

    // The inbox shrank below the threshold, so the next write does not re-fire.
    let again = s.store_memory(Parameters(put(99))).await?;
    assert!(
        again.0.auto_proposed.is_empty(),
        "inbox no longer over threshold"
    );
    Ok(())
}

#[tokio::test]
async fn answer_routes_then_serves_through_the_expert() -> anyhow::Result<()> {
    use antumbra_core::testing::EchoServe;
    let store = mem_store().await?;
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
    .await?;
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
            standing_only: false,
        }))
        .await?;
    assert!(!out.0.escalate, "the population covers the task");
    assert_eq!(out.0.expert_id.as_deref(), Some("expert:adder"));
    // EchoServe serves the prompt straight back, proving route -> serve wiring.
    assert_eq!(out.0.answer, "add two numbers");
    Ok(())
}

// A serving engine that doesn't have the routed expert's adapter must make
// `answer` ESCALATE, not error -- and never call `act` (F4). An `act` that is
// called anyway answers with an error, which `answer` passes on, so the test
// below would fail on it.
struct Unservable;
#[async_trait::async_trait]
impl antumbra_core::ports::Serve for Unservable {
    async fn act(
        &self,
        _req: ActRequest,
    ) -> antumbra_core::Result<antumbra_core::ports::ActOutput> {
        Err(antumbra_core::AntumbraError::other(
            "act must not be called when the expert is unservable",
        ))
    }
    fn can_serve(&self, _expert: &ExpertId) -> bool {
        false
    }
}

#[tokio::test]
async fn answer_escalates_when_the_routed_expert_is_not_servable() -> anyhow::Result<()> {
    let store = mem_store().await?;
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
    .await?;
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
            standing_only: false,
        }))
        .await?;
    assert!(
        out.0.escalate,
        "an unservable routed expert escalates, not errors"
    );
    assert_eq!(out.0.expert_id.as_deref(), Some("expert:adder"));
    assert!(out.0.answer.is_empty());
    Ok(())
}

// An engine still fetching its base, as a server is before its first answer
// through an expert, or one whose fetch failed. `act` is never reached: it
// answers with an error, which `answer` would pass on.
struct NotYet(antumbra_core::ports::Readiness);
#[async_trait::async_trait]
impl antumbra_core::ports::Serve for NotYet {
    async fn act(
        &self,
        _req: ActRequest,
    ) -> antumbra_core::Result<antumbra_core::ports::ActOutput> {
        Err(antumbra_core::AntumbraError::other(
            "act must not be called before the base is here",
        ))
    }
    fn prepare(&self) -> antumbra_core::ports::Readiness {
        self.0.clone()
    }
}

/// A first answer on a server whose base has never been downloaded escalates
/// at once and says the base is coming, where it used to wait out a 3 GB
/// download and time out; a failed download says why.
#[tokio::test]
async fn answer_escalates_while_the_base_is_fetched() -> anyhow::Result<()> {
    use antumbra_core::ports::Readiness;
    for (readiness, said) in [
        (
            Readiness::NotYet("the base model (Qwen/x) is downloading to this server".into()),
            "the base model (Qwen/x) is downloading to this server; escalate",
        ),
        (
            Readiness::NotYet("the base model (Qwen/x) is loading on this server".into()),
            "is loading on this server; escalate",
        ),
        (
            Readiness::Failed("no network".into()),
            "could not fetch its base model (no network)",
        ),
    ] {
        let store = mem_store().await?;
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
        .await?;
        let s = McpServer::new(
            store,
            Arc::new(FixedEmbedder::new(EMBED_DIM)),
            TenantId::new("ws:test"),
            UserId::new("user:test"),
            "h".into(),
            CompartmentId::new("comp:test:default"),
            Some(Arc::new(NotYet(readiness))),
        );
        let out = s
            .answer(Parameters(AnswerParams {
                task: "add two numbers".into(),
                repo: None,
                standing_only: false,
            }))
            .await?;
        assert!(out.0.escalate, "a base not yet here escalates");
        assert_eq!(out.0.expert_id.as_deref(), Some("expert:adder"));
        assert!(out.0.answer.is_empty());
        let note = out.0.note.unwrap_or_default();
        assert!(note.contains(said), "{note}");
    }
    Ok(())
}

#[tokio::test]
async fn answer_without_a_serving_engine_reports_not_configured() -> anyhow::Result<()> {
    let s = server().await?; // serve = None
    let out = s
        .answer(Parameters(AnswerParams {
            task: "x".into(),
            repo: None,
            standing_only: false,
        }))
        .await?;
    assert!(out.0.escalate && out.0.note.is_some());
    Ok(())
}

/// Provenance travels with the memory: a structured anchor becomes a `git:`
/// evidence entry, comes back parsed on every view, and a recall given the
/// caller's repo + branch tags each hit's scope and demotes the out-of-scope
/// ones below the in-scope ones without hiding them.
#[tokio::test]
async fn provenance_is_stored_and_scopes_recall() -> anyhow::Result<()> {
    let s = server().await?;
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
    .await?;
    s.store_memory(Parameters(store(
        "the orders route is POST /orders on main",
        "main",
        "github.com/o/r",
    )))
    .await?;
    s.store_memory(Parameters(store(
        "the orders route lives in another repo",
        "main",
        "github.com/o/other",
    )))
    .await?;

    // No context: every hit comes back with its parsed provenance and no scope.
    let plain = s
        .recall_memories(Parameters(RecallParams {
            host: None,
            repo: None,
            branch: None,
            query: "orders route".into(),
            top_k: Some(5),
            network: None,
            full: None,
            floor: None,
            session: None,
        }))
        .await?;
    assert_eq!(plain.0.memories.len(), 3);
    for m in &plain.0.memories {
        let Some(p) = m.provenance.as_ref() else {
            anyhow::bail!(
                "no provenance was parsed from the evidence of `{}`",
                m.content
            );
        };
        assert_eq!(p.commit, "b697da7");
        assert!(m.scope.is_none());
    }

    // On main in github.com/o/r: the main memory is in scope, the feature-branch
    // and other-repo memories are demoted to the tail, in that relative order.
    let scoped = s
        .recall_memories(Parameters(RecallParams {
            host: None,
            repo: Some("GitHub.com/o/r.git".into()),
            branch: Some("main".into()),
            query: "orders route".into(),
            top_k: Some(5),
            network: None,
            full: None,
            floor: None,
            session: None,
        }))
        .await?;
    let scopes = scoped
        .0
        .memories
        .iter()
        .map(|m| {
            m.scope
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("`{}` came back with no scope", m.content))
        })
        .collect::<anyhow::Result<Vec<&str>>>()?;
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
    Ok(())
}

/// A tool profile is enforced where the agent first sees the tools and where
/// it calls them: `agent` advertises its twenty-one, the REST dispatch serves those
/// and refuses the rest by name, and no profile advertises everything.
#[tokio::test]
async fn a_tool_profile_narrows_what_is_advertised_and_served() -> anyhow::Result<()> {
    let all = McpServer::all_tool_names();
    assert_eq!(all.len(), 42, "{all:?}");
    let Some(profile) = crate::profile::ToolProfile::parse("agent", &all)? else {
        anyhow::bail!("agent is a profile, not `all`");
    };
    let s = server().await?.with_tool_profile(Arc::new(profile));
    let advertised: Vec<String> = s
        .advertised_tools()
        .into_iter()
        .map(|t| t.name.to_string())
        .collect();
    assert_eq!(advertised.len(), 21, "{advertised:?}");
    assert!(advertised.iter().any(|n| n == "recall_memories"));
    assert!(advertised.iter().any(|n| n == "publish_relic"));
    assert!(!advertised.iter().any(|n| n == "share_compartment"));
    // Who may see a relic is the user's call, made in the dashboard.
    assert!(!advertised.iter().any(|n| n == "set_relic_visibility"));

    let stored = s
        .call_tool(
            "store_memory",
            serde_json::json!({ "content": "deno, not npm, in acme-api", "network": "opinion" }),
        )
        .await?;
    let Some(id) = stored["id"].as_str() else {
        anyhow::bail!("the stored memory carries no id: {stored}");
    };
    assert!(id.starts_with("memory:"));

    let Err(err) = s.call_tool("list_memories", serde_json::json!({})).await else {
        anyhow::bail!("a tool outside the profile was served");
    };
    assert!(
        err.message.contains("not in this server's tool profile")
            && err.message.contains("recall_memories"),
        "{}",
        err.message
    );

    assert_eq!(server().await?.advertised_tools().len(), 42);
    // A connection that asked for the agent's tools is advertised those.
    let known = McpServer::all_tool_names();
    let agent = crate::profile::ToolProfile::asked("agent", &known)
        .ok_or_else(|| anyhow::anyhow!("the agent profile names no tool"))?;
    assert_eq!(
        server().await?.given_only(&agent).advertised_tools().len(),
        21
    );
    Ok(())
}

/// An engine that generates on a thread of its own and never answers: its
/// call waits until it is dropped, and says when it is.
struct NeverAnswers {
    asked: Arc<tokio::sync::Notify>,
    dropped: Arc<std::sync::atomic::AtomicBool>,
}

struct SaysWhenDropped(Arc<std::sync::atomic::AtomicBool>);

impl Drop for SaysWhenDropped {
    fn drop(&mut self) {
        self.0.store(true, std::sync::atomic::Ordering::SeqCst);
    }
}

#[async_trait::async_trait]
impl antumbra_core::ports::Serve for NeverAnswers {
    async fn act(
        &self,
        _req: ActRequest,
    ) -> antumbra_core::Result<antumbra_core::ports::ActOutput> {
        let _held = SaysWhenDropped(self.dropped.clone());
        self.asked.notify_one();
        std::future::pending().await
    }

    fn answers_from_its_own_thread(&self) -> bool {
        true
    }
}

/// A client that cancels an `answer` drops the engine's call, which is how
/// the engine learns no one waits and stops decoding for it.
#[tokio::test]
async fn a_cancelled_answer_drops_the_engines_call() -> anyhow::Result<()> {
    let store = mem_store().await?;
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
    .await?;
    let asked = Arc::new(tokio::sync::Notify::new());
    let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let s = McpServer::new(
        store,
        Arc::new(FixedEmbedder::new(EMBED_DIM)),
        TenantId::new("ws:test"),
        UserId::new("user:test"),
        "h".into(),
        CompartmentId::new("comp:test:default"),
        Some(Arc::new(NeverAnswers {
            asked: asked.clone(),
            dropped: dropped.clone(),
        })),
    );
    let cancelled = tokio_util::sync::CancellationToken::new();
    let call = s.answer_until_cancelled(
        Parameters(AnswerParams {
            task: "add two numbers".into(),
            repo: None,
            standing_only: false,
        }),
        cancelled.clone(),
    );
    let cancel = async {
        // Not before the engine has the request.
        asked.notified().await;
        cancelled.cancel();
    };
    let (out, ()) = tokio::join!(call, cancel);
    assert!(out.is_err(), "a cancelled call is not answered");
    assert!(
        dropped.load(std::sync::atomic::Ordering::SeqCst),
        "the engine's call was dropped with it"
    );
    Ok(())
}
