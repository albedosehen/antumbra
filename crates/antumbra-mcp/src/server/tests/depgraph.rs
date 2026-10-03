//! The evidence graph's tools (ADR-0019): recording a dependency claim, seeing
//! it again, and walking the blast radius with the evidence attached.

use super::*;

fn server_in(store: &Store, tenant: &str, user: &str) -> McpServer {
    McpServer::new(
        store.clone(),
        Arc::new(FixedEmbedder::new(EMBED_DIM)),
        TenantId::new(tenant),
        UserId::new(user),
        "test-host".into(),
        CompartmentId::new(format!("comp:{tenant}:{user}:default")),
        None,
    )
}

async fn record(
    s: &McpServer,
    from: &str,
    to: &str,
    source: &str,
    detail: &str,
) -> RecordedDependencyOut {
    s.record_dependency(Parameters(RecordDependencyParams {
        from: from.into(),
        to: to.into(),
        source: source.into(),
        detail: Some(detail.into()),
        provenance: None,
    }))
    .await
    .unwrap()
    .0
}

async fn radius(s: &McpServer, service: &str, direction: Option<&str>) -> BlastRadiusOut {
    s.blast_radius(Parameters(BlastRadiusParams {
        service: service.into(),
        direction: direction.map(str::to_string),
        depth: None,
        min_weight: None,
    }))
    .await
    .unwrap()
    .0
}

/// What breaks if orders does: checkout directly, web through it, each with
/// the evidence for every hop. A model's claim alone (billing) stays under the
/// default floor.
#[tokio::test]
async fn blast_radius_walks_the_recorded_edges_with_their_evidence() {
    let store = Store::connect_memory(EMBED_DIM).await.unwrap();
    let s = server_in(&store, "ws:test", "user:a");
    record(
        &s,
        "github.com/acme/web",
        "github.com/acme/checkout",
        "declared",
        "package.json names @acme/checkout",
    )
    .await;
    record(
        &s,
        "github.com/acme/checkout",
        "github.com/acme/orders",
        "declared",
        "go.mod requires orders",
    )
    .await;
    record(
        &s,
        "github.com/acme/checkout",
        "github.com/acme/orders",
        "observed",
        "APM: 1.2k calls/min",
    )
    .await;
    record(
        &s,
        "github.com/acme/billing",
        "github.com/acme/orders",
        "claimed",
        "read from the code",
    )
    .await;

    let out = radius(&s, "GitHub.com/Acme/Orders", None).await;
    assert_eq!(out.service, "github.com/acme/orders");
    assert_eq!(out.direction, "dependents");
    assert_eq!(out.edges, 4);
    let reached: Vec<(&str, u32)> = out
        .reached
        .iter()
        .map(|r| (r.service.as_str(), r.depth))
        .collect();
    assert_eq!(
        reached,
        [("github.com/acme/checkout", 1), ("github.com/acme/web", 2)]
    );
    let checkout = &out.reached[0];
    let sources: Vec<&str> = checkout.path[0]
        .evidence
        .iter()
        .map(|e| e.source.as_str())
        .collect();
    assert_eq!(
        sources,
        ["declared", "observed"],
        "both sources, strongest first"
    );
    assert!(checkout.weight > 0.98, "{}", checkout.weight);
    assert_eq!(
        checkout.path[0].evidence[0].detail.as_deref(),
        Some("go.mod requires orders")
    );

    let needs = radius(&s, "github.com/acme/web", Some("dependencies")).await;
    let needed: Vec<&str> = needs.reached.iter().map(|r| r.service.as_str()).collect();
    assert_eq!(
        needed,
        ["github.com/acme/checkout", "github.com/acme/orders"]
    );

    let loose = s
        .blast_radius(Parameters(BlastRadiusParams {
            service: "github.com/acme/orders".into(),
            direction: None,
            depth: Some(1),
            min_weight: Some(0.2),
        }))
        .await
        .unwrap()
        .0;
    assert!(
        loose
            .reached
            .iter()
            .any(|r| r.service == "github.com/acme/billing"),
        "a lower floor lets the claim through"
    );
}

/// Recording the same edge from the same source again reinforces the one
/// memory, and the latest detail replaces the old.
#[tokio::test]
async fn recording_an_edge_again_reinforces_it() {
    let store = Store::connect_memory(EMBED_DIM).await.unwrap();
    let s = server_in(&store, "ws:test", "user:a");
    let first = record(&s, "a", "b", "learned", "changed together 4 times").await;
    assert!(first.created);
    assert!((first.confidence - 0.6).abs() < 1e-6);
    let again = record(&s, "A", "b/", "learned", "changed together 9 times").await;
    assert!(!again.created);
    assert_eq!(again.id, first.id);
    assert_eq!(again.reinforcement, 1);
    assert!(again.confidence > first.confidence);
    let m = memory::get(&store, &TenantId::new("ws:test"), &MemoryId::new(first.id))
        .await
        .unwrap()
        .unwrap();
    assert!(
        m.content.contains("changed together 9 times"),
        "{}",
        m.content
    );
    assert!(m.compartment.is_none(), "in the workspace's shared pool");
}

#[tokio::test]
async fn bad_input_is_refused_with_the_choices() {
    let store = Store::connect_memory(EMBED_DIM).await.unwrap();
    let s = server_in(&store, "ws:test", "user:a");
    let refuse = |from: &str, to: &str, source: &str| {
        let p = RecordDependencyParams {
            from: from.into(),
            to: to.into(),
            source: source.into(),
            detail: None,
            provenance: None,
        };
        async {
            s.record_dependency(Parameters(p))
                .await
                .err()
                .map(|e| e.message.to_string())
        }
    };
    assert!(refuse("a", "b", "rumour")
        .await
        .unwrap()
        .contains("declared, observed, learned, claimed"));
    assert!(refuse("a", "A/", "declared")
        .await
        .unwrap()
        .contains("itself"));
    assert!(refuse(" ", "b", "declared").await.unwrap().contains("both"));
    let Err(e) = s
        .blast_radius(Parameters(BlastRadiusParams {
            service: "a".into(),
            direction: Some("sideways".into()),
            depth: None,
            min_weight: None,
        }))
        .await
    else {
        panic!("an unknown direction was accepted");
    };
    assert!(
        e.message.contains("dependents or dependencies"),
        "{}",
        e.message
    );
}

/// One workspace's edges are not another's, even for the same pair of
/// services, and every member of a workspace reads them.
#[tokio::test]
async fn edges_belong_to_their_workspace_and_every_member_reads_them() {
    let store = Store::connect_memory(EMBED_DIM).await.unwrap();
    let a = server_in(&store, "ws:one", "user:a");
    let other_member = server_in(&store, "ws:one", "user:b");
    let elsewhere = server_in(&store, "ws:two", "user:c");
    let mine = record(&a, "web", "orders", "declared", "here").await;
    let theirs = record(&elsewhere, "web", "orders", "declared", "there").await;
    assert_ne!(mine.id, theirs.id);
    assert!(
        theirs.created,
        "a separate row, not a reinforcement of mine"
    );
    assert_eq!(radius(&other_member, "orders", None).await.reached.len(), 1);
    let detail = radius(&elsewhere, "orders", None).await.reached[0].path[0].evidence[0]
        .detail
        .clone();
    assert_eq!(detail.as_deref(), Some("there"));
}

#[tokio::test]
async fn the_rest_dispatcher_reaches_the_graph_tools() {
    let store = Store::connect_memory(EMBED_DIM).await.unwrap();
    let s = server_in(&store, "ws:test", "user:a");
    let recorded = s
        .call_tool(
            "record_dependency",
            serde_json::json!({ "from": "web", "to": "orders", "source": "declared" }),
        )
        .await
        .unwrap();
    assert_eq!(recorded["created"], true);
    let walked = s
        .call_tool("blast_radius", serde_json::json!({ "service": "orders" }))
        .await
        .unwrap();
    assert_eq!(walked["reached"][0]["service"], "web");
}
