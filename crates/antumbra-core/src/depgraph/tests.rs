use super::*;

use chrono::Duration;

use crate::ids::TenantId;
use crate::memory::MemoryNetwork;

/// A claim stored as its memory would be, last seen `days_ago`.
fn edge(from: &str, to: &str, source: Source, days_ago: i64, now: DateTime<Utc>) -> Edge {
    let claim = Claim::new(from, to, source, "test evidence");
    let mut m = Memory::new(
        claim.memory_id(&TenantId::new("ws:t")),
        TenantId::new("ws:t"),
        MemoryNetwork::World,
        claim.content(),
        source.confidence(),
        now - Duration::days(days_ago),
    )
    .with_evidence(claim.evidence(None));
    m.updated_at = now - Duration::days(days_ago);
    Edge::of(&m).expect("a dependency memory reads back as an edge")
}

#[test]
fn a_claim_round_trips_through_its_memory() {
    let now = Utc::now();
    let anchor = GitProvenance::new("github.com/acme/web", "abc1234").at_path("package.json");
    let claim = Claim::new(
        "GitHub.com/Acme/Web.git",
        "github.com/acme/orders/",
        Source::Declared,
        " package.json names @acme/orders-client ",
    );
    assert_eq!(claim.from, "github.com/acme/web");
    assert_eq!(claim.to, "github.com/acme/orders");
    assert_eq!(
        claim.memory_id(&TenantId::new("ws:t")),
        "memory:dep-ws-t-github-com-acme-web--github-com-acme-orders-declared"
    );
    assert_eq!(
        claim.content(),
        "github.com/acme/web depends on github.com/acme/orders (declared dependency): package.json names @acme/orders-client."
    );
    let m = Memory::new(
        claim.memory_id(&TenantId::new("ws:t")),
        TenantId::new("ws:t"),
        MemoryNetwork::World,
        claim.content(),
        0.9,
        now,
    )
    .with_evidence(claim.evidence(Some(&anchor)));
    let e = Edge::of(&m).unwrap();
    assert_eq!(
        (e.from.as_str(), e.to.as_str()),
        ("github.com/acme/web", "github.com/acme/orders")
    );
    assert_eq!(e.source, Source::Declared);
    assert_eq!(
        e.detail.as_deref(),
        Some("package.json names @acme/orders-client")
    );
    assert_eq!(e.anchor.unwrap().path.as_deref(), Some("package.json"));

    let plain = Memory::new(
        "memory:x",
        TenantId::new("ws:t"),
        MemoryNetwork::World,
        "no",
        0.5,
        now,
    );
    assert_eq!(Edge::of(&plain), None, "not a dependency");
}

#[test]
fn each_source_has_its_own_identity_so_they_reinforce_separately() {
    let declared = Claim::new("a", "b", Source::Declared, "");
    let observed = Claim::new("a", "b", Source::Observed, "");
    let t = TenantId::new("ws:t");
    assert_ne!(declared.memory_id(&t), observed.memory_id(&t));
    assert_eq!(
        declared.memory_id(&t),
        Claim::new("A", "B/", Source::Declared, "other").memory_id(&t)
    );
    assert_ne!(
        declared.memory_id(&t),
        declared.memory_id(&TenantId::new("ws:u")),
        "two workspaces never share an edge's row"
    );
    assert_eq!(Source::parse("observed"), Some(Source::Observed));
    assert_eq!(Source::parse("rumour"), None);
}

#[test]
fn an_edge_nobody_sees_again_fades_by_its_source() {
    let now = Utc::now();
    let fresh = edge("a", "b", Source::Observed, 0, now);
    let month = edge("a", "b", Source::Observed, 30, now);
    assert!((fresh.weight(now) - 0.85).abs() < 1e-3);
    assert!((month.weight(now) - 0.425).abs() < 1e-2, "one half-life");
    let declared_month = edge("a", "b", Source::Declared, 30, now);
    assert!(
        declared_month.weight(now) > month.weight(now),
        "a declaration fades slower"
    );
}

/// web -> checkout -> orders, and billing -> orders. Who breaks if orders
/// does: checkout and billing directly, web through checkout.
#[test]
fn dependents_are_reached_through_the_strongest_path_with_evidence() {
    let now = Utc::now();
    let edges = vec![
        edge("web", "checkout", Source::Declared, 0, now),
        edge("checkout", "orders", Source::Declared, 0, now),
        edge("checkout", "orders", Source::Observed, 0, now),
        edge("billing", "orders", Source::Learned, 0, now),
    ];
    let reached = blast_radius(&edges, "orders", Direction::Dependents, 3, 0.3, now);
    let order: Vec<(&str, u32)> = reached
        .iter()
        .map(|r| (r.service.as_str(), r.depth))
        .collect();
    assert_eq!(order, [("checkout", 1), ("web", 2), ("billing", 1)]);
    let checkout = &reached[0];
    // Declared 0.9 and observed 0.85 together: 1 - 0.1 * 0.15.
    assert!(
        (checkout.weight - 0.985).abs() < 1e-3,
        "{}",
        checkout.weight
    );
    assert_eq!(
        checkout.path[0].edges.len(),
        2,
        "both sources' evidence is kept"
    );
    let web = &reached[1];
    assert_eq!(web.path.len(), 2);
    assert_eq!(web.path[1].from, "web");
    assert!((web.weight - 0.985 * 0.9).abs() < 1e-3);

    let needs = blast_radius(&edges, "web", Direction::Dependencies, 3, 0.3, now);
    let needed: Vec<&str> = needs.iter().map(|r| r.service.as_str()).collect();
    assert_eq!(needed, ["checkout", "orders"]);
}

/// A model's claim alone stays under the floor; a declaration of the same
/// edge lifts it, and the claim's evidence comes along.
#[test]
fn a_claim_alone_stays_under_the_floor_until_corroborated() {
    let now = Utc::now();
    let claimed = vec![edge("web", "search", Source::Claimed, 0, now)];
    assert!(blast_radius(&claimed, "search", Direction::Dependents, 3, 0.4, now).is_empty());
    let mut corroborated = claimed.clone();
    corroborated.push(edge("web", "search", Source::Declared, 0, now));
    let reached = blast_radius(&corroborated, "search", Direction::Dependents, 3, 0.4, now);
    assert_eq!(reached.len(), 1);
    let sources: Vec<Source> = reached[0].path[0].edges.iter().map(|e| e.source).collect();
    assert_eq!(
        sources,
        [Source::Declared, Source::Claimed],
        "strongest first"
    );
}

/// Depth bounds the walk, and a cycle neither loops nor reaches back to the
/// start.
#[test]
fn depth_bounds_the_walk_and_cycles_do_not_loop() {
    let now = Utc::now();
    let edges = vec![
        edge("b", "a", Source::Declared, 0, now),
        edge("c", "b", Source::Declared, 0, now),
        edge("a", "c", Source::Declared, 0, now),
        edge("d", "c", Source::Declared, 0, now),
    ];
    let one = blast_radius(&edges, "a", Direction::Dependents, 1, 0.1, now);
    assert_eq!(
        one.iter().map(|r| r.service.as_str()).collect::<Vec<_>>(),
        ["b"]
    );
    let all = blast_radius(&edges, "a", Direction::Dependents, 6, 0.1, now);
    let mut names: Vec<&str> = all.iter().map(|r| r.service.as_str()).collect();
    names.sort_unstable();
    assert_eq!(names, ["b", "c", "d"], "the start is never reached again");
    assert!(blast_radius(&edges, "nowhere", Direction::Dependents, 6, 0.1, now).is_empty());
}
