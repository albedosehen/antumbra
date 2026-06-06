//! Penumbra graph: tenant-isolated typed edges between memories.

use chrono::Utc;

use antumbra_core::{EdgeType, MemoryEdge, MemoryId, TenantId};
use antumbra_store::repo::edge;
use antumbra_store::Store;

#[tokio::test]
async fn edges_relate_and_neighbors_are_tenant_scoped() {
    let store = Store::connect_memory(4).await.unwrap();
    let alpha = TenantId::new("ws:alpha");
    let beta = TenantId::new("ws:beta");
    let now = Utc::now();

    // alpha: a -> b (supersedes) and a -> c (references)
    edge::relate(
        &store,
        &MemoryEdge::new(
            alpha.clone(),
            "memory:a",
            "memory:b",
            EdgeType::Supersedes,
            1.0,
            now,
        ),
    )
    .await
    .unwrap();
    edge::relate(
        &store,
        &MemoryEdge::new(
            alpha.clone(),
            "memory:a",
            "memory:c",
            EdgeType::References,
            0.5,
            now,
        ),
    )
    .await
    .unwrap();
    // beta: a -> z, same `from` key but a different tenant.
    edge::relate(
        &store,
        &MemoryEdge::new(
            beta.clone(),
            "memory:a",
            "memory:z",
            EdgeType::References,
            1.0,
            now,
        ),
    )
    .await
    .unwrap();

    // alpha sees only its two edges from a; beta's a->z never leaks.
    let from_a = MemoryId::new("memory:a");
    let n = edge::neighbors(&store, &alpha, &from_a, None)
        .await
        .unwrap();
    assert_eq!(n.len(), 2, "cross-tenant edge leak");
    assert!(n.iter().all(|e| e.tenant == alpha));
    assert!(n.iter().any(|e| e.to_id == MemoryId::new("memory:b")));

    // edge-type filter narrows to the supersedes edge.
    let sup = edge::neighbors(&store, &alpha, &from_a, Some(EdgeType::Supersedes))
        .await
        .unwrap();
    assert_eq!(sup.len(), 1);
    assert_eq!(sup[0].to_id, MemoryId::new("memory:b"));

    // beta sees only its own edge.
    let nb = edge::neighbors(&store, &beta, &from_a, None).await.unwrap();
    assert_eq!(nb.len(), 1);
    assert_eq!(nb[0].to_id, MemoryId::new("memory:z"));

    // Re-relating the same pair/type is idempotent (no duplicate).
    edge::relate(
        &store,
        &MemoryEdge::new(
            alpha.clone(),
            "memory:a",
            "memory:b",
            EdgeType::Supersedes,
            0.9,
            now,
        ),
    )
    .await
    .unwrap();
    assert_eq!(
        edge::neighbors(&store, &alpha, &from_a, None)
            .await
            .unwrap()
            .len(),
        2
    );
}
