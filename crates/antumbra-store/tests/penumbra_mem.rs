//! Penumbra memory tenant-isolation regression (mirrors the data-plane POC's
//! `test_regression_tenant_isolation`): no workspace may recall, list, read by
//! id, or delete another workspace's traces, even when the other tenant's
//! trace is the nearest vector match.

use chrono::Utc;

use antumbra_core::{CompartmentId, ExpertId, Memory, MemoryId, MemoryNetwork, TenantId};
use antumbra_store::repo::memory;
use antumbra_store::Store;

#[tokio::test]
async fn list_by_compartment_gathers_only_that_compartment() {
    let store = Store::connect_memory(4).await.unwrap();
    let ws = TenantId::new("ws:a");
    let mk = |id: &str, comp: Option<&str>| {
        let m = Memory::new(id, "ws:a", MemoryNetwork::World, "c", 0.8, Utc::now())
            .with_embedding(vec![1.0, 0.0, 0.0, 0.0]);
        match comp {
            Some(c) => m.in_compartment(c),
            None => m,
        }
    };
    memory::upsert(&store, &mk("memory:1", Some("comp:x")))
        .await
        .unwrap();
    memory::upsert(&store, &mk("memory:2", Some("comp:y")))
        .await
        .unwrap();
    memory::upsert(&store, &mk("memory:3", None)).await.unwrap();

    let got = memory::list_by_compartment(&store, &ws, &CompartmentId::new("comp:x"))
        .await
        .unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].id, MemoryId::new("memory:1"));
}

fn trace(id: &str, ws: &str, net: MemoryNetwork, content: &str, embed: Vec<f32>) -> Memory {
    Memory::new(id, ws, net, content, 0.8, Utc::now()).with_embedding(embed)
}

#[tokio::test]
async fn list_and_recall_are_tenant_scoped() {
    let store = Store::connect_memory(4).await.unwrap();
    let alpha = TenantId::new("ws:alpha");
    let beta = TenantId::new("ws:beta");

    // Alpha owns one trace; beta owns a trace whose embedding is the CLOSEST to
    // the query; if isolation leaked, alpha's recall would surface beta's.
    memory::upsert(
        &store,
        &trace(
            "memory:a1",
            "ws:alpha",
            MemoryNetwork::World,
            "alpha fact",
            vec![1.0, 0.0, 0.0, 0.0],
        ),
    )
    .await
    .unwrap();
    memory::upsert(
        &store,
        &trace(
            "memory:b1",
            "ws:beta",
            MemoryNetwork::World,
            "beta secret",
            vec![0.95, 0.05, 0.0, 0.0],
        ),
    )
    .await
    .unwrap();

    // list is tenant-filtered.
    assert_eq!(memory::list(&store, &alpha).await.unwrap().len(), 1);
    assert_eq!(memory::list(&store, &beta).await.unwrap().len(), 1);

    // recall for alpha, with a query nearest to beta's trace, must return ONLY
    // alpha's trace; beta's closer vector is never a candidate.
    let hits = memory::recall(&store, &alpha, &[0.95, 0.05, 0.0, 0.0], 5, None)
        .await
        .unwrap();
    assert_eq!(hits.len(), 1, "cross-tenant recall leak");
    assert_eq!(hits[0].id, MemoryId::new("memory:a1"));
    assert_eq!(hits[0].content, "alpha fact");
}

#[tokio::test]
async fn get_by_id_is_tenant_checked() {
    let store = Store::connect_memory(4).await.unwrap();
    let alpha = TenantId::new("ws:alpha");
    let beta = TenantId::new("ws:beta");
    memory::upsert(
        &store,
        &trace(
            "memory:owned",
            "ws:beta",
            MemoryNetwork::Bank,
            "beta only",
            vec![0.0, 1.0, 0.0, 0.0],
        ),
    )
    .await
    .unwrap();

    let id = MemoryId::new("memory:owned");
    // The owner sees it; another tenant does not, even with the exact id.
    assert!(memory::get(&store, &beta, &id).await.unwrap().is_some());
    assert!(
        memory::get(&store, &alpha, &id).await.unwrap().is_none(),
        "cross-tenant read leak"
    );
}

#[tokio::test]
async fn delete_cannot_cross_tenants() {
    let store = Store::connect_memory(4).await.unwrap();
    let alpha = TenantId::new("ws:alpha");
    let beta = TenantId::new("ws:beta");
    let id = MemoryId::new("memory:keep");
    memory::upsert(
        &store,
        &trace(
            "memory:keep",
            "ws:beta",
            MemoryNetwork::World,
            "beta keeps this",
            vec![1.0, 0.0, 0.0, 0.0],
        ),
    )
    .await
    .unwrap();

    // Alpha's delete of beta's id is a no-op; beta still has it.
    memory::delete(&store, &alpha, &id).await.unwrap();
    assert!(memory::get(&store, &beta, &id).await.unwrap().is_some());

    // The owner can delete it.
    memory::delete(&store, &beta, &id).await.unwrap();
    assert!(memory::get(&store, &beta, &id).await.unwrap().is_none());
}

#[tokio::test]
async fn recall_filters_by_network() {
    let store = Store::connect_memory(4).await.unwrap();
    let ws = TenantId::new("ws:alpha");
    memory::upsert(
        &store,
        &trace(
            "memory:w",
            "ws:alpha",
            MemoryNetwork::World,
            "a world fact",
            vec![1.0, 0.0, 0.0, 0.0],
        ),
    )
    .await
    .unwrap();
    memory::upsert(
        &store,
        &trace(
            "memory:o",
            "ws:alpha",
            MemoryNetwork::Opinion,
            "an opinion",
            vec![1.0, 0.0, 0.0, 0.0],
        ),
    )
    .await
    .unwrap();

    let world = memory::recall(
        &store,
        &ws,
        &[1.0, 0.0, 0.0, 0.0],
        5,
        Some(MemoryNetwork::World),
    )
    .await
    .unwrap();
    assert_eq!(world.len(), 1);
    assert_eq!(world[0].network, MemoryNetwork::World);
}

#[tokio::test]
async fn reinforce_and_consolidate_persist() {
    let store = Store::connect_memory(4).await.unwrap();
    let ws = TenantId::new("ws:alpha");
    let id = MemoryId::new("memory:r");
    memory::upsert(
        &store,
        &trace(
            "memory:r",
            "ws:alpha",
            MemoryNetwork::World,
            "reinforce me",
            vec![1.0, 0.0, 0.0, 0.0],
        ),
    )
    .await
    .unwrap();

    let reinforced = memory::reinforce(&store, &ws, &id, Utc::now())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(reinforced.reinforcement, 1);
    assert!(reinforced.confidence > 0.8);

    let consolidated = memory::mark_consolidated(
        &store,
        &ws,
        &id,
        ExpertId::new("expert:consolidated-world"),
        Utc::now(),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(consolidated.is_consolidated());

    // Persisted: reload reflects both the reinforcement and the graduation link.
    let reloaded = memory::get(&store, &ws, &id).await.unwrap().unwrap();
    assert_eq!(reloaded.reinforcement, 1);
    assert_eq!(
        reloaded.consolidated_expert,
        Some(ExpertId::new("expert:consolidated-world"))
    );
}
