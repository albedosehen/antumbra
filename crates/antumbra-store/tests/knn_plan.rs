//! A residual filter thins what the index returned, so a recall that
//! asked for exactly `k` would come back short.
//!
//! This is the trap the index-backed KNN operator brings with it. The
//! exhaustive form these paths used to render compared every row, so a
//! `WHERE` beside it narrowed the population first and `k` meant k
//! matching rows. The graph walk knows nothing about tenants: it hands
//! back its nearest neighbours across the whole table and every other
//! predicate applies afterwards. Over-fetching and truncating is what
//! keeps the change from being a correctness regression dressed as a
//! speed-up.
//!
//! The plan half of the proof — that these paths reach their indexes
//! at all — is a unit test in `store.rs`, where the client is
//! reachable.

use chrono::Utc;

use antumbra_core::{Memory, MemoryId, MemoryNetwork, TenantId};
use antumbra_store::repo::memory;
use antumbra_store::{Store, EMBED_DIM};

fn unit(axis: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; EMBED_DIM];
    v[axis] = 1.0;
    v
}

fn trace(id: &str, tenant: &TenantId, embedding: Vec<f32>) -> Memory {
    Memory {
        id: MemoryId::new(id),
        tenant: tenant.clone(),
        network: MemoryNetwork::World,
        content: id.to_string(),
        embedding: Some(embedding),
        confidence: 1.0,
        reinforcement: 0,
        evidence: vec![],
        volatile: false,
        consolidated_expert: None,
        compartment: None,
        author: None,
        author_host: None,
        status: Default::default(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        deleted_at: None,
    }
}

/// One tenant holds a single memory among many, and its own trace is
/// nobody's nearest neighbour until every closer row has been filtered
/// away. Asking the index for `k = 1` directly would find nothing.
#[tokio::test]
async fn a_tenant_with_a_thin_share_still_recalls() {
    let store = Store::connect_memory(EMBED_DIM).await.unwrap();
    let crowd = TenantId::new("tenant:crowd");
    let thin = TenantId::new("tenant:thin");

    for n in 0..30 {
        memory::upsert(
            &store,
            &trace(&format!("memory:crowd-{n}"), &crowd, unit(0)),
        )
        .await
        .unwrap();
    }
    memory::upsert(&store, &trace("memory:thin-1", &thin, unit(1)))
        .await
        .unwrap();

    let found = memory::recall(&store, &thin, &unit(0), 1, None)
        .await
        .unwrap();
    assert_eq!(
        found.len(),
        1,
        "the over-fetch is what keeps a thin tenant's recall from coming back empty",
    );
    assert_eq!(found[0].id.as_str(), "memory:thin-1");

    // And the crowd gets what it asked for, truncated rather than
    // handed the whole pool.
    let crowded = memory::recall(&store, &crowd, &unit(0), 3, None)
        .await
        .unwrap();
    assert_eq!(crowded.len(), 3, "the pool is truncated back to k");
}

/// Tombstones are the second residual, and they thin the pool after
/// the tenant filter has already thinned it.
#[tokio::test]
async fn tombstones_do_not_eat_a_recall() {
    let store = Store::connect_memory(EMBED_DIM).await.unwrap();
    let tenant = TenantId::new("tenant:t");

    // Ten forgotten traces sitting exactly on the query axis, and one
    // live trace slightly off it: the nearest ten are all tombstones.
    for n in 0..10 {
        let mut dead = trace(&format!("memory:dead-{n}"), &tenant, unit(0));
        dead.deleted_at = Some(Utc::now());
        memory::upsert(&store, &dead).await.unwrap();
    }
    let mut near = unit(0);
    near[1] = 0.5;
    memory::upsert(&store, &trace("memory:live", &tenant, near))
        .await
        .unwrap();

    let found = memory::recall(&store, &tenant, &unit(0), 1, None)
        .await
        .unwrap();
    assert_eq!(found.len(), 1, "the live trace is behind ten tombstones");
    assert_eq!(found[0].id.as_str(), "memory:live");
}
