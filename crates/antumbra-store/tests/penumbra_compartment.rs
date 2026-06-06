//! Engine-enforced compartment ACL (Phase 1b): a memory in a private
//! compartment is invisible to another user until its compartment is granted,
//! and hidden again on revoke — enforced by the engine via the memory rule's
//! grant-graph subquery, on an unfiltered `SELECT` (no app-side WHERE).

use chrono::Utc;

use antumbra_core::{
    Capability, Compartment, CompartmentId, EdgeType, Grant, Memory, MemoryEdge, MemoryId,
    MemoryNetwork, TenantId, UserId,
};
use antumbra_store::repo::{compartment, edge, memory, principal};
use antumbra_store::Store;

fn mem(id: &str, tenant: &str, content: &str, comp: Option<&str>) -> Memory {
    let m = Memory::new(id, tenant, MemoryNetwork::World, content, 0.8, Utc::now());
    match comp {
        Some(c) => m.in_compartment(c),
        None => m,
    }
}

async fn visible_ids(store: &Store) -> Vec<String> {
    memory::all_unscoped(store)
        .await
        .unwrap()
        .into_iter()
        .map(|m| m.id.as_str().to_string())
        .collect()
}

#[tokio::test]
async fn memory_visibility_follows_compartment_grants() {
    let store = Store::connect_memory(4).await.unwrap();
    let t = TenantId::new("ws:org");
    let ua = UserId::new("user:a");
    let ub = UserId::new("user:b");
    let ca = CompartmentId::new("comp:a-private");
    let now = Utc::now();

    // Owner provisions both users, creates A's private compartment, and seeds a
    // private memory in it plus an un-compartmentalized (shared-pool) memory.
    principal::provision(&store, &t, &ua).await.unwrap();
    principal::provision(&store, &t, &ub).await.unwrap();
    compartment::create(
        &store,
        &Compartment::new(ca.clone(), t.clone(), ua.clone(), "A private", now),
    )
    .await
    .unwrap();
    memory::upsert(&store, &mem("memory:priv", "ws:org", "A private", Some("comp:a-private")))
        .await
        .unwrap();
    memory::upsert(&store, &mem("memory:pub", "ws:org", "shared pool", None))
        .await
        .unwrap();

    // userB sees the shared pool but NOT A's private compartment.
    store.signin(&t, &ub).await.unwrap();
    let seen = visible_ids(&store).await;
    assert!(seen.contains(&"memory:pub".to_string()));
    assert!(
        !seen.contains(&"memory:priv".to_string()),
        "userB must not see A's private compartment"
    );

    // Owner grants Reference on the compartment to userB.
    store.invalidate().await.unwrap();
    compartment::grant(
        &store,
        &Grant::new(t.clone(), ca.clone(), ub.clone(), Capability::Reference, ua.clone(), now),
    )
    .await
    .unwrap();

    // userB now sees A's compartment memory (grant took effect immediately).
    store.signin(&t, &ub).await.unwrap();
    assert!(
        visible_ids(&store).await.contains(&"memory:priv".to_string()),
        "grant must make A's compartment visible to B"
    );

    // Revoke → hidden again.
    store.invalidate().await.unwrap();
    compartment::revoke(&store, &t, &ca, &ub, now).await.unwrap();
    store.signin(&t, &ub).await.unwrap();
    assert!(
        !visible_ids(&store).await.contains(&"memory:priv".to_string()),
        "revoke must hide it again"
    );

    // userA always sees their own compartment plus the shared pool.
    store.invalidate().await.unwrap();
    store.signin(&t, &ua).await.unwrap();
    let seen = visible_ids(&store).await;
    assert!(seen.contains(&"memory:priv".to_string()) && seen.contains(&"memory:pub".to_string()));
}

// Deleting a compartment is a tombstone: the engine's owner subquery excludes it
// (`deleted_at IS NONE`), so its memories become invisible to the owner at once --
// and the deletion is a row that propagates rather than orphaning silently.
#[tokio::test]
async fn deleting_a_compartment_hides_its_memories_from_the_owner() {
    let store = Store::connect_memory(4).await.unwrap();
    let t = TenantId::new("ws:org");
    let ua = UserId::new("user:a");
    let ca = CompartmentId::new("comp:a-private");
    let now = Utc::now();

    principal::provision(&store, &t, &ua).await.unwrap();
    compartment::create(&store, &Compartment::new(ca.clone(), t.clone(), ua.clone(), "A", now))
        .await
        .unwrap();
    memory::upsert(&store, &mem("memory:priv", "ws:org", "A private", Some("comp:a-private")))
        .await
        .unwrap();

    // The owner sees its compartment memory.
    store.signin(&t, &ua).await.unwrap();
    assert!(visible_ids(&store).await.contains(&"memory:priv".to_string()));

    // Delete the compartment (owner mode), then the owner no longer sees it.
    store.invalidate().await.unwrap();
    compartment::delete(&store, &t, &ca, now).await.unwrap();
    store.signin(&t, &ua).await.unwrap();
    assert!(
        !visible_ids(&store).await.contains(&"memory:priv".to_string()),
        "a deleted compartment hides its memories from the owner"
    );
}

#[tokio::test]
async fn linking_into_a_compartment_requires_link_capability() {
    let store = Store::connect_memory(4).await.unwrap();
    let t = TenantId::new("ws:org");
    let ua = UserId::new("user:a");
    let ub = UserId::new("user:b");
    let ca = CompartmentId::new("comp:a");
    let now = Utc::now();

    principal::provision(&store, &t, &ua).await.unwrap();
    principal::provision(&store, &t, &ub).await.unwrap();
    compartment::create(&store, &Compartment::new(ca.clone(), t.clone(), ua.clone(), "A", now))
        .await
        .unwrap();
    memory::upsert(&store, &mem("memory:target", "ws:org", "A target", Some("comp:a")))
        .await
        .unwrap();
    memory::upsert(&store, &mem("memory:source", "ws:org", "shared source", None))
        .await
        .unwrap();

    let e = MemoryEdge::new(t.clone(), "memory:source", "memory:target", EdgeType::References, 1.0, now);
    let from = MemoryId::new("memory:source");

    // userB with only REFERENCE can read A's target but may not LINK into it.
    compartment::grant(
        &store,
        &Grant::new(t.clone(), ca.clone(), ub.clone(), Capability::Reference, ua.clone(), now),
    )
    .await
    .unwrap();
    store.signin(&t, &ub).await.unwrap();
    let _ = edge::relate(&store, &e).await; // engine denies the create (no link)
    assert!(
        edge::neighbors(&store, &t, &from, None).await.unwrap().is_empty(),
        "reference grant must not permit linking"
    );

    // Upgrade to LINK -> the edge create now succeeds.
    store.invalidate().await.unwrap();
    compartment::grant(
        &store,
        &Grant::new(t.clone(), ca.clone(), ub.clone(), Capability::Link, ua.clone(), now),
    )
    .await
    .unwrap();
    store.signin(&t, &ub).await.unwrap();
    edge::relate(&store, &e).await.unwrap();
    assert_eq!(
        edge::neighbors(&store, &t, &from, None).await.unwrap().len(),
        1,
        "link grant must permit linking"
    );
}
