//! Engine-enforced compartment ACL (Phase 1b): a memory in a private
//! compartment is invisible to another user until its compartment is granted,
//! and hidden again on revoke, enforced by the engine via the memory rule's
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
    memory::upsert(
        &store,
        &mem("memory:priv", "ws:org", "A private", Some("comp:a-private")),
    )
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
        &Grant::new(
            t.clone(),
            ca.clone(),
            ub.clone(),
            Capability::Reference,
            ua.clone(),
            now,
        ),
    )
    .await
    .unwrap();

    // userB now sees A's compartment memory (grant took effect immediately).
    store.signin(&t, &ub).await.unwrap();
    assert!(
        visible_ids(&store)
            .await
            .contains(&"memory:priv".to_string()),
        "grant must make A's compartment visible to B"
    );

    // Revoke → hidden again.
    store.invalidate().await.unwrap();
    compartment::revoke(&store, &t, &ca, &ub, now)
        .await
        .unwrap();
    store.signin(&t, &ub).await.unwrap();
    assert!(
        !visible_ids(&store)
            .await
            .contains(&"memory:priv".to_string()),
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
    compartment::create(
        &store,
        &Compartment::new(ca.clone(), t.clone(), ua.clone(), "A", now),
    )
    .await
    .unwrap();
    memory::upsert(
        &store,
        &mem("memory:priv", "ws:org", "A private", Some("comp:a-private")),
    )
    .await
    .unwrap();

    // The owner sees its compartment memory.
    store.signin(&t, &ua).await.unwrap();
    assert!(visible_ids(&store)
        .await
        .contains(&"memory:priv".to_string()));

    // Delete the compartment (owner mode), then the owner no longer sees it.
    store.invalidate().await.unwrap();
    compartment::delete(&store, &t, &ca, now).await.unwrap();
    store.signin(&t, &ua).await.unwrap();
    assert!(
        !visible_ids(&store)
            .await
            .contains(&"memory:priv".to_string()),
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
    compartment::create(
        &store,
        &Compartment::new(ca.clone(), t.clone(), ua.clone(), "A", now),
    )
    .await
    .unwrap();
    memory::upsert(
        &store,
        &mem("memory:target", "ws:org", "A target", Some("comp:a")),
    )
    .await
    .unwrap();
    memory::upsert(
        &store,
        &mem("memory:source", "ws:org", "shared source", None),
    )
    .await
    .unwrap();

    let e = MemoryEdge::new(
        t.clone(),
        "memory:source",
        "memory:target",
        EdgeType::References,
        1.0,
        now,
    );
    let from = MemoryId::new("memory:source");

    // userB with only REFERENCE can read A's target but may not LINK into it.
    compartment::grant(
        &store,
        &Grant::new(
            t.clone(),
            ca.clone(),
            ub.clone(),
            Capability::Reference,
            ua.clone(),
            now,
        ),
    )
    .await
    .unwrap();
    store.signin(&t, &ub).await.unwrap();
    let _ = edge::relate(&store, &e).await; // engine denies the create (no link)
    assert!(
        edge::neighbors(&store, &t, &from, None)
            .await
            .unwrap()
            .is_empty(),
        "reference grant must not permit linking"
    );

    // Upgrade to LINK -> the edge create now succeeds.
    store.invalidate().await.unwrap();
    compartment::grant(
        &store,
        &Grant::new(
            t.clone(),
            ca.clone(),
            ub.clone(),
            Capability::Link,
            ua.clone(),
            now,
        ),
    )
    .await
    .unwrap();
    store.signin(&t, &ub).await.unwrap();
    edge::relate(&store, &e).await.unwrap();
    assert_eq!(
        edge::neighbors(&store, &t, &from, None)
            .await
            .unwrap()
            .len(),
        1,
        "link grant must permit linking"
    );
}

// Security: a tenant member who does NOT own a compartment cannot forge a grant
// to it. The grant table's create permission requires compartment ownership, so
// B's scoped attempt to share A's private compartment with themselves fails
// closed at the engine -- no grant row is written and B still cannot see A's
// memory. (Without the owner check, any tenant member could grant themselves
// into another user's private compartment, since share_compartment accepts an
// arbitrary compartment id.)
#[tokio::test]
async fn a_non_owner_cannot_forge_a_grant_to_another_users_compartment() {
    let store = Store::connect_memory(4).await.unwrap();
    let t = TenantId::new("ws:org");
    let ua = UserId::new("user:a");
    let ub = UserId::new("user:b");
    let ca = CompartmentId::new("comp:a-private");
    let now = Utc::now();

    principal::provision(&store, &t, &ua).await.unwrap();
    principal::provision(&store, &t, &ub).await.unwrap();
    compartment::create(
        &store,
        &Compartment::new(ca.clone(), t.clone(), ua.clone(), "A", now),
    )
    .await
    .unwrap();
    memory::upsert(
        &store,
        &mem("memory:priv", "ws:org", "A private", Some("comp:a-private")),
    )
    .await
    .unwrap();

    // B (scoped, non-owner) tries to forge a grant to A's compartment.
    store.signin(&t, &ub).await.unwrap();
    let forged = Grant::new(
        t.clone(),
        ca.clone(),
        ub.clone(),
        Capability::Reference,
        ub.clone(),
        now,
    );
    let _ = compartment::grant(&store, &forged).await; // engine refuses; ignore the result

    // The forged grant did not take effect: B still cannot see A's private memory.
    assert!(
        !visible_ids(&store)
            .await
            .contains(&"memory:priv".to_string()),
        "a forged grant must not unlock a non-owned compartment"
    );

    // And no live grant row exists for the compartment (verified in owner mode).
    store.invalidate().await.unwrap();
    assert!(
        compartment::list_grants(&store, &t, &ca)
            .await
            .unwrap()
            .is_empty(),
        "the engine refused to create the forged grant"
    );
}

// The legitimate path the `share_compartment` tool drives in production: the
// OWNER, under their own scoped session, can grant on their own compartment, and
// the grantee then sees it. Proves the owner-only grant rule does not break real
// sharing (the existing visibility test grants in owner/bypass mode).
#[tokio::test]
async fn an_owner_can_share_their_own_compartment_while_scoped() {
    let store = Store::connect_memory(4).await.unwrap();
    let t = TenantId::new("ws:org");
    let ua = UserId::new("user:a");
    let ub = UserId::new("user:b");
    let ca = CompartmentId::new("comp:a-private");
    let now = Utc::now();

    principal::provision(&store, &t, &ua).await.unwrap();
    principal::provision(&store, &t, &ub).await.unwrap();
    compartment::create(
        &store,
        &Compartment::new(ca.clone(), t.clone(), ua.clone(), "A", now),
    )
    .await
    .unwrap();
    memory::upsert(
        &store,
        &mem("memory:priv", "ws:org", "A private", Some("comp:a-private")),
    )
    .await
    .unwrap();

    // A (scoped, the owner) shares the compartment with B -- must succeed.
    store.signin(&t, &ua).await.unwrap();
    compartment::grant(
        &store,
        &Grant::new(
            t.clone(),
            ca.clone(),
            ub.clone(),
            Capability::Reference,
            ua.clone(),
            now,
        ),
    )
    .await
    .unwrap();

    // B now sees A's compartment memory.
    store.signin(&t, &ub).await.unwrap();
    assert!(
        visible_ids(&store)
            .await
            .contains(&"memory:priv".to_string()),
        "the owner can share their own compartment under a scoped session"
    );
}

// Security (the write-side dual of grant-forgery): a tenant member who does NOT
// own a compartment cannot inject a memory into it. The memory CREATE permission
// (MEMORY_WRITE_RULE) requires owning the compartment or a `link` grant, so B's
// scoped write into A's private compartment fails closed -- A never finds a memory
// planted by B in their own private space. The owner's own/shared writes still work.
#[tokio::test]
async fn a_non_owner_cannot_inject_a_memory_into_another_users_compartment() {
    let store = Store::connect_memory(4).await.unwrap();
    let t = TenantId::new("ws:org");
    let ua = UserId::new("user:a");
    let ub = UserId::new("user:b");
    let ca = CompartmentId::new("comp:a-private");
    let now = Utc::now();

    principal::provision(&store, &t, &ua).await.unwrap();
    principal::provision(&store, &t, &ub).await.unwrap();
    compartment::create(
        &store,
        &Compartment::new(ca.clone(), t.clone(), ua.clone(), "A", now),
    )
    .await
    .unwrap();

    // B (scoped, non-owner) tries to plant a memory in A's private compartment.
    store.signin(&t, &ub).await.unwrap();
    let _ = memory::upsert(
        &store,
        &mem(
            "memory:injected",
            "ws:org",
            "planted by B",
            Some("comp:a-private"),
        ),
    )
    .await; // engine denies the create; ignore the result, assert the end-state

    // A (the owner) never sees a planted memory in their own space.
    store.invalidate().await.unwrap();
    store.signin(&t, &ua).await.unwrap();
    assert!(
        !visible_ids(&store)
            .await
            .contains(&"memory:injected".to_string()),
        "a non-owner must not be able to inject into a private compartment"
    );

    // The owner can still write into their own compartment and the shared pool.
    memory::upsert(
        &store,
        &mem(
            "memory:own",
            "ws:org",
            "A owns this",
            Some("comp:a-private"),
        ),
    )
    .await
    .unwrap();
    memory::upsert(&store, &mem("memory:shared", "ws:org", "shared pool", None))
        .await
        .unwrap();
    let seen = visible_ids(&store).await;
    assert!(
        seen.contains(&"memory:own".to_string()) && seen.contains(&"memory:shared".to_string()),
        "the owner writes its own compartment + the shared pool"
    );
}

#[tokio::test]
async fn compartments_of_one_name_are_listed_across_workspaces_and_owners() {
    let store = Store::connect_memory(4).await.unwrap();
    let now = Utc::now();
    for (id, tenant, owner, name) in [
        ("comp:ws:a:user:a:behaviour", "ws:a", "user:a", "behaviour"),
        ("comp:ws:b:user:b:behaviour", "ws:b", "user:b", "behaviour"),
        ("comp:ws:a:user:a:notes", "ws:a", "user:a", "notes"),
        ("comp:ws:c:user:c:behaviour", "ws:c", "user:c", "behaviour"),
    ] {
        compartment::create(
            &store,
            &Compartment::new(
                CompartmentId::new(id),
                TenantId::new(tenant),
                UserId::new(owner),
                name,
                now,
            ),
        )
        .await
        .unwrap();
    }
    compartment::delete(
        &store,
        &TenantId::new("ws:c"),
        &CompartmentId::new("comp:ws:c:user:c:behaviour"),
        now,
    )
    .await
    .unwrap();
    let mut ids: Vec<String> = compartment::list_named(&store, "behaviour")
        .await
        .unwrap()
        .into_iter()
        .map(|c| c.id.as_str().to_string())
        .collect();
    ids.sort();
    assert_eq!(
        ids,
        ["comp:ws:a:user:a:behaviour", "comp:ws:b:user:b:behaviour"]
    );
}
