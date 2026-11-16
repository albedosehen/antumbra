//! Engine-enforced compartment ACL (Phase 1b): a memory in a private
//! compartment is invisible to another user until its compartment is granted,
//! and hidden again on revoke, enforced by the engine via the memory rule's
//! grant-graph subquery, on an unfiltered `SELECT` (no app-side WHERE).

use anyhow::Result;
use chrono::Utc;

use antumbra_core::{
    Capability, Compartment, CompartmentId, EdgeType, Grant, Memory, MemoryEdge, MemoryId,
    MemoryNetwork, TenantId, UserId,
};
use antumbra_store::repo::{compartment, edge, memory, principal};
use antumbra_store::{testing, Store};

fn mem(id: &str, tenant: &str, content: &str, comp: Option<&str>) -> Memory {
    let m = Memory::new(id, tenant, MemoryNetwork::World, content, 0.8, Utc::now());
    match comp {
        Some(c) => m.in_compartment(c),
        None => m,
    }
}

async fn visible_ids(store: &Store) -> Result<Vec<String>> {
    Ok(memory::all_unscoped(store)
        .await?
        .into_iter()
        .map(|m| m.id.as_str().to_string())
        .collect())
}

#[tokio::test]
async fn memory_visibility_follows_compartment_grants() -> Result<()> {
    let store = testing::mem_store_of(4).await?;
    let t = TenantId::new("ws:org");
    let ua = UserId::new("user:a");
    let ub = UserId::new("user:b");
    let ca = CompartmentId::new("comp:a-private");
    let now = Utc::now();

    // Owner provisions both users, creates A's private compartment, and seeds a
    // private memory in it plus an un-compartmentalized (shared-pool) memory.
    testing::members(&store, &t, &[&ua, &ub]).await?;
    compartment::create(
        &store,
        &Compartment::new(ca.clone(), t.clone(), ua.clone(), "A private", now),
    )
    .await?;
    memory::upsert(
        &store,
        &mem("memory:priv", "ws:org", "A private", Some("comp:a-private")),
    )
    .await?;
    memory::upsert(&store, &mem("memory:pub", "ws:org", "shared pool", None)).await?;

    // userB sees the shared pool but NOT A's private compartment.
    store.signin(&t, &ub).await?;
    let seen = visible_ids(&store).await?;
    assert!(seen.contains(&"memory:pub".to_string()));
    assert!(
        !seen.contains(&"memory:priv".to_string()),
        "userB must not see A's private compartment"
    );

    // Owner grants Reference on the compartment to userB.
    store.invalidate().await?;
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
    .await?;

    // userB now sees A's compartment memory (grant took effect immediately).
    store.signin(&t, &ub).await?;
    assert!(
        visible_ids(&store)
            .await?
            .contains(&"memory:priv".to_string()),
        "grant must make A's compartment visible to B"
    );

    // Revoke → hidden again.
    store.invalidate().await?;
    compartment::revoke(&store, &t, &ca, &ub, now).await?;
    store.signin(&t, &ub).await?;
    assert!(
        !visible_ids(&store)
            .await?
            .contains(&"memory:priv".to_string()),
        "revoke must hide it again"
    );

    // userA always sees their own compartment plus the shared pool.
    store.invalidate().await?;
    store.signin(&t, &ua).await?;
    let seen = visible_ids(&store).await?;
    assert!(seen.contains(&"memory:priv".to_string()) && seen.contains(&"memory:pub".to_string()));
    Ok(())
}

// Deleting a compartment is a tombstone: the engine's owner subquery excludes it
// (`deleted_at IS NONE`), so its memories become invisible to the owner at once --
// and the deletion is a row that propagates rather than orphaning silently.
#[tokio::test]
async fn deleting_a_compartment_hides_its_memories_from_the_owner() -> Result<()> {
    let store = testing::mem_store_of(4).await?;
    let t = TenantId::new("ws:org");
    let ua = UserId::new("user:a");
    let ca = CompartmentId::new("comp:a-private");
    let now = Utc::now();

    principal::provision(&store, &t, &ua).await?;
    compartment::create(
        &store,
        &Compartment::new(ca.clone(), t.clone(), ua.clone(), "A", now),
    )
    .await?;
    memory::upsert(
        &store,
        &mem("memory:priv", "ws:org", "A private", Some("comp:a-private")),
    )
    .await?;

    // The owner sees its compartment memory.
    store.signin(&t, &ua).await?;
    assert!(visible_ids(&store)
        .await?
        .contains(&"memory:priv".to_string()));

    // Delete the compartment (owner mode), then the owner no longer sees it.
    store.invalidate().await?;
    compartment::delete(&store, &t, &ca, now).await?;
    store.signin(&t, &ua).await?;
    assert!(
        !visible_ids(&store)
            .await?
            .contains(&"memory:priv".to_string()),
        "a deleted compartment hides its memories from the owner"
    );
    Ok(())
}

#[tokio::test]
async fn linking_into_a_compartment_requires_link_capability() -> Result<()> {
    let store = testing::mem_store_of(4).await?;
    let t = TenantId::new("ws:org");
    let ua = UserId::new("user:a");
    let ub = UserId::new("user:b");
    let ca = CompartmentId::new("comp:a");
    let now = Utc::now();

    testing::members(&store, &t, &[&ua, &ub]).await?;
    compartment::create(
        &store,
        &Compartment::new(ca.clone(), t.clone(), ua.clone(), "A", now),
    )
    .await?;
    memory::upsert(
        &store,
        &mem("memory:target", "ws:org", "A target", Some("comp:a")),
    )
    .await?;
    memory::upsert(
        &store,
        &mem("memory:source", "ws:org", "shared source", None),
    )
    .await?;

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
    .await?;
    store.signin(&t, &ub).await?;
    let _ = edge::relate(&store, &e).await; // engine denies the create (no link)
    assert!(
        edge::neighbors(&store, &t, &from, None).await?.is_empty(),
        "reference grant must not permit linking"
    );

    // Upgrade to LINK -> the edge create now succeeds.
    store.invalidate().await?;
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
    .await?;
    store.signin(&t, &ub).await?;
    edge::relate(&store, &e).await?;
    assert_eq!(
        edge::neighbors(&store, &t, &from, None).await?.len(),
        1,
        "link grant must permit linking"
    );
    Ok(())
}

// Security: a tenant member who does NOT own a compartment cannot forge a grant
// to it. The grant table's create permission requires compartment ownership, so
// B's scoped attempt to share A's private compartment with themselves fails
// closed at the engine -- no grant row is written and B still cannot see A's
// memory. (Without the owner check, any tenant member could grant themselves
// into another user's private compartment, since share_compartment accepts an
// arbitrary compartment id.)
#[tokio::test]
async fn a_non_owner_cannot_forge_a_grant_to_another_users_compartment() -> Result<()> {
    let store = testing::mem_store_of(4).await?;
    let t = TenantId::new("ws:org");
    let ua = UserId::new("user:a");
    let ub = UserId::new("user:b");
    let ca = CompartmentId::new("comp:a-private");
    let now = Utc::now();

    testing::members(&store, &t, &[&ua, &ub]).await?;
    compartment::create(
        &store,
        &Compartment::new(ca.clone(), t.clone(), ua.clone(), "A", now),
    )
    .await?;
    memory::upsert(
        &store,
        &mem("memory:priv", "ws:org", "A private", Some("comp:a-private")),
    )
    .await?;

    // B (scoped, non-owner) tries to forge a grant to A's compartment.
    store.signin(&t, &ub).await?;
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
            .await?
            .contains(&"memory:priv".to_string()),
        "a forged grant must not unlock a non-owned compartment"
    );

    // And no live grant row exists for the compartment (verified in owner mode).
    store.invalidate().await?;
    assert!(
        compartment::list_grants(&store, &t, &ca).await?.is_empty(),
        "the engine refused to create the forged grant"
    );
    Ok(())
}

// The legitimate path the `share_compartment` tool drives in production: the
// OWNER, under their own scoped session, can grant on their own compartment, and
// the grantee then sees it. Proves the owner-only grant rule does not break real
// sharing (the existing visibility test grants in owner/bypass mode).
#[tokio::test]
async fn an_owner_can_share_their_own_compartment_while_scoped() -> Result<()> {
    let store = testing::mem_store_of(4).await?;
    let t = TenantId::new("ws:org");
    let ua = UserId::new("user:a");
    let ub = UserId::new("user:b");
    let ca = CompartmentId::new("comp:a-private");
    let now = Utc::now();

    testing::members(&store, &t, &[&ua, &ub]).await?;
    compartment::create(
        &store,
        &Compartment::new(ca.clone(), t.clone(), ua.clone(), "A", now),
    )
    .await?;
    memory::upsert(
        &store,
        &mem("memory:priv", "ws:org", "A private", Some("comp:a-private")),
    )
    .await?;

    // A (scoped, the owner) shares the compartment with B -- must succeed.
    store.signin(&t, &ua).await?;
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
    .await?;

    // B now sees A's compartment memory.
    store.signin(&t, &ub).await?;
    assert!(
        visible_ids(&store)
            .await?
            .contains(&"memory:priv".to_string()),
        "the owner can share their own compartment under a scoped session"
    );
    Ok(())
}

// Security (the write-side dual of grant-forgery): a tenant member who does NOT
// own a compartment cannot inject a memory into it. The memory CREATE permission
// (MEMORY_WRITE_RULE) requires owning the compartment or a `link` grant, so B's
// scoped write into A's private compartment fails closed -- A never finds a memory
// planted by B in their own private space. The owner's own/shared writes still work.
#[tokio::test]
async fn a_non_owner_cannot_inject_a_memory_into_another_users_compartment() -> Result<()> {
    let store = testing::mem_store_of(4).await?;
    let t = TenantId::new("ws:org");
    let ua = UserId::new("user:a");
    let ub = UserId::new("user:b");
    let ca = CompartmentId::new("comp:a-private");
    let now = Utc::now();

    testing::members(&store, &t, &[&ua, &ub]).await?;
    compartment::create(
        &store,
        &Compartment::new(ca.clone(), t.clone(), ua.clone(), "A", now),
    )
    .await?;

    // B (scoped, non-owner) tries to plant a memory in A's private compartment.
    store.signin(&t, &ub).await?;
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
    store.invalidate().await?;
    store.signin(&t, &ua).await?;
    assert!(
        !visible_ids(&store)
            .await?
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
    .await?;
    memory::upsert(&store, &mem("memory:shared", "ws:org", "shared pool", None)).await?;
    let seen = visible_ids(&store).await?;
    assert!(
        seen.contains(&"memory:own".to_string()) && seen.contains(&"memory:shared".to_string()),
        "the owner writes its own compartment + the shared pool"
    );
    Ok(())
}

#[tokio::test]
async fn compartments_of_one_name_are_listed_across_workspaces_and_owners() -> Result<()> {
    let store = testing::mem_store_of(4).await?;
    let now = Utc::now();
    for (id, tenant, owner, name) in [
        ("comp:ws:a:user:a:behavior", "ws:a", "user:a", "behavior"),
        ("comp:ws:b:user:b:behavior", "ws:b", "user:b", "behavior"),
        ("comp:ws:a:user:a:notes", "ws:a", "user:a", "notes"),
        ("comp:ws:c:user:c:behavior", "ws:c", "user:c", "behavior"),
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
        .await?;
    }
    compartment::delete(
        &store,
        &TenantId::new("ws:c"),
        &CompartmentId::new("comp:ws:c:user:c:behavior"),
        now,
    )
    .await?;
    let mut ids: Vec<String> = compartment::list_named(&store, "behavior")
        .await?
        .into_iter()
        .map(|c| c.id.as_str().to_string())
        .collect();
    ids.sort();
    assert_eq!(
        ids,
        ["comp:ws:a:user:a:behavior", "comp:ws:b:user:b:behavior"]
    );
    Ok(())
}

/// The read rule finds a memory's own compartment by its record id, the key
/// with `:`, `/`, `\`, `|` and spaces mapped to `_`. A key holding every one of
/// them still decides who sees the memory: its owner, a grantee until revoked,
/// nobody else, and not the owner either once the compartment is deleted.
#[tokio::test]
async fn the_owner_branch_holds_for_a_key_with_every_mapped_character() -> Result<()> {
    let store = testing::mem_store_of(4).await?;
    let t = TenantId::new("ws:org");
    let owner = UserId::new("user:owner");
    let other = UserId::new("user:other");
    let key = CompartmentId::new(r"comp:a/b\c|d e");
    let now = Utc::now();
    testing::members(&store, &t, &[&owner, &other]).await?;
    compartment::create(
        &store,
        &Compartment::new(key.clone(), t.clone(), owner.clone(), "odd key", now),
    )
    .await?;
    memory::upsert(
        &store,
        &mem(
            "memory:odd",
            "ws:org",
            "in the odd compartment",
            Some(key.as_str()),
        ),
    )
    .await?;
    let sees = |user: UserId| {
        let (store, t) = (&store, &t);
        async move {
            store.invalidate().await?;
            store.signin(t, &user).await?;
            Ok::<bool, anyhow::Error>(
                visible_ids(store)
                    .await?
                    .contains(&"memory:odd".to_string()),
            )
        }
    };

    assert!(
        sees(owner.clone()).await?,
        "the owner sees their own compartment"
    );
    assert!(!sees(other.clone()).await?, "another user does not");

    store.invalidate().await?;
    compartment::grant(
        &store,
        &Grant::new(
            t.clone(),
            key.clone(),
            other.clone(),
            Capability::Reference,
            owner.clone(),
            now,
        ),
    )
    .await?;
    assert!(
        sees(other.clone()).await?,
        "a grant shows it to the grantee"
    );

    store.invalidate().await?;
    compartment::revoke(&store, &t, &key, &other, now).await?;
    assert!(!sees(other.clone()).await?, "a revoke hides it again");

    store.invalidate().await?;
    compartment::delete(&store, &t, &key, now).await?;
    assert!(
        !sees(owner.clone()).await?,
        "a deleted compartment no longer shows its memories to its owner"
    );
    Ok(())
}

// A grant names its workspace, and the rule's subquery asks for the workspace
// too: a grant row in another workspace, for the same user and the same
// compartment key, opens nothing here, whatever the engine does with the grant
// table's own permission inside a permission subquery.
#[tokio::test]
async fn a_grant_in_another_workspace_opens_nothing_here() -> Result<()> {
    let store = testing::mem_store_of(4).await?;
    let here = TenantId::new("ws:here");
    let elsewhere = TenantId::new("ws:elsewhere");
    let owner = UserId::new("user:owner");
    let member = UserId::new("user:member");
    let key = CompartmentId::new("comp:shared-key");
    let now = Utc::now();
    testing::members(&store, &here, &[&owner, &member]).await?;
    compartment::create(
        &store,
        &Compartment::new(
            key.clone(),
            here.clone(),
            owner.clone(),
            "private here",
            now,
        ),
    )
    .await?;
    memory::upsert(
        &store,
        &mem(
            "memory:here-private",
            "ws:here",
            "private here",
            Some("comp:shared-key"),
        ),
    )
    .await?;
    // The owner view writes a grant in the other workspace: same key, same grantee.
    compartment::grant(
        &store,
        &Grant::new(
            elsewhere.clone(),
            key.clone(),
            member.clone(),
            Capability::Reference,
            UserId::new("user:someone-else"),
            now,
        ),
    )
    .await?;

    store.signin(&here, &member).await?;
    anyhow::ensure!(
        !visible_ids(&store)
            .await?
            .contains(&"memory:here-private".to_string()),
        "a grant in another workspace opened a private compartment here"
    );
    // A grant in this workspace still does.
    store.invalidate().await?;
    compartment::grant(
        &store,
        &Grant::new(
            here.clone(),
            key.clone(),
            member.clone(),
            Capability::Reference,
            owner.clone(),
            now,
        ),
    )
    .await?;
    store.signin(&here, &member).await?;
    anyhow::ensure!(
        visible_ids(&store)
            .await?
            .contains(&"memory:here-private".to_string()),
        "a grant in this workspace did not open the compartment"
    );
    Ok(())
}

// The compartment table's write rules are the owner's alone: another member of
// the workspace can neither take a compartment over by rewriting its owner,
// tombstone it, nor create one in someone else's name. The memory rule trusts
// the compartment row live, so this is what keeps a private compartment
// private against a member who reaches the engine without the app layer.
#[tokio::test]
async fn a_member_cannot_rewrite_another_members_compartment() -> Result<()> {
    let store = testing::mem_store_of(4).await?;
    let t = TenantId::new("ws:org");
    let a = UserId::new("user:a");
    let b = UserId::new("user:b");
    let ca = CompartmentId::new("comp:a-private");
    let now = Utc::now();
    testing::members(&store, &t, &[&a, &b]).await?;
    compartment::create(
        &store,
        &Compartment::new(ca.clone(), t.clone(), a.clone(), "A private", now),
    )
    .await?;
    memory::upsert(
        &store,
        &mem("memory:priv", "ws:org", "A private", Some("comp:a-private")),
    )
    .await?;

    // B, as a member, writes A's compartment as B's own, tombstones it, and
    // creates one in A's name. The engine refuses each either by persisting
    // nothing (the writes are upserts) or with an error, so the results are
    // not what is checked; the rows are.
    store.signin(&t, &b).await?;
    let taken = Compartment::new(ca.clone(), t.clone(), b.clone(), "mine now", now);
    compartment::create(&store, &taken).await.ok();
    compartment::delete(&store, &t, &ca, now).await.ok();
    let forged = Compartment::new(
        CompartmentId::new("comp:forged"),
        t.clone(),
        a.clone(),
        "in A's name",
        now,
    );
    compartment::create(&store, &forged).await.ok();
    anyhow::ensure!(
        !visible_ids(&store)
            .await?
            .contains(&"memory:priv".to_string()),
        "B reads A's private memory"
    );

    store.invalidate().await?;
    let Some(row) = compartment::get(&store, &t, &ca).await? else {
        anyhow::bail!("A's compartment is gone");
    };
    anyhow::ensure!(
        row.owner == a && row.name == "A private" && row.deleted_at.is_none(),
        "A's compartment changed: owner {}, name {:?}, deleted {:?}",
        row.owner.as_str(),
        row.name,
        row.deleted_at
    );
    anyhow::ensure!(
        compartment::get(&store, &t, &CompartmentId::new("comp:forged"))
            .await?
            .is_none(),
        "a compartment was created in A's name"
    );

    // The owner still can.
    store.signin(&t, &a).await?;
    compartment::create(
        &store,
        &Compartment::new(ca.clone(), t.clone(), a.clone(), "A renamed", now),
    )
    .await?;
    store.invalidate().await?;
    let Some(row) = compartment::get(&store, &t, &ca).await? else {
        anyhow::bail!("A's compartment is gone after A renamed it");
    };
    anyhow::ensure!(
        row.name == "A renamed",
        "the owner's own write did not land"
    );
    Ok(())
}
