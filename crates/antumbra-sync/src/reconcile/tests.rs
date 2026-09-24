use super::*;
use antumbra_core::{Memory, MemoryId, MemoryNetwork, TenantId};
use antumbra_store::repo::memory;
use antumbra_store::EMBED_DIM;
use chrono::Duration as ChronoDuration;

use crate::scope::Replicate;

const MEMORY: &TableSpec = &TableSpec {
    name: "memory",
    version_field: "updated_at",
    replicate: Replicate::Owned(|scope: &Scope, row| scope.holds_memory(row)),
};

async fn mem_store() -> Store {
    Store::connect_memory(EMBED_DIM).await.unwrap()
}

fn mem(id: &str, tenant: &TenantId, content: &str, at: DateTime<Utc>) -> Memory {
    Memory::new(id, tenant.clone(), MemoryNetwork::World, content, 0.9, at)
}

// A row each side is missing flows the other way: local-only pushes, remote-
// only pulls, and a second pass is a no-op (converged + idempotent).
#[tokio::test]
async fn seeds_both_directions_then_settles() {
    let (local, remote) = (mem_store().await, mem_store().await);
    let tenant = TenantId::new("t");
    let now = Utc::now();
    memory::upsert(
        &local,
        &mem("aaaaaaaa-0000-0000-0000-000000000001", &tenant, "left", now),
    )
    .await
    .unwrap();
    memory::upsert(
        &remote,
        &mem(
            "bbbbbbbb-0000-0000-0000-000000000002",
            &tenant,
            "right",
            now,
        ),
    )
    .await
    .unwrap();

    let stats = reconcile_table(&local, &remote, MEMORY).await.unwrap();
    assert_eq!(
        stats,
        ReconcileStats {
            pushed: 1,
            pulled: 1,
            refused: 0,
            declined: 0
        }
    );

    // Both stores now hold both memories.
    assert_eq!(memory::list(&local, &tenant).await.unwrap().len(), 2);
    assert_eq!(memory::list(&remote, &tenant).await.unwrap().len(), 2);

    // Converged: the next pass moves nothing.
    let again = reconcile_table(&local, &remote, MEMORY).await.unwrap();
    assert_eq!(again, ReconcileStats::default());
}

// The strictly-newer version of a shared record wins on both sides.
#[tokio::test]
async fn last_write_wins_on_conflict() {
    let (local, remote) = (mem_store().await, mem_store().await);
    let tenant = TenantId::new("t");
    let id = "cccccccc-0000-0000-0000-000000000003";
    let t0 = Utc::now();
    let t1 = t0 + ChronoDuration::seconds(5);

    // Same record, divergent: remote's copy is newer.
    memory::upsert(&local, &mem(id, &tenant, "stale", t0))
        .await
        .unwrap();
    memory::upsert(&remote, &mem(id, &tenant, "fresh", t1))
        .await
        .unwrap();

    let stats = reconcile_table(&local, &remote, MEMORY).await.unwrap();
    assert_eq!(
        stats,
        ReconcileStats {
            pushed: 0,
            pulled: 1,
            refused: 0,
            declined: 0
        },
        "newer remote pulled to local"
    );

    let mid = MemoryId::new(id);
    let on_local = memory::get(&local, &tenant, &mid).await.unwrap().unwrap();
    let on_remote = memory::get(&remote, &tenant, &mid).await.unwrap().unwrap();
    assert_eq!(on_local.content, "fresh", "local took the newer version");
    assert_eq!(on_remote.content, "fresh", "remote unchanged");
}

// A forget (tombstone) is a newer version, so it propagates to the other side
// and the trace does not resurrect; the pass then converges.
#[tokio::test]
async fn a_tombstone_propagates_and_does_not_resurrect() {
    let (local, remote) = (mem_store().await, mem_store().await);
    let tenant = TenantId::new("t");
    let id = "dddddddd-0000-0000-0000-000000000004";
    let mid = MemoryId::new(id);
    let t0 = Utc::now();

    // Both sides hold the live trace.
    memory::upsert(&local, &mem(id, &tenant, "live", t0))
        .await
        .unwrap();
    memory::upsert(&remote, &mem(id, &tenant, "live", t0))
        .await
        .unwrap();

    // Local forgets it (a tombstone, newer than remote's live copy).
    memory::soft_delete(&local, &tenant, &mid, t0 + ChronoDuration::seconds(5))
        .await
        .unwrap();

    let stats = reconcile_table(&local, &remote, MEMORY).await.unwrap();
    assert_eq!(
        stats,
        ReconcileStats {
            pushed: 1,
            pulled: 0,
            refused: 0,
            declined: 0
        },
        "tombstone pushed"
    );

    // Forgotten on both sides; the live copy did not resurrect it on local.
    assert!(memory::get(&remote, &tenant, &mid).await.unwrap().is_none());
    assert!(memory::get(&local, &tenant, &mid).await.unwrap().is_none());

    // Converged.
    assert_eq!(
        reconcile_table(&local, &remote, MEMORY).await.unwrap(),
        ReconcileStats::default()
    );
}

// Security: a revoke on one store propagates to the other (a stale live grant
// cannot keep the grantee in). The revocation tombstone out-versions the live
// copy (bumped updated_at) and wins.
#[tokio::test]
async fn a_revoke_propagates_and_cannot_be_out_voted_by_a_stale_grant() {
    use antumbra_core::{Capability, Compartment, CompartmentId, Origin, UserId};
    use antumbra_store::repo::compartment;

    const GRANT: &TableSpec = &TableSpec {
        name: "grant",
        version_field: "updated_at",
        replicate: Replicate::Owned(|scope: &Scope, row| scope.grants_own_compartment(row)),
    };
    let (local, remote) = (mem_store().await, mem_store().await);
    let tenant = TenantId::new("t");
    let comp = CompartmentId::new("comp-g");
    let bob = UserId::new("bob");
    let t0 = Utc::now();

    let new_comp = || Compartment {
        id: comp.clone(),
        tenant: tenant.clone(),
        owner: UserId::new("alice"),
        name: "shared".into(),
        origin: Origin::User,
        created_at: t0,
        updated_at: t0,
        deleted_at: None,
    };
    let grant = antumbra_core::Grant::new(
        tenant.clone(),
        comp.clone(),
        bob.clone(),
        Capability::Reference,
        UserId::new("alice"),
        t0,
    );
    // Both sides start with the compartment + the live grant.
    for s in [&local, &remote] {
        compartment::create(s, &new_comp()).await.unwrap();
        compartment::grant(s, &grant).await.unwrap();
    }

    // Local revokes bob (a newer version than remote's still-live grant).
    compartment::revoke(
        &local,
        &tenant,
        &comp,
        &bob,
        t0 + ChronoDuration::seconds(5),
    )
    .await
    .unwrap();

    let stats = reconcile_table(&local, &remote, GRANT).await.unwrap();
    assert_eq!(
        stats,
        ReconcileStats {
            pushed: 1,
            pulled: 0,
            refused: 0,
            declined: 0
        },
        "revocation pushed"
    );

    // Remote no longer lists bob as a grantee (revoked everywhere).
    assert!(
        compartment::list_grants(&remote, &tenant, &comp)
            .await
            .unwrap()
            .is_empty(),
        "the revocation reached the remote: bob is no longer a live grantee"
    );
    // And it does not resurrect from the stale side.
    assert_eq!(
        reconcile_table(&local, &remote, GRANT).await.unwrap(),
        ReconcileStats::default()
    );
}

// Incremental cursors: the first pass (empty cursors) is a full scan that
// seeds the remote and sets the watermark; a later write past the watermark is
// the only thing the next pass fetches and pushes, and the pass then settles.
#[tokio::test]
async fn incremental_pass_moves_only_what_changed_since_the_watermark() {
    let (local, remote) = (mem_store().await, mem_store().await);
    let tenant = TenantId::new("t");
    let t0 = Utc::now();
    let t1 = t0 + ChronoDuration::seconds(30);
    let mut cursors = Cursors::new();
    let tables = &[*MEMORY];

    // First pass: a full scan seeds A onto the remote.
    memory::upsert(
        &local,
        &mem("aaaaaaaa-0000-0000-0000-0000000000a1", &tenant, "A", t0),
    )
    .await
    .unwrap();
    let s = reconcile_all_since(&local, &remote, tables, &mut cursors, Duration::ZERO)
        .await
        .unwrap();
    assert_eq!(
        s,
        ReconcileStats {
            pushed: 1,
            pulled: 0,
            refused: 0,
            declined: 0
        }
    );

    // A new local write past the watermark is the only row the next pass moves.
    memory::upsert(
        &local,
        &mem("bbbbbbbb-0000-0000-0000-0000000000b2", &tenant, "B", t1),
    )
    .await
    .unwrap();
    let s = reconcile_all_since(&local, &remote, tables, &mut cursors, Duration::ZERO)
        .await
        .unwrap();
    assert_eq!(
        s,
        ReconcileStats {
            pushed: 1,
            pulled: 0,
            refused: 0,
            declined: 0
        },
        "only B moved"
    );
    assert_eq!(memory::list(&remote, &tenant).await.unwrap().len(), 2);

    // Converged: nothing new past the watermark.
    let s = reconcile_all_since(&local, &remote, tables, &mut cursors, Duration::ZERO)
        .await
        .unwrap();
    assert_eq!(s, ReconcileStats::default());
}

// The asymmetry that keeps incremental reconcile correct: a row updated on
// ONLY the remote past the watermark appears in just the remote's window, yet
// is still pulled to the local (its copy sits at or below the floor, so the
// remote's is provably newer -- no cross-side compare needed).
#[tokio::test]
async fn an_update_on_one_side_only_still_propagates_under_a_cursor() {
    let (local, remote) = (mem_store().await, mem_store().await);
    let tenant = TenantId::new("t");
    let id = "cccccccc-0000-0000-0000-0000000000c3";
    let mid = MemoryId::new(id);
    let t0 = Utc::now();
    let t2 = t0 + ChronoDuration::seconds(30);
    let mut cursors = Cursors::new();
    let tables = &[*MEMORY];

    // Seed A on both sides, advancing the watermark to t0.
    memory::upsert(&local, &mem(id, &tenant, "v0", t0))
        .await
        .unwrap();
    reconcile_all_since(&local, &remote, tables, &mut cursors, Duration::ZERO)
        .await
        .unwrap();

    // The remote alone updates A past the watermark.
    memory::upsert(&remote, &mem(id, &tenant, "v2", t2))
        .await
        .unwrap();
    let s = reconcile_all_since(&local, &remote, tables, &mut cursors, Duration::ZERO)
        .await
        .unwrap();
    assert_eq!(
        s,
        ReconcileStats {
            pushed: 0,
            pulled: 1,
            refused: 0,
            declined: 0
        },
        "remote-only update pulled"
    );
    assert_eq!(
        memory::get(&local, &tenant, &mid)
            .await
            .unwrap()
            .unwrap()
            .content,
        "v2"
    );
}

// The lookback window lowers the query floor (so a slightly-stale write is not
// skipped) but never raises it; an empty or bad mark stays a full scan.
#[test]
fn lookback_floor_subtracts_the_window() {
    let t = "2026-06-05T12:00:30+00:00";
    let lowered = lookback_floor(t, Duration::from_secs(5));
    let expected =
        DateTime::parse_from_rfc3339(t).unwrap().with_timezone(&Utc) - ChronoDuration::seconds(5);
    assert_eq!(lowered, expected.to_rfc3339());
    assert!(lowered.as_str() < t, "the floor is below the mark");
    // No window: the mark is unchanged. Empty/garbage: a full scan.
    assert_eq!(lookback_floor(t, Duration::ZERO), t);
    assert_eq!(lookback_floor("", Duration::from_secs(5)), "");
    assert_eq!(
        lookback_floor("not-a-date", Duration::from_secs(5)),
        "not-a-date"
    );
}

// A compartment deletion (tombstone) propagates and does not resurrect from a
// replica still holding the live row.
#[tokio::test]
async fn a_compartment_deletion_propagates_and_does_not_resurrect() {
    use antumbra_core::{Compartment, CompartmentId, UserId};
    use antumbra_store::repo::compartment;

    const COMPARTMENT: &TableSpec = &TableSpec {
        name: "compartment",
        version_field: "updated_at",
        replicate: Replicate::Owned(|scope: &Scope, row| scope.owns_compartment(row)),
    };
    let (local, remote) = (mem_store().await, mem_store().await);
    let tenant = TenantId::new("t");
    let id = CompartmentId::new("comp-d");
    let t0 = Utc::now();

    for s in [&local, &remote] {
        compartment::create(
            s,
            &Compartment::new(id.clone(), tenant.clone(), UserId::new("alice"), "c", t0),
        )
        .await
        .unwrap();
    }
    compartment::delete(&local, &tenant, &id, t0 + ChronoDuration::seconds(5))
        .await
        .unwrap();

    let stats = reconcile_table(&local, &remote, COMPARTMENT).await.unwrap();
    assert_eq!(
        stats,
        ReconcileStats {
            pushed: 1,
            pulled: 0,
            refused: 0,
            declined: 0
        }
    );
    assert!(
        compartment::get(&remote, &tenant, &id)
            .await
            .unwrap()
            .is_none(),
        "deletion reached remote"
    );
    assert_eq!(
        reconcile_table(&local, &remote, COMPARTMENT).await.unwrap(),
        ReconcileStats::default()
    );
}
