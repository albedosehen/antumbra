use super::*;
use antumbra_core::{DeviceProfile, DeviceRole, TenantId, UserId};
use antumbra_store::repo::{device, principal};
use antumbra_store::EMBED_DIM;

use crate::scope::Replicate;
use crate::table::PENUMBRA_TABLES;

const DEVICE: &TableSpec = &TableSpec {
    name: "device_profile",
    version_field: "updated_at",
    replicate: Replicate::Owned(|scope: &Scope, row| scope.is_own_user(row)),
};

fn tenant() -> TenantId {
    TenantId::new("ws:org")
}

/// Two members of one tenant, provisioned on both stores so either can sign
/// a session in.
async fn org() -> Result<(Store, Store, UserId, UserId)> {
    let local = Store::connect_memory(EMBED_DIM).await?;
    let remote = Store::connect_memory(EMBED_DIM).await?;
    let lily = UserId::new("user:lily");
    let oslo = UserId::new("user:oslo");
    for store in [&local, &remote] {
        principal::provision(store, &tenant(), &lily).await?;
        principal::provision(store, &tenant(), &oslo).await?;
    }
    Ok((local, remote, lily, oslo))
}

/// The test a per-user collector needs, and the reason the record-session
/// design is not sufficient on its own.
///
/// `device_profile` is tenant-readable and own-write. Under lily's session
/// the collector can see oslo's node and cannot write it. The engine
/// refuses by persisting nothing and without an error, so before this the
/// row was counted pushed and never landed -- every cycle, forever.
#[tokio::test]
async fn a_row_the_session_may_read_and_not_write_is_refused_not_pushed() -> Result<()> {
    let (local, remote, _lily, oslo) = org().await?;
    let now = Utc::now();
    // Oslo's machine, on the local store only.
    device::upsert(
        &local,
        &DeviceProfile::new(
            tenant(),
            oslo.clone(),
            "oslos-rig",
            "cuda",
            DeviceRole::Genesis,
            now,
        ),
    )
    .await?;

    // As owner, it replicates: this is today's collector, and the control
    // that proves the refusal below is about the session, not the row.
    let owner_pass = reconcile_table(&local, &remote, DEVICE).await?;
    assert_eq!((owner_pass.pushed, owner_pass.refused), (1, 0));

    // Now the same row, from a store where it has not landed, under lily.
    let (local, remote, lily, oslo) = org().await?;
    device::upsert(
        &local,
        &DeviceProfile::new(
            tenant(),
            oslo.clone(),
            "oslos-rig",
            "cuda",
            DeviceRole::Genesis,
            now,
        ),
    )
    .await?;
    local.signin(&tenant(), &lily).await?;
    remote.signin(&tenant(), &lily).await?;

    let scoped = reconcile_table(&local, &remote, DEVICE).await?;
    assert_eq!(
        (scoped.pushed, scoped.refused),
        (0, 1),
        "lily may read oslo's node and may not write it"
    );
    // And the engine really did refuse: nothing crossed.
    remote.invalidate().await?;
    assert!(
        device::list_for_user(&remote, &tenant(), &oslo)
            .await?
            .is_empty(),
        "the row did not land, which is what `refused` is reporting"
    );
    Ok(())
}

/// The other half: a row the session owns crosses normally, so `refused` is
/// reporting the permission boundary and not simply every write under a
/// record session.
#[tokio::test]
async fn a_row_the_session_owns_still_crosses() -> Result<()> {
    let (local, remote, lily, _oslo) = org().await?;
    let now = Utc::now();
    device::upsert(
        &local,
        &DeviceProfile::new(
            tenant(),
            lily.clone(),
            "her-laptop",
            "cpu",
            DeviceRole::Memory,
            now,
        ),
    )
    .await?;
    local.signin(&tenant(), &lily).await?;
    remote.signin(&tenant(), &lily).await?;

    let stats = reconcile_all(&local, &remote, PENUMBRA_TABLES).await?;
    assert_eq!(
        (stats.pushed, stats.refused),
        (1, 0),
        "her own node is hers to replicate"
    );
    remote.invalidate().await?;
    assert_eq!(
        device::list_for_user(&remote, &tenant(), &lily)
            .await?
            .len(),
        1
    );
    Ok(())
}
