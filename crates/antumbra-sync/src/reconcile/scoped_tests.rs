use super::*;
use antumbra_core::{Compartment, DeviceProfile, DeviceRole, TenantId, UserId};
use antumbra_store::repo::{compartment, device, principal};
use antumbra_store::EMBED_DIM;

use crate::config::Fabric;
use crate::table::PENUMBRA_TABLES;

fn tenant() -> TenantId {
    TenantId::new("ws:org")
}

/// Two members, each with a compartment and a machine, on the local store.
async fn org() -> Result<(Store, Store, UserId, UserId)> {
    let local = Store::connect_memory(EMBED_DIM).await?;
    let remote = Store::connect_memory(EMBED_DIM).await?;
    let lily = UserId::new("user:lily");
    let oslo = UserId::new("user:oslo");
    for store in [&local, &remote] {
        principal::provision(store, &tenant(), &lily).await?;
        principal::provision(store, &tenant(), &oslo).await?;
    }
    let now = Utc::now();
    for (id, owner) in [("comp:hers", &lily), ("comp:his", &oslo)] {
        compartment::create(
            &local,
            &Compartment::new(id, tenant(), (*owner).clone(), id, now),
        )
        .await?;
    }
    for (host, owner, role) in [
        ("her-laptop", &lily, DeviceRole::Memory),
        ("his-rig", &oslo, DeviceRole::Genesis),
    ] {
        device::upsert(
            &local,
            &DeviceProfile::new(tenant(), (*owner).clone(), host, "cpu", role, now),
        )
        .await?;
    }
    Ok((local, remote, lily, oslo))
}

/// The whole point of increment 5. A collector scoped to lily carries her
/// compartment and her machine, leaves oslo's behind, and refuses nothing --
/// `refused` is the number that must be zero, because a refusal on a table
/// the policy already narrowed means the policy and the ACL disagree.
#[tokio::test]
async fn a_scoped_collector_carries_one_fabric_and_refuses_nothing() -> Result<()> {
    let (local, remote, lily, oslo) = org().await?;
    let fabric = Fabric::new("ws:org", "user:lily");
    fabric.bind(&local).await?;
    fabric.bind(&remote).await?;
    let scope = Scope::resolve(&local, &fabric).await?;

    let stats = reconcile_all_scoped(&local, &remote, PENUMBRA_TABLES, Some(&scope)).await?;
    assert_eq!(
        stats.refused, 0,
        "a refusal here means the policy and the engine disagree"
    );
    assert!(stats.pushed > 0, "her own rows crossed");

    // Hers landed; his did not.
    remote.invalidate().await?;
    assert_eq!(
        device::list_for_user(&remote, &tenant(), &lily)
            .await?
            .len(),
        1,
        "her machine is in her fabric"
    );
    assert!(
        device::list_for_user(&remote, &tenant(), &oslo)
            .await?
            .is_empty(),
        "his machine is not"
    );
    remote.signin(&tenant(), &oslo).await?;
    assert!(
        compartment::get(
            &remote,
            &tenant(),
            &antumbra_core::CompartmentId::new("comp:his")
        )
        .await?
        .is_none(),
        "his compartment was never carried into her fabric"
    );
    Ok(())
}

/// The control. Without a scope the same pass is tenant-wide, which is what
/// every deployment does today -- so the narrowing is doing the work, not
/// some accident of the fixture.
#[tokio::test]
async fn an_unscoped_collector_still_carries_the_whole_tenant() -> Result<()> {
    let (local, remote, _lily, oslo) = org().await?;
    let stats = reconcile_all(&local, &remote, PENUMBRA_TABLES).await?;
    assert_eq!(stats.refused, 0);
    assert_eq!(
        device::list_for_user(&remote, &tenant(), &oslo)
            .await?
            .len(),
        1,
        "owner mode carries his machine too"
    );
    Ok(())
}
