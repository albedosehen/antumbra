//! Engine-enforced ownership of the device registry (ADR-0017): a node writes
//! only the row of the user running it. The table used to be owner-only, under
//! which no node could register at all; opening it for registration is what
//! makes the write rule load-bearing, because genesis is dispatched to whatever
//! machine the registry names.

use chrono::Utc;

use antumbra_core::{DeviceProfile, DeviceRole, Result, TenantId, UserId};
use antumbra_store::repo::{device, principal};
use antumbra_store::Store;

const DIM: usize = 4;

struct Org {
    store: Store,
    tenant: TenantId,
    lily: UserId,
    oslo: UserId,
}

/// Two members of one tenant, each with machines of their own.
async fn org() -> Result<Org> {
    let store = Store::connect_memory(DIM).await?;
    let tenant = TenantId::new("ws:org");
    let lily = UserId::new("user:lily");
    let oslo = UserId::new("user:oslo");
    principal::provision(&store, &tenant, &lily).await?;
    principal::provision(&store, &tenant, &oslo).await?;
    Ok(Org {
        store,
        tenant,
        lily,
        oslo,
    })
}

fn node(tenant: &TenantId, user: &UserId, host: &str, role: DeviceRole) -> DeviceProfile {
    DeviceProfile::new(tenant.clone(), user.clone(), host, "cpu", role, Utc::now())
}

#[tokio::test]
async fn a_member_cannot_declare_anothers_machine_a_trainer() -> Result<()> {
    let o = org().await?;

    // Lily's laptop, registered by lily: a memory node, as a laptop is.
    o.store.signin(&o.tenant, &o.lily).await?;
    device::upsert(
        &o.store,
        &node(&o.tenant, &o.lily, "laptop", DeviceRole::Memory),
    )
    .await?;

    // Oslo tries to re-declare it a trainer. The row he addresses is the same
    // one (the id is derived from tenant, user and host), so if the engine let
    // this through, lily's genesis would be dispatched to a machine that cannot
    // train, chosen by someone else.
    o.store.invalidate().await?;
    o.store.signin(&o.tenant, &o.oslo).await?;
    let forged = node(&o.tenant, &o.lily, "laptop", DeviceRole::Genesis);
    // The engine refuses by persisting nothing, without an error.
    device::upsert(&o.store, &forged).await.ok();

    o.store.invalidate().await?;
    o.store.signin(&o.tenant, &o.lily).await?;
    let hers = device::list_for_user(&o.store, &o.tenant, &o.lily).await?;
    assert_eq!(
        hers.iter()
            .map(|d| (d.host.as_str(), d.role))
            .collect::<Vec<_>>(),
        vec![("laptop", DeviceRole::Memory)],
        "another member must not be able to re-role a machine"
    );
    assert!(
        device::genesis_for_user(&o.store, &o.tenant, &o.lily)
            .await?
            .is_none(),
        "and so lily still has nowhere to train"
    );
    Ok(())
}

#[tokio::test]
async fn a_node_registers_its_own_user_and_dispatch_can_find_it() -> Result<()> {
    let o = org().await?;

    o.store.signin(&o.tenant, &o.oslo).await?;
    device::upsert(
        &o.store,
        &node(&o.tenant, &o.oslo, "workstation", DeviceRole::Genesis).with_vram(24_576),
    )
    .await?;
    device::upsert(
        &o.store,
        &node(&o.tenant, &o.oslo, "laptop", DeviceRole::Memory),
    )
    .await?;

    let found = device::genesis_for_user(&o.store, &o.tenant, &o.oslo).await?;
    assert_eq!(found.as_ref().map(|d| d.host.as_str()), Some("workstation"));
    assert_eq!(found.and_then(|d| d.vram_mib), Some(24_576));

    // Read is tenant-wide, because the node doing the dispatching is not always
    // the node being dispatched to.
    o.store.invalidate().await?;
    o.store.signin(&o.tenant, &o.lily).await?;
    assert_eq!(
        device::list_for_user(&o.store, &o.tenant, &o.oslo)
            .await?
            .len(),
        2,
        "a fabric is visible within its tenant"
    );

    // A tenant boundary is still a boundary.
    let elsewhere = TenantId::new("ws:elsewhere");
    assert!(
        device::list_for_user(&o.store, &elsewhere, &o.oslo)
            .await?
            .is_empty(),
        "and does not reach across tenants"
    );
    Ok(())
}
