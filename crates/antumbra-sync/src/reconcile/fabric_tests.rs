use super::*;
use antumbra_core::{
    CompartmentId, DeviceProfile, DeviceRole, GenesisRequest, GenesisStatus, TenantId, UserId,
};
use antumbra_store::repo::{device, genesis};
use antumbra_store::EMBED_DIM;
use chrono::Duration as ChronoDuration;

use crate::table::PENUMBRA_TABLES;

async fn node() -> Result<Store> {
    Store::connect_memory(EMBED_DIM).await
}

fn tenant() -> TenantId {
    TenantId::new("ws:t")
}

fn user() -> UserId {
    UserId::new("user:a")
}

/// Delivery across the fabric. The laptop cannot train and leaves a request; the
/// rig takes it. Without the fabric tables replicating, both halves work
/// perfectly and no run ever crosses between the two machines.
#[tokio::test]
async fn a_genesis_run_crosses_from_the_node_that_asked_to_the_one_that_can() -> Result<()> {
    let laptop = node().await?;
    let rig = node().await?;
    let now = Utc::now();

    // Each machine knows only itself to begin with.
    device::upsert(
        &laptop,
        &DeviceProfile::new(
            tenant(),
            user(),
            "her-laptop",
            "cpu",
            DeviceRole::Memory,
            now,
        ),
    )
    .await?;
    device::upsert(
        &rig,
        &DeviceProfile::new(
            tenant(),
            user(),
            "the-rig",
            "cuda",
            DeviceRole::Genesis,
            now,
        ),
    )
    .await?;
    assert!(
        device::genesis_for_user(&laptop, &tenant(), &user())
            .await?
            .is_none(),
        "before replication the laptop believes it is alone"
    );

    reconcile_all(&laptop, &rig, PENUMBRA_TABLES).await?;

    // Now the laptop can see where the work belongs.
    assert_eq!(
        device::genesis_for_user(&laptop, &tenant(), &user())
            .await?
            .map(|d| d.host),
        Some("the-rig".to_string())
    );

    // It leaves a run, which reaches the rig on the next pass.
    genesis::ask(
        &laptop,
        &GenesisRequest::new(
            tenant(),
            user(),
            CompartmentId::new("comp:rust"),
            "her-laptop",
            "the-rig",
            now,
        ),
    )
    .await?;
    assert!(genesis::list_open_for_user(&rig, &tenant(), &user())
        .await?
        .is_empty());

    reconcile_all(&laptop, &rig, PENUMBRA_TABLES).await?;
    let waiting = genesis::list_open_for_user(&rig, &tenant(), &user()).await?;
    assert_eq!(
        waiting
            .iter()
            .map(|r| (r.compartment.as_str(), r.from_host.as_str()))
            .collect::<Vec<_>>(),
        vec![("comp:rust", "her-laptop")]
    );

    // The rig runs it and closes it; the laptop learns the outcome.
    genesis::set_status(
        &rig,
        &waiting[0],
        GenesisStatus::Done,
        now + ChronoDuration::minutes(5),
    )
    .await?;
    reconcile_all(&laptop, &rig, PENUMBRA_TABLES).await?;
    assert!(
        genesis::list_open_for_user(&laptop, &tenant(), &user())
            .await?
            .is_empty(),
        "the asking node sees its run finished"
    );
    Ok(())
}

/// `updated_at` on these two is load-bearing, not incidental. A claim is an
/// in-place mutation, so last-write-wins has to order it above the pending
/// row it replaces -- otherwise a stale pending row wins the reconcile and
/// the same run is handed to a second trainer.
#[tokio::test]
async fn a_claim_out_versions_the_pending_row_it_replaces() -> Result<()> {
    let laptop = node().await?;
    let rig = node().await?;
    let asked = Utc::now();
    let request = GenesisRequest::new(
        tenant(),
        user(),
        CompartmentId::new("comp:rust"),
        "her-laptop",
        "the-rig",
        asked,
    );
    genesis::ask(&laptop, &request).await?;
    reconcile_all(&laptop, &rig, PENUMBRA_TABLES).await?;

    // The rig claims it, later.
    let taken = genesis::list_open_for_user(&rig, &tenant(), &user()).await?;
    genesis::set_status(
        &rig,
        &taken[0],
        GenesisStatus::Claimed,
        asked + ChronoDuration::minutes(1),
    )
    .await?;

    // Reconciling both ways must not resurrect the pending row on either
    // side, however many passes run.
    for _ in 0..3 {
        reconcile_all(&laptop, &rig, PENUMBRA_TABLES).await?;
    }
    for (name, store) in [("laptop", &laptop), ("rig", &rig)] {
        let seen = genesis::list_open_for_user(store, &tenant(), &user()).await?;
        assert_eq!(seen.len(), 1, "{name}");
        assert_eq!(
            seen[0].status,
            GenesisStatus::Claimed,
            "{name} must not resurrect the pending row"
        );
    }
    Ok(())
}
