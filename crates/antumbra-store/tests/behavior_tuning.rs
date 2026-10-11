//! The loop's readings of a workspace's audits: appended per tenant, the
//! latest in force, listed newest first, and one workspace blind to another's.

use antumbra_core::behavior::tuning::{Lean, Reading};
use antumbra_core::TenantId;
use antumbra_store::repo::behavior_tuning::{self, TuningRow};
use antumbra_store::{Store, EMBED_DIM};
use chrono::{Duration, Utc};

#[tokio::test]
async fn readings_append_per_tenant_and_the_latest_is_in_force() {
    let store = Store::connect_memory(EMBED_DIM).await.unwrap();
    let alpha = TenantId::new("ws:alpha");
    assert!(behavior_tuning::latest(&store, &alpha)
        .await
        .unwrap()
        .is_none());

    let reading = |from: f64, to: f64| Reading {
        lean: Lean::Looser,
        from,
        to,
        bets: 2,
        fresh: 6,
        base: 2,
        expert: 6,
        realized: 0.1,
    };
    let t0 = Utc::now();
    behavior_tuning::append(&store, &TuningRow::of(&alpha, &reading(0.9, 0.9), t0))
        .await
        .unwrap();
    behavior_tuning::append(
        &store,
        &TuningRow::of(&alpha, &reading(0.9, 0.92), t0 + Duration::seconds(1)),
    )
    .await
    .unwrap();

    let latest = behavior_tuning::latest(&store, &alpha)
        .await
        .unwrap()
        .expect("two readings");
    assert_eq!(latest.to, 0.92);
    let kept = latest.reading().expect("a lean this build knows");
    assert_eq!(kept.lean, Lean::Looser);
    assert!(kept.moved());

    let recent = behavior_tuning::recent(&store, &alpha, 5).await.unwrap();
    assert_eq!(
        recent.iter().map(|r| r.to).collect::<Vec<_>>(),
        [0.92, 0.9],
        "newest first"
    );
    assert_eq!(
        behavior_tuning::recent(&store, &alpha, 1)
            .await
            .unwrap()
            .len(),
        1
    );

    // Another workspace has read nothing.
    assert!(behavior_tuning::latest(&store, &TenantId::new("ws:beta"))
        .await
        .unwrap()
        .is_none());
}
