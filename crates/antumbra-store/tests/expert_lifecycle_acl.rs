//! An expert's moves are visible exactly where the expert is (ADR-0022 S-5).
//! A tenant session reads shared experts and its own private ones; if it could
//! read those experts but not their moves, a demoted expert would look active
//! to it and be routed to.

use chrono::Utc;

use antumbra_core::{
    Expert, ExpertId, ExpertStatus, Generation, Result, TenantId, TransitionCause, UserId,
};
use antumbra_store::repo::{expert, lifecycle, principal};
use antumbra_store::Store;

const DIM: usize = 4;

fn expert(id: &str, owner: Option<&str>) -> Expert {
    Expert {
        id: ExpertId::new(id),
        name: id.to_string(),
        base_model: "base".into(),
        artifact_uri: "adapters/x.safetensors".into(),
        capability_card: serde_json::json!({}),
        capability_vec: Some(vec![1.0, 0.0, 0.0, 0.0]),
        fitness: 1.0,
        frozen_at: Some(Utc::now()),
        generation: Generation::ZERO,
        owner: owner.map(UserId::new),
        compartment: None,
        placed_on: None,
        created_at: Utc::now(),
    }
}

#[tokio::test]
async fn a_tenant_session_sees_the_moves_of_the_experts_it_can_see() -> Result<()> {
    let store = Store::connect_memory(DIM).await?;
    let (tenant, ada, bob) = (
        TenantId::new("ws:t"),
        UserId::new("user:ada"),
        UserId::new("user:bob"),
    );
    principal::provision(&store, &tenant, &ada).await?;
    principal::provision(&store, &tenant, &bob).await?;
    for (id, owner) in [
        ("expert:shared", None),
        ("expert:shared-live", None),
        ("expert:ada", Some("user:ada")),
        ("expert:bob", Some("user:bob")),
    ] {
        expert::insert(&store, &expert(id, owner)).await?;
    }
    for id in ["expert:shared", "expert:ada", "expert:bob"] {
        lifecycle::transition(
            &store,
            &ExpertId::new(id),
            ExpertStatus::Dormant,
            TransitionCause::Operator { note: None },
            None,
        )
        .await?;
    }

    store.signin(&tenant, &ada).await?;
    let mut seen: Vec<(String, ExpertStatus)> = lifecycle::population(&store)
        .await?
        .into_iter()
        .map(|(e, s)| (e.id.to_string(), s))
        .collect();
    seen.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(
        seen,
        [
            ("expert:ada".to_string(), ExpertStatus::Dormant),
            ("expert:shared".to_string(), ExpertStatus::Dormant),
            ("expert:shared-live".to_string(), ExpertStatus::Active),
        ],
        "ada sees her own and the shared experts, each as it stands"
    );
    let routable: Vec<String> = lifecycle::routable(&store)
        .await?
        .into_iter()
        .map(|e| e.id.to_string())
        .collect();
    assert_eq!(routable, ["expert:shared-live"]);
    assert!(
        lifecycle::history(&store, &ExpertId::new("expert:bob"))
            .await?
            .is_empty(),
        "bob's moves stay his"
    );
    store.invalidate().await?;
    Ok(())
}
