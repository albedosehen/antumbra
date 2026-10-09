//! Retirement through the MCP surface: a dormant expert is not
//! routed to, even by a router trained while it was active, and `population`
//! says where each expert stands.

use super::*;
use antumbra_core::{ExpertStatus, TransitionCause};
use antumbra_store::repo::lifecycle;

fn shared(id: &str) -> Expert {
    let now = Utc::now();
    Expert {
        id: ExpertId::new(id),
        name: id.trim_start_matches("expert:").into(),
        base_model: "base".into(),
        artifact_uri: format!("mem://{id}"),
        capability_card: serde_json::Value::Null,
        capability_vec: None,
        fitness: 1.0,
        frozen_at: Some(now),
        generation: Generation::ZERO,
        owner: None,
        compartment: None,
        placed_on: None,
        created_at: now,
    }
}

#[tokio::test]
async fn a_dormant_expert_is_not_routed_to_and_population_says_so() {
    let s = server().await;
    expert::insert(&s.store, &shared("expert:adder"))
        .await
        .unwrap();
    router::save(
        &s.store,
        &LearnedRouter {
            weights: vec![1.0; EMBED_DIM],
            experts: vec![RouterExpert {
                id: ExpertId::new("expert:adder"),
                centroid: vec![0.0; EMBED_DIM],
            }],
            temperature: 0.1,
            floor: -1.0,
        },
    )
    .await
    .unwrap();
    let route = || {
        s.route(Parameters(RouteParams {
            task: "add two numbers".into(),
            top_k: Some(3),
        }))
    };
    assert_eq!(route().await.unwrap().0.routes.len(), 1);

    lifecycle::transition(
        &s.store,
        &ExpertId::new("expert:adder"),
        ExpertStatus::Dormant,
        TransitionCause::Operator { note: None },
        None,
    )
    .await
    .unwrap();
    let r = route().await.unwrap();
    assert!(
        r.0.escalate && r.0.routes.is_empty(),
        "the router still holds it, but masked"
    );
    assert!(
        r.0.reason
            .as_deref()
            .unwrap_or("")
            .starts_with("the experts that cover this task are dormant"),
        "{:?}",
        r.0.reason
    );

    let population = s.population().await.unwrap().0.experts;
    assert_eq!(population.len(), 1);
    assert_eq!(population[0].status, "dormant");
}
