//! The counterfactual boundary of competence, end-to-end.
//! Counterfactual search recovers C', the
//! boundary persists as actionable through surql-rs, and the gate inhibits
//! routing inside the failure scope, and only inside it. The single fake is
//! the `AcceptabilityProbe` (the real probe needs hardware-adaptive serving); the
//! search, the persistence, and the gate are all production code.

use antumbra_boundary::{find_scope, finding_to_boundary};
use antumbra_core::testing::FeatureProbe;
use antumbra_core::{BoundaryId, Expert, ExpertId, Generation, Grain};
use antumbra_gate::{route, GateConfig};
use antumbra_store::repo::{boundary, expert};
use antumbra_store::Store;
use chrono::Utc;

fn expert_at(key: &str, vec: Vec<f32>) -> Expert {
    Expert {
        id: ExpertId::new(key),
        name: key.into(),
        base_model: "base".into(),
        artifact_uri: "mem://x".into(),
        capability_card: serde_json::Value::Null,
        capability_vec: Some(vec),
        fitness: 1.0,
        frozen_at: Some(Utc::now()),
        generation: Generation::ZERO,
        owner: None,
        compartment: None,
        created_at: Utc::now(),
    }
}

#[tokio::test]
async fn actionable_boundary_inhibits_routing_inside_its_scope_only() {
    let store = Store::connect_memory(4).await.expect("connect");

    // Expert A handles the behavior that has a known failure scope; B is elsewhere.
    expert::insert(&store, &expert_at("expert:a", vec![1.0, 0.0, 0.0, 0.0]))
        .await
        .unwrap();
    expert::insert(&store, &expert_at("expert:b", vec![0.0, 1.0, 0.0, 0.0]))
        .await
        .unwrap();

    // Antumbra job #1: hold the behavior fixed, vary the context, re-probe until
    // acceptability flips. `npm install` is wrong under deno (C), right under
    // node (C'); the governing feature is the runtime.
    let probe = FeatureProbe {
        feature: "runtime".into(),
        ok_value: serde_json::json!("node"),
    };
    let fail = serde_json::json!({ "runtime": "deno", "task": "install deps" });
    let candidates = vec![("runtime".to_string(), vec![serde_json::json!("node")])];
    let finding = find_scope("run npm install", &fail, &candidates, &probe)
        .await
        .unwrap()
        .expect("a single-feature change flips acceptability");
    assert_eq!(finding.governing_feature, "runtime");

    // Job #2: persist as an actionable boundary anchored at A's region (the
    // embedded context where the behavior is known to fail).
    let context_vec = vec![1.0, 0.0, 0.0, 0.0];
    let b = finding_to_boundary(
        BoundaryId::new("boundary:a"),
        &finding,
        Grain::Project,
        1.0,
        Some(context_vec),
        None,
        Generation::ZERO,
        Utc::now(),
    );
    assert!(b.is_actionable());
    boundary::upsert(&store, &b).await.unwrap();

    // Round-trip through the substrate.
    let experts = expert::list(&store).await.unwrap();
    let boundaries = boundary::list(&store).await.unwrap();
    assert_eq!(boundaries.len(), 1);
    assert!(boundaries[0].is_actionable());

    let cfg = GateConfig::default();

    // Inside the failure scope: A matches perfectly, but the boundary cancels
    // its coverage -> the gate escalates rather than route into a known failure.
    let inside = route(&[1.0, 0.0, 0.0, 0.0], &experts, &boundaries, 1, &cfg);
    assert!(
        inside.escalate,
        "in-scope-of-failure must escalate: {inside:?}"
    );

    // Outside the failure scope: the boundary does not apply -> B routes normally.
    let outside = route(&[0.0, 1.0, 0.0, 0.0], &experts, &boundaries, 1, &cfg);
    assert!(!outside.escalate);
    assert_eq!(outside.chosen, vec![ExpertId::new("expert:b")]);

    // Control: the same population WITHOUT the boundary routes the inside task to
    // A -- proving the boundary, not coverage, caused the escalation above.
    let control = route(&[1.0, 0.0, 0.0, 0.0], &experts, &[], 1, &cfg);
    assert!(!control.escalate);
    assert_eq!(control.chosen, vec![ExpertId::new("expert:a")]);
}
