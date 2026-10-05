use super::*;
use antumbra_core::behaviour::{content, scope_evidence, status_evidence, Example, Status};
use antumbra_core::{ExpertId, Generation, TenantId, UserId};

fn spec(rule: &str) -> Spec {
    Spec {
        rule: rule.into(),
        must: vec!["x".into()],
        must_not: vec![],
        examples: vec![Example {
            task: "t".into(),
            answer: "x".into(),
        }],
        violations: vec!["y".into()],
    }
}

fn recorded(id: &str, rule: &str, status: Status, scope: &str) -> Memory {
    Memory::new(
        id,
        TenantId::new("ws:a"),
        antumbra_core::MemoryNetwork::Opinion,
        content(&spec(rule)),
        1.0,
        chrono::Utc::now(),
    )
    .with_evidence(vec![status_evidence(status), scope_evidence(scope)])
}

fn standing(scope: &str, behaviours: &[&str]) -> Expert {
    let now = chrono::Utc::now();
    Expert {
        id: ExpertId::new(format!("expert:user:a:behaviour:{scope}")),
        name: "e".into(),
        base_model: "base".into(),
        artifact_uri: "adapters/e.safetensors".into(),
        capability_card: serde_json::json!({
            "behaviours": behaviours, "scope": scope, "standing": true,
        }),
        capability_vec: None,
        fitness: 1.0,
        frozen_at: Some(now),
        generation: Generation::ZERO,
        owner: Some(UserId::new("user:a")),
        compartment: None,
        placed_on: None,
        created_at: now,
    }
}

#[test]
fn a_scope_in_step_needs_nothing_and_one_behind_is_trained() {
    let trained = vec![recorded(
        "memory:t",
        "Rule t.",
        Status::Trained,
        "everywhere",
    )];
    let taught = behaviour::learnable(&trained, "everywhere");
    let expert = standing("everywhere", &["memory:t"]);
    assert_eq!(need(&taught, Some(&expert), None), Need::Nothing);

    let mut more = trained.clone();
    more.push(recorded(
        "memory:n",
        "Rule n.",
        Status::Accepted,
        "everywhere",
    ));
    let taught = behaviour::learnable(&more, "everywhere");
    assert_eq!(
        need(&taught, Some(&expert), None),
        Need::Train(fingerprint(&taught))
    );
}

#[test]
fn a_set_that_failed_is_not_tried_again_until_it_changes() {
    let accepted = vec![recorded(
        "memory:n",
        "Rule n.",
        Status::Accepted,
        "everywhere",
    )];
    let taught = behaviour::learnable(&accepted, "everywhere");
    let print = fingerprint(&taught);
    assert_eq!(need(&taught, None, Some(&print)), Need::Nothing);

    let edited = vec![recorded(
        "memory:n",
        "Rule n, said better.",
        Status::Accepted,
        "everywhere",
    )];
    let taught = behaviour::learnable(&edited, "everywhere");
    assert_ne!(
        fingerprint(&taught),
        print,
        "a changed rule is a different set"
    );
    assert!(matches!(need(&taught, None, Some(&print)), Need::Train(_)));
}

#[test]
fn an_expert_with_nothing_left_to_teach_is_dropped() {
    let retired = vec![recorded(
        "memory:t",
        "Rule t.",
        Status::Retired,
        "everywhere",
    )];
    let taught = behaviour::learnable(&retired, "everywhere");
    let expert = standing("everywhere", &["memory:t"]);
    assert_eq!(need(&taught, Some(&expert), None), Need::Drop);
    assert_eq!(need(&taught, None, None), Need::Nothing);
}

#[test]
fn every_scope_of_a_user_is_planned() {
    let memories = vec![
        recorded("memory:e", "Rule e.", Status::Accepted, "everywhere"),
        recorded("memory:r", "Rule r.", Status::Accepted, "github.com/o/r"),
    ];
    let experts = vec![standing("github.com/o/gone", &["memory:old"])];
    let planned: Vec<(String, bool)> =
        plan_user(&memories, &experts, &Failed::new(), "ws:a", "user:a")
            .into_iter()
            .map(|p| (p.scope, matches!(p.need, Need::Train(_))))
            .collect();
    assert_eq!(
        planned,
        [
            ("everywhere".to_string(), true),
            ("github.com/o/r".to_string(), true),
            ("github.com/o/gone".to_string(), false),
        ]
    );
    let gone = plan_user(&memories, &experts, &Failed::new(), "ws:a", "user:a")
        .into_iter()
        .find(|p| p.scope == "github.com/o/gone")
        .unwrap();
    assert_eq!(gone.need, Need::Drop);
}
