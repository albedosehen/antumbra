use super::*;
use antumbra_core::behavior::{content, scope_evidence, status_evidence, Example, Status};
use antumbra_core::{ExpertId, Generation, TenantId, UserId};

fn spec(rule: &str) -> Spec {
    Spec {
        rule: rule.into(),
        must: vec!["x".into()],
        must_not: vec![],
        examples: (0..4)
            .map(|i| Example {
                task: format!("t{i}"),
                answer: "x".into(),
            })
            .collect(),
        violations: vec!["y".into()],
    }
}

#[test]
fn a_behavior_that_could_never_be_admitted_starts_no_training() {
    // Recorded with three examples, before four were required: training holds
    // none of them out, so the expert would be refused after a GPU pass.
    let mut three = recorded("memory:3", "Rule 3.", Status::Accepted, "everywhere");
    let mut short = spec("Rule 3.");
    short.examples.truncate(3);
    three.content = content(&short);
    let taught = behavior::learnable(&[three.clone()], "everywhere");
    assert!(taught.is_empty(), "nothing it could be admitted by");
    assert_eq!(need(&taught, None, None), Need::Nothing);
    let planned = plan_user(&[three], &[], &Failed::new(), "ws:a", "user:a");
    assert!(planned.iter().all(|p| p.need == Need::Nothing));
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

fn standing(scope: &str, behaviors: &[&str]) -> Expert {
    let now = chrono::Utc::now();
    Expert {
        id: ExpertId::new(format!("expert:user:a:behavior:{scope}")),
        name: "e".into(),
        base_model: "base".into(),
        artifact_uri: "adapters/e.safetensors".into(),
        capability_card: serde_json::json!({
            "behaviors": behaviors, "scope": scope, "standing": true,
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
    let taught = behavior::learnable(&trained, "everywhere");
    let expert = standing("everywhere", &["memory:t"]);
    assert_eq!(need(&taught, Some(&expert), None), Need::Nothing);

    let mut more = trained.clone();
    more.push(recorded(
        "memory:n",
        "Rule n.",
        Status::Accepted,
        "everywhere",
    ));
    let taught = behavior::learnable(&more, "everywhere");
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
    let taught = behavior::learnable(&accepted, "everywhere");
    let print = fingerprint(&taught);
    assert_eq!(need(&taught, None, Some(&print)), Need::Nothing);

    let edited = vec![recorded(
        "memory:n",
        "Rule n, said better.",
        Status::Accepted,
        "everywhere",
    )];
    let taught = behavior::learnable(&edited, "everywhere");
    assert_ne!(
        fingerprint(&taught),
        print,
        "a changed rule is a different set"
    );
    assert!(matches!(need(&taught, None, Some(&print)), Need::Train(_)));
}

#[test]
fn a_set_refused_before_a_restart_is_not_trained_again() {
    let mut accepted = vec![
        recorded("memory:a", "Rule a.", Status::Accepted, "everywhere"),
        recorded("memory:b", "Rule b.", Status::Accepted, "everywhere"),
    ];
    let set = fingerprint(&behavior::learnable(&accepted, "everywhere"));
    for m in &mut accepted {
        behavior::mark_training(
            &mut m.evidence,
            &behavior::Training {
                set: set.clone(),
                base: 0.0,
                expert: 0.0,
                learned: false,
                admitted: false,
            },
        );
    }
    // A new process: nothing failed in it yet, but the store remembers.
    let planned = plan_user(&accepted, &[], &Failed::new(), "ws:a", "user:a");
    assert_eq!(planned[0].need, Need::Nothing);

    accepted.push(recorded(
        "memory:c",
        "Rule c.",
        Status::Accepted,
        "everywhere",
    ));
    let planned = plan_user(&accepted, &[], &Failed::new(), "ws:a", "user:a");
    assert!(
        matches!(planned[0].need, Need::Train(_)),
        "a behavior added since is a new set, trained"
    );
}

#[test]
fn an_expert_admitted_without_a_behavior_it_missed_is_left_be() {
    let memories = vec![
        recorded("memory:l", "Rule l.", Status::Trained, "everywhere"),
        recorded("memory:m", "Rule m.", Status::Accepted, "everywhere"),
    ];
    let taught = behavior::learnable(&memories, "everywhere");
    let mut expert = standing("everywhere", &["memory:l"]);
    expert.capability_card["set"] = serde_json::json!(fingerprint(&taught));
    assert_eq!(
        need(&taught, Some(&expert), None),
        Need::Nothing,
        "the one it missed is accepted, but the set is the one it was trained on"
    );

    let mut more = memories.clone();
    more.push(recorded(
        "memory:n",
        "Rule n.",
        Status::Accepted,
        "everywhere",
    ));
    let taught = behavior::learnable(&more, "everywhere");
    assert!(
        matches!(need(&taught, Some(&expert), None), Need::Train(_)),
        "a behavior added since is a new set"
    );
}

#[test]
fn a_set_refused_while_one_missed_behavior_refused_all_is_trained_again() {
    // Noted under the earlier rule, when one behavior not learned kept the
    // rest out: today the same set may be admitted.
    let mut accepted = vec![recorded(
        "memory:a",
        "Rule a.",
        Status::Accepted,
        "everywhere",
    )];
    let set = fingerprint(&behavior::learnable(&accepted, "everywhere"));
    accepted[0].evidence.push(format!(
        "behavior-refused:{set} base=0.00 expert=0.00 learned=false"
    ));
    let taught = behavior::learnable(&accepted, "everywhere");
    assert!(matches!(need(&taught, None, None), Need::Train(_)));
}

#[test]
fn an_expert_with_nothing_left_to_teach_is_dropped() {
    let retired = vec![recorded(
        "memory:t",
        "Rule t.",
        Status::Retired,
        "everywhere",
    )];
    let taught = behavior::learnable(&retired, "everywhere");
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
