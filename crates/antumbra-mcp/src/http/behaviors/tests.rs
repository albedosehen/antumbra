use super::*;
use antumbra_core::behavior::{content, scope_evidence, status_evidence, Example, Status};
use antumbra_core::{ExpertId, Generation, TenantId, UserId};

/// The ceiling the planning is given where the test is not about it.
const MOST: usize = antumbra_core::behavior::MOST_PER_EXPERT;

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
fn a_behavior_that_could_never_be_admitted_starts_no_training() -> anyhow::Result<()> {
    // Recorded with three examples, before four were required: training holds
    // none of them out, so the expert would be refused after a GPU pass.
    let mut three = recorded("memory:3", "Rule 3.", Status::Accepted, "everywhere");
    let mut short = spec("Rule 3.");
    short.examples.truncate(3);
    three.content = content(&short);
    let taught = behavior::learnable(&[three.clone()], "everywhere");
    assert!(taught.is_empty(), "nothing it could be admitted by");
    assert_eq!(need(&taught, None, None), Need::Nothing);
    let planned = plan_user(&[three], &[], &Failed::new(), "ws:a", "user:a", MOST);
    assert!(planned.iter().all(|p| p.need == Need::Nothing));
    Ok(())
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
fn a_scope_in_step_needs_nothing_and_one_behind_is_trained() -> anyhow::Result<()> {
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
    Ok(())
}

#[test]
fn a_set_that_failed_is_not_tried_again_until_it_changes() -> anyhow::Result<()> {
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
    Ok(())
}

#[test]
fn a_set_refused_before_a_restart_is_not_trained_again() -> anyhow::Result<()> {
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
                held: 1,
                worse: 0.5,
            },
        );
    }
    // A new process: nothing failed in it yet, but the store remembers.
    let planned = plan_user(&accepted, &[], &Failed::new(), "ws:a", "user:a", MOST);
    assert_eq!(planned[0].need, Need::Nothing);

    accepted.push(recorded(
        "memory:c",
        "Rule c.",
        Status::Accepted,
        "everywhere",
    ));
    let planned = plan_user(&accepted, &[], &Failed::new(), "ws:a", "user:a", MOST);
    assert!(
        matches!(planned[0].need, Need::Train(_)),
        "a behavior added since is a new set, trained"
    );
    Ok(())
}

#[test]
fn an_expert_admitted_without_a_behavior_it_missed_is_left_be() -> anyhow::Result<()> {
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
    Ok(())
}

#[test]
fn a_set_refused_while_one_missed_behavior_refused_all_is_trained_again() -> anyhow::Result<()> {
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
    Ok(())
}

#[test]
fn an_expert_with_nothing_left_to_teach_is_dropped() -> anyhow::Result<()> {
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
    Ok(())
}

#[test]
fn every_scope_of_a_user_is_planned() -> anyhow::Result<()> {
    let memories = vec![
        recorded("memory:e", "Rule e.", Status::Accepted, "everywhere"),
        recorded("memory:r", "Rule r.", Status::Accepted, "github.com/o/r"),
    ];
    let experts = vec![standing("github.com/o/gone", &["memory:old"])];
    let planned: Vec<(String, bool)> =
        plan_user(&memories, &experts, &Failed::new(), "ws:a", "user:a", MOST)
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
    let Some(gone) = plan_user(&memories, &experts, &Failed::new(), "ws:a", "user:a", MOST)
        .into_iter()
        .find(|p| p.scope == "github.com/o/gone")
    else {
        anyhow::bail!("the scope whose expert has nothing left to teach was not planned");
    };
    assert_eq!(gone.need, Need::Drop);
    Ok(())
}

#[test]
fn a_scope_is_trained_on_no_more_than_its_ceiling() -> anyhow::Result<()> {
    let accepted: Vec<Memory> = (1..=3)
        .map(|n| {
            let mut m = recorded(
                &format!("memory:{n}"),
                &format!("Rule {n}."),
                Status::Accepted,
                "everywhere",
            );
            m.created_at = chrono::Utc::now() - chrono::Duration::minutes(10 - n);
            m
        })
        .collect();
    let planned = plan_user(&accepted, &[], &Failed::new(), "ws:a", "user:a", 2);
    let Some(everywhere) = planned.iter().find(|p| p.scope == "everywhere") else {
        anyhow::bail!("everywhere was not planned");
    };
    let taught: Vec<&str> = everywhere
        .taught
        .iter()
        .map(|(m, _)| m.id.as_str())
        .collect();
    assert_eq!(taught, ["memory:1", "memory:2"]);
    assert_eq!(everywhere.waiting, 1);
    assert_eq!(
        everywhere.need,
        Need::Train(fingerprint(&everywhere.taught)),
        "trained on the two that fit, not refused for the three"
    );
    Ok(())
}

/// A scope whose expert holds more than the ceiling, after a training that
/// missed one: what it missed has no room, so the set in force is what it
/// learned. It needs nothing. Trained on that set alone it would learn
/// nothing and could lose a behavior it has; before its card kept what it
/// learned, it was trained on it all the same.
#[test]
fn a_full_expert_is_not_trained_again_on_what_it_already_learned() -> anyhow::Result<()> {
    use antumbra_core::behavior::{mark_training, Training, CARD_HELD};
    let at = |n: i64, status: Status| {
        let mut m = recorded(
            &format!("memory:{n}"),
            &format!("Rule {n}."),
            status,
            "everywhere",
        );
        m.created_at = chrono::Utc::now() - chrono::Duration::minutes(10 - n);
        m
    };
    // Trained on the three it held, at a ceiling of two; the third missed.
    let before: Vec<Memory> = (1..=3).map(|n| at(n, Status::Trained)).collect();
    let set = behavior::taught(&before, "everywhere", 2).set;
    assert_eq!(set.len(), 3, "a set never drops what the expert held");
    let mut missed = at(3, Status::Accepted);
    mark_training(
        &mut missed.evidence,
        &Training {
            set: fingerprint(&set),
            base: 0.0,
            expert: 0.5,
            learned: false,
            admitted: true,
            held: 2,
            worse: 0.1,
        },
    );
    let after = vec![at(1, Status::Trained), at(2, Status::Trained), missed];
    let mut expert = standing("everywhere", &["memory:1", "memory:2"]);
    expert.capability_card["set"] = serde_json::json!(fingerprint(&set));
    expert.capability_card["missed"] = serde_json::json!(["memory:3"]);

    // Its card does not say what it learned: judged by the set it was
    // trained on, it is behind the two in force and trained on them.
    let plan = |expert: &Expert, memories: &[Memory]| {
        plan_user(
            memories,
            std::slice::from_ref(expert),
            &Failed::new(),
            "ws:a",
            "user:a",
            2,
        )
        .into_iter()
        .find(|p| p.scope == "everywhere")
        .ok_or_else(|| anyhow::anyhow!("everywhere was not planned"))
    };
    let older = plan(&expert, &after)?;
    assert_eq!(older.taught.len(), 2);
    assert_eq!(older.waiting, 1);
    assert!(matches!(older.need, Need::Train(_)));

    // Its card says it learned exactly those two: nothing to train.
    expert.capability_card[CARD_HELD] = serde_json::json!(fingerprint(&older.taught));
    assert_eq!(plan(&expert, &after)?.need, Need::Nothing);

    // The one it missed retired, or one it learned: trained.
    let mut retired = after.clone();
    retired[2] = at(3, Status::Retired);
    assert!(matches!(plan(&expert, &retired)?.need, Need::Train(_)));
    let mut fewer = after.clone();
    fewer[1] = at(2, Status::Retired);
    assert!(matches!(plan(&expert, &fewer)?.need, Need::Train(_)));
    Ok(())
}
