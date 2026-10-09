use super::*;

fn spec(n: usize) -> Spec {
    Spec {
        rule: "Name an issue branch feat/{issue}-{slug}.".into(),
        must: vec![r"feat/\d+-".into()],
        must_not: vec!["feature/".into()],
        examples: (0..n)
            .map(|i| Example {
                task: format!("Start issue #{i}."),
                answer: format!("git checkout -b feat/{i}-work"),
            })
            .collect(),
        violations: vec!["git checkout -b feature/work".into()],
    }
}

fn result(id: &str, passed: usize, total: usize) -> TaskResult {
    TaskResult {
        id: id.into(),
        passed,
        total,
    }
}

#[test]
fn every_fourth_example_is_held_out() {
    let ten = spec(10);
    let (train, held) = split(&ten.examples);
    assert_eq!((train.len(), held.len()), (8, 2));
    assert_eq!(held[0].task, "Start issue #3.");
    let three = spec(3);
    let (train, held) = split(&three.examples);
    assert_eq!((train.len(), held.len()), (3, 0), "too few to hold any out");
}

#[test]
fn the_fewest_examples_a_behavior_is_recorded_with_hold_one_out() {
    // A behavior recorded with fewer could never be admitted, and the keeper
    // spent ten minutes of GPU on each one before refusing it.
    let fewest = spec(antumbra_core::behavior::MIN_EXAMPLES);
    let (_, held) = split(&fewest.examples);
    assert!(!held.is_empty(), "nothing held out to admit the expert by");
}

#[test]
fn tasks_carry_the_check_the_answer_and_the_behavior() {
    let t = tasks("memory:b1", &spec(8));
    assert_eq!((t.train.len(), t.held.len()), (6, 2));
    let first = &t.train[0];
    assert_eq!(first.id, "memory:b1#t0");
    assert_eq!(first.skill.as_deref(), Some("memory:b1"));
    assert_eq!(
        first.completion.as_deref(),
        Some("git checkout -b feat/0-work")
    );
    let args = first.verify["args"].as_array().unwrap();
    assert_eq!(args[2], r#"["feat/\\d+-"]"#, "patterns ride as arguments");
    assert_eq!(args[3], r#"["feature/"]"#);
    assert!(t.held.iter().all(|h| h.id.starts_with("memory:b1#h")));
}

#[test]
fn the_shipped_replay_and_controls_load() {
    let prompts = replay_prompts();
    assert_eq!(prompts.len(), 325);
    assert!(prompts.iter().all(|p| p.verify == always()));
    let c = controls();
    assert_eq!(c.len(), 60);
    assert_eq!(
        c.iter()
            .filter(|t| t.id.starts_with("control-cmd-"))
            .count(),
        40
    );
    assert_eq!(
        c.iter()
            .filter(|t| t.id.starts_with("control-code-"))
            .count(),
        20
    );

    let answered = replay(
        &prompts[..3],
        &[
            (prompts[0].id.clone(), "git status".into()),
            (prompts[1].id.clone(), "   ".into()),
        ],
    );
    assert_eq!(
        answered.len(),
        1,
        "an empty answer and a missing one are dropped"
    );
    assert_eq!(answered[0].completion.as_deref(), Some("git status"));
}

#[test]
fn an_expert_that_learned_and_kept_the_rest_is_admitted() {
    let ids = vec!["memory:b1".to_string(), "memory:b2".to_string()];
    let base = vec![
        result("memory:b1#h0", 0, 1),
        result("memory:b1#h1", 0, 1),
        result("memory:b2#h0", 0, 1),
        result("control-cmd-0", 1, 1),
        result("control-cmd-1", 1, 1),
    ];
    let expert = vec![
        result("memory:b1#h0", 1, 1),
        result("memory:b1#h1", 1, 1),
        result("memory:b2#h0", 1, 1),
        result("control-cmd-0", 1, 1),
        result("control-cmd-1", 1, 1),
    ];
    let v = admit(&ids, &base, &expert);
    assert!(v.admitted, "{:?}", v.reasons);
    assert_eq!(
        v.controls,
        [ControlScore {
            family: "control-cmd".into(),
            base: 1.0,
            expert: 1.0
        }]
    );
    assert!(v.behaviors.iter().all(|b| b.admitted));
}

#[test]
fn an_expert_that_did_not_learn_or_broke_the_controls_is_refused() {
    let ids = vec!["memory:b1".to_string()];
    let base = vec![result("memory:b1#h0", 0, 2), result("control-cmd-0", 4, 4)];
    let unlearned = vec![result("memory:b1#h0", 1, 2), result("control-cmd-0", 4, 4)];
    let v = admit(&ids, &base, &unlearned);
    assert!(!v.admitted);
    assert!(v.missed()[0].contains("held out 0.50"), "{:?}", v.missed());
    assert!(
        v.reasons
            .iter()
            .any(|r| r.starts_with("no behavior was learned")),
        "{:?}",
        v.reasons
    );

    let interfering = vec![result("memory:b1#h0", 2, 2), result("control-cmd-0", 2, 4)];
    let v = admit(&ids, &base, &interfering);
    assert!(!v.admitted);
    assert!(
        v.reasons.iter().any(|r| r.starts_with("control-cmd fell")),
        "{:?}",
        v.reasons
    );

    let v = admit(&["memory:none".to_string()], &base, &interfering);
    assert!(v.learned().is_empty(), "no held-out tasks, nothing learned");
    assert!(!v.admitted);
    assert!(
        !admit(&[], &base, &base).admitted,
        "nothing taught, nothing admitted"
    );
}

#[test]
fn each_family_of_controls_is_held_on_its_own() {
    let ids = vec!["memory:b1".to_string()];
    let mut base = vec![result("memory:b1#h0", 0, 2)];
    let mut expert = vec![result("memory:b1#h0", 2, 2)];
    // Forty commands held, and three of twenty code answers lost: pooled, the
    // fall is 3/60, inside the slack; on its own, code fell 0.15.
    for i in 0..40 {
        base.push(result(&format!("control-cmd-{i}"), 1, 1));
        expert.push(result(&format!("control-cmd-{i}"), 1, 1));
    }
    for i in 0..20 {
        base.push(result(&format!("control-code-{i}"), 1, 1));
        expert.push(result(&format!("control-code-{i}"), usize::from(i >= 3), 1));
    }
    let v = admit(&ids, &base, &expert);
    assert!(!v.admitted);
    let families: Vec<&str> = v.controls.iter().map(|c| c.family.as_str()).collect();
    assert_eq!(families, ["control-cmd", "control-code"]);
    assert!(
        v.reasons.iter().any(|r| r.starts_with("control-code fell")),
        "{:?}",
        v.reasons
    );
}

#[test]
fn a_behavior_the_base_already_follows_holds_rather_than_rises() {
    let ids = vec!["memory:new".to_string(), "memory:known".to_string()];
    let base = vec![
        result("memory:new#h0", 0, 2),
        result("memory:known#h0", 2, 2),
        result("control-cmd-0", 1, 1),
    ];
    let expert = vec![
        result("memory:new#h0", 2, 2),
        result("memory:known#h0", 2, 2),
        result("control-cmd-0", 1, 1),
    ];
    let v = admit(&ids, &base, &expert);
    assert!(v.admitted, "{:?}", v.reasons);

    let forgot = vec![
        result("memory:new#h0", 2, 2),
        result("memory:known#h0", 1, 2),
        result("control-cmd-0", 1, 1),
    ];
    let v = admit(&ids, &base, &forgot);
    assert!(!v.admitted, "one the base followed must still be held");
    assert!(
        v.reasons
            .iter()
            .any(|r| r.starts_with("memory:known fell from 1.00 to 0.50")),
        "{:?}",
        v.reasons
    );

    let known = vec!["memory:known".to_string()];
    let v = admit(&known, &base, &expert);
    assert!(!v.admitted);
    assert!(
        v.reasons
            .iter()
            .any(|r| r.starts_with("no behavior was learned")),
        "{:?}",
        v.reasons
    );
}

#[test]
fn an_expert_is_admitted_with_what_it_learned_and_without_what_it_missed() {
    let ids = vec!["memory:learned".to_string(), "memory:missed".to_string()];
    let base = vec![
        result("memory:learned#h0", 0, 2),
        result("memory:missed#h0", 0, 2),
        result("control-cmd-0", 1, 1),
    ];
    // The one it missed scored no better than the base model, and no worse.
    let expert = vec![
        result("memory:learned#h0", 2, 2),
        result("memory:missed#h0", 1, 2),
        result("control-cmd-0", 1, 1),
    ];
    let v = admit(&ids, &base, &expert);
    assert!(v.admitted, "{:?}", v.reasons);
    assert_eq!(v.learned(), ["memory:learned"]);
    assert_eq!(v.missed().len(), 1);
    assert!(
        v.missed()[0].starts_with("memory:missed: held out 0.50 against the base's 0.00"),
        "{:?}",
        v.missed()
    );

    let broke = vec![
        result("memory:learned#h0", 2, 2),
        result("memory:missed#h0", 1, 2),
        result("control-cmd-0", 0, 1),
    ];
    assert!(
        !admit(&ids, &base, &broke).admitted,
        "the controls still hold it back"
    );
}
