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

/// Admission at the recipe's default regression confidence, 0.9.
fn admit(behaviors: &[String], base: &[TaskResult], expert: &[TaskResult]) -> Verdict {
    super::admit(behaviors, base, expert, 0.9)
}

fn result(id: &str, passed: usize, total: usize) -> TaskResult {
    TaskResult {
        id: id.into(),
        passed,
        total,
    }
}

#[test]
fn every_third_example_is_held_out() {
    let ten = spec(10);
    let (train, held) = split(&ten.examples);
    assert_eq!(
        (train.len(), held.len()),
        (7, 3),
        "three held out: one miss still clears the bar"
    );
    assert_eq!(held[0].task, "Start issue #2.");
    let three = spec(3);
    let (train, held) = split(&three.examples);
    assert_eq!((train.len(), held.len()), (2, 1));
    let two = spec(2);
    let (train, held) = split(&two.examples);
    assert_eq!((train.len(), held.len()), (2, 0), "too few to hold any out");
}

/// The closed form against what can be checked by hand, and the evidence it
/// carries: the same lost fraction says more over more tasks.
#[test]
fn how_likely_the_expert_is_worse_follows_the_evidence() {
    let close = |a: f64, b: f64| (a - b).abs() < 1e-9;
    assert!(close(prob_beta_greater(1, 1, 1, 1), 0.5), "two flat priors");
    assert!(
        close(prob_beta_greater(2, 1, 1, 2), 5.0 / 6.0),
        "one pass against one fail"
    );
    assert!(
        close(prob_beta_greater(4, 1, 4, 1), 0.5),
        "the same evidence twice"
    );
    assert!(
        close(prob_worse(2, 2, 2) + 0.0, 0.5),
        "nothing lost, nothing gained"
    );
    // One task lost of a few is not much evidence; all of them is.
    let one_of_three = prob_worse(3, 2, 3);
    assert!((0.7..0.8).contains(&one_of_three), "{one_of_three}");
    assert!(prob_worse(3, 0, 3) > 0.98);
    assert!(
        prob_worse(1, 0, 3) < 0.9,
        "the base had one of three itself: losing it is not a refusal at the default"
    );
    // The same fraction lost over more tasks is more evidence.
    let fifteen_of_eighteen = prob_worse(18, 15, 18);
    assert!(fifteen_of_eighteen > 0.9, "{fifteen_of_eighteen}");
    assert!(fifteen_of_eighteen > prob_worse(3, 2, 3));
    assert!(
        close(prob_worse(0, 0, 0), 0.5),
        "no held-out task says nothing"
    );
    // Gaining is the mirror of losing.
    assert!(close(prob_worse(2, 3, 3) + prob_worse(3, 2, 3), 1.0));
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
fn an_audit_asks_the_later_examples_under_the_behavior_s_own_check() {
    let later = spec(2).examples;
    let t = audit_tasks("memory:b1", &spec(8), &later);
    assert_eq!(t.len(), 2);
    assert_eq!(t[0].id, "memory:b1#a0");
    assert_eq!(t[1].id, "memory:b1#a1");
    assert!(t.iter().all(|t| t.skill.as_deref() == Some("memory:b1")));
    assert_eq!(
        t[1].completion.as_deref(),
        Some("git checkout -b feat/1-work")
    );
    assert_eq!(t[0].verify["behavior"]["must"], json!([r"feat/\d+-"]));
    assert!(audit_tasks("memory:b1", &spec(8), &[]).is_empty());
    // Counted under the behavior, like its held-out tasks.
    let results = [result("memory:b1#a0", 1, 1), result("memory:b1#a1", 0, 1)];
    assert_eq!(counts(&results, "memory:b1#"), (1, 2));
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
    let check = &first.verify["behavior"];
    assert_eq!(
        check["must"],
        json!([r"feat/\d+-"]),
        "the behavior's own check"
    );
    assert_eq!(check["must_not"], json!(["feature/"]));
    assert!(first.verify.get("program").is_none(), "judged in-process");
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

    // One task lost of two is four in five odds of being worse, below the
    // recipe's 0.9, so the set is admitted; both lost is near certain and
    // refuses it.
    let slipped = vec![
        result("memory:new#h0", 2, 2),
        result("memory:known#h0", 1, 2),
        result("control-cmd-0", 1, 1),
    ];
    let v = super::admit(&ids, &base, &slipped, 0.9);
    assert!(v.admitted, "{:?}", v.reasons);
    let known = v.behaviors.iter().find(|s| s.id == "memory:known").unwrap();
    assert_eq!(known.held, 2);
    assert!((0.75..0.85).contains(&known.worse), "{}", known.worse);
    let forgot = vec![
        result("memory:new#h0", 2, 2),
        result("memory:known#h0", 0, 2),
        result("control-cmd-0", 1, 1),
    ];
    let v = super::admit(&ids, &base, &forgot, 0.9);
    assert!(!v.admitted, "one the base followed must still be held");
    assert!(
        v.reasons.iter().any(
            |r| r.starts_with("memory:known fell from 1.00 to 0.00 held out: 9")
                && r.contains("% likely worse than the base model over 2 tasks")
        ),
        "{:?}",
        v.reasons
    );
    // The same evidence is read more strictly by a stricter recipe.
    let v = super::admit(&ids, &base, &slipped, 0.7);
    assert!(!v.admitted, "at 0.7, one task lost of two refuses");

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

/// What a behavior was missed on is its first held-out task the expert
/// failed, with the expert's answer. A behavior that passed every held-out
/// task has none, and neither does a control.
#[test]
fn a_behavior_is_missed_on_its_first_failed_held_out_answer() {
    let a = tasks("memory:a", &spec(6));
    let b = tasks("memory:b", &spec(6));
    let control = controls().remove(0);
    let mut held = a.held.clone();
    held.extend(b.held.clone());
    held.push(control.clone());
    let draw = |task: &CorpusTask, completion: &str, passed: bool| Draw {
        task: task.id.clone(),
        completion: completion.into(),
        passed,
    };
    let draws = vec![
        draw(&a.held[0], "git checkout -b feat/2-work", true),
        draw(&a.held[1], "git switch -c feature/work", false),
        draw(&b.held[0], "git checkout -b feat/2-work", true),
        draw(&b.held[1], "git checkout -b feat/5-work", true),
        draw(&control, "something else", false),
    ];
    let found = misses(&held, &draws);
    assert_eq!(found.len(), 1, "{found:?}");
    assert_eq!(found[0].0, "memory:a");
    assert_eq!(found[0].1.task, a.held[1].prompt);
    assert_eq!(found[0].1.answer, "git switch -c feature/work");
}
