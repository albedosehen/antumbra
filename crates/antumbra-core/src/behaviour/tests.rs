use super::*;

fn ssh() -> Spec {
    Spec {
        rule: "Reach a host by its SSH alias, never by a key and an address.".into(),
        must: vec![r"\bssh\s".into()],
        must_not: vec![
            r"\d+\.\d+\.\d+\.\d+".into(),
            r"ssh\s+-i\b".into(),
            "@".into(),
        ],
        examples: vec![
            Example {
                task: "Run df -h on kuskokwim (10.0.0.132, user nyx).".into(),
                answer: "ssh kuskokwim 'df -h'".into(),
            },
            Example {
                task: "Show uptime on shaman.".into(),
                answer: "ssh shaman uptime".into(),
            },
            Example {
                task: "List containers on aur0.".into(),
                answer: "ssh aur0 'docker ps'".into(),
            },
            Example {
                task: "Check the GPU on em0.".into(),
                answer: "ssh em0 nvidia-smi".into(),
            },
        ],
        violations: vec![
            "ssh nyx@10.0.0.132 'df -h'".into(),
            "ssh -i ~/.ssh/id_ed25519 kuskokwim".into(),
        ],
    }
}

#[test]
fn a_behaviour_with_a_discriminating_check_has_no_problems() {
    assert_eq!(ssh().problems(), Vec::<String>::new());
    assert_eq!(ssh().follows("ssh em0 nvidia-smi"), Some(true));
    assert_eq!(ssh().follows("ssh nano@10.0.0.51 nvidia-smi"), Some(false));
}

#[test]
fn a_check_that_only_refuses_is_not_enough() {
    let mut spec = ssh();
    spec.must.clear();
    let problems = spec.problems();
    assert!(
        problems.iter().any(|p| p.contains("`must` pattern")),
        "{problems:?}"
    );
}

#[test]
fn the_check_is_shown_to_discriminate_before_it_is_trusted() {
    let mut spec = ssh();
    spec.violations.push("ssh kuskokwim uptime".into());
    spec.examples[1].answer = "ssh nyx@10.0.0.130 uptime".into();
    let problems = spec.problems();
    assert!(
        problems
            .iter()
            .any(|p| p.contains("passes the violating answer `ssh kuskokwim uptime`")),
        "{problems:?}"
    );
    assert!(
        problems
            .iter()
            .any(|p| p.contains("refuses the example answer")),
        "{problems:?}"
    );
}

#[test]
fn too_few_examples_no_violations_and_bad_patterns_are_named() {
    let mut spec = ssh();
    spec.examples.truncate(1);
    spec.violations.clear();
    let problems = spec.problems();
    assert!(
        problems.iter().any(|p| p.contains("at least 4")),
        "{problems:?}"
    );
    spec = ssh();
    spec.examples.truncate(3);
    assert!(
        spec.problems().iter().any(|p| p.contains("one in four")),
        "three examples hold none out"
    );
    assert!(
        problems.iter().any(|p| p.contains("no violating answer")),
        "{problems:?}"
    );

    spec = ssh();
    spec.must.push("(unclosed".into());
    assert!(spec
        .problems()
        .iter()
        .any(|p| p.contains("does not compile")));
    assert_eq!(spec.follows("ssh kuskokwim"), None);
}

#[test]
fn lookarounds_work_as_the_trainers_checks_allow() {
    let mut spec = ssh();
    spec.must = vec![r"\bFILE_TYPE(?!_ID)\b".into()];
    assert_eq!(spec.follows("ORDER BY FILE_TYPE"), Some(true));
    assert_eq!(spec.follows("ORDER BY FILE_TYPE_ID"), Some(false));
}

#[test]
fn the_content_leads_with_the_rule_and_round_trips() {
    let spec = ssh();
    let text = content(&spec);
    assert!(text.starts_with("Reach a host by its SSH alias"));
    assert_eq!(spec_of(&text), Some(spec));
    assert_eq!(spec_of("a memory with no behaviour block"), None);
}

#[test]
fn status_scope_and_supersession_ride_in_evidence() {
    let mut evidence = vec![
        "git:github.com/a/b@abc1234".to_string(),
        status_evidence(Status::Proposed),
        scope_evidence(&normalize_scope(Some("GitHub.com/A/B"))),
    ];
    set_status(&mut evidence, Status::Accepted);
    evidence.push(supersedes_evidence("memory:old"));
    let state = State::of(&evidence).expect("a behaviour");
    assert_eq!(state.status, Status::Accepted);
    assert_eq!(state.scope, "github.com/a/b");
    assert_eq!(state.supersedes.as_deref(), Some("memory:old"));
    assert_eq!(
        evidence
            .iter()
            .filter(|e| e.starts_with("behaviour-status:"))
            .count(),
        1,
        "one status at a time"
    );
    assert_eq!(normalize_scope(None), EVERYWHERE);
    assert_eq!(
        State::of(&["git:x@abc1234".to_string()]),
        None,
        "not a behaviour"
    );
    assert_eq!(Status::parse("Retired"), Some(Status::Retired));

    mark_trained(&mut evidence, "expert:user:a:behaviour:everywhere");
    mark_trained(&mut evidence, "expert:user:a:behaviour:everywhere");
    let trained = State::of(&evidence).expect("a behaviour");
    assert_eq!(trained.status, Status::Trained);
    assert_eq!(
        trained.expert.as_deref(),
        Some("expert:user:a:behaviour:everywhere")
    );
    assert_eq!(
        evidence
            .iter()
            .filter(|e| e.starts_with("behaviour-expert:"))
            .count(),
        1,
        "one expert at a time"
    );
}

fn recorded(id: &str, status: Status, scope: &str) -> Memory {
    Memory::new(
        id,
        TenantId::new("ws:a"),
        crate::MemoryNetwork::Opinion,
        content(&ssh()),
        1.0,
        chrono::Utc::now(),
    )
    .with_evidence(vec![status_evidence(status), scope_evidence(scope)])
}

fn standing_expert(scope: &str, behaviours: &[&str]) -> Expert {
    let now = chrono::Utc::now();
    Expert {
        id: crate::ExpertId::new(format!("expert:user:a:behaviour:{scope}")),
        name: "e".into(),
        base_model: "base".into(),
        artifact_uri: "adapters/e.safetensors".into(),
        capability_card: serde_json::json!({
            "behaviours": behaviours, "scope": scope, "standing": true,
        }),
        capability_vec: None,
        fitness: 1.0,
        frozen_at: Some(now),
        generation: crate::Generation::ZERO,
        owner: Some(UserId::new("user:a")),
        compartment: None,
        placed_on: None,
        created_at: now,
    }
}

#[test]
fn a_standing_expert_learns_the_accepted_and_trained_behaviours_of_its_scope() {
    let all = vec![
        recorded("memory:accepted", Status::Accepted, EVERYWHERE),
        recorded("memory:trained", Status::Trained, EVERYWHERE),
        recorded("memory:proposed", Status::Proposed, EVERYWHERE),
        recorded("memory:retired", Status::Retired, EVERYWHERE),
        recorded("memory:other", Status::Accepted, "github.com/a/b"),
    ];
    let ids: Vec<String> = learnable(&all, EVERYWHERE)
        .into_iter()
        .map(|(m, _)| m.id.as_str().to_string())
        .collect();
    assert_eq!(ids, ["memory:accepted", "memory:trained"]);
    // One recorded before four examples were required is left out: it could
    // never be admitted.
    let mut short = ssh();
    short.examples.truncate(3);
    let mut legacy = recorded("memory:three", Status::Accepted, EVERYWHERE);
    legacy.content = content(&short);
    assert!(learnable(&[legacy], EVERYWHERE).is_empty());
    let experts = [standing_expert("github.com/c/d", &[])];
    assert_eq!(
        scopes(&all, &experts),
        [EVERYWHERE, "github.com/a/b", "github.com/c/d"]
    );
}

#[test]
fn an_expert_is_stale_when_its_behaviours_moved_on() {
    let trained = vec![recorded("memory:t", Status::Trained, EVERYWHERE)];
    let both = standing_expert(EVERYWHERE, &["memory:t"]);
    let t = learnable(&trained, EVERYWHERE);
    assert!(!stale(&t, Some(&both)), "it holds exactly what is in force");
    assert!(stale(&t, None), "something to teach and no expert");
    assert!(!stale(&[], None));

    let mut more = trained.clone();
    more.push(recorded("memory:new", Status::Accepted, EVERYWHERE));
    assert!(
        stale(&learnable(&more, EVERYWHERE), Some(&both)),
        "a new one"
    );

    let retired = vec![recorded("memory:t", Status::Retired, EVERYWHERE)];
    assert!(
        stale(&learnable(&retired, EVERYWHERE), Some(&both)),
        "nothing left to hold"
    );

    let again = vec![recorded("memory:t", Status::Accepted, EVERYWHERE)];
    assert!(
        stale(&learnable(&again, EVERYWHERE), Some(&both)),
        "recorded again since it was trained"
    );
}
