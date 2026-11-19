use super::*;
use anyhow::Context as _;

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
fn a_training_shadow_names_the_scope_it_trained() -> anyhow::Result<()> {
    let id = shadow_id(
        &TenantId::new("ws:default"),
        &UserId::new("user:default"),
        "code",
        "0123456789abcdef",
    );
    assert_eq!(
        id.as_str(),
        "shadow:ws:default:user:default:behavior:code:0123456789ab"
    );
    assert_eq!(shadow_scope(&id), Some("code"));
    // A scope with a colon of its own is read whole.
    let repo = shadow_id(
        &TenantId::new("ws:a"),
        &UserId::new("user:b"),
        "repo:acme/orders",
        "ff",
    );
    assert_eq!(shadow_scope(&repo), Some("repo:acme/orders"));
    // The generational loop's shadows name no scope.
    for other in ["shadow:g3", "shadow:run-1:g3:7", "behavior:code:ab"] {
        assert_eq!(shadow_scope(&ShadowId::new(other)), None, "{other}");
    }
    Ok(())
}

#[test]
fn a_behavior_with_a_discriminating_check_has_no_problems() -> anyhow::Result<()> {
    assert_eq!(ssh().problems(), Vec::<String>::new());
    assert_eq!(ssh().follows("ssh em0 nvidia-smi"), Some(true));
    assert_eq!(ssh().follows("ssh nano@10.0.0.51 nvidia-smi"), Some(false));
    Ok(())
}

#[test]
fn a_check_that_only_refuses_is_not_enough() -> anyhow::Result<()> {
    let mut spec = ssh();
    spec.must.clear();
    let problems = spec.problems();
    assert!(
        problems.iter().any(|p| p.contains("`must` pattern")),
        "{problems:?}"
    );
    Ok(())
}

#[test]
fn the_check_is_shown_to_discriminate_before_it_is_trusted() -> anyhow::Result<()> {
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
    Ok(())
}

#[test]
fn too_few_examples_no_violations_and_bad_patterns_are_named() -> anyhow::Result<()> {
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
        spec.problems().iter().any(|p| p.contains("one in three")),
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
    Ok(())
}

#[test]
fn lookarounds_work_as_the_trainers_checks_allow() -> anyhow::Result<()> {
    let mut spec = ssh();
    spec.must = vec![r"\bFILE_TYPE(?!_ID)\b".into()];
    assert_eq!(spec.follows("ORDER BY FILE_TYPE"), Some(true));
    assert_eq!(spec.follows("ORDER BY FILE_TYPE_ID"), Some(false));
    Ok(())
}

#[test]
fn the_content_leads_with_the_rule_and_round_trips() -> anyhow::Result<()> {
    let spec = ssh();
    let text = content(&spec);
    assert!(text.starts_with("Reach a host by its SSH alias"));
    assert_eq!(spec_of(&text), Some(spec));
    assert_eq!(spec_of("a memory with no behavior block"), None);
    Ok(())
}

#[test]
fn status_scope_and_supersession_ride_in_evidence() -> anyhow::Result<()> {
    let mut evidence = vec![
        "git:github.com/a/b@abc1234".to_string(),
        status_evidence(Status::Proposed),
        scope_evidence(&normalize_scope(Some("GitHub.com/A/B"))),
    ];
    set_status(&mut evidence, Status::Accepted);
    evidence.push(supersedes_evidence("memory:old"));
    let state = State::of(&evidence).context("a behavior")?;
    assert_eq!(state.status, Status::Accepted);
    assert_eq!(state.scope, "github.com/a/b");
    assert_eq!(state.supersedes.as_deref(), Some("memory:old"));
    assert_eq!(
        evidence
            .iter()
            .filter(|e| e.starts_with("behavior-status:"))
            .count(),
        1,
        "one status at a time"
    );
    assert_eq!(normalize_scope(None), EVERYWHERE);
    assert_eq!(
        State::of(&["git:x@abc1234".to_string()]),
        None,
        "not a behavior"
    );
    assert_eq!(Status::parse("Retired"), Some(Status::Retired));

    mark_trained(&mut evidence, "expert:user:a:behavior:everywhere");
    mark_trained(&mut evidence, "expert:user:a:behavior:everywhere");
    let trained = State::of(&evidence).context("a behavior")?;
    assert_eq!(trained.status, Status::Trained);
    assert_eq!(
        trained.expert.as_deref(),
        Some("expert:user:a:behavior:everywhere")
    );
    assert_eq!(
        evidence
            .iter()
            .filter(|e| e.starts_with("behavior-expert:"))
            .count(),
        1,
        "one expert at a time"
    );
    Ok(())
}

#[test]
fn behaviors_stored_before_the_us_spelling_are_still_read() -> anyhow::Result<()> {
    // Content, evidence and expert cards written with `behaviour`.
    let legacy_content = content(&ssh()).replace("```behavior", "```behaviour");
    assert_eq!(spec_of(&legacy_content), Some(ssh()));
    let mut evidence = vec![
        "behaviour-status:accepted".to_string(),
        "behaviour-scope:github.com/a/b".to_string(),
        "behaviour-supersedes:memory:old".to_string(),
    ];
    let state = State::of(&evidence).context("a behavior")?;
    assert_eq!(state.status, Status::Accepted);
    assert_eq!(state.scope, "github.com/a/b");
    assert_eq!(state.supersedes.as_deref(), Some("memory:old"));
    // A status set now replaces the old-spelled one rather than sitting beside it.
    set_status(&mut evidence, Status::Retired);
    assert!(evidence.iter().all(|e| !e.starts_with("behaviour-status:")));
    assert_eq!(
        State::of(&evidence).context("a behavior")?.status,
        Status::Retired
    );

    let mut card = standing_expert(EVERYWHERE, &[]);
    card.capability_card = serde_json::json!({ "behaviours": ["memory:t"], "standing": true });
    assert_eq!(taught_by(&card), ["memory:t"]);
    assert_eq!(
        compartment_id(&TenantId::new("ws:a"), &UserId::new("user:a")).as_str(),
        "comp:ws:a:user:a:behaviour",
        "the compartment keeps the id it is stored under"
    );
    Ok(())
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

fn standing_expert(scope: &str, behaviors: &[&str]) -> Expert {
    let now = chrono::Utc::now();
    Expert {
        id: crate::ExpertId::new(format!("expert:user:a:behavior:{scope}")),
        name: "e".into(),
        base_model: "base".into(),
        artifact_uri: "adapters/e.safetensors".into(),
        capability_card: serde_json::json!({
            "behaviors": behaviors, "scope": scope, "standing": true,
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
fn a_standing_expert_is_a_user_s_in_one_workspace_only() -> anyhow::Result<()> {
    let (a, b) = (TenantId::new("ws:a"), TenantId::new("ws:b"));
    let user = UserId::new("user:a");
    let mut e = standing_expert(EVERYWHERE, &[]);
    assert!(
        !is_standing_for(&e, &a, &user),
        "without its compartment, no workspace can claim it"
    );
    e.compartment = Some(compartment_id(&a, &user));
    assert!(is_standing_for(&e, &a, &user));
    assert!(
        !is_standing_for(&e, &b, &user),
        "the same user id in another workspace"
    );
    assert!(!is_standing_for(&e, &a, &UserId::new("user:b")));
    e.capability_card = serde_json::json!({ "scope": EVERYWHERE });
    assert!(!is_standing_for(&e, &a, &user), "not standing");
    Ok(())
}

#[test]
fn a_standing_expert_learns_the_accepted_and_trained_behaviors_of_its_scope() -> anyhow::Result<()>
{
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
    Ok(())
}

#[test]
fn an_expert_is_stale_when_its_behaviors_moved_on() -> anyhow::Result<()> {
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
    Ok(())
}

#[test]
fn a_set_is_known_by_its_behaviors_and_their_content_in_any_order() -> anyhow::Result<()> {
    let a = recorded("memory:a", Status::Accepted, EVERYWHERE);
    let b = recorded("memory:b", Status::Accepted, EVERYWHERE);
    let ab = learnable(&[a.clone(), b.clone()], EVERYWHERE);
    let ba = learnable(&[b.clone(), a.clone()], EVERYWHERE);
    assert_eq!(fingerprint(&ab), fingerprint(&ba));
    // Stored with each training and on the expert, so it must not move
    // between builds.
    assert_eq!(fingerprint(&ab), "d87da319b008f6da");
    let mut spec = ssh();
    spec.rule = "Reach a host by its alias.".into();
    let mut changed = b;
    changed.content = content(&spec);
    assert_ne!(
        fingerprint(&ab),
        fingerprint(&learnable(&[a, changed], EVERYWHERE)),
        "a behavior recorded again with a changed rule makes a different set"
    );
    Ok(())
}

fn trained_in(set: &str, expert: f32, learned: bool, admitted: bool) -> Training {
    Training {
        set: set.into(),
        base: 0.0,
        expert,
        learned,
        admitted,
        held: 3,
        worse: 0.02,
    }
}

#[test]
fn a_training_note_keeps_the_counts_behind_its_verdict_and_reads_older_notes() -> anyhow::Result<()>
{
    let mut evidence = Vec::new();
    let slipped = Training {
        set: "00112233aabbccdd".into(),
        base: 1.0,
        expert: 0.67,
        learned: false,
        admitted: true,
        held: 3,
        worse: 0.74,
    };
    mark_training(&mut evidence, &slipped);
    assert_eq!(
        evidence[0],
        "behavior-training:00112233aabbccdd base=1.00 expert=0.67 learned=false admitted=true held=3 worse=0.74"
    );
    assert_eq!(training(&evidence), Some(slipped.clone()));
    assert!(
        slipped.describe().starts_with(
            "not learned (held-out 0.67 against the base model's 1.00, 74% likely worse)"
        ),
        "{}",
        slipped.describe()
    );
    // A note from before the counts were kept still reads, saying nothing
    // about them.
    let older = vec![
        "behavior-training:ffeeddcc00112233 base=0.00 expert=1.00 learned=true admitted=true"
            .to_string(),
    ];
    let read = training(&older).context("an older note still reads")?;
    assert_eq!((read.held, read.worse), (0, 0.5));
    assert_eq!(
        read.describe(),
        "learned (held-out 1.00 against the base model's 0.00); its scope's expert serves it"
    );
    Ok(())
}

#[test]
fn an_audit_is_noted_beside_the_others_and_read_back_in_order() -> anyhow::Result<()> {
    let mut m = recorded("memory:a", Status::Accepted, EVERYWHERE);
    mark_training(
        &mut m.evidence,
        &trained_in("00112233aabbccdd", 1.0, true, true),
    );
    assert!(audits(&m.evidence).is_empty());
    let first = Audit {
        set: "00112233aabbccdd".into(),
        fresh: 2,
        base: 1,
        expert: 2,
        worse: Some(0.74),
    };
    let second = Audit {
        set: "00112233aabbccdd".into(),
        fresh: 1,
        base: 0,
        expert: 0,
        worse: None,
    };
    mark_audit(&mut m.evidence, &first);
    mark_audit(&mut m.evidence, &second);
    assert_eq!(
        audits(&m.evidence),
        [first.clone(), second.clone()],
        "audits keep: each is of different examples"
    );
    assert!(
        m.evidence
            .iter()
            .any(|e| e == "behavior-audit:00112233aabbccdd fresh=2 base=1 expert=2 worse=0.74"),
        "{:?}",
        m.evidence
    );
    assert!(
        m.evidence
            .iter()
            .any(|e| e == "behavior-audit:00112233aabbccdd fresh=1 base=0 expert=0"),
        "no bet known, none written"
    );
    assert_eq!(
        first.describe(),
        "audited on 2 later examples: the expert passed 2, the base model 1"
    );
    assert_eq!(
        second.describe(),
        "audited on 1 later example: the expert passed 0, the base model 0"
    );
    assert_eq!(
        m.evidence
            .iter()
            .filter(|e| e.starts_with("behavior-audit:"))
            .count(),
        2
    );
    let state = State::of(&m.evidence).context("a behavior")?;
    assert_eq!(state.status, Status::Accepted, "a note leaves the status");
    assert!(
        training(&m.evidence).is_some(),
        "and the training note it checks"
    );
    Ok(())
}

#[test]
fn the_fresh_examples_are_the_ones_a_card_was_neither_trained_nor_audited_on() -> anyhow::Result<()>
{
    let a = recorded("memory:a", Status::Trained, EVERYWHERE);
    let b = recorded("memory:b", Status::Accepted, EVERYWHERE);
    let taught = learnable(&[a.clone(), b.clone()], EVERYWHERE);
    let prints: Vec<String> = ssh().examples.iter().map(example_print).collect();
    assert_eq!(prints[0].len(), 16);
    assert_ne!(prints[0], prints[1]);
    assert_eq!(
        example_print(&ssh().examples[0]),
        prints[0],
        "a print is of the example alone"
    );

    // The card taught a's first three examples and audited nothing; b it
    // never taught.
    let card = serde_json::json!({
        "standing": true, "scope": EVERYWHERE, "set": "x",
        CARD_EXAMPLES: { "memory:a": prints[..3] },
    });
    let fresh = fresh_examples(&taught, &card);
    assert_eq!(
        fresh.len(),
        1,
        "b was not on the card: no decision to check"
    );
    assert_eq!(fresh[0].0, "memory:a");
    assert_eq!(fresh[0].1, [ssh().examples[3].clone()]);

    // Audited once, that example is no longer fresh.
    let card = serde_json::json!({
        "standing": true, "scope": EVERYWHERE, "set": "x",
        CARD_EXAMPLES: { "memory:a": prints[..3] },
        CARD_AUDITED: { "memory:a": [prints[3]] },
    });
    assert!(fresh_examples(&taught, &card).is_empty());

    // A card from before examples were kept audits nothing.
    let card = serde_json::json!({ "standing": true, "scope": EVERYWHERE, "set": "x" });
    assert!(fresh_examples(&taught, &card).is_empty());
    Ok(())
}

#[test]
fn the_bet_an_audit_checks_is_read_off_the_card_under_either_id() -> anyhow::Result<()> {
    let card = serde_json::json!({
        "standing": true, "scope": EVERYWHERE, "set": "x",
        CARD_SCORES: {
            "memory:a": { "base": 1.0, "expert": 0.67, "held": 3, "worse": 0.74, "learned": false },
            "memory:b": { "base": 0.0, "expert": 1.0, "held": 3, "worse": 0.02, "learned": true },
        },
    });
    assert_eq!(card_worse(&card, &["memory:a"]), Some(0.74));
    assert_eq!(
        card_worse(&card, &["memory:a2", "memory:a"]),
        Some(0.74),
        "the one it supersedes"
    );
    assert_eq!(card_worse(&card, &["memory:b"]), Some(0.02));
    assert_eq!(card_worse(&card, &["memory:c"]), None);
    let older = serde_json::json!({ "standing": true, "scope": EVERYWHERE, "set": "x" });
    assert_eq!(card_worse(&older, &["memory:a"]), None);
    Ok(())
}

/// Recording a behavior again is how it gains examples: the new one carries
/// the old one's id, and the card is read under both.
#[test]
fn a_behavior_recorded_again_is_audited_as_the_one_it_supersedes() -> anyhow::Result<()> {
    let mut more = ssh();
    more.examples.push(Example {
        task: "Tail the journal on aur0.".into(),
        answer: "ssh aur0 'journalctl -f'".into(),
    });
    let again = Memory::new(
        "memory:a2",
        TenantId::new("ws:a"),
        crate::MemoryNetwork::Opinion,
        content(&more),
        1.0,
        chrono::Utc::now(),
    )
    .with_evidence(vec![
        status_evidence(Status::Accepted),
        scope_evidence(EVERYWHERE),
        supersedes_evidence("memory:a"),
    ]);
    let mut retired = recorded("memory:a", Status::Trained, EVERYWHERE);
    retired.evidence.push(status_evidence(Status::Retired));
    let taught = learnable(&[retired, again], EVERYWHERE);
    assert_eq!(taught.len(), 1, "the retired one teaches nothing");
    let prints: Vec<String> = ssh().examples.iter().map(example_print).collect();

    // The card taught the earlier one; the later one's new example is fresh.
    let card = serde_json::json!({
        "standing": true, "scope": EVERYWHERE, "set": "x",
        CARD_EXAMPLES: { "memory:a": prints },
    });
    let fresh = fresh_examples(&taught, &card);
    assert_eq!(fresh.len(), 1, "{fresh:?}");
    assert_eq!(fresh[0].0, "memory:a2", "noted on the one now taught");
    assert_eq!(fresh[0].1, [more.examples[4].clone()]);

    // Audited under either id, it is not fresh again.
    let print = example_print(&more.examples[4]);
    for id in ["memory:a", "memory:a2"] {
        let card = serde_json::json!({
            "standing": true, "scope": EVERYWHERE, "set": "x",
            CARD_EXAMPLES: { "memory:a": prints },
            CARD_AUDITED: { id: [print] },
        });
        assert!(fresh_examples(&taught, &card).is_empty(), "{id}");
    }

    // A card that taught neither says nothing about it.
    let card = serde_json::json!({
        "standing": true, "scope": EVERYWHERE, "set": "x",
        CARD_EXAMPLES: { "memory:other": prints },
    });
    assert!(fresh_examples(&taught, &card).is_empty());
    Ok(())
}

#[test]
fn a_training_is_noted_on_each_behavior_and_read_back() -> anyhow::Result<()> {
    let mut m = recorded("memory:a", Status::Accepted, EVERYWHERE);
    // A refusal noted under the earlier rule goes with the next note.
    m.evidence
        .push("behavior-refused:00112233aabbccdd base=0.00 expert=0.00 learned=false".into());
    let first = trained_in("00112233aabbccdd", 0.5, false, false);
    mark_training(&mut m.evidence, &first);
    assert_eq!(training(&m.evidence), Some(first));
    assert!(!m.evidence.iter().any(|e| e.starts_with(REFUSED)));
    let second = trained_in("ffeeddcc00112233", 1.0, true, true);
    mark_training(&mut m.evidence, &second);
    assert_eq!(
        training(&m.evidence),
        Some(second.clone()),
        "the later one replaces it"
    );
    assert_eq!(
        m.evidence
            .iter()
            .filter(|e| e.starts_with(TRAINING))
            .count(),
        1
    );
    assert_eq!(
        second.describe(),
        "learned (held-out 1.00 against the base model's 0.00); its scope's expert serves it"
    );
    assert_eq!(
        State::of(&m.evidence).context("a behavior")?.status,
        Status::Accepted,
        "a note leaves the status as it was"
    );

    mark_trained(&mut m.evidence, "expert:user:a:behavior:everywhere");
    assert_eq!(
        training(&m.evidence),
        Some(second),
        "and so does training it"
    );
    mark_untrained(&mut m.evidence);
    let state = State::of(&m.evidence).context("a behavior")?;
    assert_eq!((state.status, state.expert), (Status::Accepted, None));
    Ok(())
}

#[test]
fn a_set_is_refused_only_when_every_behavior_in_it_notes_its_expert_refused() -> anyhow::Result<()>
{
    let mut a = recorded("memory:a", Status::Accepted, EVERYWHERE);
    let mut b = recorded("memory:b", Status::Accepted, EVERYWHERE);
    let set = fingerprint(&learnable(&[a.clone(), b.clone()], EVERYWHERE));
    assert!(!refused(&learnable(&[a.clone(), b.clone()], EVERYWHERE)));
    mark_training(&mut a.evidence, &trained_in(&set, 0.0, false, false));
    assert!(
        !refused(&learnable(&[a.clone(), b.clone()], EVERYWHERE)),
        "b was not in that training"
    );
    mark_training(&mut b.evidence, &trained_in(&set, 1.0, true, false));
    assert!(refused(&learnable(&[a.clone(), b.clone()], EVERYWHERE)));

    let c = recorded("memory:c", Status::Accepted, EVERYWHERE);
    assert!(
        !refused(&learnable(&[a.clone(), b.clone(), c], EVERYWHERE)),
        "a behavior added since makes a new set"
    );
    assert!(!refused(&[]));

    mark_training(&mut a.evidence, &trained_in(&set, 0.0, false, true));
    mark_training(&mut b.evidence, &trained_in(&set, 1.0, true, true));
    assert!(
        !refused(&learnable(&[a, b], EVERYWHERE)),
        "an admitted expert is no refusal"
    );
    Ok(())
}

#[test]
fn an_expert_that_records_its_set_is_in_step_until_the_set_changes() -> anyhow::Result<()> {
    let memories = vec![
        recorded("memory:l", Status::Trained, EVERYWHERE),
        recorded("memory:m", Status::Accepted, EVERYWHERE),
    ];
    let taught = learnable(&memories, EVERYWHERE);
    let mut expert = standing_expert(EVERYWHERE, &["memory:l"]);
    expert.capability_card["set"] = serde_json::json!(fingerprint(&taught));
    assert_eq!(trained_set(&expert), Some(fingerprint(&taught).as_str()));
    assert!(
        !stale(&taught, Some(&expert)),
        "memory:m is accepted and not held, but it was in the set trained"
    );

    let retired = vec![
        recorded("memory:l", Status::Trained, EVERYWHERE),
        recorded("memory:m", Status::Retired, EVERYWHERE),
    ];
    assert!(stale(&learnable(&retired, EVERYWHERE), Some(&expert)));
    assert!(stale(&[], Some(&expert)), "nothing left to hold");
    Ok(())
}

#[test]
fn a_behavior_cites_the_memories_it_was_drawn_from_in_its_evidence() -> anyhow::Result<()> {
    let evidence = vec![
        status_evidence(Status::Proposed),
        scope_evidence(EVERYWHERE),
        source_evidence("memory:a"),
        source_evidence("memory:b"),
    ];
    let state = State::of(&evidence).context("a behavior")?;
    assert_eq!(state.sources, ["memory:a", "memory:b"]);
    assert_eq!(sources(&evidence), ["memory:a", "memory:b"]);
    assert!(
        State::of(&[status_evidence(Status::Accepted)])
            .context("a behavior")?
            .sources
            .is_empty(),
        "one the user stated cites nothing"
    );
    Ok(())
}

/// An example answer may be a fenced code block of its own: a Mermaid
/// diagram, a Rust snippet. Its fences sit inside the spec's JSON, and the
/// spec still reads back whole. Three behaviors recorded on a live store were
/// stored and never listed or trained before this held.
#[test]
fn a_spec_whose_examples_hold_code_fences_reads_back_whole() -> anyhow::Result<()> {
    let mut spec = ssh();
    spec.examples.push(Example {
        task: "Draw the load, pose and render flow.".into(),
        answer: "```mermaid\nflowchart LR\n  load --> pose --> render\n```".into(),
    });
    spec.examples.push(Example {
        task: "Read the token file.".into(),
        answer: "```rust\nlet token = std::fs::read_to_string(&path)?;\n```".into(),
    });
    spec.violations
        .push("```\nload -> pose -> render\n```".into());
    let text = content(&spec);
    assert_eq!(spec_of(&text), Some(spec));
    Ok(())
}

#[test]
fn a_recipe_defaults_to_what_nine_behaviors_were_learned_with_and_refuses_nonsense(
) -> anyhow::Result<()> {
    let d = Recipe::default();
    assert_eq!((d.rank, d.rounds), (16, 3));
    assert!((d.learning_rate - 1.5e-4).abs() < 1e-12);
    assert!((d.regression_confidence - 0.9).abs() < 1e-12);
    assert_eq!(d.describe(), "rank 16, 3 rounds, 1.5e-4, refusing at 0.90");
    assert_eq!(
        Recipe::new(32, 1, 1e-4, 0.95)
            .map_err(anyhow::Error::msg)?
            .describe(),
        "rank 32, 1 round, 1e-4, refusing at 0.95"
    );
    assert!(
        Recipe::new(0, 3, 1.5e-4, 0.9).is_err(),
        "rank 0 trains nothing"
    );
    assert!(Recipe::new(512, 3, 1.5e-4, 0.9).is_err());
    assert!(Recipe::new(16, 0, 1.5e-4, 0.9).is_err(), "no rounds");
    assert!(Recipe::new(16, 3, 0.0, 0.9).is_err(), "no learning");
    assert!(
        Recipe::new(16, 3, 0.5, 0.9).is_err(),
        "far past where 3e-4 broke nine"
    );
    assert!(
        Recipe::new(16, 3, 1.5e-4, 0.5).is_err(),
        "a coin toss would refuse"
    );
    assert!(
        Recipe::new(16, 3, 1.5e-4, 1.0).is_err(),
        "nothing would ever refuse"
    );
    Ok(())
}

/// A training's seed follows what it teaches: the same behavior under another
/// id seeds the same, the set's order does not matter, and a behavior changed
/// seeds another. The set's print keeps the ids; the seed does not.
#[test]
fn a_training_is_seeded_from_what_it_teaches_not_where_it_is_stored() -> anyhow::Result<()> {
    let taught = |id: &str| (recorded(id, Status::Accepted, EVERYWHERE), ssh());
    let here = vec![taught("memory:a")];
    let elsewhere = vec![taught("memory:elsewhere")];
    assert_eq!(seed(&here), seed(&elsewhere));
    assert_ne!(fingerprint(&here), fingerprint(&elsewhere));

    let mut changed = taught("memory:b");
    changed.0.content.push_str(" Changed.");
    let two = vec![here[0].clone(), changed.clone()];
    let swapped = vec![changed, here[0].clone()];
    assert_eq!(seed(&two), seed(&swapped), "order does not matter");
    assert_ne!(seed(&two), seed(&here));
    Ok(())
}

// A proposed behavior the server may accept for its user: one that can be
// taught, left untouched for as long as the workspace sets.
#[test]
fn a_proposed_behavior_is_due_for_acceptance_once_it_has_waited() -> anyhow::Result<()> {
    let now = chrono::Utc::now();
    let day = chrono::Duration::hours(24);
    let mut waited = recorded("memory:waited", Status::Proposed, EVERYWHERE);
    waited.updated_at = now - chrono::Duration::hours(25);
    assert!(due_for_acceptance(&waited, day, now));
    // Not yet: it was recorded, or recorded again, an hour ago.
    let mut fresh = waited.clone();
    fresh.updated_at = now - chrono::Duration::hours(1);
    assert!(!due_for_acceptance(&fresh, day, now));
    // A wait of nothing leaves accepting to the user.
    assert!(!due_for_acceptance(&waited, chrono::Duration::zero(), now));
    // What is accepted, trained or retired is not the server's to accept.
    for status in [Status::Accepted, Status::Trained, Status::Retired] {
        let mut settled = recorded("memory:settled", status, EVERYWHERE);
        settled.updated_at = waited.updated_at;
        assert!(!due_for_acceptance(&settled, day, now), "{status:?}");
    }
    // One that could not be taught is left for its user to mend.
    let mut lacking = waited.clone();
    let mut spec = ssh();
    spec.examples.truncate(1);
    lacking.content = content(&spec);
    assert!(!due_for_acceptance(&lacking, day, now));
    // And a memory that is no behavior is nobody's to accept.
    let plain = Memory::new(
        "memory:plain",
        TenantId::new("ws:a"),
        crate::MemoryNetwork::World,
        "a fact",
        1.0,
        waited.updated_at,
    );
    assert!(!due_for_acceptance(&plain, day, now));
    Ok(())
}

/// A behavior recorded `minutes_ago`, so the order the set fills in is known.
fn recorded_at(id: &str, status: Status, minutes_ago: i64) -> Memory {
    let mut m = recorded(id, status, EVERYWHERE);
    m.created_at = chrono::Utc::now() - chrono::Duration::minutes(minutes_ago);
    m
}

/// `m` with a note that its scope's last training read the base model at
/// `base` on it, and whether it was learned.
fn noted(mut m: Memory, base: f32, learned: bool) -> Memory {
    let training = Training {
        base,
        ..trained_in("set-before", if learned { 1.0 } else { 0.0 }, learned, true)
    };
    mark_training(&mut m.evidence, &training);
    m
}

fn ids(set: &[(Memory, Spec)]) -> Vec<&str> {
    set.iter().map(|(m, _)| m.id.as_str()).collect()
}

#[test]
fn a_set_holds_what_the_expert_knows_then_the_oldest_untried_then_the_missed() {
    let all = vec![
        recorded_at("memory:trained", Status::Trained, 500),
        noted(
            recorded_at("memory:missed-old", Status::Accepted, 400),
            0.0,
            false,
        ),
        recorded_at("memory:new", Status::Accepted, 10),
        recorded_at("memory:old", Status::Accepted, 300),
        recorded_at("memory:middle", Status::Accepted, 200),
        recorded_at("memory:proposed", Status::Proposed, 900),
    ];
    let taught = taught(&all, EVERYWHERE, 3);
    // What the expert holds, then the two oldest never tried. A behavior it
    // was taught and missed gives its room to one not yet tried, however old.
    assert_eq!(
        ids(&taught.set),
        ["memory:trained", "memory:old", "memory:middle"]
    );
    assert_eq!(ids(&taught.waiting), ["memory:new", "memory:missed-old"]);
    assert!(taught.followed.is_empty());
    // With room for all, the missed one is taught again too, last.
    let roomy = super::taught(&all, EVERYWHERE, 10);
    assert_eq!(roomy.set.len(), 5);
    assert_eq!(
        roomy.set.last().map(|(m, _)| m.id.as_str()),
        Some("memory:missed-old")
    );
    assert!(roomy.waiting.is_empty());
}

#[test]
fn a_set_never_drops_what_the_expert_already_learned() {
    let all: Vec<Memory> = (0..5)
        .map(|i| recorded_at(&format!("memory:trained-{i}"), Status::Trained, 100 - i))
        .chain([recorded_at("memory:accepted", Status::Accepted, 1)])
        .collect();
    // Five learned, a ceiling of three: all five stay, and nothing is added,
    // since a set that dropped one would unteach it.
    let taught = taught(&all, EVERYWHERE, 3);
    assert_eq!(taught.set.len(), 5);
    assert_eq!(ids(&taught.waiting), ["memory:accepted"]);
    // The same set from pass to pass, so a full expert is not retrained.
    assert_eq!(
        fingerprint(&taught.set),
        fingerprint(&super::taught(&all, EVERYWHERE, 3).set)
    );
}

/// An expert as its training leaves it: taught `set` with `missed` among
/// them not learned, its card recording both prints.
fn trained_on(set: &[(Memory, Spec)], missed: &[&str]) -> Expert {
    let learned: Vec<(Memory, Spec)> = set
        .iter()
        .filter(|(m, _)| !missed.contains(&m.id.as_str()))
        .cloned()
        .collect();
    let mut expert = standing_expert(EVERYWHERE, &ids(&learned));
    expert.capability_card["set"] = serde_json::json!(fingerprint(set));
    expert.capability_card[CARD_HELD] = serde_json::json!(fingerprint(&learned));
    expert.capability_card["missed"] = serde_json::json!(missed);
    expert
}

#[test]
fn an_expert_that_holds_the_set_in_force_is_not_behind_it() {
    // Four taught at a ceiling of three (three it held already), one missed.
    let before = vec![
        recorded_at("memory:a", Status::Trained, 400),
        recorded_at("memory:b", Status::Trained, 300),
        recorded_at("memory:c", Status::Trained, 200),
        recorded_at("memory:d", Status::Trained, 100),
    ];
    let expert = trained_on(&taught(&before, EVERYWHERE, 3).set, &["memory:d"]);

    // After it, the missed one is accepted again and has no room: the set in
    // force is the three it learned.
    let after = |d: Memory| {
        vec![
            recorded_at("memory:a", Status::Trained, 400),
            recorded_at("memory:b", Status::Trained, 300),
            recorded_at("memory:c", Status::Trained, 200),
            d,
        ]
    };
    let missed = || noted(recorded_at("memory:d", Status::Accepted, 100), 0.0, false);
    let now = taught(&after(missed()), EVERYWHERE, 3);
    assert_eq!(ids(&now.set), ["memory:a", "memory:b", "memory:c"]);
    assert_eq!(ids(&now.waiting), ["memory:d"]);
    assert!(
        stale(&now.set, Some(&expert)),
        "not the set it was trained on"
    );
    assert!(holds(&now, &expert), "but exactly what it learned of it");
    assert!(!behind(&now, Some(&expert)));

    // The missed one retired: a change the expert is trained for.
    let retired = taught(
        &after(recorded_at("memory:d", Status::Retired, 100)),
        EVERYWHERE,
        3,
    );
    assert!(
        behind(&retired, Some(&expert)),
        "what it missed was retired"
    );

    // Room for the missed one again: it is taught again.
    let roomy = taught(&after(missed()), EVERYWHERE, 4);
    assert_eq!(roomy.set.len(), 4);
    assert!(!behind(&roomy, Some(&expert)), "the set it was trained on");
    let mut wider = after(missed());
    wider.push(recorded_at("memory:e", Status::Accepted, 50));
    assert!(
        behind(&taught(&wider, EVERYWHERE, 4), Some(&expert)),
        "one never tried takes the room"
    );

    // One it learned retired, or recorded again: not what it holds.
    let mut fewer = after(missed());
    fewer[2] = recorded_at("memory:c", Status::Retired, 200);
    assert!(behind(&taught(&fewer, EVERYWHERE, 3), Some(&expert)));
    let mut again = after(missed());
    again[1].content = format!("{} ", again[1].content);
    assert!(behind(&taught(&again, EVERYWHERE, 3), Some(&expert)));

    // An expert minted before its card kept what it learned is judged by the
    // set it was trained on, as before.
    let mut older = expert.clone();
    if let Some(card) = older.capability_card.as_object_mut() {
        card.remove(CARD_HELD);
    }
    assert!(behind(&now, Some(&older)));
    assert!(behind(&now, None), "no expert at all");
    assert!(!behind(&Taught::default(), None), "and nothing to teach");
}

#[test]
fn a_behavior_the_base_already_follows_is_left_out_of_the_set() {
    let all = vec![
        noted(
            recorded_at("memory:the-base-had-it", Status::Trained, 50),
            1.0,
            true,
        ),
        noted(
            recorded_at("memory:at-the-bar", Status::Accepted, 40),
            LEARNED_AT,
            false,
        ),
        noted(
            recorded_at("memory:below", Status::Accepted, 30),
            0.33,
            false,
        ),
        recorded_at("memory:untried", Status::Accepted, 20),
    ];
    let taught = taught(&all, EVERYWHERE, 10);
    assert_eq!(
        ids(&taught.followed),
        ["memory:the-base-had-it", "memory:at-the-bar"]
    );
    assert_eq!(ids(&taught.set), ["memory:untried", "memory:below"]);
    // Read from the scope's last training only: a behavior never trained
    // says nothing about the base yet.
    assert!(!followed_by_base(&all[3].evidence));
    // And it takes no room: two left out, two taught, under a ceiling of two.
    assert_eq!(super::taught(&all, EVERYWHERE, 2).set.len(), 2);
}

#[test]
fn the_ceiling_admission_and_the_base_rule_share_one_bar() {
    // Admission's learned bar and the base rule read the same number.
    assert!((LEARNED_AT - 0.6).abs() < f32::EPSILON);
    assert_eq!(MOST_PER_EXPERT, 10);
    assert!(taught(&[], EVERYWHERE, MOST_PER_EXPERT).set.is_empty());
}
