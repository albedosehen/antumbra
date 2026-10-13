
use super::*;

#[test]
fn a_standing_expert_trains_at_the_rate_nine_behaviors_held_at() {
    let cfg = standing_config(&behavior::Recipe::default());
    assert!(cfg.learning_rate <= 1.5e-4, "3e-4 broke nine behaviors");
    assert_eq!((cfg.rounds, cfg.max_new_tokens), (3, 256));
    assert_eq!(cfg.lora_rank, RaftConfig::default().lora_rank);
}

/// A recipe at another rank keeps the scale every expert is trained and
/// served at: alpha moves with the rank.
#[test]
fn a_recipe_moves_alpha_with_the_rank_and_keeps_the_scale() {
    let recipe = behavior::Recipe::new(32, 5, 1e-4, 0.9).unwrap();
    let cfg = standing_config(&recipe);
    assert_eq!((cfg.lora_rank, cfg.rounds), (32, 5));
    assert!((cfg.learning_rate - 1e-4).abs() < 1e-12);
    assert!((cfg.lora_scale() - RaftConfig::default().lora_scale()).abs() < 1e-12);
    assert_eq!(cfg.max_new_tokens, 256);
}

#[test]
fn a_standing_expert_is_named_for_its_workspace_owner_and_scope() {
    let user = UserId::new("user:owner");
    let (a, b) = (TenantId::new("ws:a"), TenantId::new("ws:b"));
    assert_eq!(
        expert_id(&a, &user, "everywhere").as_str(),
        "expert:ws:a:user:owner:behavior:everywhere"
    );
    assert_ne!(
        expert_id(&a, &user, "everywhere"),
        expert_id(&b, &user, "everywhere"),
        "a user id once shared by every hosted workspace's first user"
    );
}

/// An admitted training of `ids` in `scope`, every behavior learned.
fn admitted(ids: &[&str], adapter: &str) -> Trained {
    let learned = |id: &&str| antumbra_train::behave::BehaviorScore {
        id: id.to_string(),
        base: 0.0,
        expert: 1.0,
        admitted: true,
        held: 3,
        worse: 0.02,
    };
    Trained {
        verdict: Verdict {
            admitted: true,
            behaviors: ids.iter().map(learned).collect(),
            controls: Vec::new(),
            reasons: Vec::new(),
        },
        replay: 0,
        adapter_uri: adapter.into(),
        base_model: "base".into(),
        fitness: 1.0,
        capability_vec: vec![0.0; EMBED_DIM],
        recipe: behavior::Recipe::default(),
    }
}

/// A plan for `tenant`'s `user:owner` everywhere, of one behavior held in
/// memory only: minting needs nothing from the store but the experts.
fn owner_plan(tenant: &str) -> Plan {
    use antumbra_core::behavior::{content, Example};
    let spec = Spec {
        rule: "A rule.".into(),
        must: vec!["x".into()],
        must_not: vec![],
        examples: (0..4)
            .map(|i| Example {
                task: format!("t{i}"),
                answer: "x".into(),
            })
            .collect(),
        violations: vec!["y".into()],
    };
    let m = Memory::new(
        "memory:rule",
        TenantId::new(tenant),
        antumbra_core::MemoryNetwork::Opinion,
        content(&spec),
        1.0,
        Utc::now(),
    );
    Plan {
        tenant: TenantId::new(tenant),
        user: UserId::new("user:owner"),
        scope: "everywhere".into(),
        taught: vec![(m, spec)],
    }
}

#[tokio::test]
async fn two_workspaces_with_the_same_user_keep_their_own_standing_experts() {
    let store = Store::connect_memory(EMBED_DIM).await.unwrap();
    let (a, b) = (owner_plan("ws:a"), owner_plan("ws:b"));
    let (ea, _) = mint(
        &store,
        &a,
        admitted(&["memory:rule"], "adapters/a.safetensors"),
        "rig",
    )
    .await
    .unwrap()
    .expert
    .unwrap();
    let (eb, _) = mint(
        &store,
        &b,
        admitted(&["memory:rule"], "adapters/b.safetensors"),
        "rig",
    )
    .await
    .unwrap()
    .expert
    .unwrap();
    assert_ne!(ea, eb);
    let ea = expert::get(&store, &ea).await.unwrap().expect("a's kept");
    let eb = expert::get(&store, &eb).await.unwrap().expect("b's kept");
    assert_eq!(ea.artifact_uri, "adapters/a.safetensors");
    assert_eq!(eb.artifact_uri, "adapters/b.safetensors");
    let owner = UserId::new("user:owner");
    assert!(behavior::is_standing_for(&ea, &a.tenant, &owner));
    assert!(!behavior::is_standing_for(&ea, &b.tenant, &owner));
    assert!(behavior::is_standing_for(&eb, &b.tenant, &owner));
}

#[tokio::test]
async fn minting_retires_this_workspace_s_expert_under_the_old_id_and_no_other() {
    let store = Store::connect_memory(EMBED_DIM).await.unwrap();
    let owner = UserId::new("user:owner");
    let (a, b) = (owner_plan("ws:a"), owner_plan("ws:b"));
    let old = |tenant: &TenantId| Expert {
        id: legacy_expert_id(&owner, "everywhere"),
        name: "old".into(),
        base_model: "base".into(),
        artifact_uri: "adapters/old.safetensors".into(),
        capability_card: serde_json::json!({ "standing": true, "scope": "everywhere" }),
        capability_vec: None,
        fitness: 1.0,
        frozen_at: Some(Utc::now()),
        generation: Generation::ZERO,
        owner: Some(owner.clone()),
        compartment: Some(behavior::compartment_id(tenant, &owner)),
        placed_on: Some("rig".into()),
        created_at: Utc::now(),
    };
    let legacy = legacy_expert_id(&owner, "everywhere");

    // Another workspace's, under the old id: minting here leaves it.
    expert::insert(&store, &old(&b.tenant)).await.unwrap();
    mint(
        &store,
        &a,
        admitted(&["memory:rule"], "adapters/a.safetensors"),
        "rig",
    )
    .await
    .unwrap();
    assert!(
        expert::get(&store, &legacy).await.unwrap().is_some(),
        "not this workspace's to retire"
    );

    // This workspace's, under the old id: minting here retires it.
    expert::delete(&store, &legacy).await.unwrap();
    expert::insert(&store, &old(&a.tenant)).await.unwrap();
    mint(
        &store,
        &a,
        admitted(&["memory:rule"], "adapters/a2.safetensors"),
        "rig",
    )
    .await
    .unwrap();
    assert!(expert::get(&store, &legacy).await.unwrap().is_none());
    let now = expert::get(&store, &expert_id(&a.tenant, &owner, "everywhere"))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(now.artifact_uri, "adapters/a2.safetensors");
}

#[tokio::test]
async fn mint_marks_only_the_behaviors_still_as_they_were_taught() {
    use antumbra_core::behavior::{content, scope_evidence, status_evidence, Example};
    let store = Store::connect_memory(EMBED_DIM).await.unwrap();
    let (tenant, user) = (TenantId::new("ws:a"), UserId::new("user:a"));
    let spec = Spec {
        rule: "A rule.".into(),
        must: vec!["x".into()],
        must_not: vec![],
        examples: (0..4)
            .map(|i| Example {
                task: format!("t{i}"),
                answer: "x".into(),
            })
            .collect(),
        violations: vec!["y".into()],
    };
    let comp = behavior::compartment_id(&tenant, &user);
    let put = |id: &str, status: Status| {
        Memory::new(
            id,
            tenant.clone(),
            antumbra_core::MemoryNetwork::Opinion,
            content(&spec),
            1.0,
            Utc::now(),
        )
        .in_compartment(comp.clone())
        .with_evidence(vec![status_evidence(status), scope_evidence("everywhere")])
    };
    for m in [
        put("memory:kept", Status::Accepted),
        put("memory:gone", Status::Accepted),
    ] {
        memory::upsert(&store, &m).await.unwrap();
    }
    let plan = plan(&store, &tenant, &user, "everywhere")
        .await
        .unwrap()
        .unwrap();
    let mut taught = plan.behaviors();
    taught.sort();
    assert_eq!(taught, ["memory:gone", "memory:kept"]);
    // Retired while it trained.
    memory::upsert(&store, &put("memory:gone", Status::Retired))
        .await
        .unwrap();

    let learned = |id: &str| antumbra_train::behave::BehaviorScore {
        id: id.into(),
        base: 0.0,
        expert: 1.0,
        admitted: true,
        held: 3,
        worse: 0.02,
    };
    let trained = Trained {
        verdict: Verdict {
            admitted: true,
            behaviors: vec![learned("memory:kept"), learned("memory:gone")],
            controls: Vec::new(),
            reasons: Vec::new(),
        },
        replay: 0,
        adapter_uri: "adapters/x.safetensors".into(),
        base_model: "base".into(),
        fitness: 1.0,
        capability_vec: vec![0.0; EMBED_DIM],
        recipe: behavior::Recipe::default(),
    };
    let report = mint(&store, &plan, trained, "rig").await.unwrap();
    let minted = report.expert.unwrap().0;
    assert_eq!(minted.as_str(), "expert:ws:a:user:a:behavior:everywhere");
    let status = |id: &str| {
        let store = store.clone();
        let tenant = tenant.clone();
        let id = id.to_string();
        async move {
            let m = memory::get(&store, &tenant, &MemoryId::new(id))
                .await
                .unwrap()
                .unwrap();
            State::of(&m.evidence).unwrap().status
        }
    };
    assert_eq!(status("memory:kept").await, Status::Trained);
    assert_eq!(status("memory:gone").await, Status::Retired);
    let e = expert::get(&store, &minted).await.unwrap().unwrap();
    assert_eq!(
        e.placed_on.as_deref(),
        Some("rig"),
        "placed where its adapter is"
    );
}

/// A store holding two accepted behaviors everywhere, `memory:learned`
/// and `memory:missed`, planned; and a training of them where the first
/// was learned and the second scored 0.50 against the base's 0.00.
async fn learned_and_missed(admitted: bool) -> (Store, Plan, Trained) {
    use antumbra_core::behavior::{content, scope_evidence, status_evidence, Example};
    use antumbra_train::behave::BehaviorScore;
    let store = Store::connect_memory(EMBED_DIM).await.unwrap();
    let (tenant, user) = (TenantId::new("ws:a"), UserId::new("user:a"));
    let comp = behavior::compartment_id(&tenant, &user);
    for (id, rule) in [
        ("memory:learned", "Rule one."),
        ("memory:missed", "Rule two."),
    ] {
        let spec = Spec {
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
        };
        let m = Memory::new(
            id,
            tenant.clone(),
            antumbra_core::MemoryNetwork::Opinion,
            content(&spec),
            1.0,
            Utc::now(),
        )
        .in_compartment(comp.clone())
        .with_evidence(vec![
            status_evidence(Status::Accepted),
            scope_evidence("everywhere"),
        ]);
        memory::upsert(&store, &m).await.unwrap();
    }
    let plan = plan(&store, &tenant, &user, "everywhere")
        .await
        .unwrap()
        .unwrap();
    let score = |id: &str, expert: f32, admitted: bool| BehaviorScore {
        id: id.into(),
        base: 0.0,
        expert,
        admitted,
        held: 3,
        worse: 0.5,
    };
    let trained = Trained {
        verdict: Verdict {
            admitted,
            behaviors: vec![
                score("memory:learned", 1.0, true),
                score("memory:missed", 0.5, false),
            ],
            controls: Vec::new(),
            reasons: if admitted {
                Vec::new()
            } else {
                vec!["control-cmd fell from 1.00 to 0.50".into()]
            },
        },
        replay: 0,
        adapter_uri: "adapters/x.safetensors".into(),
        base_model: "base".into(),
        fitness: 0.5,
        capability_vec: vec![0.0; EMBED_DIM],
        recipe: behavior::Recipe::default(),
    };
    (store, plan, trained)
}

/// A behavior as stored now: its status, and how its last training went.
async fn stored(store: &Store, id: &str) -> (Status, behavior::Training) {
    let m = memory::get(store, &TenantId::new("ws:a"), &MemoryId::new(id))
        .await
        .unwrap()
        .unwrap();
    (
        State::of(&m.evidence).unwrap().status,
        behavior::training(&m.evidence).unwrap(),
    )
}

#[tokio::test]
async fn an_expert_is_minted_with_what_it_learned_and_left_be_until_its_set_changes() {
    let (store, plan, trained) = learned_and_missed(true).await;
    let report = mint(&store, &plan, trained, "rig").await.unwrap();
    let (id, _) = report.expert.expect("minted");
    let e = expert::get(&store, &id).await.unwrap().unwrap();
    assert_eq!(behavior::taught_by(&e), ["memory:learned"]);
    assert_eq!(
        e.capability_card["missed"],
        serde_json::json!(["memory:missed"])
    );
    let set = behavior::fingerprint(&plan.taught);
    assert_eq!(behavior::trained_set(&e), Some(set.as_str()));
    // The card keeps each behavior's scores at admission, for the audit.
    assert!(behavior::card_worse(&e.capability_card, &["memory:missed"]).is_some());
    assert!(
        behavior::card_worse(&e.capability_card, &["memory:later", "memory:learned"]).is_some(),
        "read under the id a later recording supersedes"
    );
    assert_eq!(
        behavior::card_worse(&e.capability_card, &["memory:other"]),
        None
    );
    assert_eq!(
        e.capability_card[behavior::CARD_SCORES]["memory:learned"]["learned"],
        serde_json::json!(true)
    );

    let (status, t) = stored(&store, "memory:learned").await;
    assert_eq!(status, Status::Trained);
    assert_eq!(
        (t.set.as_str(), t.learned, t.admitted),
        (set.as_str(), true, true)
    );
    let (status, t) = stored(&store, "memory:missed").await;
    assert_eq!(status, Status::Accepted, "not in the expert");
    assert_eq!((t.expert, t.learned, t.admitted), (0.5, false, true));
    assert!(t.describe().contains("serves the others without it"));

    // Read again, the expert is in step with the set, the one it missed
    // still accepted: nothing to train until a behavior changes.
    let again = plan_again(&store).await;
    assert!(!behavior::stale(&again.taught, Some(&e)));
    assert!(!behavior::refused(&again.taught));
}

#[tokio::test]
async fn a_refused_expert_leaves_each_behavior_its_own_rates() {
    let (store, plan, trained) = learned_and_missed(false).await;
    let report = mint(&store, &plan, trained, "rig").await.unwrap();
    assert!(report.expert.is_none(), "nothing minted");

    let set = behavior::fingerprint(&plan.taught);
    let (status, t) = stored(&store, "memory:learned").await;
    assert_eq!(status, Status::Accepted);
    assert_eq!(
        (t.set.as_str(), t.expert, t.learned, t.admitted),
        (set.as_str(), 1.0, true, false)
    );
    let (_, t) = stored(&store, "memory:missed").await;
    assert_eq!((t.expert, t.learned, t.admitted), (0.5, false, false));

    // Read again, the same set is refused as it stands.
    assert!(behavior::refused(&plan_again(&store).await.taught));
}

async fn plan_again(store: &Store) -> Plan {
    plan(
        store,
        &TenantId::new("ws:a"),
        &UserId::new("user:a"),
        "everywhere",
    )
    .await
    .unwrap()
    .unwrap()
}

/// The GPU step of an audit is not run here; what it would measure, and
/// what is noted of the measurement, are.
#[tokio::test]
async fn an_audit_measures_only_what_the_card_never_saw_and_notes_it() {
    use antumbra_core::behavior::{content, Example};
    let (store, plan, trained) = learned_and_missed(true).await;
    let report = mint(&store, &plan, trained, "rig").await.unwrap();
    let (id, _) = report.expert.expect("minted");
    let e = expert::get(&store, &id).await.unwrap().unwrap();
    // The card names every taught example, so a later one stands out.
    let prints = &e.capability_card[behavior::CARD_EXAMPLES]["memory:missed"];
    assert_eq!(prints.as_array().map(Vec::len), Some(4));
    assert!(
        behavior::fresh_examples(&plan.taught, &e.capability_card).is_empty(),
        "nothing newer than the card"
    );

    // The user adds an example to the behavior it missed.
    let tenant = TenantId::new("ws:a");
    let mut m = memory::get(&store, &tenant, &MemoryId::new("memory:missed"))
        .await
        .unwrap()
        .unwrap();
    let mut spec = behavior::spec_of(&m.content).unwrap();
    spec.examples.push(Example {
        task: "t4".into(),
        answer: "x".into(),
    });
    m.content = content(&spec);
    memory::upsert(&store, &m).await.unwrap();
    let again = plan_again(&store).await;
    assert!(behavior::stale(&again.taught, Some(&e)), "a new set");
    let fresh = behavior::fresh_examples(&again.taught, &e.capability_card);
    assert_eq!(fresh.len(), 1, "{fresh:?}");
    assert_eq!(fresh[0].0, "memory:missed");
    assert_eq!(fresh[0].1, [spec.examples[4].clone()]);

    // Measured, the audit is noted on the behavior and on the card.
    let print = behavior::example_print(&fresh[0].1[0]);
    let set = behavior::trained_set(&e).unwrap().to_string();
    let audited = Audited {
        expert: id.clone(),
        set: set.clone(),
        behaviors: vec![(
            "memory:missed".into(),
            behavior::Audit {
                set: set.clone(),
                fresh: 1,
                base: 0,
                expert: 1,
                worse: behavior::card_worse(&e.capability_card, &["memory:missed"]),
            },
            vec![print.clone()],
        )],
    };
    assert_eq!(
        (audited.fresh(), audited.base(), audited.passed()),
        (1, 0, 1)
    );
    note_audit(&store, &again, &audited).await.unwrap();
    let m = memory::get(&store, &tenant, &MemoryId::new("memory:missed"))
        .await
        .unwrap()
        .unwrap();
    let notes = behavior::audits(&m.evidence);
    assert_eq!(notes.len(), 1);
    assert_eq!(
        (
            notes[0].set.as_str(),
            notes[0].fresh,
            notes[0].base,
            notes[0].expert
        ),
        (set.as_str(), 1, 0, 1)
    );
    assert!(
        behavior::training(&m.evidence).is_some(),
        "the training note it checks stays"
    );
    assert_eq!(State::of(&m.evidence).unwrap().status, Status::Accepted);
    let e = expert::get(&store, &id).await.unwrap().unwrap();
    assert_eq!(
        e.capability_card[behavior::CARD_AUDITED]["memory:missed"],
        serde_json::json!([print])
    );
    assert_eq!(
        behavior::trained_set(&e),
        Some(set.as_str()),
        "the card is otherwise as it was"
    );
    assert!(
        behavior::fresh_examples(&again.taught, &e.capability_card).is_empty(),
        "audited once, not again"
    );
    // The behavior that gained nothing was not audited.
    let m = memory::get(&store, &tenant, &MemoryId::new("memory:learned"))
        .await
        .unwrap()
        .unwrap();
    assert!(behavior::audits(&m.evidence).is_empty());
}

/// Recording a behavior again with more examples retires it under a new
/// id: the audit follows it there, and the card learns the new id.
#[tokio::test]
async fn an_audit_follows_a_behavior_recorded_again() {
    use antumbra_core::behavior::{
        content, scope_evidence, status_evidence, supersedes_evidence, Example,
    };
    let (store, plan, trained) = learned_and_missed(true).await;
    let report = mint(&store, &plan, trained, "rig").await.unwrap();
    let (id, _) = report.expert.expect("minted");
    let e = expert::get(&store, &id).await.unwrap().unwrap();
    let tenant = TenantId::new("ws:a");

    // Recorded again with one more example, superseding the old.
    let mut old = memory::get(&store, &tenant, &MemoryId::new("memory:missed"))
        .await
        .unwrap()
        .unwrap();
    let mut spec = behavior::spec_of(&old.content).unwrap();
    spec.examples.push(Example {
        task: "t4".into(),
        answer: "x".into(),
    });
    let again = Memory::new(
        "memory:missed2",
        tenant.clone(),
        antumbra_core::MemoryNetwork::Opinion,
        content(&spec),
        1.0,
        Utc::now(),
    )
    .in_compartment(behavior::compartment_id(&tenant, &UserId::new("user:a")))
    .with_evidence(vec![
        status_evidence(Status::Accepted),
        scope_evidence("everywhere"),
        supersedes_evidence("memory:missed"),
    ]);
    memory::upsert(&store, &again).await.unwrap();
    old.evidence.push(status_evidence(Status::Retired));
    memory::upsert(&store, &old).await.unwrap();

    let plan = plan_again(&store).await;
    let mut ids = plan.behaviors();
    ids.sort();
    assert_eq!(ids, ["memory:learned", "memory:missed2"]);
    let fresh = behavior::fresh_examples(&plan.taught, &e.capability_card);
    assert_eq!(fresh.len(), 1, "{fresh:?}");
    assert_eq!(fresh[0].0, "memory:missed2");
    assert_eq!(fresh[0].1, [spec.examples[4].clone()]);

    let print = behavior::example_print(&spec.examples[4]);
    let set = behavior::trained_set(&e).unwrap().to_string();
    let audited = Audited {
        expert: id.clone(),
        set: set.clone(),
        behaviors: vec![(
            "memory:missed2".into(),
            behavior::Audit {
                set,
                fresh: 1,
                base: 1,
                expert: 1,
                worse: None,
            },
            vec![print.clone()],
        )],
    };
    note_audit(&store, &plan, &audited).await.unwrap();
    let m = memory::get(&store, &tenant, &MemoryId::new("memory:missed2"))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        behavior::audits(&m.evidence).len(),
        1,
        "noted on the new one"
    );
    let e = expert::get(&store, &id).await.unwrap().unwrap();
    let card = &e.capability_card;
    assert_eq!(
        card[behavior::CARD_EXAMPLES]["memory:missed2"],
        card[behavior::CARD_EXAMPLES]["memory:missed"],
        "the card learns the new id"
    );
    assert_eq!(
        card[behavior::CARD_AUDITED]["memory:missed2"],
        serde_json::json!([print])
    );
    assert!(
        behavior::fresh_examples(&plan.taught, &e.capability_card).is_empty(),
        "audited once, not again"
    );
}
