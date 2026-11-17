use super::*;

use antumbra_store::testing::mem_store;

#[test]
fn a_standing_expert_trains_at_the_rate_nine_behaviors_held_at() -> anyhow::Result<()> {
    let cfg = standing_config(&behavior::Recipe::default());
    assert!(cfg.learning_rate <= 1.5e-4, "3e-4 broke nine behaviors");
    assert_eq!((cfg.rounds, cfg.max_new_tokens), (3, 256));
    assert_eq!(cfg.lora_rank, RaftConfig::default().lora_rank);
    Ok(())
}

/// A recipe at another rank keeps the scale every expert is trained and
/// served at: alpha moves with the rank.
#[test]
fn a_recipe_moves_alpha_with_the_rank_and_keeps_the_scale() -> anyhow::Result<()> {
    let recipe = behavior::Recipe::new(32, 5, 1e-4, 0.9).map_err(anyhow::Error::msg)?;
    let cfg = standing_config(&recipe);
    assert_eq!((cfg.lora_rank, cfg.rounds), (32, 5));
    assert!((cfg.learning_rate - 1e-4).abs() < 1e-12);
    assert!((cfg.lora_scale() - RaftConfig::default().lora_scale()).abs() < 1e-12);
    assert_eq!(cfg.max_new_tokens, 256);
    Ok(())
}

#[test]
fn a_standing_expert_is_named_for_its_workspace_owner_and_scope() -> anyhow::Result<()> {
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
    Ok(())
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
        seal: None,
        drift: Vec::new(),
        base_model: "base".into(),
        fitness: 1.0,
        base_fitness: 0.0,
        capability_vec: vec![0.0; EMBED_DIM],
        recipe: behavior::Recipe::default(),
        misses: Vec::new(),
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

/// The expert `report` minted, which it must have.
fn minted(report: BehaveReport) -> anyhow::Result<(ExpertId, String)> {
    let Some(expert) = report.expert else {
        anyhow::bail!("nothing minted");
    };
    Ok(expert)
}

/// The expert under `id`, which the store must hold.
async fn expert_in(store: &Store, id: &ExpertId) -> anyhow::Result<Expert> {
    let Some(e) = expert::get(store, id).await? else {
        anyhow::bail!("no expert {id} in the store");
    };
    Ok(e)
}

/// `tenant`'s memory `id`, which the store must hold.
async fn memory_in(store: &Store, tenant: &TenantId, id: &str) -> anyhow::Result<Memory> {
    let Some(m) = memory::get(store, tenant, &MemoryId::new(id)).await? else {
        anyhow::bail!("no memory {id} in the store");
    };
    Ok(m)
}

/// The status `m`'s evidence puts it at, which a behavior must have.
fn status_of(m: &Memory) -> anyhow::Result<Status> {
    let Some(state) = State::of(&m.evidence) else {
        anyhow::bail!("{} has no state", m.id.as_str());
    };
    Ok(state.status)
}

/// The behavior `m` records, which its content must spell out.
fn spec_in(m: &Memory) -> anyhow::Result<Spec> {
    let Some(spec) = behavior::spec_of(&m.content) else {
        anyhow::bail!("{} holds no behavior", m.id.as_str());
    };
    Ok(spec)
}

/// The set `expert`'s card says it was trained on, which it must name.
fn set_of(expert: &Expert) -> anyhow::Result<String> {
    let Some(set) = behavior::trained_set(expert) else {
        anyhow::bail!("the card of {} names no set", expert.id);
    };
    Ok(set.to_string())
}

#[tokio::test]
async fn two_workspaces_with_the_same_user_keep_their_own_standing_experts() -> anyhow::Result<()> {
    let store = mem_store().await?;
    let (a, b) = (owner_plan("ws:a"), owner_plan("ws:b"));
    let (ea, _) = minted(
        mint(
            &store,
            &a,
            admitted(&["memory:rule"], "adapters/a.safetensors"),
            "rig",
        )
        .await?,
    )?;
    let (eb, _) = minted(
        mint(
            &store,
            &b,
            admitted(&["memory:rule"], "adapters/b.safetensors"),
            "rig",
        )
        .await?,
    )?;
    assert_ne!(ea, eb);
    let ea = expert_in(&store, &ea).await?;
    let eb = expert_in(&store, &eb).await?;
    assert_eq!(ea.artifact_uri, "adapters/a.safetensors");
    assert_eq!(eb.artifact_uri, "adapters/b.safetensors");
    let owner = UserId::new("user:owner");
    assert!(behavior::is_standing_for(&ea, &a.tenant, &owner));
    assert!(!behavior::is_standing_for(&ea, &b.tenant, &owner));
    assert!(behavior::is_standing_for(&eb, &b.tenant, &owner));
    Ok(())
}

#[tokio::test]
async fn minting_retires_this_workspace_s_expert_under_the_old_id_and_no_other(
) -> anyhow::Result<()> {
    let store = mem_store().await?;
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
    expert::insert(&store, &old(&b.tenant)).await?;
    mint(
        &store,
        &a,
        admitted(&["memory:rule"], "adapters/a.safetensors"),
        "rig",
    )
    .await?;
    assert!(
        expert::get(&store, &legacy).await?.is_some(),
        "not this workspace's to retire"
    );

    // This workspace's, under the old id: minting here retires it.
    expert::delete(&store, &legacy).await?;
    expert::insert(&store, &old(&a.tenant)).await?;
    mint(
        &store,
        &a,
        admitted(&["memory:rule"], "adapters/a2.safetensors"),
        "rig",
    )
    .await?;
    assert!(expert::get(&store, &legacy).await?.is_none());
    let now = expert_in(&store, &expert_id(&a.tenant, &owner, "everywhere")).await?;
    assert_eq!(now.artifact_uri, "adapters/a2.safetensors");
    Ok(())
}

#[tokio::test]
async fn mint_marks_only_the_behaviors_still_as_they_were_taught() -> anyhow::Result<()> {
    use antumbra_core::behavior::{content, scope_evidence, status_evidence, Example};
    let store = mem_store().await?;
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
        memory::upsert(&store, &m).await?;
    }
    let Some(plan) = plan(&store, &tenant, &user, "everywhere").await? else {
        anyhow::bail!("nothing to plan");
    };
    let mut taught = plan.behaviors();
    taught.sort();
    assert_eq!(taught, ["memory:gone", "memory:kept"]);
    // Retired while it trained.
    memory::upsert(&store, &put("memory:gone", Status::Retired)).await?;

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
        seal: None,
        drift: Vec::new(),
        base_model: "base".into(),
        fitness: 1.0,
        base_fitness: 0.0,
        capability_vec: vec![0.0; EMBED_DIM],
        recipe: behavior::Recipe::default(),
        misses: Vec::new(),
    };
    let (id, _) = minted(mint(&store, &plan, trained, "rig").await?)?;
    assert_eq!(id.as_str(), "expert:ws:a:user:a:behavior:everywhere");
    let status = |id: &str| {
        let store = store.clone();
        let tenant = tenant.clone();
        let id = id.to_string();
        async move { status_of(&memory_in(&store, &tenant, &id).await?) }
    };
    assert_eq!(status("memory:kept").await?, Status::Trained);
    assert_eq!(status("memory:gone").await?, Status::Retired);
    let e = expert_in(&store, &id).await?;
    assert_eq!(
        e.placed_on.as_deref(),
        Some("rig"),
        "placed where its adapter is"
    );
    Ok(())
}

/// A store holding two accepted behaviors everywhere, `memory:learned`
/// and `memory:missed`, planned; and a training of them where the first
/// was learned and the second scored 0.50 against the base's 0.00.
async fn learned_and_missed(admitted: bool) -> anyhow::Result<(Store, Plan, Trained)> {
    use antumbra_core::behavior::{content, scope_evidence, status_evidence, Example};
    use antumbra_train::behave::BehaviorScore;
    let store = mem_store().await?;
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
        memory::upsert(&store, &m).await?;
    }
    let Some(plan) = plan(&store, &tenant, &user, "everywhere").await? else {
        anyhow::bail!("nothing to plan");
    };
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
        seal: None,
        drift: Vec::new(),
        base_model: "base".into(),
        fitness: 0.5,
        base_fitness: 0.25,
        capability_vec: vec![0.0; EMBED_DIM],
        recipe: behavior::Recipe::default(),
        misses: Vec::new(),
    };
    Ok((store, plan, trained))
}

/// A behavior as stored now: its status, and how its last training went.
async fn stored(store: &Store, id: &str) -> anyhow::Result<(Status, behavior::Training)> {
    let m = memory_in(store, &TenantId::new("ws:a"), id).await?;
    let Some(training) = behavior::training(&m.evidence) else {
        anyhow::bail!("{id} has no training noted");
    };
    Ok((status_of(&m)?, training))
}

#[tokio::test]
async fn an_expert_is_minted_with_what_it_learned_and_left_be_until_its_set_changes(
) -> anyhow::Result<()> {
    let (store, plan, trained) = learned_and_missed(true).await?;
    let (id, _) = minted(mint(&store, &plan, trained, "rig").await?)?;
    let e = expert_in(&store, &id).await?;
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

    let (status, t) = stored(&store, "memory:learned").await?;
    assert_eq!(status, Status::Trained);
    assert_eq!(
        (t.set.as_str(), t.learned, t.admitted),
        (set.as_str(), true, true)
    );
    let (status, t) = stored(&store, "memory:missed").await?;
    assert_eq!(status, Status::Accepted, "not in the expert");
    assert_eq!((t.expert, t.learned, t.admitted), (0.5, false, true));
    assert!(t.describe().contains("serves the others without it"));

    // Read again, the expert is in step with the set, the one it missed
    // still accepted: nothing to train until a behavior changes.
    let again = plan_again(&store).await?;
    assert!(!behavior::stale(&again.taught, Some(&e)));
    assert!(!behavior::refused(&again.taught));
    Ok(())
}

#[tokio::test]
async fn a_refused_expert_leaves_each_behavior_its_own_rates() -> anyhow::Result<()> {
    let (store, plan, trained) = learned_and_missed(false).await?;
    let report = mint(&store, &plan, trained, "rig").await?;
    assert!(report.expert.is_none(), "nothing minted");

    let set = behavior::fingerprint(&plan.taught);
    let (status, t) = stored(&store, "memory:learned").await?;
    assert_eq!(status, Status::Accepted);
    assert_eq!(
        (t.set.as_str(), t.expert, t.learned, t.admitted),
        (set.as_str(), 1.0, true, false)
    );
    let (_, t) = stored(&store, "memory:missed").await?;
    assert_eq!((t.expert, t.learned, t.admitted), (0.5, false, false));

    // Read again, the same set is refused as it stands.
    assert!(behavior::refused(&plan_again(&store).await?.taught));
    Ok(())
}

/// What a behavior's last training noted it was missed on, as stored now.
async fn noted_miss(store: &Store, id: &str) -> anyhow::Result<Option<Missed>> {
    let m = memory_in(store, &TenantId::new("ws:a"), id).await?;
    Ok(missed::of(&m.evidence))
}

/// A behavior not learned notes the held-out task it was missed on and the
/// expert's answer, whether the expert was admitted or refused. One learned
/// notes none, even when it failed a held-out task on its way over the bar,
/// and a note an earlier training left on it goes.
#[tokio::test]
async fn a_missed_behavior_notes_what_the_expert_answered_instead() -> anyhow::Result<()> {
    for admitted in [true, false] {
        let (store, plan, mut trained) = learned_and_missed(admitted).await?;
        trained.misses = vec![
            ("memory:learned".into(), Missed::new("t2", "wrong once")),
            (
                "memory:missed".into(),
                Missed::new("t2", "not the behavior"),
            ),
        ];
        let tenant = TenantId::new("ws:a");
        let mut earlier = memory_in(&store, &tenant, "memory:learned").await?;
        missed::mark(&mut earlier.evidence, Some(&Missed::new("t0", "before")));
        memory::upsert(&store, &earlier).await?;

        mint(&store, &plan, trained, "rig").await?;

        let miss = noted_miss(&store, "memory:missed").await?;
        assert_eq!(
            miss.map(|m| (m.task, m.answer)),
            Some(("t2".into(), "not the behavior".into())),
            "admitted: {admitted}"
        );
        assert_eq!(
            noted_miss(&store, "memory:learned").await?,
            None,
            "admitted: {admitted}"
        );
    }
    Ok(())
}

/// The shadow `plan`'s training is recorded as, as stored now.
async fn shadow_of(store: &Store, plan: &Plan) -> anyhow::Result<Shadow> {
    let Some(shadow) = shadow::get(store, &shadow_id(plan)).await? else {
        anyhow::bail!("the plan's training has no shadow");
    };
    Ok(shadow)
}

/// A training is a shadow in the penumbra: exploring once begun, then scored
/// from the base model's pass rate to its own and graduated when admitted,
/// in the row it began in.
#[tokio::test]
async fn an_admitted_training_graduates_its_shadow() -> anyhow::Result<()> {
    let (store, plan, trained) = learned_and_missed(true).await?;
    let begun = begin(&store, &plan).await?;
    assert_eq!(begun.parent_expert, None, "no expert before the first");
    let exploring = shadow_of(&store, &plan).await?;
    assert_eq!(exploring.status, ShadowStatus::Exploring);

    mint(&store, &plan, trained, "rig").await?;
    let sh = shadow_of(&store, &plan).await?;
    assert_eq!(sh.status, ShadowStatus::Graduated);
    assert_eq!(sh.reward_curve, vec![0.25, 0.5]);
    assert_eq!(sh.adapter_uri.as_deref(), Some("adapters/x.safetensors"));
    assert_eq!(sh.created_at, exploring.created_at, "the row begun");
    Ok(())
}

#[tokio::test]
async fn a_refused_training_prunes_its_shadow_though_never_begun() -> anyhow::Result<()> {
    let (store, plan, trained) = learned_and_missed(false).await?;
    mint(&store, &plan, trained, "rig").await?;
    let sh = shadow_of(&store, &plan).await?;
    assert_eq!(sh.status, ShadowStatus::Pruned);
    assert_eq!(sh.reward_curve, vec![0.25, 0.5]);
    Ok(())
}

/// A training that fails is pruned, and one settled stays as it is. The
/// next training of the scope descends from the expert the last minted.
#[tokio::test]
async fn a_failed_training_is_pruned_and_the_next_descends_from_the_expert() -> anyhow::Result<()> {
    let (store, plan, trained) = learned_and_missed(true).await?;
    begin(&store, &plan).await?;
    abandon(&store, &plan).await?;
    assert_eq!(shadow_of(&store, &plan).await?.status, ShadowStatus::Pruned);
    abandon(&store, &plan).await?;
    assert_eq!(shadow_of(&store, &plan).await?.status, ShadowStatus::Pruned);

    mint(&store, &plan, trained, "rig").await?;
    assert_eq!(
        shadow_of(&store, &plan).await?.status,
        ShadowStatus::Graduated
    );
    let again = begin(&store, &plan).await?;
    assert_eq!(
        again.parent_expert,
        Some(expert_id(&plan.tenant, &plan.user, &plan.scope))
    );
    Ok(())
}

/// The boundaries beside `plan`'s scope's standing expert.
async fn boundaries_of(store: &Store, plan: &Plan) -> anyhow::Result<Vec<FailureBoundary>> {
    let prefix = boundary_prefix(&expert_id(&plan.tenant, &plan.user, &plan.scope));
    Ok(boundary::list(store)
        .await?
        .into_iter()
        .filter(|b| b.id.as_str().starts_with(&prefix))
        .collect())
}

/// A boundary beside `plan`'s scope's expert, for `behavior`, as an earlier
/// training would have left it.
fn earlier_boundary(plan: &Plan, behavior: &str) -> FailureBoundary {
    let prefix = boundary_prefix(&expert_id(&plan.tenant, &plan.user, &plan.scope));
    FailureBoundary {
        id: BoundaryId::new(format!("{prefix}{behavior}")),
        behavior: "An earlier rule.".into(),
        fail_context: serde_json::json!({ "scope": plan.scope }),
        near_ok_context: None,
        governing_features: Vec::new(),
        grain: None,
        context_vec: None,
        ok_context_vec: None,
        confidence: 1.0,
        generation: Generation::ZERO,
        owner: Some(plan.user.clone()),
        created_at: Utc::now(),
    }
}

/// A behavior the training missed stands as an open boundary beside the
/// expert, its owner's: the rule, the task it was missed on and the answer
/// given, inert in routing, as confident as the share of its held-out tasks
/// failed. One the training learned has none, and one no longer taught goes.
#[tokio::test]
async fn a_missed_behavior_stands_as_an_open_boundary_of_its_owner() -> anyhow::Result<()> {
    for admitted in [true, false] {
        let (store, plan, mut trained) = learned_and_missed(admitted).await?;
        boundary::upsert(&store, &earlier_boundary(&plan, "memory:retired")).await?;
        trained.misses = vec![(
            "memory:missed".into(),
            Missed::new("t2", "not the behavior"),
        )];
        mint(&store, &plan, trained, "rig").await?;

        let found = boundaries_of(&store, &plan).await?;
        assert_eq!(found.len(), 1, "admitted: {admitted}");
        let b = &found[0];
        assert_eq!(b.behavior, "Rule two.");
        assert!(!b.is_actionable(), "open, so it never gates routing");
        assert_eq!(b.owner, Some(UserId::new("user:a")));
        assert_eq!(b.fail_context["scope"], "everywhere");
        assert_eq!(b.fail_context["task"], "t2");
        assert_eq!(b.fail_context["answer"], "not the behavior");
        assert!((b.confidence - 0.5).abs() < 1e-6);
    }
    Ok(())
}

/// A boundary stands until an admitted expert learns its behavior: a refused
/// training that learned it leaves it, since nothing serves it. Dropping the
/// expert takes its boundaries, and no other scope's.
#[tokio::test]
async fn a_boundary_goes_once_an_expert_learns_it_or_is_dropped() -> anyhow::Result<()> {
    let (store, plan, trained) = learned_and_missed(true).await?;
    let mut all_learned = trained.clone();
    for s in &mut all_learned.verdict.behaviors {
        s.admitted = true;
        s.expert = 1.0;
    }
    mint(&store, &plan, trained, "rig").await?;
    assert_eq!(boundaries_of(&store, &plan).await?.len(), 1);

    let mut refused = all_learned.clone();
    refused.verdict.admitted = false;
    mint(&store, &plan, refused, "rig").await?;
    assert_eq!(boundaries_of(&store, &plan).await?.len(), 1, "refused");

    mint(&store, &plan, all_learned, "rig").await?;
    assert!(boundaries_of(&store, &plan).await?.is_empty(), "learned");

    let elsewhere = Plan {
        scope: "everywhere-else".into(),
        ..plan.clone()
    };
    boundary::upsert(&store, &earlier_boundary(&plan, "memory:missed")).await?;
    boundary::upsert(&store, &earlier_boundary(&elsewhere, "memory:missed")).await?;
    forget_boundaries(&store, &expert_id(&plan.tenant, &plan.user, &plan.scope)).await?;
    assert!(boundaries_of(&store, &plan).await?.is_empty(), "dropped");
    assert_eq!(boundaries_of(&store, &elsewhere).await?.len(), 1);
    Ok(())
}

/// A standing expert minted before trainings were recorded gets its
/// graduated shadow, and its missed behavior its boundary, from its card and
/// the miss noted on the behavior: once, and only while it is in step with
/// the set.
#[tokio::test]
async fn a_standing_expert_from_before_is_recorded_from_its_card() -> anyhow::Result<()> {
    let (recorded, plan, mut trained) = learned_and_missed(true).await?;
    trained.misses = vec![(
        "memory:missed".into(),
        Missed::new("t2", "not the behavior"),
    )];
    let (id, _) = minted(mint(&recorded, &plan, trained, "rig").await?)?;
    let e = expert_in(&recorded, &id).await?;

    // The same behaviors, and the expert, where the training was not recorded.
    let (store, _, _) = learned_and_missed(true).await?;
    let tenant = TenantId::new("ws:a");
    let mut m = memory_in(&store, &tenant, "memory:missed").await?;
    missed::mark(
        &mut m.evidence,
        Some(&Missed::new("t2", "not the behavior")),
    );
    memory::upsert(&store, &m).await?;
    expert::insert(&store, &e).await?;
    let plan = plan_again(&store).await?;

    let mut elsewhere = e.clone();
    elsewhere.capability_card["set"] = "another set".into();
    record_standing(&store, &plan, &elsewhere).await?;
    assert!(
        shadow::get(&store, &shadow_id(&plan)).await?.is_none(),
        "not in step with the set"
    );

    record_standing(&store, &plan, &e).await?;
    let sh = shadow_of(&store, &plan).await?;
    assert_eq!(sh.status, ShadowStatus::Graduated);
    assert_eq!(sh.reward_curve, vec![0.5]);
    assert_eq!(sh.adapter_uri.as_deref(), Some("adapters/x.safetensors"));
    let found = boundaries_of(&store, &plan).await?;
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].behavior, "Rule two.");
    assert_eq!(found[0].fail_context["answer"], "not the behavior");
    assert!((found[0].confidence - 0.5).abs() < 1e-6);

    boundary::delete(&store, &found[0].id).await?;
    record_standing(&store, &plan, &e).await?;
    assert!(boundaries_of(&store, &plan).await?.is_empty(), "once");
    Ok(())
}

async fn plan_again(store: &Store) -> anyhow::Result<Plan> {
    let Some(plan) = plan(
        store,
        &TenantId::new("ws:a"),
        &UserId::new("user:a"),
        "everywhere",
    )
    .await?
    else {
        anyhow::bail!("nothing to plan");
    };
    Ok(plan)
}

/// The GPU step of an audit is not run here; what it would measure, and
/// what is noted of the measurement, are.
#[tokio::test]
async fn an_audit_measures_only_what_the_card_never_saw_and_notes_it() -> anyhow::Result<()> {
    use antumbra_core::behavior::{content, Example};
    let (store, plan, trained) = learned_and_missed(true).await?;
    let (id, _) = minted(mint(&store, &plan, trained, "rig").await?)?;
    let e = expert_in(&store, &id).await?;
    // The card names every taught example, so a later one stands out.
    let prints = &e.capability_card[behavior::CARD_EXAMPLES]["memory:missed"];
    assert_eq!(prints.as_array().map(Vec::len), Some(4));
    assert!(
        behavior::fresh_examples(&plan.taught, &e.capability_card).is_empty(),
        "nothing newer than the card"
    );

    // The user adds an example to the behavior it missed.
    let tenant = TenantId::new("ws:a");
    let mut m = memory_in(&store, &tenant, "memory:missed").await?;
    let mut spec = spec_in(&m)?;
    spec.examples.push(Example {
        task: "t4".into(),
        answer: "x".into(),
    });
    m.content = content(&spec);
    memory::upsert(&store, &m).await?;
    let again = plan_again(&store).await?;
    assert!(behavior::stale(&again.taught, Some(&e)), "a new set");
    let fresh = behavior::fresh_examples(&again.taught, &e.capability_card);
    assert_eq!(fresh.len(), 1, "{fresh:?}");
    assert_eq!(fresh[0].0, "memory:missed");
    assert_eq!(fresh[0].1, [spec.examples[4].clone()]);

    // Measured, the audit is noted on the behavior and on the card.
    let print = behavior::example_print(&fresh[0].1[0]);
    let set = set_of(&e)?;
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
    note_audit(&store, &again, &audited).await?;
    let m = memory_in(&store, &tenant, "memory:missed").await?;
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
    assert_eq!(status_of(&m)?, Status::Accepted);
    let e = expert_in(&store, &id).await?;
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
    let m = memory_in(&store, &tenant, "memory:learned").await?;
    assert!(behavior::audits(&m.evidence).is_empty());
    Ok(())
}

/// Recording a behavior again with more examples retires it under a new
/// id: the audit follows it there, and the card learns the new id.
#[tokio::test]
async fn an_audit_follows_a_behavior_recorded_again() -> anyhow::Result<()> {
    use antumbra_core::behavior::{
        content, scope_evidence, status_evidence, supersedes_evidence, Example,
    };
    let (store, plan, trained) = learned_and_missed(true).await?;
    let (id, _) = minted(mint(&store, &plan, trained, "rig").await?)?;
    let e = expert_in(&store, &id).await?;
    let tenant = TenantId::new("ws:a");

    // Recorded again with one more example, superseding the old.
    let mut old = memory_in(&store, &tenant, "memory:missed").await?;
    let mut spec = spec_in(&old)?;
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
    memory::upsert(&store, &again).await?;
    old.evidence.push(status_evidence(Status::Retired));
    memory::upsert(&store, &old).await?;

    let plan = plan_again(&store).await?;
    let mut ids = plan.behaviors();
    ids.sort();
    assert_eq!(ids, ["memory:learned", "memory:missed2"]);
    let fresh = behavior::fresh_examples(&plan.taught, &e.capability_card);
    assert_eq!(fresh.len(), 1, "{fresh:?}");
    assert_eq!(fresh[0].0, "memory:missed2");
    assert_eq!(fresh[0].1, [spec.examples[4].clone()]);

    let print = behavior::example_print(&spec.examples[4]);
    let set = set_of(&e)?;
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
    note_audit(&store, &plan, &audited).await?;
    let m = memory_in(&store, &tenant, "memory:missed2").await?;
    assert_eq!(
        behavior::audits(&m.evidence).len(),
        1,
        "noted on the new one"
    );
    let e = expert_in(&store, &id).await?;
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
    Ok(())
}

/// The trainer judges an answer by a behavior's check exactly as recording
/// judged its examples and violations: one matcher, one dialect. A violation
/// ending in a newline is the case the trainer's old Python check read the
/// other way, since Python's `$` also matches before a final newline.
#[tokio::test]
async fn the_trainer_judges_an_answer_as_recording_judged_the_examples() -> anyhow::Result<()> {
    use antumbra_core::behavior::Example;
    use antumbra_core::ports::{Verifier, VerifyRequest};
    let spec = Spec {
        rule: "Stage files by name.".into(),
        must: vec![r"(?i)^git add (?!-A\b)\S+$".into()],
        must_not: vec!["--force".into()],
        examples: ["git add src/lib.rs", "GIT ADD Cargo.toml"]
            .iter()
            .map(|a| Example {
                task: "Stage it.".into(),
                answer: a.to_string(),
            })
            .collect(),
        violations: vec![
            "git add -A".into(),
            "git add src/lib.rs\n".into(),
            "git add --force x".into(),
        ],
    };
    let verify = antumbra_train::behave::verify(&spec);
    let verifier = antumbra_critic::CommandVerifier;
    let answers = spec
        .examples
        .iter()
        .map(|e| e.answer.clone())
        .chain(spec.violations.iter().cloned());
    for answer in answers {
        let req = VerifyRequest {
            run_id: RunId::new("r"),
            step_idx: 0,
            dimension: "exec".into(),
            artifact: serde_json::json!({ "completion": answer, "verify": verify }),
        };
        let Some(follows) = spec.follows(&answer) else {
            anyhow::bail!("{answer:?} is judged by no check");
        };
        assert_eq!(verifier.verify(&req).await?.passed, follows, "{answer:?}");
    }
    assert_eq!(spec.follows("git add src/lib.rs\n"), Some(false));
    Ok(())
}

/// An expert whose card is `card`, in the store, under the standing id.
async fn expert_with_card(store: &Store, card: serde_json::Value) -> anyhow::Result<ExpertId> {
    let tenant = TenantId::new("ws:a");
    let owner = UserId::new("user:owner");
    let id = expert_id(&tenant, &owner, "everywhere");
    expert::insert(
        store,
        &Expert {
            id: id.clone(),
            name: "standing".into(),
            base_model: "base".into(),
            artifact_uri: "adapters/standing.safetensors".into(),
            capability_card: card,
            capability_vec: None,
            fitness: 1.0,
            frozen_at: Some(Utc::now()),
            generation: Generation::ZERO,
            owner: Some(owner.clone()),
            compartment: Some(behavior::compartment_id(&tenant, &owner)),
            placed_on: Some("rig".into()),
            created_at: Utc::now(),
        },
    )
    .await?;
    Ok(id)
}

/// One behavior audited on two later examples, one print to note.
fn one_audit(expert: &ExpertId) -> Audited {
    Audited {
        expert: expert.clone(),
        set: "set".into(),
        behaviors: vec![(
            "memory:rule".into(),
            behavior::Audit {
                set: "set".into(),
                fresh: 2,
                base: 1,
                expert: 2,
                worse: None,
            },
            vec!["print-a".into()],
        )],
    }
}

// A card that is not an object, or whose audited section is not, is a row
// this code never wrote: the audit is refused in words naming the expert,
// where indexing into it would have panicked the keeper's task, and the card
// is left as it was.
#[tokio::test]
async fn note_audit_refuses_a_card_that_is_not_an_object() -> anyhow::Result<()> {
    let store = mem_store().await?;
    let plan = owner_plan("ws:a");
    for odd in [
        serde_json::json!(["legacy"]),
        serde_json::json!("legacy"),
        serde_json::json!({ "standing": true, "audited": ["not", "an", "object"] }),
    ] {
        let id = expert_with_card(&store, odd.clone()).await?;
        let Err(e) = note_audit(&store, &plan, &one_audit(&id)).await else {
            anyhow::bail!("{odd} was indexed into");
        };
        anyhow::ensure!(
            e.to_string().contains("not an object") && e.to_string().contains(id.as_str()),
            "{e}"
        );
        let Some(kept) = expert::get(&store, &id).await? else {
            anyhow::bail!("the expert is gone");
        };
        anyhow::ensure!(
            kept.capability_card == odd,
            "the card changed: {}",
            kept.capability_card
        );
        expert::delete(&store, &id).await?;
    }
    Ok(())
}

// A card that is nothing yet, or an object without the sections, gets them
// and the audit's prints under the behavior.
#[tokio::test]
async fn note_audit_makes_the_sections_a_card_lacks() -> anyhow::Result<()> {
    let store = mem_store().await?;
    let plan = owner_plan("ws:a");
    for start in [
        serde_json::Value::Null,
        serde_json::json!({ "standing": true }),
        serde_json::json!({ "standing": true, "audited": null }),
    ] {
        let id = expert_with_card(&store, start.clone()).await?;
        note_audit(&store, &plan, &one_audit(&id)).await?;
        let Some(noted) = expert::get(&store, &id).await? else {
            anyhow::bail!("the expert is gone");
        };
        anyhow::ensure!(
            noted.capability_card[behavior::CARD_AUDITED]["memory:rule"]
                == serde_json::json!(["print-a"]),
            "from {start}: {}",
            noted.capability_card
        );
        expert::delete(&store, &id).await?;
    }
    Ok(())
}

/// An admitted training of `ids` whose adapter was promoted to `adapter`,
/// three bytes with the digest `sha256`.
fn sealed(ids: &[&str], adapter: &str, sha256: &str) -> Trained {
    Trained {
        seal: Some(adapter::Seal {
            sha256: sha256.into(),
            bytes: 3,
        }),
        ..admitted(ids, adapter)
    }
}

/// The adapter the scope's standing expert is served from, and the seal on
/// its card.
async fn served(store: &Store, plan: &Plan) -> anyhow::Result<(String, serde_json::Value)> {
    let id = expert_id(&plan.tenant, &plan.user, &plan.scope);
    let Some(e) = expert::get(store, &id).await? else {
        anyhow::bail!("{} is not minted", id.as_str());
    };
    let seal = e.capability_card[adapter::CARD_ADAPTER].clone();
    Ok((e.artifact_uri, seal))
}

#[tokio::test]
async fn an_expert_s_row_names_its_adapter_and_its_card_seals_it() -> anyhow::Result<()> {
    let store = Store::connect_memory(EMBED_DIM).await?;
    let plan = owner_plan("ws:a");

    let first = sealed(&["memory:rule"], "adapters/e.aa.safetensors", "aa");
    let report = mint(&store, &plan, first, "rig").await?;

    assert_eq!(
        report.expert.map(|(_, uri)| uri).as_deref(),
        Some("adapters/e.aa.safetensors")
    );
    assert_eq!(
        report.superseded, None,
        "the first of a scope replaces none"
    );
    assert_eq!(
        served(&store, &plan).await?,
        (
            "adapters/e.aa.safetensors".to_string(),
            serde_json::json!({ "sha256": "aa", "bytes": 3 })
        )
    );
    Ok(())
}

#[tokio::test]
async fn an_admitted_retraining_reports_the_adapter_it_replaces() -> anyhow::Result<()> {
    let store = Store::connect_memory(EMBED_DIM).await?;
    let plan = owner_plan("ws:a");
    let train = |file: &str, sha256: &str| sealed(&["memory:rule"], file, sha256);
    mint(
        &store,
        &plan,
        train("adapters/e.aa.safetensors", "aa"),
        "rig",
    )
    .await?;

    let again = mint(
        &store,
        &plan,
        train("adapters/e.bb.safetensors", "bb"),
        "rig",
    )
    .await?;
    assert_eq!(
        again.superseded.as_deref(),
        Some("adapters/e.aa.safetensors")
    );
    assert_eq!(
        served(&store, &plan).await?,
        (
            "adapters/e.bb.safetensors".to_string(),
            serde_json::json!({ "sha256": "bb", "bytes": 3 })
        )
    );

    // The same bytes admitted again are the same file: nothing to remove.
    let same = mint(
        &store,
        &plan,
        train("adapters/e.bb.safetensors", "bb"),
        "rig",
    )
    .await?;
    assert_eq!(same.superseded, None);

    // Minted on another machine, the file before it is on a disk that
    // machine cannot reach: it has nothing of its own to remove.
    let moved = mint(
        &store,
        &plan,
        train("adapters/e.cc.safetensors", "cc"),
        "other",
    )
    .await?;
    assert_eq!(moved.superseded, None);
    Ok(())
}

/// The regression, at the row: a refused retraining leaves the expert, the
/// adapter its row names and the seal on its card as admission left them.
#[tokio::test]
async fn a_refused_retraining_leaves_the_expert_its_adapter() -> anyhow::Result<()> {
    let store = Store::connect_memory(EMBED_DIM).await?;
    let plan = owner_plan("ws:a");
    let first = sealed(&["memory:rule"], "adapters/e.aa.safetensors", "aa");
    mint(&store, &plan, first, "rig").await?;
    let before = served(&store, &plan).await?;

    let mut refused = admitted(&["memory:rule"], "adapters/candidates/e.safetensors");
    refused.verdict.admitted = false;
    let report = mint(&store, &plan, refused, "rig").await?;

    assert!(report.expert.is_none(), "nothing is minted");
    assert_eq!(report.superseded, None, "and nothing is to be removed");
    assert_eq!(served(&store, &plan).await?, before);
    assert_eq!(before.0, "adapters/e.aa.safetensors");
    Ok(())
}

/// A quarter of a nat apart on average over two positions, one of them
/// ranking another token first.
fn two_positions() -> antumbra_train::quant::Drift {
    use antumbra_train::quant::{Drift, Reading};
    let reading = |logprob, top1| Reading { logprob, top1 };
    let mut drift = Drift::default();
    drift.add(
        &[reading(-1.0, 1), reading(-2.0, 2)],
        &[reading(-1.5, 1), reading(-2.0, 3)],
    );
    drift
}

#[tokio::test]
async fn an_expert_s_card_records_what_each_quantized_base_did_to_it() -> anyhow::Result<()> {
    let store = Store::connect_memory(EMBED_DIM).await?;
    let plan = owner_plan("ws:a");
    let trained = Trained {
        drift: vec![
            (ServeQuant::Q8_0, antumbra_train::quant::Drift::default()),
            (ServeQuant::Q4K, two_positions()),
        ],
        ..sealed(&["memory:rule"], "adapters/e.aa.safetensors", "aa")
    };

    let report = mint(&store, &plan, trained, "rig").await?;

    assert_eq!(
        report
            .drift
            .iter()
            .map(|(form, _)| *form)
            .collect::<Vec<_>>(),
        [ServeQuant::Q8_0, ServeQuant::Q4K],
        "the report says what was measured, in the order it was asked for"
    );
    let Some(e) = expert::get(&store, &expert_id(&plan.tenant, &plan.user, &plan.scope)).await?
    else {
        anyhow::bail!("the expert was not minted");
    };
    assert_eq!(
        e.capability_card[drift::CARD_DRIFT],
        serde_json::json!({
            "q8_0": { "positions": 0, "mean_shift": 0.0, "largest_shift": 0.0, "flips": 0 },
            "q4_k": { "positions": 2, "mean_shift": 0.25, "largest_shift": 0.5, "flips": 1 },
        })
    );
    Ok(())
}

#[tokio::test]
async fn an_expert_measured_on_nothing_has_no_drift_on_its_card() -> anyhow::Result<()> {
    let store = Store::connect_memory(EMBED_DIM).await?;
    let plan = owner_plan("ws:a");
    let trained = sealed(&["memory:rule"], "adapters/e.aa.safetensors", "aa");

    let report = mint(&store, &plan, trained, "rig").await?;

    assert!(report.drift.is_empty());
    let Some(e) = expert::get(&store, &expert_id(&plan.tenant, &plan.user, &plan.scope)).await?
    else {
        anyhow::bail!("the expert was not minted");
    };
    assert_eq!(e.capability_card.get(drift::CARD_DRIFT), None);
    Ok(())
}
