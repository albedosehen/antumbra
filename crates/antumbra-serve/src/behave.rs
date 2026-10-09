//! Train a user's accepted behaviors into a private standing expert, on the GPU.
//!
//! The procedure:
//! 1. Score the base model on each behavior's held-out examples and on
//!    controls no behavior governs.
//! 2. Collect its own answers to everyday prompts as replay.
//! 3. Train on the behaviors' examples beside that replay.
//! 4. Score again, and mint the expert, holding the behaviors it learned,
//!    when at least one clearly rose, none it missed got worse, and the
//!    controls held (`antumbra_train::behave::admit`).
//!
//! The expert is the user's own, private, and one per scope: a repository,
//! or everywhere. A re-run replaces it.
//!
//! The work is three phases, so a server can hold its store only for the
//! first and the last: [`plan`] reads the behaviors, [`train`] runs on the
//! GPU and touches no store, and [`mint`] writes the expert.
//! [`train_behaviors`] runs all three.

use chrono::Utc;

use antumbra_core::behavior::{self, learnable, Spec, State, Status};
use antumbra_core::ports::Embedder;
use antumbra_core::{
    Expert, ExpertId, Generation, Memory, MemoryId, Result, RunId, TenantId, UserId,
};
use antumbra_store::repo::{expert, memory, principal};
use antumbra_store::{Store, EMBED_DIM};
use antumbra_train::behave::{self as course, Verdict};
use antumbra_train::{
    capture_corrections, eval_pass_rate, CandleModelLoader, CorpusTask, ModelLoader, RaftConfig,
};

/// What a run did: the expert minted, if admitted, and the scores behind it.
#[derive(Debug, Clone)]
pub struct BehaveReport {
    /// The behaviors taught, by id.
    pub behaviors: Vec<String>,
    /// How many base answers were replayed beside them.
    pub replay: usize,
    pub verdict: Verdict,
    /// The minted expert and its adapter, when admitted.
    pub expert: Option<(ExpertId, String)>,
}

/// The private standing expert for a user's behaviors in `scope`.
pub fn expert_id(user: &UserId, scope: &str) -> ExpertId {
    ExpertId::new(format!("expert:{}:behavior:{scope}", user.as_str()))
}

/// The training a standing expert gets when nobody chose one: three epochs at
/// 1.5e-4, with answers up to 256 tokens so no replay answer is cut short.
/// `antumbra behave` defaults to the same. Validation trained four behaviors
/// at 3e-4; at that rate nine taught together broke the expert (command
/// controls 34/40 to 20/40), and at 1.5e-4 the same nine were all learned with
/// every control held, as were the four.
pub fn standing_config() -> RaftConfig {
    RaftConfig {
        rounds: 3,
        max_new_tokens: 256,
        learning_rate: 1.5e-4,
        ..RaftConfig::default()
    }
}

/// What a run teaches: one user's behaviors in one scope, as read.
#[derive(Debug, Clone)]
pub struct Plan {
    pub tenant: TenantId,
    pub user: UserId,
    pub scope: String,
    pub taught: Vec<(Memory, Spec)>,
}

impl Plan {
    /// The behaviors `memories` hold for `scope`. `None` when there is
    /// nothing to teach.
    pub fn of(tenant: &TenantId, user: &UserId, scope: &str, memories: &[Memory]) -> Option<Plan> {
        let taught = learnable(memories, scope);
        (!taught.is_empty()).then(|| Plan {
            tenant: tenant.clone(),
            user: user.clone(),
            scope: scope.to_string(),
            taught,
        })
    }

    /// The behaviors taught, by id.
    pub fn behaviors(&self) -> Vec<String> {
        self.taught
            .iter()
            .map(|(m, _)| m.id.as_str().to_string())
            .collect()
    }
}

/// Read `user`'s behaviors in `scope`. `None` when there is nothing to teach.
pub async fn plan(
    store: &Store,
    tenant: &TenantId,
    user: &UserId,
    scope: &str,
) -> Result<Option<Plan>> {
    principal::provision(store, tenant, user).await?;
    let compartment = behavior::compartment_id(tenant, user);
    let memories = memory::list_by_compartment(store, tenant, &compartment).await?;
    Ok(Plan::of(tenant, user, scope, &memories))
}

/// A trained expert's scores and, when admitted, what minting needs.
#[derive(Debug, Clone)]
pub struct Trained {
    pub verdict: Verdict,
    pub replay: usize,
    pub adapter_uri: String,
    pub base_model: String,
    pub fitness: f32,
    /// The centroid of the tasks it was taught, which routes it for its owner.
    pub capability_vec: Vec<f32>,
}

/// Score the base, train beside its own replay, and score again. Touches no
/// store. `cfg` sets the base model, epochs and learning rate; scoring is
/// greedy, one answer per task.
pub async fn train(plan: &Plan, embedder: &dyn Embedder, cfg: &RaftConfig) -> Result<Trained> {
    let (mut tasks, mut held): (Vec<CorpusTask>, Vec<CorpusTask>) = (Vec::new(), Vec::new());
    for (m, spec) in &plan.taught {
        let t = course::tasks(m.id.as_str(), spec);
        tasks.extend(t.train);
        held.extend(t.held);
    }
    held.extend(course::controls());
    // Each training prompt with the behavior it teaches.
    let taught_prompts: Vec<(Option<String>, String)> = tasks
        .iter()
        .map(|t| (t.skill.clone(), t.prompt.clone()))
        .collect();

    let cfg = RaftConfig {
        temperature: 0.0,
        samples_per_task: 1,
        ..cfg.clone()
    };
    let run = RunId::new(expert_id(&plan.user, &plan.scope).as_str().to_string());
    let loader = CandleModelLoader::new(cfg.clone());
    let verifier = antumbra_critic::CommandVerifier;
    let mut model = ModelLoader::load(&loader, &cfg.base_model, None).await?;

    let base = eval_pass_rate(&mut model, &verifier, &held, &run, 1).await?;
    let prompts = course::replay_prompts();
    let answered = eval_pass_rate(&mut model, &verifier, &prompts, &run, 1).await?;
    let answers: Vec<(String, String)> = answered
        .draws
        .into_iter()
        .map(|d| (d.task, d.completion))
        .collect();
    let replay = course::replay(&prompts, &answers);
    let replayed = replay.len();
    tasks.extend(replay);

    let out = capture_corrections(&mut model, &verifier, &tasks, &[], &run, &cfg, &[]).await?;
    let scores = eval_pass_rate(&mut model, &verifier, &held, &run, 1).await?;
    let verdict = course::admit(&plan.behaviors(), &base.per_task, &scores.per_task);

    // The expert routes by the tasks of the behaviors it learned.
    let learned = verdict.learned();
    let routed: Vec<&str> = taught_prompts
        .iter()
        .filter(|(skill, _)| skill.as_deref().is_some_and(|s| learned.contains(&s)))
        .map(|(_, p)| p.as_str())
        .collect();
    let mut acc = vec![0.0f32; EMBED_DIM];
    if verdict.admitted {
        for p in &routed {
            for (x, b) in acc.iter_mut().zip(embedder.embed(p).await?) {
                *x += b;
            }
        }
    }
    let n = routed.len().max(1) as f32;
    Ok(Trained {
        verdict,
        replay: replayed,
        adapter_uri: out.adapter_uri,
        base_model: cfg.base_model.clone(),
        fitness: scores.pass_rate,
        capability_vec: acc.iter().map(|v| v / n).collect(),
    })
}

/// Each behavior `plan` taught that is still as it was taught, read again:
/// one retired or recorded again while it trained is left out, for the next
/// run to see.
async fn still_as_taught(store: &Store, plan: &Plan) -> Result<Vec<Memory>> {
    let mut kept = Vec::new();
    for (taught, _) in &plan.taught {
        let Some(m) = memory::get(store, &plan.tenant, &MemoryId::new(taught.id.as_str())).await?
        else {
            continue;
        };
        let unchanged = m.content == taught.content
            && State::of(&m.evidence)
                .is_some_and(|s| matches!(s.status, Status::Accepted | Status::Trained));
        if unchanged {
            kept.push(m);
        }
    }
    Ok(kept)
}

/// Note on each behavior still as it was taught how this training went for
/// it, and, when the expert was admitted, mark the ones it learned trained
/// into `expert` and the ones it missed untrained.
async fn note(store: &Store, plan: &Plan, verdict: &Verdict, expert: Option<&str>) -> Result<()> {
    let set = behavior::fingerprint(&plan.taught);
    let now = Utc::now();
    for mut m in still_as_taught(store, plan).await? {
        let score = verdict.behaviors.iter().find(|s| s.id == m.id.as_str());
        let learned = score.is_some_and(|s| s.admitted);
        let training = behavior::Training {
            set: set.clone(),
            base: score.map_or(0.0, |s| s.base),
            expert: score.map_or(0.0, |s| s.expert),
            learned,
            admitted: expert.is_some(),
        };
        behavior::mark_training(&mut m.evidence, &training);
        match expert {
            Some(e) if learned => behavior::mark_trained(&mut m.evidence, e),
            Some(_) => behavior::mark_untrained(&mut m.evidence),
            None => {}
        }
        m.updated_at = now;
        memory::upsert(store, &m).await?;
    }
    Ok(())
}

/// Mint the standing expert when `trained` was admitted, replacing the one
/// before. It holds the behaviors it learned, and its card records the set it
/// was trained on, so the keeper leaves it be until that set changes. It is
/// placed on `host`, the machine whose disk holds its adapter.
///
/// Either way each behavior notes how the training went for it: its held-out
/// rates, whether it was learned, and whether the expert was admitted. A set
/// whose expert was refused is not trained again until it changes. A
/// behavior retired or recorded again while it trained is left as it now is:
/// the next run sees it.
pub async fn mint(
    store: &Store,
    plan: &Plan,
    trained: Trained,
    host: &str,
) -> Result<BehaveReport> {
    let ids = plan.behaviors();
    if !trained.verdict.admitted {
        note(store, plan, &trained.verdict, None).await?;
        return Ok(BehaveReport {
            behaviors: ids,
            replay: trained.replay,
            verdict: trained.verdict,
            expert: None,
        });
    }
    let id = expert_id(&plan.user, &plan.scope);
    let verdict = trained.verdict;
    let learned = verdict.learned();
    let rules: Vec<&str> = plan
        .taught
        .iter()
        .filter(|(m, _)| learned.contains(&m.id.as_str()))
        .map(|(_, s)| s.rule.as_str())
        .collect();
    let missed: Vec<&str> = ids
        .iter()
        .map(String::as_str)
        .filter(|i| !learned.contains(i))
        .collect();
    let now = Utc::now();
    let e = Expert {
        id: id.clone(),
        name: id.as_str().to_string(),
        base_model: trained.base_model,
        artifact_uri: trained.adapter_uri.clone(),
        capability_card: serde_json::json!({
            "behaviors": learned, "rules": rules, "missed": missed,
            "set": behavior::fingerprint(&plan.taught), "scope": plan.scope,
            "private": true, "standing": true,
            "controls": verdict.controls.iter().map(|c| serde_json::json!({
                "family": c.family, "base": c.base, "expert": c.expert,
            })).collect::<Vec<_>>(),
        }),
        capability_vec: Some(trained.capability_vec),
        fitness: trained.fitness,
        frozen_at: Some(now),
        generation: Generation::ZERO,
        owner: Some(plan.user.clone()),
        compartment: Some(behavior::compartment_id(&plan.tenant, &plan.user)),
        // The adapter is on `host`'s disk, and adapters stay out of sync, so
        // the row says which machine can open it.
        placed_on: Some(host.to_string()),
        created_at: now,
    };
    expert::delete(store, &e.id).await?;
    expert::insert(store, &e).await?;
    note(store, plan, &verdict, Some(id.as_str())).await?;
    Ok(BehaveReport {
        behaviors: ids,
        replay: trained.replay,
        verdict,
        expert: Some((e.id, trained.adapter_uri)),
    })
}

/// Train `user`'s behaviors in `scope` into their standing expert: [`plan`],
/// [`train`] and [`mint`] in one. `None` when there is nothing to teach.
pub async fn train_behaviors(
    store: &Store,
    embedder: &dyn Embedder,
    tenant: &TenantId,
    user: &UserId,
    scope: &str,
    cfg: &RaftConfig,
) -> Result<Option<BehaveReport>> {
    let Some(plan) = plan(store, tenant, user, scope).await? else {
        return Ok(None);
    };
    let trained = train(&plan, embedder, cfg).await?;
    Ok(Some(
        mint(store, &plan, trained, &antumbra_core::this_host()).await?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_standing_expert_trains_at_the_rate_nine_behaviors_held_at() {
        let cfg = standing_config();
        assert!(cfg.learning_rate <= 1.5e-4, "3e-4 broke nine behaviors");
        assert_eq!((cfg.rounds, cfg.max_new_tokens), (3, 256));
    }

    #[test]
    fn a_standing_expert_is_named_for_its_owner_and_scope() {
        assert_eq!(
            expert_id(&UserId::new("user:a"), "everywhere").as_str(),
            "expert:user:a:behavior:everywhere"
        );
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
        };
        let report = mint(&store, &plan, trained, "rig").await.unwrap();
        let minted = report.expert.unwrap().0;
        assert_eq!(minted.as_str(), "expert:user:a:behavior:everywhere");
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
}
