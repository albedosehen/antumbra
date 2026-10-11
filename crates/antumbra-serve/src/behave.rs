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

/// The private standing expert for a user's behaviors in `scope`, in
/// `tenant`. The workspace is in the id, and so in the adapter's file name:
/// every hosted workspace's first user has the same id, and without it two
/// workspaces would replace each other's expert and adapter.
pub fn expert_id(tenant: &TenantId, user: &UserId, scope: &str) -> ExpertId {
    ExpertId::new(format!(
        "expert:{}:{}:behavior:{scope}",
        tenant.as_str(),
        user.as_str()
    ))
}

/// The id a standing expert was minted under before the workspace was in it.
fn legacy_expert_id(user: &UserId, scope: &str) -> ExpertId {
    ExpertId::new(format!("expert:{}:behavior:{scope}", user.as_str()))
}

/// The training a standing expert gets, from the node's `recipe`: its rank,
/// epochs and learning rate, with answers up to 256 tokens so no replay answer
/// is cut short. `antumbra behave` builds its config the same way. Alpha moves
/// with the rank, so `alpha / rank`, the scale every expert is trained and
/// served at, stays what the default is. The default recipe is three epochs at
/// 1.5e-4 and rank 16: validation trained four behaviors at 3e-4; at that rate
/// nine taught together broke the expert (command controls 34/40 to 20/40), and
/// at 1.5e-4 the same nine were all learned with every control held.
pub fn standing_config(recipe: &behavior::Recipe) -> RaftConfig {
    let base = RaftConfig::default();
    RaftConfig {
        lora_rank: recipe.rank,
        lora_alpha: base.lora_alpha * recipe.rank as f64 / base.lora_rank as f64,
        rounds: recipe.rounds,
        max_new_tokens: 256,
        learning_rate: recipe.learning_rate,
        ..base
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
    /// What it was trained with, kept on the card so a measurement can be
    /// read back against its settings.
    pub recipe: behavior::Recipe,
}

/// Score the base, train beside its own replay, and score again. Touches no
/// store. `cfg` sets the base model, epochs and learning rate; scoring is
/// greedy, one answer per task.
pub async fn train(
    plan: &Plan,
    embedder: &dyn Embedder,
    cfg: &RaftConfig,
    confidence: f64,
) -> Result<Trained> {
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
    let run = RunId::new(
        expert_id(&plan.tenant, &plan.user, &plan.scope)
            .as_str()
            .to_string(),
    );
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
    let verdict = course::admit(
        &plan.behaviors(),
        &base.per_task,
        &scores.per_task,
        confidence,
    );

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
        recipe: behavior::Recipe {
            rank: cfg.lora_rank,
            rounds: cfg.rounds,
            learning_rate: cfg.learning_rate,
            regression_confidence: confidence,
        },
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
            held: score.map_or(0, |s| s.held),
            worse: score.map_or(0.5, |s| s.worse),
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

/// A served expert measured, beside the base model, on the examples its
/// behaviors gained since it was trained: ground truth the training never saw.
#[derive(Debug, Clone)]
pub struct Audited {
    pub expert: ExpertId,
    /// The set the expert was trained on, from its card.
    pub set: String,
    /// Each behavior with later examples: its audit, and the prints of the
    /// examples measured.
    pub behaviors: Vec<(String, behavior::Audit, Vec<String>)>,
}

impl Audited {
    /// How many later examples were measured, over every behavior.
    pub fn fresh(&self) -> usize {
        self.behaviors.iter().map(|(_, a, _)| a.fresh).sum()
    }

    /// How many of them the base model passed.
    pub fn base(&self) -> usize {
        self.behaviors.iter().map(|(_, a, _)| a.base).sum()
    }

    /// How many the expert passed.
    pub fn passed(&self) -> usize {
        self.behaviors.iter().map(|(_, a, _)| a.expert).sum()
    }
}

/// Score `current`, the scope's served expert, and the base model on the
/// examples `plan`'s behaviors gained since `current` was trained, before the
/// changed set replaces it. Touches no store. `None` when nothing is new, or
/// the card is from before examples were kept on it.
pub async fn audit(plan: &Plan, current: &Expert, cfg: &RaftConfig) -> Result<Option<Audited>> {
    let fresh = behavior::fresh_examples(&plan.taught, &current.capability_card);
    if fresh.is_empty() {
        return Ok(None);
    }
    let set = current
        .capability_card
        .get("set")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    let mut tasks = Vec::new();
    for (id, examples) in &fresh {
        let Some((_, spec)) = plan.taught.iter().find(|(m, _)| m.id.as_str() == id) else {
            continue;
        };
        tasks.extend(course::audit_tasks(id, spec, examples));
    }
    let cfg = RaftConfig {
        temperature: 0.0,
        samples_per_task: 1,
        ..cfg.clone()
    };
    let run = RunId::new(format!("{}#audit", current.id.as_str()));
    let loader = CandleModelLoader::new(cfg.clone());
    let verifier = antumbra_critic::CommandVerifier;
    let mut model = ModelLoader::load(&loader, &cfg.base_model, None).await?;
    let base = eval_pass_rate(&mut model, &verifier, &tasks, &run, 1).await?;
    drop(model);
    let mut model = ModelLoader::load(
        &loader,
        &cfg.base_model,
        Some(current.artifact_uri.as_str()),
    )
    .await?;
    let expert = eval_pass_rate(&mut model, &verifier, &tasks, &run, 1).await?;
    let behaviors = fresh
        .iter()
        .map(|(id, examples)| {
            let prefix = format!("{id}#");
            let (b, n) = course::counts(&base.per_task, &prefix);
            let (e, _) = course::counts(&expert.per_task, &prefix);
            // The bet admission made on it, under its id or the one it
            // supersedes.
            let earlier = plan
                .taught
                .iter()
                .find(|(m, _)| m.id.as_str() == id)
                .and_then(|(m, _)| State::of(&m.evidence))
                .and_then(|s| s.supersedes);
            let ids: Vec<&str> = std::iter::once(id.as_str())
                .chain(earlier.as_deref())
                .collect();
            let audit = behavior::Audit {
                set: set.clone(),
                fresh: n,
                base: b,
                expert: e,
                worse: behavior::card_worse(&current.capability_card, &ids),
            };
            let prints = examples.iter().map(behavior::example_print).collect();
            (id.clone(), audit, prints)
        })
        .collect();
    Ok(Some(Audited {
        expert: current.id.clone(),
        set,
        behaviors,
    }))
}

/// Note `audited` on each behavior it measured that is still as taught, and
/// on the expert's card the examples now audited, so a later audit measures
/// only what is newer still. A behavior recorded again takes over the card's
/// entry for the one it supersedes, so a further recording is still followed.
/// The audit changes nothing else: not the expert, not the behaviors'
/// status, not the training to come.
pub async fn note_audit(store: &Store, plan: &Plan, audited: &Audited) -> Result<()> {
    let now = Utc::now();
    let mut lineage: Vec<(String, String)> = Vec::new();
    for mut m in still_as_taught(store, plan).await? {
        let Some((_, audit, _)) = audited
            .behaviors
            .iter()
            .find(|(id, _, _)| id == m.id.as_str())
        else {
            continue;
        };
        if let Some(earlier) = State::of(&m.evidence).and_then(|s| s.supersedes) {
            lineage.push((m.id.as_str().to_string(), earlier));
        }
        behavior::mark_audit(&mut m.evidence, audit);
        m.updated_at = now;
        memory::upsert(store, &m).await?;
    }
    if let Some(mut e) = expert::get(store, &audited.expert).await? {
        let card = &mut e.capability_card;
        for (id, earlier) in &lineage {
            for key in [behavior::CARD_EXAMPLES, behavior::CARD_AUDITED] {
                if card[key].get(id.as_str()).is_none() {
                    if let Some(known) = card[key].get(earlier.as_str()).cloned() {
                        card[key][id.as_str()] = known;
                    }
                }
            }
        }
        for (id, _, prints) in &audited.behaviors {
            let slot = &mut card[behavior::CARD_AUDITED][id.as_str()];
            let mut list = slot.as_array().cloned().unwrap_or_default();
            list.extend(prints.iter().map(|p| serde_json::Value::String(p.clone())));
            *slot = serde_json::Value::Array(list);
        }
        expert::delete(store, &e.id).await?;
        expert::insert(store, &e).await?;
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
    let id = expert_id(&plan.tenant, &plan.user, &plan.scope);
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
    // Each taught behavior's examples by print, so an audit can tell the
    // ones it gains later from these.
    let examples: serde_json::Map<String, serde_json::Value> = plan
        .taught
        .iter()
        .map(|(m, s)| {
            let prints: Vec<String> = s.examples.iter().map(behavior::example_print).collect();
            (m.id.as_str().to_string(), serde_json::json!(prints))
        })
        .collect();
    let mut card = serde_json::json!({
        "behaviors": learned, "rules": rules, "missed": missed,
        "set": behavior::fingerprint(&plan.taught), "scope": plan.scope,
        "private": true, "standing": true,
        "recipe": {
            "rank": trained.recipe.rank, "rounds": trained.recipe.rounds,
            "learning_rate": trained.recipe.learning_rate,
            "regression_confidence": trained.recipe.regression_confidence,
        },
        "controls": verdict.controls.iter().map(|c| serde_json::json!({
            "family": c.family, "base": c.base, "expert": c.expert,
        })).collect::<Vec<_>>(),
    });
    card[behavior::CARD_EXAMPLES] = serde_json::Value::Object(examples);
    // Each behavior's held-out scores at admission: the bet an audit checks.
    let scores: serde_json::Map<String, serde_json::Value> = verdict
        .behaviors
        .iter()
        .map(|s| {
            let score = serde_json::json!({
                "base": s.base, "expert": s.expert, "held": s.held,
                "worse": s.worse, "learned": s.admitted,
            });
            (s.id.clone(), score)
        })
        .collect();
    card[behavior::CARD_SCORES] = serde_json::Value::Object(scores);
    let e = Expert {
        id: id.clone(),
        name: id.as_str().to_string(),
        base_model: trained.base_model,
        artifact_uri: trained.adapter_uri.clone(),
        capability_card: card,
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
    // The scope's expert minted under the id without the workspace goes,
    // when it is this workspace's: the scope keeps one. Another workspace's,
    // under the same old id, is left as it is.
    let legacy = legacy_expert_id(&plan.user, &plan.scope);
    if let Some(old) = expert::get(store, &legacy).await? {
        if behavior::is_standing_for(&old, &plan.tenant, &plan.user) {
            expert::delete(store, &legacy).await?;
        }
    }
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
    confidence: f64,
) -> Result<Option<BehaveReport>> {
    let Some(plan) = plan(store, tenant, user, scope).await? else {
        return Ok(None);
    };
    let trained = train(&plan, embedder, cfg, confidence).await?;
    Ok(Some(
        mint(store, &plan, trained, &antumbra_core::this_host()).await?,
    ))
}

#[cfg(test)]
mod tests {
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
            "every hosted workspace's first user is user:owner"
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
}
