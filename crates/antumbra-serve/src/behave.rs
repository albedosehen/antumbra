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

use antumbra_core::behavior::missed::{self, Missed};
use antumbra_core::behavior::{self, learnable, Spec, State, Status};
use antumbra_core::ports::Embedder;
use antumbra_core::{
    BoundaryId, Expert, ExpertId, FailureBoundary, Generation, Memory, MemoryId, Result, RunId,
    Shadow, ShadowId, ShadowStatus, TenantId, UserId,
};
use antumbra_store::repo::{boundary, expert, memory, principal, shadow};
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

/// The seed a training of `plan` runs under: [`behavior::seed`] of what it
/// teaches, so the same behaviors train the same way for any user, under any
/// scope, in any process.
pub fn seed_of(plan: &Plan) -> u64 {
    behavior::seed(&plan.taught)
}

/// A trained expert's scores and, when admitted, what minting needs.
#[derive(Debug, Clone)]
pub struct Trained {
    pub verdict: Verdict,
    pub replay: usize,
    pub adapter_uri: String,
    pub base_model: String,
    pub fitness: f32,
    /// The base model's pass rate on the same held-out tasks, before
    /// training: where the shadow's reward curve starts.
    pub base_fitness: f32,
    /// The centroid of the tasks it was taught, which routes it for its owner.
    pub capability_vec: Vec<f32>,
    /// What it was trained with, kept on the card so a measurement can be
    /// read back against its settings.
    pub recipe: behavior::Recipe,
    /// Each behavior's first held-out task the expert failed, with its
    /// answer, by behavior id: noted on the ones it did not learn.
    pub misses: Vec<(String, Missed)>,
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

    // Seeded from the set taught, so the same set trains the same way in any
    // process, whatever was trained before it; the node's generation seed
    // shifts it for measuring how much a training varies.
    let cfg = RaftConfig {
        temperature: 0.0,
        samples_per_task: 1,
        seed: Some(seed_of(plan)),
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
        misses: course::misses(&held, &scores.draws),
        verdict,
        replay: replayed,
        adapter_uri: out.adapter_uri,
        base_model: cfg.base_model.clone(),
        fitness: scores.pass_rate,
        base_fitness: base.pass_rate,
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
/// into `expert` and the ones it missed untrained. One it did not learn also
/// notes the held-out task it was missed on and the expert's answer, from
/// `misses`; one it learned has that note cleared.
async fn note(
    store: &Store,
    plan: &Plan,
    verdict: &Verdict,
    misses: &[(String, Missed)],
    expert: Option<&str>,
) -> Result<()> {
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
        let miss = misses
            .iter()
            .find(|(id, _)| !learned && id == m.id.as_str())
            .map(|(_, x)| x);
        missed::mark(&mut m.evidence, miss);
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

/// The shadow a training of `plan` is recorded as: its scope's standing
/// expert, under `shadow:`, and the set it teaches, so the same set trained
/// again is the same row and a changed set a new one.
pub fn shadow_id(plan: &Plan) -> ShadowId {
    let set = behavior::fingerprint(&plan.taught);
    ShadowId::new(format!(
        "shadow:{}:{}:behavior:{}:{}",
        plan.tenant.as_str(),
        plan.user.as_str(),
        plan.scope,
        &set[..set.len().min(12)]
    ))
}

/// Record a training of `plan` as begun: its shadow, exploring, in the
/// penumbra while the GPU works. The expert it would replace is its parent.
/// [`mint`] settles it, and [`abandon`] prunes it when the training fails.
pub async fn begin(store: &Store, plan: &Plan) -> Result<Shadow> {
    let current = expert_id(&plan.tenant, &plan.user, &plan.scope);
    let parent = expert::get(store, &current).await?.map(|e| e.id);
    let mut sh = Shadow::spawn(shadow_id(plan), Generation::ZERO, parent, Utc::now());
    sh.advance_to(ShadowStatus::Exploring)?;
    shadow::upsert(store, &sh).await?;
    Ok(sh)
}

/// Prune the shadow [`begin`] recorded for `plan`, when its training failed
/// before it could be scored. One already settled is left as it is.
pub async fn abandon(store: &Store, plan: &Plan) -> Result<()> {
    let Some(mut sh) = shadow::get(store, &shadow_id(plan)).await? else {
        return Ok(());
    };
    if sh.status.is_terminal() {
        return Ok(());
    }
    sh.advance_to(ShadowStatus::Pruned)?;
    shadow::upsert(store, &sh).await
}

/// Settle `plan`'s shadow from its training: scored on the held-out tasks,
/// the curve running from the base model's pass rate to the trained one's,
/// then graduated into the standing expert when admitted, else pruned. A
/// training [`begin`] did not record starts its shadow here, from `parent`.
async fn settle(
    store: &Store,
    plan: &Plan,
    parent: Option<ExpertId>,
    adapter_uri: String,
    curve: Vec<f32>,
    admitted: bool,
) -> Result<()> {
    let id = shadow_id(plan);
    let mut sh = match shadow::get(store, &id).await? {
        Some(sh) if !sh.status.is_terminal() => sh,
        _ => Shadow::spawn(id, Generation::ZERO, parent, Utc::now()),
    };
    if sh.status == ShadowStatus::Spawning {
        sh.advance_to(ShadowStatus::Exploring)?;
    }
    if sh.status == ShadowStatus::Exploring {
        sh.advance_to(ShadowStatus::Scoring)?;
    }
    sh.adapter_uri = Some(adapter_uri);
    sh.reward_curve = curve;
    sh.advance_to(if admitted {
        ShadowStatus::Graduated
    } else {
        ShadowStatus::Pruned
    })?;
    shadow::upsert(store, &sh).await
}

/// The prefix of every boundary recorded beside the standing expert `expert`.
fn boundary_prefix(expert: &ExpertId) -> String {
    format!("boundary:{}:", expert.as_str())
}

/// Bring the boundaries beside `plan`'s standing expert in step with its
/// latest training. Each behavior the training missed is an open boundary:
/// the rule, held fixed, and the held-out task the expert did not follow it
/// on, with its answer. Open (no context where it is acceptable), so it is
/// recorded and never gates routing. Its confidence is the share of the
/// behavior's held-out tasks the expert failed. It belongs to the user, as
/// the rule does. One an admitted expert learned, or one no longer taught,
/// goes; one a refused training learned stays, since nothing serves it.
async fn record_boundaries(
    store: &Store,
    plan: &Plan,
    verdict: &Verdict,
    misses: &[(String, Missed)],
) -> Result<()> {
    let prefix = boundary_prefix(&expert_id(&plan.tenant, &plan.user, &plan.scope));
    let learned = verdict.learned();
    let now = Utc::now();
    for (m, spec) in &plan.taught {
        let id = m.id.as_str();
        let Some(score) = verdict.behaviors.iter().find(|s| s.id == id && !s.admitted) else {
            continue;
        };
        let mut context = serde_json::json!({ "scope": plan.scope });
        if let Some((_, miss)) = misses.iter().find(|(b, _)| b == id) {
            context["task"] = miss.task.clone().into();
            context["answer"] = miss.answer.clone().into();
        }
        let b = FailureBoundary {
            id: BoundaryId::new(format!("{prefix}{id}")),
            behavior: spec.rule.clone(),
            fail_context: context,
            near_ok_context: None,
            governing_features: Vec::new(),
            grain: None,
            context_vec: None,
            ok_context_vec: None,
            confidence: (1.0 - score.expert).clamp(0.0, 1.0),
            generation: Generation::ZERO,
            owner: Some(plan.user.clone()),
            created_at: now,
        };
        boundary::upsert(store, &b).await?;
    }
    let taught = plan.behaviors();
    for b in boundary::list(store).await? {
        let Some(behavior) = b.id.as_str().strip_prefix(&prefix) else {
            continue;
        };
        let gone = !taught.iter().any(|t| t == behavior);
        let served = verdict.admitted && learned.contains(&behavior);
        if gone || served {
            boundary::delete(store, &b.id).await?;
        }
    }
    Ok(())
}

/// Remove the boundaries recorded beside `expert`: a standing expert dropped
/// with nothing left to teach takes them with it.
pub async fn forget_boundaries(store: &Store, expert: &ExpertId) -> Result<()> {
    let prefix = boundary_prefix(expert);
    for b in boundary::list(store).await? {
        if b.id.as_str().starts_with(&prefix) {
            boundary::delete(store, &b.id).await?;
        }
    }
    Ok(())
}

/// Record a standing expert minted before trainings were: its graduated
/// shadow, from its card, and an open boundary for each behavior it missed,
/// with the miss noted on the behavior. Only while `expert` is in step with
/// the set `plan` teaches, and only once: one whose shadow is recorded is
/// left be. The card keeps the trained pass rate and not the base model's,
/// so the curve is that one reading, and a card without a behavior's scores
/// reads its pass rate as even.
pub async fn record_standing(store: &Store, plan: &Plan, expert: &Expert) -> Result<()> {
    let set = behavior::fingerprint(&plan.taught);
    if behavior::trained_set(expert) != Some(set.as_str()) {
        return Ok(());
    }
    let id = shadow_id(plan);
    if shadow::get(store, &id).await?.is_some() {
        return Ok(());
    }
    let scores = expert.capability_card.get(behavior::CARD_SCORES);
    let score = |id: &str, key: &str| scores.and_then(|s| s.get(id)?.get(key)?.as_f64());
    let learned = behavior::taught_by(expert);
    let verdict = Verdict {
        admitted: true,
        behaviors: plan
            .taught
            .iter()
            .map(|(m, _)| {
                let id = m.id.as_str();
                course::BehaviorScore {
                    id: id.to_string(),
                    base: score(id, "base").unwrap_or(0.0) as f32,
                    expert: score(id, "expert").unwrap_or(0.5) as f32,
                    admitted: learned.iter().any(|l| l == id),
                    held: score(id, "held").unwrap_or(0.0) as usize,
                    worse: score(id, "worse").unwrap_or(0.5) as f32,
                }
            })
            .collect(),
        controls: Vec::new(),
        reasons: Vec::new(),
    };
    let misses: Vec<(String, Missed)> = plan
        .taught
        .iter()
        .filter_map(|(m, _)| missed::of(&m.evidence).map(|x| (m.id.as_str().to_string(), x)))
        .collect();
    record_boundaries(store, plan, &verdict, &misses).await?;
    let mut sh = Shadow::spawn(id, Generation::ZERO, None, expert.created_at);
    sh.adapter_uri = Some(expert.artifact_uri.clone());
    sh.reward_curve = vec![expert.fitness];
    for to in [
        ShadowStatus::Exploring,
        ShadowStatus::Scoring,
        ShadowStatus::Graduated,
    ] {
        sh.advance_to(to)?;
    }
    shadow::upsert(store, &sh).await
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
/// rates, whether it was learned, and whether the expert was admitted; and one
/// not learned notes the task it was missed on and the expert's answer. A set
/// whose expert was refused is not trained again until it changes. A
/// behavior retired or recorded again while it trained is left as it now is:
/// the next run sees it.
///
/// Either way, too, the training settles its shadow in the penumbra,
/// graduated or pruned, and the behaviors it missed stand as open boundaries
/// beside the expert in the antumbra.
pub async fn mint(
    store: &Store,
    plan: &Plan,
    trained: Trained,
    host: &str,
) -> Result<BehaveReport> {
    let ids = plan.behaviors();
    let id = expert_id(&plan.tenant, &plan.user, &plan.scope);
    // The expert this training would replace: its shadow's parent.
    let parent = expert::get(store, &id).await?.map(|e| e.id);
    let curve = vec![trained.base_fitness, trained.fitness];
    record_boundaries(store, plan, &trained.verdict, &trained.misses).await?;
    if !trained.verdict.admitted {
        note(store, plan, &trained.verdict, &trained.misses, None).await?;
        settle(store, plan, parent, trained.adapter_uri, curve, false).await?;
        return Ok(BehaveReport {
            behaviors: ids,
            replay: trained.replay,
            verdict: trained.verdict,
            expert: None,
        });
    }
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
    note(store, plan, &verdict, &trained.misses, Some(id.as_str())).await?;
    settle(
        store,
        plan,
        parent,
        trained.adapter_uri.clone(),
        curve,
        true,
    )
    .await?;
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
    begin(store, &plan).await?;
    let trained = match train(&plan, embedder, cfg, confidence).await {
        Ok(t) => t,
        Err(e) => {
            abandon(store, &plan).await?;
            return Err(e);
        }
    };
    Ok(Some(
        mint(store, &plan, trained, &antumbra_core::this_host()).await?,
    ))
}

#[cfg(test)]
mod tests;
