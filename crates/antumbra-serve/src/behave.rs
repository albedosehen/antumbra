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
mod tests;
