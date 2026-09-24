//! # antumbra-loop: the durable generational loop
//!
//! The durable generational loop as a resumable state-machine-in-DB:
//! `grow -> explore -> score -> decide -> consolidate -> grow`. The head is
//! persisted after every transition, so killing the process and restarting
//! resumes from the last checkpoint: the state value *is* the checkpoint.
//!
//! Each generation now writes its full lineage to the substrate: the shadow
//! model that holds the plasticity and its status transitions, the per-step
//! reward signals from the critic, an evaluation run against the unified
//! substrate, the graduated frozen expert, and, on prune, an open-negative
//! failure boundary (not yet actionable, because the loop has no probe to
//! recover C' until serving is wired). Driven entirely by the injected ports,
//! so it runs with fakes.

use chrono::Utc;
use sha2::{Digest, Sha256};

use antumbra_boundary::finding_to_boundary;
use antumbra_core::generational::{GenerationHead, LoopCommand, LoopState};
use antumbra_core::ports::{Embedder, TrainOutcome, Trainer};
use antumbra_core::slice::Partition;
use antumbra_core::{
    BoundaryId, EvalStatus, EvaluationRun, Expert, ExpertId, FailureBoundary, Generation, Grain,
    Result, RewardSignal, RunId, ShadowId, ShadowStatus, SubjectKind, TrainingRecipe,
};
use antumbra_eclipse::instrument::GenerationReport as InstrumentReport;
use antumbra_eclipse::{Trend, Watch};
use antumbra_store::repo::{
    boundary, evaluation, expert, generation, loop_control, reward, shadow,
};
use antumbra_store::Store;

mod cohort;
mod measure;
mod recipe;
pub mod search;
pub use cohort::{CohortMember, Remeasure};
use measure::Measurement;

/// The shadow a generation of a run trains. It also names that generation's
/// recipe row, so the two are found from each other.
pub(crate) fn shadow_id(run_id: &RunId, generation: Generation) -> ShadowId {
    ShadowId::new(format!("{run_id}:g{}", generation.0))
}

/// Confidence stamped on a correction-derived boundary. The context pair is
/// ground-truth verified, so the scope is trustworthy -- but confidence tempers
/// the inhibition magnitude at the counterfactual boundary, so it is high, not
/// absolute.
const CORRECTION_BOUNDARY_CONFIDENCE: f32 = 0.8;

/// Render a behavior placed in a context into the text the embedder turns into a
/// boundary's context vector. Both C and C' embed through this, so they share
/// the behavior and differ only by context -- exactly the contrast the
/// relative-margin inhibition reads to separate near-identical scopes (the
/// counterfactual boundary feeding the boundary-conditioned gate). `Value`'s
/// `Display` is compact JSON.
fn render_scope(behavior: &str, context: &serde_json::Value) -> String {
    format!("{behavior} | {context}")
}

#[derive(Debug, Clone)]
pub struct LoopConfig {
    /// Fitness at or above which a shadow graduates into a frozen expert.
    pub graduate_threshold: f32,
    /// The shared base every adapter rides on (the frozen-expert population).
    pub base_model: String,
    /// How the corpus is split for the standing instruments (ADR-0022): which
    /// tasks selection may see, which are held out, which are audited. Carried
    /// in the config rather than derived, because the seed decides what every
    /// measurement means and a generation has to record which one it ran under.
    ///
    /// The trainer is asked to enforce it, so it changes what a run learns:
    /// held-out and audit tasks are measured and never trained on, and fitness
    /// is computed over visible tasks alone. `None` (the default) trains on
    /// every task and leaves generations unmeasured, which is the honest shape
    /// for a run with nothing held out. It is off by default because the corpora
    /// shipped with the repository are a handful of tasks each, and several hash
    /// entirely into the withheld slices.
    pub partition: Option<Partition>,
    /// The audit slice is measured every `audit_every` generations, counting
    /// from generation 0, and skipped in between (ADR-0022: "evaluated every k
    /// generations and only logged"). The record names no k. The default is
    /// the largest one the default [`Watch`] can always read: it asks for 4
    /// audited generations in a window of 10, and 10 consecutive generations
    /// hold at least 5 multiples of 2 but only 3 of 3. A test holds the two
    /// dials together. Zero is read as one.
    pub audit_every: u32,
    /// The recipe each generation's shadow trains under (ADR-0022 S-1).
    /// `None` (the default) trains under the trainer's own settings, as every
    /// run did before the recipe was searched. Either way the trainer reports
    /// the recipe it used, and that is what the generation's recipe row holds.
    /// A searched run starts from it.
    pub recipe: Option<TrainingRecipe>,
    /// Search the recipe (ADR-0022 S-1): each generation trains a cohort under
    /// recipes the search proposes, every member from the base, and carries the
    /// best forward. `None` (the default) trains one shadow under `recipe`.
    pub search: Option<search::SearchPolicy>,
    /// Judge graduation on a re-measurement (ADR-0022 S-1's fourth
    /// constraint): the carried-forward shadow is evaluated again, on the
    /// held-out slice when there is one, once per fresh seed, and the threshold
    /// applies to the mean. `None` (the default) judges on training fitness, or
    /// on its shrunk form under a search.
    pub remeasure: Option<Remeasure>,
    /// How the audit-slice trend is read across generations: the window, how
    /// many audited generations it needs, and the share of a search gain the
    /// audit slice must show for the gain to count as carried.
    pub watch: Watch,
}

impl Default for LoopConfig {
    fn default() -> Self {
        Self {
            graduate_threshold: 0.5,
            base_model: "code-base".into(),
            partition: None,
            audit_every: 2,
            watch: Watch::default(),
            recipe: None,
            search: None,
            remeasure: None,
        }
    }
}

/// What one generation did.
#[derive(Debug, Clone)]
pub struct GenerationReport {
    pub generation: Generation,
    pub shadow: ShadowId,
    pub fitness: f32,
    pub graduated: bool,
    /// Per-round pass-rate (RAFT reward curve); rising means the adapter is
    /// learning to satisfy the verifier.
    pub reward_curve: Vec<f32>,
    /// Frozen experts whose regression fingerprint drifted this generation: the
    /// no-forgetting kill criterion firing. Empty when the freeze held
    /// (the expected case); a non-empty list is a serious integrity alarm.
    pub regressions: Vec<ExpertId>,
    /// What the standing instruments made of this generation (ADR-0022): the
    /// visible-minus-held-out gap banded by task size, the audit slice, and the
    /// impossible set. `None` when nothing was held out, when the trainer did
    /// not confirm it withheld what was asked, or when it reported no per-task
    /// results -- in each case a report would be a number that looks like a
    /// measurement and is not.
    pub instruments: Option<InstrumentReport>,
    /// What the audit slice says across this run's measured generations,
    /// ending at this one: whether a climbing search score is carrying
    /// competence the loop cannot select for. `None` when this generation was
    /// not measured; `Inconclusive` until enough generations have been.
    pub trend: Option<Trend>,
    /// The recipe the shadow trained under, as its trainer reported it. `None`
    /// when the trainer did not say, in which case no recipe row was written.
    pub recipe: Option<TrainingRecipe>,
    /// Every shadow the generation trained, the carried-forward one included:
    /// one without a search, the cohort with one.
    pub cohort: Vec<CohortMember>,
    /// The number the graduation threshold was applied to: the carried-forward
    /// shadow's fitness, shrunk toward the cohort's mean when it was chosen
    /// from a cohort, or the mean of its re-measurement when the loop took one.
    pub graduation_score: f32,
    /// The re-measurement graduation was judged on, when the loop took one.
    pub remeasured: Option<antumbra_core::ports::Remeasurement>,
}

/// The frozen-expert regression fingerprint: `sha256` of the adapter's bytes, so a
/// frozen expert whose weights file is mutated under it is caught. When the file
/// is absent (the demo trainer, or an artifact not present on this node) it falls
/// back to a digest **of the uri**: still deterministic and distinct per adapter
/// (so re-pointing a frozen expert to a different missing path still trips the
/// check), but without echoing the raw path into the stored/exposed fingerprint.
fn fingerprint(adapter_uri: &str) -> String {
    let digest = |bytes: &[u8]| -> String {
        Sha256::digest(bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    };
    match std::fs::read(adapter_uri) {
        Ok(bytes) => format!("sha256:{}", digest(&bytes)),
        Err(_) => format!("absent:{}", digest(adapter_uri.as_bytes())),
    }
}

/// Drives the loop over a [`Store`] using injected ports. Holds no durable
/// state of its own; everything lives in the head + the store.
pub struct GenerationLoop<'a> {
    store: &'a Store,
    trainer: &'a dyn Trainer,
    embedder: &'a dyn Embedder,
    cfg: LoopConfig,
}

impl<'a> GenerationLoop<'a> {
    pub fn new(
        store: &'a Store,
        trainer: &'a dyn Trainer,
        embedder: &'a dyn Embedder,
        cfg: LoopConfig,
    ) -> Self {
        Self {
            store,
            trainer,
            embedder,
            cfg,
        }
    }

    /// Resume from the persisted head, or initialize and persist a fresh one.
    pub async fn resume_or_init(&self, run_id: &RunId) -> Result<GenerationHead> {
        if let Some(head) = generation::load_head(self.store, run_id).await? {
            return Ok(head);
        }
        let head = GenerationHead::new(run_id.clone(), Utc::now());
        generation::save_head(self.store, &head).await?;
        Ok(head)
    }

    /// Run exactly one generation, starting and ending at `Grow` (the
    /// resumable boundary). The head is persisted after every transition and
    /// the full lineage is written to the substrate.
    pub async fn run_generation(&self, head: &mut GenerationHead) -> Result<GenerationReport> {
        let run_id = head.run_id.clone();
        let generation = head.generation;

        // grow -> explore: spawn the shadows that hold the plasticity and
        // train them on verified outcomes, withholding what the partition keeps
        // from selection. One, or a cohort when the recipe is searched.
        let holdout = self.holdout_for(generation);
        let members = self.train_cohort(head, holdout).await?;
        let (winner, cohort) = self.select(members).await?;
        let cohort::Member {
            shadow: mut sh,
            outcome,
            recipe,
        } = winner;
        let shadow_id = sh.id.clone();
        let mut measured = self
            .measure(&run_id, generation, holdout.as_ref(), &outcome)
            .await?;

        // explore -> score: record the per-step reward from the critic.
        self.advance(head, LoopState::Score).await?;
        sh.advance_to(ShadowStatus::Scoring)?;
        shadow::upsert(self.store, &sh).await?;
        self.record_rewards(&run_id, &outcome).await?;
        // Persist any actionable boundaries this run surfaced (capture path); a
        // no-op for discovery runs.
        self.persist_correction_boundaries(&run_id, generation, &outcome)
            .await?;
        let fitness = outcome.final_fitness;

        // score -> decide: graduate the winner or prune + log a boundary.
        self.advance(head, LoopState::Decide).await?;
        // ADR-0022: a pass on an impossible task is proof of a shortcut, and a
        // shortcut makes every other number in the generation unreadable, so
        // the generation fails whole rather than scoring a little lower.
        let shortcut = measured.instruments.as_ref().filter(|m| m.failed());
        if let Some(report) = shortcut {
            eprintln!(
                "instruments: generation {} passed impossible task(s) {:?}; it fails whole and does not graduate",
                generation.0, report.impossible_passed
            );
        }
        let shortcut = shortcut.is_some();
        let (graduation_score, remeasured) = self
            .judge(&run_id, generation, &shadow_id, &outcome, &cohort, shortcut)
            .await?;
        measured.remeasured = remeasured.clone();
        let graduated = graduation_score >= self.cfg.graduate_threshold && !shortcut;
        if graduated {
            sh.advance_to(ShadowStatus::Graduated)?;
            shadow::upsert(self.store, &sh).await?;
            self.graduate(&run_id, generation, &outcome.adapter_uri, fitness, &outcome)
                .await?;
        } else {
            sh.advance_to(ShadowStatus::Pruned)?;
            shadow::upsert(self.store, &sh).await?;
            self.log_open_boundary(generation, &shadow_id).await?;
        }
        self.record_evaluation(
            &run_id, &shadow_id, generation, graduated, &outcome, &measured,
        )
        .await?;

        // decide -> consolidate -> grow (the last step bumps the generation).
        self.advance(head, LoopState::Consolidate).await?;
        self.advance(head, LoopState::Grow).await?;

        // The population just changed; verify every frozen expert is still
        // byte-identical to its freeze baseline (the no-forgetting tripwire).
        let regressions = self.check_no_forgetting(&run_id).await?;

        Ok(GenerationReport {
            generation,
            shadow: shadow_id,
            fitness,
            graduated,
            reward_curve: outcome.reward_curve.clone(),
            regressions,
            instruments: measured.instruments,
            trend: measured.trend,
            recipe,
            cohort,
            graduation_score,
            remeasured,
        })
    }

    /// Run generations until the head reaches `target_generation`.
    pub async fn run_until(
        &self,
        run_id: &RunId,
        target_generation: u32,
    ) -> Result<Vec<GenerationReport>> {
        let mut head = self.resume_or_init(run_id).await?;
        self.recover(&mut head).await?;
        let mut reports = Vec::new();
        while head.generation.0 < target_generation {
            // Cooperative stop: an operator can halt the run between
            // generations. Default (no control) is Run, so this is a no-op then.
            if loop_control::load(self.store, run_id).await? == LoopCommand::Halt {
                self.advance(&mut head, LoopState::Paused).await?;
                loop_control::clear(self.store, run_id).await?;
                break;
            }
            reports.push(self.run_generation(&mut head).await?);
        }
        Ok(reports)
    }

    /// Bring a head the process left anywhere but a generation boundary back to
    /// one, so the run can go on.
    ///
    /// The loop checkpoints where it is, not the work in flight: a shadow
    /// half-trained when the process died has nothing to resume from. So a
    /// generation interrupted before its decision was written (in `Explore`,
    /// `Score` or `Decide`) is run again from its start, through the same
    /// `Paused -> Grow` edge an operator's halt takes, and does not cross the
    /// generation boundary. What it wrote on the way is keyed so the second
    /// run replaces the first: shadows and recipe rows by shadow, the
    /// graduated expert by generation, and the audit trend keeps a
    /// generation's latest evaluation. One interrupted in `Consolidate` had
    /// written everything, so it is finished rather than repeated. Before this,
    /// a run killed mid-training (an out-of-memory on the GPU, a reboot) could
    /// never be resumed: every generation opens by moving to `Explore`, which
    /// a head already there refuses.
    async fn recover(&self, head: &mut GenerationHead) -> Result<()> {
        match head.state {
            LoopState::Grow => Ok(()),
            LoopState::Paused => self.advance(head, LoopState::Grow).await,
            LoopState::Consolidate => self.advance(head, LoopState::Grow).await,
            LoopState::Explore | LoopState::Score | LoopState::Decide => {
                eprintln!(
                    "loop: generation {} was interrupted in {:?}; running it again from its start",
                    head.generation.0, head.state
                );
                self.advance(head, LoopState::Paused).await?;
                self.advance(head, LoopState::Grow).await
            }
        }
    }

    /// Persist the training reward curve as source-tagged signals for the critic.
    async fn record_rewards(&self, run_id: &RunId, outcome: &TrainOutcome) -> Result<()> {
        let now = Utc::now();
        let signals: Vec<RewardSignal> = outcome
            .reward_curve
            .iter()
            .enumerate()
            .map(|(i, &v)| RewardSignal::verifier(run_id.clone(), i as u32, "fitness", v, now))
            .collect();
        reward::insert_many(self.store, &signals).await
    }

    /// One measured run per generation, recorded to the unified substrate.
    async fn record_evaluation(
        &self,
        run_id: &RunId,
        shadow_id: &ShadowId,
        generation: Generation,
        graduated: bool,
        outcome: &TrainOutcome,
        measured: &Measurement,
    ) -> Result<()> {
        let eval = EvaluationRun {
            run_id: run_id.clone(),
            subject_kind: SubjectKind::Shadow,
            subject_id: shadow_id.to_string(),
            corpus_task_id: format!("gen:{}", generation.0),
            status: if graduated {
                EvalStatus::Success
            } else {
                EvalStatus::Failure
            },
            // The instruments ride with the fitness they qualify, so a reader
            // of this row cannot get the score without the measurement of
            // whether the score means anything (ADR-0022). The partition seed
            // goes with them: a reseed repartitions the corpus and invalidates
            // every gap measured before it, so a generation has to say which
            // split it was read under.
            metrics: Some(serde_json::json!({
                "fitness": outcome.final_fitness,
                "partition_seed": self.cfg.partition.map(|p| p.seed),
                "instruments": measured.instruments,
                "trend": measured.trend,
                "remeasured": measured.remeasured,
            })),
            regression_fingerprint: Some(outcome.adapter_uri.clone()),
            created_at: Utc::now(),
        };
        evaluation::insert(self.store, &eval).await
    }

    /// Freeze a graduated shadow into a new expert and add it to the population
    /// (the frozen-expert population). The capability vector is learned from
    /// **evaluated behavior**: the centroid of the embeddings of the tasks the
    /// shadow provably solved, scoped by the counterfactual boundary the gate
    /// routes against, so routing reflects what the expert demonstrably does,
    /// not a hand-written label. Falls back to a generic descriptor only when
    /// the trainer reported no exemplars.
    async fn graduate(
        &self,
        run_id: &RunId,
        generation: Generation,
        adapter_uri: &str,
        fitness: f32,
        outcome: &TrainOutcome,
    ) -> Result<()> {
        let capability_vec = self
            .capability_vector(&outcome.capability_exemplars, generation)
            .await?;
        let now = Utc::now();
        let expert = Expert {
            id: ExpertId::new(format!("expert:{run_id}:g{}", generation.0)),
            name: format!("{run_id}-g{}", generation.0),
            base_model: self.cfg.base_model.clone(),
            artifact_uri: adapter_uri.to_string(),
            capability_card: serde_json::json!({
                "generation": generation.0,
                "exemplars": outcome.capability_exemplars,
            }),
            capability_vec: Some(capability_vec),
            fitness,
            frozen_at: Some(now),
            generation,
            owner: None,
            compartment: None,
            // The weights land on this machine's disk and stay there
            // (ADR-0017 keeps adapters out of sync scope), so the row says
            // which machine that was.
            placed_on: Some(antumbra_core::this_host()),
            created_at: now,
        };
        // A generation run again after an interruption may already have
        // graduated once. Its expert is keyed by generation, so the second
        // decision replaces the first rather than adding a twin to the
        // population.
        if expert::get(self.store, &expert.id).await?.is_some() {
            expert::delete(self.store, &expert.id).await?;
        }
        expert::insert(self.store, &expert).await?;
        // Snapshot the freeze baseline for the no-forgetting tripwire (the frozen
        // population versus the plastic shadow): this fingerprint must never
        // change while the expert is frozen in the population. Stored once, at
        // graduation, and re-checked each later
        // generation by `check_no_forgetting`.
        let baseline = EvaluationRun {
            run_id: run_id.clone(),
            subject_kind: SubjectKind::Expert,
            subject_id: expert.id.to_string(),
            corpus_task_id: format!("freeze:g{}", generation.0),
            status: EvalStatus::Success,
            metrics: Some(serde_json::json!({ "fitness": fitness, "event": "freeze" })),
            regression_fingerprint: Some(fingerprint(adapter_uri)),
            created_at: now,
        };
        evaluation::insert(self.store, &baseline).await?;
        // A new expert may resolve the failure region of an open scope: retire
        // every actionable counterfactual boundary its capability now covers. The
        // gap the boundary marked is filled, so it should stop gating routing.
        self.retire_covered_boundaries(&expert).await?;
        Ok(())
    }

    /// Retire every boundary whose failure region this expert now covers: by its
    /// capability vector it sits closer to the failure context than to C', so the
    /// scope it marked is resolved and must no longer inhibit routing (the
    /// retire-on-correction boundary lifecycle). Returns how many were retired. A
    /// no-op for an expert with no capability vector or against open boundaries
    /// (which lack the
    /// embedded contrastive pair `is_covered_by` needs).
    async fn retire_covered_boundaries(&self, expert: &Expert) -> Result<usize> {
        let Some(cap) = expert.capability_vec.as_deref() else {
            return Ok(0);
        };
        let mut retired = 0;
        for b in boundary::list(self.store).await? {
            if b.is_covered_by(cap) {
                boundary::delete(self.store, &b.id).await?;
                retired += 1;
            }
        }
        Ok(retired)
    }

    /// The no-forgetting tripwire (frozen population versus plastic shadow):
    /// re-fingerprint every frozen expert and compare to its freeze baseline. A
    /// mismatch means a frozen
    /// expert's weights changed under it (the kill criterion firing), and the
    /// expert's id is returned (and logged). Experts with no baseline (frozen
    /// before this check existed) are skipped. Logged, not fatal: a drift is a
    /// loud alarm, but halting every other expert's progress on it is the
    /// operator's call, not the loop's. Public so an operator can run the audit on
    /// demand, not only as part of a generation.
    pub async fn check_no_forgetting(&self, run_id: &RunId) -> Result<Vec<ExpertId>> {
        let mut regressions = Vec::new();
        for frozen in expert::list(self.store).await? {
            let id = frozen.id.to_string();
            let Some(baseline) =
                evaluation::latest_for_subject(self.store, SubjectKind::Expert, &id).await?
            else {
                continue;
            };
            let current = EvaluationRun {
                run_id: run_id.clone(),
                subject_kind: SubjectKind::Expert,
                subject_id: id,
                corpus_task_id: baseline.corpus_task_id.clone(),
                status: EvalStatus::Success,
                metrics: None,
                regression_fingerprint: Some(fingerprint(&frozen.artifact_uri)),
                created_at: Utc::now(),
            };
            if !baseline.fingerprint_matches(&current) {
                eprintln!(
                    "no-forgetting KILL CRITERION: frozen expert {} drifted (baseline {:?} != now {:?})",
                    frozen.id, baseline.regression_fingerprint, current.regression_fingerprint
                );
                regressions.push(frozen.id);
            }
        }
        Ok(regressions)
    }

    /// The capability vector: the mean of the embeddings of solved-task prompts
    /// (cosine, used downstream by the gate, is scale-invariant so the centroid
    /// need not be renormalized). With no exemplars, embed a generic descriptor.
    async fn capability_vector(
        &self,
        exemplars: &[String],
        generation: Generation,
    ) -> Result<Vec<f32>> {
        if exemplars.is_empty() {
            let descriptor = format!("specialist for generation {}", generation.0);
            return self.embedder.embed(&descriptor).await;
        }
        let mut centroid = vec![0.0f32; self.embedder.dim()];
        for prompt in exemplars {
            let v = self.embedder.embed(prompt).await?;
            // A dimension drift (e.g. the embedder model changed mid-run) would
            // silently truncate the centroid via zip; refuse it instead of storing
            // a corrupted capability vector the gate would then route against.
            if v.len() != centroid.len() {
                return Err(antumbra_core::AntumbraError::other(format!(
                    "embedder returned dim {} but the capability centroid is {}",
                    v.len(),
                    centroid.len()
                )));
            }
            for (c, x) in centroid.iter_mut().zip(v.iter()) {
                *c += *x;
            }
        }
        let n = exemplars.len() as f32;
        centroid.iter_mut().for_each(|c| *c /= n);
        Ok(centroid)
    }

    /// Persist any **actionable** counterfactual boundaries a capture run surfaced:
    /// each is a verified correction's contrastive pair (C incorrect / C'
    /// acceptable). The loop owns the embedder, so it renders each context into
    /// the same space task vectors live in and stores both embeddings, letting
    /// the relative-margin inhibition fire only inside the failure scope. Unlike
    /// `log_open_boundary`, these gate routing -- a C' was recovered. No-op for
    /// discovery (RAFT) runs, whose findings list is empty.
    async fn persist_correction_boundaries(
        &self,
        run_id: &RunId,
        generation: Generation,
        outcome: &TrainOutcome,
    ) -> Result<()> {
        let now = Utc::now();
        for (i, finding) in outcome.boundary_findings.iter().enumerate() {
            let fail_vec = self
                .embedder
                .embed(&render_scope(&finding.behavior, &finding.fail_context))
                .await?;
            let ok_vec = self
                .embedder
                .embed(&render_scope(&finding.behavior, &finding.near_ok_context))
                .await?;
            let boundary = finding_to_boundary(
                BoundaryId::new(format!("boundary:{run_id}:g{}:{i}", generation.0)),
                finding,
                // A correction names where it applies, not how wide; Project is
                // the conservative default grain until the scope carries one.
                Grain::Project,
                // Ground-truth-verified, so the scope is trustworthy enough to
                // gate; confidence still tempers the inhibition magnitude.
                CORRECTION_BOUNDARY_CONFIDENCE,
                Some(fail_vec),
                Some(ok_vec),
                generation,
                now,
            );
            boundary::upsert(self.store, &boundary).await?;
        }
        Ok(())
    }

    /// On prune, log an **open-negative** counterfactual boundary: the failure is
    /// recorded, but it is *not actionable* until counterfactual search
    /// recovers a C', which needs the (still-stubbed) serving probe. Storing
    /// it open: an un-scoped negative must never gate routing.
    async fn log_open_boundary(&self, generation: Generation, shadow_id: &ShadowId) -> Result<()> {
        let b = FailureBoundary {
            id: BoundaryId::new(format!("boundary:{shadow_id}")),
            behavior: format!("approach of {shadow_id}"),
            fail_context: serde_json::json!({ "generation": generation.0 }),
            near_ok_context: None,
            governing_features: Vec::new(),
            grain: None,
            context_vec: None,
            ok_context_vec: None,
            confidence: 0.3,
            generation,
            created_at: Utc::now(),
        };
        boundary::upsert(self.store, &b).await
    }

    /// Apply a guarded state transition and persist the head (the checkpoint).
    async fn advance(&self, head: &mut GenerationHead, to: LoopState) -> Result<()> {
        head.advance_to(to, Utc::now())?;
        generation::save_head(self.store, head).await
    }
}
