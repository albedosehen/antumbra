//! # antumbra-loop — ADR-0008
//!
//! The durable generational loop as a resumable state-machine-in-DB:
//! `grow -> explore -> score -> decide -> consolidate -> grow`. The head is
//! persisted after every transition, so killing the process and restarting
//! resumes from the last checkpoint — the state value *is* the checkpoint.
//!
//! Each generation now writes its full lineage to the substrate: the shadow
//! and its status transitions (ADR-0002), the per-step reward signals
//! (ADR-0003), an evaluation run (ADR-0007), the graduated expert (ADR-0001),
//! and — on prune — an open-negative failure boundary (ADR-0004; not yet
//! actionable, because the loop has no probe to recover C' until serving is
//! wired). Driven entirely by the injected ports, so it runs with fakes.

use chrono::Utc;

use antumbra_core::generational::{GenerationHead, LoopState};
use antumbra_core::ports::{Embedder, TrainOutcome, TrainRequest, Trainer};
use antumbra_core::{
    BoundaryId, EvalStatus, EvaluationRun, Expert, ExpertId, FailureBoundary, Generation, Result,
    RewardSignal, RunId, Shadow, ShadowId, ShadowStatus, SubjectKind,
};
use antumbra_store::repo::{boundary, evaluation, expert, generation, reward, shadow};
use antumbra_store::Store;

#[derive(Debug, Clone)]
pub struct LoopConfig {
    /// Fitness at or above which a shadow graduates into a frozen expert.
    pub graduate_threshold: f32,
    /// The shared base every adapter rides on (ADR-0001).
    pub base_model: String,
    /// Cap on shadow training steps per generation.
    pub max_steps: u32,
}

impl Default for LoopConfig {
    fn default() -> Self {
        Self {
            graduate_threshold: 0.5,
            base_model: "code-base".into(),
            max_steps: 8,
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
    /// Per-round pass-rate (RAFT reward curve) — rising means the adapter is
    /// learning to satisfy the verifier.
    pub reward_curve: Vec<f32>,
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
        let shadow_id = ShadowId::new(format!("{run_id}:g{}", generation.0));

        // grow -> explore: spawn the shadow (ADR-0002).
        let mut sh = Shadow::spawn(shadow_id.clone(), generation, None, Utc::now());
        shadow::upsert(self.store, &sh).await?;
        self.advance(head, LoopState::Explore).await?;
        sh.advance_to(ShadowStatus::Exploring)?;
        shadow::upsert(self.store, &sh).await?;

        // train on verified outcomes.
        let outcome = self
            .trainer
            .train_shadow(TrainRequest {
                shadow: shadow_id.clone(),
                base_model: self.cfg.base_model.clone(),
                corpus_task_ids: Vec::new(),
                max_steps: self.cfg.max_steps,
            })
            .await?;
        sh.adapter_uri = Some(outcome.adapter_uri.clone());
        sh.reward_curve = outcome.reward_curve.clone();

        // explore -> score: record the per-step reward (ADR-0003).
        self.advance(head, LoopState::Score).await?;
        sh.advance_to(ShadowStatus::Scoring)?;
        shadow::upsert(self.store, &sh).await?;
        self.record_rewards(&run_id, &outcome).await?;
        let fitness = outcome.final_fitness;

        // score -> decide: graduate the winner or prune + log a boundary.
        self.advance(head, LoopState::Decide).await?;
        let graduated = fitness >= self.cfg.graduate_threshold;
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
            &run_id, &shadow_id, generation, fitness, graduated, &outcome,
        )
        .await?;

        // decide -> consolidate -> grow (the last step bumps the generation).
        self.advance(head, LoopState::Consolidate).await?;
        self.advance(head, LoopState::Grow).await?;

        Ok(GenerationReport {
            generation,
            shadow: shadow_id,
            fitness,
            graduated,
            reward_curve: outcome.reward_curve.clone(),
        })
    }

    /// Run generations until the head reaches `target_generation`.
    pub async fn run_until(
        &self,
        run_id: &RunId,
        target_generation: u32,
    ) -> Result<Vec<GenerationReport>> {
        let mut head = self.resume_or_init(run_id).await?;
        let mut reports = Vec::new();
        while head.generation.0 < target_generation {
            reports.push(self.run_generation(&mut head).await?);
        }
        Ok(reports)
    }

    /// Persist the training reward curve as source-tagged signals (ADR-0003).
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

    /// One measured run per generation (ADR-0007).
    async fn record_evaluation(
        &self,
        run_id: &RunId,
        shadow_id: &ShadowId,
        generation: Generation,
        fitness: f32,
        graduated: bool,
        outcome: &TrainOutcome,
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
            metrics: Some(serde_json::json!({ "fitness": fitness })),
            regression_fingerprint: Some(outcome.adapter_uri.clone()),
            created_at: Utc::now(),
        };
        evaluation::insert(self.store, &eval).await
    }

    /// Freeze a graduated shadow into a new expert and add it to the population
    /// (ADR-0001). The capability vector is learned from **evaluated behavior**:
    /// the centroid of the embeddings of the tasks the shadow provably solved
    /// (ADR-0004/0005), so routing reflects what the expert demonstrably does —
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
            created_at: now,
        };
        expert::insert(self.store, &expert).await
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
            for (c, x) in centroid.iter_mut().zip(v.iter()) {
                *c += *x;
            }
        }
        let n = exemplars.len() as f32;
        centroid.iter_mut().for_each(|c| *c /= n);
        Ok(centroid)
    }

    /// On prune, log an **open-negative** boundary (ADR-0004): the failure is
    /// recorded, but it is *not actionable* until counterfactual search
    /// recovers a C' — which needs the (still-stubbed) serving probe. Storing
    /// it open is honest: an un-scoped negative must never gate routing.
    async fn log_open_boundary(&self, generation: Generation, shadow_id: &ShadowId) -> Result<()> {
        let b = FailureBoundary {
            id: BoundaryId::new(format!("boundary:{shadow_id}")),
            behavior: format!("approach of {shadow_id}"),
            fail_context: serde_json::json!({ "generation": generation.0 }),
            near_ok_context: None,
            governing_features: Vec::new(),
            grain: None,
            context_vec: None,
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
