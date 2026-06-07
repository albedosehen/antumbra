//! Port traits — the seams where Antumbra meets the GPU and the outside world.
//!
//! The domain core (loop, gate, boundary, critic aggregation) is written
//! entirely against these traits, so it is exercisable with in-memory fakes
//! (see [`crate::testing`]) and the heavy implementations — candle QLoRA
//! training (ADR-0002), llama.cpp / mistral.rs serving (ADR-0006) — drop in
//! later as the only changed pieces.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::boundary::BoundaryFinding;
use crate::error::Result;
use crate::ids::{ExpertId, RunId, ShadowId};

// --- serving (ADR-0006) ---------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActRequest {
    pub task_id: String,
    pub prompt: String,
    /// The adapters the gate selected to blend in latent space (ADR-0005).
    #[serde(default)]
    pub adapters: Vec<ExpertId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepOutput {
    pub step_idx: u32,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActOutput {
    pub steps: Vec<StepOutput>,
    pub final_output: String,
}

/// Runs the base + selected adapters to produce output for a task.
#[async_trait]
pub trait Serve: Send + Sync {
    async fn act(&self, req: ActRequest) -> Result<ActOutput>;

    /// Whether this engine can serve `expert` right now (its adapter is resident /
    /// registered). The `answer` tool checks this so a routed-but-unservable expert
    /// — e.g. a private expert minted after the engine snapshotted its population —
    /// escalates cleanly instead of surfacing a "no adapter registered" error.
    /// Defaults to `true` for engines that pin a single adapter or echo any input.
    fn can_serve(&self, _expert: &ExpertId) -> bool {
        true
    }
}

/// The optional flagship escalation tier (ADR-0005): consulted only when the
/// gate says out-of-scope / low-confidence, and shrinks over time.
#[async_trait]
pub trait FlagshipTier: Send + Sync {
    async fn escalate(&self, req: ActRequest) -> Result<ActOutput>;
}

// --- training (ADR-0002) --------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainRequest {
    pub shadow: ShadowId,
    pub base_model: String,
    pub corpus_task_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainOutcome {
    pub adapter_uri: String,
    /// Per-step fitness; the anti-collapse guard reads this curve.
    pub reward_curve: Vec<f32>,
    pub final_fitness: f32,
    /// Prompts the shadow provably solved (verified-correct) by the final round.
    /// The expert's capability vector is learned from these evaluated behaviors
    /// rather than a hand-written description (ADR-0004/0005).
    pub capability_exemplars: Vec<String>,
    /// Actionable boundary findings this run produced: each is a verified
    /// contrastive context pair (C incorrect / C' acceptable) the loop embeds
    /// and persists as a scope that gates routing (ADR-0004). The capture path
    /// surfaces these from corrections; discovery-only runs leave it empty.
    #[serde(default)]
    pub boundary_findings: Vec<BoundaryFinding>,
}

/// Trains a shadow adapter on verified outcomes. The heaviest real component;
/// in v0 this is a DIY candle QLoRA path, stubbed behind this trait until built.
#[async_trait]
pub trait Trainer: Send + Sync {
    async fn train_shadow(&self, req: TrainRequest) -> Result<TrainOutcome>;
}

// --- reward (ADR-0003) ----------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyRequest {
    pub run_id: RunId,
    pub step_idx: u32,
    pub dimension: String,
    /// Whatever the verifier inspects: a command result, a JSON body, a diff.
    pub artifact: serde_json::Value,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct VerifierVerdict {
    pub passed: bool,
    pub value: f32,
}

/// A primary, trusted verifier (tests / schema / exec). Ground truth.
#[async_trait]
pub trait Verifier: Send + Sync {
    async fn verify(&self, req: &VerifyRequest) -> Result<VerifierVerdict>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticScore {
    pub step_idx: u32,
    pub dimension: String,
    pub value: f32,
}

/// A PRM-style densifier that interpolates per-step credit between verifier
/// checkpoints. Never authoritative (ADR-0003).
#[async_trait]
pub trait Critic: Send + Sync {
    async fn densify(&self, output: &ActOutput) -> Result<Vec<CriticScore>>;
}

// --- embedding + boundary probe (ADR-0004/0005) ---------------------------

/// Produces capability / context vectors for routing-as-retrieval and boundary
/// lookup.
#[async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
    fn dim(&self) -> usize;
}

/// Replays the frozen population to judge whether a behavior is acceptable in a
/// given context. This is what makes counterfactual search affordable
/// (ADR-0004): cheap, repeatable re-probing over frozen experts.
#[async_trait]
pub trait AcceptabilityProbe: Send + Sync {
    async fn acceptable(&self, behavior: &str, context: &serde_json::Value) -> Result<bool>;
}
