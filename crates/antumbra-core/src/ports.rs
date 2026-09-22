//! Port traits: the seams where Antumbra meets the GPU and the outside world.
//!
//! The domain core (loop, gate, boundary, critic aggregation) is written
//! entirely against these traits, so it is exercisable with in-memory fakes
//! (see [`crate::testing`]) and the heavy implementations (candle QLoRA
//! training, llama.cpp / mistral.rs hardware-adaptive serving) drop in
//! later as the only changed pieces.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::boundary::BoundaryFinding;
use crate::error::Result;
use crate::ids::{ExpertId, RunId, ShadowId};

// --- serving (hardware-adaptive) ------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActRequest {
    pub task_id: String,
    pub prompt: String,
    /// The adapters the gate selected to blend in latent space.
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
    /// (e.g. a private expert minted after the engine snapshotted its population)
    /// escalates cleanly instead of surfacing a "no adapter registered" error.
    /// Defaults to `true` for engines that pin a single adapter or echo any input.
    fn can_serve(&self, _expert: &ExpertId) -> bool {
        true
    }

    /// Register (or replace) an expert's adapter in the live engine, so a route
    /// to a freshly-minted expert (autonomous consolidation, training) serves
    /// without a restart. Default no-op for engines that pin a single adapter or
    /// snapshot their population at build time.
    fn register_expert(&self, _expert: &ExpertId, _adapter_uri: &str) {}
}

/// The optional flagship escalation tier: consulted only when the
/// gate says out-of-scope / low-confidence, and shrinks over time.
#[async_trait]
pub trait FlagshipTier: Send + Sync {
    async fn escalate(&self, req: ActRequest) -> Result<ActOutput>;
}

// --- training -------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainRequest {
    pub shadow: ShadowId,
    pub base_model: String,
    pub corpus_task_ids: Vec<String>,
}

/// One corpus task, as the final training round found it.
///
/// The loop needs this to say anything honest about a generation. Aggregate
/// fitness cannot be sliced -- a visible-minus-held-out gap computed from one
/// number is not a measurement of anything (ADR-0022) -- and the trainer knows
/// the per-task answer already, because it is what it averages to get fitness.
/// It simply used to throw it away.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskOutcome {
    pub task_id: String,
    pub passed: bool,
    /// What the corpus counts as the size of this task. The instruments never
    /// interpret it beyond ordering, so any consistent measure will do; the
    /// trainer supplies prompt length, which is a proxy rather than a claim.
    pub size: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainOutcome {
    pub adapter_uri: String,
    /// Per-step fitness; the anti-collapse guard reads this curve.
    pub reward_curve: Vec<f32>,
    pub final_fitness: f32,
    /// Prompts the shadow provably solved (verified-correct) by the final round.
    /// The expert's capability vector is learned from these evaluated behaviors
    /// rather than a hand-written description.
    pub capability_exemplars: Vec<String>,
    /// Actionable boundary findings this run produced: each is a verified
    /// contrastive context pair (C incorrect / C' acceptable) the loop embeds
    /// and persists as a scope that gates routing. The capture path
    /// surfaces these from corrections; discovery-only runs leave it empty.
    #[serde(default)]
    pub boundary_findings: Vec<BoundaryFinding>,
    /// Per-task results from the final round, in corpus order. The same round
    /// `capability_exemplars` reflects, so the two describe one adapter.
    #[serde(default)]
    pub per_task: Vec<TaskOutcome>,
}

/// Trains a shadow adapter on verified outcomes. The heaviest real component;
/// in v0 this is a DIY candle QLoRA path, stubbed behind this trait until built.
#[async_trait]
pub trait Trainer: Send + Sync {
    async fn train_shadow(&self, req: TrainRequest) -> Result<TrainOutcome>;
}

// --- reward ----------------------------------------------------------------

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
/// checkpoints. Never authoritative.
#[async_trait]
pub trait Critic: Send + Sync {
    async fn densify(&self, output: &ActOutput) -> Result<Vec<CriticScore>>;
}

// --- embedding + boundary probe -------------------------------------------

/// Produces capability / context vectors for routing-as-retrieval and boundary
/// lookup.
#[async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
    fn dim(&self) -> usize;
}

/// Re-scores wide hybrid-recall candidates with a general-purpose cross-encoder
/// over (query, candidate-content) pairs — the precision stage downstream of the
/// wide RRF recall (P-2 rerank). A single-vector dense retriever has a
/// dimension-bounded recall ceiling; a cross-encoder reads the query and each
/// candidate *jointly*, so it reorders the recalled pool far more precisely than
/// the bi-encoder scores can.
#[async_trait]
pub trait Reranker: Send + Sync {
    /// Score each `(id, text)` candidate against `query` and return the candidate
    /// ids in descending relevance order. The result is a permutation of the input
    /// ids — callers reorder their already-fetched rows by it, then truncate to
    /// top-k — so an implementation must neither drop nor duplicate an id. Returns
    /// `Err` only on a real transport/parse fault; the caller then degrades to the
    /// pre-rerank order rather than failing recall.
    async fn rerank(&self, query: &str, candidates: &[(String, String)]) -> Result<Vec<String>>;
}

/// Replays the frozen population to judge whether a behavior is acceptable in a
/// given context. This is what makes counterfactual search affordable:
/// cheap, repeatable re-probing over frozen experts.
#[async_trait]
pub trait AcceptabilityProbe: Send + Sync {
    async fn acceptable(&self, behavior: &str, context: &serde_json::Value) -> Result<bool>;
}
