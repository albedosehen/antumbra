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

/// A question with a known answer space, asked of a [`TypedDecider`] (ADR-0024).
///
/// The point of naming the answer space is that the answer comes back as a
/// calibrated probability rather than as a distance the caller has to interpret.
/// Most of what this system decides is not text — route or escalate, in scope or
/// not, does this memory answer this query — and each has been answered by
/// comparing an uncalibrated scalar against a hand-set threshold.
#[derive(Debug, Clone, PartialEq)]
pub enum Question {
    /// Pick one option, with a distribution over all of them.
    ///
    /// The answer space is bounded deliberately: accuracy collapses on large
    /// label spaces, so ADR-0024 requires every `Choice` in the system to offer
    /// fewer than twenty options and asserts it with a test.
    Choice { options: Vec<String> },
    /// An expectation on an ordinal scale, for "how much" rather than "which".
    Score { low: f32, high: f32 },
    /// Is this statement true, as a calibrated probability. The primitive the
    /// relevance floor needs (ADR-0023 B-2): "does this memory answer this
    /// query" is a `Noul`, and the floor reads its probability.
    Noul,
}

/// The most options a [`Question::Choice`] may offer.
///
/// ADR-0024 Validation 6 caps this at twenty, on two grounds that agree:
/// options share a fixed token budget, and accuracy on typed decisions degrades
/// sharply past roughly twenty labels — the model card this design follows
/// scores 0.425 on a 77-label benchmark against 0.870 for a system without that
/// weakness. Sixteen leaves headroom under the ceiling rather than sitting on it.
pub const MAX_CHOICE_OPTIONS: usize = 16;

// The record's ceiling, checked at compile time rather than by a test: raising
// the constant past twenty should fail the build, not a test run.
const _: () = assert!(MAX_CHOICE_OPTIONS < 20);

impl Question {
    /// Whether this question can be answered well, as opposed to merely answered.
    ///
    /// A caller checks this before spending a forward pass. Refusing here is the
    /// point: a `Choice` over fifty labels returns a confident number that means
    /// nothing, and that is worse than no answer, because the caller's next move
    /// is a threshold.
    pub fn is_answerable(&self) -> bool {
        match self {
            // Two or more, and not past the bound: one option is not a decision,
            // and none is a caller error.
            Question::Choice { options } => (2..=MAX_CHOICE_OPTIONS).contains(&options.len()),
            // A scale needs width, or the expectation it asks for is a constant.
            Question::Score { low, high } => high > low,
            Question::Noul => true,
        }
    }
}

/// What a [`TypedDecider`] answers.
///
/// Every variant carries a probability rather than a score, because the caller's
/// next move is a threshold and a threshold on an uncalibrated number is the
/// defect ADR-0024 exists to remove.
#[derive(Debug, Clone, PartialEq)]
pub enum Answer {
    /// The chosen index into the question's options, and the full distribution.
    /// The distribution is the part that matters: it is what distinguishes
    /// "ambiguous between two good options" from "none of these fit", which one
    /// scalar margin conflates and no threshold over it can separate.
    Choice { index: usize, probs: Vec<f32> },
    /// The expectation on the requested scale.
    Score { expected: f32 },
    /// The probability that the statement is true.
    Noul { probability: f32 },
}

/// Answers questions with known answer spaces, as calibrated probabilities
/// (ADR-0024).
///
/// The contract that makes this worth having is the training objective rather
/// than the interface: an implementation must be trained against a strictly
/// proper scoring rule over outcomes a VERIFIER produced, so that reporting an
/// honest probability is the only way to score well. A head trained on its own
/// past answers, or on a critic's, violates ADR-0022's anchor invariant and is
/// not an admissible implementation of this port however well it performs.
///
/// Answering a batch in one call is deliberate: the questions about one state
/// share an encoding, so asking them together is what makes this cheap enough to
/// sit in a serving path at all.
#[async_trait]
pub trait TypedDecider: Send + Sync {
    /// Answer every question about `state`, in order. The returned vector is the
    /// same length as `questions`, and each answer's variant matches its
    /// question's.
    async fn decide(&self, state: &str, questions: &[Question]) -> Result<Vec<Answer>>;
}

/// Replays the frozen population to judge whether a behavior is acceptable in a
/// given context. This is what makes counterfactual search affordable:
/// cheap, repeatable re-probing over frozen experts.
#[async_trait]
pub trait AcceptabilityProbe: Send + Sync {
    async fn acceptable(&self, behavior: &str, context: &serde_json::Value) -> Result<bool>;
}

#[cfg(test)]
mod typed_decisions {
    use super::*;

    /// ADR-0024 Validation 6: every `Choice` in the system offers fewer than
    /// twenty options, because accuracy collapses on large label spaces and the
    /// options share a fixed token budget.
    ///
    /// The bound is asserted here, on the type, rather than left to each caller
    /// to remember. A question that cannot be answered well is not worth asking
    /// cheaply.
    #[test]
    fn a_choice_answer_space_is_bounded() {
        let ok = Question::Choice {
            options: (0..MAX_CHOICE_OPTIONS).map(|i| i.to_string()).collect(),
        };
        assert!(ok.is_answerable(), "a question at the bound is answerable");
        let too_many = Question::Choice {
            options: (0..MAX_CHOICE_OPTIONS + 1).map(|i| i.to_string()).collect(),
        };
        assert!(
            !too_many.is_answerable(),
            "past the bound the question must be refused rather than answered badly"
        );
    }

    /// An empty or single-option choice is not a decision, and asking it wastes a
    /// forward pass on an answer the caller already has.
    #[test]
    fn a_choice_with_nothing_to_choose_is_not_answerable() {
        assert!(!Question::Choice { options: vec![] }.is_answerable());
        assert!(!Question::Choice {
            options: vec!["only".into()]
        }
        .is_answerable());
    }

    /// A scale has to have width, or the expectation it asks for is a constant.
    #[test]
    fn a_score_needs_a_real_scale() {
        assert!(Question::Score {
            low: 0.0,
            high: 1.0
        }
        .is_answerable());
        assert!(!Question::Score {
            low: 1.0,
            high: 1.0
        }
        .is_answerable());
        assert!(
            !Question::Score {
                low: 1.0,
                high: 0.0
            }
            .is_answerable(),
            "an inverted scale is a caller error, not a question"
        );
    }

    /// A `Noul` is always answerable: its answer space is fixed.
    #[test]
    fn a_noul_is_always_answerable() {
        assert!(Question::Noul.is_answerable());
    }
}
