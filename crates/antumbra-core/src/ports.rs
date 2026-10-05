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
use crate::error::{AntumbraError, Result};
use crate::ids::{ExpertId, RunId, ShadowId, VerifierId};
use crate::recipe::TrainingRecipe;
use crate::router::LearnedRouter;
use crate::slice::Holdout;

// --- serving (hardware-adaptive) ------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActRequest {
    pub task_id: String,
    pub prompt: String,
    /// The adapters the gate selected to blend in latent space.
    #[serde(default)]
    pub adapters: Vec<ExpertId>,
    /// Each adapter's weight in the blend, by position; an adapter without one
    /// weighs 1.0.
    #[serde(default)]
    pub weights: Vec<f32>,
}

impl ActRequest {
    /// A request served by `adapters`, each at full weight.
    pub fn new(
        task_id: impl Into<String>,
        prompt: impl Into<String>,
        adapters: Vec<ExpertId>,
    ) -> Self {
        Self {
            task_id: task_id.into(),
            prompt: prompt.into(),
            adapters,
            weights: Vec::new(),
        }
    }

    /// The adapters with their weights.
    pub fn blend(&self) -> Vec<(ExpertId, f32)> {
        self.adapters
            .iter()
            .enumerate()
            .map(|(i, e)| (e.clone(), self.weights.get(i).copied().unwrap_or(1.0)))
            .collect()
    }
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
    /// What this run must withhold from learning (ADR-0022). `None` learns from
    /// every task and measures nothing apart, which is the right shape for a
    /// run with nothing held out: its generation carries no instruments.
    #[serde(default)]
    pub holdout: Option<Holdout>,
    /// The recipe to train under (ADR-0022 S-1). `None` trains under the
    /// trainer's own configuration, which is what every run did before the
    /// recipe was searched.
    #[serde(default)]
    pub recipe: Option<TrainingRecipe>,
    /// The grow step's choice of what to learn (ADR-0022 S-3). When not empty,
    /// the shadow learns only from these tasks, among those its holdout lets
    /// it learn from. Withheld tasks are measured as always, so a focused run
    /// carries the same instruments as a full one. Empty learns from every
    /// visible task, as every run did before.
    #[serde(default)]
    pub focus: Vec<String>,
    /// The adapter the shadow starts from instead of fresh factors: the grow
    /// step's warm start (ADR-0022 S-3), the expert that serves the region it
    /// chose. `None` starts from the trainer's own configuration.
    #[serde(default)]
    pub parent_adapter: Option<String>,
    /// Train on the verifier's reward alone, whatever critic the trainer
    /// holds: the run has set its critic aside (ADR-0022 S-2's standing
    /// fallback).
    #[serde(default)]
    pub verifier_only: bool,
}

/// One corpus task, as the final training round found it.
///
/// The loop needs this to say anything true about a generation. Aggregate
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
    /// The corpus marked this task unsatisfiable. The partition never assigns
    /// that slice -- such a task is authored, not drawn -- so it has to travel
    /// with the result for the instruments to see it.
    #[serde(default)]
    pub impossible: bool,
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
    /// `capability_exemplars` reflects, so the two describe one adapter. Under
    /// a holdout this includes the tasks measured without being learned from.
    #[serde(default)]
    pub per_task: Vec<TaskOutcome>,
    /// The holdout this run actually enforced, echoed back. The loop measures a
    /// generation only when this matches what it asked for, so a trainer that
    /// ignores the request (an older one, or one that cannot split its corpus)
    /// produces an unmeasured generation rather than a gap over tasks it
    /// learned from.
    #[serde(default)]
    pub holdout: Option<Holdout>,
    /// The recipe this run actually trained under, echoed back whether or not
    /// one was asked for. The loop records a recipe row only from this echo,
    /// so a row never names settings a run did not use.
    #[serde(default)]
    pub recipe: Option<TrainingRecipe>,
    /// The named verifiers whose passes this run trained on, with how many
    /// (ADR-0022 S-4). A verifier later quarantined can then be traced to
    /// what it taught.
    #[serde(default)]
    pub granted_by: Vec<crate::verifier::VerifierGrant>,
    /// Every answer a named verifier judged on a task this run learned from,
    /// passed or not, so the loop can recheck the verdicts against anchored
    /// truth (ADR-0022 S-4).
    #[serde(default)]
    pub judged: Vec<crate::verifier::JudgedSample>,
    /// How the critic that shaped this run read against the verifier, and
    /// against its twin, on the answers it scored (ADR-0022 S-2). `None`
    /// without a critic.
    #[serde(default)]
    pub critic_watch: Option<crate::critic::CriticWatch>,
}

/// Measure a trained shadow again for graduation (ADR-0022 S-1): the
/// fitness a search ranked by is a noisy estimate chosen for being high, and
/// the record's fourth constraint puts the graduation threshold on a fresh
/// measurement instead: a fresh slice, a new seed, at least three repeats.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RemeasureRequest {
    pub shadow: ShadowId,
    pub base_model: String,
    pub adapter_uri: String,
    /// The holdout the shadow's trainer confirmed it enforced. Its held-out
    /// slice is the fresh one: frozen away from training and search, and
    /// reserved for gating graduation. `None` re-draws the tasks the shadow
    /// trained on, which answers the noise but not generalization.
    #[serde(default)]
    pub holdout: Option<Holdout>,
    /// One full evaluation per seed, each drawing from its own.
    pub seeds: Vec<u64>,
}

/// What a re-measurement found.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Remeasurement {
    /// Pass rate per seed, in seed order.
    pub pass_rates: Vec<f32>,
    /// Whether the tasks were the held-out slice, never trained on, rather
    /// than the tasks the shadow trained on.
    pub held_out: bool,
    /// How many tasks each evaluation covered.
    pub tasks: usize,
}

impl Remeasurement {
    /// The mean pass rate over the seeds, which the graduation threshold is
    /// applied to. `None` with no evaluations.
    pub fn mean(&self) -> Option<f32> {
        (!self.pass_rates.is_empty())
            .then(|| self.pass_rates.iter().sum::<f32>() / self.pass_rates.len() as f32)
    }
}

/// A task the population is judged on, as routing needs it: an id and the
/// prompt a router embeds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskPrompt {
    pub id: String,
    pub prompt: String,
    /// The region the task belongs to: its skill, the unit the grow step
    /// chooses among (ADR-0022 S-3).
    #[serde(default)]
    pub region: String,
}

/// Score the base model under one adapter, or alone, on named tasks: the
/// measurement a leave-one-out contribution is made of (ADR-0022 S-5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluateRequest {
    /// Names the evaluation, for the verifier's scratch space and the logs.
    pub label: String,
    pub base_model: String,
    /// The adapter to evaluate under. `None` scores the base model alone,
    /// which is what a task no expert covers falls back to.
    #[serde(default)]
    pub adapter_uri: Option<String>,
    pub task_ids: Vec<String>,
    /// One evaluation per seed. The same seeds on both sides of a comparison
    /// make it a paired one.
    pub seeds: Vec<u64>,
}

/// Each task's pass rate, averaged over the seeds, by task id. A task the
/// trainer does not know is absent rather than scored zero.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TaskScores {
    pub scores: std::collections::BTreeMap<String, f32>,
}

/// Merge two adapters into one of the same rank (ADR-0022 S-5). With `out` of
/// `None` nothing is written, and the outcome alone is the test of whether the
/// two are siblings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MergeRequest {
    pub left: String,
    pub right: String,
    #[serde(default)]
    pub out: Option<String>,
}

/// What a merge kept: the share of the averaged delta's energy that fits in
/// the population's rank, which is how far the two adapters' subspaces
/// overlap.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MergeOutcome {
    pub rank: usize,
    pub retained: f32,
}

/// Trains a shadow adapter on verified outcomes. The heaviest real component;
/// in v0 this is a DIY candle QLoRA path, stubbed behind this trait until built.
#[async_trait]
pub trait Trainer: Send + Sync {
    async fn train_shadow(&self, req: TrainRequest) -> Result<TrainOutcome>;

    /// Re-measure a trained shadow's adapter: the number graduation is judged
    /// on when the loop asks for it. The default refuses, so a trainer that
    /// cannot re-measure is never taken to have done it.
    async fn remeasure(&self, req: RemeasureRequest) -> Result<Remeasurement> {
        let _ = req;
        Err(AntumbraError::Unimplemented("re-measurement"))
    }

    /// The tasks a population's contribution is measured on: the visible
    /// slice under `holdout`, impossible tasks left out. Never the held-out or
    /// audit slice, which no selection may touch, and demotion is one. The
    /// default refuses.
    async fn live_tasks(&self, holdout: Option<Holdout>) -> Result<Vec<TaskPrompt>> {
        let _ = holdout;
        Err(AntumbraError::Unimplemented("live tasks"))
    }

    /// Score an adapter, or the base model alone, on named tasks. The default
    /// refuses, so a trainer that cannot evaluate is never taken to have.
    async fn evaluate(&self, req: EvaluateRequest) -> Result<TaskScores> {
        let _ = req;
        Err(AntumbraError::Unimplemented("task evaluation"))
    }

    /// Merge two adapters at the population's rank. The default refuses.
    async fn merge(&self, req: MergeRequest) -> Result<MergeOutcome> {
        let _ = req;
        Err(AntumbraError::Unimplemented("adapter merging"))
    }

    /// Train the learned router over embedded capability exemplars, each
    /// labeled with the expert it describes. The default refuses, so a
    /// trainer that cannot train one leaves the gate as it is.
    async fn train_router(&self, exemplars: &[(ExpertId, Vec<f32>)]) -> Result<LearnedRouter> {
        let _ = exemplars;
        Err(AntumbraError::Unimplemented("router training"))
    }

    /// The right answer a task's corpus carries for it, when it carries one.
    /// The loop's recheck takes it as a known-good case for a synthesized
    /// verifier, labeled by the task's anchor like any other answer, so a
    /// verifier the policy has not yet answered right can still be judged
    /// (ADR-0022 S-4). The default has none.
    async fn reference(&self, task_id: &str) -> Option<String> {
        let _ = task_id;
        None
    }
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

/// The verifier namespace as the reward path sees it (ADR-0022 S-4): which
/// named verifiers may grant reward now. Read-only, so nothing that trains
/// can write a verifier or change one's trust through it.
#[async_trait]
pub trait TrustedVerifiers: Send + Sync {
    /// The spec verifier `id` checks `task` with, if it may grant reward for
    /// it now: it applies to the task, it is trusted and intact, and, when
    /// synthesized, it is inside its time to live.
    async fn trusted_spec(&self, id: &VerifierId, task: &str) -> Result<Option<serde_json::Value>>;
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

    /// One completion of `prompt`, scored: by default its steps densified,
    /// read by the weakest. A critic that reads the task itself overrides it.
    /// `None` when nothing was scored.
    async fn score(&self, prompt: &str, completion: &str) -> Result<Option<f32>> {
        let _ = prompt;
        let trace = ActOutput {
            steps: vec![StepOutput {
                step_idx: 0,
                content: completion.to_string(),
            }],
            final_output: completion.to_string(),
        };
        Ok(crate::critic::weakest_step(&self.densify(&trace).await?))
    }
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

/// The same signal a [`Reranker`] orders by, as NUMBERS rather than an order.
///
/// Separate from `Reranker` on purpose. That port returns a permutation because
/// ordering is all a precision stage needs, and ADR-0023 B-2 is emphatic that
/// the score is trustworthy WITHIN a query and not across queries — so handing
/// callers a raw magnitude invites exactly the mistake that record warns about.
///
/// A relevance floor needs the magnitude anyway, because "is this good enough"
/// is an across-query comparison by construction. The resolution is not to
/// expose the raw score to callers but to CALIBRATE it: fit
/// [`Platt`](crate::platt::Platt) against verifier-produced labels and let the
/// floor read a probability. This port is the input to that fit and to the
/// decider built on it, which is why it is named for scoring rather than for
/// ranking.
#[async_trait]
pub trait RelevanceScorer: Send + Sync {
    /// Score each text against `query`, in the order given. The result has one
    /// entry per input text, positionally aligned, so a caller can zip it with
    /// its own rows. Higher means more relevant; the scale is the model's own
    /// and means nothing across queries until calibrated.
    async fn relevance(&self, query: &str, texts: &[String]) -> Result<Vec<f32>>;
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
/// proper scoring rule over outcomes a VERIFIER produced, so that reporting the
/// true probability is the only way to score well. A head trained on its own
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

/// The state a relevance floor judges: one query and the memories recalled for
/// it, one `Noul` question each, in order. It travels through
/// [`TypedDecider`]'s one-string state as JSON, so a query or a memory may
/// hold any text: a prompt of many lines, a memory with a markdown rule in it.
/// The recall path encodes it and a floor decodes it, through this one type,
/// so the two cannot drift apart.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelevanceState {
    pub query: String,
    pub memories: Vec<String>,
}

impl RelevanceState {
    pub fn encode(&self) -> String {
        serde_json::to_string(self).expect("two strings and a list of them serialize")
    }

    /// `None` for a state that is not one of these.
    pub fn decode(state: &str) -> Option<Self> {
        serde_json::from_str(state).ok()
    }
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

#[cfg(test)]
mod act_request_tests {
    use super::*;

    #[test]
    fn an_adapter_without_a_weight_weighs_one() {
        let mut req = ActRequest::new("t", "p", vec![ExpertId::new("a"), ExpertId::new("b")]);
        assert_eq!(
            req.blend(),
            vec![(ExpertId::new("a"), 1.0), (ExpertId::new("b"), 1.0)]
        );
        req.weights = vec![0.4];
        assert_eq!(
            req.blend(),
            vec![(ExpertId::new("a"), 0.4), (ExpertId::new("b"), 1.0)]
        );
    }

    #[test]
    fn a_request_without_weights_still_parses() {
        let req: ActRequest =
            serde_json::from_str(r#"{"task_id":"t","prompt":"p","adapters":["a"]}"#).unwrap();
        assert!(req.weights.is_empty());
    }
}
