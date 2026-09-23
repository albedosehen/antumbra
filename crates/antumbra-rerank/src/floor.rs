//! The relevance floor ADR-0023 B-2 asked for, built from the signal ADR-0023
//! B-2 ruled out.
//!
//! B-2 concluded that no fixed threshold over the cross-encoder can carry a
//! floor, because the score is trustworthy within a query and not across them.
//! That was measured on ten memories against a handful of varied nonsense
//! queries, and it is true of a THRESHOLD.
//!
//! It is not true of a CALIBRATION, which the record never tried. Fitting a
//! logistic over `log10(score)` against verifier-produced labels — 800 balanced
//! pairs from 400 distinct queries, fit on half and measured on the other half —
//! gives accuracy 0.803, F1 0.797 and an expected calibration error of 0.033 at
//! the natural 0.5 floor. The tuned threshold the same data supports reaches
//! 0.785 F1, and does so IN SAMPLE, having chosen its cut point on the rows it
//! was scored against. The calibrated probability is both better and honest
//! about being out of sample.
//!
//! **Why this satisfies [`TypedDecider`] rather than dodging it.** The port
//! requires an implementation trained against a strictly proper scoring rule
//! over outcomes a verifier produced. The labels come from
//! `scripts/d2-labels.sh`, whose verifier is span provenance with the span
//! excised, derived from no model; the fit minimises log loss, which is strictly
//! proper. A head fitted on a model's own answers would violate ADR-0022's
//! anchor invariant. This is not that.
//!
//! What it is NOT: a general typed decider. It answers `Noul` and nothing else,
//! because a cross-encoder scores pair relevance and has no opinion about a
//! `Choice` or a `Score`. ADR-0024's D-1 still needs the real head.

use antumbra_core::platt::Platt;
use antumbra_core::ports::{Answer, Question, RelevanceScorer, TypedDecider};
use antumbra_core::{AntumbraError, Result};
use async_trait::async_trait;
use std::sync::Arc;

/// The separator `above_floor` joins its per-memory states with.
const STATE_SEPARATOR: &str = "\n---\n";
const QUERY_PREFIX: &str = "QUERY: ";
const MEMORY_PREFIX: &str = "MEMORY: ";

/// Fitted on 800 balanced pairs drawn from 400 distinct queries against
/// `ws:default`, by `antumbra_core::platt::Platt::fit` minimising log loss, and
/// reported on the held-out half (`calibrating_the_reranker`).
///
/// These are defaults rather than constants of nature. They were fitted against
/// ONE deployment's corpus and ONE reranker (`BAAI/bge-reranker-base` via
/// text-embeddings-inference), and a different model or a very different corpus
/// wants a refit — which is why [`CalibratedFloor::with_calibration`] exists and
/// why the fitting procedure ships beside the numbers.
pub const FITTED: Platt = Platt { a: 1.630, b: 5.362 };

/// Answers "does this memory answer this query" as a calibrated probability, by
/// scoring the pair with a cross-encoder and mapping the score through [`FITTED`].
pub struct CalibratedFloor {
    scorer: Arc<dyn RelevanceScorer>,
    calibration: Platt,
}

impl CalibratedFloor {
    /// With the fitted calibration.
    pub fn new(scorer: Arc<dyn RelevanceScorer>) -> Self {
        Self {
            scorer,
            calibration: FITTED,
        }
    }

    /// With a calibration of your own, from a refit on your corpus.
    pub fn with_calibration(scorer: Arc<dyn RelevanceScorer>, calibration: Platt) -> Self {
        Self {
            scorer,
            calibration,
        }
    }

    /// Split the joined state back into its `(query, memory)` pairs.
    ///
    /// The state arrives as one string because [`TypedDecider`] takes one state
    /// and many questions — the questions about one state share an encoding,
    /// which is what makes a batch cheap. For this decider they do not share an
    /// encoding at all, so the join has to be undone. That is a wart in the
    /// contract rather than in this code, and it is written out here so the next
    /// reader sees the seam instead of guessing at it.
    fn pairs(state: &str) -> Vec<(String, String)> {
        state
            .split(STATE_SEPARATOR)
            .filter_map(|chunk| {
                let rest = chunk.strip_prefix(QUERY_PREFIX)?;
                let (query, memory) = rest.split_once('\n')?;
                Some((
                    query.to_string(),
                    memory.strip_prefix(MEMORY_PREFIX)?.to_string(),
                ))
            })
            .collect()
    }
}

#[async_trait]
impl TypedDecider for CalibratedFloor {
    async fn decide(&self, state: &str, questions: &[Question]) -> Result<Vec<Answer>> {
        if questions.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(bad) = questions.iter().find(|q| !matches!(q, Question::Noul)) {
            // Refuse rather than answer the wrong variant. `above_floor` keeps a
            // row whose answer does not match its question, so a wrong variant
            // would silently disable the floor instead of reporting a problem.
            return Err(AntumbraError::other(format!(
                "CalibratedFloor answers Noul only, asked {bad:?}"
            )));
        }
        let pairs = Self::pairs(state);
        if pairs.len() != questions.len() {
            return Err(AntumbraError::other(format!(
                "state parsed into {} pairs for {} questions",
                pairs.len(),
                questions.len()
            )));
        }
        // Every pair shares one query, so this is one call rather than N.
        let query = pairs[0].0.clone();
        let texts: Vec<String> = pairs.into_iter().map(|(_, m)| m).collect();
        let scores = self.scorer.relevance(&query, &texts).await?;
        if scores.len() != questions.len() {
            return Err(AntumbraError::other(format!(
                "scorer returned {} scores for {} texts",
                scores.len(),
                questions.len()
            )));
        }
        Ok(scores
            .into_iter()
            .map(|s| Answer::Noul {
                probability: self.calibration.probability(s),
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FixedScorer(Vec<f32>);

    #[async_trait]
    impl RelevanceScorer for FixedScorer {
        async fn relevance(&self, _q: &str, _t: &[String]) -> Result<Vec<f32>> {
            Ok(self.0.clone())
        }
    }

    fn state(query: &str, memories: &[&str]) -> String {
        memories
            .iter()
            .map(|m| format!("QUERY: {query}\nMEMORY: {m}"))
            .collect::<Vec<_>>()
            .join(STATE_SEPARATOR)
    }

    /// The seam that matters: the state `above_floor` builds must parse back
    /// into exactly the pairs that went in. If this drifts, the floor silently
    /// stops being applied.
    #[test]
    fn the_joined_state_round_trips() {
        let s = state("what is a boundary", &["a memory", "another memory"]);
        let pairs = CalibratedFloor::pairs(&s);
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].0, "what is a boundary");
        assert_eq!(pairs[0].1, "a memory");
        assert_eq!(pairs[1].1, "another memory");
    }

    /// A high score reads as relevant and a low one does not, through the
    /// calibration actually shipped. This pins the fitted constants against an
    /// accidental sign flip, which would empty every recall.
    #[tokio::test]
    async fn the_shipped_calibration_separates_the_measured_bands() {
        // 7.8e-4 is the threshold the same data supports; 3.7e-5 is the score a
        // memory got against a query about nothing, both from the measurements
        // in ADR-0023 B-2 and ADR-0024 D-2.
        let scorer = Arc::new(FixedScorer(vec![5e-3, 3.7e-5]));
        let floor = CalibratedFloor::new(scorer);
        let answers = floor
            .decide(&state("q", &["relevant", "not"]), &vec![Question::Noul; 2])
            .await
            .expect("decide");
        let p = |a: &Answer| match a {
            Answer::Noul { probability } => *probability,
            _ => panic!("wrong variant"),
        };
        assert!(p(&answers[0]) > 0.5, "got {}", p(&answers[0]));
        assert!(p(&answers[1]) < 0.5, "got {}", p(&answers[1]));
    }

    /// A question this decider cannot answer is refused, not guessed. Returning
    /// the wrong variant would make `above_floor` keep the row and the floor
    /// would quietly do nothing.
    #[tokio::test]
    async fn a_question_it_cannot_answer_is_refused() {
        let floor = CalibratedFloor::new(Arc::new(FixedScorer(vec![1.0])));
        let err = floor
            .decide(
                &state("q", &["m"]),
                &[Question::Score {
                    low: 0.0,
                    high: 1.0,
                }],
            )
            .await;
        assert!(err.is_err(), "a Score question must be refused");
    }

    /// A malformed state is an error rather than a silent short answer: returning
    /// fewer answers than questions makes `above_floor` skip the floor, which is
    /// the safe direction but must be reported.
    #[tokio::test]
    async fn a_state_that_does_not_parse_is_reported() {
        let floor = CalibratedFloor::new(Arc::new(FixedScorer(vec![1.0, 1.0])));
        let err = floor
            .decide("not a state at all", &vec![Question::Noul; 2])
            .await;
        assert!(err.is_err());
    }
}
