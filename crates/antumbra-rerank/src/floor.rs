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
//! was scored against. The calibrated probability is both better and
//! measured out of sample.
//!
//! **Why this satisfies [`TypedDecider`] rather than dodging it.** The port
//! requires an implementation trained against a strictly proper scoring rule
//! over outcomes a verifier produced. The labels come from
//! `scripts/d2-labels.sh`, whose verifier is span provenance with the span
//! excised, derived from no model; the fit minimizes log loss, which is strictly
//! proper. A head fitted on a model's own answers would violate ADR-0022's
//! anchor invariant. This is not that.
//!
//! What it is NOT: a general typed decider. It answers `Noul` and nothing else,
//! because a cross-encoder scores pair relevance and has no opinion about a
//! `Choice` or a `Score`. ADR-0024's D-1 still needs the real head.

use antumbra_core::platt::Platt;
use antumbra_core::ports::{Answer, Question, RelevanceScorer, RelevanceState, TypedDecider};
use antumbra_core::{AntumbraError, Result};
use async_trait::async_trait;
use std::sync::Arc;

/// Fitted on 800 balanced pairs drawn from 400 distinct queries against
/// `ws:default`, by `antumbra_core::platt::Platt::fit` minimizing log loss, and
/// reported on the held-out half (`calibrating_the_reranker`).
///
/// These are defaults rather than constants of nature. They were fitted against
/// ONE deployment's corpus and ONE reranker (`BAAI/bge-reranker-base` via
/// text-embeddings-inference), and a different model or a very different corpus
/// wants a refit — which is why [`CalibratedFloor::with_calibration`] exists and
/// why the fitting procedure ships beside the numbers.
pub const FITTED: Platt = Platt { a: 1.630, b: 5.362 };

/// The same fit against `Alibaba-NLP/gte-reranker-modernbert-base`, served by
/// text-embeddings-inference, on the same 800 pairs (ADR-0024 D-2): on the
/// held-out half, accuracy 0.882 and F1 0.885 at the 0.5 floor, with an
/// expected calibration error of 0.043, where the `bge-reranker-base` fit
/// reaches 0.803, 0.797 and 0.033. It reads up to 8,192 tokens where that one
/// reads 512, so a long memory is scored whole.
pub const FITTED_GTE_MODERNBERT: Platt = Platt {
    a: 31.361,
    b: 5.233,
};

/// The calibrations fitted so far, by the model id a rerank endpoint reports.
/// A fit belongs to one model's scores: another model's fit read over them is
/// not a floor at all, which is why an unknown model gets none.
pub fn calibration_for(model: &str) -> Option<Platt> {
    match model.trim() {
        "BAAI/bge-reranker-base" => Some(FITTED),
        "Alibaba-NLP/gte-reranker-modernbert-base" => Some(FITTED_GTE_MODERNBERT),
        _ => None,
    }
}

/// Once the endpoint answers: whether the calibration the floor started with
/// suits the model it serves. `None` when it does; otherwise what is wrong and
/// what to do. Startup may have chosen from configuration alone, before the
/// endpoint was up, and a fit read over another model's scores is no floor.
pub fn recheck(served: &str, running: Option<Platt>) -> Option<String> {
    match (calibration_for(served), running) {
        (Some(fit), Some(running)) if fit == running => None,
        (None, None) => None,
        (Some(_), Some(_)) => Some(format!(
            "the reranker serves {served}, but the relevance floor runs on another model's calibration; restart the server to pick up {served}'s"
        )),
        (Some(_), None) => Some(format!(
            "the reranker serves {served}, which has a fitted calibration, but the relevance floor is off; restart the server to turn it on"
        )),
        (None, Some(_)) => Some(format!(
            "the reranker serves {served}, which no calibration is fitted for, yet the relevance floor runs on another model's: its probabilities mean nothing for these scores; set ANTUMBRA_RERANK_MODEL={served} (the floor then stays off) or pass --floor-calibration, and restart"
        )),
    }
}

/// `a,b` as a calibration, for one fitted by hand.
pub fn parse_calibration(text: &str) -> Option<Platt> {
    let (a, b) = text.split_once(',')?;
    let a: f32 = a.trim().parse().ok()?;
    let b: f32 = b.trim().parse().ok()?;
    (a.is_finite() && b.is_finite() && a > 0.0).then_some(Platt { a, b })
}

/// Which calibration the floor runs on and why, or why there is no floor.
#[derive(Debug, Clone, PartialEq)]
pub enum FloorChoice {
    Calibrated { calibration: Platt, because: String },
    Off { because: String },
}

/// Choose the floor's calibration. One given by hand wins. Otherwise the model
/// the endpoint says it serves decides, since that is what scores; failing
/// that, the configured model; failing both, the `bge-reranker-base` fit, the
/// default before any model was named. A named model with no fit has no
/// floor.
pub fn choose(
    explicit: Option<Platt>,
    reported: Option<&str>,
    configured: Option<&str>,
) -> FloorChoice {
    if let Some(calibration) = explicit {
        return FloorChoice::Calibrated {
            calibration,
            because: "the calibration given with --floor-calibration".into(),
        };
    }
    let (model, whose) = match (reported, configured) {
        (Some(m), _) => (m, "the model the endpoint serves"),
        (None, Some(m)) => (m, "the configured model; the endpoint did not say what it serves"),
        (None, None) => {
            return FloorChoice::Calibrated {
                calibration: FITTED,
                because: "the BAAI/bge-reranker-base fit, the default: neither the endpoint nor the configuration named a model".into(),
            }
        }
    };
    let mismatch = match (reported, configured) {
        (Some(r), Some(c)) if r.trim() != c.trim() => format!(" (configured as {c})"),
        _ => String::new(),
    };
    match calibration_for(model) {
        Some(calibration) => FloorChoice::Calibrated {
            calibration,
            because: format!("fitted for {model}, {whose}{mismatch}"),
        },
        None => FloorChoice::Off {
            because: format!(
                "no calibration is fitted for {model}, {whose}{mismatch}; fit one with scripts/d2-relevance-baseline.sh and the calibrating_the_reranker test, and pass it with --floor-calibration"
            ),
        },
    }
}

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
        // The state arrives as one string because the port takes one state and
        // many questions; for this decider the questions share a query and
        // nothing else, so it is decoded back into the query and its memories.
        let Some(RelevanceState { query, memories }) = RelevanceState::decode(state) else {
            return Err(AntumbraError::other(
                "the state is not a relevance state (a query and its memories)",
            ));
        };
        if memories.len() != questions.len() {
            return Err(AntumbraError::other(format!(
                "the state holds {} memories for {} questions",
                memories.len(),
                questions.len()
            )));
        }
        // Every memory shares one query, so this is one call rather than N.
        let texts = memories;
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
        RelevanceState {
            query: query.to_string(),
            memories: memories.iter().map(|m| m.to_string()).collect(),
        }
        .encode()
    }

    /// Records what it was asked, and scores every text the same.
    struct Recording(std::sync::Mutex<Vec<(String, Vec<String>)>>);

    #[async_trait]
    impl RelevanceScorer for Recording {
        async fn relevance(&self, q: &str, t: &[String]) -> Result<Vec<f32>> {
            self.0.lock().unwrap().push((q.to_string(), t.to_vec()));
            Ok(vec![5e-3; t.len()])
        }
    }

    /// The seam that matters: the scorer is asked about exactly the query and
    /// memories that went in, whatever text they hold. A query of several lines,
    /// and a memory with a markdown rule in it, broke the text join this
    /// replaced, and every such recall went out unfloored.
    #[tokio::test]
    async fn the_scorer_sees_the_query_and_memories_that_went_in() {
        let scorer = Arc::new(Recording(std::sync::Mutex::new(Vec::new())));
        let floor = CalibratedFloor::new(scorer.clone());
        let query = "why does recall\nskip the floor\n\nQUERY: on long prompts";
        let memories = ["first\n---\nsecond", "MEMORY: plain", ""];
        let answers = floor
            .decide(&state(query, &memories), &vec![Question::Noul; 3])
            .await
            .expect("decide");
        assert_eq!(answers.len(), 3);
        let asked = scorer.0.lock().unwrap().clone();
        assert_eq!(
            asked,
            [(
                query.to_string(),
                memories.iter().map(|m| m.to_string()).collect::<Vec<_>>()
            )]
        );
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

    /// The fit follows the model that scores: the endpoint's own answer over
    /// the configuration, a hand-fitted one over both, and no floor for a model
    /// nobody has fitted.
    #[test]
    fn the_calibration_follows_the_model_that_scores() {
        let gte = "Alibaba-NLP/gte-reranker-modernbert-base";
        let bge = "BAAI/bge-reranker-base";
        let fit = |c: FloorChoice| match c {
            FloorChoice::Calibrated { calibration, .. } => Some(calibration),
            FloorChoice::Off { .. } => None,
        };
        assert_eq!(
            fit(choose(None, Some(gte), Some(bge))),
            Some(FITTED_GTE_MODERNBERT)
        );
        assert_eq!(
            fit(choose(None, None, Some(gte))),
            Some(FITTED_GTE_MODERNBERT)
        );
        assert_eq!(fit(choose(None, None, None)), Some(FITTED));
        assert_eq!(fit(choose(None, Some("acme/reranker"), Some(gte))), None);
        let mine = Platt { a: 2.0, b: 1.0 };
        assert_eq!(
            fit(choose(Some(mine), Some("acme/reranker"), None)),
            Some(mine)
        );
        let FloorChoice::Calibrated { because, .. } = choose(None, Some(gte), Some(bge)) else {
            panic!("calibrated");
        };
        assert!(
            because.contains("configured as BAAI/bge-reranker-base"),
            "{because}"
        );
    }

    /// Checked against what the endpoint turns out to serve: quiet when the fit
    /// is that model's, a warning for every way it can be wrong.
    #[test]
    fn the_running_calibration_is_checked_against_the_served_model() {
        let gte = "Alibaba-NLP/gte-reranker-modernbert-base";
        assert_eq!(recheck(gte, Some(FITTED_GTE_MODERNBERT)), None);
        assert_eq!(recheck("acme/reranker", None), None);
        let warned =
            |served: &str, running: Option<Platt>| recheck(served, running).unwrap_or_default();
        assert!(warned(gte, Some(FITTED)).contains("another model's calibration"));
        assert!(warned(gte, None).contains("the relevance floor is off"));
        assert!(warned("acme/reranker", Some(FITTED)).contains("mean nothing"));
    }

    #[test]
    fn a_hand_fitted_calibration_parses_or_is_refused() {
        assert_eq!(
            parse_calibration(" 31.361, 5.233"),
            Some(FITTED_GTE_MODERNBERT)
        );
        assert_eq!(parse_calibration("31.361"), None);
        assert_eq!(
            parse_calibration("-1,2"),
            None,
            "a higher score must read as more likely"
        );
        assert_eq!(parse_calibration("a,b"), None);
    }

    /// Each fit separates its own model's bands: a relevant and an irrelevant
    /// score from the gte measurement fall on either side of 0.5.
    #[test]
    fn the_gte_calibration_separates_its_own_scores() {
        assert!(FITTED_GTE_MODERNBERT.probability(0.9) > 0.9);
        assert!(FITTED_GTE_MODERNBERT.probability(0.2) < 0.5);
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
