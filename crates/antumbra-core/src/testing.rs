//! In-memory fakes of every port, so the loop, gate, and boundary engine run
//! end-to-end with no GPU, model, or database. Enabled by the `testing` feature
//! and always available under `cfg(test)`.

use async_trait::async_trait;

use crate::error::Result;
use crate::ports::{
    AcceptabilityProbe, ActOutput, ActRequest, Critic, CriticScore, Embedder, Reranker, Serve,
    StepOutput, TrainOutcome, TrainRequest, Trainer, Verifier, VerifierVerdict, VerifyRequest,
};

/// Deterministic embedder: maps text to a fixed-dimension unit-ish vector by
/// hashing characters into buckets. No randomness, so tests are reproducible.
#[derive(Debug, Clone)]
pub struct FixedEmbedder {
    dim: usize,
}

impl FixedEmbedder {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

#[async_trait]
impl Embedder for FixedEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut v = vec![0.0f32; self.dim];
        for (i, b) in text.bytes().enumerate() {
            v[i % self.dim] += (b as f32) / 255.0;
        }
        Ok(v)
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

/// Deterministic reranker for tests: reorders candidates by a scripted rule with
/// no network, mirroring [`FixedEmbedder`]'s role for the rerank stage. Either
/// promotes any candidate whose content contains a token to the front (the rest
/// keep their input order), or always fails (to exercise the degrade-on-error
/// path).
#[derive(Debug, Clone)]
pub struct ScriptedReranker {
    promote_substring: Option<String>,
    fail: bool,
}

impl ScriptedReranker {
    /// Promote every candidate whose content contains `token` to the front,
    /// preserving the relative order of the rest. Models a cross-encoder that
    /// lifts an exact-match document the bi-encoder ranked lower.
    pub fn promoting_content_substring(token: impl Into<String>) -> Self {
        Self {
            promote_substring: Some(token.into()),
            fail: false,
        }
    }

    /// A reranker that always returns `Err`, so a caller's degrade-to-pre-rerank
    /// path can be tested.
    pub fn failing() -> Self {
        Self {
            promote_substring: None,
            fail: true,
        }
    }
}

#[async_trait]
impl Reranker for ScriptedReranker {
    async fn rerank(&self, _query: &str, candidates: &[(String, String)]) -> Result<Vec<String>> {
        if self.fail {
            return Err(crate::error::AntumbraError::other(
                "scripted rerank failure",
            ));
        }
        let token = self.promote_substring.as_deref().unwrap_or("");
        let (mut promoted, mut rest): (Vec<String>, Vec<String>) = (Vec::new(), Vec::new());
        for (id, text) in candidates {
            if !token.is_empty() && text.contains(token) {
                promoted.push(id.clone());
            } else {
                rest.push(id.clone());
            }
        }
        promoted.extend(rest);
        Ok(promoted)
    }
}

/// Echoes the prompt back as a single-step output.
#[derive(Debug, Clone, Default)]
pub struct EchoServe;

#[async_trait]
impl Serve for EchoServe {
    async fn act(&self, req: ActRequest) -> Result<ActOutput> {
        Ok(ActOutput {
            steps: vec![StepOutput {
                step_idx: 0,
                content: req.prompt.clone(),
            }],
            final_output: req.prompt,
        })
    }
}

/// Trainer that returns a preset rising reward curve and a synthetic adapter
/// path. `final_fitness` drives whether the loop graduates or prunes the shadow.
#[derive(Debug, Clone)]
pub struct ScriptedTrainer {
    pub final_fitness: f32,
    pub curve: Vec<f32>,
    pub capability_exemplars: Vec<String>,
    pub boundary_findings: Vec<crate::BoundaryFinding>,
    /// Per-task results the loop slices for the standing instruments. Empty by
    /// default, which is what a trainer that reports only aggregate fitness
    /// looks like -- and the loop has to stay honest about that rather than
    /// inventing a report from one number.
    pub per_task: Vec<crate::ports::TaskOutcome>,
    /// Behave like a trainer that predates the holdout: learn from everything,
    /// report everything, and echo no holdout back. The loop must then decline
    /// to measure the generation rather than trust per-task results whose
    /// held-out half was learned from.
    pub ignores_holdout: bool,
    /// The recipe this trainer is configured with, echoed back when a request
    /// names none, as a real trainer reports the settings it actually used.
    pub own_recipe: Option<crate::TrainingRecipe>,
    /// Behave like a trainer that predates the recipe: train under its own
    /// settings whatever is asked and echo no recipe back. The loop must then
    /// write no recipe row rather than one naming settings nobody ran.
    pub ignores_recipe: bool,
    /// What a re-measurement reports, cycled across the seeds asked for.
    /// `None` refuses re-measurement, as a trainer without it does.
    pub remeasured: Option<Vec<f32>>,
}

/// The recipe a scripted trainer is configured with.
pub const SCRIPTED_RECIPE: crate::TrainingRecipe = crate::TrainingRecipe {
    learning_rate: 1e-4,
    batch_size: 1,
    kl_beta: 0.04,
};

impl ScriptedTrainer {
    /// A trainer whose shadow graduates.
    pub fn graduating() -> Self {
        Self {
            final_fitness: 0.9,
            curve: vec![0.1, 0.4, 0.7, 0.9],
            capability_exemplars: Vec::new(),
            boundary_findings: Vec::new(),
            per_task: Vec::new(),
            ignores_holdout: false,
            own_recipe: Some(SCRIPTED_RECIPE),
            ignores_recipe: false,
            remeasured: None,
        }
    }

    /// A trainer whose shadow collapses and should be pruned.
    pub fn collapsing() -> Self {
        Self {
            final_fitness: 0.0,
            curve: vec![0.0, 0.0, 0.0],
            capability_exemplars: Vec::new(),
            boundary_findings: Vec::new(),
            per_task: Vec::new(),
            ignores_holdout: false,
            own_recipe: Some(SCRIPTED_RECIPE),
            ignores_recipe: false,
            remeasured: None,
        }
    }

    /// A graduating trainer that reports the prompts its shadow solved, so the
    /// loop derives the capability vector from evaluated behavior.
    pub fn graduating_with_exemplars(exemplars: Vec<String>) -> Self {
        Self {
            capability_exemplars: exemplars,
            ..Self::graduating()
        }
    }

    /// A graduating trainer that also surfaces a verified correction's boundary
    /// finding, so the loop's actionable-boundary persistence can be exercised.
    pub fn graduating_with_boundary(finding: crate::BoundaryFinding) -> Self {
        Self {
            boundary_findings: vec![finding],
            ..Self::graduating()
        }
    }
}

#[async_trait]
impl Trainer for ScriptedTrainer {
    async fn train_shadow(&self, req: TrainRequest) -> Result<TrainOutcome> {
        // A scripted run learns nothing, so it withholds trivially; what it
        // must still model is the reporting: under a holdout it reports only
        // the tasks that run measures, as a real trainer does.
        let holdout = req.holdout.filter(|_| !self.ignores_holdout);
        let per_task = self
            .per_task
            .iter()
            .filter(|t| holdout.is_none_or(|h| t.impossible || h.measures(&t.task_id)))
            .cloned()
            .collect();
        Ok(TrainOutcome {
            adapter_uri: format!("memory://adapter/{}", req.shadow),
            reward_curve: self.curve.clone(),
            final_fitness: self.final_fitness,
            capability_exemplars: self.capability_exemplars.clone(),
            boundary_findings: self.boundary_findings.clone(),
            per_task,
            holdout,
            recipe: if self.ignores_recipe {
                None
            } else {
                req.recipe.or(self.own_recipe)
            },
        })
    }

    async fn remeasure(
        &self,
        req: crate::ports::RemeasureRequest,
    ) -> Result<crate::ports::Remeasurement> {
        let Some(rates) = &self.remeasured else {
            return Err(crate::AntumbraError::Unimplemented("re-measurement"));
        };
        Ok(crate::ports::Remeasurement {
            pass_rates: (0..req.seeds.len())
                .filter_map(|i| rates.get(i % rates.len().max(1)).copied())
                .collect(),
            held_out: req.holdout.is_some(),
            tasks: self.per_task.len(),
        })
    }
}

/// Verifier that passes when the artifact's `marker` field equals `expect`.
#[derive(Debug, Clone)]
pub struct MarkerVerifier {
    pub expect: String,
}

#[async_trait]
impl Verifier for MarkerVerifier {
    async fn verify(&self, req: &VerifyRequest) -> Result<VerifierVerdict> {
        let passed = req
            .artifact
            .get("marker")
            .and_then(|m| m.as_str())
            .map(|m| m == self.expect)
            .unwrap_or(false);
        Ok(VerifierVerdict {
            passed,
            value: if passed { 1.0 } else { 0.0 },
        })
    }
}

/// Critic that assigns a flat per-step densified score.
#[derive(Debug, Clone)]
pub struct FlatCritic {
    pub value: f32,
}

#[async_trait]
impl Critic for FlatCritic {
    async fn densify(&self, output: &ActOutput) -> Result<Vec<CriticScore>> {
        Ok(output
            .steps
            .iter()
            .map(|s| CriticScore {
                step_idx: s.step_idx,
                dimension: "critic".into(),
                value: self.value,
            })
            .collect())
    }
}

/// Acceptability probe keyed on a single governing feature: the behavior is
/// acceptable exactly when `context[feature] == ok_value`. This lets a boundary
/// test assert that counterfactual search recovers `feature` as the governing
/// dimension.
#[derive(Debug, Clone)]
pub struct FeatureProbe {
    pub feature: String,
    pub ok_value: serde_json::Value,
}

#[async_trait]
impl AcceptabilityProbe for FeatureProbe {
    async fn acceptable(&self, _behavior: &str, context: &serde_json::Value) -> Result<bool> {
        Ok(context.get(&self.feature) == Some(&self.ok_value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn fixed_embedder_is_deterministic() {
        let e = FixedEmbedder::new(8);
        assert_eq!(
            e.embed("hello").await.unwrap(),
            e.embed("hello").await.unwrap()
        );
        assert_eq!(e.dim(), 8);
    }

    #[tokio::test]
    async fn scripted_reranker_promotes_matching_content() {
        let r = ScriptedReranker::promoting_content_substring("PROMOTE");
        let cands = vec![
            ("a".to_string(), "ordinary text".to_string()),
            ("b".to_string(), "carries PROMOTE token".to_string()),
            ("c".to_string(), "also ordinary".to_string()),
        ];
        let out = r.rerank("q", &cands).await.unwrap();
        assert_eq!(out, vec!["b", "a", "c"], "matching id leads, rest stable");
    }

    #[tokio::test]
    async fn scripted_reranker_failing_returns_err() {
        assert!(ScriptedReranker::failing()
            .rerank("q", &[("a".to_string(), "x".to_string())])
            .await
            .is_err());
    }

    #[tokio::test]
    async fn marker_verifier_checks_the_field() {
        let v = MarkerVerifier {
            expect: "ok".into(),
        };
        let req = VerifyRequest {
            run_id: crate::ids::RunId::new("run:1"),
            step_idx: 0,
            dimension: "tests".into(),
            artifact: serde_json::json!({"marker": "ok"}),
        };
        assert!(v.verify(&req).await.unwrap().passed);
    }

    #[tokio::test]
    async fn feature_probe_recovers_governing_value() {
        let p = FeatureProbe {
            feature: "runtime".into(),
            ok_value: serde_json::json!("node"),
        };
        assert!(p
            .acceptable("npm install", &serde_json::json!({"runtime": "node"}))
            .await
            .unwrap());
        assert!(!p
            .acceptable("npm install", &serde_json::json!({"runtime": "deno"}))
            .await
            .unwrap());
    }
}

/// A [`TypedDecider`](crate::ports::TypedDecider) whose answers are decided by a
/// substring rule, so a caller's use of the port can be tested before any head
/// exists (ADR-0024).
///
/// It is deliberately honest about its own confidence: the probability it
/// reports is the one it was configured with, not 1.0, because a caller that
/// only ever sees certainty will not exercise the threshold that is the whole
/// reason for asking a typed question.
pub struct ScriptedDecider {
    /// A state containing this substring is judged true / in-scope.
    token: String,
    /// What to report when the token is present. The complement is reported when
    /// it is absent, so a caller sees both sides of its threshold.
    confidence: f32,
    fail: bool,
}

impl ScriptedDecider {
    /// Answers `Noul` with `confidence` when `token` is in the state, and with
    /// `1 - confidence` when it is not; `Choice` picks the option containing the
    /// token, and `Score` returns the midpoint of the requested scale.
    pub fn on_substring(token: impl Into<String>, confidence: f32) -> Self {
        Self {
            token: token.into(),
            confidence,
            fail: false,
        }
    }

    /// A decider that always returns `Err`, so a caller's degrade path can be
    /// tested. ADR-0024 requires a head that fails to load to fall back to the
    /// path it replaced rather than to nothing.
    pub fn failing() -> Self {
        Self {
            token: String::new(),
            confidence: 0.0,
            fail: true,
        }
    }
}

#[async_trait]
impl crate::ports::TypedDecider for ScriptedDecider {
    async fn decide(
        &self,
        state: &str,
        questions: &[crate::ports::Question],
    ) -> Result<Vec<crate::ports::Answer>> {
        use crate::ports::{Answer, Question};
        if self.fail {
            return Err(crate::AntumbraError::other("scripted decider: failing"));
        }
        let hit = !self.token.is_empty() && state.contains(&self.token);
        Ok(questions
            .iter()
            .map(|q| match q {
                Question::Noul => Answer::Noul {
                    probability: if hit {
                        self.confidence
                    } else {
                        1.0 - self.confidence
                    },
                },
                Question::Score { low, high } => Answer::Score {
                    expected: (low + high) / 2.0,
                },
                Question::Choice { options } => {
                    let index = options
                        .iter()
                        .position(|o| !self.token.is_empty() && o.contains(&self.token))
                        .unwrap_or(0);
                    // Mass on the pick, the rest spread evenly: a distribution a
                    // caller can actually read, rather than a one-hot that hides
                    // whether the head was torn between two options.
                    let n = options.len().max(1);
                    let rest = if n > 1 {
                        (1.0 - self.confidence) / (n - 1) as f32
                    } else {
                        0.0
                    };
                    let probs = (0..n)
                        .map(|i| if i == index { self.confidence } else { rest })
                        .collect();
                    Answer::Choice { index, probs }
                }
            })
            .collect())
    }
}

#[cfg(test)]
mod scripted_decider {
    use super::*;
    use crate::ports::{Answer, Question, TypedDecider};

    /// The seam works before any head exists, which is the point of defining the
    /// port first: a caller of ADR-0024's D-2 floor can be written and tested
    /// against this, and the trained head drops in as the only changed piece.
    #[tokio::test]
    async fn a_noul_answers_both_sides_of_a_threshold() {
        let d = ScriptedDecider::on_substring("relevant", 0.9);
        let hit = d
            .decide("this is relevant", &[Question::Noul])
            .await
            .unwrap();
        let miss = d.decide("this is not", &[Question::Noul]).await.unwrap();
        match (&hit[0], &miss[0]) {
            (Answer::Noul { probability: a }, Answer::Noul { probability: b }) => {
                assert!(*a > 0.8 && *b < 0.2, "got {a} and {b}");
            }
            other => panic!("a Noul question must get a Noul answer, got {other:?}"),
        }
    }

    /// A choice returns a DISTRIBUTION, not just a pick. That is the half the
    /// prototype margin could not express: "torn between two good options" and
    /// "none of these fit" are different answers, and only the distribution
    /// distinguishes them.
    #[tokio::test]
    async fn a_choice_returns_a_distribution_and_not_only_a_pick() {
        let d = ScriptedDecider::on_substring("rust", 0.7);
        let q = Question::Choice {
            options: vec![
                "python expert".into(),
                "rust expert".into(),
                "go expert".into(),
            ],
        };
        let out = d.decide("write a rust trait", &[q]).await.unwrap();
        match &out[0] {
            Answer::Choice { index, probs } => {
                assert_eq!(*index, 1, "the rust option is chosen");
                assert_eq!(probs.len(), 3);
                let total: f32 = probs.iter().sum();
                assert!(
                    (total - 1.0).abs() < 1e-5,
                    "probs must sum to 1, got {total}"
                );
                assert!(probs[1] > probs[0] && probs[1] > probs[2]);
            }
            other => panic!("expected a Choice answer, got {other:?}"),
        }
    }

    /// One state, several questions, one call: the questions about a state share
    /// an encoding, and answering them together is what makes this affordable in
    /// a serving path.
    #[tokio::test]
    async fn a_batch_answers_in_order_and_matches_each_variant() {
        let d = ScriptedDecider::on_substring("x", 0.8);
        let out = d
            .decide(
                "x",
                &[
                    Question::Noul,
                    Question::Score {
                        low: 0.0,
                        high: 10.0,
                    },
                    Question::Choice {
                        options: vec!["a".into(), "x".into()],
                    },
                ],
            )
            .await
            .unwrap();
        assert_eq!(out.len(), 3, "one answer per question, in order");
        assert!(matches!(out[0], Answer::Noul { .. }));
        assert!(matches!(out[1], Answer::Score { expected } if (expected - 5.0).abs() < 1e-6));
        assert!(matches!(out[2], Answer::Choice { index: 1, .. }));
    }

    /// ADR-0024 requires a head that fails to load to degrade to the path it
    /// replaced rather than to nothing, so the failure has to be visible.
    #[tokio::test]
    async fn a_failing_decider_reports_the_failure() {
        let d = ScriptedDecider::failing();
        assert!(d.decide("anything", &[Question::Noul]).await.is_err());
    }
}
