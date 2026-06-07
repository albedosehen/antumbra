//! In-memory fakes of every port, so the loop, gate, and boundary engine run
//! end-to-end with no GPU, model, or database. Enabled by the `testing` feature
//! and always available under `cfg(test)`.

use async_trait::async_trait;

use crate::error::Result;
use crate::ports::{
    AcceptabilityProbe, ActOutput, ActRequest, Critic, CriticScore, Embedder, Serve, StepOutput,
    TrainOutcome, TrainRequest, Trainer, Verifier, VerifierVerdict, VerifyRequest,
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
}

impl ScriptedTrainer {
    /// A trainer whose shadow graduates.
    pub fn graduating() -> Self {
        Self {
            final_fitness: 0.9,
            curve: vec![0.1, 0.4, 0.7, 0.9],
            capability_exemplars: Vec::new(),
            boundary_findings: Vec::new(),
        }
    }

    /// A trainer whose shadow collapses and should be pruned.
    pub fn collapsing() -> Self {
        Self {
            final_fitness: 0.0,
            curve: vec![0.0, 0.0, 0.0],
            capability_exemplars: Vec::new(),
            boundary_findings: Vec::new(),
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
        Ok(TrainOutcome {
            adapter_uri: format!("memory://adapter/{}", req.shadow),
            reward_curve: self.curve.clone(),
            final_fitness: self.final_fitness,
            capability_exemplars: self.capability_exemplars.clone(),
            boundary_findings: self.boundary_findings.clone(),
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
