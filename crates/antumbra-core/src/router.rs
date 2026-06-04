//! A learned router (ADR-0009 — the north-star gate, in its routing form).
//!
//! The v0 gate routes by raw cosine to each expert's capability centroid. Frozen
//! sentence embeddings compress general and specific experts into the same band,
//! so a specialist barely outscores a generalist (EXP-012: a 0.020 margin). This
//! applies a **learned per-dimension metric** that amplifies the directions which
//! actually separate experts, then routes by cosine to per-expert centroids in
//! that reweighted space. The metric is trained on the population's own solved
//! exemplars (free labels) in `antumbra-train`; inference here is pure arithmetic
//! so the gate stays dependency-light.

use serde::{Deserialize, Serialize};

use crate::ExpertId;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterExpert {
    pub id: ExpertId,
    /// Centroid of this expert's exemplars in the reweighted, L2-normalized space.
    pub centroid: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedRouter {
    /// Per-dimension metric (learned feature weighting) over the raw embedding.
    pub weights: Vec<f32>,
    pub experts: Vec<RouterExpert>,
    /// Softmax temperature on the cosine logits.
    pub temperature: f32,
}

impl LearnedRouter {
    /// Apply the learned metric and L2-normalize.
    pub fn project(&self, x: &[f32]) -> Vec<f32> {
        let scaled: Vec<f32> = x
            .iter()
            .zip(&self.weights)
            .map(|(a, w)| a * w)
            .collect();
        let norm = scaled.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
        scaled.iter().map(|v| v / norm).collect()
    }

    /// Routing probabilities over the experts for a task embedding, best first.
    pub fn route(&self, task: &[f32]) -> Vec<(ExpertId, f32)> {
        let t = self.project(task);
        let temp = self.temperature.max(1e-4);
        let logits: Vec<f32> = self
            .experts
            .iter()
            .map(|e| t.iter().zip(&e.centroid).map(|(a, b)| a * b).sum::<f32>() / temp)
            .collect();
        let max = logits.iter().copied().fold(f32::MIN, f32::max);
        let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
        let sum: f32 = exps.iter().sum::<f32>().max(1e-9);
        let mut out: Vec<(ExpertId, f32)> = self
            .experts
            .iter()
            .zip(exps)
            .map(|(e, x)| (e.id.clone(), x / sum))
            .collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A metric that zeroes the shared first dimension and keeps the
    /// discriminative second/third separates two otherwise-similar experts.
    #[test]
    fn reweighting_separates_near_identical_centroids() {
        let router = LearnedRouter {
            // Down-weight the shared dim (0), keep the discriminative dims.
            weights: vec![0.0, 1.0, 1.0],
            experts: vec![
                RouterExpert {
                    id: ExpertId::new("general"),
                    centroid: vec![0.0, 1.0, 0.0],
                },
                RouterExpert {
                    id: ExpertId::new("specific"),
                    centroid: vec![0.0, 0.0, 1.0],
                },
            ],
            temperature: 0.1,
        };
        // Raw cosine of this task to both centroids is high and close (shared
        // dim 0 dominates); under the learned metric it lands on "specific".
        let ranked = router.route(&[0.95, 0.1, 0.3]);
        assert_eq!(ranked[0].0, ExpertId::new("specific"));
        assert!(ranked[0].1 > 0.5);
    }
}
