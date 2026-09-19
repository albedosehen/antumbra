//! A learned router (the north-star gate, in its routing form).
//!
//! The v0 gate routes by raw cosine to each expert's capability centroid. Frozen
//! sentence embeddings compress general and specific tasks into nearly the same direction (a 0.020 margin between them). This
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
    /// In-distribution confidence floor: the calibrated lower bound on the
    /// nearest-centroid similarity (in the learned metric space) for a task to
    /// be considered covered by the population. A task below it is out of
    /// distribution -> escalate, not route (selective prediction; DynMoLE-style
    /// uncertainty gating; the learned gate).
    #[serde(default)]
    pub floor: f32,
}

/// Dot product, or `None` when the vectors are different lengths. A dimension
/// mismatch (e.g. a router trained at one embedding width meeting a task embedded
/// at another) must not silently `zip`-truncate into a plausible-but-wrong
/// similarity; the caller treats `None` as "no match" and abstains, mirroring the
/// length guard in [`crate::expert`]'s `cosine_similarity`.
fn aligned_dot(a: &[f32], b: &[f32]) -> Option<f32> {
    (a.len() == b.len()).then(|| a.iter().zip(b).map(|(x, y)| x * y).sum())
}

impl LearnedRouter {
    /// Apply the learned metric and L2-normalize. Returns an empty vector when the
    /// task width does not match the learned `weights` (no valid projection), so
    /// downstream scoring abstains rather than routing on a truncated vector.
    pub fn project(&self, x: &[f32]) -> Vec<f32> {
        if x.len() != self.weights.len() {
            return Vec::new();
        }
        let scaled: Vec<f32> = x.iter().zip(&self.weights).map(|(a, w)| a * w).collect();
        let norm = scaled.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
        scaled.iter().map(|v| v / norm).collect()
    }

    /// The nearest-centroid similarity in the learned space: the absolute
    /// confidence that *some* expert covers this task (unlike the softmax,
    /// which is purely relative and always picks a max). A dim mismatch yields
    /// `f32::MIN` (no expert matched), so `covers` reports out-of-distribution.
    pub fn top_similarity(&self, task: &[f32]) -> f32 {
        let t = self.project(task);
        self.experts
            .iter()
            .filter_map(|e| aligned_dot(&t, &e.centroid))
            .fold(f32::MIN, f32::max)
    }

    /// Whether the population covers this task at all (vs out-of-distribution).
    pub fn covers(&self, task: &[f32]) -> bool {
        self.top_similarity(task) >= self.floor
    }

    /// Routing probabilities over the experts for a task embedding, best first.
    /// Experts whose centroid width does not match the projected task are not
    /// candidates; if none match, the result is empty (escalate, don't route).
    pub fn route(&self, task: &[f32]) -> Vec<(ExpertId, f32)> {
        let t = self.project(task);
        let temp = self.temperature.max(1e-4);
        let scored: Vec<(&RouterExpert, f32)> = self
            .experts
            .iter()
            .filter_map(|e| aligned_dot(&t, &e.centroid).map(|s| (e, s / temp)))
            .collect();
        if scored.is_empty() {
            return Vec::new();
        }
        let max = scored.iter().map(|(_, l)| *l).fold(f32::MIN, f32::max);
        let exps: Vec<f32> = scored.iter().map(|(_, l)| (l - max).exp()).collect();
        let sum: f32 = exps.iter().sum::<f32>().max(1e-9);
        let mut out: Vec<(ExpertId, f32)> = scored
            .iter()
            .zip(exps)
            .map(|((e, _), x)| (e.id.clone(), x / sum))
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
            floor: 0.5,
        };
        // Raw cosine of this task to both centroids is high and close (shared
        // dim 0 dominates); under the learned metric it lands on "specific".
        let ranked = router.route(&[0.95, 0.1, 0.3]);
        assert_eq!(ranked[0].0, ExpertId::new("specific"));
        assert!(ranked[0].1 > 0.5);
    }

    #[test]
    fn covers_in_distribution_and_abstains_out_of_distribution() {
        let router = LearnedRouter {
            weights: vec![0.0, 1.0, 1.0],
            experts: vec![RouterExpert {
                id: ExpertId::new("e"),
                centroid: vec![0.0, 1.0, 0.0],
            }],
            temperature: 0.1,
            floor: 0.5,
        };
        // Aligned with the expert's discriminative axis -> covered.
        assert!(router.covers(&[0.1, 1.0, 0.0]));
        // The metric zeroes dim 0, so a task only on dim 0 projects to ~nothing
        // -> top similarity below the floor -> not covered (out of distribution).
        assert!(!router.covers(&[1.0, 0.0, 0.0]));
    }

    // A task embedded at the wrong width must abstain (escalate), not route on a
    // silently truncated dot product.
    #[test]
    fn a_dimension_mismatch_abstains_instead_of_routing() {
        let router = LearnedRouter {
            weights: vec![1.0, 1.0, 1.0],
            experts: vec![RouterExpert {
                id: ExpertId::new("e"),
                centroid: vec![0.0, 1.0, 0.0],
            }],
            temperature: 0.1,
            floor: -1.0, // a permissive floor: only the dim guard should abstain
        };
        // Wrong width (2 vs 3): no valid projection -> not covered, no route.
        assert!(!router.covers(&[1.0, 1.0]));
        assert!(router.route(&[1.0, 1.0]).is_empty());
        // Correct width still routes.
        assert_eq!(router.route(&[0.0, 1.0, 0.0]).len(), 1);
    }
}
