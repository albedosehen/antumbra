//! The counterfactual boundary — the antumbra, the keystone. ADR-0004.
//!
//! A boundary is a *context-scoped conditional*, never a negation of the goal.
//! It holds a behavior fixed and records the region of context where that
//! behavior is correct, the governing features that the scope depends on, the
//! grain of that scope, and the minimal contrastive pair (C incorrect / C'
//! correct) that evidences it.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::ids::{BoundaryId, Generation};

/// Resolution at which a scope applies. Over-generalizing the grain is exactly
/// the false-inhibition failure mode ADR-0004 exists to prevent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Grain {
    File,
    Project,
    Client,
    Session,
    /// Scope spans everything — the most dangerous claim; assert it rarely.
    Global,
}

/// A recovered scope: the behavior held fixed, the governing feature, and the
/// contrastive context pair (C where the behavior is incorrect, C' the nearest
/// context where it is acceptable). The output of counterfactual scope search
/// (`antumbra-boundary`) and the carrier a verified correction surfaces so the
/// loop can promote it to an actionable [`FailureBoundary`]. Lives in core so
/// both the search crate and a [`crate::ports::TrainOutcome`] can name it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundaryFinding {
    pub behavior: String,
    pub governing_feature: String,
    /// C — where the behavior was judged incorrect.
    pub fail_context: serde_json::Value,
    /// C' — the nearest context where it is acceptable.
    pub near_ok_context: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureBoundary {
    pub id: BoundaryId,
    /// The behavior B, held fixed while context is varied (e.g. "run `npm install`").
    pub behavior: String,
    /// The context C where B was judged incorrect.
    pub fail_context: serde_json::Value,
    /// The nearest context C' where B is acceptable, once recovered.
    #[serde(default)]
    pub near_ok_context: Option<serde_json::Value>,
    /// Which context dimensions the scope depends on (file, brand, client, ...).
    #[serde(default)]
    pub governing_features: Vec<String>,
    #[serde(default)]
    pub grain: Option<Grain>,
    /// Embedded context for inhibitory-penalty KNN lookup (ADR-0005/0007).
    #[serde(default)]
    pub context_vec: Option<Vec<f32>>,
    /// Embedded C' (the acceptable context). When present, inhibition is
    /// *relative* — closer to the failure than to C' — which separates
    /// near-identical contexts an absolute radius cannot (ADR-0004).
    #[serde(default)]
    pub ok_context_vec: Option<Vec<f32>>,
    #[serde(default)]
    pub confidence: f32,
    #[serde(default)]
    pub generation: Generation,
    pub created_at: DateTime<Utc>,
}

impl FailureBoundary {
    /// A boundary is only *actionable* once the contrastive pair is closed —
    /// i.e. we have found a C' where the same behavior is acceptable. Until
    /// then it is an open negative, not yet a scope, and must not gate routing
    /// (doing so would risk global, over-generalized inhibition).
    pub fn is_actionable(&self) -> bool {
        self.near_ok_context.is_some() && !self.governing_features.is_empty()
    }

    /// Inhibitory weight to apply to a candidate context, in `[0, 1]`.
    ///
    /// Zero unless the boundary is actionable *and* the candidate is close
    /// enough (in embedded context space) to the known-incorrect region. This
    /// is the "inhibit only within the incorrect scope" rule of ADR-0004 made
    /// concrete: outside the scope the penalty is exactly zero.
    /// True when an expert (by capability vector) covers this boundary's
    /// failure region — it sits closer to the failure context than to C', the
    /// same relative test the inhibition uses. A captured correction that lands
    /// here *resolves* the boundary, so the lifecycle can retire it (ADR-0004).
    pub fn is_covered_by(&self, expert_vec: &[f32]) -> bool {
        let (Some(fail), Some(ok)) = (self.context_vec.as_deref(), self.ok_context_vec.as_deref())
        else {
            return false;
        };
        crate::expert::cosine_similarity(fail, expert_vec)
            > crate::expert::cosine_similarity(ok, expert_vec)
    }

    pub fn inhibition_for(&self, candidate_vec: &[f32], radius: f32) -> f32 {
        if !self.is_actionable() {
            return 0.0;
        }
        let Some(fail) = self.context_vec.as_deref() else {
            return 0.0;
        };
        let conf = self.confidence.clamp(0.0, 1.0);
        let sim_fail = crate::expert::cosine_similarity(fail, candidate_vec);

        match self.ok_context_vec.as_deref() {
            // Relative scope (preferred): fire only when the candidate sits
            // closer to the failure context than to the acceptable one (C').
            // The shared background cancels in the difference, so contexts that
            // differ only slightly (same task, different project) separate by
            // the *sign* of the margin — the gate's top-1-minus-top-2 idea,
            // applied to the boundary (ADR-0004; an absolute radius cannot).
            Some(ok) => {
                let sim_ok = crate::expert::cosine_similarity(ok, candidate_vec);
                let margin = sim_fail - sim_ok;
                if margin <= 0.0 {
                    0.0
                } else {
                    (margin / RELATIVE_SENSITIVITY).clamp(0.0, 1.0) * conf
                }
            }
            // Legacy absolute radius (boundaries with no embedded C').
            None => {
                if sim_fail < radius {
                    0.0
                } else {
                    let span = (1.0 - radius).max(1e-6);
                    ((sim_fail - radius) / span) * conf
                }
            }
        }
    }
}

/// Relative-margin scale at which inhibition saturates. Sentence-embedding
/// cosine compresses near-identical contexts into a narrow band, so the
/// discriminative fail-vs-C' margin is small; this is the calibratable knob
/// that turns that small margin into a usable penalty.
const RELATIVE_SENSITIVITY: f32 = 0.05;

#[cfg(test)]
mod tests {
    use super::*;

    fn boundary(actionable: bool, ctx: Vec<f32>, confidence: f32) -> FailureBoundary {
        FailureBoundary {
            id: BoundaryId::new("boundary:1"),
            behavior: "run `npm install`".into(),
            fail_context: serde_json::json!({"runtime": "deno"}),
            near_ok_context: actionable.then(|| serde_json::json!({"runtime": "node"})),
            governing_features: if actionable {
                vec!["runtime".into()]
            } else {
                vec![]
            },
            grain: Some(Grain::Project),
            context_vec: Some(ctx),
            ok_context_vec: None,
            confidence,
            generation: Generation::ZERO,
            created_at: Utc::now(),
        }
    }

    #[test]
    fn open_negatives_never_inhibit() {
        let b = boundary(false, vec![1.0, 0.0], 1.0);
        assert!(!b.is_actionable());
        assert_eq!(b.inhibition_for(&[1.0, 0.0], 0.5), 0.0);
    }

    #[test]
    fn inhibition_is_zero_outside_scope() {
        let b = boundary(true, vec![1.0, 0.0], 1.0);
        // Orthogonal candidate is far from the incorrect region -> no penalty.
        assert_eq!(b.inhibition_for(&[0.0, 1.0], 0.5), 0.0);
    }

    #[test]
    fn inhibition_grows_inside_scope_scaled_by_confidence() {
        let b = boundary(true, vec![1.0, 0.0], 0.5);
        // Identical context -> max proximity, scaled by confidence 0.5.
        let p = b.inhibition_for(&[1.0, 0.0], 0.0);
        assert!((p - 0.5).abs() < 1e-6);
    }

    /// Relative scope separates near-identical contexts an absolute radius
    /// cannot: with a fail and a C' that are *both* highly similar to two
    /// candidates, inhibition fires for the one nearer the failure and is zero
    /// for the one nearer C' — by the sign of the margin.
    #[test]
    fn relative_scope_fires_only_nearer_the_failure() {
        let mut b = boundary(true, vec![1.0, 0.05, 0.0], 1.0);
        b.ok_context_vec = Some(vec![1.0, 0.0, 0.05]); // C' — same background, tilted

        // Candidate tilted toward the failure axis -> positive margin -> inhibit.
        let near_fail = b.inhibition_for(&[1.0, 0.1, 0.0], 0.5);
        assert!(near_fail > 0.0);
        // Candidate tilted toward C' -> negative margin -> no inhibition, even
        // though its absolute similarity to the failure is high.
        let near_ok = b.inhibition_for(&[1.0, 0.0, 0.1], 0.5);
        assert_eq!(near_ok, 0.0);
    }

    #[test]
    fn covered_by_an_expert_in_the_failure_region() {
        let mut b = boundary(true, vec![1.0, 0.05, 0.0], 1.0);
        b.ok_context_vec = Some(vec![1.0, 0.0, 0.05]); // C'
                                                       // An expert whose capability sits on the failure side resolves it.
        assert!(b.is_covered_by(&[1.0, 0.1, 0.0]));
        // One sitting on the C' side does not.
        assert!(!b.is_covered_by(&[1.0, 0.0, 0.1]));
    }
}
