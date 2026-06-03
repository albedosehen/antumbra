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
    pub fn inhibition_for(&self, candidate_vec: &[f32], radius: f32) -> f32 {
        if !self.is_actionable() {
            return 0.0;
        }
        let Some(ctx) = self.context_vec.as_deref() else {
            return 0.0;
        };
        let sim = crate::expert::cosine_similarity(ctx, candidate_vec);
        if sim < radius {
            0.0
        } else {
            // Scale within the in-scope band by both proximity and confidence.
            let span = (1.0 - radius).max(1e-6);
            ((sim - radius) / span) * self.confidence.clamp(0.0, 1.0)
        }
    }
}

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
}
