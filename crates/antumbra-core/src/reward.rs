//! Reward signals: verifiable-first, critic-densified. ADR-0003.
//!
//! The discipline of ADR-0003 is encoded in the type: every signal is tagged
//! with its [`RewardSource`], verifiers are primary and trusted, and the critic
//! can only *densify* between verifier checkpoints; it can never override one.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::ids::RunId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RewardSource {
    /// Ground truth: a test passed, a schema matched, an exec check succeeded.
    Verifier,
    /// PRM-style interpolation between verifier checkpoints. Never authoritative.
    Critic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardSignal {
    pub run_id: RunId,
    pub step_idx: u32,
    /// What was measured: `tests`, `schema`, `exec`, `critic`, ...
    pub dimension: String,
    pub value: f32,
    pub source: RewardSource,
    pub created_at: DateTime<Utc>,
}

impl RewardSignal {
    pub fn verifier(
        run_id: RunId,
        step_idx: u32,
        dimension: impl Into<String>,
        value: f32,
        now: DateTime<Utc>,
    ) -> Self {
        Self::new(
            run_id,
            step_idx,
            dimension,
            value,
            RewardSource::Verifier,
            now,
        )
    }

    pub fn critic(
        run_id: RunId,
        step_idx: u32,
        dimension: impl Into<String>,
        value: f32,
        now: DateTime<Utc>,
    ) -> Self {
        Self::new(
            run_id,
            step_idx,
            dimension,
            value,
            RewardSource::Critic,
            now,
        )
    }

    pub fn new(
        run_id: RunId,
        step_idx: u32,
        dimension: impl Into<String>,
        value: f32,
        source: RewardSource,
        now: DateTime<Utc>,
    ) -> Self {
        RewardSignal {
            run_id,
            step_idx,
            dimension: dimension.into(),
            value,
            source,
            created_at: now,
        }
    }

    pub fn is_verifier(&self) -> bool {
        self.source == RewardSource::Verifier
    }
}

/// Fold a step's source-tagged signals into one scalar, honoring ADR-0003:
/// if any verifier reading exists it bounds the result, and the critic may only
/// move the value *within* the verifier-trusted envelope, never past it.
///
/// Concretely: the verifier mean is the anchor; the critic mean is blended in at
/// `critic_weight`, then clamped to never exceed the verifier anchor when the
/// verifier failed the step (a critic cannot rescue a verified failure).
pub fn fold_step(signals: &[RewardSignal], critic_weight: f32) -> f32 {
    let (mut vsum, mut vn) = (0.0f32, 0u32);
    let (mut csum, mut cn) = (0.0f32, 0u32);
    for s in signals {
        match s.source {
            RewardSource::Verifier => {
                vsum += s.value;
                vn += 1;
            }
            RewardSource::Critic => {
                csum += s.value;
                cn += 1;
            }
        }
    }

    match (vn, cn) {
        (0, 0) => 0.0,
        (0, _) => csum / cn as f32, // no verifier this step: pure densifier
        (_, 0) => vsum / vn as f32, // verifier only: ground truth
        (_, _) => {
            let v = vsum / vn as f32;
            let c = csum / cn as f32;
            let blended = v * (1.0 - critic_weight) + c * critic_weight;
            // The critic cannot push a verified-failed step above its verifier
            // anchor: ground truth bounds the densifier.
            if v <= 0.0 {
                blended.min(v.max(0.0))
            } else {
                blended
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(source: RewardSource, value: f32) -> RewardSignal {
        RewardSignal::new(RunId::new("run:1"), 0, "x", value, source, Utc::now())
    }

    #[test]
    fn verifier_only_is_ground_truth() {
        let signals = vec![
            r(RewardSource::Verifier, 1.0),
            r(RewardSource::Verifier, 0.0),
        ];
        assert_eq!(fold_step(&signals, 0.5), 0.5);
    }

    #[test]
    fn critic_cannot_rescue_a_verified_failure() {
        // verifier failed (0.0); a glowing critic must not lift the step above 0.
        let signals = vec![r(RewardSource::Verifier, 0.0), r(RewardSource::Critic, 1.0)];
        assert_eq!(fold_step(&signals, 0.9), 0.0);
    }

    #[test]
    fn critic_densifies_within_a_passing_step() {
        let signals = vec![r(RewardSource::Verifier, 1.0), r(RewardSource::Critic, 0.5)];
        let folded = fold_step(&signals, 0.5);
        assert!((folded - 0.75).abs() < 1e-6);
    }

    #[test]
    fn pure_critic_when_no_verifier() {
        let signals = vec![r(RewardSource::Critic, 0.4), r(RewardSource::Critic, 0.6)];
        assert!((fold_step(&signals, 0.5) - 0.5).abs() < 1e-6);
    }
}
