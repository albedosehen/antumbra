//! # antumbra-critic: credit assignment
//!
//! Verifiable signals are the primary reward; the critic is a secondary
//! *densifier* that interpolates per-step credit between verifier checkpoints
//! and can never override a verifier. This crate runs the [`Verifier`] and
//! [`Critic`] ports, tags every signal with its source, and folds them with
//! the core [`fold_step`] discipline.

pub mod governed;
pub mod trust;
pub mod verifiers;
pub use governed::Governed;
pub use verifiers::CommandVerifier;

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};

use antumbra_core::ports::{ActOutput, Critic, Verifier, VerifyRequest};
use antumbra_core::{fold_step, Result, RewardSignal, RunId};

/// One step's folded reward.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepReward {
    pub step_idx: u32,
    pub folded: f32,
}

/// The outcome of assessing a trace: every source-tagged signal, the per-step
/// folded rewards, and a single scalar for fitness: the weakest step.
#[derive(Debug, Clone)]
pub struct Critique {
    pub signals: Vec<RewardSignal>,
    pub steps: Vec<StepReward>,
    pub total: f32,
}

/// Run every verifier against every request, emitting primary (trusted)
/// signals. Verifiers are ground truth.
pub async fn run_verifiers(
    run_id: &RunId,
    verifiers: &[&dyn Verifier],
    requests: &[VerifyRequest],
    now: DateTime<Utc>,
) -> Result<Vec<RewardSignal>> {
    let mut out = Vec::new();
    for req in requests {
        for verifier in verifiers {
            let verdict = verifier.verify(req).await?;
            // pass/fail is authoritative (the environment is the truth): a failed verdict contributes
            // zero reward regardless of any partial `value` it reports, so a
            // not-quite-passing step can never be rescued into a verified win.
            let value = if verdict.passed { verdict.value } else { 0.0 };
            out.push(RewardSignal::verifier(
                run_id.clone(),
                req.step_idx,
                req.dimension.clone(),
                value,
                now,
            ));
        }
    }
    Ok(out)
}

/// Run the densifier over a trace, emitting secondary (critic) signals.
pub async fn run_critic(
    run_id: &RunId,
    critic: &dyn Critic,
    output: &ActOutput,
    now: DateTime<Utc>,
) -> Result<Vec<RewardSignal>> {
    let scores = critic.densify(output).await?;
    Ok(scores
        .into_iter()
        .map(|s| RewardSignal::critic(run_id.clone(), s.step_idx, s.dimension, s.value, now))
        .collect())
}

/// Fold source-tagged signals per step. Ground truth bounds the densifier
/// (see [`fold_step`]).
pub fn aggregate(signals: &[RewardSignal], critic_weight: f32) -> Critique {
    let mut by_step: BTreeMap<u32, Vec<RewardSignal>> = BTreeMap::new();
    for signal in signals {
        by_step
            .entry(signal.step_idx)
            .or_default()
            .push(signal.clone());
    }
    let steps: Vec<StepReward> = by_step
        .iter()
        .map(|(step, sigs)| StepReward {
            step_idx: *step,
            folded: fold_step(sigs, critic_weight),
        })
        .collect();
    // The weakest step, not the mean or the sum: a sum pays for verbose
    // vacuous steps.
    let total = steps
        .iter()
        .map(|s| s.folded)
        .reduce(f32::min)
        .unwrap_or(0.0);
    Critique {
        signals: signals.to_vec(),
        steps,
        total,
    }
}

/// Full assessment: verifiers (primary) plus an optional critic (densifier),
/// folded into a [`Critique`].
#[allow(clippy::too_many_arguments)]
pub async fn assess(
    run_id: &RunId,
    output: &ActOutput,
    requests: &[VerifyRequest],
    verifiers: &[&dyn Verifier],
    critic: Option<&dyn Critic>,
    critic_weight: f32,
    now: DateTime<Utc>,
) -> Result<Critique> {
    let mut signals = run_verifiers(run_id, verifiers, requests, now).await?;
    if let Some(critic) = critic {
        signals.extend(run_critic(run_id, critic, output, now).await?);
    }
    Ok(aggregate(&signals, critic_weight))
}

#[cfg(test)]
mod tests {
    use super::*;
    use antumbra_core::ports::StepOutput;
    use antumbra_core::testing::{FlatCritic, MarkerVerifier};

    fn output() -> ActOutput {
        ActOutput {
            steps: vec![StepOutput {
                step_idx: 0,
                content: "ok".into(),
            }],
            final_output: "ok".into(),
        }
    }

    fn request(marker: &str) -> VerifyRequest {
        VerifyRequest {
            run_id: RunId::new("run:1"),
            step_idx: 0,
            dimension: "tests".into(),
            artifact: serde_json::json!({ "marker": marker }),
        }
    }

    #[tokio::test]
    async fn verifier_pass_yields_full_credit() {
        let verifier = MarkerVerifier {
            expect: "ok".into(),
        };
        let run = RunId::new("run:1");
        let critique = assess(
            &run,
            &output(),
            &[request("ok")],
            &[&verifier],
            None,
            0.5,
            Utc::now(),
        )
        .await
        .unwrap();
        assert_eq!(critique.total, 1.0);
        assert!(critique.signals.iter().all(RewardSignal::is_verifier));
    }

    // A misbehaving verifier that reports partial credit on a FAILED step. The
    // critic must still score it zero -- `passed` is authoritative, so the
    // `value` of a failed verdict can never leak into the reward (the environment is the truth).
    struct PartialCreditOnFail;
    #[async_trait::async_trait]
    impl antumbra_core::ports::Verifier for PartialCreditOnFail {
        async fn verify(
            &self,
            _req: &VerifyRequest,
        ) -> antumbra_core::Result<antumbra_core::ports::VerifierVerdict> {
            Ok(antumbra_core::ports::VerifierVerdict {
                passed: false,
                value: 0.3,
            })
        }
    }

    #[tokio::test]
    async fn a_failed_verdict_contributes_zero_despite_a_nonzero_value() {
        let signals = run_verifiers(
            &RunId::new("run:1"),
            &[&PartialCreditOnFail],
            &[request("anything")],
            Utc::now(),
        )
        .await
        .unwrap();
        assert_eq!(signals.len(), 1);
        assert_eq!(
            signals[0].value, 0.0,
            "a failed verdict's partial value is dropped"
        );
    }

    #[tokio::test]
    async fn critic_cannot_rescue_a_verified_failure() {
        let verifier = MarkerVerifier {
            expect: "ok".into(),
        };
        let critic = FlatCritic { value: 1.0 };
        let run = RunId::new("run:1");
        // verifier sees the wrong marker -> fails; glowing critic must not lift it.
        let critique = assess(
            &run,
            &output(),
            &[request("wrong")],
            &[&verifier],
            Some(&critic),
            0.9,
            Utc::now(),
        )
        .await
        .unwrap();
        assert_eq!(critique.total, 0.0);
    }

    #[test]
    fn a_trace_totals_to_its_weakest_step_not_its_mean() {
        let run = RunId::new("run:1");
        let now = Utc::now();
        let signals = vec![
            RewardSignal::verifier(run.clone(), 0, "tests", 1.0, now),
            RewardSignal::verifier(run.clone(), 1, "tests", 0.2, now),
            RewardSignal::verifier(run, 2, "tests", 0.9, now),
        ];
        assert_eq!(aggregate(&signals, 0.5).total, 0.2);
        assert_eq!(aggregate(&[], 0.5).total, 0.0);
    }
}
