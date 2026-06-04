//! Capture — internalize a *given, verified* correction (ADR-0004/0009).
//!
//! RAFT (`raft.rs`) discovers a skill from the model's own verified-correct
//! samples. Capture is the other intake path into the same population: a
//! correction supplied from outside — a human's "no, this project uses deno" —
//! is checked by the verifier and, if it holds, fine-tuned into a frozen expert.
//! Discovery and capture are two ways competence enters the brain; both are
//! gated on ground truth, neither trusts unverified teacher text.

use antumbra_core::ports::{TrainOutcome, Verifier, VerifyRequest};
use antumbra_core::{Result, RunId};
use serde_json::json;

use crate::config::RaftConfig;
use crate::eval::eval_pass_rate;
use crate::model::{CausalLm, CorpusTask, SftExample};

fn verify_request(run_id: &RunId, idx: usize, task: &CorpusTask, completion: &str) -> VerifyRequest {
    VerifyRequest {
        run_id: run_id.clone(),
        step_idx: idx as u32,
        dimension: "exec".into(),
        artifact: json!({
            "task": task.id,
            "completion": completion,
            "marker": completion,
            "verify": task.verify,
        }),
    }
}

/// Verify each supplied correction, fine-tune the adapter on the ones that hold
/// for `cfg.rounds` epochs, then measure whether the expert now reproduces them.
/// `final_fitness` is that internalized pass-rate — the honest "did it stick".
pub async fn capture_corrections(
    model: &mut (dyn CausalLm + Send),
    verifier: &dyn Verifier,
    tasks: &[CorpusTask],
    run_id: &RunId,
    cfg: &RaftConfig,
) -> Result<TrainOutcome> {
    // A correction we cannot check is not trusted into the population.
    let mut winners: Vec<SftExample> = Vec::new();
    let mut solved: Vec<String> = Vec::new();
    for (i, task) in tasks.iter().enumerate() {
        let Some(correction) = task.completion.as_deref() else {
            continue;
        };
        let req = verify_request(run_id, i, task, correction);
        if verifier.verify(&req).await?.passed {
            winners.push(SftExample {
                prompt: task.prompt.clone(),
                completion: correction.to_string(),
            });
            if !solved.contains(&task.prompt) {
                solved.push(task.prompt.clone());
            }
        }
    }

    for _ in 0..cfg.rounds.max(1) {
        if !winners.is_empty() {
            model.sft_step(&winners).await?;
        }
    }

    // Honest fitness: does the expert now generate the correction on its own?
    let learned = eval_pass_rate(model, verifier, tasks, run_id, cfg.samples_per_task).await?;

    let safe = run_id.as_str().replace([':', '/', '\\'], "_");
    let adapter_uri = format!("{}/{safe}.safetensors", cfg.adapter_dir);
    model.save_adapter(&adapter_uri)?;

    Ok(TrainOutcome {
        adapter_uri,
        reward_curve: vec![learned.pass_rate],
        final_fitness: learned.pass_rate,
        capability_exemplars: solved,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::CausalLm;
    use antumbra_core::testing::MarkerVerifier;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// Emits the base prior until it is taught, then emits the correction —
    /// so the post-capture eval reflects the internalized behavior.
    struct Learner {
        taught: AtomicBool,
    }

    #[async_trait]
    impl CausalLm for Learner {
        async fn generate(&mut self, _prompt: &str, n: usize) -> Result<Vec<String>> {
            let out = if self.taught.load(Ordering::SeqCst) {
                "deno install"
            } else {
                "npm install"
            };
            Ok(vec![out.to_string(); n])
        }
        async fn sft_step(&mut self, batch: &[SftExample]) -> Result<f32> {
            // Learns only from the supplied (verified) correction.
            if batch.iter().any(|e| e.completion.contains("deno")) {
                self.taught.store(true, Ordering::SeqCst);
            }
            Ok(0.0)
        }
        fn save_adapter(&self, _path: &str) -> Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn captures_and_internalizes_a_verified_correction() {
        let mut lm = Learner {
            taught: AtomicBool::new(false),
        };
        let verifier = MarkerVerifier {
            expect: "deno install".into(),
        };
        let tasks = vec![CorpusTask::new("p", "add a dep").with_completion("deno install")];
        let cfg = RaftConfig {
            rounds: 2,
            samples_per_task: 4,
            ..RaftConfig::default()
        };
        let out = capture_corrections(&mut lm, &verifier, &tasks, &RunId::new("cap:g0"), &cfg)
            .await
            .unwrap();
        // The correction verified, was internalized, and the expert now emits it.
        assert_eq!(out.final_fitness, 1.0);
        assert_eq!(out.capability_exemplars, vec!["add a dep"]);
    }

    #[tokio::test]
    async fn an_unverifiable_correction_is_not_trusted() {
        let mut lm = Learner {
            taught: AtomicBool::new(false),
        };
        // The verifier wants "deno" but the supplied correction is npm -> rejected.
        let verifier = MarkerVerifier {
            expect: "deno install".into(),
        };
        let tasks = vec![CorpusTask::new("p", "add a dep").with_completion("npm install")];
        let cfg = RaftConfig {
            rounds: 2,
            samples_per_task: 4,
            ..RaftConfig::default()
        };
        let out = capture_corrections(&mut lm, &verifier, &tasks, &RunId::new("cap:g1"), &cfg)
            .await
            .unwrap();
        assert_eq!(out.final_fitness, 0.0);
        assert!(out.capability_exemplars.is_empty());
    }
}
