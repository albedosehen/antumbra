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
use crate::consolidate::interleave_replay;
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
///
/// `replay` is a rehearsal buffer of already-consolidated `prompt -> behavior`
/// pairs (EXP-021): when `cfg.replay_ratio > 0` it is interleaved into every SFT
/// round so consolidating new memories does not clobber old skills. Plain
/// capture passes an empty buffer (replay off).
pub async fn capture_corrections(
    model: &mut (dyn CausalLm + Send),
    verifier: &dyn Verifier,
    tasks: &[CorpusTask],
    run_id: &RunId,
    cfg: &RaftConfig,
    replay: &[SftExample],
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

    // Nothing verified means nothing to capture: the batch will be empty (replay
    // is scaled by the winner count), so training is a no-op and the saved adapter
    // stays at its untrained init -- a silent failure that looks like graduation
    // succeeded. Surface it, and name the usual Windows cause (the `python`
    // verifier resolving to the Store stub; set ANTUMBRA_PYTHON).
    let with_corrections = tasks.iter().filter(|t| t.completion.is_some()).count();
    if winners.is_empty() && with_corrections > 0 {
        eprintln!(
            "warning: 0 of {with_corrections} correction(s) verified for {} -- nothing \
             captured, the adapter will be UNTRAINED. Is the verifier runnable? (on Windows, \
             set ANTUMBRA_PYTHON to a real interpreter)",
            run_id.as_str()
        );
    }

    // Interleave the rehearsal buffer once; reused each round. With replay off
    // (empty buffer or ratio 0) this is exactly the winners.
    let batch = interleave_replay(&winners, replay, cfg.replay_ratio);
    for _ in 0..cfg.rounds.max(1) {
        if !batch.is_empty() {
            model.sft_step(&batch).await?;
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
        let out = capture_corrections(&mut lm, &verifier, &tasks, &RunId::new("cap:g0"), &cfg, &[])
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
        let out = capture_corrections(&mut lm, &verifier, &tasks, &RunId::new("cap:g1"), &cfg, &[])
            .await
            .unwrap();
        assert_eq!(out.final_fitness, 0.0);
        assert!(out.capability_exemplars.is_empty());
    }

    /// Records every prompt it is fine-tuned on, so a test can assert the
    /// rehearsal buffer actually reached the SFT batch.
    struct Recorder {
        seen: std::sync::Mutex<Vec<String>>,
    }

    #[async_trait]
    impl CausalLm for Recorder {
        async fn generate(&mut self, _prompt: &str, n: usize) -> Result<Vec<String>> {
            Ok(vec!["deno install".to_string(); n])
        }
        async fn sft_step(&mut self, batch: &[SftExample]) -> Result<f32> {
            let mut seen = self.seen.lock().unwrap();
            seen.extend(batch.iter().map(|e| e.prompt.clone()));
            Ok(0.0)
        }
        fn save_adapter(&self, _path: &str) -> Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn replay_buffer_is_rehearsed_during_consolidation() {
        let mut lm = Recorder {
            seen: std::sync::Mutex::new(Vec::new()),
        };
        let verifier = MarkerVerifier {
            expect: "deno install".into(),
        };
        let tasks = vec![CorpusTask::new("p", "new project").with_completion("deno install")];
        // A previously-consolidated skill to rehearse so it is not clobbered.
        let replay = vec![SftExample {
            prompt: "old skill".into(),
            completion: "reverse a string".into(),
        }];
        let cfg = RaftConfig {
            rounds: 1,
            samples_per_task: 2,
            replay_ratio: 1.0,
            ..RaftConfig::default()
        };
        capture_corrections(&mut lm, &verifier, &tasks, &RunId::new("cons:g0"), &cfg, &replay)
            .await
            .unwrap();
        let seen = lm.seen.lock().unwrap();
        // Both the new memory and the rehearsed old skill were trained on.
        assert!(seen.iter().any(|p| p == "new project"));
        assert!(seen.iter().any(|p| p == "old skill"));
    }
}
