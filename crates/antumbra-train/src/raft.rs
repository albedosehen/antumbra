//! RAFT: reward-ranked fine-tuning over verified outcomes.
//!
//! Each round: sample `K` completions per task, **verify** each (the
//! verifier is ground truth, the environment is the truth), keep the winners, and SFT the LoRA adapter on
//! them. The model learns from its *own verified-correct* generations, not a
//! teacher's text (plasticity grounded in verification). The per-round pass-rate is the reward curve.

use serde_json::json;

use antumbra_core::ports::{TaskOutcome, TrainOutcome, Verifier, VerifyRequest};
use antumbra_core::{Result, RunId};

use crate::config::RaftConfig;
use crate::model::{CausalLm, CorpusTask, SftExample};

/// Run RAFT for `cfg.rounds` rounds and return the trained adapter outcome.
pub async fn raft_train(
    model: &mut (dyn CausalLm + Send),
    verifier: &dyn Verifier,
    tasks: &[CorpusTask],
    run_id: &RunId,
    cfg: &RaftConfig,
) -> Result<TrainOutcome> {
    let mut reward_curve = Vec::with_capacity(cfg.rounds);
    // The prompts solved in the final round become the expert's capability
    // exemplars: what it provably does, learned from evaluated behavior.
    let mut capability_exemplars: Vec<String> = Vec::new();
    // Per-task results from the final round, kept for the standing instruments
    // (ADR-0022): a generation cannot be sliced into visible, held-out and
    // audit from one aggregate number, and this is where the per-task answer
    // exists. It was already being computed and discarded.
    let mut per_task: Vec<TaskOutcome> = Vec::new();

    for _round in 0..cfg.rounds {
        let mut winners: Vec<SftExample> = Vec::new();
        let mut solved: Vec<String> = Vec::new();
        let (mut total, mut passed) = (0usize, 0usize);
        let mut round_tasks: Vec<TaskOutcome> = Vec::new();

        for task in tasks {
            let samples = model.generate(&task.prompt, cfg.samples_per_task).await?;
            // A task counts as passed when any sample of it verified, which is
            // the same reading `solved` takes: the adapter can do it.
            let mut task_passed = false;
            for (i, sample) in samples.iter().enumerate() {
                total += 1;
                let req = VerifyRequest {
                    run_id: run_id.clone(),
                    step_idx: i as u32,
                    dimension: "exec".into(),
                    artifact: json!({
                        "task": task.id,
                        "completion": sample,
                        "marker": sample,
                        "verify": task.verify,
                    }),
                };
                if verifier.verify(&req).await?.passed {
                    passed += 1;
                    task_passed = true;
                    if !solved.contains(&task.prompt) {
                        solved.push(task.prompt.clone());
                    }
                    winners.push(SftExample {
                        prompt: task.prompt.clone(),
                        completion: sample.clone(),
                    });
                }
            }
            round_tasks.push(TaskOutcome {
                task_id: task.id.clone(),
                passed: task_passed,
                // Prompt length as the size proxy: the corpus declares no size
                // of its own, and the instruments only order by it.
                size: task.prompt.chars().count() as u32,
            });
        }

        reward_curve.push(if total == 0 {
            0.0
        } else {
            passed as f32 / total as f32
        });
        // Keep the latest round's solved set (reflects the trained adapter),
        // and its per-task results for the same reason.
        capability_exemplars = solved;
        per_task = round_tasks;

        // Anti-collapse (shadow plasticity discipline): only train on verified positives; an empty
        // winner set means no update this round (never reinforce nothing).
        if !winners.is_empty() {
            model.sft_step(&winners).await?;
        }
    }

    // Sanitize the run id for a filesystem path (record ids contain ':', which
    // is illegal in Windows filenames).
    let safe = run_id.as_str().replace([':', '/', '\\'], "_");
    let adapter_uri = format!("{}/{safe}.safetensors", cfg.adapter_dir);
    model.save_adapter(&adapter_uri)?;
    let final_fitness = reward_curve.last().copied().unwrap_or(0.0);

    Ok(TrainOutcome {
        adapter_uri,
        reward_curve,
        final_fitness,
        capability_exemplars,
        // RAFT discovers skills, not scopes; boundaries come from the capture path.
        boundary_findings: Vec::new(),
        per_task,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::CausalLm;
    use antumbra_core::testing::MarkerVerifier;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A fake LM whose "skill" rises each time it is trained: it emits `skill`
    /// passing completions out of `n`, so RAFT's pass-rate climbs round on round.
    struct FakeLm {
        skill: AtomicUsize,
    }

    #[async_trait]
    impl CausalLm for FakeLm {
        async fn generate(&mut self, _prompt: &str, n_samples: usize) -> Result<Vec<String>> {
            let skill = self.skill.load(Ordering::SeqCst).min(n_samples);
            Ok((0..n_samples)
                .map(|i| if i < skill { "PASS" } else { "FAIL" }.to_string())
                .collect())
        }

        async fn sft_step(&mut self, _batch: &[SftExample]) -> Result<f32> {
            self.skill.fetch_add(1, Ordering::SeqCst);
            Ok(0.1)
        }

        fn save_adapter(&self, _path: &str) -> Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn raft_lifts_pass_rate_on_verified_winners() {
        let mut lm = FakeLm {
            skill: AtomicUsize::new(1),
        };
        let verifier = MarkerVerifier {
            expect: "PASS".into(),
        };
        let tasks = vec![CorpusTask::new("t1", "complete the function")];
        let cfg = RaftConfig {
            samples_per_task: 4,
            rounds: 3,
            ..RaftConfig::default()
        };

        let out = raft_train(&mut lm, &verifier, &tasks, &RunId::new("shadow:g0"), &cfg)
            .await
            .unwrap();

        assert_eq!(out.reward_curve.len(), 3);
        // pass-rate climbs as the adapter trains on verified winners
        assert!(out.reward_curve.last().unwrap() > out.reward_curve.first().unwrap());
        assert!(out.final_fitness > 0.0);
        assert!(out.adapter_uri.ends_with("shadow_g0.safetensors"));
        // capability is learned from the task it provably solved
        assert_eq!(out.capability_exemplars, vec!["complete the function"]);
    }

    #[tokio::test]
    async fn no_winners_means_no_collapse() {
        // skill 0 -> never passes -> winners always empty -> rate stays 0,
        // and sft_step is never called (we don't reinforce nothing).
        let mut lm = FakeLm {
            skill: AtomicUsize::new(0),
        };
        let verifier = MarkerVerifier {
            expect: "PASS".into(),
        };
        let tasks = vec![CorpusTask::new("t1", "p")];
        let cfg = RaftConfig {
            samples_per_task: 4,
            rounds: 2,
            ..RaftConfig::default()
        };
        let out = raft_train(&mut lm, &verifier, &tasks, &RunId::new("shadow:g1"), &cfg)
            .await
            .unwrap();
        assert_eq!(out.final_fitness, 0.0);
        assert!(out.reward_curve.iter().all(|&r| r == 0.0));
        // nothing solved -> no capability exemplars
        assert!(out.capability_exemplars.is_empty());
    }
}
