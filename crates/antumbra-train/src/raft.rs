//! RAFT — reward-ranked fine-tuning over verified outcomes (ADR-0010).
//!
//! Each round: sample `K` completions per task, **verify** each (the ADR-0003
//! verifier is ground truth), keep the winners, and SFT the LoRA adapter on
//! them. The model learns from its *own verified-correct* generations — not a
//! teacher's text (ADR-0002/0003). The per-round pass-rate is the reward curve.

use serde_json::json;

use antumbra_core::ports::{TrainOutcome, VerifyRequest, Verifier};
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

    for _round in 0..cfg.rounds {
        let mut winners: Vec<SftExample> = Vec::new();
        let (mut total, mut passed) = (0usize, 0usize);

        for task in tasks {
            let samples = model.generate(&task.prompt, cfg.samples_per_task).await?;
            for (i, sample) in samples.iter().enumerate() {
                total += 1;
                let req = VerifyRequest {
                    run_id: run_id.clone(),
                    step_idx: i as u32,
                    dimension: "exec".into(),
                    artifact: json!({ "task": task.id, "completion": sample, "marker": sample }),
                };
                if verifier.verify(&req).await?.passed {
                    passed += 1;
                    winners.push(SftExample {
                        prompt: task.prompt.clone(),
                        completion: sample.clone(),
                    });
                }
            }
        }

        reward_curve.push(if total == 0 {
            0.0
        } else {
            passed as f32 / total as f32
        });

        // Anti-collapse (ADR-0002): only train on verified positives; an empty
        // winner set means no update this round (never reinforce nothing).
        if !winners.is_empty() {
            model.sft_step(&winners).await?;
        }
    }

    let adapter_uri = format!("{}/{run_id}.safetensors", cfg.adapter_dir);
    model.save_adapter(&adapter_uri)?;
    let final_fitness = reward_curve.last().copied().unwrap_or(0.0);

    Ok(TrainOutcome {
        adapter_uri,
        reward_curve,
        final_fitness,
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
        async fn generate(&self, _prompt: &str, n_samples: usize) -> Result<Vec<String>> {
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
        let tasks = vec![CorpusTask {
            id: "t1".into(),
            prompt: "complete the function".into(),
        }];
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
        assert!(out.adapter_uri.ends_with("shadow:g0.safetensors"));
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
        let tasks = vec![CorpusTask {
            id: "t1".into(),
            prompt: "p".into(),
        }];
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
    }
}
