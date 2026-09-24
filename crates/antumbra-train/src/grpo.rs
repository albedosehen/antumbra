//! GRPO: group-relative policy optimization, the v1 efficiency upgrade over
//! RAFT. Same sample -> verify loop, but instead of SFT on the winners it takes
//! a clipped policy-gradient step weighted by each sample's *group-relative*
//! advantage, with a KL leash to a frozen reference. Critic-free: the group
//! mean is the baseline.
//!
//! The loss math and the loop are model-agnostic and CPU-tested; the candle
//! policy/reference forwards live behind the [`GrpoLm`] seam (GPU, MT-4).

use async_trait::async_trait;
use candle_core::Tensor;
use candle_nn::ops::log_softmax;
use serde_json::json;

use antumbra_core::ports::{TaskOutcome, TrainOutcome, Verifier, VerifyRequest};
use antumbra_core::{count_grant, AntumbraError, Result, RunId, TrainingRecipe, VerifierGrant};

use crate::config::RaftConfig;
use crate::model::CorpusTask;

/// Group-relative advantage: `(r - mean) / std`. A group with zero spread (all
/// pass or all fail) carries no signal, so its advantages are zero.
pub fn group_advantages(rewards: &[f32]) -> Vec<f32> {
    let n = rewards.len();
    if n == 0 {
        return Vec::new();
    }
    let mean = rewards.iter().sum::<f32>() / n as f32;
    let var = rewards.iter().map(|r| (r - mean).powi(2)).sum::<f32>() / n as f32;
    let std = var.sqrt();
    if std < 1e-6 {
        return vec![0.0; n];
    }
    rewards.iter().map(|r| (r - mean) / std).collect()
}

/// Per-token log-probability of the realized next token: shift, `log_softmax`,
/// gather. `(b, seq, vocab)` + `(b, seq)` -> `(b, seq-1)`. Gradients flow into
/// whatever produced `logits` (the LoRA adapter).
pub fn token_logprobs(logits: &Tensor, input_ids: &Tensor) -> candle_core::Result<Tensor> {
    let (_b, seq, _v) = logits.dims3()?;
    let shift_logits = logits.narrow(1, 0, seq - 1)?;
    let shift_labels = input_ids.narrow(1, 1, seq - 1)?.contiguous()?;
    log_softmax(&shift_logits, 2)?
        .gather(&shift_labels.unsqueeze(2)?, 2)?
        .squeeze(2)
}

/// The GRPO objective as a loss to minimize (negated `J`). All tensors are
/// `(b, t)` over completion tokens except `advantages` which is `(b, 1)` and
/// broadcasts. Only `policy_lp` carries gradient; `old_lp`, `ref_lp`,
/// `advantages`, and `mask` are constants.
pub fn grpo_loss(
    policy_lp: &Tensor,
    old_lp: &Tensor,
    ref_lp: &Tensor,
    advantages: &Tensor,
    mask: &Tensor,
    clip_eps: f64,
    kl_beta: f64,
) -> candle_core::Result<Tensor> {
    let ratio = (policy_lp - old_lp)?.exp()?;
    let clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)?;
    let adv = advantages.broadcast_as(ratio.shape())?;
    let policy_gradient = (&ratio * &adv)?.minimum(&(&clipped * &adv)?)?;

    // Unbiased KL estimator: exp(ref - policy) - (ref - policy) - 1 >= 0.
    let delta = (ref_lp - policy_lp)?;
    let kl = ((delta.exp()? - &delta)? - 1.0)?;

    let per_token = (policy_gradient - (kl * kl_beta)?)?;
    let masked = (per_token * mask)?;
    let tokens = mask.sum_all()?.to_scalar::<f32>()?.max(1.0);
    (masked.sum_all()? / tokens as f64)?.neg()
}

/// One sampled completion plus the policy log-probs at sampling time
/// (`π_old`), captured so the ratio can be formed on the next forward.
#[derive(Debug, Clone)]
pub struct GrpoSample {
    pub completion: String,
    pub tokens: Vec<u32>,
    pub old_logprobs: Vec<f32>,
}

/// A scored group member ready for the update: its tokens, `π_old` and `π_ref`
/// per-token log-probs, and its group-relative advantage.
#[derive(Debug, Clone)]
pub struct GrpoExperience {
    pub tokens: Vec<u32>,
    pub old_logprobs: Vec<f32>,
    pub ref_logprobs: Vec<f32>,
    pub advantage: f32,
}

/// The model seam GRPO drives (cf. [`crate::model::CausalLm`] for RAFT). The
/// candle Qwen + LoRA realization is the GPU piece; the loop is fake-tested.
#[async_trait]
pub trait GrpoLm {
    /// Sample a group of completions, capturing each one's `π_old` log-probs.
    async fn sample_group(&mut self, prompt: &str, group: usize) -> Result<Vec<GrpoSample>>;
    /// Reference (frozen base, LoRA off) per-token log-probs for given tokens.
    async fn reference_logprobs(&mut self, prompt: &str, tokens: &[u32]) -> Result<Vec<f32>>;
    /// One GRPO optimization step over the advantaged group; returns the loss.
    async fn grpo_step(
        &mut self,
        prompt: &str,
        group: &[GrpoExperience],
        cfg: &RaftConfig,
    ) -> Result<f32>;
    fn save_adapter(&self, path: &str) -> Result<()>;

    /// Draw every later sample from `seed` (cf. [`crate::model::CausalLm::seed_draws`]).
    /// The default refuses, so a model that cannot seed says so.
    fn seed_draws(&mut self, seed: u64) -> Result<()> {
        let _ = seed;
        Err(AntumbraError::Unimplemented("seeded draws"))
    }
}

/// Builds a fresh [`GrpoLm`] for a shadow (cf. [`crate::model::ModelLoader`]).
#[async_trait]
pub trait GrpoModelLoader: Send + Sync {
    type Model: GrpoLm + Send;

    /// Build the model to train under `recipe`, or under the loader's own
    /// configuration when `None`. Required, so every loader decides what a
    /// recipe means for it: one that took a recipe and quietly trained under
    /// something else would make the recipe echoed back to the loop a lie.
    async fn load_trained(
        &self,
        base_model: &str,
        parent_adapter: Option<&str>,
        recipe: Option<&TrainingRecipe>,
    ) -> Result<Self::Model>;

    /// Build the model under the loader's own configuration: inference, and any
    /// training that searches nothing.
    async fn load(&self, base_model: &str, parent_adapter: Option<&str>) -> Result<Self::Model> {
        self.load_trained(base_model, parent_adapter, None).await
    }
}

/// Run GRPO for `cfg.rounds` rounds (group size = `cfg.samples_per_task`) and
/// return the trained adapter outcome. The per-round pass-rate is the reward
/// curve, exactly as RAFT, so the loop swaps in behind the trainer.
///
/// `withheld` tasks are measured in the final round and never learned from,
/// under the same contract as [`crate::raft::raft_train`]: no policy step, no
/// fitness, only a per-task result.
pub async fn grpo_train(
    model: &mut (dyn GrpoLm + Send),
    verifier: &dyn Verifier,
    tasks: &[CorpusTask],
    withheld: &[CorpusTask],
    run_id: &RunId,
    cfg: &RaftConfig,
) -> Result<TrainOutcome> {
    let mut reward_curve = Vec::with_capacity(cfg.rounds);
    let mut capability_exemplars: Vec<String> = Vec::new();
    // Per-task results from the final round, for the standing instruments
    // (ADR-0022). The same reading RAFT takes: a task passed when any sample
    // of it verified.
    let mut per_task: Vec<TaskOutcome> = Vec::new();
    // Every pass in a group that steps is reward granted (ADR-0022 S-4).
    let mut granted_by: Vec<VerifierGrant> = Vec::new();

    let last_round = cfg.rounds.saturating_sub(1);
    for round in 0..cfg.rounds {
        let mut solved: Vec<String> = Vec::new();
        let (mut total, mut passed) = (0usize, 0usize);
        let mut round_tasks: Vec<TaskOutcome> = Vec::new();

        let measured: &[CorpusTask] = if round == last_round { withheld } else { &[] };
        let learned = tasks.iter().map(|t| (t, true));
        for (task, learn) in learned.chain(measured.iter().map(|t| (t, false))) {
            let mut task_passed = false;
            let samples = model
                .sample_group(&task.prompt, cfg.samples_per_task)
                .await?;

            let mut rewards = Vec::with_capacity(samples.len());
            for (i, sample) in samples.iter().enumerate() {
                let req = VerifyRequest {
                    run_id: run_id.clone(),
                    step_idx: i as u32,
                    dimension: "exec".into(),
                    artifact: json!({
                        "task": task.id,
                        "completion": sample.completion,
                        "marker": sample.completion,
                        "verify": task.verify,
                    }),
                };
                let won = verifier.verify(&req).await?.passed;
                rewards.push(if won { 1.0 } else { 0.0 });
                task_passed |= won;
                if !learn {
                    continue;
                }
                total += 1;
                if won {
                    passed += 1;
                    if !solved.contains(&task.prompt) {
                        solved.push(task.prompt.clone());
                    }
                }
            }
            // Recorded before any early exit below. A group with no spread (every
            // sample passed, or every one failed) takes no step, but it is still a
            // measured task: leaving it out would drop exactly the tasks the
            // adapter has mastered or cannot do from the per-task results.
            round_tasks.push(TaskOutcome {
                task_id: task.id.clone(),
                passed: task_passed,
                size: task.prompt.chars().count() as u32,
                impossible: task.impossible,
            });
            // A withheld task is measured and nothing more.
            if !learn {
                continue;
            }

            let advantages = group_advantages(&rewards);
            // No spread -> no learning signal; skip the step (never reinforce
            // nothing, mirroring RAFT's empty-winner guard).
            if advantages.iter().all(|a| a.abs() < 1e-6) {
                continue;
            }
            // The group steps, so every pass in it is reward granted.
            for _ in rewards.iter().filter(|&&r| r > 0.0) {
                count_grant(&mut granted_by, &task.verify);
            }

            let mut group = Vec::with_capacity(samples.len());
            for (sample, advantage) in samples.iter().zip(advantages) {
                let ref_logprobs = model
                    .reference_logprobs(&task.prompt, &sample.tokens)
                    .await?;
                group.push(GrpoExperience {
                    tokens: sample.tokens.clone(),
                    old_logprobs: sample.old_logprobs.clone(),
                    ref_logprobs,
                    advantage,
                });
            }
            model.grpo_step(&task.prompt, &group, cfg).await?;
        }

        reward_curve.push(if total == 0 {
            0.0
        } else {
            passed as f32 / total as f32
        });
        capability_exemplars = solved;
        per_task = round_tasks;
    }

    let safe = run_id.as_str().replace([':', '/', '\\'], "_");
    let adapter_uri = format!("{}/{safe}.safetensors", cfg.adapter_dir);
    model.save_adapter(&adapter_uri)?;
    let final_fitness = reward_curve.last().copied().unwrap_or(0.0);

    Ok(TrainOutcome {
        adapter_uri,
        reward_curve,
        final_fitness,
        capability_exemplars,
        per_task,
        boundary_findings: Vec::new(),
        holdout: None,
        recipe: None,
        granted_by,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use antumbra_core::testing::MarkerVerifier;
    use candle_core::{Device, Tensor};
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn advantages_are_group_normalized_and_zero_without_spread() {
        let a = group_advantages(&[1.0, 0.0, 0.0, 0.0]);
        // mean 0.25, so the winner is positive and the losers negative.
        assert!(a[0] > 0.0 && a[1] < 0.0);
        assert!((a.iter().sum::<f32>()).abs() < 1e-5); // zero-mean
        assert_eq!(group_advantages(&[1.0, 1.0, 1.0]), vec![0.0, 0.0, 0.0]);
        assert_eq!(group_advantages(&[0.0, 0.0]), vec![0.0, 0.0]);
    }

    fn t(data: &[f32], dev: &Device) -> Tensor {
        Tensor::from_vec(data.to_vec(), (1, data.len()), dev).unwrap()
    }

    #[test]
    fn grpo_loss_is_negative_advantage_when_on_policy_and_no_kl() {
        let dev = Device::Cpu;
        let lp = t(&[-0.5, -0.5], &dev);
        // policy == old (ratio 1) and policy == ref (KL 0); advantage +1.
        let adv = Tensor::from_vec(vec![1.0f32], (1, 1), &dev).unwrap();
        let mask = t(&[1.0, 1.0], &dev);
        let loss = grpo_loss(&lp, &lp, &lp, &adv, &mask, 0.2, 0.04)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        // -mean(min(1*1, clip*1)) = -1
        assert!((loss + 1.0).abs() < 1e-5, "loss {loss}");
    }

    #[test]
    fn grpo_loss_kl_penalizes_divergence_from_reference() {
        let dev = Device::Cpu;
        let policy = t(&[-0.5, -0.5], &dev);
        let reference = t(&[-1.5, -1.5], &dev); // policy != ref -> KL > 0
        let adv = Tensor::from_vec(vec![0.0f32], (1, 1), &dev).unwrap(); // no PG term
        let mask = t(&[1.0, 1.0], &dev);
        let loss = grpo_loss(&policy, &policy, &reference, &adv, &mask, 0.2, 1.0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        // loss = -(- beta*KL) = beta*KL > 0
        assert!(loss > 0.0, "expected positive KL loss, got {loss}");
    }

    /// A fake whose skill rises with each GRPO step, so the pass-rate climbs.
    struct FakeGrpoLm {
        skill: AtomicUsize,
    }

    #[async_trait]
    impl GrpoLm for FakeGrpoLm {
        async fn sample_group(&mut self, _prompt: &str, group: usize) -> Result<Vec<GrpoSample>> {
            let skill = self.skill.load(Ordering::SeqCst).min(group);
            Ok((0..group)
                .map(|i| GrpoSample {
                    completion: if i < skill { "PASS" } else { "FAIL" }.into(),
                    tokens: vec![1, 2],
                    old_logprobs: vec![-0.1, -0.1],
                })
                .collect())
        }
        async fn reference_logprobs(&mut self, _prompt: &str, tokens: &[u32]) -> Result<Vec<f32>> {
            Ok(vec![-0.2; tokens.len()])
        }
        async fn grpo_step(
            &mut self,
            _prompt: &str,
            _group: &[GrpoExperience],
            _cfg: &RaftConfig,
        ) -> Result<f32> {
            self.skill.fetch_add(1, Ordering::SeqCst);
            Ok(0.0)
        }
        fn save_adapter(&self, _path: &str) -> Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn grpo_lifts_pass_rate_over_rounds() {
        let mut lm = FakeGrpoLm {
            skill: AtomicUsize::new(1),
        };
        let verifier = MarkerVerifier {
            expect: "PASS".into(),
        };
        let tasks = vec![CorpusTask::new("t1", "do the thing")];
        let cfg = RaftConfig {
            samples_per_task: 4,
            rounds: 3,
            ..RaftConfig::default()
        };
        let out = grpo_train(
            &mut lm,
            &verifier,
            &tasks,
            &[],
            &RunId::new("shadow:g0"),
            &cfg,
        )
        .await
        .unwrap();
        assert_eq!(out.reward_curve.len(), 3);
        assert!(out.reward_curve.last().unwrap() > out.reward_curve.first().unwrap());
        assert_eq!(out.capability_exemplars, vec!["do the thing"]);
    }

    /// Answers every prompt the same way each time: all pass for a mastered
    /// prompt, half pass for any other. Records which prompts took a step.
    #[derive(Default)]
    struct Recording {
        mastered: Vec<String>,
        stepped: Vec<String>,
    }

    #[async_trait]
    impl GrpoLm for Recording {
        async fn sample_group(&mut self, prompt: &str, group: usize) -> Result<Vec<GrpoSample>> {
            let all = self.mastered.iter().any(|m| m == prompt);
            Ok((0..group)
                .map(|i| GrpoSample {
                    completion: if all || i % 2 == 0 { "PASS" } else { "FAIL" }.into(),
                    tokens: vec![1, 2],
                    old_logprobs: vec![-0.1, -0.1],
                })
                .collect())
        }
        async fn reference_logprobs(&mut self, _prompt: &str, tokens: &[u32]) -> Result<Vec<f32>> {
            Ok(vec![-0.2; tokens.len()])
        }
        async fn grpo_step(
            &mut self,
            prompt: &str,
            _group: &[GrpoExperience],
            _cfg: &RaftConfig,
        ) -> Result<f32> {
            self.stepped.push(prompt.to_string());
            Ok(0.0)
        }
        fn save_adapter(&self, _path: &str) -> Result<()> {
            Ok(())
        }
    }

    /// Two things the per-task results must not do: lose a task because its
    /// group had no spread, and let a withheld task take a policy step.
    #[tokio::test]
    async fn every_measured_task_is_reported_and_no_withheld_one_is_learned() -> Result<()> {
        let mut lm = Recording {
            mastered: vec!["mastered".into()],
            ..Recording::default()
        };
        let verifier = MarkerVerifier {
            expect: "PASS".into(),
        };
        let tasks = vec![
            CorpusTask::new("seen:mixed", "mixed"),
            CorpusTask::new("seen:mastered", "mastered"),
        ];
        let withheld = vec![CorpusTask::new("held:mixed", "held, mixed")];
        let cfg = RaftConfig {
            samples_per_task: 4,
            rounds: 2,
            ..RaftConfig::default()
        };
        let out = grpo_train(
            &mut lm,
            &verifier,
            &tasks,
            &withheld,
            &RunId::new("shadow:held"),
            &cfg,
        )
        .await?;

        let reported: Vec<&str> = out.per_task.iter().map(|t| t.task_id.as_str()).collect();
        // `seen:mastered` has no spread, so it takes no step; it was still
        // measured, and a mastered task missing from the results would bias
        // every rate computed over them downward.
        assert_eq!(reported, ["seen:mixed", "seen:mastered", "held:mixed"]);
        assert!(
            lm.stepped.iter().all(|p| p == "mixed"),
            "only the learned task with spread steps: {:?}",
            lm.stepped
        );
        // Fitness over the learned tasks: 2 of 4 plus 4 of 4.
        assert_eq!(out.final_fitness, 0.75);
        Ok(())
    }
}
