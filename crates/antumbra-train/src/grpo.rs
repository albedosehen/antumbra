//! GRPO — group-relative policy optimization (ADR-0011), the v1 upgrade over
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

use antumbra_core::ports::{TrainOutcome, Verifier, VerifyRequest};
use antumbra_core::{Result, RunId};

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
}

/// Builds a fresh [`GrpoLm`] for a shadow (cf. [`crate::model::ModelLoader`]).
#[async_trait]
pub trait GrpoModelLoader: Send + Sync {
    type Model: GrpoLm + Send;
    async fn load(&self, base_model: &str, parent_adapter: Option<&str>) -> Result<Self::Model>;
}

/// Run GRPO for `cfg.rounds` rounds (group size = `cfg.samples_per_task`) and
/// return the trained adapter outcome. The per-round pass-rate is the reward
/// curve, exactly as RAFT (ADR-0010), so the loop swaps in behind the trainer.
pub async fn grpo_train(
    model: &mut (dyn GrpoLm + Send),
    verifier: &dyn Verifier,
    tasks: &[CorpusTask],
    run_id: &RunId,
    cfg: &RaftConfig,
) -> Result<TrainOutcome> {
    let mut reward_curve = Vec::with_capacity(cfg.rounds);
    let mut capability_exemplars: Vec<String> = Vec::new();

    for _round in 0..cfg.rounds {
        let mut solved: Vec<String> = Vec::new();
        let (mut total, mut passed) = (0usize, 0usize);

        for task in tasks {
            let samples = model
                .sample_group(&task.prompt, cfg.samples_per_task)
                .await?;

            let mut rewards = Vec::with_capacity(samples.len());
            for (i, sample) in samples.iter().enumerate() {
                total += 1;
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
                if won {
                    passed += 1;
                    if !solved.contains(&task.prompt) {
                        solved.push(task.prompt.clone());
                    }
                }
            }

            let advantages = group_advantages(&rewards);
            // No spread -> no learning signal; skip the step (never reinforce
            // nothing, mirroring RAFT's empty-winner guard).
            if advantages.iter().all(|a| a.abs() < 1e-6) {
                continue;
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
        boundary_findings: Vec::new(),
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
        let out = grpo_train(&mut lm, &verifier, &tasks, &RunId::new("shadow:g0"), &cfg)
            .await
            .unwrap();
        assert_eq!(out.reward_curve.len(), 3);
        assert!(out.reward_curve.last().unwrap() > out.reward_curve.first().unwrap());
        assert_eq!(out.capability_exemplars, vec!["do the thing"]);
    }
}
