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

use std::sync::Arc;

use antumbra_core::critic::{shaped_advantages, watch};
use antumbra_core::ports::{Critic, TaskOutcome, TrainOutcome, Verifier, VerifyRequest};
use antumbra_core::{
    count_grant, AntumbraError, JudgedSample, Result, RunId, TrainingRecipe, VerifierGrant,
};

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

/// A critic and how far it may shape advantage (ADR-0022 S-2). It reorders
/// samples only inside the parts the verifier made, scaled by how well it
/// tracks the verifier; see [`shaped_advantages`]. Fitness, and so
/// graduation, still reads verifier bits alone.
#[derive(Clone)]
pub struct CriticShaping {
    pub critic: Arc<dyn Critic>,
    pub weight: f32,
    /// A second critic, trained on another seed, that scores the same
    /// answers and shapes nothing: its agreement with the critic is the
    /// instrument the record keeps it for.
    pub twin: Option<Arc<dyn Critic>>,
}

/// Each completion's score from `critic`; `None` when any went unscored.
async fn scored_by(
    critic: &dyn Critic,
    prompt: &str,
    samples: &[GrpoSample],
) -> Result<Option<Vec<f32>>> {
    let mut out = Vec::with_capacity(samples.len());
    for sample in samples {
        match critic.score(prompt, &sample.completion).await? {
            Some(score) => out.push(score),
            None => return Ok(None),
        }
    }
    Ok(Some(out))
}

impl CriticShaping {
    /// Each completion's score; `None` when any went unscored, and then the
    /// group is not shaped at all.
    async fn scores(&self, prompt: &str, samples: &[GrpoSample]) -> Result<Option<Vec<f32>>> {
        scored_by(self.critic.as_ref(), prompt, samples).await
    }
}

/// What the critic scored over a run, for its watch.
#[derive(Default)]
struct Watched {
    scores: Vec<f32>,
    passed: Vec<bool>,
    twin: Vec<f32>,
    /// A group the twin could not score: its agreement is then unread.
    twin_missed: bool,
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
    critic: Option<&CriticShaping>,
) -> Result<TrainOutcome> {
    let mut reward_curve = Vec::with_capacity(cfg.rounds);
    let mut capability_exemplars: Vec<String> = Vec::new();
    // Per-task results from the final round, for the standing instruments
    // (ADR-0022). The same reading RAFT takes: a task passed when any sample
    // of it verified.
    let mut per_task: Vec<TaskOutcome> = Vec::new();
    // Every pass in a group that steps is reward granted (ADR-0022 S-4).
    let mut granted_by: Vec<VerifierGrant> = Vec::new();
    // Every verdict a named verifier gave on a learned task, for the loop's
    // recheck against anchored truth.
    let mut judged: Vec<JudgedSample> = Vec::new();
    let mut watched = Watched::default();

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
            let group_judged = judged.len();
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
                judged.extend(JudgedSample::named(
                    &task.verify,
                    &task.id,
                    &sample.completion,
                    won,
                ));
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

            let mut advantages = group_advantages(&rewards);
            // No spread -> no learning signal; skip the step (never reinforce
            // nothing, mirroring RAFT's empty-winner guard). The critic cannot
            // make a signal the verifier did not.
            if advantages.iter().all(|a| a.abs() < 1e-6) {
                // A group that takes no step rewards nothing it passed.
                for j in &mut judged[group_judged..] {
                    j.rewarded = false;
                }
                continue;
            }
            if let Some(shaping) = critic {
                if let Some(scores) = shaping.scores(&task.prompt, &samples).await? {
                    let bits: Vec<bool> = rewards.iter().map(|&r| r > 0.0).collect();
                    if let Some(twin) = &shaping.twin {
                        match scored_by(twin.as_ref(), &task.prompt, &samples).await? {
                            Some(t) => watched.twin.extend(t),
                            None => watched.twin_missed = true,
                        }
                    }
                    advantages = shaped_advantages(&bits, &scores, shaping.weight);
                    watched.scores.extend(scores);
                    watched.passed.extend(bits);
                }
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
        judged,
        critic_watch: critic.map(|shaping| {
            let twin =
                (shaping.twin.is_some() && !watched.twin_missed).then_some(watched.twin.as_slice());
            watch(&watched.scores, &watched.passed, twin)
        }),
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

    /// Samples four fixed completions and records the advantages each step
    /// was given, and never improves.
    struct RecordingLm {
        stepped: Vec<Vec<f32>>,
    }

    #[async_trait]
    impl GrpoLm for RecordingLm {
        async fn sample_group(&mut self, _prompt: &str, _group: usize) -> Result<Vec<GrpoSample>> {
            Ok(["PASS 3", "PASS 9", "FAIL 1", "FAIL 7"]
                .iter()
                .map(|c| GrpoSample {
                    completion: c.to_string(),
                    tokens: vec![1],
                    old_logprobs: vec![-0.1],
                })
                .collect())
        }
        async fn reference_logprobs(&mut self, _prompt: &str, tokens: &[u32]) -> Result<Vec<f32>> {
            Ok(vec![-0.2; tokens.len()])
        }
        async fn grpo_step(
            &mut self,
            _prompt: &str,
            group: &[GrpoExperience],
            _cfg: &RaftConfig,
        ) -> Result<f32> {
            self.stepped
                .push(group.iter().map(|e| e.advantage).collect());
            Ok(0.0)
        }
        fn save_adapter(&self, _path: &str) -> Result<()> {
            Ok(())
        }
    }

    /// Passes a completion that starts with PASS.
    struct Prefix;

    #[async_trait]
    impl Verifier for Prefix {
        async fn verify(
            &self,
            req: &VerifyRequest,
        ) -> Result<antumbra_core::ports::VerifierVerdict> {
            let passed = req.artifact["completion"]
                .as_str()
                .is_some_and(|c| c.starts_with("PASS"));
            Ok(antumbra_core::ports::VerifierVerdict {
                passed,
                value: if passed { 1.0 } else { 0.0 },
            })
        }
    }

    /// Scores a completion by its last digit, read upside down when `invert`.
    struct Digit {
        invert: bool,
    }

    #[async_trait]
    impl Critic for Digit {
        async fn densify(
            &self,
            output: &antumbra_core::ports::ActOutput,
        ) -> Result<Vec<antumbra_core::ports::CriticScore>> {
            let d = output
                .final_output
                .chars()
                .last()
                .and_then(|c| c.to_digit(10))
                .unwrap_or(0) as f32;
            let pass = output.final_output.starts_with("PASS");
            // Tracks the verifier across the group (every PASS above every FAIL),
            // or the opposite, and orders each part by its digit either way.
            let score = match (self.invert, pass) {
                (false, true) | (true, false) => 10.0 + d,
                _ => d,
            };
            Ok(vec![antumbra_core::ports::CriticScore {
                step_idx: 0,
                dimension: "critic".into(),
                value: score,
            }])
        }
    }

    async fn stepped_with(critic: Option<CriticShaping>) -> (Vec<f32>, TrainOutcome) {
        let mut lm = RecordingLm {
            stepped: Vec::new(),
        };
        let cfg = RaftConfig {
            samples_per_task: 4,
            rounds: 1,
            ..RaftConfig::default()
        };
        let tasks = vec![CorpusTask::new("t1", "p1")];
        let out = grpo_train(
            &mut lm,
            &Prefix,
            &tasks,
            &[],
            &RunId::new("r"),
            &cfg,
            critic.as_ref(),
        )
        .await
        .unwrap();
        (lm.stepped.remove(0), out)
    }

    /// Every answer the critic scored is read against the verifier, and a
    /// twin scoring the same answers is read against the critic.
    #[tokio::test]
    async fn a_critic_is_watched_against_the_verifier_and_its_twin() {
        let (_, plain) = stepped_with(None).await;
        assert_eq!(plain.critic_watch, None);
        let shaping = CriticShaping {
            critic: Arc::new(Digit { invert: false }),
            weight: 1.0,
            twin: Some(Arc::new(Digit { invert: true })),
        };
        let (_, out) = stepped_with(Some(shaping)).await;
        let w = out.critic_watch.expect("watched");
        assert_eq!(w.n, 4);
        assert!(w.correlation.unwrap() > 0.0, "{w:?}");
        // The inverted twin ranks the answers against the critic.
        assert!(w.twin_agreement.unwrap() < 0.0, "{w:?}");
    }

    #[tokio::test]
    async fn a_critic_reorders_only_inside_the_verifiers_parts_and_fitness_ignores_it() {
        let (plain, plain_out) = stepped_with(None).await;
        assert_eq!(plain[0], plain[1]);
        let tracking = CriticShaping {
            critic: Arc::new(Digit { invert: false }),
            weight: 1.0,
            twin: None,
        };
        let (shaped, out) = stepped_with(Some(tracking)).await;
        // "PASS 9" over "PASS 3" and "FAIL 7" over "FAIL 1", and every pass
        // still over every fail.
        assert!(shaped[1] > shaped[0] && shaped[3] > shaped[2], "{shaped:?}");
        assert!(
            shaped[0].min(shaped[1]) > shaped[2].max(shaped[3]),
            "{shaped:?}"
        );
        // Graduation reads verifier bits only.
        assert_eq!(out.reward_curve, plain_out.reward_curve);

        let inverted = CriticShaping {
            critic: Arc::new(Digit { invert: true }),
            weight: 1.0,
            twin: None,
        };
        let (flipped, _) = stepped_with(Some(inverted)).await;
        assert!(
            flipped[1] < flipped[0] && flipped[3] < flipped[2],
            "{flipped:?}"
        );
        assert!(
            flipped[0].min(flipped[1]) > flipped[2].max(flipped[3]),
            "{flipped:?}"
        );
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
            None,
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

    /// A named verifier's verdicts are all kept for the recheck, and a pass is
    /// a reward only when its group stepped.
    #[tokio::test]
    async fn judged_answers_are_kept_and_rewarded_only_when_their_group_steps() -> Result<()> {
        let mut lm = Recording {
            mastered: vec!["mastered".into()],
            ..Recording::default()
        };
        let verifier = MarkerVerifier {
            expect: "PASS".into(),
        };
        let named = serde_json::json!({ "verifier": "verifier:a" });
        let tasks = vec![
            CorpusTask::new("mixed", "mixed").with_verify(named.clone()),
            CorpusTask::new("mastered", "mastered").with_verify(named),
        ];
        let cfg = RaftConfig {
            samples_per_task: 4,
            rounds: 1,
            ..RaftConfig::default()
        };
        let out = grpo_train(
            &mut lm,
            &verifier,
            &tasks,
            &[],
            &RunId::new("shadow:judged"),
            &cfg,
            None,
        )
        .await?;
        let (mixed, mastered): (Vec<&JudgedSample>, Vec<&JudgedSample>) =
            out.judged.iter().partition(|j| j.task == "mixed");
        assert_eq!(mixed.len(), 4);
        assert_eq!(mixed.iter().filter(|j| j.rewarded).count(), 2);
        assert!(mixed.iter().all(|j| j.rewarded == j.passed));
        // Every answer passed, so the group had no spread and took no step.
        assert_eq!(mastered.iter().filter(|j| j.passed).count(), 4);
        assert_eq!(mastered.iter().filter(|j| j.rewarded).count(), 0);
        assert_eq!(out.granted_by[0].passes, 2);
        Ok(())
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
            None,
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
