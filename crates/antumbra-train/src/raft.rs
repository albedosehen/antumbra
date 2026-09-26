//! RAFT: reward-ranked fine-tuning over verified outcomes.
//!
//! Each round: sample `K` completions per task, **verify** each (the
//! verifier is ground truth, the environment is the truth), keep the winners, and SFT the LoRA adapter on
//! them. The model learns from its *own verified-correct* generations, not a
//! teacher's text (plasticity grounded in verification). The per-round pass-rate is the reward curve.

use serde_json::json;

use antumbra_core::ports::{TaskOutcome, TrainOutcome, Verifier, VerifyRequest};
use antumbra_core::{count_grant, JudgedSample, Result, RunId, VerifierGrant};

use crate::config::RaftConfig;
use crate::model::{CausalLm, CorpusTask, SftExample};

/// Run RAFT for `cfg.rounds` rounds and return the trained adapter outcome.
///
/// `tasks` are learned from and are the only tasks fitness is computed over.
/// `withheld` are measured in the final round, against the same adapter the
/// learned tasks' final results describe, and never learned from: no winner
/// of theirs reaches an SFT step and no pass of theirs reaches the reward
/// curve. Pass an empty slice when nothing is held out (ADR-0022).
pub async fn raft_train(
    model: &mut (dyn CausalLm + Send),
    verifier: &dyn Verifier,
    tasks: &[CorpusTask],
    withheld: &[CorpusTask],
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
    // Every winner a named verifier passed is reward it granted (ADR-0022 S-4).
    let mut granted_by: Vec<VerifierGrant> = Vec::new();
    // Every verdict a named verifier gave on a learned task, for the loop's
    // recheck against anchored truth.
    let mut judged: Vec<JudgedSample> = Vec::new();

    let last_round = cfg.rounds.saturating_sub(1);
    for round in 0..cfg.rounds {
        let mut winners: Vec<SftExample> = Vec::new();
        let mut solved: Vec<String> = Vec::new();
        let (mut total, mut passed) = (0usize, 0usize);
        let mut round_tasks: Vec<TaskOutcome> = Vec::new();

        // Only the final round's per-task results are kept, so only the final
        // round spends samples on the withheld tasks.
        let measured: &[CorpusTask] = if round == last_round { withheld } else { &[] };
        let learned = tasks.iter().map(|t| (t, true));
        for (task, learn) in learned.chain(measured.iter().map(|t| (t, false))) {
            let samples = model.generate(&task.prompt, cfg.samples_per_task).await?;
            // A task counts as passed when any sample of it verified, which is
            // the same reading `solved` takes: the adapter can do it.
            let mut task_passed = false;
            for (i, sample) in samples.iter().enumerate() {
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
                let verified = verifier.verify(&req).await?.passed;
                task_passed |= verified;
                // A withheld task is measured and nothing more: its passes are
                // not fitness and its winners are not training data.
                if !learn {
                    continue;
                }
                judged.extend(JudgedSample::named(
                    &task.verify,
                    &task.id,
                    sample,
                    verified,
                ));
                total += 1;
                if verified {
                    passed += 1;
                    if !solved.contains(&task.prompt) {
                        solved.push(task.prompt.clone());
                    }
                    winners.push(SftExample {
                        prompt: task.prompt.clone(),
                        completion: sample.clone(),
                    });
                    count_grant(&mut granted_by, &task.verify);
                }
            }
            round_tasks.push(TaskOutcome {
                task_id: task.id.clone(),
                passed: task_passed,
                // Prompt length as the size proxy: the corpus declares no size
                // of its own, and the instruments only order by it.
                size: task.prompt.chars().count() as u32,
                impossible: task.impossible,
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
        // The trainer adapter, which chose the split, says what was enforced.
        holdout: None,
        recipe: None,
        granted_by,
        judged,
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

        let out = raft_train(
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
        // pass-rate climbs as the adapter trains on verified winners
        assert!(out.reward_curve.last().unwrap() > out.reward_curve.first().unwrap());
        assert!(out.final_fitness > 0.0);
        assert!(out.adapter_uri.ends_with("shadow_g0.safetensors"));
        // capability is learned from the task it provably solved
        assert_eq!(out.capability_exemplars, vec!["complete the function"]);
    }

    #[tokio::test]
    async fn every_trained_on_pass_of_a_named_verifier_is_counted_as_its_grant() {
        let mut lm = FakeLm {
            skill: AtomicUsize::new(1),
        };
        let verifier = MarkerVerifier {
            expect: "PASS".into(),
        };
        let named = serde_json::json!({ "verifier": "verifier:a" });
        let tasks = vec![
            CorpusTask::new("t1", "p1").with_verify(named.clone()),
            CorpusTask::new("t2", "p2").with_verify(serde_json::json!({ "program": "x" })),
        ];
        let withheld = vec![CorpusTask::new("t3", "p3").with_verify(named)];
        let cfg = RaftConfig {
            samples_per_task: 2,
            rounds: 2,
            ..RaftConfig::default()
        };
        let out = raft_train(
            &mut lm,
            &verifier,
            &tasks,
            &withheld,
            &RunId::new("r"),
            &cfg,
        )
        .await
        .unwrap();
        // One winner in the first round and two in the second, for t1 only:
        // t2's spec is its own, and t3 is measured, never trained on.
        assert_eq!(
            out.granted_by,
            vec![antumbra_core::VerifierGrant {
                verifier: antumbra_core::VerifierId::new("verifier:a"),
                passes: 3
            }]
        );
        // Every verdict it gave on t1 is kept for the recheck, the failed one
        // too; a winner is a reward.
        assert_eq!(out.judged.len(), 4);
        assert!(out
            .judged
            .iter()
            .all(|j| j.task == "t1" && j.verifier.as_str() == "verifier:a"));
        assert_eq!(out.judged.iter().filter(|j| j.passed).count(), 3);
        assert!(out.judged.iter().all(|j| j.rewarded == j.passed));
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
        let out = raft_train(
            &mut lm,
            &verifier,
            &tasks,
            &[],
            &RunId::new("shadow:g1"),
            &cfg,
        )
        .await
        .unwrap();
        assert_eq!(out.final_fitness, 0.0);
        assert!(out.reward_curve.iter().all(|&r| r == 0.0));
        // nothing solved -> no capability exemplars
        assert!(out.capability_exemplars.is_empty());
    }

    /// Passes every prompt except the ones named, and records what it was
    /// asked and what it was trained on.
    #[derive(Default)]
    struct Recording {
        fails: Vec<String>,
        asked: Vec<String>,
        trained: Vec<String>,
    }

    #[async_trait]
    impl CausalLm for Recording {
        async fn generate(&mut self, prompt: &str, n_samples: usize) -> Result<Vec<String>> {
            self.asked.push(prompt.to_string());
            let answer = if self.fails.iter().any(|f| f == prompt) {
                "FAIL"
            } else {
                "PASS"
            };
            Ok(vec![answer.to_string(); n_samples])
        }

        async fn sft_step(&mut self, batch: &[SftExample]) -> Result<f32> {
            self.trained
                .extend(batch.iter().map(|example| example.prompt.clone()));
            Ok(0.1)
        }

        fn save_adapter(&self, _path: &str) -> Result<()> {
            Ok(())
        }
    }

    /// ADR-0022: a withheld task is measured and never learned from. Its
    /// winners reach no SFT step, its passes reach no fitness, and it costs
    /// samples only in the round whose results are kept.
    #[tokio::test]
    async fn a_withheld_task_is_measured_and_never_learned_from() -> Result<()> {
        let mut lm = Recording {
            fails: vec!["held, failing".into()],
            ..Recording::default()
        };
        let verifier = MarkerVerifier {
            expect: "PASS".into(),
        };
        let tasks = vec![CorpusTask::new("seen", "learn me")];
        let withheld = vec![
            CorpusTask::new("held:pass", "held, passing"),
            CorpusTask::new("held:fail", "held, failing"),
        ];
        let cfg = RaftConfig {
            samples_per_task: 4,
            rounds: 3,
            ..RaftConfig::default()
        };
        let out = raft_train(
            &mut lm,
            &verifier,
            &tasks,
            &withheld,
            &RunId::new("shadow:held"),
            &cfg,
        )
        .await?;

        assert!(
            lm.trained.iter().all(|p| p == "learn me"),
            "a withheld winner reached training: {:?}",
            lm.trained
        );
        // Counting the withheld tasks would make this 8 of 12.
        assert_eq!(
            out.final_fitness, 1.0,
            "fitness reads the learned tasks only"
        );
        assert_eq!(out.capability_exemplars, vec!["learn me"]);
        let passed: Vec<(&str, bool)> = out
            .per_task
            .iter()
            .map(|t| (t.task_id.as_str(), t.passed))
            .collect();
        assert_eq!(
            passed,
            [("seen", true), ("held:pass", true), ("held:fail", false)]
        );
        let asked = |prompt: &str| lm.asked.iter().filter(|p| *p == prompt).count();
        assert_eq!(asked("learn me"), 3);
        assert_eq!(asked("held, passing"), 1, "only the kept round measures it");
        Ok(())
    }
}
