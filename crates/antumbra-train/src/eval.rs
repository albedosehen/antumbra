//! Eval: an adapter's pass-rate on a corpus, with no training.
//!
//! This is one RAFT measurement round without the SFT step or the save: sample
//! `K` per task, verify each, report the fraction that pass. Used to
//! detect catastrophic forgetting: a monolithic adapter continually fine-tuned
//! on a new skill is re-scored on the old one, and the drop is the forgetting.

use serde_json::json;

use antumbra_core::ports::{Verifier, VerifyRequest};
use antumbra_core::{Result, RunId};

use crate::model::{CausalLm, CorpusTask};

#[derive(Debug, Clone)]
pub struct TaskResult {
    pub id: String,
    pub passed: usize,
    pub total: usize,
}

impl TaskResult {
    pub fn rate(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            self.passed as f32 / self.total as f32
        }
    }
}

/// One sampled completion and its verdict.
#[derive(Debug, Clone, PartialEq)]
pub struct Draw {
    pub task: String,
    pub completion: String,
    pub passed: bool,
}

#[derive(Debug, Clone)]
pub struct EvalOutcome {
    pub pass_rate: f32,
    pub passed: usize,
    pub total: usize,
    /// Per-task pass counts, so a controller can train only the tasks that fail.
    pub per_task: Vec<TaskResult>,
    /// A few raw completions, for eyeballing what the model actually emits.
    pub examples: Vec<String>,
    /// Every completion drawn, with its verdict, in task order.
    pub draws: Vec<Draw>,
}

/// Sample `samples` completions per task, verify each, and return the pass-rate
/// across all `(task, sample)` pairs, the same denominator RAFT reports, so an
/// eval score is directly comparable to a training reward curve.
pub async fn eval_pass_rate(
    model: &mut (dyn CausalLm + Send),
    verifier: &dyn Verifier,
    tasks: &[CorpusTask],
    run_id: &RunId,
    samples: usize,
) -> Result<EvalOutcome> {
    let (mut total, mut passed) = (0usize, 0usize);
    let mut examples: Vec<String> = Vec::new();
    let mut per_task: Vec<TaskResult> = Vec::new();
    let mut all: Vec<Draw> = Vec::new();
    for task in tasks {
        let draws = model.generate(&task.prompt, samples).await?;
        let mut task_passed = 0usize;
        for (i, sample) in draws.iter().enumerate() {
            total += 1;
            if examples.len() < 5 {
                examples.push(sample.clone());
            }
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
            if verified {
                passed += 1;
                task_passed += 1;
            }
            all.push(Draw {
                task: task.id.clone(),
                completion: sample.clone(),
                passed: verified,
            });
        }
        per_task.push(TaskResult {
            id: task.id.clone(),
            passed: task_passed,
            total: draws.len(),
        });
    }
    let pass_rate = if total == 0 {
        0.0
    } else {
        passed as f32 / total as f32
    };
    Ok(EvalOutcome {
        pass_rate,
        passed,
        total,
        per_task,
        examples,
        draws: all,
    })
}

/// Re-measure a trained adapter (ADR-0022 S-1's fourth constraint): one full
/// evaluation per seed, each drawing from its own, so graduation is judged on
/// independent measurements rather than the one noisy number a search ranked
/// by. Returns the pass rate per seed, in seed order. Fails if the model cannot
/// seed its draws, rather than returning repeats of one stream.
pub async fn remeasure(
    model: &mut (dyn CausalLm + Send),
    verifier: &dyn Verifier,
    tasks: &[CorpusTask],
    run_id: &RunId,
    samples: usize,
    seeds: &[u64],
) -> Result<Vec<f32>> {
    let mut rates = Vec::with_capacity(seeds.len());
    for &seed in seeds {
        model.seed_draws(seed)?;
        rates.push(
            eval_pass_rate(model, verifier, tasks, run_id, samples)
                .await?
                .pass_rate,
        );
    }
    Ok(rates)
}

/// The record of one evaluation worth keeping: what was scored, under what
/// budget, and how every task fared. The aggregate hides the thing a corpus
/// is calibrated by -- which tasks the model always passes or never passes,
/// since neither kind teaches it anything -- so the per-task counts are the
/// point of it.
pub fn report(
    corpus: &str,
    base_model: &str,
    adapter: Option<&str>,
    samples: usize,
    max_new_tokens: usize,
    out: &EvalOutcome,
) -> serde_json::Value {
    let tasks: Vec<serde_json::Value> = out
        .per_task
        .iter()
        .map(|t| json!({ "id": t.id, "passed": t.passed, "total": t.total }))
        .collect();
    json!({
        "corpus": corpus,
        "base_model": base_model,
        "adapter": adapter,
        "samples": samples,
        "max_new_tokens": max_new_tokens,
        "pass_rate": out.pass_rate,
        "passed": out.passed,
        "total": out.total,
        "tasks": tasks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::SftExample;
    use antumbra_core::testing::MarkerVerifier;
    use async_trait::async_trait;

    /// Emits `skill` passing draws out of `n`; never trains (eval is read-only).
    struct FixedLm {
        skill: usize,
    }

    #[async_trait]
    impl CausalLm for FixedLm {
        async fn generate(&mut self, _prompt: &str, n: usize) -> Result<Vec<String>> {
            let s = self.skill.min(n);
            Ok((0..n)
                .map(|i| if i < s { "PASS" } else { "FAIL" }.to_string())
                .collect())
        }
        async fn sft_step(&mut self, _batch: &[SftExample]) -> Result<f32> {
            Ok(0.0)
        }
        fn save_adapter(&self, _path: &str) -> Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn pass_rate_is_passed_over_total() {
        let mut lm = FixedLm { skill: 3 };
        let verifier = MarkerVerifier {
            expect: "PASS".into(),
        };
        let tasks = vec![CorpusTask::new("t1", "p1"), CorpusTask::new("t2", "p2")];
        let out = eval_pass_rate(&mut lm, &verifier, &tasks, &RunId::new("eval"), 4)
            .await
            .unwrap();
        // 3 of every 4 draws pass, across both tasks: 6 / 8.
        assert_eq!((out.passed, out.total), (6, 8));
        assert!((out.pass_rate - 0.75).abs() < 1e-6);
    }

    #[tokio::test]
    async fn every_draw_is_kept_with_its_task_and_verdict() {
        let mut lm = FixedLm { skill: 1 };
        let verifier = MarkerVerifier {
            expect: "PASS".into(),
        };
        let tasks = vec![CorpusTask::new("t1", "p1"), CorpusTask::new("t2", "p2")];
        let out = eval_pass_rate(&mut lm, &verifier, &tasks, &RunId::new("eval"), 2)
            .await
            .unwrap();
        let kept: Vec<(&str, &str, bool)> = out
            .draws
            .iter()
            .map(|d| (d.task.as_str(), d.completion.as_str(), d.passed))
            .collect();
        assert_eq!(
            kept,
            [
                ("t1", "PASS", true),
                ("t1", "FAIL", false),
                ("t2", "PASS", true),
                ("t2", "FAIL", false)
            ]
        );
    }

    #[tokio::test]
    async fn the_report_keeps_every_task_and_what_was_scored() {
        let mut lm = FixedLm { skill: 1 };
        let verifier = MarkerVerifier {
            expect: "PASS".into(),
        };
        let tasks = vec![CorpusTask::new("t1", "p1"), CorpusTask::new("t2", "p2")];
        let out = eval_pass_rate(&mut lm, &verifier, &tasks, &RunId::new("eval"), 2)
            .await
            .unwrap();
        let record = report("c.json", "base", None, 2, 256, &out);
        assert_eq!(record["base_model"], "base");
        assert_eq!(record["adapter"], serde_json::Value::Null);
        assert_eq!(record["max_new_tokens"], 256);
        let ids: Vec<&str> = record["tasks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, ["t1", "t2"]);
        assert_eq!(record["tasks"][0]["passed"], 1);
        assert_eq!(record["tasks"][0]["total"], 2);
    }
}
