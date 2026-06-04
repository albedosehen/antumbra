//! Eval — an adapter's pass-rate on a corpus, with no training (ADR-0010).
//!
//! This is one RAFT measurement round without the SFT step or the save: sample
//! `K` per task, verify each, report the fraction that pass. EXP-010 uses it to
//! detect catastrophic forgetting — a monolithic adapter continually fine-tuned
//! on a new skill is re-scored on the old one, and the drop is the forgetting.

use serde_json::json;

use antumbra_core::ports::{Verifier, VerifyRequest};
use antumbra_core::{Result, RunId};

use crate::model::{CausalLm, CorpusTask};

#[derive(Debug, Clone)]
pub struct EvalOutcome {
    pub pass_rate: f32,
    pub passed: usize,
    pub total: usize,
    /// A few raw completions, for eyeballing what the model actually emits.
    pub examples: Vec<String>,
}

/// Sample `samples` completions per task, verify each, and return the pass-rate
/// across all `(task, sample)` pairs — the same denominator RAFT reports, so an
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
    for task in tasks {
        let draws = model.generate(&task.prompt, samples).await?;
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
            if verifier.verify(&req).await?.passed {
                passed += 1;
            }
        }
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
        examples,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use crate::model::SftExample;
    use antumbra_core::testing::MarkerVerifier;

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
}
