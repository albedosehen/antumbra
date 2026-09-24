//! What a leave-one-out contribution is measured with (ADR-0022 S-5): the
//! live tasks, and scores for the base model under one adapter, or alone, on a
//! named subset of them. The loop routes and does the arithmetic; the trainer
//! only scores, through the same evaluation re-measurement uses.

use std::collections::{BTreeMap, HashSet};

use antumbra_core::ports::{EvaluateRequest, TaskPrompt, TaskScores, Verifier};
use antumbra_core::slice::Holdout;
use antumbra_core::{Result, RunId};

use crate::eval::eval_pass_rate;
use crate::holdout::{split, Split};
use crate::model::{CausalLm, CorpusTask, ModelLoader};

/// The visible slice under `holdout`, as routing needs it. Impossible tasks
/// are left out, as they are from learning.
pub(crate) fn live_tasks(
    tasks: Vec<CorpusTask>,
    holdout: Option<&Holdout>,
) -> Result<Vec<TaskPrompt>> {
    let Split { learn, .. } = split(tasks, holdout)?;
    Ok(learn
        .into_iter()
        .map(|t| TaskPrompt {
            id: t.id,
            prompt: t.prompt,
        })
        .collect())
}

/// Score the named tasks under the requested adapter, or the base alone,
/// once per seed, each task's pass rate averaged over the seeds.
pub(crate) async fn evaluate_with<L: ModelLoader>(
    loader: &L,
    tasks: Vec<CorpusTask>,
    verifier: &dyn Verifier,
    samples: usize,
    req: EvaluateRequest,
) -> Result<TaskScores> {
    let wanted: HashSet<&str> = req.task_ids.iter().map(String::as_str).collect();
    let subset: Vec<CorpusTask> = tasks
        .into_iter()
        .filter(|t| !t.impossible && wanted.contains(t.id.as_str()))
        .collect();
    if subset.is_empty() || req.seeds.is_empty() {
        return Ok(TaskScores::default());
    }
    let mut model = loader
        .load(&req.base_model, req.adapter_uri.as_deref())
        .await?;
    let run_id = RunId::new(req.label.as_str());
    let mut sums: BTreeMap<String, f32> = BTreeMap::new();
    for &seed in &req.seeds {
        model.seed_draws(seed)?;
        let out = eval_pass_rate(&mut model, verifier, &subset, &run_id, samples).await?;
        for t in out.per_task {
            *sums.entry(t.id).or_default() += t.passed as f32 / t.total.max(1) as f32;
        }
    }
    let n = req.seeds.len() as f32;
    Ok(TaskScores {
        scores: sums.into_iter().map(|(id, s)| (id, s / n)).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::SftExample;
    use antumbra_core::slice::Partition;
    use antumbra_core::testing::MarkerVerifier;
    use antumbra_core::{AntumbraError, TrainingRecipe};
    use async_trait::async_trait;

    /// Answers "right" under the adapter named "good", "wrong" otherwise, and
    /// under "coin" alternates by draw, from a seeded start.
    struct Lm {
        adapter: Option<String>,
        draw: u64,
    }

    #[async_trait]
    impl CausalLm for Lm {
        async fn generate(&mut self, _prompt: &str, n: usize) -> Result<Vec<String>> {
            Ok((0..n)
                .map(|_| {
                    self.draw += 1;
                    let right = match self.adapter.as_deref() {
                        Some("good") => true,
                        Some("coin") => self.draw.is_multiple_of(2),
                        _ => false,
                    };
                    if right { "right" } else { "wrong" }.to_string()
                })
                .collect())
        }
        async fn sft_step(&mut self, _batch: &[SftExample]) -> Result<f32> {
            Err(AntumbraError::other("scoring never trains"))
        }
        fn save_adapter(&self, _path: &str) -> Result<()> {
            Ok(())
        }
        fn seed_draws(&mut self, seed: u64) -> Result<()> {
            self.draw = seed;
            Ok(())
        }
    }

    struct Loader;

    #[async_trait]
    impl ModelLoader for Loader {
        type Model = Lm;
        async fn load_trained(
            &self,
            _base: &str,
            parent: Option<&str>,
            _recipe: Option<&TrainingRecipe>,
        ) -> Result<Lm> {
            Ok(Lm {
                adapter: parent.map(str::to_string),
                draw: 0,
            })
        }
    }

    fn task(id: &str) -> CorpusTask {
        CorpusTask::new(id, format!("prompt for {id}"))
    }

    fn request(adapter: Option<&str>, ids: &[&str], seeds: &[u64]) -> EvaluateRequest {
        EvaluateRequest {
            label: "contribution:test".into(),
            base_model: "base".into(),
            adapter_uri: adapter.map(str::to_string),
            task_ids: ids.iter().map(|s| s.to_string()).collect(),
            seeds: seeds.to_vec(),
        }
    }

    fn verifier() -> MarkerVerifier {
        MarkerVerifier {
            expect: "right".into(),
        }
    }

    #[tokio::test]
    async fn an_adapter_and_the_base_alone_are_scored_per_task() -> Result<()> {
        let tasks = || vec![task("a"), task("b"), task("c")];
        let good = evaluate_with(
            &Loader,
            tasks(),
            &verifier(),
            2,
            request(Some("good"), &["a", "b"], &[1]),
        )
        .await?;
        assert_eq!(good.scores.len(), 2, "only the tasks asked for");
        assert!(good.scores.values().all(|&s| s == 1.0));
        let base = evaluate_with(
            &Loader,
            tasks(),
            &verifier(),
            2,
            request(None, &["c"], &[1]),
        )
        .await?;
        assert_eq!(base.scores.get("c"), Some(&0.0));
        Ok(())
    }

    /// Each task's score is its pass rate averaged over the seeds.
    #[tokio::test]
    async fn a_score_is_averaged_over_the_seeds() -> Result<()> {
        // One draw a seed: seed 0 draws 1 (wrong), seed 1 draws 2 (right),
        // seed 3 draws 4 (right).
        let s = evaluate_with(
            &Loader,
            vec![task("a")],
            &verifier(),
            1,
            request(Some("coin"), &["a"], &[0, 1, 3]),
        )
        .await?;
        let a = s.scores.get("a").copied().unwrap_or(f32::NAN);
        assert!((a - 2.0 / 3.0).abs() < 1e-6, "{a}");
        Ok(())
    }

    #[tokio::test]
    async fn unknown_tasks_or_no_seeds_score_nothing() -> Result<()> {
        let none = evaluate_with(
            &Loader,
            vec![task("a")],
            &verifier(),
            2,
            request(Some("good"), &["zzz"], &[1]),
        )
        .await?;
        assert!(none.scores.is_empty());
        let unseeded = evaluate_with(
            &Loader,
            vec![task("a")],
            &verifier(),
            2,
            request(Some("good"), &["a"], &[]),
        )
        .await?;
        assert!(unseeded.scores.is_empty());
        Ok(())
    }

    /// The live tasks are the visible slice: nothing held out or audited, and
    /// no impossible task.
    #[test]
    fn live_tasks_are_the_visible_slice() -> Result<()> {
        let mut tasks: Vec<CorpusTask> = (0..40).map(|i| task(&format!("t{i}"))).collect();
        let mut impossible = task("never");
        impossible.impossible = true;
        tasks.push(impossible);
        let holdout = Holdout {
            partition: Partition::default(),
            audit: true,
        };
        let live = live_tasks(tasks.clone(), Some(&holdout))?;
        assert!(!live.is_empty() && live.len() < 40);
        assert!(live.iter().all(|t| holdout.learns_from(&t.id)));
        assert!(live.iter().all(|t| t.id != "never"));
        assert_eq!(live_tasks(tasks, None)?.len(), 40);
        Ok(())
    }
}
