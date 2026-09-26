//! Scores the loop has already paid for, kept for the rest of the run.
//!
//! An evaluation is a pure function of the adapter, the task and the seeds: a
//! frozen expert's weights never change (the no-forgetting tripwire checks
//! it), and the base model is frozen by construction. The contribution
//! measurement draws the same seeds every generation, so an unchanged
//! population measures the same. With the baseline on, it also scores every
//! expert on every live task. So from the second generation on, nearly
//! everything it asks for is a score it asked for before, and only a new
//! expert's tasks cost an evaluation. The loop asks through this cache, and
//! the trainer is asked only for what is missing.
//!
//! An adapter is named by its uri, which the loop never writes twice within a
//! run: every shadow, merge and graduate gets a uri of its own. The cache
//! lives as long as the loop, one process, so a generation retrained after a
//! crash starts with nothing cached.

use std::collections::HashMap;
use std::sync::Mutex;

use antumbra_core::ports::{EvaluateRequest, TaskScores, Trainer};
use antumbra_core::Result;

/// What an evaluation of one task depends on.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Key {
    base_model: String,
    adapter_uri: Option<String>,
    task: String,
    seeds: Vec<u64>,
}

impl Key {
    fn of(req: &EvaluateRequest, task: &str) -> Self {
        Key {
            base_model: req.base_model.clone(),
            adapter_uri: req.adapter_uri.clone(),
            task: task.to_string(),
            seeds: req.seeds.clone(),
        }
    }
}

/// How many task evaluations the loop asked for, and how many of them were
/// already known.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Evaluations {
    pub asked: u64,
    pub reused: u64,
}

impl Evaluations {
    /// What was asked since `before`.
    pub fn since(&self, before: Evaluations) -> Evaluations {
        Evaluations {
            asked: self.asked - before.asked,
            reused: self.reused - before.reused,
        }
    }
}

#[derive(Default)]
pub(crate) struct ScoreCache {
    known: Mutex<(HashMap<Key, f32>, Evaluations)>,
}

impl ScoreCache {
    pub(crate) fn counted(&self) -> Evaluations {
        self.known.lock().map(|k| k.1).unwrap_or_default()
    }

    /// `req`'s scores, asking `trainer` only for the tasks not already known.
    pub(crate) async fn evaluate(
        &self,
        trainer: &dyn Trainer,
        req: EvaluateRequest,
    ) -> Result<TaskScores> {
        let mut scores = TaskScores::default();
        let mut missing = Vec::new();
        if let Ok(mut known) = self.known.lock() {
            for task in &req.task_ids {
                match known.0.get(&Key::of(&req, task)) {
                    Some(&score) => {
                        scores.scores.insert(task.clone(), score);
                    }
                    None => missing.push(task.clone()),
                }
            }
            known.1.asked += req.task_ids.len() as u64;
            known.1.reused += (req.task_ids.len() - missing.len()) as u64;
        } else {
            missing.clone_from(&req.task_ids);
        }
        if missing.is_empty() {
            return Ok(scores);
        }
        let fresh = trainer
            .evaluate(EvaluateRequest {
                task_ids: missing,
                ..req.clone()
            })
            .await?;
        if let Ok(mut known) = self.known.lock() {
            for (task, &score) in &fresh.scores {
                known.0.insert(Key::of(&req, task), score);
            }
        }
        scores.scores.extend(fresh.scores);
        Ok(scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antumbra_core::ports::{TrainOutcome, TrainRequest};
    use async_trait::async_trait;

    /// Scores every task 0.5 and remembers what it was asked.
    #[derive(Default)]
    struct Counting {
        asked: Mutex<Vec<Vec<String>>>,
    }

    #[async_trait]
    impl Trainer for Counting {
        async fn train_shadow(&self, _req: TrainRequest) -> Result<TrainOutcome> {
            unreachable!("only evaluation is exercised")
        }

        async fn evaluate(&self, req: EvaluateRequest) -> Result<TaskScores> {
            self.asked.lock().unwrap().push(req.task_ids.clone());
            Ok(TaskScores {
                scores: req
                    .task_ids
                    .iter()
                    .filter(|t| *t != "unscorable")
                    .map(|t| (t.clone(), 0.5))
                    .collect(),
            })
        }
    }

    fn req(adapter: Option<&str>, tasks: &[&str], seeds: &[u64]) -> EvaluateRequest {
        EvaluateRequest {
            label: "test".into(),
            base_model: "base".into(),
            adapter_uri: adapter.map(str::to_string),
            task_ids: tasks.iter().map(|t| t.to_string()).collect(),
            seeds: seeds.to_vec(),
        }
    }

    #[tokio::test]
    async fn only_what_is_not_known_is_asked_for() -> Result<()> {
        let trainer = Counting::default();
        let cache = ScoreCache::default();
        cache
            .evaluate(&trainer, req(Some("a"), &["t1", "t2"], &[1, 2]))
            .await?;
        let again = cache
            .evaluate(&trainer, req(Some("a"), &["t1", "t2", "t3"], &[1, 2]))
            .await?;
        assert_eq!(again.scores.len(), 3);
        assert_eq!(
            *trainer.asked.lock().unwrap(),
            vec![vec!["t1".to_string(), "t2".into()], vec!["t3".into()]]
        );
        assert_eq!(
            cache.counted(),
            Evaluations {
                asked: 5,
                reused: 2
            }
        );
        Ok(())
    }

    #[tokio::test]
    async fn another_adapter_other_seeds_or_the_base_are_other_scores() -> Result<()> {
        let trainer = Counting::default();
        let cache = ScoreCache::default();
        cache
            .evaluate(&trainer, req(Some("a"), &["t1"], &[1]))
            .await?;
        cache
            .evaluate(&trainer, req(Some("b"), &["t1"], &[1]))
            .await?;
        cache
            .evaluate(&trainer, req(Some("a"), &["t1"], &[2]))
            .await?;
        cache.evaluate(&trainer, req(None, &["t1"], &[1])).await?;
        assert_eq!(trainer.asked.lock().unwrap().len(), 4);
        assert_eq!(cache.counted().reused, 0);
        Ok(())
    }

    #[tokio::test]
    async fn a_task_the_trainer_could_not_score_is_asked_for_again() -> Result<()> {
        let trainer = Counting::default();
        let cache = ScoreCache::default();
        let first = cache
            .evaluate(&trainer, req(None, &["unscorable"], &[1]))
            .await?;
        assert!(first.scores.is_empty());
        cache
            .evaluate(&trainer, req(None, &["unscorable"], &[1]))
            .await?;
        assert_eq!(trainer.asked.lock().unwrap().len(), 2);
        Ok(())
    }

    #[test]
    fn counts_since_a_point_are_the_difference() {
        let before = Evaluations {
            asked: 10,
            reused: 4,
        };
        let now = Evaluations {
            asked: 25,
            reused: 16,
        };
        assert_eq!(
            now.since(before),
            Evaluations {
                asked: 15,
                reused: 12
            }
        );
    }
}
