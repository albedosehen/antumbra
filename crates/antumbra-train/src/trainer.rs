//! `RaftTrainer`: the [`Trainer`] port realized as the RAFT loop.
//!
//! It owns the config, a model loader (the candle base+LoRA factory), a corpus
//! resolver, and the verifier (ground truth: the environment is the truth). `train_shadow` loads a
//! fresh adapter, resolves the request's tasks, and runs `raft_train`. Only the
//! model loader touches the GPU, so this orchestration is tested with fakes.
//!
//! Each realization is also where a request's holdout is enforced (ADR-0022):
//! the corpus is split into what the run learns from and what it only
//! measures before the model is loaded, and the outcome echoes the holdout so
//! the loop can tell a measured generation from one that was merely labelled.

use std::sync::Arc;

use async_trait::async_trait;

use antumbra_core::ports::{TrainOutcome, TrainRequest, Trainer, Verifier};
use antumbra_core::{Result, RunId};

use crate::config::RaftConfig;
use crate::grpo::{grpo_train, GrpoModelLoader};
use crate::holdout::{split, Split};
use crate::model::{Corpus, ModelLoader};
use crate::raft::raft_train;
use crate::teach::capture_corrections;

pub struct RaftTrainer<L: ModelLoader, C: Corpus> {
    config: RaftConfig,
    loader: L,
    corpus: C,
    verifier: Arc<dyn Verifier>,
}

impl<L: ModelLoader, C: Corpus> RaftTrainer<L, C> {
    pub fn new(config: RaftConfig, loader: L, corpus: C, verifier: Arc<dyn Verifier>) -> Self {
        Self {
            config,
            loader,
            corpus,
            verifier,
        }
    }
}

#[async_trait]
impl<L: ModelLoader, C: Corpus> Trainer for RaftTrainer<L, C> {
    async fn train_shadow(&self, req: TrainRequest) -> Result<TrainOutcome> {
        let Split { learn, withheld } = split(
            self.corpus.tasks(&req.corpus_task_ids),
            req.holdout.as_ref(),
        )?;
        let mut model = self
            .loader
            .load(&req.base_model, self.config.parent_adapter.as_deref())
            .await?;
        let run_id = RunId::new(req.shadow.as_str());
        let outcome = raft_train(
            &mut model,
            self.verifier.as_ref(),
            &learn,
            &withheld,
            &run_id,
            &self.config,
        )
        .await?;
        Ok(TrainOutcome {
            holdout: req.holdout,
            ..outcome
        })
    }
}

/// The [`Trainer`] port realized as the GRPO loop (v1 efficiency); same shape as
/// [`RaftTrainer`] but over a [`GrpoModelLoader`].
pub struct GrpoTrainer<L: GrpoModelLoader, C: Corpus> {
    config: RaftConfig,
    loader: L,
    corpus: C,
    verifier: Arc<dyn Verifier>,
}

impl<L: GrpoModelLoader, C: Corpus> GrpoTrainer<L, C> {
    pub fn new(config: RaftConfig, loader: L, corpus: C, verifier: Arc<dyn Verifier>) -> Self {
        Self {
            config,
            loader,
            corpus,
            verifier,
        }
    }
}

#[async_trait]
impl<L: GrpoModelLoader, C: Corpus> Trainer for GrpoTrainer<L, C> {
    async fn train_shadow(&self, req: TrainRequest) -> Result<TrainOutcome> {
        let Split { learn, withheld } = split(
            self.corpus.tasks(&req.corpus_task_ids),
            req.holdout.as_ref(),
        )?;
        let mut model = self
            .loader
            .load(&req.base_model, self.config.parent_adapter.as_deref())
            .await?;
        let run_id = RunId::new(req.shadow.as_str());
        let outcome = grpo_train(
            &mut model,
            self.verifier.as_ref(),
            &learn,
            &withheld,
            &run_id,
            &self.config,
        )
        .await?;
        Ok(TrainOutcome {
            holdout: req.holdout,
            ..outcome
        })
    }
}

/// The [`Trainer`] port realized as correction capture (capture intake into the composed model): same
/// shape as [`RaftTrainer`], but it internalizes the corpus's supplied,
/// verifier-checked corrections instead of discovering them by sampling.
pub struct CaptureTrainer<L: ModelLoader, C: Corpus> {
    config: RaftConfig,
    loader: L,
    corpus: C,
    verifier: Arc<dyn Verifier>,
}

impl<L: ModelLoader, C: Corpus> CaptureTrainer<L, C> {
    pub fn new(config: RaftConfig, loader: L, corpus: C, verifier: Arc<dyn Verifier>) -> Self {
        Self {
            config,
            loader,
            corpus,
            verifier,
        }
    }
}

#[async_trait]
impl<L: ModelLoader, C: Corpus> Trainer for CaptureTrainer<L, C> {
    async fn train_shadow(&self, req: TrainRequest) -> Result<TrainOutcome> {
        let Split { learn, withheld } = split(
            self.corpus.tasks(&req.corpus_task_ids),
            req.holdout.as_ref(),
        )?;
        let mut model = self
            .loader
            .load(&req.base_model, self.config.parent_adapter.as_deref())
            .await?;
        let run_id = RunId::new(req.shadow.as_str());
        let outcome = capture_corrections(
            &mut model,
            self.verifier.as_ref(),
            &learn,
            &withheld,
            &run_id,
            &self.config,
            &[],
        )
        .await?;
        Ok(TrainOutcome {
            holdout: req.holdout,
            ..outcome
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{CausalLm, CorpusTask, SftExample};
    use antumbra_core::slice::{Holdout, Partition};
    use antumbra_core::testing::MarkerVerifier;
    use antumbra_core::ShadowId;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct FakeLm {
        skill: AtomicUsize,
    }

    #[async_trait]
    impl CausalLm for FakeLm {
        async fn generate(&mut self, _prompt: &str, n: usize) -> Result<Vec<String>> {
            let s = self.skill.load(Ordering::SeqCst).min(n);
            Ok((0..n)
                .map(|i| if i < s { "PASS" } else { "FAIL" }.to_string())
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

    struct FakeLoader;
    #[async_trait]
    impl ModelLoader for FakeLoader {
        type Model = FakeLm;
        async fn load(&self, _base: &str, _parent: Option<&str>) -> Result<FakeLm> {
            Ok(FakeLm {
                skill: AtomicUsize::new(1),
            })
        }
    }

    struct OneTaskCorpus;
    impl Corpus for OneTaskCorpus {
        fn tasks(&self, _ids: &[String]) -> Vec<CorpusTask> {
            vec![CorpusTask::new("t1", "complete the function")]
        }
    }

    #[tokio::test]
    async fn train_shadow_runs_raft_end_to_end() {
        let trainer = RaftTrainer::new(
            RaftConfig {
                samples_per_task: 4,
                rounds: 3,
                ..RaftConfig::default()
            },
            FakeLoader,
            OneTaskCorpus,
            Arc::new(MarkerVerifier {
                expect: "PASS".into(),
            }),
        );
        let req = TrainRequest {
            shadow: ShadowId::new("shadow:g0"),
            base_model: "Qwen/Qwen2.5-Coder-1.5B".into(),
            corpus_task_ids: vec![],
            holdout: None,
        };
        let out = trainer.train_shadow(req).await.unwrap();
        assert!(out.final_fitness > 0.0);
        assert!(out.reward_curve.last().unwrap() > out.reward_curve.first().unwrap());
    }

    /// Counts loads, so a refusal can be shown to come before the model does.
    struct CountingLoader(Arc<AtomicUsize>);
    #[async_trait]
    impl ModelLoader for CountingLoader {
        type Model = FakeLm;
        async fn load(&self, _base: &str, _parent: Option<&str>) -> Result<FakeLm> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(FakeLm {
                skill: AtomicUsize::new(1),
            })
        }
    }

    /// Named for the slices the default partition puts them in (pinned in core).
    struct IdCorpus(&'static [&'static str]);
    impl Corpus for IdCorpus {
        fn tasks(&self, _ids: &[String]) -> Vec<CorpusTask> {
            self.0
                .iter()
                .map(|id| CorpusTask::new(*id, format!("prompt for {id}")))
                .collect()
        }
    }

    fn held_out_request(holdout: Holdout) -> TrainRequest {
        TrainRequest {
            shadow: ShadowId::new("shadow:held"),
            base_model: "Qwen/Qwen2.5-Coder-1.5B".into(),
            corpus_task_ids: vec![],
            holdout: Some(holdout),
        }
    }

    fn raft() -> RaftConfig {
        RaftConfig {
            samples_per_task: 4,
            rounds: 2,
            ..RaftConfig::default()
        }
    }

    fn passing() -> Arc<MarkerVerifier> {
        Arc::new(MarkerVerifier {
            expect: "PASS".into(),
        })
    }

    #[tokio::test]
    async fn train_shadow_enforces_the_holdout_and_says_so() -> Result<()> {
        let holdout = Holdout {
            partition: Partition::default(),
            audit: false,
        };
        let loads = Arc::new(AtomicUsize::new(0));
        let trainer = RaftTrainer::new(
            raft(),
            CountingLoader(loads.clone()),
            IdCorpus(&["task:0", "task:1", "task:19"]),
            passing(),
        );
        let out = trainer.train_shadow(held_out_request(holdout)).await?;
        assert_eq!(out.holdout, Some(holdout), "what was enforced is echoed");
        let reported: Vec<&str> = out.per_task.iter().map(|t| t.task_id.as_str()).collect();
        // The audit task is not due, so it is neither learned nor measured.
        assert_eq!(reported, ["task:0", "task:1"]);
        assert_eq!(loads.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[tokio::test]
    async fn a_corpus_with_nothing_to_learn_is_refused_before_the_model_loads() {
        let loads = Arc::new(AtomicUsize::new(0));
        // Both ids of the shipped arithmetic corpus hash into the held-out slice.
        let trainer = RaftTrainer::new(
            raft(),
            CountingLoader(loads.clone()),
            IdCorpus(&["add", "multiply"]),
            passing(),
        );
        let holdout = Holdout {
            partition: Partition::default(),
            audit: true,
        };
        assert!(trainer
            .train_shadow(held_out_request(holdout))
            .await
            .is_err());
        assert_eq!(loads.load(Ordering::SeqCst), 0, "no base model for nothing");
    }
}
