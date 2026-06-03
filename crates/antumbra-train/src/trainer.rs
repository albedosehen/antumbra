//! `RaftTrainer` — the [`Trainer`] port realized as the RAFT loop (ADR-0010).
//!
//! It owns the config, a model loader (the candle base+LoRA factory), a corpus
//! resolver, and the verifier (ground truth, ADR-0003). `train_shadow` loads a
//! fresh adapter, resolves the request's tasks, and runs `raft_train`. Only the
//! model loader touches the GPU, so this orchestration is tested with fakes.

use std::sync::Arc;

use async_trait::async_trait;

use antumbra_core::ports::{TrainOutcome, TrainRequest, Trainer, Verifier};
use antumbra_core::{Result, RunId};

use crate::config::RaftConfig;
use crate::model::{Corpus, ModelLoader};
use crate::raft::raft_train;

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
        let mut model = self.loader.load(&req.base_model, None).await?;
        let tasks = self.corpus.tasks(&req.corpus_task_ids);
        let run_id = RunId::new(req.shadow.as_str());
        raft_train(&mut model, self.verifier.as_ref(), &tasks, &run_id, &self.config).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{CausalLm, CorpusTask, SftExample};
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
            vec![CorpusTask {
                id: "t1".into(),
                prompt: "complete the function".into(),
            }]
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
            max_steps: 4,
        };
        let out = trainer.train_shadow(req).await.unwrap();
        assert!(out.final_fitness > 0.0);
        assert!(out.reward_curve.last().unwrap() > out.reward_curve.first().unwrap());
    }
}
