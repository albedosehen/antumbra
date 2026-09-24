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
//! The same goes for the recipe (S-1): the run trains under the one requested,
//! or under the trainer's own when none is, and echoes whichever it used.

use std::sync::Arc;

use async_trait::async_trait;

use antumbra_core::ports::{
    RemeasureRequest, Remeasurement, TrainOutcome, TrainRequest, Trainer, Verifier,
};
use antumbra_core::slice::{Holdout, Slice};
use antumbra_core::{Result, RunId, TrainingRecipe};

use crate::config::RaftConfig;
use crate::grpo::{grpo_train, GrpoModelLoader};
use crate::grpo_eval::EvalLoader;
use crate::holdout::{split, Split};
use crate::model::{Corpus, CorpusTask, ModelLoader};
use crate::raft::raft_train;
use crate::teach::capture_corrections;

/// The recipe a run trains under: the one its request names, or the trainer's
/// own. Checked before anything loads, so an untrainable recipe costs nothing.
fn recipe_for(req: &TrainRequest, config: &RaftConfig) -> Result<TrainingRecipe> {
    let recipe = req.recipe.unwrap_or_else(|| config.recipe());
    recipe.validate()?;
    Ok(recipe)
}

/// The tasks graduation is re-measured on: the held-out slice when the shadow
/// trained under a holdout that holds any out, and otherwise the tasks it
/// trained on. Never the audit slice, which no decision may read, and never an
/// impossible task, which is its own alarm. The flag says which it was.
fn remeasure_slice(
    tasks: Vec<CorpusTask>,
    holdout: Option<&Holdout>,
) -> Result<(Vec<CorpusTask>, bool)> {
    if let Some(h) = holdout {
        let held: Vec<CorpusTask> = tasks
            .iter()
            .filter(|t| !t.impossible && h.partition.of(&t.id) == Slice::HeldOut)
            .cloned()
            .collect();
        if !held.is_empty() {
            return Ok((held, true));
        }
    }
    let Split { learn, .. } = split(tasks, holdout)?;
    Ok((learn, false))
}

/// Re-measure a trained adapter on its slice, once per seed, for any trainer
/// whose loader builds a [`crate::model::CausalLm`].
async fn remeasure_with<L: ModelLoader>(
    loader: &L,
    tasks: Vec<CorpusTask>,
    verifier: &dyn Verifier,
    samples: usize,
    req: RemeasureRequest,
) -> Result<Remeasurement> {
    let (slice, held_out) = remeasure_slice(tasks, req.holdout.as_ref())?;
    let mut model = loader.load(&req.base_model, Some(&req.adapter_uri)).await?;
    let pass_rates = crate::eval::remeasure(
        &mut model,
        verifier,
        &slice,
        &RunId::new(req.shadow.as_str()),
        samples,
        &req.seeds,
    )
    .await?;
    Ok(Remeasurement {
        pass_rates,
        held_out,
        tasks: slice.len(),
    })
}

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
        let recipe = recipe_for(&req, &self.config)?;
        let Split { learn, withheld } = split(
            self.corpus.tasks(&req.corpus_task_ids),
            req.holdout.as_ref(),
        )?;
        let config = self.config.with_recipe(&recipe);
        let mut model = self
            .loader
            .load_trained(
                &req.base_model,
                config.parent_adapter.as_deref(),
                Some(&recipe),
            )
            .await?;
        let run_id = RunId::new(req.shadow.as_str());
        let outcome = raft_train(
            &mut model,
            self.verifier.as_ref(),
            &learn,
            &withheld,
            &run_id,
            &config,
        )
        .await?;
        Ok(TrainOutcome {
            holdout: req.holdout,
            recipe: Some(recipe),
            ..outcome
        })
    }

    async fn remeasure(&self, req: RemeasureRequest) -> Result<Remeasurement> {
        remeasure_with(
            &self.loader,
            self.corpus.tasks(&[]),
            self.verifier.as_ref(),
            self.config.samples_per_task,
            req,
        )
        .await
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
        let recipe = recipe_for(&req, &self.config)?;
        let Split { learn, withheld } = split(
            self.corpus.tasks(&req.corpus_task_ids),
            req.holdout.as_ref(),
        )?;
        let config = self.config.with_recipe(&recipe);
        let mut model = self
            .loader
            .load_trained(
                &req.base_model,
                config.parent_adapter.as_deref(),
                Some(&recipe),
            )
            .await?;
        let run_id = RunId::new(req.shadow.as_str());
        let outcome = grpo_train(
            &mut model,
            self.verifier.as_ref(),
            &learn,
            &withheld,
            &run_id,
            &config,
        )
        .await?;
        Ok(TrainOutcome {
            holdout: req.holdout,
            recipe: Some(recipe),
            ..outcome
        })
    }

    async fn remeasure(&self, req: RemeasureRequest) -> Result<Remeasurement> {
        remeasure_with(
            &EvalLoader(&self.loader),
            self.corpus.tasks(&[]),
            self.verifier.as_ref(),
            self.config.samples_per_task,
            req,
        )
        .await
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
        let recipe = recipe_for(&req, &self.config)?;
        let Split { learn, withheld } = split(
            self.corpus.tasks(&req.corpus_task_ids),
            req.holdout.as_ref(),
        )?;
        let config = self.config.with_recipe(&recipe);
        let mut model = self
            .loader
            .load_trained(
                &req.base_model,
                config.parent_adapter.as_deref(),
                Some(&recipe),
            )
            .await?;
        let run_id = RunId::new(req.shadow.as_str());
        let outcome = capture_corrections(
            &mut model,
            self.verifier.as_ref(),
            &learn,
            &withheld,
            &run_id,
            &config,
            &[],
        )
        .await?;
        Ok(TrainOutcome {
            holdout: req.holdout,
            recipe: Some(recipe),
            ..outcome
        })
    }

    async fn remeasure(&self, req: RemeasureRequest) -> Result<Remeasurement> {
        remeasure_with(
            &self.loader,
            self.corpus.tasks(&[]),
            self.verifier.as_ref(),
            self.config.samples_per_task,
            req,
        )
        .await
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
        async fn load_trained(
            &self,
            _base: &str,
            _parent: Option<&str>,
            _recipe: Option<&TrainingRecipe>,
        ) -> Result<FakeLm> {
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
            recipe: None,
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
        async fn load_trained(
            &self,
            _base: &str,
            _parent: Option<&str>,
            _recipe: Option<&TrainingRecipe>,
        ) -> Result<FakeLm> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(FakeLm {
                skill: AtomicUsize::new(1),
            })
        }
    }

    /// Keeps the recipe each load was asked to build under.
    struct RecordingLoader(Arc<std::sync::Mutex<Vec<Option<TrainingRecipe>>>>);
    #[async_trait]
    impl ModelLoader for RecordingLoader {
        type Model = FakeLm;
        async fn load_trained(
            &self,
            _base: &str,
            _parent: Option<&str>,
            recipe: Option<&TrainingRecipe>,
        ) -> Result<FakeLm> {
            if let Ok(mut seen) = self.0.lock() {
                seen.push(recipe.copied());
            }
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
            recipe: None,
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

    /// Passes every draw whose seed is even; keeps the seeds it was given.
    struct SeededLm {
        seeds: Arc<std::sync::Mutex<Vec<u64>>>,
        next: Option<u64>,
    }

    #[async_trait]
    impl CausalLm for SeededLm {
        async fn generate(&mut self, _prompt: &str, n: usize) -> Result<Vec<String>> {
            Ok((0..n)
                .map(|_| {
                    let draw = self.next.map_or(1, |s| {
                        self.next = Some(s + 1);
                        s
                    });
                    if draw.is_multiple_of(2) {
                        "PASS"
                    } else {
                        "FAIL"
                    }
                    .to_string()
                })
                .collect())
        }
        async fn sft_step(&mut self, _batch: &[SftExample]) -> Result<f32> {
            Ok(0.0)
        }
        fn save_adapter(&self, _path: &str) -> Result<()> {
            Ok(())
        }
        fn seed_draws(&mut self, seed: u64) -> Result<()> {
            if let Ok(mut seeds) = self.seeds.lock() {
                seeds.push(seed);
            }
            self.next = Some(seed);
            Ok(())
        }
    }

    struct SeededLoader(Arc<std::sync::Mutex<Vec<u64>>>);
    #[async_trait]
    impl ModelLoader for SeededLoader {
        type Model = SeededLm;
        async fn load_trained(
            &self,
            _base: &str,
            parent: Option<&str>,
            _recipe: Option<&TrainingRecipe>,
        ) -> Result<SeededLm> {
            assert_eq!(
                parent,
                Some("adapters/winner.safetensors"),
                "re-measures the adapter"
            );
            Ok(SeededLm {
                seeds: self.0.clone(),
                next: None,
            })
        }
    }

    fn remeasure_request(holdout: Option<Holdout>) -> RemeasureRequest {
        RemeasureRequest {
            shadow: ShadowId::new("shadow:winner"),
            base_model: "Qwen/Qwen2.5-Coder-1.5B".into(),
            adapter_uri: "adapters/winner.safetensors".into(),
            holdout,
            seeds: vec![10, 11, 12],
        }
    }

    /// Under a holdout, graduation is measured on the held-out slice only:
    /// not the visible task it trained on, and not the audit task no decision
    /// may read. Each seed is one evaluation, in order.
    #[tokio::test]
    async fn remeasurement_uses_the_held_out_slice_under_each_seed() -> Result<()> {
        let seeds = Arc::new(std::sync::Mutex::new(Vec::new()));
        let trainer = RaftTrainer::new(
            raft(),
            SeededLoader(seeds.clone()),
            IdCorpus(&["task:0", "task:1", "task:19"]),
            passing(),
        );
        let holdout = Holdout {
            partition: Partition::default(),
            audit: true,
        };
        let m = trainer.remeasure(remeasure_request(Some(holdout))).await?;
        assert!(m.held_out);
        assert_eq!(m.tasks, 1, "task:1 is the one held-out task");
        assert_eq!(*seeds.lock().unwrap(), vec![10, 11, 12]);
        // Four draws per evaluation (`raft()`), from each seed in turn: seed 10
        // draws 10..14 (two even), seed 11 draws 11..15 (two even), and so on.
        assert_eq!(m.pass_rates, vec![0.5, 0.5, 0.5]);
        Ok(())
    }

    #[tokio::test]
    async fn without_a_holdout_the_trained_tasks_are_redrawn() -> Result<()> {
        let seeds = Arc::new(std::sync::Mutex::new(Vec::new()));
        let trainer = RaftTrainer::new(
            raft(),
            SeededLoader(seeds),
            IdCorpus(&["task:0", "task:1", "task:19"]),
            passing(),
        );
        let m = trainer.remeasure(remeasure_request(None)).await?;
        assert!(!m.held_out);
        assert_eq!((m.tasks, m.pass_rates.len()), (3, 3));
        Ok(())
    }

    /// A model that cannot seed its draws is refused rather than measured
    /// three times on one stream.
    #[tokio::test]
    async fn a_model_that_cannot_seed_is_not_remeasured() {
        let trainer = RaftTrainer::new(raft(), FakeLoader, OneTaskCorpus, passing());
        let req = RemeasureRequest {
            adapter_uri: "x".into(),
            ..remeasure_request(None)
        };
        assert!(trainer.remeasure(req).await.is_err());
    }

    fn recipe_request(recipe: Option<TrainingRecipe>) -> TrainRequest {
        TrainRequest {
            shadow: ShadowId::new("shadow:recipe"),
            base_model: "Qwen/Qwen2.5-Coder-1.5B".into(),
            corpus_task_ids: vec![],
            holdout: None,
            recipe,
        }
    }

    /// The requested recipe is the one the model is built under, and the one
    /// the outcome reports.
    #[tokio::test]
    async fn a_requested_recipe_reaches_the_model_and_is_echoed() -> Result<()> {
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let trainer = RaftTrainer::new(
            raft(),
            RecordingLoader(seen.clone()),
            OneTaskCorpus,
            passing(),
        );
        let asked = TrainingRecipe {
            learning_rate: 3e-4,
            batch_size: 4,
            kl_beta: 0.1,
        };
        let out = trainer.train_shadow(recipe_request(Some(asked))).await?;
        assert_eq!(out.recipe, Some(asked));
        assert_eq!(*seen.lock().unwrap(), vec![Some(asked)]);
        Ok(())
    }

    /// With none requested, the run trains under the trainer's own settings,
    /// and says which they were rather than reporting nothing.
    #[tokio::test]
    async fn without_a_request_the_trainers_own_recipe_is_used_and_echoed() -> Result<()> {
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let config = RaftConfig {
            learning_rate: 5e-5,
            batch_size: 2,
            ..raft()
        };
        let own = config.recipe();
        let trainer = RaftTrainer::new(
            config,
            RecordingLoader(seen.clone()),
            OneTaskCorpus,
            passing(),
        );
        let out = trainer.train_shadow(recipe_request(None)).await?;
        assert_eq!(out.recipe, Some(own));
        assert_eq!(*seen.lock().unwrap(), vec![Some(own)]);
        Ok(())
    }

    #[tokio::test]
    async fn an_untrainable_recipe_is_refused_before_the_model_loads() {
        let loads = Arc::new(AtomicUsize::new(0));
        let trainer = RaftTrainer::new(
            raft(),
            CountingLoader(loads.clone()),
            OneTaskCorpus,
            passing(),
        );
        let bad = TrainingRecipe {
            learning_rate: -1.0,
            batch_size: 1,
            kl_beta: 0.0,
        };
        assert!(trainer
            .train_shadow(recipe_request(Some(bad)))
            .await
            .is_err());
        assert_eq!(loads.load(Ordering::SeqCst), 0);
    }
}
