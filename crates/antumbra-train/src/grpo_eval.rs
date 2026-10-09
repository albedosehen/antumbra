//! A GRPO model seen as a [`CausalLm`] for evaluation, so a GRPO-trained
//! adapter is re-measured by the same code as a RAFT one rather
//! than by a copy of it. Generation is the group sampler's completions; seeding
//! passes straight through, so a model that cannot seed still refuses. The
//! view evaluates and never trains.

use async_trait::async_trait;

use antumbra_core::{AntumbraError, Result, TrainingRecipe};

use crate::grpo::{GrpoLm, GrpoModelLoader};
use crate::model::{CausalLm, ModelLoader, SftExample};

/// A GRPO model, evaluated as a causal LM.
pub(crate) struct Evaluating<M>(pub M);

#[async_trait]
impl<M: GrpoLm + Send> CausalLm for Evaluating<M> {
    async fn generate(&mut self, prompt: &str, n_samples: usize) -> Result<Vec<String>> {
        Ok(self
            .0
            .sample_group(prompt, n_samples)
            .await?
            .into_iter()
            .map(|s| s.completion)
            .collect())
    }

    async fn sft_step(&mut self, _batch: &[SftExample]) -> Result<f32> {
        Err(AntumbraError::other(
            "a GRPO model seen for evaluation does not train through SFT",
        ))
    }

    fn save_adapter(&self, path: &str) -> Result<()> {
        self.0.save_adapter(path)
    }

    fn seed_draws(&mut self, seed: u64) -> Result<()> {
        self.0.seed_draws(seed)
    }
}

/// Loads GRPO models as [`Evaluating`] ones.
pub(crate) struct EvalLoader<'a, L>(pub &'a L);

#[async_trait]
impl<L: GrpoModelLoader> ModelLoader for EvalLoader<'_, L> {
    type Model = Evaluating<L::Model>;

    async fn load_trained(
        &self,
        base_model: &str,
        parent_adapter: Option<&str>,
        recipe: Option<&TrainingRecipe>,
    ) -> Result<Self::Model> {
        Ok(Evaluating(
            self.0
                .load_trained(base_model, parent_adapter, recipe)
                .await?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RaftConfig;
    use crate::grpo::{GrpoExperience, GrpoSample};

    /// Samples "draw {n}" for the n-th draw of its current seed; seeds only
    /// when `seedable`.
    struct Sampler {
        seedable: bool,
        next: u64,
    }

    #[async_trait]
    impl GrpoLm for Sampler {
        async fn sample_group(&mut self, _prompt: &str, group: usize) -> Result<Vec<GrpoSample>> {
            Ok((0..group)
                .map(|_| {
                    let n = self.next;
                    self.next += 1;
                    GrpoSample {
                        completion: format!("draw {n}"),
                        tokens: Vec::new(),
                        old_logprobs: Vec::new(),
                    }
                })
                .collect())
        }
        async fn reference_logprobs(&mut self, _prompt: &str, _tokens: &[u32]) -> Result<Vec<f32>> {
            Ok(Vec::new())
        }
        async fn grpo_step(
            &mut self,
            _prompt: &str,
            _group: &[GrpoExperience],
            _cfg: &RaftConfig,
        ) -> Result<f32> {
            Ok(0.0)
        }
        fn save_adapter(&self, _path: &str) -> Result<()> {
            Ok(())
        }
        fn seed_draws(&mut self, seed: u64) -> Result<()> {
            if !self.seedable {
                return Err(AntumbraError::Unimplemented("seeded draws"));
            }
            self.next = seed;
            Ok(())
        }
    }

    #[tokio::test]
    async fn generation_is_the_groups_completions_under_its_seed() -> Result<()> {
        let mut lm = Evaluating(Sampler {
            seedable: true,
            next: 0,
        });
        lm.seed_draws(100)?;
        assert_eq!(
            lm.generate("p", 3).await?,
            ["draw 100", "draw 101", "draw 102"]
        );
        lm.seed_draws(100)?;
        assert_eq!(
            lm.generate("p", 1).await?,
            ["draw 100"],
            "the same seed repeats"
        );
        Ok(())
    }

    struct SamplerLoader;

    #[async_trait]
    impl GrpoModelLoader for SamplerLoader {
        type Model = Sampler;
        async fn load_trained(
            &self,
            _base: &str,
            _parent: Option<&str>,
            _recipe: Option<&TrainingRecipe>,
        ) -> Result<Sampler> {
            Ok(Sampler {
                seedable: true,
                next: 0,
            })
        }
    }

    struct TwoTasks;

    impl crate::model::Corpus for TwoTasks {
        fn tasks(&self, _ids: &[String]) -> Vec<crate::model::CorpusTask> {
            vec![
                crate::model::CorpusTask::new("a", "first"),
                crate::model::CorpusTask::new("b", "second"),
            ]
        }
    }

    /// Through the GRPO trainer itself: two tasks of two draws each per seed.
    /// Seed 7 draws 7 to 10, one of which is the passing "draw 7" (1 of 4);
    /// seed 9 draws 9 to 12, none of which is.
    #[tokio::test]
    async fn a_grpo_trainer_remeasures_through_the_view() -> Result<()> {
        use antumbra_core::ports::{RemeasureRequest, Trainer};
        let trainer = crate::trainer::GrpoTrainer::new(
            RaftConfig {
                samples_per_task: 2,
                ..RaftConfig::default()
            },
            SamplerLoader,
            TwoTasks,
            std::sync::Arc::new(antumbra_core::testing::MarkerVerifier {
                expect: "draw 7".into(),
            }),
        );
        let m = trainer
            .remeasure(RemeasureRequest {
                shadow: antumbra_core::ShadowId::new("shadow:grpo"),
                base_model: "base".into(),
                adapter_uri: "adapter".into(),
                holdout: None,
                seeds: vec![7, 9],
            })
            .await?;
        assert_eq!(m.pass_rates, vec![0.25, 0.0]);
        assert_eq!((m.tasks, m.held_out), (2, false));
        Ok(())
    }

    #[tokio::test]
    async fn a_model_that_cannot_seed_still_refuses_and_nothing_trains() {
        let mut lm = Evaluating(Sampler {
            seedable: false,
            next: 0,
        });
        assert!(lm.seed_draws(1).is_err());
        assert!(lm.sft_step(&[]).await.is_err());
    }
}
