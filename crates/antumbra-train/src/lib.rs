//! # antumbra-train (shadow models hold the plasticity; the candle QLoRA trainer)
//!
//! The DIY `candle` path that trains shadow adapters from **verified outcomes**
//! via RAFT-style reward-ranked LoRA fine-tuning.
//!
//! Built bottom-up and tested on CPU: the LoRA training primitive ([`lora`]),
//! the SFT objective ([`objective`]), the RAFT loop ([`raft`]), and the
//! [`Trainer`] realizations ([`trainer::RaftTrainer`]/`GrpoTrainer`/`CaptureTrainer`).
//! The candle Qwen2.5-Coder + LoRA [`CausalLm`] (loaded by [`CandleModelLoader`])
//! is the real, GPU-validated body behind the `models` feature; with `models`
//! off it compiles to a CPU stub so the control flow stays exercisable.

use async_trait::async_trait;

use antumbra_core::{AntumbraError, Result, TrainingRecipe};

pub mod compose;
pub mod config;
pub mod consolidate;
pub mod corpus;
pub mod decision;
pub mod decode;
pub mod device;
pub mod eval;
pub mod grpo;
mod grpo_eval;
pub mod harness;
pub mod holdout;
pub mod lora;
pub mod memory;
pub mod model;
pub mod objective;
pub mod raft;
pub mod router;
pub mod teach;
pub mod trainer;

#[cfg(feature = "models")]
pub mod models;

pub use compose::compose_adapters;
pub use config::{RaftConfig, TrainDtype};
pub use consolidate::{
    interleave_replay, replay_from_tasks, score_memory, ConsolidationPolicy, Verdict,
};
pub use corpus::JsonCorpus;
pub use eval::{eval_pass_rate, EvalOutcome, TaskResult};
pub use grpo::{grpo_train, GrpoExperience, GrpoLm, GrpoModelLoader, GrpoSample};
pub use harness::{
    metabolize, normalize_traces, parse_traces as parse_harness_traces, HarnessStep, HarnessTrace,
    MetabolizePolicy,
};
pub use holdout::{split as split_holdout, Split};
pub use memory::{import as import_memories, ImportPolicy, ImportedTask, Intake, MemoryRecord};
pub use model::{CausalLm, Corpus, CorpusTask, ModelLoader, SftExample};
pub use raft::raft_train;
pub use router::train_learned_router;
pub use teach::capture_corrections;
pub use trainer::{CaptureTrainer, GrpoTrainer, RaftTrainer};

/// Placeholder for the candle model until MT-1 lands; implements [`CausalLm`]
/// so the loader's associated type is complete, but every method reports
/// `Unimplemented`.
#[derive(Debug, Default)]
pub struct PendingModel;

#[async_trait]
impl CausalLm for PendingModel {
    async fn generate(&mut self, _prompt: &str, _n: usize) -> Result<Vec<String>> {
        Err(AntumbraError::Unimplemented(
            "candle Qwen2.5-Coder generation (MT-1)",
        ))
    }
    async fn sft_step(&mut self, _batch: &[SftExample]) -> Result<f32> {
        Err(AntumbraError::Unimplemented("candle LoRA sft_step (MT-1)"))
    }
    fn save_adapter(&self, _path: &str) -> Result<()> {
        Err(AntumbraError::Unimplemented("candle adapter save (MT-1)"))
    }
}

#[async_trait]
impl GrpoLm for PendingModel {
    async fn sample_group(&mut self, _prompt: &str, _g: usize) -> Result<Vec<GrpoSample>> {
        Err(AntumbraError::Unimplemented("candle GRPO sampling (MT-1)"))
    }
    async fn reference_logprobs(&mut self, _prompt: &str, _tokens: &[u32]) -> Result<Vec<f32>> {
        Err(AntumbraError::Unimplemented("candle GRPO reference (MT-1)"))
    }
    async fn grpo_step(
        &mut self,
        _prompt: &str,
        _group: &[GrpoExperience],
        _cfg: &RaftConfig,
    ) -> Result<f32> {
        Err(AntumbraError::Unimplemented("candle GRPO step (MT-1)"))
    }
    fn save_adapter(&self, _path: &str) -> Result<()> {
        Err(AntumbraError::Unimplemented("candle adapter save (MT-1)"))
    }
}

/// Loads the candle Qwen2.5-Coder base + a fresh LoRA adapter (MT-1, GPU-
/// validated). The real body (candle-transformers + hf-hub + tokenizers + LoRA
/// injection) is behind the `models` feature; without it this is a CPU stub.
pub struct CandleModelLoader {
    pub config: RaftConfig,
}

impl CandleModelLoader {
    pub fn new(config: RaftConfig) -> Self {
        Self { config }
    }

    /// Build the candle Qwen + LoRA model (shared by the RAFT and GRPO loaders).
    #[cfg(feature = "models")]
    fn load_qwen(
        &self,
        base_model: &str,
        parent_adapter: Option<&str>,
        recipe: Option<&TrainingRecipe>,
    ) -> Result<models::QwenCausalLm> {
        let mut config = match recipe {
            Some(recipe) => self.config.with_recipe(recipe),
            None => self.config.clone(),
        };
        config.base_model = base_model.to_string();
        let device = device::best_device().map_err(|e| AntumbraError::other(e.to_string()))?;
        let mut model = models::QwenCausalLm::load(device, config)?;
        if let Some(adapter) = parent_adapter {
            model.load_adapter(adapter)?;
        }
        Ok(model)
    }
}

#[cfg(feature = "models")]
#[async_trait]
impl ModelLoader for CandleModelLoader {
    type Model = models::QwenCausalLm;

    async fn load_trained(
        &self,
        base: &str,
        parent: Option<&str>,
        recipe: Option<&TrainingRecipe>,
    ) -> Result<models::QwenCausalLm> {
        self.load_qwen(base, parent, recipe)
    }
}

#[cfg(feature = "models")]
#[async_trait]
impl GrpoModelLoader for CandleModelLoader {
    type Model = models::QwenCausalLm;

    async fn load_trained(
        &self,
        base: &str,
        parent: Option<&str>,
        recipe: Option<&TrainingRecipe>,
    ) -> Result<models::QwenCausalLm> {
        self.load_qwen(base, parent, recipe)
    }
}

#[cfg(not(feature = "models"))]
#[async_trait]
impl GrpoModelLoader for CandleModelLoader {
    type Model = PendingModel;

    async fn load_trained(
        &self,
        _base: &str,
        _parent: Option<&str>,
        _recipe: Option<&TrainingRecipe>,
    ) -> Result<PendingModel> {
        Err(AntumbraError::Unimplemented(
            "candle GRPO load requires antumbra-train built with --features models",
        ))
    }
}

#[cfg(not(feature = "models"))]
#[async_trait]
impl ModelLoader for CandleModelLoader {
    type Model = PendingModel;

    async fn load_trained(
        &self,
        _base_model: &str,
        _parent_adapter: Option<&str>,
        _recipe: Option<&TrainingRecipe>,
    ) -> Result<PendingModel> {
        Err(AntumbraError::Unimplemented(
            "candle Qwen2.5-Coder load requires antumbra-train built with --features models",
        ))
    }
}

// The stub loaders only exist (and only return `Unimplemented`) in the
// non-models build; under `--features models` `load` returns a real
// `QwenCausalLm`, so this test is non-models only.
#[cfg(all(test, not(feature = "models")))]
mod tests {
    use super::*;

    #[tokio::test]
    async fn candle_loader_reports_unimplemented() {
        let loader = CandleModelLoader::new(RaftConfig::default());
        // `load` is now defined on two loader traits; name one explicitly.
        let err = ModelLoader::load(&loader, "Qwen/Qwen2.5-Coder-1.5B", None)
            .await
            .unwrap_err();
        assert!(matches!(err, AntumbraError::Unimplemented(_)));
        let err = GrpoModelLoader::load(&loader, "Qwen/Qwen2.5-Coder-1.5B", None)
            .await
            .unwrap_err();
        assert!(matches!(err, AntumbraError::Unimplemented(_)));
    }
}
