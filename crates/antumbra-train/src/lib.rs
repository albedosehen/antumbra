//! # antumbra-train — ADR-0002 / ADR-0010
//!
//! The DIY `candle` path that trains shadow adapters from **verified outcomes**
//! via RAFT-style reward-ranked LoRA fine-tuning.
//!
//! Built bottom-up and tested on CPU: the LoRA training primitive ([`lora`]),
//! the SFT objective ([`objective`]), the RAFT loop ([`raft`]), and the
//! [`Trainer`] realization ([`trainer::RaftTrainer`]) are all in place. The one
//! remaining GPU-validated piece (MT-1) is the candle Qwen2.5-Coder + LoRA
//! [`CausalLm`], loaded by [`CandleModelLoader`] — currently a typed seam.

use async_trait::async_trait;

use antumbra_core::{AntumbraError, Result};

pub mod config;
pub mod device;
pub mod lora;
pub mod model;
pub mod objective;
pub mod raft;
pub mod trainer;

pub use config::{RaftConfig, TrainDtype};
pub use model::{CausalLm, Corpus, CorpusTask, ModelLoader, SftExample};
pub use raft::raft_train;
pub use trainer::RaftTrainer;

/// Placeholder for the candle model until MT-1 lands; implements [`CausalLm`]
/// so the loader's associated type is complete, but every method reports
/// `Unimplemented`.
#[derive(Debug, Default)]
pub struct PendingModel;

#[async_trait]
impl CausalLm for PendingModel {
    async fn generate(&self, _prompt: &str, _n: usize) -> Result<Vec<String>> {
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

/// Loads the candle Qwen2.5-Coder base + a fresh LoRA adapter (MT-1). The real
/// body — candle-transformers + hf-hub + tokenizers + LoRA injection — runs on
/// the GPU and lands behind a `models` feature; this is its typed seam.
pub struct CandleModelLoader {
    pub config: RaftConfig,
}

impl CandleModelLoader {
    pub fn new(config: RaftConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl ModelLoader for CandleModelLoader {
    type Model = PendingModel;

    async fn load(&self, _base_model: &str, _parent_adapter: Option<&str>) -> Result<PendingModel> {
        Err(AntumbraError::Unimplemented(
            "candle Qwen2.5-Coder load (MT-1: candle-transformers + hf-hub + LoRA)",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn candle_loader_reports_unimplemented() {
        let loader = CandleModelLoader::new(RaftConfig::default());
        let err = loader
            .load("Qwen/Qwen2.5-Coder-1.5B", None)
            .await
            .unwrap_err();
        assert!(matches!(err, AntumbraError::Unimplemented(_)));
    }
}
