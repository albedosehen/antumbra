//! # antumbra-train — ADR-0002 / ADR-0010
//!
//! The DIY `candle` path that trains shadow adapters from **verified outcomes**
//! via RAFT-style reward-ranked LoRA fine-tuning. v0 builds bottom-up: the
//! candle LoRA training primitive ([`lora`]) is in place and tested on CPU; the
//! base-model generation + full RAFT loop (MT-1/MT-3) land next and run on the
//! 3090 Ti. The [`Trainer`] port is unchanged, so the loop adopts the real
//! trainer without upstream edits.

use async_trait::async_trait;

use antumbra_core::ports::{TrainOutcome, TrainRequest, Trainer};
use antumbra_core::{AntumbraError, Result};

pub mod config;
pub mod device;
pub mod lora;
pub mod model;
pub mod raft;

pub use config::{RaftConfig, TrainDtype};
pub use model::{CausalLm, CorpusTask, SftExample};
pub use raft::raft_train;

/// candle-backed QLoRA trainer. Carries the config the real path will need;
/// `train_shadow` is not yet implemented.
#[derive(Debug, Clone)]
pub struct CandleTrainer {
    /// Path to the shared 4-bit base the LoRA rides on.
    pub base_model_uri: String,
    /// Where graduated adapter checkpoints are written.
    pub adapter_dir: String,
}

impl CandleTrainer {
    pub fn new(base_model_uri: impl Into<String>, adapter_dir: impl Into<String>) -> Self {
        Self {
            base_model_uri: base_model_uri.into(),
            adapter_dir: adapter_dir.into(),
        }
    }
}

#[async_trait]
impl Trainer for CandleTrainer {
    async fn train_shadow(&self, _req: TrainRequest) -> Result<TrainOutcome> {
        Err(AntumbraError::Unimplemented(
            "DIY candle QLoRA shadow training (ADR-0002)",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn trainer_seam_reports_unimplemented() {
        let trainer = CandleTrainer::new("mem://base", "/adapters");
        let req = TrainRequest {
            shadow: antumbra_core::ShadowId::new("shadow:1"),
            base_model: "code-base".into(),
            corpus_task_ids: vec![],
            max_steps: 4,
        };
        let err = trainer.train_shadow(req).await.unwrap_err();
        assert!(matches!(err, AntumbraError::Unimplemented(_)));
    }
}
