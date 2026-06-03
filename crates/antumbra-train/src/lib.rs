//! # antumbra-train — ADR-0002 (seam)
//!
//! The heaviest real component: a DIY `candle` QLoRA path (NF4 4-bit frozen
//! base + LoRA) that trains shadow adapters and the gate. It is not yet built;
//! this crate provides the [`Trainer`] seam so the loop already runs against
//! the fake trainer and the real one drops in here unchanged.

use async_trait::async_trait;

use antumbra_core::ports::{TrainOutcome, TrainRequest, Trainer};
use antumbra_core::{AntumbraError, Result};

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
