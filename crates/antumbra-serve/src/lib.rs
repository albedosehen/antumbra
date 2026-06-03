//! # antumbra-serve — ADR-0006 (seam)
//!
//! S-LoRA-style serving: one shared base plus a library of hot-swappable LoRA
//! adapters on a single GPU, via `llama-cpp-2` (GGUF + LoRA) or `mistral.rs`.
//! Not yet wired; this crate provides the [`Serve`] seam so routing and the
//! loop run against the fake server today and the real engine drops in here.

use async_trait::async_trait;

use antumbra_core::ports::{ActOutput, ActRequest, Serve};
use antumbra_core::{AntumbraError, Result};

/// Multi-adapter server over a shared base. `act` is not yet implemented.
#[derive(Debug, Clone)]
pub struct MultiAdapterServe {
    /// GGUF / safetensors path of the shared, code-capable base.
    pub base_model_uri: String,
    /// Directory of frozen LoRA adapters to hot-swap.
    pub adapter_dir: String,
}

impl MultiAdapterServe {
    pub fn new(base_model_uri: impl Into<String>, adapter_dir: impl Into<String>) -> Self {
        Self {
            base_model_uri: base_model_uri.into(),
            adapter_dir: adapter_dir.into(),
        }
    }
}

#[async_trait]
impl Serve for MultiAdapterServe {
    async fn act(&self, _req: ActRequest) -> Result<ActOutput> {
        Err(AntumbraError::Unimplemented(
            "llama-cpp-2 / mistral.rs multi-adapter serving (ADR-0006)",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn serve_seam_reports_unimplemented() {
        let serve = MultiAdapterServe::new("mem://base.gguf", "/adapters");
        let req = ActRequest {
            task_id: "t:1".into(),
            prompt: "hello".into(),
            adapters: vec![],
        };
        let err = serve.act(req).await.unwrap_err();
        assert!(matches!(err, AntumbraError::Unimplemented(_)));
    }
}
