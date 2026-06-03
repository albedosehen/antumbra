//! Candle-backed serving (ADR-0006). v0 serves a single frozen LoRA adapter
//! over the shared base by reusing the antumbra-train Qwen2.5-Coder + LoRA
//! model: load the base, load the adapter weights, generate. This is what turns
//! a graduated expert into an answer. Multi-adapter hot-swap (S-LoRA-style) and
//! the llama-cpp-2 / mistral.rs backends are the deferred richer ADR-0006.

use async_trait::async_trait;

use antumbra_core::ports::{ActOutput, ActRequest, Serve, StepOutput};
use antumbra_core::Result;
use antumbra_train::{CandleModelLoader, CausalLm, ModelLoader, RaftConfig};

/// Serves one frozen adapter over a shared base. Loading happens per `act` so
/// the server is stateless; the heavy base weights come from the hf-hub cache.
pub struct CandleServe {
    base_model: String,
    adapter_uri: String,
    config: RaftConfig,
}

impl CandleServe {
    pub fn new(
        base_model: impl Into<String>,
        adapter_uri: impl Into<String>,
        config: RaftConfig,
    ) -> Self {
        Self {
            base_model: base_model.into(),
            adapter_uri: adapter_uri.into(),
            config,
        }
    }
}

#[async_trait]
impl Serve for CandleServe {
    async fn act(&self, req: ActRequest) -> Result<ActOutput> {
        let loader = CandleModelLoader::new(self.config.clone());
        let mut model = loader
            .load(&self.base_model, Some(&self.adapter_uri))
            .await?;
        let mut outputs = model.generate(&req.prompt, 1).await?;
        let final_output = outputs.drain(..).next().unwrap_or_default();
        Ok(ActOutput {
            steps: vec![StepOutput {
                step_idx: 0,
                content: final_output.clone(),
            }],
            final_output,
        })
    }
}
