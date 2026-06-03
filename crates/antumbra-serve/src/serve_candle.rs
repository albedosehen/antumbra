//! Candle-backed serving (ADR-0006). v0 serves a single frozen LoRA adapter
//! over the shared base by reusing the antumbra-train Qwen2.5-Coder + LoRA
//! model: load the base, load the adapter weights, generate. This is what turns
//! a graduated expert into an answer. Multi-adapter hot-swap (S-LoRA-style) and
//! the llama-cpp-2 / mistral.rs backends are the deferred richer ADR-0006.

use async_trait::async_trait;
use tokio::sync::Mutex;

use antumbra_core::ports::{ActOutput, ActRequest, Serve, StepOutput};
use antumbra_core::Result;
use antumbra_train::models::QwenCausalLm;
use antumbra_train::{CandleModelLoader, CausalLm, ModelLoader, RaftConfig};

/// Serves a shared base, optionally with one frozen adapter on top. The model
/// is loaded once on the first `act` and **cached** for the life of the server,
/// so best-of-K probing and repeated answers do not pay the multi-GB reload
/// each call. `adapter` is `None` for base-only serving (e.g. the probe).
pub struct CandleServe {
    base_model: String,
    adapter: Option<String>,
    config: RaftConfig,
    model: Mutex<Option<QwenCausalLm>>,
}

impl CandleServe {
    pub fn new(base_model: impl Into<String>, adapter: Option<String>, config: RaftConfig) -> Self {
        Self {
            base_model: base_model.into(),
            adapter,
            config,
            model: Mutex::new(None),
        }
    }
}

#[async_trait]
impl Serve for CandleServe {
    async fn act(&self, req: ActRequest) -> Result<ActOutput> {
        let mut guard = self.model.lock().await;
        if guard.is_none() {
            let loader = CandleModelLoader::new(self.config.clone());
            *guard = Some(
                loader
                    .load(&self.base_model, self.adapter.as_deref())
                    .await?,
            );
        }
        let model = guard.as_mut().expect("model loaded above");
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
