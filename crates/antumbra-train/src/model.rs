//! The model seam the RAFT loop drives. A [`CausalLm`] generates candidate
//! completions and trains its LoRA adapter on verified winners. The candle
//! Qwen2.5-Coder + LoRA realization (MT-1) is GPU-validated; the RAFT control
//! flow is tested against a fake, mirroring how the rest of Antumbra seams off
//! heavy dependencies.

use async_trait::async_trait;

use antumbra_core::Result;

/// A verifiable task drawn from the corpus (your repo): a prompt to complete.
#[derive(Debug, Clone)]
pub struct CorpusTask {
    pub id: String,
    pub prompt: String,
}

/// One supervised example: train the adapter to produce `completion` for `prompt`.
#[derive(Debug, Clone)]
pub struct SftExample {
    pub prompt: String,
    pub completion: String,
}

#[async_trait]
pub trait CausalLm {
    /// Sample `n_samples` completions for a prompt (the RAFT rollouts).
    /// `&mut` because decoding advances the model's KV cache.
    async fn generate(&mut self, prompt: &str, n_samples: usize) -> Result<Vec<String>>;
    /// One supervised fine-tuning step over verified winners; returns the loss.
    async fn sft_step(&mut self, batch: &[SftExample]) -> Result<f32>;
    /// Persist the trained LoRA adapter (safetensors).
    fn save_adapter(&self, path: &str) -> Result<()>;
}

/// Builds a fresh [`CausalLm`] for a shadow: the shared base plus a new LoRA
/// adapter (optionally warm-started from a parent expert's adapter). The candle
/// Qwen2.5-Coder realization is the GPU-validated impl (MT-1).
#[async_trait]
pub trait ModelLoader: Send + Sync {
    type Model: CausalLm + Send;
    async fn load(&self, base_model: &str, parent_adapter: Option<&str>) -> Result<Self::Model>;
}

/// Resolves corpus task ids (from a `TrainRequest`) to verifiable prompts.
pub trait Corpus: Send + Sync {
    fn tasks(&self, task_ids: &[String]) -> Vec<CorpusTask>;
}
