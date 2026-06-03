//! Trainer configuration (ADR-0010). RAFT-style reward-ranked LoRA fine-tuning
//! over an f16 frozen, code-capable base.

use candle_core::DType;

/// Compute precision. Ampere (3090 Ti) does f16/bf16 well; Pascal (1080) is
/// gimped at f16, so f32 is the fallback there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainDtype {
    F16,
    Bf16,
    F32,
}

impl TrainDtype {
    pub fn to_candle(self) -> DType {
        match self {
            TrainDtype::F16 => DType::F16,
            TrainDtype::Bf16 => DType::BF16,
            TrainDtype::F32 => DType::F32,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RaftConfig {
    /// Hugging Face id of the shared, code-capable base (ADR-0010 locks
    /// Qwen2.5-Coder-1.5B for v0).
    pub base_model: String,
    /// Directory where graduated adapter checkpoints are written.
    pub adapter_dir: String,
    /// LoRA rank and scaling (alpha); `scale = alpha / rank`.
    pub lora_rank: usize,
    pub lora_alpha: f64,
    /// Completions sampled per corpus task before reward-ranking (the K in RAFT).
    pub samples_per_task: usize,
    /// RAFT rounds (sample -> verify -> SFT) per shadow.
    pub rounds: usize,
    /// Max tokens generated per completion.
    pub max_new_tokens: usize,
    pub learning_rate: f64,
    pub dtype: TrainDtype,
    /// Sampling temperature for generation; higher = more diverse draws (so
    /// best-of-K actually explores). `<= 0` is greedy/argmax.
    pub temperature: f64,
    /// GRPO PPO-clip epsilon (ADR-0011). Unused by RAFT.
    pub clip_eps: f64,
    /// GRPO KL-to-reference penalty weight (ADR-0011). Unused by RAFT.
    pub kl_beta: f64,
    /// Quantize the frozen base to 4-bit Q4_K (QLoRA-proper, ADR-0011);
    /// dequantized in the forward. A capacity lever for larger bases.
    pub quantize_base: bool,
}

impl Default for RaftConfig {
    fn default() -> Self {
        Self {
            base_model: "Qwen/Qwen2.5-Coder-1.5B".to_string(),
            adapter_dir: "adapters".to_string(),
            lora_rank: 16,
            lora_alpha: 32.0,
            samples_per_task: 8,
            rounds: 4,
            max_new_tokens: 256,
            learning_rate: 1e-4,
            // bf16 on GPU: same exponent range as f32, so the transformer
            // forward can't overflow like f16 does (CPU is forced to f32).
            dtype: TrainDtype::Bf16,
            temperature: 0.8,
            clip_eps: 0.2,
            kl_beta: 0.04,
            quantize_base: false,
        }
    }
}

impl RaftConfig {
    /// LoRA scaling factor `alpha / rank`.
    pub fn lora_scale(&self) -> f64 {
        self.lora_alpha / self.lora_rank as f64
    }
}
