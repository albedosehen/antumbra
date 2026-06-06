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
    /// Nucleus (top-p) sampling cutoff: keep the smallest set of tokens whose
    /// cumulative probability reaches `top_p`, then sample within it
    /// (Holtzman et al, 1904.09751). `>= 1.0` disables truncation. Ignored when
    /// greedy. Off by default so RAFT exploration is unchanged; serving sets it.
    pub top_p: f64,
    /// Repetition penalty over the generated continuation (Keskar et al,
    /// 1909.05858): `> 1.0` divides the logit of an already-generated token,
    /// suppressing loops. `1.0` is off. Serving raises it (~1.2); training leaves
    /// it off so best-of-K draws stay faithful to the policy.
    pub repetition_penalty: f64,
    /// Block any token that would complete an `n`-gram already present in the
    /// generated continuation (a hard anti-loop guard). `0` is off.
    pub no_repeat_ngram_size: usize,
    /// GRPO PPO-clip epsilon (ADR-0011). Unused by RAFT.
    pub clip_eps: f64,
    /// GRPO KL-to-reference penalty weight (ADR-0011). Unused by RAFT.
    pub kl_beta: f64,
    /// Quantize the frozen base to 4-bit Q4_K (QLoRA-proper, ADR-0011);
    /// dequantized in the forward. A capacity lever for larger bases.
    pub quantize_base: bool,
    /// Warm-start the LoRA from this saved adapter instead of fresh factors, so
    /// training *continues* a prior expert. `None` trains from scratch. EXP-010
    /// uses it for the monolithic continual-fine-tune arm.
    pub parent_adapter: Option<String>,
    /// Rehearsal examples per winner interleaved into capture SFT (EXP-021).
    /// `0.0` is replay off (plain capture); consolidating many memories at once
    /// sets it `> 0` to rehearse already-consolidated skills and resist
    /// catastrophic interference (the complementary-learning-systems fix).
    pub replay_ratio: f64,
    /// Accumulate gradients over the whole SFT batch and apply **one** optimizer
    /// step (the mean gradient), instead of one step per example (batch-of-1
    /// SGD). True mini-batch descent: the gradient is far less noisy, so a higher
    /// learning rate is stable and the adapter no longer over-updates toward
    /// whatever example it saw last ("Beware of the Batch Size"). `false` keeps
    /// the per-example path. Off by default so the validated recipe is unchanged.
    pub grad_accumulation: bool,
}

impl Default for RaftConfig {
    fn default() -> Self {
        Self {
            // Instruct (chat-tuned) base: for short instruction-following skills
            // it produces clean, controllable output where the raw completion
            // base rambles (model-quality #2). The chat template is applied
            // automatically (QwenCausalLm detects the `-Instruct` name).
            base_model: "Qwen/Qwen2.5-Coder-1.5B-Instruct".to_string(),
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
            top_p: 1.0,
            repetition_penalty: 1.0,
            no_repeat_ngram_size: 0,
            clip_eps: 0.2,
            kl_beta: 0.04,
            quantize_base: false,
            parent_adapter: None,
            replay_ratio: 0.0,
            grad_accumulation: false,
        }
    }
}

impl RaftConfig {
    /// LoRA scaling factor `alpha / rank`.
    pub fn lora_scale(&self) -> f64 {
        self.lora_alpha / self.lora_rank as f64
    }

    /// A config tuned for **serving** a learned skill rather than RAFT
    /// exploration: the caller's `temperature` (0 = greedy) plus a repetition
    /// penalty, an n-gram block, and nucleus truncation, so generation is the
    /// learned mode without the degeneration greedy/likelihood decoding is prone
    /// to (Holtzman et al, Keskar et al). Training keeps the defaults (off) so
    /// best-of-K draws stay faithful to the policy.
    pub fn for_serving(max_new_tokens: usize, temperature: f64) -> Self {
        Self {
            max_new_tokens,
            temperature,
            top_p: 0.9,
            repetition_penalty: 1.2,
            no_repeat_ngram_size: 3,
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn train_dtype_maps_to_candle() {
        assert_eq!(TrainDtype::F16.to_candle(), DType::F16);
        assert_eq!(TrainDtype::Bf16.to_candle(), DType::BF16);
        assert_eq!(TrainDtype::F32.to_candle(), DType::F32);
    }

    #[test]
    fn default_is_an_instruct_base_with_decode_off() {
        let cfg = RaftConfig::default();
        assert!(cfg.base_model.contains("Instruct"));
        assert_eq!(cfg.top_p, 1.0);
        assert_eq!(cfg.repetition_penalty, 1.0);
        assert_eq!(cfg.no_repeat_ngram_size, 0);
        assert!(!cfg.grad_accumulation);
    }

    #[test]
    fn lora_scale_is_alpha_over_rank() {
        let cfg = RaftConfig {
            lora_rank: 16,
            lora_alpha: 32.0,
            ..RaftConfig::default()
        };
        assert_eq!(cfg.lora_scale(), 2.0);
    }

    #[test]
    fn for_serving_turns_the_decode_policy_on() {
        let cfg = RaftConfig::for_serving(48, 0.0);
        assert_eq!(cfg.max_new_tokens, 48);
        assert_eq!(cfg.temperature, 0.0);
        assert_eq!(cfg.top_p, 0.9);
        assert_eq!(cfg.repetition_penalty, 1.2);
        assert_eq!(cfg.no_repeat_ngram_size, 3);
    }
}
