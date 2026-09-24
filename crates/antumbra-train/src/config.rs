//! Trainer configuration. RAFT-style reward-ranked LoRA fine-tuning
//! over an f16 frozen, code-capable base.

use candle_core::DType;

use antumbra_core::TrainingRecipe;

/// Compute precision. Ampere (3090 Ti) does f16/bf16 well; Pascal (1080) is
/// gimped at f16, so f32 is the fallback there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainDtype {
    F16,
    Bf16,
    F32,
}

impl std::str::FromStr for TrainDtype {
    type Err = antumbra_core::AntumbraError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "f16" => Ok(TrainDtype::F16),
            "bf16" => Ok(TrainDtype::Bf16),
            "f32" => Ok(TrainDtype::F32),
            other => Err(antumbra_core::AntumbraError::other(format!(
                "unknown dtype `{other}` (use f32, bf16 or f16)"
            ))),
        }
    }
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
    /// Hugging Face id of the shared, code-capable base (the trainer locks
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
    /// GRPO PPO-clip epsilon. Unused by RAFT.
    pub clip_eps: f64,
    /// GRPO KL-to-reference penalty weight. Unused by RAFT.
    pub kl_beta: f64,
    /// Quantize the frozen base to 4-bit Q4_K (QLoRA-proper, v1 efficiency);
    /// dequantized in the forward. A capacity lever for larger bases.
    pub quantize_base: bool,
    /// Warm-start the LoRA from this saved adapter instead of fresh factors, so
    /// training *continues* a prior expert. `None` trains from scratch.
    /// Used for the monolithic continual-fine-tune arm.
    pub parent_adapter: Option<String>,
    /// Rehearsal examples per winner interleaved into capture SFT.
    /// `0.0` is replay off (plain capture); consolidating many memories at once
    /// sets it `> 0` to rehearse already-consolidated skills and resist
    /// catastrophic interference (the complementary-learning-systems fix).
    pub replay_ratio: f64,
    /// Verified winners per optimizer step. `1` (the default, the validated
    /// recipe) steps on each example in turn. `n > 1` steps on the mean
    /// gradient of `n` at a time: true mini-batch descent, far less noisy, so a
    /// higher learning rate is stable and the adapter no longer over-updates
    /// toward whatever example it saw last ("Beware of the Batch Size"). Each
    /// example is backpropagated before the next is run, so memory does not
    /// grow with `n`.
    pub batch_size: usize,
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
            batch_size: 1,
        }
    }
}

impl RaftConfig {
    /// The searched part of this configuration (ADR-0022 S-1): what a run
    /// trains under when its request names no recipe.
    pub fn recipe(&self) -> TrainingRecipe {
        TrainingRecipe {
            learning_rate: self.learning_rate,
            batch_size: u32::try_from(self.batch_size).unwrap_or(u32::MAX),
            kl_beta: self.kl_beta,
        }
    }

    /// This configuration with `recipe` in place of its searched part. Nothing
    /// outside the recipe changes, rank least of all.
    pub fn with_recipe(&self, recipe: &TrainingRecipe) -> Self {
        Self {
            learning_rate: recipe.learning_rate,
            batch_size: recipe.batch_size as usize,
            kl_beta: recipe.kl_beta,
            ..self.clone()
        }
    }

    /// LoRA scaling factor `alpha / rank`. A rank-0 adapter (an empty or corrupt
    /// load) scales to `0.0` rather than `0/0 = NaN`, so a bad adapter contributes
    /// nothing instead of poisoning every logit with NaN.
    pub fn lora_scale(&self) -> f64 {
        if self.lora_rank == 0 {
            return 0.0;
        }
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
    fn train_dtype_parses_by_name_and_refuses_the_unknown() {
        assert_eq!("f32".parse::<TrainDtype>().ok(), Some(TrainDtype::F32));
        assert_eq!(" BF16 ".parse::<TrainDtype>().ok(), Some(TrainDtype::Bf16));
        assert_eq!("f16".parse::<TrainDtype>().ok(), Some(TrainDtype::F16));
        assert!("fp8".parse::<TrainDtype>().is_err());
    }

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
        assert_eq!(
            cfg.batch_size, 1,
            "one step per example is the validated recipe"
        );
    }

    #[test]
    fn a_recipe_replaces_only_the_searched_settings() {
        let base = RaftConfig::default();
        let recipe = TrainingRecipe {
            learning_rate: 3e-4,
            batch_size: 4,
            kl_beta: 0.1,
        };
        let cfg = base.with_recipe(&recipe);
        assert_eq!(cfg.recipe(), recipe);
        assert_eq!(
            (
                cfg.lora_rank,
                cfg.lora_alpha,
                cfg.rounds,
                cfg.samples_per_task
            ),
            (
                base.lora_rank,
                base.lora_alpha,
                base.rounds,
                base.samples_per_task
            ),
            "rank, alpha and the rest stay as configured"
        );
        assert_eq!(base.with_recipe(&base.recipe()).recipe(), base.recipe());
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
    fn lora_scale_is_zero_not_nan_for_a_rank_zero_adapter() {
        let cfg = RaftConfig {
            lora_rank: 0,
            lora_alpha: 32.0,
            ..RaftConfig::default()
        };
        let scale = cfg.lora_scale();
        assert!(scale.is_finite(), "a rank-0 adapter must not yield NaN");
        assert_eq!(scale, 0.0);
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
