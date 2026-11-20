//! Trainer configuration. RAFT-style reward-ranked LoRA fine-tuning
//! over an f16 frozen, code-capable base.

use candle_core::quantized::GgmlDType;
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

/// How a model loaded only to serve holds its frozen base: multiplied in a
/// quantized form by candle's quantized kernels, never dequantized. On a CPU,
/// where every decoded token reads every weight, Q8_0 reads about a quarter of
/// the f32 bytes, Q6_K about a fifth and Q4_K about a sixth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServeQuant {
    Q8_0,
    Q6K,
    Q4K,
}

impl std::str::FromStr for ServeQuant {
    type Err = antumbra_core::AntumbraError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "q8_0" | "q8" => Ok(ServeQuant::Q8_0),
            "q6_k" | "q6k" | "q6" => Ok(ServeQuant::Q6K),
            "q4_k" | "q4k" | "q4" => Ok(ServeQuant::Q4K),
            other => Err(antumbra_core::AntumbraError::other(format!(
                "unknown serving quantization `{other}` (use q8_0, q6_k or q4_k)"
            ))),
        }
    }
}

impl ServeQuant {
    pub fn to_ggml(self) -> GgmlDType {
        match self {
            ServeQuant::Q8_0 => GgmlDType::Q8_0,
            ServeQuant::Q6K => GgmlDType::Q6K,
            ServeQuant::Q4K => GgmlDType::Q4K,
        }
    }

    /// How this build serves on this machine when nothing says otherwise. A
    /// build with no GPU backend serves on the CPU, where the dense f32 base
    /// reads about 6 GB a token, so it serves quantized: from Q6_K where the
    /// CPU multiplies six-bit tiles ([`crate::device::six_bit_by_default`]), and
    /// from Q8_0 on any other, where Q6_K would go through slower kernels
    /// than Q8_0 does. A CUDA build keeps the dense base it trained on. A
    /// Metal build serves as a build with no GPU does, from the CPU
    /// ([`crate::device::serves_from_cpu`]), where a token is decoded at more
    /// than twice the rate Metal decodes it from the dense base; `off` keeps
    /// the dense base on Metal.
    ///
    /// No quantized form is the dense base. Q8_0 moves an expert least: over
    /// two experts' held-out positions (2078 and 865) its log-probabilities
    /// shift from the dense base's by 0.10 and 0.09 nats on average, Q6_K's
    /// by 0.15 and 0.12, and Q4_K's by 0.41 and 0.36. Q8_0 is also the most
    /// to read at every token, and so the slowest to decode: on a CPU that
    /// multiplies through Antumbra's tiles Q6_K decodes about 1.3 times as
    /// fast and Q4_K about 1.7 times. Q6_K is the default where it is that
    /// much faster: a third more tokens a second for half again the shift.
    /// What the shift does to answers is too small to tell from chance at
    /// either form: of the two experts' 113 held-out tasks, 80 answers
    /// followed their behavior over Q8_0, 82 over Q6_K and 77 over Q4_K, and
    /// the count of positions whose first-ranked token changes (about 80,
    /// about 100 and about 290 of the first expert's 2078) moves by ten or
    /// twenty with nothing but the order a sum is added in.
    pub fn for_build() -> Option<Self> {
        let on_gpu = cfg!(any(feature = "cuda", feature = "metal"));
        if on_gpu && !crate::device::serves_from_cpu() {
            None
        } else if crate::device::six_bit_by_default() {
            Some(ServeQuant::Q6K)
        } else {
            Some(ServeQuant::Q8_0)
        }
    }

    /// What an operator's setting names: `auto` is [`Self::for_build`], `off`
    /// keeps the base dense, and `q8_0`, `q6_k` or `q4_k` name a form on any
    /// build.
    pub fn from_setting(setting: &str) -> Result<Option<Self>, antumbra_core::AntumbraError> {
        match setting.trim().to_ascii_lowercase().as_str() {
            "auto" | "" => Ok(Self::for_build()),
            "off" | "none" | "dense" => Ok(None),
            other => other.parse().map(Some).map_err(|_| {
                antumbra_core::AntumbraError::other(format!(
                    "unknown serving quantization `{other}` (use auto, off, q8_0, q6_k or q4_k)"
                ))
            }),
        }
    }

    /// The forms an operator's setting asks an expert's drift be measured on
    /// when it is admitted. `auto`, and a setting left empty, is Q8_0 and
    /// Q6_K, the two forms a build with no GPU backend serves by default (the
    /// second where the CPU multiplies six-bit tiles); `off` is none;
    /// and `q8_0`, `q6_k`, `q4_k` or several with commas name them, each
    /// measured once.
    pub fn drift_from_setting(setting: &str) -> Result<Vec<Self>, antumbra_core::AntumbraError> {
        let setting = setting.trim().to_ascii_lowercase();
        match setting.as_str() {
            "auto" | "" => return Ok(vec![ServeQuant::Q8_0, ServeQuant::Q6K]),
            "off" | "none" => return Ok(Vec::new()),
            _ => {}
        }
        let mut forms = Vec::new();
        for name in setting.split(',') {
            let name = name.trim();
            let form: Self = name.parse().map_err(|_| {
                antumbra_core::AntumbraError::other(format!(
                    "unknown quantization `{name}` to measure drift on (use auto, off, q8_0, \
                     q6_k, q4_k, or several with commas)"
                ))
            })?;
            if !forms.contains(&form) {
                forms.push(form);
            }
        }
        Ok(forms)
    }

    /// The name a setting or a log line uses for this form.
    pub fn name(self) -> &'static str {
        match self {
            ServeQuant::Q8_0 => "q8_0",
            ServeQuant::Q6K => "q6_k",
            ServeQuant::Q4K => "q4_k",
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
    /// suppressing loops. `1.0` is off, which it is everywhere unless a
    /// caller sets it: it rewords most of what a model would have said, and
    /// moves a digit it has already written.
    pub repetition_penalty: f64,
    /// Block any token that would complete an `n`-gram already present in the
    /// generated continuation. `0` is off, which it is everywhere unless a
    /// caller sets it: a block on 3-grams cannot let a name of three tokens be
    /// said twice. A generation that loops is ended by [`crate::decode::loops`]
    /// whatever these two are.
    pub no_repeat_ngram_size: usize,
    /// GRPO PPO-clip epsilon. Unused by RAFT.
    pub clip_eps: f64,
    /// GRPO KL-to-reference penalty weight. Unused by RAFT.
    pub kl_beta: f64,
    /// Quantize the frozen base to 4-bit Q4_K (QLoRA-proper, v1 efficiency);
    /// dequantized in the forward. A capacity lever for larger bases.
    pub quantize_base: bool,
    /// Load the model to serve, with its frozen base quantized this way and
    /// multiplied in that form (see [`ServeQuant`]). Each layer is quantized as
    /// it loads, so the dense base is never held whole, and the head is held
    /// as Q8_0. A model loaded so cannot train: the quantized product carries
    /// no gradient. `None` keeps the base dense (or `quantize_base`'s form).
    pub serve_quant: Option<ServeQuant>,
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
    /// Seed the run's own randomness: the adapter's starting factors and the
    /// order its examples are trained in, each a stream of this seed shifted
    /// by the process's generation seed. `None` takes both from the process's
    /// running state, so a run then depends on what the process did before
    /// it: the same set trained first and third in one server came out learned
    /// and refused.
    pub seed: Option<u64>,
    /// Decode greedily several tokens a pass where the model can
    /// ([`crate::draft`]): guesses ride along with the token in hand and are
    /// kept where they are the model's own next token, so what is written
    /// is what a token at a time writes. The guesses come from what the
    /// model wrote for earlier requests through the same blend of experts,
    /// which it remembers between them. Off unless a caller sets it;
    /// serving sets it.
    pub read_ahead: bool,
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
            serve_quant: None,
            parent_adapter: None,
            replay_ratio: 0.0,
            batch_size: 1,
            seed: None,
            read_ahead: false,
        }
    }
}

impl RaftConfig {
    /// The searched part of this configuration: what a run
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

    /// A config for **serving** a learned skill rather than RAFT exploration:
    /// the caller's `temperature` (0 = greedy), nucleus truncation for when it
    /// samples (Holtzman et al), and the base held the way this build serves
    /// ([`ServeQuant::for_build`]): quantized where there is no GPU backend, so
    /// a model loaded from this config there cannot train.
    ///
    /// It decodes as admission and the audits do, with no repetition penalty
    /// and no block on repeated n-grams, so what was verified of an expert is
    /// what it answers. Until 0.4.5 serving turned both on (1.2, and 3-grams)
    /// and admission neither. Over one expert's 78 held-out tasks the same
    /// number of answers followed their behavior either way, and 14 were
    /// worded as admission had read them: the block cannot let a name of
    /// three tokens appear twice (`fn add(a: i64, b: i64) -> i64` came back as
    /// `fn add(a: i64, b: i32) -> i6`), and the penalty reworded four answers
    /// in five. What they were for, a model that repeats itself without end,
    /// is ended where it starts ([`crate::decode::loops`]).
    ///
    /// And it reads ahead ([`RaftConfig::read_ahead`]): the same tokens,
    /// in fewer passes over the weights.
    pub fn for_serving(max_new_tokens: usize, temperature: f64) -> Self {
        Self {
            max_new_tokens,
            temperature,
            top_p: 0.9,
            serve_quant: ServeQuant::for_build(),
            read_ahead: true,
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antumbra_core::Result;

    #[test]
    fn train_dtype_parses_by_name_and_refuses_the_unknown() -> Result<()> {
        assert_eq!("f32".parse::<TrainDtype>().ok(), Some(TrainDtype::F32));
        assert_eq!(" BF16 ".parse::<TrainDtype>().ok(), Some(TrainDtype::Bf16));
        assert_eq!("f16".parse::<TrainDtype>().ok(), Some(TrainDtype::F16));
        assert!("fp8".parse::<TrainDtype>().is_err());
        Ok(())
    }

    #[test]
    fn train_dtype_maps_to_candle() -> Result<()> {
        assert_eq!(TrainDtype::F16.to_candle(), DType::F16);
        assert_eq!(TrainDtype::Bf16.to_candle(), DType::BF16);
        assert_eq!(TrainDtype::F32.to_candle(), DType::F32);
        Ok(())
    }

    #[test]
    fn default_is_an_instruct_base_with_decode_off() -> Result<()> {
        let cfg = RaftConfig::default();
        assert!(cfg.base_model.contains("Instruct"));
        assert_eq!(cfg.top_p, 1.0);
        assert_eq!(cfg.repetition_penalty, 1.0);
        assert_eq!(cfg.no_repeat_ngram_size, 0);
        assert_eq!(
            cfg.batch_size, 1,
            "one step per example is the validated recipe"
        );
        Ok(())
    }

    #[test]
    fn a_recipe_replaces_only_the_searched_settings() -> Result<()> {
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
        Ok(())
    }

    #[test]
    fn lora_scale_is_alpha_over_rank() -> Result<()> {
        let cfg = RaftConfig {
            lora_rank: 16,
            lora_alpha: 32.0,
            ..RaftConfig::default()
        };
        assert_eq!(cfg.lora_scale(), 2.0);
        Ok(())
    }

    #[test]
    fn lora_scale_is_zero_not_nan_for_a_rank_zero_adapter() -> Result<()> {
        let cfg = RaftConfig {
            lora_rank: 0,
            lora_alpha: 32.0,
            ..RaftConfig::default()
        };
        let scale = cfg.lora_scale();
        assert!(scale.is_finite(), "a rank-0 adapter must not yield NaN");
        assert_eq!(scale, 0.0);
        Ok(())
    }

    #[test]
    fn a_serving_quantization_parses_by_its_ggml_name() -> Result<()> {
        for (name, want) in [
            ("q8_0", ServeQuant::Q8_0),
            ("Q8", ServeQuant::Q8_0),
            ("q6_k", ServeQuant::Q6K),
            ("Q6", ServeQuant::Q6K),
            ("q4_k", ServeQuant::Q4K),
            (" Q4K ", ServeQuant::Q4K),
        ] {
            assert_eq!(name.parse::<ServeQuant>()?, want, "{name}");
        }
        assert!("q2_k".parse::<ServeQuant>().is_err());
        assert_eq!(ServeQuant::Q4K.to_ggml(), GgmlDType::Q4K);
        assert_eq!(ServeQuant::Q6K.to_ggml(), GgmlDType::Q6K);
        assert_eq!(RaftConfig::default().serve_quant, None);
        Ok(())
    }

    /// A build that serves from the CPU serves quantized unless told
    /// otherwise, and one that serves from its GPU serves dense; either can
    /// be told.
    #[test]
    fn a_setting_names_the_build_default_a_dense_base_or_a_form() -> Result<()> {
        // On a CPU, Q6_K where it multiplies six-bit tiles and Q8_0 where it
        // does not.
        let build = ServeQuant::for_build();
        let on_gpu = cfg!(any(feature = "cuda", feature = "metal"));
        if on_gpu && !crate::device::serves_from_cpu() {
            assert_eq!(build, None);
        } else {
            let fast = crate::device::six_bit_by_default();
            assert_eq!(
                build,
                Some(if fast {
                    ServeQuant::Q6K
                } else {
                    ServeQuant::Q8_0
                })
            );
        }
        assert_eq!(ServeQuant::from_setting("auto")?, build);
        assert_eq!(ServeQuant::from_setting("")?, build);
        for dense in ["off", "none", " Dense "] {
            assert_eq!(ServeQuant::from_setting(dense)?, None, "{dense}");
        }
        for q in [ServeQuant::Q8_0, ServeQuant::Q6K, ServeQuant::Q4K] {
            assert_eq!(ServeQuant::from_setting(q.name())?, Some(q));
        }
        let refused = ServeQuant::from_setting("q2_k");
        assert!(
            refused.is_err_and(|e| e.to_string().contains("auto, off, q8_0, q6_k or q4_k")),
            "a setting that names nothing says what a setting may be"
        );
        Ok(())
    }

    #[test]
    fn a_drift_setting_names_no_form_one_or_several() -> Result<()> {
        let (q8, q6, q4) = (ServeQuant::Q8_0, ServeQuant::Q6K, ServeQuant::Q4K);
        for (setting, want) in [
            ("auto", vec![q8, q6]),
            ("", vec![q8, q6]),
            ("off", vec![]),
            (" None ", vec![]),
            ("q4_k", vec![q4]),
            ("q6_k", vec![q6]),
            ("q8_0,q4_k", vec![q8, q4]),
            ("q8_0,q6_k,q4_k", vec![q8, q6, q4]),
            (" Q4K , q8 ", vec![q4, q8]),
            // A form named twice is measured once.
            ("q8_0,q8,q8_0", vec![q8]),
        ] {
            assert_eq!(
                ServeQuant::drift_from_setting(setting)?,
                want,
                "{setting:?}"
            );
        }
        for refused in ["q2_k", "q8_0,", "q8_0;q4_k", "dense"] {
            let named = ServeQuant::drift_from_setting(refused);
            assert!(
                named.is_err_and(|e| e.to_string().contains("use auto, off, q8_0")),
                "{refused:?} names no form and says what a setting may be"
            );
        }
        Ok(())
    }

    /// Serving decodes as admission does: the penalties are off in both, and
    /// only the nucleus cutoff, which greedy decoding never reads, differs.
    #[test]
    fn serving_decodes_as_admission_does() -> Result<()> {
        let cfg = RaftConfig::for_serving(48, 0.0);
        assert_eq!(cfg.max_new_tokens, 48);
        assert_eq!(cfg.temperature, 0.0);
        assert_eq!(cfg.top_p, 0.9);
        let admission = RaftConfig::default();
        assert_eq!(cfg.repetition_penalty, admission.repetition_penalty);
        assert_eq!(cfg.no_repeat_ngram_size, admission.no_repeat_ngram_size);
        assert_eq!((cfg.repetition_penalty, cfg.no_repeat_ngram_size), (1.0, 0));
        // Serving holds the base as the build serves it; training never does.
        assert_eq!(cfg.serve_quant, ServeQuant::for_build());
        assert_eq!(RaftConfig::default().serve_quant, None);
        // Serving reads ahead; nothing else does unless it asks.
        assert!(cfg.read_ahead && !RaftConfig::default().read_ahead);
        Ok(())
    }
}
