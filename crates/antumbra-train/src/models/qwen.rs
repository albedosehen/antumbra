//! candle Qwen2.5-Coder + LoRA: the real [`CausalLm`] (the candle QLoRA trainer, MT-1).
//!
//! Architecture vendored from candle-transformers' `qwen2` (GQA attention,
//! RoPE, RMSNorm, KV cache) with the seven projection linears per layer swapped
//! for [`LoraLinear`]: the base weights load **frozen** from the HF safetensors;
//! only the LoRA `A`/`B` factors are trainable `Var`s, and every product
//! against a base weight goes through [`frozen_matmul_t`], which computes no
//! gradient for the weight. Generation uses the KV cache; `sft_step` runs a
//! full-sequence forward, the completion-masked cross-entropy
//! ([`crate::objective::causal_lm_loss`]), and a backward over the LoRA
//! `VarMap`. Validated on the 3090 Ti (needs weights).

use std::path::PathBuf;
use std::sync::Arc;

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, Result as CResult, Tensor, D};
use candle_nn::init::Init;
use candle_nn::{Activation, Embedding, Module, Optimizer, VarBuilder, VarMap};

use rand::distr::weighted::WeightedIndex;
use rand::distr::Distribution;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use async_trait::async_trait;

use antumbra_core::{AntumbraError, Result};

use crate::accumulate::GradSum;
use crate::config::RaftConfig;
use crate::frozen::frozen_matmul_t;
use crate::grpo::{token_logprobs, GrpoExperience, GrpoLm, GrpoSample};
use crate::model::{CausalLm, SftExample};
use crate::objective::causal_lm_loss;

fn ce(err: candle_core::Error) -> AntumbraError {
    AntumbraError::other(format!("candle: {err}"))
}

// --- config ----------------------------------------------------------------

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    #[serde(default = "default_act")]
    pub hidden_act: Activation,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
}

fn default_act() -> Activation {
    Activation::Silu
}
fn default_true() -> bool {
    true
}

mod layers;
use layers::*;

// --- the model -------------------------------------------------------------

pub struct QwenLora {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head_w: Tensor,
    device: Device,
    dtype: DType,
    varmap: VarMap,
}

impl QwenLora {
    fn build(
        cfg: &Config,
        base: &VarBuilder,
        lora_vb: &VarBuilder,
        varmap: VarMap,
        rank: usize,
        scale: f64,
    ) -> CResult<Self> {
        let vb_m = base.pp("model");
        let lora_m = lora_vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary = Arc::new(RotaryEmbedding::new(base.dtype(), cfg, base.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::load(
                rotary.clone(),
                cfg,
                rank,
                scale,
                &vb_m.pp("layers").pp(i),
                &lora_m.pp("layers").pp(i),
            )?);
        }
        let norm = RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, &vb_m.pp("norm"))?;
        let lm_head_w = if base.contains_tensor("lm_head.weight") {
            base.get((cfg.vocab_size, cfg.hidden_size), "lm_head.weight")?
        } else {
            embed_tokens.embeddings().clone()
        };
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head_w,
            device: base.device().clone(),
            dtype: base.dtype(),
            varmap,
        })
    }

    fn causal_mask(&self, seq: usize, offset: usize) -> CResult<Tensor> {
        let mask: Vec<f32> = (0..seq)
            .flat_map(|i| (0..seq).map(move |j| if i < j { f32::NEG_INFINITY } else { 0.0 }))
            .collect();
        let mask = Tensor::from_slice(&mask, (seq, seq), &self.device)?;
        let mask = if offset > 0 {
            let zeros = Tensor::zeros((seq, offset), DType::F32, &self.device)?;
            Tensor::cat(&[&zeros, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.to_dtype(self.dtype)
    }

    /// Hidden states for every position: `(b, seq, hidden)`.
    fn hidden(&mut self, input_ids: &Tensor, offset: usize, use_cache: bool) -> CResult<Tensor> {
        let (_b, seq) = input_ids.dims2()?;
        let mask = if seq <= 1 {
            None
        } else {
            Some(self.causal_mask(seq, offset)?)
        };
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, mask.as_ref(), offset, use_cache)?;
        }
        self.norm.forward(&xs)
    }

    fn clear_cache(&mut self) {
        for l in self.layers.iter_mut() {
            l.clear_cache();
        }
    }

    /// Toggle the LoRA delta across every projection. Off = the frozen base
    /// alone (the GRPO reference policy); on = base + adapter (the policy).
    fn set_lora(&mut self, on: bool) {
        for l in self.layers.iter_mut() {
            l.set_lora(on);
        }
    }

    /// Toggle autograd tracking across every projection. Off for generation:
    /// the forward is detached, so nothing is retained between tokens (critical
    /// for the quantized base, where each token re-materializes the weights).
    fn set_grad(&mut self, on: bool) {
        for l in self.layers.iter_mut() {
            l.set_grad(on);
        }
    }

    /// Quantize every projection's frozen base weight to 4-bit (v1 efficiency).
    fn quantize_base(&mut self, device: &Device) -> CResult<()> {
        for l in self.layers.iter_mut() {
            l.quantize_base(device)?;
        }
        Ok(())
    }
}

// --- the trainer-facing wrapper (load + generate + train) ------------------

pub struct QwenCausalLm {
    model: QwenLora,
    tokenizer: tokenizers::Tokenizer,
    opt: candle_nn::AdamW,
    eos: u32,
    max_new_tokens: usize,
    temperature: f64,
    top_p: f64,
    repetition_penalty: f64,
    no_repeat_ngram_size: usize,
    /// `true` for an `-Instruct` base: prompts are wrapped in the Qwen chat
    /// template (so the model is prompted the way it was tuned) and generation
    /// stops at `<|im_end|>` instead of `<|endoftext|>`.
    chat: bool,
    /// Verified winners per optimizer step: 1 steps on each in turn, more steps
    /// on the mean gradient of that many at a time. See [`RaftConfig::batch_size`].
    batch_size: usize,
    /// When set by [`CausalLm::seed_draws`], the seed later draws come from and
    /// how many have been taken from it; otherwise draws come from the
    /// process-wide nonce.
    draws: Option<(u64, u64)>,
}

/// Process-global generation nonce, so every `generate` call (even repeated
/// single-sample calls from a best-of-K probe over freshly-loaded models) seeds
/// a distinct RNG and yields a *different* draw.
static GEN_NONCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl QwenCausalLm {
    /// The seed the `i`-th draw of a call samples from. Seeded (`seed_draws`),
    /// the n-th draw since takes the n-th seed of that stream, so the same seed
    /// repeats the same draws. Otherwise a fresh process-wide nonce per draw,
    /// so repeated calls (best-of-K over reloaded models) diverge instead of
    /// all seeding identically.
    fn next_draw_seed(&mut self, i: usize) -> u64 {
        match &mut self.draws {
            Some((base, taken)) => {
                let seed = base
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(*taken);
                *taken += 1;
                seed
            }
            None => {
                let nonce = GEN_NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                0xA17_u64
                    .wrapping_mul(nonce.wrapping_add(1))
                    .wrapping_add(i as u64)
            }
        }
    }

    /// Load Qwen2.5-Coder + a fresh LoRA adapter from the Hugging Face hub.
    pub fn load(device: Device, cfg: RaftConfig) -> Result<Self> {
        // hf-hub 1.0's blocking client `block_on`s an async runtime, which panics
        // inside an existing tokio runtime (the CLI/MCP load from async). Run the
        // downloads on a dedicated thread that has no ambient runtime.
        let base_model = cfg.base_model.as_str();
        let (tok_path, cfg_path, weights_path) = std::thread::scope(|s| {
            s.spawn(|| -> Result<(PathBuf, PathBuf, PathBuf)> {
                let client = hf_hub::HFClientSync::new()
                    .map_err(|e| AntumbraError::other(format!("hf-hub: {e}")))?;
                let (owner, name) = base_model.split_once('/').unwrap_or(("", base_model));
                let repo = client.model(owner, name);
                let tok = repo
                    .download_file()
                    .filename("tokenizer.json")
                    .send()
                    .map_err(|e| AntumbraError::other(format!("hf-hub tokenizer: {e}")))?;
                let config = repo
                    .download_file()
                    .filename("config.json")
                    .send()
                    .map_err(|e| AntumbraError::other(format!("hf-hub config: {e}")))?;
                let weights = repo
                    .download_file()
                    .filename("model.safetensors")
                    .send()
                    .map_err(|e| AntumbraError::other(format!("hf-hub weights: {e}")))?;
                Ok((tok, config, weights))
            })
            .join()
        })
        .map_err(|_| AntumbraError::other("hf-hub download thread panicked".to_string()))??;
        let weights: Vec<PathBuf> = vec![weights_path];

        let tokenizer = tokenizers::Tokenizer::from_file(tok_path)
            .map_err(|e| AntumbraError::other(format!("tokenizer: {e}")))?;
        let config_bytes =
            std::fs::read(cfg_path).map_err(|e| AntumbraError::other(e.to_string()))?;
        let model_cfg: Config = serde_json::from_slice(&config_bytes)?;

        // Train in f32: keeps the frozen base, the LoRA factors, and the loss in
        // one precision (candle autograd flows through the f32 base matmuls into
        // the LoRA factors; the base, not being a Var, gets no gradient). On CPU
        // force f32 regardless of config; f16/bf16 only pay off on a GPU.
        let dtype = if matches!(device, Device::Cpu) {
            DType::F32
        } else {
            cfg.dtype.to_candle()
        };
        let base =
            unsafe { VarBuilder::from_mmaped_safetensors(&weights, dtype, &device).map_err(ce)? };
        let varmap = VarMap::new();
        let lora_vb = VarBuilder::from_varmap(&varmap, dtype, &device);
        let mut model = QwenLora::build(
            &model_cfg,
            &base,
            &lora_vb,
            varmap,
            cfg.lora_rank,
            cfg.lora_scale(),
        )
        .map_err(ce)?;
        // 4-bit QLoRA base (v1 efficiency): dequant-in-forward. A capacity lever for
        // larger bases; off by default on the 1.5B base.
        if cfg.quantize_base {
            model.quantize_base(&device).map_err(ce)?;
        }

        let opt = candle_nn::AdamW::new(
            model.varmap.all_vars(),
            candle_nn::ParamsAdamW {
                lr: cfg.learning_rate,
                ..Default::default()
            },
        )
        .map_err(ce)?;

        // An `-Instruct` base is chat-tuned: prompt it with the chat template and
        // stop the assistant turn at `<|im_end|>`.
        let chat = cfg.base_model.contains("Instruct");
        let eos_token = if chat { "<|im_end|>" } else { "<|endoftext|>" };
        let eos = tokenizer
            .token_to_id(eos_token)
            .ok_or_else(|| AntumbraError::other(format!("tokenizer missing {eos_token}")))?;

        Ok(Self {
            model,
            tokenizer,
            opt,
            eos,
            max_new_tokens: cfg.max_new_tokens,
            temperature: cfg.temperature,
            top_p: cfg.top_p,
            repetition_penalty: cfg.repetition_penalty,
            no_repeat_ngram_size: cfg.no_repeat_ngram_size,
            chat,
            batch_size: cfg.batch_size.max(1),
            draws: None,
        })
    }

    /// Load trained LoRA weights (saved by `save_adapter`) into this model's
    /// VarMap by matching variable names. Used to serve a graduated expert and
    /// to warm-start continual training from a parent adapter. The backend and
    /// dtype must match the ones the adapter was trained with.
    pub fn load_adapter(&mut self, path: &str) -> Result<()> {
        self.model
            .varmap
            .load(path)
            .map_err(|e| AntumbraError::other(format!("load adapter `{path}`: {e}")))
    }

    /// Wrap a raw prompt in the Qwen chat template for an `-Instruct` base, so
    /// the model is prompted the way it was tuned (a `user` turn, then the open
    /// `assistant` turn it completes). A plain base sees the prompt unchanged.
    /// Applied at *every* prompt-encoding site so training and serving agree.
    /// Trim generated text to the completion that is verified and trained on.
    /// A chat model's code sits inside a fence, which the completion-model
    /// stops would cut away; see [`crate::decode::trim_chat_answer`].
    fn trim(&self, text: &str) -> String {
        if self.chat {
            crate::decode::trim_chat_answer(text)
        } else {
            crate::decode::truncate_at_stops(text, crate::decode::DEFAULT_STOPS)
        }
    }

    fn wrap_prompt(&self, prompt: &str) -> String {
        if self.chat {
            format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
        } else {
            prompt.to_string()
        }
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text, true)
            .map_err(|e| AntumbraError::other(format!("encode: {e}")))?
            .get_ids()
            .to_vec())
    }

    fn sample_one(&mut self, prompt: &str, seed: u64) -> Result<String> {
        self.model.clear_cache();
        self.model.set_grad(false); // generation needs no gradients
        let mut tokens = self.encode(&self.wrap_prompt(prompt))?;
        let prompt_len = tokens.len();
        let mut rng = StdRng::seed_from_u64(seed);
        let policy = crate::decode::DecodePolicy {
            temperature: self.temperature,
            top_p: self.top_p,
            repetition_penalty: self.repetition_penalty,
            no_repeat_ngram_size: self.no_repeat_ngram_size,
        };

        for index in 0..self.max_new_tokens {
            let ctx_len = if index == 0 { tokens.len() } else { 1 };
            let start = tokens.len() - ctx_len;
            let input = Tensor::new(&tokens[start..], &self.model.device)
                .map_err(ce)?
                .unsqueeze(0)
                .map_err(ce)?;
            let hidden = self.model.hidden(&input, start, true).map_err(ce)?;
            let last = hidden.narrow(1, ctx_len - 1, 1).map_err(ce)?;
            let logits = frozen_matmul_t(&last, &self.model.lm_head_w)
                .map_err(ce)?
                .squeeze(0)
                .map_err(ce)?
                .squeeze(0)
                .map_err(ce)?
                .to_dtype(DType::F32)
                .map_err(ce)?;
            // Decode policy over the generated continuation only (not the prompt).
            let logits_vec: Vec<f32> = logits.to_vec1().map_err(ce)?;
            let next =
                crate::decode::pick_token(logits_vec, &tokens[prompt_len..], &policy, &mut rng);
            if next == self.eos {
                break;
            }
            tokens.push(next);
        }

        let text = self
            .tokenizer
            .decode(&tokens[prompt_len..], true)
            .map_err(|e| AntumbraError::other(format!("decode: {e}")))?;
        // Trim the run-on so the verified, trained-on completion is the clean one.
        Ok(self.trim(&text))
    }

    /// Full-sequence logits `(1, seq, vocab)` in f32, LoRA on (the policy) or
    /// off (the GRPO reference). No KV cache: we score the whole sequence. The
    /// LoRA gate is restored to on afterwards.
    fn forward_logits(&mut self, ids: &[u32], lora_on: bool) -> Result<Tensor> {
        self.model.clear_cache();
        self.model.set_lora(lora_on);
        self.model.set_grad(true); // scoring forward: the policy pass needs grad
        let input = Tensor::new(ids, &self.model.device)
            .map_err(ce)?
            .unsqueeze(0)
            .map_err(ce)?;
        let logits = (|| {
            let hidden = self.model.hidden(&input, 0, false)?;
            frozen_matmul_t(&hidden, &self.model.lm_head_w)?.to_dtype(DType::F32)
        })();
        self.model.set_lora(true);
        logits.map_err(ce)
    }

    /// Sample one completion, capturing each generated token's `π_old` log-prob
    /// (under the model's softmax, temperature aside) for the GRPO ratio.
    fn sample_one_with_logprobs(&mut self, prompt: &str, seed: u64) -> Result<GrpoSample> {
        self.model.clear_cache();
        self.model.set_lora(true);
        self.model.set_grad(false); // π_old is a fixed snapshot; no graph needed
        let mut tokens = self.encode(&self.wrap_prompt(prompt))?;
        let mut gen_tokens: Vec<u32> = Vec::new();
        let mut old_logprobs: Vec<f32> = Vec::new();
        let mut rng = StdRng::seed_from_u64(seed);
        let temp = self.temperature;

        for index in 0..self.max_new_tokens {
            let ctx_len = if index == 0 { tokens.len() } else { 1 };
            let start = tokens.len() - ctx_len;
            let input = Tensor::new(&tokens[start..], &self.model.device)
                .map_err(ce)?
                .unsqueeze(0)
                .map_err(ce)?;
            let hidden = self.model.hidden(&input, start, true).map_err(ce)?;
            let last = hidden.narrow(1, ctx_len - 1, 1).map_err(ce)?;
            let logits = frozen_matmul_t(&last, &self.model.lm_head_w)
                .map_err(ce)?
                .squeeze(0)
                .map_err(ce)?
                .squeeze(0)
                .map_err(ce)?
                .to_dtype(DType::F32)
                .map_err(ce)?;
            let next = sample_token(&logits, temp, &mut rng)?;
            if next == self.eos {
                break;
            }
            let lp = candle_nn::ops::log_softmax(&logits, 0)
                .map_err(ce)?
                .get(next as usize)
                .map_err(ce)?
                .to_scalar::<f32>()
                .map_err(ce)?;
            tokens.push(next);
            gen_tokens.push(next);
            old_logprobs.push(lp);
        }

        let text = self
            .tokenizer
            .decode(&gen_tokens, true)
            .map_err(|e| AntumbraError::other(format!("decode: {e}")))?;
        Ok(GrpoSample {
            completion: self.trim(&text),
            tokens: gen_tokens,
            old_logprobs,
        })
    }

    fn write_adapter(&self, path: &str) -> Result<()> {
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent).map_err(|e| AntumbraError::other(e.to_string()))?;
        }
        self.model
            .varmap
            .save(path)
            .map_err(|e| AntumbraError::other(format!("save adapter: {e}")))
    }

    /// Forward + masked LM loss for one example, with autograd tracked. Returns
    /// the loss tensor (graph attached) or `None` if the example is too short to
    /// supervise. The caller drives backward/step -- per example (`train_one`) or
    /// accumulated over a batch (`sft_step` with a `batch_size` above one).
    fn forward_loss(&mut self, example: &SftExample) -> Result<Option<Tensor>> {
        self.model.clear_cache();
        self.model.set_grad(true); // training forward must be tracked
        let wrapped = self.wrap_prompt(&example.prompt);
        let prompt_ids = self.encode(&wrapped)?;
        let full_text = format!("{}{}", wrapped, example.completion);
        let mut full_ids = self.encode(&full_text)?;
        if full_ids.len() <= prompt_ids.len() || full_ids.len() < 2 {
            return Ok(None);
        }
        // Supervise an EOS after the completion so the model learns to *stop*
        // there; without it a short taught completion runs on and degenerates.
        full_ids.push(self.eos);

        // Robust completion boundary: tokenizing the prompt alone vs inside the
        // full text can merge/split the boundary token (e.g. ": " + "bun"), so
        // the prompt length mis-masks the first completion token. Supervise
        // everything past the shared prefix -- the true completion start.
        let boundary = prompt_ids
            .iter()
            .zip(full_ids.iter())
            .take_while(|(a, b)| a == b)
            .count();
        let mask: Vec<f32> = (0..full_ids.len())
            .map(|i| if i >= boundary { 1.0 } else { 0.0 })
            .collect();
        let input = Tensor::new(full_ids.as_slice(), &self.model.device)
            .map_err(ce)?
            .unsqueeze(0)
            .map_err(ce)?;
        let mask = Tensor::from_vec(mask, (1, full_ids.len()), &self.model.device).map_err(ce)?;

        let hidden = self.model.hidden(&input, 0, false).map_err(ce)?;
        let logits = frozen_matmul_t(&hidden, &self.model.lm_head_w)
            .map_err(ce)?
            .to_dtype(DType::F32)
            .map_err(ce)?;
        let loss = causal_lm_loss(&logits, &input, &mask).map_err(ce)?;
        Ok(Some(loss))
    }

    /// One example, one optimizer step (batch-of-1 SGD). The per-example path.
    fn train_one(&mut self, example: &SftExample) -> Result<f32> {
        match self.forward_loss(example)? {
            Some(loss) => {
                self.opt.backward_step(&loss).map_err(ce)?;
                loss.to_scalar::<f32>().map_err(ce)
            }
            None => Ok(0.0),
        }
    }

    /// Shuffled mini-batch SGD: one step on the **mean gradient of `per_step`
    /// examples** rather than one example at a time. It is a genuine mini-batch
    /// update -- far less noisy than batch-of-1, stable at a higher learning rate
    /// ("Beware of the Batch Size") and free of last-example dominance within
    /// the group.
    ///
    /// Each example is backpropagated as soon as its loss is computed and its
    /// adapter gradients summed ([`GradSum`]), so only one forward graph is
    /// alive at a time and a group of any size costs one example's memory.
    /// Stepping on the mean loss instead held every example's graph until the
    /// one backward, and two examples of workbench code ran the 24 GB card out.
    /// Groups rather than the whole batch still, because a group is the batch
    /// size the recipe asks for. The batch is shuffled first so groups are
    /// random across rounds. Returns the mean supervised loss over the steps
    /// taken.
    fn train_batch(&mut self, batch: &[SftExample], per_step: usize) -> Result<f32> {
        let nonce = GEN_NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut order: Vec<usize> = (0..batch.len()).collect();
        let mut rng = StdRng::seed_from_u64(0x5F37_u64.wrapping_mul(nonce.wrapping_add(1)));
        order.shuffle(&mut rng);

        let vars = self.model.varmap.all_vars();
        let mut total = 0.0f32;
        let mut steps = 0usize;
        for group in order.chunks(per_step.max(1)) {
            let mut sum = GradSum::new();
            let mut group_loss = 0.0f32;
            for &i in group {
                if let Some(loss) = self.forward_loss(&batch[i])? {
                    group_loss += loss.to_scalar::<f32>().map_err(ce)?;
                    sum.add(&loss, &vars).map_err(ce)?;
                }
            }
            let n = sum.len();
            if sum.step(&mut self.opt, &vars).map_err(ce)? {
                total += group_loss / n as f32;
                steps += 1;
            }
        }
        if steps == 0 {
            return Ok(0.0);
        }
        Ok(total / steps as f32)
    }
}

/// Temperature sampling from logits over the vocabulary.
fn sample_token(logits: &Tensor, temp: f64, rng: &mut StdRng) -> Result<u32> {
    if temp <= 0.0 {
        return logits
            .argmax(D::Minus1)
            .map_err(ce)?
            .to_scalar::<u32>()
            .map_err(ce);
    }
    let scaled = (logits / temp).map_err(ce)?;
    let probs = candle_nn::ops::softmax_last_dim(&scaled).map_err(ce)?;
    let probs: Vec<f32> = probs.to_vec1().map_err(ce)?;
    // Defensive: if the distribution is degenerate (NaN/Inf/all-zero logits),
    // fall back to greedy rather than erroring.
    match WeightedIndex::new(&probs) {
        Ok(dist) => Ok(dist.sample(rng) as u32),
        Err(_) => logits
            .argmax(D::Minus1)
            .map_err(ce)?
            .to_scalar::<u32>()
            .map_err(ce),
    }
}

#[async_trait]
impl CausalLm for QwenCausalLm {
    async fn generate(&mut self, prompt: &str, n_samples: usize) -> Result<Vec<String>> {
        let mut out = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let seed = self.next_draw_seed(i);
            out.push(self.sample_one(prompt, seed)?);
        }
        Ok(out)
    }

    async fn sft_step(&mut self, batch: &[SftExample]) -> Result<f32> {
        if batch.is_empty() {
            return Ok(0.0);
        }
        // True mini-batch descent: accumulate the batch gradient, one step. Order
        // is irrelevant (summation commutes), and no example dominates by being
        // trained last, so no shuffle is needed.
        if self.batch_size > 1 {
            return self.train_batch(batch, self.batch_size);
        }
        // Each example is one SGD step (batch-of-1). Shuffle the order every call
        // so no single example is consistently trained *last* and dominates the
        // LoRA; otherwise the adapter collapses to the final example instead of
        // learning the prompt-conditioned mapping. The nonce varies the shuffle
        // per round.
        let nonce = GEN_NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut order: Vec<usize> = (0..batch.len()).collect();
        let mut rng = StdRng::seed_from_u64(0x5F37_u64.wrapping_mul(nonce.wrapping_add(1)));
        order.shuffle(&mut rng);
        let mut total = 0.0f32;
        for &i in &order {
            total += self.train_one(&batch[i])?;
        }
        Ok(total / batch.len() as f32)
    }

    fn save_adapter(&self, path: &str) -> Result<()> {
        self.write_adapter(path)
    }

    fn seed_draws(&mut self, seed: u64) -> Result<()> {
        self.draws = Some((seed, 0));
        Ok(())
    }
}

#[async_trait]
impl GrpoLm for QwenCausalLm {
    async fn sample_group(&mut self, prompt: &str, group: usize) -> Result<Vec<GrpoSample>> {
        let mut out = Vec::with_capacity(group);
        for i in 0..group {
            let seed = self.next_draw_seed(i);
            out.push(self.sample_one_with_logprobs(prompt, seed)?);
        }
        Ok(out)
    }

    fn seed_draws(&mut self, seed: u64) -> Result<()> {
        self.draws = Some((seed, 0));
        Ok(())
    }

    async fn reference_logprobs(&mut self, prompt: &str, tokens: &[u32]) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        let prompt_ids = self.encode(&self.wrap_prompt(prompt))?;
        let plen = prompt_ids.len();
        let mut ids = prompt_ids;
        ids.extend_from_slice(tokens);
        let logits = self.forward_logits(&ids, false)?; // base only
        let input = Tensor::new(ids.as_slice(), &self.model.device)
            .map_err(ce)?
            .unsqueeze(0)
            .map_err(ce)?;
        let all_lp = token_logprobs(&logits, &input).map_err(ce)?;
        all_lp
            .narrow(1, plen - 1, tokens.len())
            .map_err(ce)?
            .flatten_all()
            .map_err(ce)?
            .to_vec1::<f32>()
            .map_err(ce)
    }

    async fn grpo_step(
        &mut self,
        prompt: &str,
        group: &[GrpoExperience],
        cfg: &RaftConfig,
    ) -> Result<f32> {
        let dev = self.model.device.clone();
        let prompt_ids = self.encode(&self.wrap_prompt(prompt))?;
        let plen = prompt_ids.len();
        // One backward step per group member (as RAFT steps per winner), so only
        // a single forward graph is alive at a time; a combined-group backward
        // holds G forwards and OOMs the card.
        let (mut total, mut steps) = (0.0f32, 0usize);
        for exp in group {
            let t = exp.tokens.len();
            if t == 0 || exp.old_logprobs.len() != t || exp.ref_logprobs.len() != t {
                continue;
            }
            let mut ids = prompt_ids.clone();
            ids.extend_from_slice(&exp.tokens);
            let logits = self.forward_logits(&ids, true)?; // policy, grad on LoRA
            let input = Tensor::new(ids.as_slice(), &dev)
                .map_err(ce)?
                .unsqueeze(0)
                .map_err(ce)?;
            let policy_lp = token_logprobs(&logits, &input)
                .map_err(ce)?
                .narrow(1, plen - 1, t)
                .map_err(ce)?;
            let old_lp = Tensor::from_vec(exp.old_logprobs.clone(), (1, t), &dev).map_err(ce)?;
            let ref_lp = Tensor::from_vec(exp.ref_logprobs.clone(), (1, t), &dev).map_err(ce)?;
            let adv = Tensor::from_vec(vec![exp.advantage], (1, 1), &dev).map_err(ce)?;
            let mask = Tensor::ones((1, t), DType::F32, &dev).map_err(ce)?;
            let loss = crate::grpo::grpo_loss(
                &policy_lp,
                &old_lp,
                &ref_lp,
                &adv,
                &mask,
                cfg.clip_eps,
                cfg.kl_beta,
            )
            .map_err(ce)?;
            self.opt.backward_step(&loss).map_err(ce)?;
            total += loss.to_scalar::<f32>().map_err(ce)?;
            steps += 1;
        }
        if steps == 0 {
            return Ok(0.0);
        }
        Ok(total / steps as f32)
    }

    fn save_adapter(&self, path: &str) -> Result<()> {
        self.write_adapter(path)
    }
}
