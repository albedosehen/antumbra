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

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
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
use crate::config::{RaftConfig, ServeQuant};
use crate::frozen::frozen_matmul_t;
use crate::grpo::{token_logprobs, GrpoExperience, GrpoLm, GrpoSample};
use crate::model::{CausalLm, SftExample};
use crate::objective::causal_lm_loss;

fn ce(err: candle_core::Error) -> AntumbraError {
    crate::candle_err("candle")(err)
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

mod adapt;
mod ahead;
mod attend;
mod kv;
mod layers;
mod pack;
mod phases;
mod rowwise;
mod sketch;
mod slices;
mod tiles;
#[cfg(target_arch = "x86_64")]
mod wide;
use layers::*;

// --- the model -------------------------------------------------------------

/// The form the head is served in over a body served in `body`. The head
/// picks every token, so it is held finer than a four-bit body: at six bits
/// (Q6_K), the form a Q4_K_M file gives its output weight, and 57 MB a token
/// less to read on a 1.5B base than the eight bits it was held at before.
/// Over a six-bit body it is six bits, as the body is: at eight one expert's
/// log-probabilities shifted from the dense base's by 0.140 nats on average
/// where at six they shift by 0.144, less than the order a sum is added in
/// moves them, for those 57 MB a token. Over an eight-bit body it is eight
/// bits, as the body is: at six they shifted by 0.105 where at eight they
/// shift by 0.098.
fn head_form(body: GgmlDType) -> GgmlDType {
    match body {
        GgmlDType::Q4K | GgmlDType::Q6K => GgmlDType::Q6K,
        _ => GgmlDType::Q8_0,
    }
}

pub struct QwenLora {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head_w: Tensor,
    /// The head as a model loaded to serve reads it ([`head_form`]), in place
    /// of the dense `lm_head_w`. Either this or `lm_head_t` is `Some` exactly
    /// when the model was loaded to serve.
    lm_head_q: Option<QMatMul>,
    /// The served head as tiles, in place of `lm_head_q`, on a CPU that can
    /// multiply through them.
    lm_head_t: Option<tiles::Tiled>,
    /// The tiled head's rows as a sketch, where this CPU multiplies one: a
    /// token that is the head's first is chosen through it, from the dense
    /// rows of `lm_head_w`.
    lm_head_s: Option<sketch::Sketch>,
    /// Whether the model was loaded to serve: its base is quantized and
    /// carries no gradient.
    serves: bool,
    /// What rows are computed in where they are stepped through as plain
    /// values, kept from one token to the next.
    scratch: Scratch,
    device: Device,
    dtype: DType,
    varmap: VarMap,
}

impl QwenLora {
    /// With `serve`, every projection's base is quantized to that type as its
    /// layer loads (so the dense base is never held whole) and multiplied in
    /// that form, and the head is held in the form that goes with it
    /// ([`head_form`]): the model can only serve.
    fn build(
        cfg: &Config,
        base: &VarBuilder,
        lora_vb: &VarBuilder,
        varmap: VarMap,
        rank: usize,
        scale: f64,
        serve: Option<GgmlDType>,
    ) -> CResult<Self> {
        let vb_m = base.pp("model");
        let lora_m = lora_vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary = Arc::new(RotaryEmbedding::new(base.dtype(), cfg, base.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let mut layer = DecoderLayer::load(
                rotary.clone(),
                cfg,
                rank,
                scale,
                &vb_m.pp("layers").pp(i),
                &lora_m.pp("layers").pp(i),
            )?;
            if let Some(dtype) = serve {
                layer.serve_quantized(dtype)?;
            }
            layers.push(layer);
        }
        let norm = RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, &vb_m.pp("norm"))?;
        let lm_head_w = if base.contains_tensor("lm_head.weight") {
            base.get((cfg.vocab_size, cfg.hidden_size), "lm_head.weight")?
        } else {
            embed_tokens.embeddings().clone()
        };
        // Tied to the embeddings, the head is the largest weight there is, and
        // read whole every token; the lookup still needs the dense rows.
        let (lm_head_q, lm_head_t, lm_head_s) = match serve {
            Some(body) => {
                let form = head_form(body);
                let cpu = lm_head_w.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
                let tiled = match base.device().is_cpu() {
                    true => tiles::Tiled::quantize(&cpu, form)?,
                    false => None,
                };
                match tiled {
                    Some(tiled) => (None, Some(tiled), sketch::Sketch::of(&cpu)?),
                    None => {
                        let q = QTensor::quantize_onto(&cpu, form, base.device())?;
                        (Some(QMatMul::from_qtensor(q)?), None, None)
                    }
                }
            }
            None => (None, None, None),
        };
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head_w,
            lm_head_q,
            lm_head_t,
            lm_head_s,
            serves: serve.is_some(),
            scratch: Scratch::default(),
            device: base.device().clone(),
            dtype: base.dtype(),
            varmap,
        })
    }

    /// Logits from hidden states, through the quantized head when the model
    /// was loaded to serve.
    fn logits(&self, hidden: &Tensor) -> CResult<Tensor> {
        if let Some(tiled) = &self.lm_head_t {
            return one(tiles::forward(&[tiled], hidden)?);
        }
        match &self.lm_head_q {
            Some(q) => served_matmul(q, hidden),
            None => frozen_matmul_t(hidden, &self.lm_head_w),
        }
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

    /// The adapter's factors were set: the copies plain rows read of them
    /// are let go, and made again when rows are next read.
    fn adapters_changed(&mut self) {
        for l in self.layers.iter_mut() {
            l.forget_packs();
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

/// How long a generation's two phases took: reading the prompt in one pass,
/// which produces the first token, and each token after it. They are bound by
/// different things on a CPU (the first by compute, the second by how fast the
/// weights can be read), so one rate over both describes neither.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GenTiming {
    pub prompt_tokens: usize,
    pub prompt_secs: f64,
    /// Tokens decoded after the first.
    pub decoded: usize,
    pub decode_secs: f64,
    /// Passes over the weights those tokens took: as many as the tokens
    /// where each has a pass of its own, fewer where a model reads ahead.
    pub passes: usize,
}

impl GenTiming {
    fn record(&mut self, index: usize, ctx_len: usize, took: std::time::Duration) {
        if index == 0 {
            self.prompt_tokens = ctx_len;
            self.prompt_secs = took.as_secs_f64();
        } else {
            self.passed(1, took);
        }
    }

    /// One pass over the weights that decided `tokens` tokens.
    fn passed(&mut self, tokens: usize, took: std::time::Duration) {
        self.decoded += tokens;
        self.decode_secs += took.as_secs_f64();
        self.passes += 1;
    }
}

/// How many blends of experts a model remembers its own writing for.
const BLENDS_REMEMBERED: usize = 8;

/// What a model that reads ahead guesses from: what it wrote before, apart
/// for each blend of experts it wrote through, and room for the generation
/// in hand.
#[derive(Default)]
struct Memories {
    /// The generation in hand: its prompt and what it has written.
    near: crate::draft::Drafts,
    /// What was written through each blend, the latest used last.
    far: Vec<(String, crate::draft::Drafts)>,
    /// What guesses have been worth on this machine, whatever the blend.
    odds: ahead::Odds,
}

impl Memories {
    /// Where `blend`'s memory is, made if it is new, and moved to the end as
    /// the latest used. The blend longest unused is forgotten for it.
    fn place(&mut self, blend: &str) -> usize {
        let held = match self.far.iter().position(|(name, _)| name == blend) {
            Some(at) => self.far.remove(at),
            None => (blend.to_string(), crate::draft::Drafts::new()),
        };
        if self.far.len() >= BLENDS_REMEMBERED {
            self.far.remove(0);
        }
        self.far.push(held);
        self.far.len() - 1
    }
}

/// One request being answered beside others.
struct Sequence {
    tokens: Vec<u32>,
    prompt_len: usize,
    blend: u16,
    rng: StdRng,
    /// The answer, once the request has ended.
    answer: Option<String>,
}

/// The requests one model is answering side by side
/// ([`QwenCausalLm::admit`], [`QwenCausalLm::advance`]): a place for each, a
/// request's own from the pass it is taken in to the moment its answer is
/// taken.
#[derive(Default)]
pub struct Together {
    places: Vec<Option<Sequence>>,
    /// The prompts' tokens and the time they took to read, and the tokens
    /// decoded after each request's first over the time the shared passes
    /// took.
    pub timing: GenTiming,
}

impl Together {
    /// Requests one pass takes at most.
    pub const AT_ONCE: usize = 32;

    /// A place no request holds, if there is one.
    fn free(&mut self) -> Option<usize> {
        if let Some(place) = self.places.iter().position(Option::is_none) {
            return Some(place);
        }
        (self.places.len() < Self::AT_ONCE).then(|| {
            self.places.push(None);
            self.places.len() - 1
        })
    }

    /// Requests still taking tokens.
    pub fn answering(&self) -> usize {
        self.places
            .iter()
            .flatten()
            .filter(|s| s.answer.is_none())
            .count()
    }

    /// Requests that hold a place: answering, or ended with their answer
    /// not yet taken.
    pub fn held(&self) -> usize {
        self.places.iter().flatten().count()
    }

    /// Let go of the request at `place`, answered or not: it takes no more
    /// tokens and its place is free for another, whose prompt is read into
    /// a cache cleared for it.
    pub fn leave(&mut self, place: usize) {
        if let Some(slot) = self.places.get_mut(place) {
            *slot = None;
        }
    }

    /// The answer of the request at `place`, once it has ended, and its
    /// place with it: free for another request from then on. `None` while
    /// it is still answering, or when no request is there.
    pub fn take(&mut self, place: usize) -> Option<String> {
        let slot = self.places.get_mut(place)?;
        let answer = slot.as_mut()?.answer.take()?;
        *slot = None;
        Some(answer)
    }
}

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
    /// When the run is seeded ([`RaftConfig::seed`]), the seed its training
    /// shuffles come from and how many it has taken; otherwise each shuffle
    /// comes from the process-wide nonce.
    train_order: Option<(u64, u64)>,
    /// The phases of the last generation, for a caller measuring serving.
    timing: Option<GenTiming>,
    /// What greedy generation reads ahead with ([`RaftConfig::read_ahead`]),
    /// when it does.
    ahead: Option<Memories>,
    /// The blend of experts the adapter is set to, as a name, and whether
    /// the adapter is on: which memory a generation guesses from and adds
    /// to. Writing through one blend is never guessed from under another.
    blend: String,
    adapter_on: bool,
}

/// What a seeded run's two streams are told apart by: the adapter's starting
/// factors and the order it trains in.
const INIT_STREAM: u64 = 0x1A17;
const ORDER_STREAM: u64 = 0x5F37;

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
                crate::seed::shifted(
                    0xA17_u64
                        .wrapping_mul(nonce.wrapping_add(1))
                        .wrapping_add(i as u64),
                    crate::seed::generation_seed(),
                )
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
            cfg.serve_quant.map(ServeQuant::to_ggml),
        )
        .map_err(ce)?;
        // 4-bit QLoRA base (v1 efficiency): dequant-in-forward. A capacity lever for
        // larger bases; off by default on the 1.5B base. A model loaded to serve
        // has its base quantized already.
        if cfg.quantize_base && cfg.serve_quant.is_none() {
            model.quantize_base(&device).map_err(ce)?;
        }
        // A seeded run starts from factors drawn from its seed, not from
        // wherever the device's generator has got to in this process.
        let seeded = cfg
            .seed
            .map(|s| crate::seed::shifted(s, crate::seed::generation_seed()));
        if let Some(s) = seeded {
            seed_lora_a(&model.varmap, s ^ INIT_STREAM).map_err(ce)?;
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
            train_order: seeded.map(|s| (s ^ ORDER_STREAM, 0)),
            timing: None,
            ahead: cfg.read_ahead.then(Memories::default),
            blend: String::new(),
            adapter_on: true,
        })
    }

    /// Load trained LoRA weights (saved by `save_adapter`) into this model's
    /// VarMap by matching variable names. Used to serve a graduated expert and
    /// to warm-start continual training from a parent adapter. Each factor is
    /// cast to the model's dtype, so an expert trained in bf16 on the GPU loads
    /// into the f32 model the CPU runs.
    pub fn load_adapter(&mut self, path: &str) -> Result<()> {
        // Whatever the load did or did not set, no copy of the factors
        // outlives it.
        self.model.adapters_changed();
        self.blend = format!("one:{path}");
        crate::compose::restore_adapter(&self.model.varmap, path)
    }

    /// The phases of the last generation this model ran, if it has run one.
    pub fn last_timing(&self) -> Option<GenTiming> {
        self.timing
    }

    /// Generate with the adapter's delta or without it. Off, every projection
    /// is the frozen base alone, as the GRPO reference pass reads it, and
    /// [`CausalLm::generate`] answers as the base model would. For measuring
    /// what an adapter costs a token; whoever turns it off turns it back on.
    pub fn set_adapter(&mut self, on: bool) {
        self.model.set_lora(on);
        self.adapter_on = on;
    }

    /// Teacher-force `completion` after `prompt` through the adapter, with no
    /// gradient: for each of the completion's tokens and the end-of-turn after
    /// it, the log-probability the model gives it and the token the model ranks
    /// first there. The text is tokenized as training tokenizes an example, so
    /// the positions are the ones an expert was taught on. A reading of the
    /// served policy, for comparing one base form against another
    /// ([`crate::quant`]).
    pub fn teacher_forced(
        &mut self,
        prompt: &str,
        completion: &str,
    ) -> Result<Vec<crate::quant::Reading>> {
        let wrapped = self.wrap_prompt(prompt);
        let prompt_ids = self.encode(&wrapped)?;
        let mut ids = self.encode(&format!("{wrapped}{completion}"))?;
        if ids.len() <= prompt_ids.len() {
            return Ok(Vec::new());
        }
        ids.push(self.eos);
        // The completion starts where the two tokenizations part, as in
        // `forward_loss`: the boundary token can merge across the join.
        let boundary = prompt_ids
            .iter()
            .zip(ids.iter())
            .take_while(|(a, b)| a == b)
            .count()
            .max(1);
        self.model.clear_cache();
        self.model.set_lora(true);
        self.model.set_grad(false);
        let input = Tensor::new(ids.as_slice(), &self.model.device)
            .map_err(ce)?
            .unsqueeze(0)
            .map_err(ce)?;
        let hidden = self.model.hidden(&input, 0, false).map_err(ce)?;
        // The logits at position i - 1 predict token i.
        let logits = self
            .model
            .logits(
                &hidden
                    .narrow(1, boundary - 1, ids.len() - boundary)
                    .map_err(ce)?,
            )
            .map_err(ce)?
            .squeeze(0)
            .map_err(ce)?
            .to_dtype(DType::F32)
            .map_err(ce)?;
        let logprobs = candle_nn::ops::log_softmax(&logits, D::Minus1)
            .map_err(ce)?
            .to_vec2::<f32>()
            .map_err(ce)?;
        let top1 = logits
            .argmax(D::Minus1)
            .map_err(ce)?
            .to_vec1::<u32>()
            .map_err(ce)?;
        Ok(logprobs
            .iter()
            .zip(top1)
            .zip(&ids[boundary..])
            .map(|((row, top1), &token)| crate::quant::Reading {
                logprob: row[token as usize],
                top1,
            })
            .collect())
    }

    /// Refuse a training step on a model loaded to serve: its quantized base
    /// product has no backward, so the step would reach every layer's factors
    /// with the wrong gradient.
    fn ensure_trainable(&self) -> Result<()> {
        if self.model.serves {
            return Err(AntumbraError::other(
                "this model was loaded to serve, with a quantized base that carries no gradient; it cannot train",
            ));
        }
        Ok(())
    }

    /// Load `(adapter_path, weight)` adapters as one blend: stacked, each block
    /// contributing `weight` times its delta, and zero-padded to this model's
    /// rank (`crate::compose::blend`). A model built at a multiple of the
    /// trained rank, with alpha raised in step, serves one adapter exactly as a
    /// model at the trained rank would, and a blend without a rebuild.
    pub fn load_blend(&mut self, specs: &[(String, f32)]) -> Result<()> {
        self.model.adapters_changed();
        self.blend = specs
            .iter()
            .map(|(path, weight)| format!("{path}*{weight}"))
            .collect::<Vec<_>>()
            .join("+");
        // A poisoned lock holds the same map of vars, none half-set: taken as is.
        let vars = self
            .model
            .varmap
            .data()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        crate::compose::set_blend(&vars, specs)
    }

    /// Keep a blend of experts beside the one that is loaded, as blend
    /// `blend` (one or more): [`QwenCausalLm::generate_each_of`] answers a
    /// request with it while other requests in the same pass are answered
    /// with others. The blend is loaded as [`QwenCausalLm::load_blend`]
    /// loads it and its factors are kept as rows read them; the blend that
    /// was loaded before is no longer, so load it again, or keep it too.
    /// `false`, with the blend loaded and nothing kept, where this model
    /// does not read rows as plain values (a GPU, a dense base, a CPU with
    /// no tiles): there one blend is served at a time.
    pub fn keep_blend(&mut self, blend: u16, specs: &[(String, f32)]) -> Result<bool> {
        self.load_blend(specs)?;
        self.model.set_grad(false);
        self.model.keep_blend(blend).map_err(ce)
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

    /// The logits after the last of `tokens`, the first of which is at
    /// `start`, through the tensor path with the cache on.
    fn forwarded_logits(&mut self, tokens: &[u32], start: usize) -> Result<Vec<f32>> {
        let input = Tensor::new(tokens, &self.model.device)
            .map_err(ce)?
            .unsqueeze(0)
            .map_err(ce)?;
        let hidden = self.model.hidden(&input, start, true).map_err(ce)?;
        let last = hidden.narrow(1, tokens.len() - 1, 1).map_err(ce)?;
        self.model
            .logits(&last)
            .map_err(ce)?
            .squeeze(0)
            .map_err(ce)?
            .squeeze(0)
            .map_err(ce)?
            .to_dtype(DType::F32)
            .map_err(ce)?
            .to_vec1()
            .map_err(ce)
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
        let mut timing = GenTiming::default();

        // Several tokens a pass, where the token is the largest logit and
        // the model reads ahead: the same tokens as the loop below writes.
        let mut written = false;
        if let (true, Some(memories)) = (policy.takes_the_first(), &mut self.ahead) {
            let blend = match self.adapter_on {
                true => self.blend.as_str(),
                false => "",
            };
            let at = memories.place(blend);
            let ahead = ahead::Ahead {
                eos: self.eos,
                most: self.max_new_tokens,
                near: &mut memories.near,
                far: &memories.far[at].1,
                odds: &mut memories.odds,
            };
            let passes =
                ahead::decode(&mut self.model, &mut tokens, prompt_len, ahead, &mut timing)
                    .map_err(ce)?;
            if passes.is_some() {
                // What it wrote is remembered for this blend's next
                // generation; the prompt is not.
                memories.far[at].1.learn(&tokens, prompt_len);
                memories.far[at]
                    .1
                    .keep(&tokens[..prompt_len], &tokens[prompt_len..]);
                written = true;
            }
        }

        // A token at a time, where that was not done.
        let steps = match written {
            true => 0,
            false => self.max_new_tokens,
        };
        for index in 0..steps {
            let step = std::time::Instant::now();
            let ctx_len = if index == 0 { tokens.len() } else { 1 };
            let start = tokens.len() - ctx_len;
            // The prompt, then each token after it: through plain rows
            // where the model reads rows that way, as tensors otherwise. A
            // policy whose token is the largest logit says so, and has the
            // token itself back where the head can find it without being
            // read whole.
            let stepped = self
                .model
                .step(&tokens[start..], start, policy.takes_the_first())
                .map_err(ce)?;
            let next = match stepped {
                Some(Stepped::First(next)) => next,
                logits => {
                    let mut logits = match logits {
                        Some(Stepped::Logits(logits)) => logits,
                        _ => self.forwarded_logits(&tokens[start..], start)?,
                    };
                    // Decode policy over the generated continuation only
                    // (not the prompt).
                    let next = crate::decode::pick_token_in(
                        &mut logits,
                        &tokens[prompt_len..],
                        &policy,
                        &mut rng,
                    );
                    // The room the logits are in goes back for the next
                    // token's.
                    self.model.keep(logits);
                    next
                }
            };
            timing.record(index, ctx_len, step.elapsed());
            if next == self.eos {
                break;
            }
            tokens.push(next);
            // A generation that has run into a loop is over: it would repeat
            // itself to its token limit, and nothing after it is an answer.
            if crate::decode::loops(&tokens[prompt_len..]) {
                break;
            }
        }
        self.timing = Some(timing);

        let text = self
            .tokenizer
            .decode(&tokens[prompt_len..], true)
            .map_err(|e| AntumbraError::other(format!("decode: {e}")))?;
        // Trim the run-on so the verified, trained-on completion is the clean one.
        Ok(self.trim(&text))
    }

    /// An answer to each of `prompts`, decoded side by side: for each prompt
    /// what [`CausalLm::generate`] gives it alone under a greedy policy, token
    /// for token, and under a sampling one with `seed` plus its place among
    /// the prompts. Each prompt is read into a cache of its own; after that
    /// every sequence still answering takes its next token from one pass
    /// over the weights, which on a CPU is read from memory once for all of
    /// them, and a sequence that ends leaves the rest to go on. Where the
    /// model does not read rows as plain values (a GPU, a dense base, a CPU
    /// with no tiles) the prompts are answered one after another.
    ///
    /// The timing kept is the whole run's: the prompts' tokens and the time
    /// they took to read, and the tokens decoded after each sequence's first
    /// over the time the shared passes took.
    pub fn generate_each(&mut self, prompts: &[&str], seed: u64) -> Result<Vec<String>> {
        let of: Vec<(&str, u16)> = prompts.iter().map(|prompt| (*prompt, 0)).collect();
        self.generate_each_of(&of, seed)
    }

    /// [`QwenCausalLm::generate_each`] with a blend of experts for each
    /// prompt: 0 for the blend that is loaded, or one kept with
    /// [`QwenCausalLm::keep_blend`]. Each answer is the one its prompt gets
    /// alone from a model that has that blend loaded. The base's weights
    /// are read once for every request in a pass whatever their blends;
    /// each request reads its own blend's factors beside them.
    pub fn generate_each_of(&mut self, prompts: &[(&str, u16)], seed: u64) -> Result<Vec<String>> {
        let mut answers = vec![String::new(); prompts.len()];
        let mut timing = GenTiming::default();
        // A run of prompts at a time, as many as one pass takes.
        for (run, some) in prompts.chunks(Together::AT_ONCE).enumerate() {
            let from = run * Together::AT_ONCE;
            let mut together = Together::default();
            let mut places = Vec::with_capacity(some.len());
            for (i, (prompt, blend)) in some.iter().enumerate() {
                let own = seed.wrapping_add((from + i) as u64);
                let Some(place) = self.admit(&mut together, prompt, *blend, own)? else {
                    // This model reads no rows this way: one prompt after
                    // another, as it always has.
                    if prompts.iter().any(|(_, blend)| *blend != 0) {
                        return Err(AntumbraError::other(
                            "a kept blend was asked for of a model that keeps none: it reads no rows as plain values",
                        ));
                    }
                    for (i, (prompt, _)) in prompts.iter().enumerate() {
                        answers[i] = self.sample_one(prompt, seed.wrapping_add(i as u64))?;
                    }
                    return Ok(answers);
                };
                places.push(place);
            }
            while together.answering() > 0 {
                self.advance(&mut together)?;
            }
            for (i, place) in places.into_iter().enumerate() {
                answers[from + i] = together.take(place).unwrap_or_default();
            }
            timing.prompt_tokens += together.timing.prompt_tokens;
            timing.prompt_secs += together.timing.prompt_secs;
            timing.decoded += together.timing.decoded;
            timing.decode_secs += together.timing.decode_secs;
        }
        self.timing = Some(timing);
        Ok(answers)
    }

    /// Take a request in among those `together` is answering: its prompt is
    /// read into a cache of its own, and it takes its tokens from the passes
    /// that follow ([`QwenCausalLm::advance`]) beside the requests already
    /// there, which do not wait for it. `blend` is 0 for the blend that is
    /// loaded or one kept with [`QwenCausalLm::keep_blend`]. What comes back
    /// is the request's place, by which its answer is taken when it is done.
    /// `None`, with nothing read, when there is no room for another or when
    /// this model does not read rows as plain values.
    pub fn admit(
        &mut self,
        together: &mut Together,
        prompt: &str,
        blend: u16,
        seed: u64,
    ) -> Result<Option<usize>> {
        let Some(place) = together.free() else {
            return Ok(None);
        };
        self.model.set_grad(false);
        self.model.sequences(Together::AT_ONCE);
        let policy = self.policy();
        let first = policy.takes_the_first();
        let tokens = self.encode(&self.wrap_prompt(prompt))?;
        let mut sequence = Sequence {
            prompt_len: tokens.len(),
            tokens,
            blend,
            rng: StdRng::seed_from_u64(seed),
            answer: None,
        };
        self.model.clear_sequence(place).map_err(ce)?;
        let began = std::time::Instant::now();
        let stepped = self
            .model
            .step_in(place, blend, &sequence.tokens, 0, first)
            .map_err(ce)?;
        let next = match stepped {
            Some(Stepped::First(next)) => next,
            Some(Stepped::Logits(mut logits)) => {
                let next =
                    crate::decode::pick_token_in(&mut logits, &[], &policy, &mut sequence.rng);
                self.model.keep(logits);
                next
            }
            None => return Ok(None),
        };
        together.timing.prompt_tokens += sequence.prompt_len;
        together.timing.prompt_secs += began.elapsed().as_secs_f64();
        if self.max_new_tokens == 0 || next == self.eos {
            sequence.answer = Some(self.answer_of(&sequence)?);
        } else {
            sequence.tokens.push(next);
        }
        together.places[place] = Some(sequence);
        Ok(Some(place))
    }

    /// One pass for every request `together` is still answering: each takes
    /// its next token, and one that ends there has its answer ready
    /// ([`Together::take`]). The places of those that ended in this pass are
    /// what comes back.
    pub fn advance(&mut self, together: &mut Together) -> Result<Vec<usize>> {
        let policy = self.policy();
        let first = policy.takes_the_first();
        let each: Vec<Next> = together
            .places
            .iter()
            .enumerate()
            .filter_map(|(place, sequence)| {
                let sequence = sequence.as_ref().filter(|s| s.answer.is_none())?;
                sequence.tokens.last().map(|token| Next {
                    sequence: place,
                    token: *token,
                    position: sequence.tokens.len() - 1,
                    blend: sequence.blend,
                })
            })
            .collect();
        if each.is_empty() {
            return Ok(Vec::new());
        }
        let began = std::time::Instant::now();
        let Some(stepped) = self.model.step_each(&each, first).map_err(ce)? else {
            return Err(AntumbraError::other(
                "a model that read its prompts as plain rows stopped reading rows that way",
            ));
        };
        together.timing.decoded += each.len();
        together.timing.decode_secs += began.elapsed().as_secs_f64();
        let mut ended = Vec::new();
        for (next, stepped) in each.iter().zip(stepped) {
            let Some(sequence) = together.places[next.sequence].as_mut() else {
                continue;
            };
            let token = match stepped {
                Stepped::First(token) => token,
                Stepped::Logits(mut logits) => crate::decode::pick_token_in(
                    &mut logits,
                    &sequence.tokens[sequence.prompt_len..],
                    &policy,
                    &mut sequence.rng,
                ),
            };
            let mut done = token == self.eos;
            if !done {
                sequence.tokens.push(token);
                let answered = &sequence.tokens[sequence.prompt_len..];
                done = answered.len() >= self.max_new_tokens || crate::decode::loops(answered);
            }
            if done {
                let answer = self.answer_of(sequence)?;
                sequence.answer = Some(answer);
                ended.push(next.sequence);
            }
        }
        Ok(ended)
    }

    /// The decode policy this model's settings come to.
    fn policy(&self) -> crate::decode::DecodePolicy {
        crate::decode::DecodePolicy {
            temperature: self.temperature,
            top_p: self.top_p,
            repetition_penalty: self.repetition_penalty,
            no_repeat_ngram_size: self.no_repeat_ngram_size,
        }
    }

    /// What a sequence has answered, as text.
    fn answer_of(&self, sequence: &Sequence) -> Result<String> {
        let text = self
            .tokenizer
            .decode(&sequence.tokens[sequence.prompt_len..], true)
            .map_err(|e| AntumbraError::other(format!("decode: {e}")))?;
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
            self.model.logits(&hidden)?.to_dtype(DType::F32)
        })();
        self.model.set_lora(true);
        logits.map_err(ce)
    }

    /// The next-token logits after `prompt`, under the adapter, with no
    /// gradient: a reading, not a training pass.
    fn next_logits(&mut self, prompt: &str) -> Result<Vec<f32>> {
        self.model.clear_cache();
        self.model.set_lora(true);
        self.model.set_grad(false);
        let ids = self.encode(&self.wrap_prompt(prompt))?;
        let input = Tensor::new(ids.as_slice(), &self.model.device)
            .map_err(ce)?
            .unsqueeze(0)
            .map_err(ce)?;
        let hidden = self.model.hidden(&input, 0, false).map_err(ce)?;
        let last = hidden.narrow(1, ids.len() - 1, 1).map_err(ce)?;
        self.model
            .logits(&last)
            .map_err(ce)?
            .flatten_all()
            .map_err(ce)?
            .to_dtype(DType::F32)
            .map_err(ce)?
            .to_vec1()
            .map_err(ce)
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
        let mut timing = GenTiming::default();

        for index in 0..self.max_new_tokens {
            let step = std::time::Instant::now();
            let ctx_len = if index == 0 { tokens.len() } else { 1 };
            let start = tokens.len() - ctx_len;
            let input = Tensor::new(&tokens[start..], &self.model.device)
                .map_err(ce)?
                .unsqueeze(0)
                .map_err(ce)?;
            let hidden = self.model.hidden(&input, start, true).map_err(ce)?;
            let last = hidden.narrow(1, ctx_len - 1, 1).map_err(ce)?;
            let logits = self
                .model
                .logits(&last)
                .map_err(ce)?
                .squeeze(0)
                .map_err(ce)?
                .squeeze(0)
                .map_err(ce)?
                .to_dtype(DType::F32)
                .map_err(ce)?;
            let next = sample_token(&logits, temp, &mut rng)?;
            timing.record(index, ctx_len, step.elapsed());
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
        self.timing = Some(timing);

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
        let logits = self
            .model
            .logits(&hidden)
            .map_err(ce)?
            .to_dtype(DType::F32)
            .map_err(ce)?;
        let loss = causal_lm_loss(&logits, &input, &mask).map_err(ce)?;
        Ok(Some(loss))
    }

    /// One example, one optimizer step (batch-of-1 SGD). The per-example path.
    fn train_one(&mut self, example: &SftExample) -> Result<f32> {
        self.ensure_trainable()?;
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
        self.ensure_trainable()?;
        // A seeded run takes its n-th shuffle from its own stream; otherwise
        // the shuffle is the process's next.
        let stream = match &mut self.train_order {
            Some((base, taken)) => {
                let s = base
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(*taken);
                *taken += 1;
                s
            }
            None => {
                let nonce = GEN_NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                crate::seed::shifted(
                    ORDER_STREAM.wrapping_mul(nonce.wrapping_add(1)),
                    crate::seed::generation_seed(),
                )
            }
        };
        let mut order: Vec<usize> = (0..batch.len()).collect();
        let mut rng = StdRng::seed_from_u64(stream);
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
        let mut rng = StdRng::seed_from_u64(crate::seed::shifted(
            0x5F37_u64.wrapping_mul(nonce.wrapping_add(1)),
            crate::seed::generation_seed(),
        ));
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

    async fn choose(&mut self, prompt: &str, choices: &[&str]) -> Result<Vec<f32>> {
        let logits = self.next_logits(prompt)?;
        let mut picked = Vec::with_capacity(choices.len());
        for choice in choices {
            let first = self.encode(choice)?.first().copied().ok_or_else(|| {
                AntumbraError::other(format!("choice `{choice}` encodes to nothing"))
            })?;
            let logit = logits.get(first as usize).copied().ok_or_else(|| {
                AntumbraError::other(format!("choice `{choice}` is out of vocabulary"))
            })?;
            picked.push(logit);
        }
        Ok(crate::critic::softmax(&picked))
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
        self.ensure_trainable()?;
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
