//! candle Qwen2.5-Coder + LoRA — the real [`CausalLm`] (ADR-0010, MT-1).
//!
//! Architecture vendored from candle-transformers' `qwen2` (GQA attention,
//! RoPE, RMSNorm, KV cache) with the seven projection linears per layer swapped
//! for [`LoraLinear`]: the base weights load **frozen** from the HF safetensors;
//! only the LoRA `A`/`B` factors are trainable `Var`s. Generation uses the KV
//! cache; `sft_step` runs a full-sequence forward, the completion-masked
//! cross-entropy ([`crate::objective::causal_lm_loss`]), and `backward_step`
//! over the LoRA `VarMap`. Validated on the 3090 Ti (needs weights).

use std::path::PathBuf;
use std::sync::Arc;

use candle_core::{DType, Device, Result as CResult, Tensor, D};
use candle_nn::init::Init;
use candle_nn::{Activation, Embedding, Module, Optimizer, VarBuilder, VarMap};

use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::StdRng;
use rand::SeedableRng;

use async_trait::async_trait;

use antumbra_core::{AntumbraError, Result};

use crate::config::RaftConfig;
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

// --- linear-algebra helpers ------------------------------------------------

/// `x · Wᵀ` with candle's broadcast-over-batch-dims convention (mirrors
/// `candle_nn::Linear`), for 2-, 3-, and 4-D `x`.
fn matmul_t(x: &Tensor, w: &Tensor) -> CResult<Tensor> {
    let w = match *x.dims() {
        [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
        [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
        _ => w.t()?,
    };
    x.matmul(&w)
}

/// GQA: repeat each KV head `n_rep` times to match the query heads.
fn repeat_kv(x: Tensor, n_rep: usize) -> CResult<Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }
    let (b, kv_heads, seq, head_dim) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, kv_heads, n_rep, seq, head_dim))?
        .reshape((b, kv_heads * n_rep, seq, head_dim))
}

// --- a LoRA-wrapped linear over a frozen base ------------------------------

struct LoraLinear {
    base_w: Tensor,
    base_b: Option<Tensor>,
    a: Tensor,
    b: Tensor,
    scale: f64,
}

impl LoraLinear {
    /// Load the frozen base weight (+ optional bias) from `base`, and register
    /// fresh trainable LoRA factors in `lora`.
    fn load(
        in_f: usize,
        out_f: usize,
        bias: bool,
        base: &VarBuilder,
        lora: &VarBuilder,
        rank: usize,
        scale: f64,
    ) -> CResult<Self> {
        let base_w = base.get((out_f, in_f), "weight")?;
        let base_b = if bias {
            Some(base.get(out_f, "bias")?)
        } else {
            None
        };
        let a = lora.get_with_hints(
            (rank, in_f),
            "lora_a",
            Init::Randn {
                mean: 0.0,
                stdev: 0.02,
            },
        )?;
        let b = lora.get_with_hints((out_f, rank), "lora_b", candle_nn::init::ZERO)?;
        Ok(Self {
            base_w,
            base_b,
            a,
            b,
            scale,
        })
    }

    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let mut out = matmul_t(x, &self.base_w)?;
        let lora = matmul_t(&matmul_t(x, &self.a)?, &self.b)?.affine(self.scale, 0.0)?;
        out = (out + lora)?;
        match &self.base_b {
            Some(bias) => out.broadcast_add(bias),
            None => Ok(out),
        }
    }
}

// --- RMSNorm (frozen) ------------------------------------------------------

struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn load(size: usize, eps: f64, vb: &VarBuilder) -> CResult<Self> {
        Ok(Self {
            weight: vb.get(size, "weight")?,
            eps,
        })
    }

    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let in_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let x_normed = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        x_normed.to_dtype(in_dtype)?.broadcast_mul(&self.weight)
    }
}

// --- rotary embedding ------------------------------------------------------

struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> CResult<Self> {
        let dim = cfg.hidden_size / cfg.num_attention_heads;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> CResult<(Tensor, Tensor)> {
        let (_b, _h, seq_len, _d) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q, k))
    }
}

// --- attention -------------------------------------------------------------

struct Attention {
    q_proj: LoraLinear,
    k_proj: LoraLinear,
    v_proj: LoraLinear,
    o_proj: LoraLinear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary: Arc<RotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Attention {
    fn load(
        rotary: Arc<RotaryEmbedding>,
        cfg: &Config,
        rank: usize,
        scale: f64,
        base: &VarBuilder,
        lora: &VarBuilder,
    ) -> CResult<Self> {
        let h = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = h / num_heads;
        Ok(Self {
            q_proj: LoraLinear::load(
                h,
                num_heads * head_dim,
                true,
                &base.pp("q_proj"),
                &lora.pp("q_proj"),
                rank,
                scale,
            )?,
            k_proj: LoraLinear::load(
                h,
                num_kv_heads * head_dim,
                true,
                &base.pp("k_proj"),
                &lora.pp("k_proj"),
                rank,
                scale,
            )?,
            v_proj: LoraLinear::load(
                h,
                num_kv_heads * head_dim,
                true,
                &base.pp("v_proj"),
                &lora.pp("v_proj"),
                rank,
                scale,
            )?,
            o_proj: LoraLinear::load(
                num_heads * head_dim,
                h,
                false,
                &base.pp("o_proj"),
                &lora.pp("o_proj"),
                rank,
                scale,
            )?,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
            hidden_size: h,
            rotary,
            kv_cache: None,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        mask: Option<&Tensor>,
        offset: usize,
        use_cache: bool,
    ) -> CResult<Tensor> {
        let (b, q_len, _) = xs.dims3()?;
        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let q = q
            .reshape((b, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = self.rotary.apply(&q, &k, offset)?;

        let (k, v) = match &self.kv_cache {
            Some((pk, pv)) if use_cache => (Tensor::cat(&[pk, &k], 2)?, Tensor::cat(&[pv, &v], 2)?),
            _ => (k, v),
        };
        if use_cache {
            self.kv_cache = Some((k.clone(), v.clone()));
        }

        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let scale = 1f64 / (self.head_dim as f64).sqrt();
        let attn = (q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let attn = match mask {
            Some(m) => attn.broadcast_add(m)?,
            None => attn,
        };
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;
        let out = out.transpose(1, 2)?.reshape((b, q_len, self.hidden_size))?;
        self.o_proj.forward(&out)
    }

    fn clear_cache(&mut self) {
        self.kv_cache = None;
    }
}

// --- MLP -------------------------------------------------------------------

struct Mlp {
    gate_proj: LoraLinear,
    up_proj: LoraLinear,
    down_proj: LoraLinear,
    act: Activation,
}

impl Mlp {
    fn load(
        cfg: &Config,
        rank: usize,
        scale: f64,
        base: &VarBuilder,
        lora: &VarBuilder,
    ) -> CResult<Self> {
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        Ok(Self {
            gate_proj: LoraLinear::load(
                h,
                i,
                false,
                &base.pp("gate_proj"),
                &lora.pp("gate_proj"),
                rank,
                scale,
            )?,
            up_proj: LoraLinear::load(
                h,
                i,
                false,
                &base.pp("up_proj"),
                &lora.pp("up_proj"),
                rank,
                scale,
            )?,
            down_proj: LoraLinear::load(
                i,
                h,
                false,
                &base.pp("down_proj"),
                &lora.pp("down_proj"),
                rank,
                scale,
            )?,
            act: cfg.hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> CResult<Tensor> {
        let lhs = self.act.forward(&self.gate_proj.forward(xs)?)?;
        let rhs = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(lhs * rhs)?)
    }
}

// --- decoder layer ---------------------------------------------------------

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_ln: RmsNorm,
    post_attn_ln: RmsNorm,
}

impl DecoderLayer {
    fn load(
        rotary: Arc<RotaryEmbedding>,
        cfg: &Config,
        rank: usize,
        scale: f64,
        base: &VarBuilder,
        lora: &VarBuilder,
    ) -> CResult<Self> {
        Ok(Self {
            self_attn: Attention::load(
                rotary,
                cfg,
                rank,
                scale,
                &base.pp("self_attn"),
                &lora.pp("self_attn"),
            )?,
            mlp: Mlp::load(cfg, rank, scale, &base.pp("mlp"), &lora.pp("mlp"))?,
            input_ln: RmsNorm::load(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                &base.pp("input_layernorm"),
            )?,
            post_attn_ln: RmsNorm::load(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                &base.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        mask: Option<&Tensor>,
        offset: usize,
        use_cache: bool,
    ) -> CResult<Tensor> {
        let residual = xs;
        let xs = self.input_ln.forward(xs)?;
        let xs = self.self_attn.forward(&xs, mask, offset, use_cache)?;
        let xs = (residual + xs)?;
        let residual = &xs;
        let h = self.mlp.forward(&self.post_attn_ln.forward(&xs)?)?;
        residual + h
    }

    fn clear_cache(&mut self) {
        self.self_attn.clear_cache();
    }
}

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
}

// --- the trainer-facing wrapper (load + generate + train) ------------------

pub struct QwenCausalLm {
    model: QwenLora,
    tokenizer: tokenizers::Tokenizer,
    opt: candle_nn::AdamW,
    eos: u32,
    max_new_tokens: usize,
}

impl QwenCausalLm {
    /// Load Qwen2.5-Coder + a fresh LoRA adapter from the Hugging Face hub.
    pub fn load(device: Device, cfg: RaftConfig) -> Result<Self> {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| AntumbraError::other(format!("hf-hub: {e}")))?;
        let repo = api.model(cfg.base_model.clone());
        let tok_path = repo
            .get("tokenizer.json")
            .map_err(|e| AntumbraError::other(format!("hf-hub tokenizer: {e}")))?;
        let cfg_path = repo
            .get("config.json")
            .map_err(|e| AntumbraError::other(format!("hf-hub config: {e}")))?;
        let weights: Vec<PathBuf> = vec![repo
            .get("model.safetensors")
            .map_err(|e| AntumbraError::other(format!("hf-hub weights: {e}")))?];

        let tokenizer = tokenizers::Tokenizer::from_file(tok_path)
            .map_err(|e| AntumbraError::other(format!("tokenizer: {e}")))?;
        let config_bytes =
            std::fs::read(cfg_path).map_err(|e| AntumbraError::other(e.to_string()))?;
        let model_cfg: Config = serde_json::from_slice(&config_bytes)?;

        // Train in f32: keeps the frozen base, the LoRA factors, and the loss in
        // one precision (candle autograd flows through the f32 base matmuls into
        // the LoRA factors; the base, not being a Var, gets no gradient). On CPU
        // force f32 regardless of config — f16/bf16 only pay off on a GPU.
        let dtype = if matches!(device, Device::Cpu) {
            DType::F32
        } else {
            cfg.dtype.to_candle()
        };
        let base =
            unsafe { VarBuilder::from_mmaped_safetensors(&weights, dtype, &device).map_err(ce)? };
        let varmap = VarMap::new();
        let lora_vb = VarBuilder::from_varmap(&varmap, dtype, &device);
        let model = QwenLora::build(
            &model_cfg,
            &base,
            &lora_vb,
            varmap,
            cfg.lora_rank,
            cfg.lora_scale(),
        )
        .map_err(ce)?;

        let opt = candle_nn::AdamW::new(
            model.varmap.all_vars(),
            candle_nn::ParamsAdamW {
                lr: cfg.learning_rate,
                ..Default::default()
            },
        )
        .map_err(ce)?;

        let eos = tokenizer
            .token_to_id("<|endoftext|>")
            .ok_or_else(|| AntumbraError::other("tokenizer missing <|endoftext|>"))?;

        Ok(Self {
            model,
            tokenizer,
            opt,
            eos,
            max_new_tokens: cfg.max_new_tokens,
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
        let mut tokens = self.encode(prompt)?;
        let prompt_len = tokens.len();
        let mut rng = StdRng::seed_from_u64(seed);
        let temp = 0.8f64;

        for index in 0..self.max_new_tokens {
            let ctx_len = if index == 0 { tokens.len() } else { 1 };
            let start = tokens.len() - ctx_len;
            let input = Tensor::new(&tokens[start..], &self.model.device)
                .map_err(ce)?
                .unsqueeze(0)
                .map_err(ce)?;
            let hidden = self.model.hidden(&input, start, true).map_err(ce)?;
            let last = hidden.narrow(1, ctx_len - 1, 1).map_err(ce)?;
            let logits = matmul_t(&last, &self.model.lm_head_w)
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
            tokens.push(next);
        }

        let text = self
            .tokenizer
            .decode(&tokens[prompt_len..], true)
            .map_err(|e| AntumbraError::other(format!("decode: {e}")))?;
        // Trim the run-on so the verified, trained-on completion is the clean one.
        Ok(crate::decode::truncate_at_stops(
            &text,
            crate::decode::DEFAULT_STOPS,
        ))
    }

    fn train_one(&mut self, example: &SftExample) -> Result<f32> {
        self.model.clear_cache();
        let prompt_ids = self.encode(&example.prompt)?;
        let full_text = format!("{}{}", example.prompt, example.completion);
        let full_ids = self.encode(&full_text)?;
        if full_ids.len() <= prompt_ids.len() || full_ids.len() < 2 {
            return Ok(0.0);
        }

        let mask: Vec<f32> = (0..full_ids.len())
            .map(|i| if i >= prompt_ids.len() { 1.0 } else { 0.0 })
            .collect();
        let input = Tensor::new(full_ids.as_slice(), &self.model.device)
            .map_err(ce)?
            .unsqueeze(0)
            .map_err(ce)?;
        let mask = Tensor::from_vec(mask, (1, full_ids.len()), &self.model.device).map_err(ce)?;

        let hidden = self.model.hidden(&input, 0, false).map_err(ce)?;
        let logits = matmul_t(&hidden, &self.model.lm_head_w)
            .map_err(ce)?
            .to_dtype(DType::F32)
            .map_err(ce)?;
        let loss = causal_lm_loss(&logits, &input, &mask).map_err(ce)?;
        self.opt.backward_step(&loss).map_err(ce)?;
        loss.to_scalar::<f32>().map_err(ce)
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
            out.push(self.sample_one(prompt, 0xA17_u64.wrapping_mul(i as u64 + 1))?);
        }
        Ok(out)
    }

    async fn sft_step(&mut self, batch: &[SftExample]) -> Result<f32> {
        if batch.is_empty() {
            return Ok(0.0);
        }
        let mut total = 0.0f32;
        for ex in batch {
            total += self.train_one(ex)?;
        }
        Ok(total / batch.len() as f32)
    }

    fn save_adapter(&self, path: &str) -> Result<()> {
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent).map_err(|e| AntumbraError::other(e.to_string()))?;
        }
        self.model
            .varmap
            .save(path)
            .map_err(|e| AntumbraError::other(format!("save adapter: {e}")))
    }
}
