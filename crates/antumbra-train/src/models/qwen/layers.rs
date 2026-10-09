//! The pieces a Qwen decoder is built from: the LoRA-wrapped linear, the
//! norms, the rotary embedding, attention, the MLP, and the layer that
//! stacks them.
//!
//! A child module of `qwen`, so it sees the config and helpers next door and
//! the model above builds from these; they moved here only because the file
//! had grown past the size rule.

use super::*;

// --- linear-algebra helpers ------------------------------------------------

/// `x · Wᵀ` with candle's broadcast-over-batch-dims convention (mirrors
/// `candle_nn::Linear`), for 2-, 3-, and 4-D `x`.
pub(super) fn matmul_t(x: &Tensor, w: &Tensor) -> CResult<Tensor> {
    let w = match *x.dims() {
        [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
        [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
        _ => w.t()?,
    };
    x.matmul(&w)
}

/// GQA: repeat each KV head `n_rep` times to match the query heads.
pub(super) fn repeat_kv(x: Tensor, n_rep: usize) -> CResult<Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }
    let (b, kv_heads, seq, head_dim) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, kv_heads, n_rep, seq, head_dim))?
        .reshape((b, kv_heads * n_rep, seq, head_dim))
}

// --- a LoRA-wrapped linear over a frozen base ------------------------------

/// The frozen base weight: a dense tensor, or a 4-bit Q4_K `QTensor` that is
/// dequantized in the forward (QLoRA-proper, v1 efficiency). Either way it is a
/// constant; gradients only reach the LoRA factors.
pub(super) enum BaseWeight {
    Dense(Tensor),
    Quantized(QTensor),
}

pub(super) struct LoraLinear {
    base: BaseWeight,
    base_b: Option<Tensor>,
    a: Tensor,
    b: Tensor,
    scale: f64,
    /// When false the forward is base-only (the GRPO reference pass).
    enabled: bool,
    /// When false the LoRA factors are detached so the forward is *not* tracked
    /// (generation). Without this, sampling retains the whole autograd graph in
    /// the KV cache, and with a quantized base, every token's re-materialized
    /// weights are retained too, OOMing the card.
    grad: bool,
}

impl LoraLinear {
    /// Load the frozen base weight (+ optional bias) from `base`, and register
    /// fresh trainable LoRA factors in `lora`.
    pub(super) fn load(
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
            base: BaseWeight::Dense(base_w),
            base_b,
            a,
            b,
            scale,
            enabled: true,
            grad: true,
        })
    }

    pub(super) fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let base_w = match &self.base {
            BaseWeight::Dense(w) => w.clone(),
            BaseWeight::Quantized(q) => q.dequantize(x.device())?.to_dtype(x.dtype())?,
        };
        // The base is frozen, so its product computes no gradient for the
        // weight: candle's plain matmul would build one for every base
        // weight on every training step and never use it.
        let mut out = frozen_matmul_t(x, &base_w)?;
        if self.enabled {
            // Detaching the factors when grad is off keeps generation off the
            // autograd graph: same values, no retained activations or dequant.
            let (a, b) = if self.grad {
                (self.a.clone(), self.b.clone())
            } else {
                (self.a.detach(), self.b.detach())
            };
            let lora = matmul_t(&matmul_t(x, &a)?, &b)?.affine(self.scale, 0.0)?;
            out = (out + lora)?;
        }
        match &self.base_b {
            Some(bias) => out.broadcast_add(bias),
            None => Ok(out),
        }
    }

    pub(super) fn set_enabled(&mut self, on: bool) {
        self.enabled = on;
    }

    pub(super) fn set_grad(&mut self, on: bool) {
        self.grad = on;
    }

    /// Quantize the frozen base weight to 4-bit Q4_K (v1 efficiency). Q4_K needs the
    /// `in` dim divisible by 256, which holds for every Qwen projection.
    pub(super) fn quantize(&mut self, device: &Device) -> CResult<()> {
        if let BaseWeight::Dense(w) = &self.base {
            let cpu = w.to_device(&Device::Cpu)?;
            let q = QTensor::quantize_onto(&cpu, GgmlDType::Q4K, device)?;
            self.base = BaseWeight::Quantized(q);
        }
        Ok(())
    }
}

// --- RMSNorm (frozen) ------------------------------------------------------

pub(super) struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub(super) fn load(size: usize, eps: f64, vb: &VarBuilder) -> CResult<Self> {
        Ok(Self {
            weight: vb.get(size, "weight")?,
            eps,
        })
    }

    pub(super) fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let in_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let x_normed = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        x_normed.to_dtype(in_dtype)?.broadcast_mul(&self.weight)
    }
}

// --- rotary embedding ------------------------------------------------------

pub(super) struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    pub(super) fn new(dtype: DType, cfg: &Config, dev: &Device) -> CResult<Self> {
        let dim = cfg.hidden_size / cfg.num_attention_heads;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        // Positions and angles are computed in f32 whatever the model computes
        // in, and only the finished sin/cos tables take the model's dtype. bf16
        // carries 8 bits of mantissa: position 257 is 256 in it, and past 512
        // positions come in steps of four, so neighboring tokens were rotated
        // as if they sat at the same place -- and an angle of a few hundred
        // radians keeps almost none of its fraction, so the fast-rotating
        // dimensions got sin and cos of the wrong angle. On the GPU that showed
        // up as duplicated tokens under greedy decoding ("than than", "+= +=")
        // that an f32 run of the same prompt did not produce.
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    pub(super) fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> CResult<(Tensor, Tensor)> {
        let (_b, _h, seq_len, _d) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q, k))
    }
}

// --- attention -------------------------------------------------------------

pub(super) struct Attention {
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
    pub(super) fn load(
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

    pub(super) fn forward(
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

    pub(super) fn clear_cache(&mut self) {
        self.kv_cache = None;
    }

    pub(super) fn set_lora(&mut self, on: bool) {
        self.q_proj.set_enabled(on);
        self.k_proj.set_enabled(on);
        self.v_proj.set_enabled(on);
        self.o_proj.set_enabled(on);
    }

    pub(super) fn set_grad(&mut self, on: bool) {
        self.q_proj.set_grad(on);
        self.k_proj.set_grad(on);
        self.v_proj.set_grad(on);
        self.o_proj.set_grad(on);
    }

    pub(super) fn quantize_base(&mut self, device: &Device) -> CResult<()> {
        self.q_proj.quantize(device)?;
        self.k_proj.quantize(device)?;
        self.v_proj.quantize(device)?;
        self.o_proj.quantize(device)
    }
}

// --- MLP -------------------------------------------------------------------

pub(super) struct Mlp {
    gate_proj: LoraLinear,
    up_proj: LoraLinear,
    down_proj: LoraLinear,
    act: Activation,
}

impl Mlp {
    pub(super) fn load(
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

    pub(super) fn forward(&self, xs: &Tensor) -> CResult<Tensor> {
        let lhs = self.act.forward(&self.gate_proj.forward(xs)?)?;
        let rhs = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(lhs * rhs)?)
    }

    pub(super) fn set_lora(&mut self, on: bool) {
        self.gate_proj.set_enabled(on);
        self.up_proj.set_enabled(on);
        self.down_proj.set_enabled(on);
    }

    pub(super) fn set_grad(&mut self, on: bool) {
        self.gate_proj.set_grad(on);
        self.up_proj.set_grad(on);
        self.down_proj.set_grad(on);
    }

    pub(super) fn quantize_base(&mut self, device: &Device) -> CResult<()> {
        self.gate_proj.quantize(device)?;
        self.up_proj.quantize(device)?;
        self.down_proj.quantize(device)
    }
}

// --- decoder layer ---------------------------------------------------------

pub(super) struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_ln: RmsNorm,
    post_attn_ln: RmsNorm,
}

impl DecoderLayer {
    pub(super) fn load(
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

    pub(super) fn forward(
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

    pub(super) fn clear_cache(&mut self) {
        self.self_attn.clear_cache();
    }

    pub(super) fn set_lora(&mut self, on: bool) {
        self.self_attn.set_lora(on);
        self.mlp.set_lora(on);
    }

    pub(super) fn set_grad(&mut self, on: bool) {
        self.self_attn.set_grad(on);
        self.mlp.set_grad(on);
    }

    pub(super) fn quantize_base(&mut self, device: &Device) -> CResult<()> {
        self.self_attn.quantize_base(device)?;
        self.mlp.quantize_base(device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Config {
        Config {
            vocab_size: 16,
            hidden_size: 128,
            intermediate_size: 128,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            max_position_embeddings: 2048,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            hidden_act: Activation::Silu,
            tie_word_embeddings: true,
        }
    }

    fn largest_gap(a: &Tensor, b: &Tensor) -> CResult<f32> {
        (a - b.to_dtype(DType::F32)?)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()
    }

    /// The rotary table is a function of position, not of the precision the
    /// model computes in: a bf16 model gets the f32 table rounded to bf16,
    /// which is within a few thousandths everywhere. Built in bf16 it was off
    /// by whole units at long positions, which is the difference between a
    /// token's position and some other token's.
    #[test]
    fn the_rotary_table_does_not_depend_on_the_compute_precision() -> CResult<()> {
        let dev = Device::Cpu;
        let exact = RotaryEmbedding::new(DType::F32, &config(), &dev)?;
        let half = RotaryEmbedding::new(DType::BF16, &config(), &dev)?;
        let sin = largest_gap(&exact.sin, &half.sin)?;
        let cos = largest_gap(&exact.cos, &half.cos)?;
        assert!(
            sin < 0.01 && cos < 0.01,
            "bf16 table drifts: sin {sin}, cos {cos}"
        );
        Ok(())
    }

    /// Neighboring positions past 256 are distinct in the bf16 table. bf16
    /// cannot hold 257, so positions computed in it collapsed pairwise.
    #[test]
    fn neighboring_long_positions_stay_distinct_in_bf16() -> CResult<()> {
        let half = RotaryEmbedding::new(DType::BF16, &config(), &Device::Cpu)?;
        let a = half.sin.get(300)?.to_dtype(DType::F32)?;
        let b = half.sin.get(301)?.to_dtype(DType::F32)?;
        let gap = (a - b)?.abs()?.max(0)?.to_scalar::<f32>()?;
        assert!(
            gap > 0.1,
            "positions 300 and 301 are nearly the same rotation: {gap}"
        );
        Ok(())
    }
}
