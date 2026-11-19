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

/// `x · Wᵀ` through candle's quantized kernels. They take a contiguous f32 (or
/// f16) input, so `x` is brought to f32 and the product handed back in `x`'s
/// dtype; on the CPU, where the model is f32 already, both casts are free.
pub(super) fn served_matmul(w: &QMatMul, x: &Tensor) -> CResult<Tensor> {
    let dtype = x.dtype();
    w.forward(&x.to_dtype(DType::F32)?.contiguous()?)?
        .to_dtype(dtype)
}

/// Grouped-query attention: `q` is `(b, heads, q_len, head_dim)`, `k` and `v`
/// are `(b, kv_heads, kv_len, head_dim)`, and each KV head serves `n_rep`
/// query heads in order. Returns `(b, heads, q_len, head_dim)`.
///
/// The query heads that share a KV head are stacked as rows against it, so
/// every product is the one a K or V repeated `n_rep` times would give, and
/// nothing of the cache's length is copied to make it. K is multiplied
/// through a transposed view, which the product reads in place. Repeating K
/// and V copied the whole cache six times over on this base, and making the
/// transposed K contiguous copied it again element by element, in every
/// layer for every token: on a CPU that copy alone was about a quarter of a
/// decoded token.
pub(super) fn grouped_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    n_rep: usize,
) -> CResult<Tensor> {
    let (b, heads, q_len, head_dim) = q.dims4()?;
    let (_, kv_heads, kv_len, _) = k.dims4()?;
    let rows = n_rep * q_len;
    let q = q.contiguous()?.reshape((b, kv_heads, rows, head_dim))?;
    let scale = 1f64 / (head_dim as f64).sqrt();
    let attn = (q.matmul(&k.contiguous()?.t()?)? * scale)?;
    let attn = match mask {
        // The mask is per query position, so it is added with the stacked
        // heads set apart again.
        Some(m) => attn
            .reshape((b, kv_heads, n_rep, q_len, kv_len))?
            .broadcast_add(m)?
            .reshape((b, kv_heads, rows, kv_len))?,
        None => attn,
    };
    let attn = candle_nn::ops::softmax_last_dim(&attn)?;
    attn.matmul(&v.contiguous()?)?
        .reshape((b, heads, q_len, head_dim))
}

/// GQA by repeating each KV head `n_rep` times to match the query heads: what
/// [`grouped_attention`] replaced, kept as the reference it is tested against.
#[cfg(test)]
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
///
/// `Served` is the third form, for a model loaded only to serve: the weight is
/// multiplied in its quantized form and never dequantized, so a decoded token
/// reads the quantized bytes and no more. That product has no backward, so it
/// would also cut the gradient to every earlier layer's factors; a served
/// model does not train.
pub(super) enum BaseWeight {
    Dense(Tensor),
    Quantized(QTensor),
    Served(QMatMul),
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

/// The spread a fresh adapter's `lora_a` factors are drawn with; `lora_b`
/// starts at zero, so a fresh adapter changes nothing until it trains.
pub(super) const LORA_A_STDEV: f64 = 0.02;

/// Draw every `lora_a` factor in `varmap` again from `seed`, with the spread a
/// fresh one is drawn with, in the order of their names. The device's own
/// generator gives each new adapter whatever its stream has reached (and the
/// CPU's cannot be seeded at all), so a training seeded here starts from the
/// same factors in any process, on any device.
pub(super) fn seed_lora_a(varmap: &VarMap, seed: u64) -> CResult<()> {
    use rand::RngExt as _;
    // A poisoned lock holds the same map of vars, none half-set: taken as is.
    let vars = varmap
        .data()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut names: Vec<&String> = vars.keys().filter(|n| n.ends_with("lora_a")).collect();
    names.sort();
    let mut rng = StdRng::seed_from_u64(seed);
    // Box-Muller: two uniforms make one standard normal.
    let mut normal = || {
        let u1: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
        let u2: f64 = rng.random::<f64>();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    };
    for name in names {
        let var = &vars[name];
        let values: Vec<f32> = (0..var.elem_count())
            .map(|_| (normal() * LORA_A_STDEV) as f32)
            .collect();
        let drawn = Tensor::from_vec(values, var.shape(), var.device())?.to_dtype(var.dtype())?;
        var.set(&drawn)?;
    }
    Ok(())
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
                stdev: LORA_A_STDEV,
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

    /// The frozen base's product with `x`, before the adapter and the bias.
    fn base_product(&self, x: &Tensor) -> CResult<Tensor> {
        // The base is frozen, so its product computes no gradient for the
        // weight: candle's plain matmul would build one for every base
        // weight on every training step and never use it.
        match &self.base {
            BaseWeight::Dense(w) => frozen_matmul_t(x, w),
            BaseWeight::Quantized(q) => {
                frozen_matmul_t(x, &q.dequantize(x.device())?.to_dtype(x.dtype())?)
            }
            BaseWeight::Served(q) => served_matmul(q, x),
        }
    }

    fn with_bias(&self, out: Tensor) -> CResult<Tensor> {
        match &self.base_b {
            Some(bias) => out.broadcast_add(bias),
            None => Ok(out),
        }
    }

    /// Whether this projection's delta can be read straight from its factors:
    /// the adapter is on, in f32 on the CPU, with nothing to differentiate.
    fn reads_directly(&self) -> bool {
        self.enabled
            && !self.grad
            && self.a.device().is_cpu()
            && self.a.dtype() == DType::F32
            && self.b.dtype() == DType::F32
    }

    /// Every projection in `group` applied to the same `x`: what `forward`
    /// gives each. For one decoded row on the CPU with nothing to
    /// differentiate, the group's adapter deltas are computed together, in
    /// one pass over their factors on every core; a prompt's rows, training
    /// and every other device are each projection's own forward.
    pub(super) fn forward_group<const N: usize>(
        x: &Tensor,
        group: [&LoraLinear; N],
    ) -> CResult<[Tensor; N]> {
        let one_row = x.dims().last().is_some_and(|inn| x.elem_count() == *inn);
        let direct = one_row
            && x.device().is_cpu()
            && x.dtype() == DType::F32
            && !x.track_op()
            && group.iter().all(|l| l.reads_directly());
        let outs: Vec<Tensor> = if direct {
            let factors: Vec<adapt::Factors> = group
                .iter()
                .map(|l| adapt::Factors {
                    a: &l.a,
                    b: &l.b,
                    scale: l.scale,
                })
                .collect();
            let deltas = adapt::grouped_delta(x, &factors)?;
            group
                .iter()
                .zip(deltas)
                .map(|(l, delta)| l.with_bias((l.base_product(x)? + delta)?))
                .collect::<CResult<_>>()?
        } else {
            group.iter().map(|l| l.forward(x)).collect::<CResult<_>>()?
        };
        outs.try_into()
            .map_err(|_| candle_core::Error::msg("a projection group lost an output"))
    }

    pub(super) fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let mut out = self.base_product(x)?;
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
        self.with_bias(out)
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

    /// Hold the frozen base for serving, quantized to `dtype` and multiplied
    /// in that form. Q4_K needs the `in` dim divisible by 256 and Q8_0 by 32,
    /// which holds for every Qwen projection. The dense weight is dropped here,
    /// so a model built layer by layer never holds its whole dense base.
    pub(super) fn serve_quantized(&mut self, dtype: GgmlDType) -> CResult<()> {
        let w = match &self.base {
            BaseWeight::Dense(w) => w.clone(),
            BaseWeight::Quantized(q) => q.dequantize(&Device::Cpu)?,
            BaseWeight::Served(_) => return Ok(()),
        };
        let device = w.device().clone();
        let cpu = w.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let q = QTensor::quantize_onto(&cpu, dtype, &device)?;
        self.base = BaseWeight::Served(QMatMul::from_qtensor(q)?);
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
        let [q, k, v] = LoraLinear::forward_group(xs, [&self.q_proj, &self.k_proj, &self.v_proj])?;

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

        // One decoded token on the CPU, with nothing to differentiate, reads
        // the cache directly. A prompt, a mask, a gradient, another precision
        // or another device keeps the tensor ops.
        let direct = q_len == 1
            && mask.is_none()
            && q.device().is_cpu()
            && q.dtype() == DType::F32
            && !(q.track_op() || k.track_op() || v.track_op());
        let out = if direct {
            attend::decode_attention(&q, &k, &v)?
        } else {
            grouped_attention(&q, &k, &v, mask, self.num_kv_groups)?
        };
        let out = out.transpose(1, 2)?.reshape((b, q_len, self.hidden_size))?;
        let [out] = LoraLinear::forward_group(&out, [&self.o_proj])?;
        Ok(out)
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

    pub(super) fn serve_quantized(&mut self, dtype: GgmlDType) -> CResult<()> {
        self.q_proj.serve_quantized(dtype)?;
        self.k_proj.serve_quantized(dtype)?;
        self.v_proj.serve_quantized(dtype)?;
        self.o_proj.serve_quantized(dtype)
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
        let [gate, up] = LoraLinear::forward_group(xs, [&self.gate_proj, &self.up_proj])?;
        let lhs = self.act.forward(&gate)?;
        let [out] = LoraLinear::forward_group(&(lhs * up)?, [&self.down_proj])?;
        Ok(out)
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

    pub(super) fn serve_quantized(&mut self, dtype: GgmlDType) -> CResult<()> {
        self.gate_proj.serve_quantized(dtype)?;
        self.up_proj.serve_quantized(dtype)?;
        self.down_proj.serve_quantized(dtype)
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

    pub(super) fn serve_quantized(&mut self, dtype: GgmlDType) -> CResult<()> {
        self.self_attn.serve_quantized(dtype)?;
        self.mlp.serve_quantized(dtype)
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

    /// Attention as it was computed before: K and V repeated to the query
    /// heads' count, one product per head.
    fn repeated_attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        n_rep: usize,
    ) -> CResult<Tensor> {
        let head_dim = q.dim(3)?;
        let k = repeat_kv(k.clone(), n_rep)?.contiguous()?;
        let v = repeat_kv(v.clone(), n_rep)?.contiguous()?;
        let scale = 1f64 / (head_dim as f64).sqrt();
        let attn = (q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let attn = match mask {
            Some(m) => attn.broadcast_add(m)?,
            None => attn,
        };
        candle_nn::ops::softmax_last_dim(&attn)?.matmul(&v)
    }

    /// Stacking the query heads that share a KV head gives what repeating K
    /// and V gives: one decoded token against a cache, and a whole prompt
    /// under its causal mask.
    #[test]
    fn grouped_attention_is_attention_over_repeated_keys_and_values() -> CResult<()> {
        let dev = Device::Cpu;
        let (kv_heads, n_rep, head_dim) = (2, 3, 8);
        let draw =
            |heads: usize, len: usize| Tensor::randn(0f32, 1.0, (1, heads, len, head_dim), &dev);

        // A decode step: one new position, five in the cache.
        let (q, k, v) = (
            draw(kv_heads * n_rep, 1)?,
            draw(kv_heads, 5)?,
            draw(kv_heads, 5)?,
        );
        let grouped = grouped_attention(&q, &k, &v, None, n_rep)?;
        assert_eq!(grouped.dims(), &[1, kv_heads * n_rep, 1, head_dim]);
        let gap = largest_gap(&grouped, &repeated_attention(&q, &k, &v, None, n_rep)?)?;
        assert!(gap < 1e-5, "a decode step differs by {gap}");

        // A prompt of four positions after one cached, under the causal mask:
        // position i sees the cached one and positions up to its own.
        let (q, k, v) = (
            draw(kv_heads * n_rep, 4)?,
            draw(kv_heads, 5)?,
            draw(kv_heads, 5)?,
        );
        let mask: Vec<f32> = (0..4)
            .flat_map(|i| (0..5).map(move |j| if j > i + 1 { f32::NEG_INFINITY } else { 0.0 }))
            .collect();
        let mask = Tensor::from_vec(mask, (4, 5), &dev)?;
        let grouped = grouped_attention(&q, &k, &v, Some(&mask), n_rep)?;
        let gap = largest_gap(
            &grouped,
            &repeated_attention(&q, &k, &v, Some(&mask), n_rep)?,
        )?;
        assert!(gap < 1e-5, "a masked prompt differs by {gap}");
        Ok(())
    }

    /// A fresh adapter's factors, as a layer registers them, on the CPU.
    fn fresh_factors() -> CResult<VarMap> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        for layer in ["q", "v"] {
            let vb = vb.pp(layer);
            vb.get_with_hints(
                (4, 32),
                "lora_a",
                Init::Randn {
                    mean: 0.0,
                    stdev: LORA_A_STDEV,
                },
            )?;
            vb.get_with_hints((32, 4), "lora_b", candle_nn::init::ZERO)?;
        }
        Ok(varmap)
    }

    /// A bias-free LoRA linear over `base`, generating (grad off), scale 2.
    fn linear(base: BaseWeight, a: &Tensor, b: &Tensor) -> LoraLinear {
        LoraLinear {
            base,
            base_b: None,
            a: a.clone(),
            b: b.clone(),
            scale: 2.0,
            enabled: true,
            grad: false,
        }
    }

    /// A base served quantized multiplies close to the dense one, and the
    /// adapter on top of it is added exactly: the LoRA branch never touches
    /// the quantized weight.
    #[test]
    fn a_served_base_tracks_the_dense_one_and_keeps_the_adapter_exact() -> CResult<()> {
        let dev = Device::Cpu;
        let (out_f, in_f, rank) = (8, 256, 4);
        let w = Tensor::randn(0f32, 1.0, (out_f, in_f), &dev)?;
        let a = Tensor::randn(0f32, 0.1, (rank, in_f), &dev)?;
        let b = Tensor::randn(0f32, 0.1, (out_f, rank), &dev)?;
        let x = Tensor::randn(0f32, 1.0, (1, 3, in_f), &dev)?;
        let dense = linear(BaseWeight::Dense(w.clone()), &a, &b).forward(&x)?;
        let peak = dense.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        for (dtype, bound) in [(GgmlDType::Q8_0, 0.03), (GgmlDType::Q4K, 0.15)] {
            let mut served = linear(BaseWeight::Dense(w.clone()), &a, &b);
            served.serve_quantized(dtype)?;
            assert!(matches!(served.base, BaseWeight::Served(_)));
            let with = served.forward(&x)?;
            let gap = largest_gap(&dense, &with)? / peak;
            assert!(gap < bound, "{dtype:?}: off by {gap} of the peak");
            served.set_enabled(false);
            let without = served.forward(&x)?;
            let lora = matmul_t(&matmul_t(&x, &a)?, &b)?.affine(2.0, 0.0)?;
            assert!(largest_gap(&(with - without)?, &lora)? < 1e-4, "{dtype:?}");
        }
        Ok(())
    }

    /// Projections applied as a group give what each gives alone: one decoded
    /// row through the grouped deltas, over a dense base and a served one, and
    /// a prompt's rows, a silenced adapter or a training one through each
    /// projection's own forward.
    #[test]
    fn a_group_of_linears_gives_what_each_gives_alone() -> CResult<()> {
        let dev = Device::Cpu;
        let (in_f, rank) = (256, 4);
        let make = |out_f: usize| -> CResult<LoraLinear> {
            let w = Tensor::randn(0f32, 1.0, (out_f, in_f), &dev)?;
            let a = Tensor::randn(0f32, 0.1, (rank, in_f), &dev)?;
            let b = Tensor::randn(0f32, 0.1, (out_f, rank), &dev)?;
            Ok(linear(BaseWeight::Dense(w), &a, &b))
        };
        let (mut one, mut two) = (make(8)?, make(24)?);
        let row = Tensor::randn(0f32, 1.0, (1, 1, in_f), &dev)?;
        let rows = Tensor::randn(0f32, 1.0, (1, 3, in_f), &dev)?;

        let [first, second] = LoraLinear::forward_group(&row, [&one, &two])?;
        assert!(largest_gap(&first, &one.forward(&row)?)? < 1e-4);
        assert!(largest_gap(&second, &two.forward(&row)?)? < 1e-4);
        let [many] = LoraLinear::forward_group(&rows, [&two])?;
        assert_eq!(largest_gap(&many, &two.forward(&rows)?)?, 0.0, "a prompt");

        one.serve_quantized(GgmlDType::Q8_0)?;
        let [served] = LoraLinear::forward_group(&row, [&one])?;
        assert!(largest_gap(&served, &one.forward(&row)?)? < 1e-4, "served");

        two.set_enabled(false);
        let [bare] = LoraLinear::forward_group(&row, [&two])?;
        assert_eq!(largest_gap(&bare, &two.forward(&row)?)?, 0.0, "silenced");
        two.set_enabled(true);
        two.set_grad(true);
        let [tracked] = LoraLinear::forward_group(&row, [&two])?;
        assert_eq!(largest_gap(&tracked, &two.forward(&row)?)?, 0.0, "training");
        Ok(())
    }

    fn factor(varmap: &VarMap, name: &str) -> CResult<Vec<f32>> {
        let vars = varmap
            .data()
            .lock()
            .map_err(|_| candle_core::Error::msg("the varmap lock is poisoned"))?;
        vars[name].flatten_all()?.to_vec1::<f32>()
    }

    /// Two adapters seeded alike start from the same factors, whatever the
    /// device's generator drew for them first; another seed starts elsewhere.
    /// Only `lora_a` is drawn: `lora_b` stays zero, so a fresh adapter still
    /// changes nothing, and the spread is a fresh adapter's.
    #[test]
    fn a_seeded_adapter_starts_from_its_seed_and_no_other_state() -> CResult<()> {
        let (one, two, other) = (fresh_factors()?, fresh_factors()?, fresh_factors()?);
        assert_ne!(
            factor(&one, "q.lora_a")?,
            factor(&two, "q.lora_a")?,
            "unseeded, they differ"
        );
        seed_lora_a(&one, 7)?;
        seed_lora_a(&two, 7)?;
        seed_lora_a(&other, 8)?;
        for name in ["q.lora_a", "v.lora_a"] {
            assert_eq!(factor(&one, name)?, factor(&two, name)?, "{name}");
            assert_ne!(factor(&one, name)?, factor(&other, name)?, "{name}");
        }
        assert_ne!(factor(&one, "q.lora_a")?, factor(&one, "v.lora_a")?);
        assert!(factor(&one, "q.lora_b")?.iter().all(|x| *x == 0.0));
        let drawn = factor(&one, "q.lora_a")?;
        let sd = (drawn.iter().map(|x| x * x).sum::<f32>() / drawn.len() as f32).sqrt();
        assert!((0.01..0.03).contains(&sd), "spread {sd}");
        Ok(())
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
