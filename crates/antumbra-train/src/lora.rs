//! A trainable LoRA linear layer over a frozen base weight (ADR-0001/0002).
//!
//! `y = x · Wᵀ + scale · (x · Aᵀ) · Bᵀ`, where `W` is the frozen base (no
//! gradient) and `A`, `B` are the trainable low-rank factors. `B` starts at
//! zero (the LoRA convention), so a freshly attached adapter is a no-op until
//! it learns. Gradients flow only into `A`/`B`, never the base: the ADR-0001
//! freeze, by construction.
//!
//! This is a **standalone reference layer**: its CPU test (`lora_adapter_trains_on_cpu`)
//! is the workspace's GPU-free proof that the candle LoRA-training stack works
//! (gradients reach the adapter, the base is untouched). The production trainer
//! integrates an equivalent LoRA directly into the model (see the private
//! `LoraLinear` in [`crate::models`]'s Qwen) rather than using this layer.

use candle_core::{Result, Tensor};
use candle_nn::init::Init;
use candle_nn::VarBuilder;

pub struct LoraLinear {
    /// Frozen base weight, shape `(out, in)`. Not a `Var`, so it carries no gradient.
    base_w: Tensor,
    /// Trainable `(rank, in)`.
    a: Tensor,
    /// Trainable `(out, rank)`, zero-initialized.
    b: Tensor,
    scale: f64,
}

impl LoraLinear {
    /// Attach trainable LoRA factors (registered in `vb`'s VarMap) over a frozen
    /// base weight.
    pub fn new(
        vb: VarBuilder,
        in_features: usize,
        out_features: usize,
        rank: usize,
        base_w: Tensor,
        scale: f64,
    ) -> Result<Self> {
        let a = vb.get_with_hints(
            (rank, in_features),
            "lora_a",
            Init::Randn {
                mean: 0.0,
                stdev: 0.02,
            },
        )?;
        let b = vb.get_with_hints((out_features, rank), "lora_b", candle_nn::init::ZERO)?;
        Ok(Self {
            base_w,
            a,
            b,
            scale,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let base = x.matmul(&self.base_w.t()?)?;
        let lora = x
            .matmul(&self.a.t()?)?
            .matmul(&self.b.t()?)?
            .affine(self.scale, 0.0)?;
        base.add(&lora)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};

    /// Proves the candle training stack works inside the workspace (Windows,
    /// CPU): a LoRA adapter over a zero base learns a random linear map, and the
    /// loss falls sharply. This is MT-2 in miniature (gradients reach the
    /// adapter; the base is untouched).
    #[test]
    fn lora_adapter_trains_on_cpu() -> Result<()> {
        let dev = Device::Cpu;
        let (in_f, out_f, rank, batch) = (8usize, 4usize, 4usize, 32usize);

        // Frozen base is zero, so the adapter must learn the whole map.
        let base_w = Tensor::zeros((out_f, in_f), DType::F32, &dev)?;
        // The target we want the adapter to approximate.
        let target_w = Tensor::randn(0f32, 1.0, (out_f, in_f), &dev)?;
        let x = Tensor::randn(0f32, 1.0, (batch, in_f), &dev)?;
        let y = x.matmul(&target_w.t()?)?;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let lora = LoraLinear::new(vb, in_f, out_f, rank, base_w, 2.0)?;

        let mut opt = AdamW::new(
            varmap.all_vars(),
            ParamsAdamW {
                lr: 0.05,
                ..Default::default()
            },
        )?;

        let mut first = 0f32;
        let mut last = 0f32;
        for step in 0..300 {
            let pred = lora.forward(&x)?;
            let loss = (pred - &y)?.sqr()?.mean_all()?;
            let value = loss.to_scalar::<f32>()?;
            if step == 0 {
                first = value;
            }
            last = value;
            opt.backward_step(&loss)?;
        }
        assert!(
            last < first * 0.25,
            "LoRA training should cut loss sharply: {first} -> {last}"
        );
        Ok(())
    }
}
