//! Two of a layer's small steps in one pass each, for generation on the CPU:
//! the MLP's gate and the RMS norm.
//!
//! As tensor ops the gate is an activation and a product, each a pass over
//! the MLP's width on one thread, with an exponential for every value: 3.6 ms
//! of a decoded token on a 1.5B base. The norm is six ops, each allocating
//! its result, 57 times a token. Neither has a backward here: they are used
//! where nothing is differentiated, and training keeps the tensor ops.

use std::sync::{Mutex, PoisonError};

use candle_core::{bail, CpuStorage, CustomOp2, Layout, Result, Shape, Tensor};

use super::slices::{dot, values};

/// `silu(gate) · up`, value for value what the tensor ops give, spread over
/// candle's barrier pool. Both are f32 on the CPU and of one shape.
pub(super) fn silu_times(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    gate.contiguous()?
        .apply_op2_no_bwd(&up.contiguous()?, &SiluTimes)
}

struct SiluTimes;

/// Values one unit of work covers.
const SPAN: usize = 1024;

impl CustomOp2 for SiluTimes {
    fn name(&self) -> &'static str {
        "silu-times"
    }

    fn cpu_fwd(
        &self,
        gate: &CpuStorage,
        gate_layout: &Layout,
        up: &CpuStorage,
        up_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        if gate_layout.shape() != up_layout.shape() {
            bail!(
                "silu-times: a gate of {:?} against {:?}",
                gate_layout.dims(),
                up_layout.dims()
            )
        }
        let gate = values(gate, gate_layout, "silu-times: the gate")?;
        let up = values(up, up_layout, "silu-times: the other side")?;
        let mut out = vec![0f32; gate.len()];
        {
            // Each span is written by exactly one unit of work: the locks are
            // never contended.
            let spans: Vec<Mutex<&mut [f32]>> = out.chunks_mut(SPAN).map(Mutex::new).collect();
            candle_core::utils::barrier_pool().execute_chunked(spans.len(), |units| {
                for unit in units {
                    let mut span = spans[unit].lock().unwrap_or_else(PoisonError::into_inner);
                    let from = unit * SPAN;
                    let to = from + span.len();
                    for ((o, g), u) in span.iter_mut().zip(&gate[from..to]).zip(&up[from..to]) {
                        // candle's own silu, then its product.
                        *o = g / (1.0 + (-g).exp()) * u;
                    }
                }
            });
        }
        Ok((CpuStorage::F32(out), gate_layout.shape().clone()))
    }
}

/// The RMS norm of each row of `x` over its last dimension, scaled by
/// `weight`: `x / sqrt(mean(x²) + eps) · weight`. Both are f32 on the CPU.
pub(super) fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    x.contiguous()?
        .apply_op2_no_bwd(&weight.contiguous()?, &RmsNormRows { eps: eps as f32 })
}

struct RmsNormRows {
    eps: f32,
}

impl CustomOp2 for RmsNormRows {
    fn name(&self) -> &'static str {
        "rms-norm-rows"
    }

    fn cpu_fwd(
        &self,
        x: &CpuStorage,
        x_layout: &Layout,
        weight: &CpuStorage,
        weight_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let width = x_layout.dims().last().copied().unwrap_or(0);
        if width == 0 || weight_layout.dims() != [width] {
            bail!(
                "rms norm: rows of {:?} against a weight of {:?}",
                x_layout.dims(),
                weight_layout.dims()
            )
        }
        let x = values(x, x_layout, "rms norm: the input")?;
        let weight = values(weight, weight_layout, "rms norm: the weight")?;
        let mut out = vec![0f32; x.len()];
        for (o_row, x_row) in out.chunks_exact_mut(width).zip(x.chunks_exact(width)) {
            let mean = dot(x_row, x_row) / width as f32;
            let root = (mean + self.eps).sqrt();
            for ((o, v), w) in o_row.iter_mut().zip(x_row).zip(weight) {
                *o = v / root * w;
            }
        }
        Ok((CpuStorage::F32(out), x_layout.shape().clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, D};

    fn largest_gap(a: &Tensor, b: &Tensor) -> Result<f32> {
        (a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()
    }

    /// The gate in one pass is the tensor ops' gate to the last bit, for one
    /// row and for a prompt's, at a width that does not fill its last span.
    #[test]
    fn the_gate_in_one_pass_is_the_tensor_ops_gate() -> Result<()> {
        let dev = Device::Cpu;
        for shape in [(1, 1, 8960), (1, 3, 1000), (2, 2, 7)] {
            let gate = Tensor::randn(0f32, 3.0, shape, &dev)?;
            let up = Tensor::randn(0f32, 1.0, shape, &dev)?;
            let want = (gate.silu()? * &up)?;
            let got = silu_times(&gate, &up)?;
            assert_eq!(got.dims(), want.dims());
            assert_eq!(largest_gap(&got, &want)?, 0.0, "{shape:?}");
        }
        let gate = Tensor::zeros((1, 1, 8), DType::F32, &dev)?;
        let other = Tensor::zeros((1, 1, 9), DType::F32, &dev)?;
        assert!(silu_times(&gate, &other).is_err(), "shapes differ");
        Ok(())
    }

    /// The norm in one pass is the tensor ops' norm, for one row and several.
    #[test]
    fn the_norm_in_one_pass_is_the_tensor_ops_norm() -> Result<()> {
        let dev = Device::Cpu;
        let eps = 1e-6;
        for shape in [(1, 1, 1536), (2, 5, 64), (1, 4, 17)] {
            let x = Tensor::randn(0f32, 2.0, shape, &dev)?;
            let weight = Tensor::randn(1f32, 0.2, shape.2, &dev)?;
            let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
            let want = x
                .broadcast_div(&(variance + eps)?.sqrt()?)?
                .broadcast_mul(&weight)?;
            let got = rms_norm(&x, &weight, eps)?;
            assert_eq!(got.dims(), want.dims());
            let peak = want.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            let gap = largest_gap(&got, &want)? / peak;
            assert!(gap < 1e-5, "{shape:?}: off by {gap} of the peak");
        }
        let x = Tensor::zeros((1, 1, 8), DType::F32, &dev)?;
        let weight = Tensor::zeros(9, DType::F32, &dev)?;
        assert!(rms_norm(&x, &weight, eps).is_err(), "widths differ");
        Ok(())
    }
}
