//! `x · Wᵀ` against a frozen weight, with no gradient for the weight.
//!
//! candle's backward computes a gradient for both arguments of every matmul,
//! whether or not anything is training the argument, and keeps each one in the
//! gradient store until the optimizer has stepped. For LoRA over a frozen base
//! that is a full-size gradient for every base weight on every training step:
//! as much memory as the base itself, and as many FLOPs as the forward, for
//! gradients nothing reads. On the 1.5B base it is about 3 GB of a 24 GB card.
//!
//! This is the same product as a custom op whose backward returns the input's
//! gradient alone, so gradients still reach the LoRA factors upstream of it.

use candle_core::backend::BackendStorage;
use candle_core::{
    bail, CpuStorage, CudaStorage, CustomOp1, Error, Layout, MetalStorage, Result, Shape, Storage,
    Tensor,
};

/// `x · wᵀ` for `w` of shape `(out, in)` and `x` of shape `(.., in)`, with the
/// batch dimensions of `x` broadcast over `w`, as `candle_nn::Linear` does.
/// `w` is a constant: no gradient is computed for it, even if it is a `Var`.
pub fn frozen_matmul_t(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let op = FrozenMatmulT { w: w.clone() };
    let out = op.operands(x.layout())?.out;
    if out.elem_count() == 0 || w.elem_count() == 0 {
        return Tensor::zeros(out, x.dtype(), x.device());
    }
    x.apply_op1(op)
}

struct FrozenMatmulT {
    w: Tensor,
}

/// How one product is laid out for the storage-level matmul.
struct Operands {
    /// `(batch, m, n, k)`.
    bmnk: (usize, usize, usize, usize),
    /// The weight as the right operand: transposed to `(in, out)` and
    /// broadcast over the batch dimensions of `x`.
    rhs: Layout,
    out: Shape,
}

impl FrozenMatmulT {
    fn operands(&self, x: &Layout) -> Result<Operands> {
        let dims = x.dims();
        let (out_f, in_f) = self.w.dims2()?;
        let rank = dims.len();
        if rank < 2 || dims[rank - 1] != in_f {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: x.shape().clone(),
                rhs: self.w.shape().clone(),
                op: "frozen-matmul-t",
            }
            .bt());
        }
        let batch = &dims[..rank - 2];
        let m = dims[rank - 2];
        let rhs = self
            .w
            .layout()
            .transpose(0, 1)?
            .broadcast_as(Shape::from(batch).extend(&[in_f, out_f]))?;
        Ok(Operands {
            bmnk: (batch.iter().product(), m, out_f, in_f),
            rhs,
            out: Shape::from(batch).extend(&[m, out_f]),
        })
    }

    fn fwd<S: BackendStorage>(&self, x: &S, layout: &Layout, w: &S) -> Result<(S, Shape)> {
        let ops = self.operands(layout)?;
        Ok((x.matmul(w, ops.bmnk, layout, &ops.rhs)?, ops.out))
    }
}

impl CustomOp1 for FrozenMatmulT {
    fn name(&self) -> &'static str {
        "frozen-matmul-t"
    }

    fn cpu_fwd(&self, x: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let (w, _) = self.w.storage_and_layout();
        match &*w {
            Storage::Cpu(w) => self.fwd(x, layout, w),
            _ => bail!("frozen-matmul-t: the weight is not on the cpu with its input"),
        }
    }

    fn cuda_fwd(&self, x: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        let (w, _) = self.w.storage_and_layout();
        match &*w {
            Storage::Cuda(w) => self.fwd(x, layout, w),
            _ => bail!("frozen-matmul-t: the weight is not on the gpu with its input"),
        }
    }

    fn metal_fwd(&self, x: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        let (w, _) = self.w.storage_and_layout();
        match &*w {
            Storage::Metal(w) => self.fwd(x, layout, w),
            _ => bail!("frozen-matmul-t: the weight is not on the gpu with its input"),
        }
    }

    /// `d(x · wᵀ)/dx = grad · w`, and nothing for `w`.
    fn bwd(&self, _x: &Tensor, _res: &Tensor, grad: &Tensor) -> Result<Option<Tensor>> {
        let batch = &grad.dims()[..grad.rank() - 2];
        let w = self.w.broadcast_left(Shape::from(batch))?;
        grad.matmul(&w).map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Var};

    fn ramp(shape: &[usize], dev: &Device) -> Tensor {
        let n: usize = shape.iter().product();
        let values: Vec<f32> = (0..n).map(|i| ((i * 7 % 11) as f32 - 5.0) / 4.0).collect();
        Tensor::from_vec(values, shape, dev).unwrap()
    }

    /// The product as `candle_nn::Linear` computes it, weight gradient and all.
    fn plain(x: &Tensor, w: &Tensor) -> Tensor {
        let batch = &x.dims()[..x.rank() - 2];
        x.matmul(&w.broadcast_left(Shape::from(batch)).unwrap().t().unwrap())
            .unwrap()
    }

    fn max_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    #[test]
    fn the_product_is_the_linear_one_at_every_rank() {
        let dev = Device::Cpu;
        let w = ramp(&[5, 4], &dev);
        for shape in [&[3, 4][..], &[2, 3, 4], &[2, 2, 3, 4]] {
            let x = ramp(shape, &dev);
            let got = frozen_matmul_t(&x, &w).unwrap();
            assert_eq!(got.dims(), plain(&x, &w).dims());
            assert!(max_diff(&got, &plain(&x, &w)) < 1e-5, "at {shape:?}");
        }
    }

    #[test]
    fn a_strided_input_is_read_through_its_layout() {
        let dev = Device::Cpu;
        let w = ramp(&[5, 4], &dev);
        let x = ramp(&[2, 4, 3], &dev).transpose(1, 2).unwrap();
        assert!(!x.is_contiguous());
        let got = frozen_matmul_t(&x, &w).unwrap();
        assert!(max_diff(&got, &plain(&x, &w)) < 1e-5);
    }

    /// The input's gradient is the one the plain product gives, and the store
    /// keeps no gradient of the weight's shape, which the plain product does.
    #[test]
    fn the_input_gets_its_gradient_and_the_weight_gets_none() {
        let dev = Device::Cpu;
        let w = ramp(&[5, 4], &dev);
        let x = Var::from_tensor(&ramp(&[2, 3, 4], &dev)).unwrap();
        let r = ramp(&[2, 3, 5], &dev);
        let loss = |y: Tensor| (y * &r).unwrap().sum_all().unwrap();

        let frozen = loss(frozen_matmul_t(&x, &w).unwrap()).backward().unwrap();
        let reference = loss(plain(&x, &w)).backward().unwrap();
        let gx = frozen.get(&x).expect("a gradient for the input");
        assert!(max_diff(gx, reference.get(&x).unwrap()) < 1e-5);

        let weight_shaped = |store: &candle_core::backprop::GradStore| {
            store
                .get_ids()
                .filter_map(|id| store.get_id(*id))
                .any(|g| g.dims() == [2, 4, 5])
        };
        assert!(weight_shaped(&reference), "the plain product keeps one");
        assert!(!weight_shaped(&frozen));
    }

    #[test]
    fn an_input_of_the_wrong_width_is_refused() {
        let dev = Device::Cpu;
        let w = ramp(&[5, 4], &dev);
        assert!(frozen_matmul_t(&ramp(&[2, 3, 6], &dev), &w).is_err());
        assert!(frozen_matmul_t(&ramp(&[4], &dev), &w).is_err());
    }

    #[test]
    fn an_empty_input_gives_an_empty_product() {
        let dev = Device::Cpu;
        let w = ramp(&[5, 4], &dev);
        let x = Tensor::zeros((2, 0, 4), DType::F32, &dev).unwrap();
        assert_eq!(frozen_matmul_t(&x, &w).unwrap().dims(), [2, 0, 5]);
    }
}
