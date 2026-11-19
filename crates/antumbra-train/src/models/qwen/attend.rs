//! Attention for one decoded token on the CPU, read straight from the cache.
//!
//! A decoded token's attention is, for each query head, a score against every
//! cached position, a softmax over them, and the weighted sum of the cached
//! values: a few hundred thousand multiply-adds a layer. As tensor ops those
//! are matrix products one or six rows tall, which the matmul library runs on
//! one thread at a small fraction of the CPU's speed: about a sixth of a
//! decoded token on a 1.5B base. Here each query head is one unit of work on
//! candle's barrier pool, the threads its quantized products already run on.
//!
//! It has no backward, so it is used only where no gradient can flow: the
//! prompt pass and training keep the tensor ops.

use std::sync::{Mutex, PoisonError};

use candle_core::{bail, CpuStorage, CustomOp3, Layout, Result, Shape, Tensor};

use super::slices::{dot, values};

/// Attention of one query position per head against `kv_len` cached ones, for
/// grouped-query attention where each KV head serves `heads / kv_heads` query
/// heads in order. `q` is `(b, heads, 1, d)`, `k` and `v` are
/// `(b, kv_heads, kv_len, d)`, all f32 on the CPU; the result is
/// `(b, heads, 1, d)`, what `grouped_attention` returns for
/// the same inputs with no mask.
pub(super) fn decode_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    q.contiguous()?
        .apply_op3_no_bwd(&k.contiguous()?, &v.contiguous()?, &DecodeAttention)
}

struct DecodeAttention;

impl CustomOp3 for DecodeAttention {
    fn name(&self) -> &'static str {
        "decode-attention"
    }

    fn cpu_fwd(
        &self,
        q_storage: &CpuStorage,
        q_layout: &Layout,
        k_storage: &CpuStorage,
        k_layout: &Layout,
        v_storage: &CpuStorage,
        v_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (b, heads, q_len, d) = q_layout.shape().dims4()?;
        let (k_b, kv_heads, kv_len, k_d) = k_layout.shape().dims4()?;
        if q_len != 1
            || k_b != b
            || k_d != d
            || v_layout.shape() != k_layout.shape()
            || kv_heads == 0
            || kv_len == 0
            || !heads.is_multiple_of(kv_heads)
        {
            bail!(
                "decode attention: a query of {:?} does not fit keys of {:?} and values of {:?}",
                q_layout.dims(),
                k_layout.dims(),
                v_layout.dims()
            )
        }
        let n_rep = heads / kv_heads;
        let q = values(q_storage, q_layout, "decode attention: the query")?;
        let k = values(k_storage, k_layout, "decode attention: the keys")?;
        let v = values(v_storage, v_layout, "decode attention: the values")?;
        let scale = 1f32 / (d as f32).sqrt();

        let mut out = vec![0f32; b * heads * d];
        if d > 0 {
            // One row of the result per query head, each written by exactly
            // one unit of work: the locks are never contended.
            let rows: Vec<Mutex<&mut [f32]>> = out.chunks_mut(d).map(Mutex::new).collect();
            candle_core::utils::barrier_pool().execute_chunked(rows.len(), |units| {
                let mut weights = vec![0f32; kv_len];
                for unit in units {
                    let (batch, head) = (unit / heads, unit % heads);
                    let cached = (batch * kv_heads + head / n_rep) * kv_len * d;
                    let keys = &k[cached..cached + kv_len * d];
                    let vals = &v[cached..cached + kv_len * d];
                    let query = &q[unit * d..(unit + 1) * d];

                    let mut max = f32::NEG_INFINITY;
                    for (w, key) in weights.iter_mut().zip(keys.chunks_exact(d)) {
                        *w = dot(query, key) * scale;
                        max = max.max(*w);
                    }
                    let mut sum = 0f32;
                    for w in weights.iter_mut() {
                        *w = (*w - max).exp();
                        sum += *w;
                    }

                    let mut row = rows[unit].lock().unwrap_or_else(PoisonError::into_inner);
                    for (w, val) in weights.iter().zip(vals.chunks_exact(d)) {
                        let w = w / sum;
                        for (o, x) in row.iter_mut().zip(val) {
                            *o += w * x;
                        }
                    }
                }
            });
        }
        Ok((CpuStorage::F32(out), Shape::from((b, heads, 1, d))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn largest_gap(a: &Tensor, b: &Tensor) -> Result<f32> {
        (a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()
    }

    /// One decoded token's attention from the cache is what grouped attention
    /// over the same query, keys and values gives, for every grouping and for
    /// one cached position as for many.
    #[test]
    fn decode_attention_is_grouped_attention_for_one_query() -> Result<()> {
        let dev = Device::Cpu;
        for (b, kv_heads, n_rep, kv_len, d) in [
            (1, 2, 6, 1, 128),
            (1, 2, 6, 37, 128),
            (1, 2, 6, 300, 128),
            (2, 3, 2, 17, 40),
            (1, 4, 1, 9, 24),
        ] {
            let heads = kv_heads * n_rep;
            let q = Tensor::randn(0f32, 1.0, (b, heads, 1, d), &dev)?;
            let k = Tensor::randn(0f32, 1.0, (b, kv_heads, kv_len, d), &dev)?;
            let v = Tensor::randn(0f32, 1.0, (b, kv_heads, kv_len, d), &dev)?;
            let want = super::super::layers::grouped_attention(&q, &k, &v, None, n_rep)?;
            let got = decode_attention(&q, &k, &v)?;
            assert_eq!(got.dims(), want.dims());
            let gap = largest_gap(&got, &want)?;
            assert!(
                gap < 1e-4,
                "{b}x{kv_heads}x{n_rep}, {kv_len} cached: off by {gap}"
            );
        }
        Ok(())
    }

    /// A query that does not fit its keys, or more than one query position,
    /// is refused rather than read out of bounds.
    #[test]
    fn decode_attention_refuses_what_does_not_fit() -> Result<()> {
        let dev = Device::Cpu;
        let k = Tensor::zeros((1, 2, 5, 8), DType::F32, &dev)?;
        let two = Tensor::zeros((1, 4, 2, 8), DType::F32, &dev)?;
        assert!(
            decode_attention(&two, &k, &k).is_err(),
            "two query positions"
        );
        let odd = Tensor::zeros((1, 3, 1, 8), DType::F32, &dev)?;
        assert!(
            decode_attention(&odd, &k, &k).is_err(),
            "heads not a multiple"
        );
        let wide = Tensor::zeros((1, 4, 1, 16), DType::F32, &dev)?;
        assert!(
            decode_attention(&wide, &k, &k).is_err(),
            "head size differs"
        );
        Ok(())
    }
}
