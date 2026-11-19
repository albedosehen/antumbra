//! An adapter's deltas for one decoded token on the CPU, for every projection
//! that reads the same input, in one pass over the factors.
//!
//! An adapter adds `scale · B · (A · x)` to a projection. As tensor ops that is
//! two matrix products one row tall, which the matmul library runs on one
//! thread, at all 196 projections of every decoded token. What bounds it is
//! how fast one core reads the factors (74 MB a token at rank 16 on a 1.5B
//! base, 295 MB at a four-expert blend's rank): 19 and 40 ms of a decoded
//! token. Here the projections that share an input (q, k and v; gate and up)
//! are one unit of dispatch on candle's barrier pool, the threads its quantized
//! products already run on, so the factors are read by every core.
//!
//! The factors are read where the trainer keeps them, at each call: nothing is
//! copied or packed, so a loaded, blended or retrained adapter is what the
//! next token uses. There is no backward: this is for generation, where the
//! factors are detached anyway. A prompt's rows, training and every GPU path
//! keep the tensor ops.

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};

use candle_core::{bail, CpuStorage, CustomOp1, Layout, Result, Shape, Storage, Tensor, D};

use super::slices::{dot, values};

/// One projection's adapter: `a` is `(rank, in)`, `b` is `(out, rank)`.
pub(super) struct Factors<'a> {
    pub(super) a: &'a Tensor,
    pub(super) b: &'a Tensor,
    pub(super) scale: f64,
}

/// `scale · B · (A · x)` for each projection in `group`, all reading the one
/// row `x` (`(.., in)` with every leading dimension 1), with no gradient. The
/// factors are f32 on the CPU with `x`. Each result is `(.., out)` for its
/// projection.
pub(super) fn grouped_delta(x: &Tensor, group: &[Factors]) -> Result<Vec<Tensor>> {
    let op = GroupedDelta {
        factors: group
            .iter()
            .map(|f| (f.a.clone(), f.b.clone(), f.scale as f32))
            .collect(),
    };
    let all = x.contiguous()?.apply_op1_no_bwd(&op)?;
    let mut deltas = Vec::with_capacity(group.len());
    let mut start = 0;
    for f in group {
        let out = f.b.dim(0)?;
        deltas.push(all.narrow(D::Minus1, start, out)?);
        start += out;
    }
    Ok(deltas)
}

struct GroupedDelta {
    factors: Vec<(Tensor, Tensor, f32)>,
}

/// One projection's factors as the op reads them.
struct Projection<'a> {
    a: &'a [f32],
    b: &'a [f32],
    rank: usize,
    out: usize,
    scale: f32,
    /// Where this projection's `rank` values of `A · x` start.
    through: usize,
}

/// Output values one unit of the B side covers.
const SPAN: usize = 512;
/// Rank rows a thread takes at a time on the A side.
const ROWS: usize = 4;

/// Every projection's delta into `out`, which holds them one after another.
/// Two phases in one dispatch: each rank row's `A · x`, then, once all of
/// those are in, each span of outputs against them.
fn deltas_into(x: &[f32], group: &[Projection], out: &mut [f32]) {
    let inn = x.len();
    // (projection, rank row) for each value of the A side.
    let rows: Vec<(usize, usize)> = group
        .iter()
        .enumerate()
        .flat_map(|(p, proj)| (0..proj.rank).map(move |j| (p, j)))
        .collect();
    let through: Vec<AtomicU32> = rows.iter().map(|_| AtomicU32::new(0)).collect();
    // (projection, first output) for each span of the B side, and the part of
    // `out` it alone writes: the locks are never contended.
    let mut spans = Vec::new();
    let mut rest = out;
    for (p, proj) in group.iter().enumerate() {
        let (mut mine, tail) = rest.split_at_mut(proj.out);
        rest = tail;
        // An adapter of no rank adds nothing: its outputs stay zero.
        if proj.rank == 0 {
            continue;
        }
        let mut start = 0;
        while !mine.is_empty() {
            let (head, tail) = mine.split_at_mut(SPAN.min(mine.len()));
            spans.push((p, start, Mutex::new(head)));
            start += SPAN;
            mine = tail;
        }
    }
    let longest = group.iter().map(|p| p.rank).max().unwrap_or(0);
    let (next_row, rows_done, next_span) = (
        AtomicUsize::new(0),
        AtomicUsize::new(0),
        AtomicUsize::new(0),
    );

    candle_core::utils::barrier_pool().execute(|_| {
        loop {
            let first = next_row.fetch_add(ROWS, Ordering::Relaxed);
            if first >= rows.len() {
                break;
            }
            let last = rows.len().min(first + ROWS);
            for (value, &(p, j)) in through[first..last].iter().zip(&rows[first..last]) {
                let proj = &group[p];
                let row = &proj.a[j * inn..(j + 1) * inn];
                value.store((dot(row, x) * proj.scale).to_bits(), Ordering::Relaxed);
            }
            rows_done.fetch_add(last - first, Ordering::Release);
        }
        // Every row taken is finished by the thread that took it, so this
        // ends: the B side reads all of a projection's A side.
        while rows_done.load(Ordering::Acquire) < rows.len() {
            std::hint::spin_loop();
        }
        let mut mine = vec![0f32; longest];
        while let Some((p, start, span)) = spans.get(next_span.fetch_add(1, Ordering::Relaxed)) {
            let proj = &group[*p];
            let mine = &mut mine[..proj.rank];
            let at = &through[proj.through..proj.through + proj.rank];
            for (m, value) in mine.iter_mut().zip(at) {
                *m = f32::from_bits(value.load(Ordering::Relaxed));
            }
            let b = &proj.b[start * proj.rank..];
            let mut span = span.lock().unwrap_or_else(PoisonError::into_inner);
            for (o, b_row) in span.iter_mut().zip(b.chunks_exact(proj.rank)) {
                *o = dot(b_row, mine);
            }
        }
    });
}

impl CustomOp1 for GroupedDelta {
    fn name(&self) -> &'static str {
        "grouped-adapter-delta"
    }

    fn cpu_fwd(&self, x: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let dims = layout.dims();
        let inn = dims.last().copied().unwrap_or(0);
        if inn == 0 || layout.shape().elem_count() != inn {
            bail!("adapter delta: one row is read at a time, not {dims:?}")
        }
        let x = values(x, layout, "adapter delta: the input")?;

        // Each factor's storage is held for the length of the op.
        let held: Vec<_> = self
            .factors
            .iter()
            .map(|(a, b, _)| (a.storage_and_layout(), b.storage_and_layout()))
            .collect();
        let mut group = Vec::with_capacity(held.len());
        let (mut through, mut total) = (0, 0);
        for (((a_storage, a_layout), (b_storage, b_layout)), (a, b, scale)) in
            held.iter().zip(&self.factors)
        {
            let (rank, a_in) = a.dims2()?;
            let (out, b_rank) = b.dims2()?;
            if b_rank != rank || a_in != inn {
                bail!(
                    "adapter delta: an input of {dims:?} does not fit factors of {:?} and {:?}",
                    a.dims(),
                    b.dims()
                )
            }
            let (Storage::Cpu(a_storage), Storage::Cpu(b_storage)) = (&**a_storage, &**b_storage)
            else {
                bail!("adapter delta: the factors are not on the cpu with their input")
            };
            group.push(Projection {
                a: values(a_storage, a_layout, "adapter delta: lora_a")?,
                b: values(b_storage, b_layout, "adapter delta: lora_b")?,
                rank,
                out,
                scale: *scale,
                through,
            });
            through += rank;
            total += out;
        }

        let mut out = vec![0f32; total];
        deltas_into(x, &group, &mut out);
        let mut shape = dims.to_vec();
        if let Some(last) = shape.last_mut() {
            *last = total;
        }
        Ok((CpuStorage::F32(out), Shape::from(shape)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn largest_gap(a: &Tensor, b: &Tensor) -> Result<f32> {
        (a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()
    }

    /// The delta as the tensor ops compute it.
    fn by_tensor_ops(x: &Tensor, a: &Tensor, b: &Tensor, scale: f64) -> Result<Tensor> {
        x.broadcast_matmul(&a.t()?)?
            .broadcast_matmul(&b.t()?)?
            .affine(scale, 0.0)
    }

    /// A group's deltas are the ones the tensor ops give each projection, for
    /// projections of different widths and ranks sharing an input, for one
    /// projection alone, and at a rank that is not a multiple of the lanes.
    #[test]
    fn a_groups_deltas_are_each_projections_own() -> Result<()> {
        let dev = Device::Cpu;
        for (inn, shapes) in [
            (1536, vec![(16, 1536), (16, 256), (16, 256)]),
            (1536, vec![(64, 8960), (64, 8960)]),
            (8960, vec![(16, 1536)]),
            (96, vec![(5, 40), (3, 1200), (20, 7)]),
        ] {
            let x = Tensor::randn(0f32, 1.0, (1, 1, inn), &dev)?;
            let made = shapes
                .iter()
                .map(|&(rank, out)| {
                    Ok((
                        Tensor::randn(0f32, 0.3, (rank, inn), &dev)?,
                        Tensor::randn(0f32, 0.3, (out, rank), &dev)?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            let group: Vec<Factors> = made
                .iter()
                .map(|(a, b)| Factors { a, b, scale: 2.0 })
                .collect();
            let deltas = grouped_delta(&x, &group)?;
            assert_eq!(deltas.len(), made.len());
            for (delta, (a, b)) in deltas.iter().zip(&made) {
                let want = by_tensor_ops(&x, a, b, 2.0)?;
                assert_eq!(delta.dims(), want.dims());
                let peak = want.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
                let gap = largest_gap(delta, &want)? / peak;
                assert!(
                    gap < 1e-4,
                    "{:?} from {inn}: off by {gap} of the peak",
                    b.dims()
                );
            }
        }
        Ok(())
    }

    /// An adapter of no rank adds nothing, beside one that does.
    #[test]
    fn an_adapter_of_no_rank_adds_nothing() -> Result<()> {
        let dev = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (1, 1, 32), &dev)?;
        let (a0, b0) = (
            Tensor::zeros((0, 32), DType::F32, &dev)?,
            Tensor::zeros((24, 0), DType::F32, &dev)?,
        );
        let (a1, b1) = (
            Tensor::randn(0f32, 0.3, (4, 32), &dev)?,
            Tensor::randn(0f32, 0.3, (8, 4), &dev)?,
        );
        let group = [
            Factors {
                a: &a0,
                b: &b0,
                scale: 2.0,
            },
            Factors {
                a: &a1,
                b: &b1,
                scale: 2.0,
            },
        ];
        let deltas = grouped_delta(&x, &group)?;
        let none = deltas[0].abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        assert_eq!(none, 0.0);
        assert_eq!(deltas[0].dims(), &[1, 1, 24]);
        assert!(largest_gap(&deltas[1], &by_tensor_ops(&x, &a1, &b1, 2.0)?)? < 1e-4);
        Ok(())
    }

    /// More than one row, or factors that do not fit the input or each other,
    /// are refused rather than read out of bounds.
    #[test]
    fn deltas_refuse_what_does_not_fit() -> Result<()> {
        let dev = Device::Cpu;
        let a = Tensor::zeros((4, 32), DType::F32, &dev)?;
        let b = Tensor::zeros((8, 4), DType::F32, &dev)?;
        let fits = [Factors {
            a: &a,
            b: &b,
            scale: 1.0,
        }];
        let rows = Tensor::zeros((1, 3, 32), DType::F32, &dev)?;
        assert!(grouped_delta(&rows, &fits).is_err(), "three rows");
        let narrow = Tensor::zeros((1, 1, 16), DType::F32, &dev)?;
        assert!(grouped_delta(&narrow, &fits).is_err(), "input width");
        let x = Tensor::zeros((1, 1, 32), DType::F32, &dev)?;
        let b_wrong = Tensor::zeros((8, 5), DType::F32, &dev)?;
        let mismatch = [Factors {
            a: &a,
            b: &b_wrong,
            scale: 1.0,
        }];
        assert!(grouped_delta(&x, &mismatch).is_err(), "ranks differ");
        Ok(())
    }
}
