//! Attention for one decoded token on the CPU, read straight from the cache.
//!
//! A decoded token's attention is, for each query head, a score against every
//! cached position, a softmax over them, and the weighted sum of the cached
//! values: a few hundred thousand multiply-adds a layer. As tensor ops those
//! are matrix products one or six rows tall, which the matmul library runs on
//! one thread at a small fraction of the CPU's speed.
//!
//! Here the unit of work is a run of one KV head's positions, and the query
//! heads that share that KV head are scored against a position together. So
//! each key and each value is read once a token, by one thread: a unit a
//! query head read its KV head's whole cache once for each of the heads
//! sharing it, six times over on a 1.5B base, and a decoded token is bound by
//! how fast memory is read. The work is two passes with the softmax between
//! them, one dispatch to candle's barrier pool: every run's scores, then,
//! once those are all in and each head's largest is known, every run's
//! weighted values, each run into room of its own. The runs are then added
//! in their order, and how the positions fall into runs depends only on how
//! many there are: a token's attention is the same sum, to the bit, whichever
//! thread took which run and however many threads there were. A handful of
//! positions are done where the caller stands.
//!
//! It has no backward, so it is used only where no gradient can flow: the
//! prompt pass and training keep the tensor ops.

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};

use candle_core::{bail, Device, Result, Storage, Tensor};

use super::kv::KvStore;
use super::slices::{dot, values};

/// Cached positions below which a token's attention is not worth a dispatch.
const INLINE_BELOW: usize = 32;
/// Runs a KV head's positions are split into, about; a run is a multiple of
/// `RUN_STEP` positions, no fewer than `RUN_LEAST` and no more than
/// `RUN_MOST`.
const RUNS: usize = 16;
const RUN_STEP: usize = 16;
const RUN_LEAST: usize = 32;
const RUN_MOST: usize = 256;

/// The two passes over a run of positions. `scores(q, keys, dim, scale, out,
/// stride, most)` writes head `r`'s score against position `p` to
/// `out[r * stride + p]` and raises `most[r]` to the largest; `weigh(scores,
/// stride, most, values, dim, acc, sums)` adds each position's values,
/// weighted by `e` to its score less `most[r]`, into `acc[r * dim..]`, and
/// the weights into `sums[r]`.
#[derive(Clone, Copy)]
struct Kernels {
    scores: Scoring,
    weigh: Weighing,
}

type Scoring = unsafe fn(&[f32], &[f32], usize, f32, *mut f32, usize, &mut [f32]);
type Weighing = unsafe fn(*const f32, usize, &[f32], &[f32], usize, &mut [f32], &mut [f32]);

/// SAFETY (for the caller): `out` has room at `r * stride + p` for every
/// head `r` of `q` and position `p` of `keys`, and no other thread writes
/// there.
unsafe fn plain_scores(
    q: &[f32],
    keys: &[f32],
    dim: usize,
    scale: f32,
    out: *mut f32,
    stride: usize,
    most: &mut [f32],
) {
    for (p, key) in keys.chunks_exact(dim).enumerate() {
        for (r, (query, top)) in q.chunks_exact(dim).zip(most.iter_mut()).enumerate() {
            let score = dot(query, key) * scale;
            *out.add(r * stride + p) = score;
            *top = top.max(score);
        }
    }
}

/// SAFETY (for the caller): `scores` holds a score at `r * stride + p` for
/// every head `r` of `most` and position `p` of `values`.
unsafe fn plain_weigh(
    scores: *const f32,
    stride: usize,
    most: &[f32],
    values: &[f32],
    dim: usize,
    acc: &mut [f32],
    sums: &mut [f32],
) {
    for (r, ((top, row), sum)) in most
        .iter()
        .zip(acc.chunks_exact_mut(dim))
        .zip(sums.iter_mut())
        .enumerate()
    {
        for (p, value) in values.chunks_exact(dim).enumerate() {
            let weight = (*scores.add(r * stride + p) - top).exp();
            *sum += weight;
            for (o, x) in row.iter_mut().zip(value) {
                *o += weight * x;
            }
        }
    }
}

const PLAIN: Kernels = Kernels {
    scores: plain_scores,
    weigh: plain_weigh,
};

/// The widest kernels this CPU has for heads of `dim` values.
fn kernels(dim: usize) -> Kernels {
    #[cfg(target_arch = "x86_64")]
    if super::tiles::available() && super::wide::attends(dim) {
        return Kernels {
            scores: super::wide::scores,
            weigh: super::wide::weigh,
        };
    }
    let _ = dim;
    PLAIN
}

/// Room attention works in, kept by its caller from one token to the next.
#[derive(Default)]
pub(super) struct Work {
    /// Every head's score against every position.
    scores: Vec<f32>,
    /// For each run of a KV head's positions, its query heads' weighted
    /// values and then their weights, summed over the run.
    sums: Vec<f32>,
}

/// Where every unit writes its own scores and no other unit's.
struct Scores(*mut f32);

// SAFETY: each unit of work writes the scores of its own run of positions,
// which no other unit covers, and they are read only after every unit has
// written.
unsafe impl Sync for Scores {}

fn raise(slot: &AtomicU32, to: f32) {
    // Nothing to do when the slot already holds as much.
    let _ = slot.fetch_update(Ordering::AcqRel, Ordering::Acquire, |bits| {
        (f32::from_bits(bits) < to).then(|| to.to_bits())
    });
}

/// Attention of one query position per head against every position in
/// `store`, for grouped-query attention where each KV head serves
/// `heads / kv_heads` query heads in order. `q` is each query head's `dim`
/// values, one head after another; `out` takes each head's `dim` attended
/// values in the same order. What `grouped_attention` gives for the same
/// query, keys and values with no mask.
pub(super) fn attend(q: &[f32], store: &KvStore, out: &mut [f32], work: &mut Work) -> Result<()> {
    attend_with(kernels(store.dim()), q, store, out, work)
}

fn attend_with(
    kernels: Kernels,
    q: &[f32],
    store: &KvStore,
    out: &mut [f32],
    work: &mut Work,
) -> Result<()> {
    let (kv_heads, dim, len) = (store.heads(), store.dim(), store.len());
    if dim == 0
        || kv_heads == 0
        || len == 0
        || q.is_empty()
        || !q.len().is_multiple_of(kv_heads * dim)
        || out.len() != q.len()
    {
        bail!(
            "decode attention: a query of {} values into {} does not fit {len} cached positions of {kv_heads} heads of {dim}",
            q.len(),
            out.len()
        )
    }
    let heads = q.len() / dim;
    let rep = heads / kv_heads;
    let scale = 1f32 / (dim as f32).sqrt();
    let Work { scores, sums } = work;
    scores.clear();
    scores.resize(heads * len, 0.0);

    // A unit is one KV head's run of positions. How long a run is depends on
    // how many positions there are and on nothing else, so the sums below are
    // the same sums on any number of threads.
    let run = len
        .div_ceil(RUNS)
        .next_multiple_of(RUN_STEP)
        .clamp(RUN_LEAST, RUN_MOST);
    let runs = len.div_ceil(run);
    let units = kv_heads * runs;
    let unit = |u: usize| {
        let (kv_head, first) = (u / runs, u % runs * run);
        (kv_head, first, len.min(first + run))
    };

    let at = Scores(scores.as_mut_ptr());
    let most: Vec<AtomicU32> = (0..heads)
        .map(|_| AtomicU32::new(f32::NEG_INFINITY.to_bits()))
        .collect();
    let (next_scored, scored, next_weighed) = (
        AtomicUsize::new(0),
        AtomicUsize::new(0),
        AtomicUsize::new(0),
    );
    // Each unit sums into room of its own: its heads' weighted values, then
    // their weights. One lock for every thread to add its sums under is asked
    // for by all of them at once as the pass ends, and a thread that waits
    // for a lock sleeps for longer than a layer's attention takes; and sums
    // kept by thread would depend on which thread took which run.
    let per = rep * dim + rep;
    sums.clear();
    sums.resize(units * per, 0.0);
    let rooms: Vec<Mutex<&mut [f32]>> = sums.chunks_exact_mut(per).map(Mutex::new).collect();

    let pool = candle_core::utils::barrier_pool();
    let work = |_: usize| {
        let at = &at;
        let mut top = vec![0f32; rep];
        loop {
            let u = next_scored.fetch_add(1, Ordering::Relaxed);
            if u >= units {
                break;
            }
            let (kv_head, first, last) = unit(u);
            top.fill(f32::NEG_INFINITY);
            // SAFETY: the kernels were chosen for this CPU and this `dim`;
            // head `kv_head * rep + r`'s scores for these positions are at
            // `(kv_head * rep + r) * len + p`, inside `scores`, and this unit
            // alone covers them.
            unsafe {
                (kernels.scores)(
                    &q[kv_head * rep * dim..(kv_head + 1) * rep * dim],
                    &store.keys(kv_head)[first * dim..last * dim],
                    dim,
                    scale,
                    at.0.add(kv_head * rep * len + first),
                    len,
                    &mut top,
                );
            }
            for (slot, top) in most[kv_head * rep..].iter().zip(&top) {
                raise(slot, *top);
            }
            scored.fetch_add(1, Ordering::Release);
        }
        // Every unit taken is finished by the thread that took it, so this
        // ends: the softmax needs each head's largest score over them all.
        while scored.load(Ordering::Acquire) < units {
            std::hint::spin_loop();
        }
        loop {
            let u = next_weighed.fetch_add(1, Ordering::Relaxed);
            if u >= units {
                break;
            }
            let (kv_head, first, last) = unit(u);
            for (top, slot) in top.iter_mut().zip(&most[kv_head * rep..]) {
                *top = f32::from_bits(slot.load(Ordering::Acquire));
            }
            // This unit's room, which no other unit locks.
            let mut room = rooms[u].lock().unwrap_or_else(PoisonError::into_inner);
            let (acc, weights) = room.split_at_mut(rep * dim);
            // SAFETY: every score was written before the wait above ended,
            // and these are the places the scoring pass wrote this unit's.
            unsafe {
                (kernels.weigh)(
                    at.0.add(kv_head * rep * len + first).cast_const(),
                    len,
                    &top,
                    &store.values(kv_head)[first * dim..last * dim],
                    dim,
                    acc,
                    weights,
                );
            }
        }
    };
    // A handful of positions are not worth a dispatch.
    if len < INLINE_BELOW || pool.n_workers() == 0 {
        work(0);
    } else {
        pool.execute(work);
    }

    // Each head's weighted values over its weights, the runs added in their
    // order.
    out.fill(0.0);
    let mut weights = vec![0f32; heads];
    for (u, room) in rooms.into_iter().enumerate() {
        let room = room.into_inner().unwrap_or_else(PoisonError::into_inner);
        let (acc, sum) = room.split_at(rep * dim);
        let kv_head = u / runs;
        for (o, a) in out[kv_head * rep * dim..].iter_mut().zip(acc) {
            *o += a;
        }
        for (w, s) in weights[kv_head * rep..].iter_mut().zip(sum) {
            *w += s;
        }
    }
    for (row, weight) in out.chunks_exact_mut(dim).zip(&weights) {
        for o in row.iter_mut() {
            *o /= weight;
        }
    }
    Ok(())
}

/// [`attend`] for a query held as a tensor of `(1, heads, 1, dim)`, f32 in
/// host memory: the attended values in the same shape.
pub(super) fn one(q: &Tensor, store: &KvStore) -> Result<Tensor> {
    let (b, heads, q_len, dim) = q.dims4()?;
    if b != 1 || q_len != 1 || dim != store.dim() {
        bail!(
            "decode attention: a query of {:?} does not fit a cache of {} heads of {}",
            q.dims(),
            store.heads(),
            store.dim()
        )
    }
    let q = q.contiguous()?;
    let mut out = vec![0f32; heads * dim];
    {
        let (storage, layout) = q.storage_and_layout();
        let Storage::Cpu(storage) = &*storage else {
            bail!("decode attention: the query is not in host memory")
        };
        let q = values(storage, layout, "decode attention: the query")?;
        attend(q, store, &mut out, &mut Work::default())?;
    }
    Tensor::from_vec(out, (1, heads, 1, dim), &Device::Cpu)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    fn largest_gap(a: &Tensor, b: &Tensor) -> Result<f32> {
        (a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()
    }

    fn stored(k: &Tensor, v: &Tensor) -> Result<KvStore> {
        let (_, kv_heads, _, d) = k.dims4()?;
        let mut store = KvStore::new(kv_heads, d);
        store.append_tensors(k, v)?;
        Ok(store)
    }

    /// One decoded token's attention from the cache is what grouped attention
    /// over the same query, keys and values gives: for every grouping, for
    /// one cached position, for a handful done where the caller stands and
    /// for enough to dispatch, through this CPU's widest kernels and through
    /// the plain ones.
    #[test]
    fn decode_attention_is_grouped_attention_for_one_query() -> Result<()> {
        let dev = Device::Cpu;
        for (kv_heads, n_rep, kv_len, d) in [
            (2, 6, 1, 128),
            (2, 6, 37, 128),
            (2, 6, 300, 128),
            (2, 6, 1000, 128),
            (2, 7, 45, 64),
            (2, 2, 33, 32),
            (3, 2, 17, 40),
            (4, 1, 9, 24),
        ] {
            let heads = kv_heads * n_rep;
            let q = Tensor::randn(0f32, 1.0, (1, heads, 1, d), &dev)?;
            let k = Tensor::randn(0f32, 1.0, (1, kv_heads, kv_len, d), &dev)?;
            let v = Tensor::randn(0f32, 1.0, (1, kv_heads, kv_len, d), &dev)?;
            let want = super::super::layers::grouped_attention(&q, &k, &v, None, n_rep)?;
            let store = stored(&k, &v)?;
            let got = one(&q, &store)?;
            assert_eq!(got.dims(), want.dims());
            let gap = largest_gap(&got, &want)?;
            assert!(
                gap < 1e-4,
                "{kv_heads}x{n_rep}, {kv_len} cached: off by {gap}"
            );

            let flat = q.flatten_all()?.to_vec1::<f32>()?;
            let mut plain = vec![0f32; flat.len()];
            attend_with(PLAIN, &flat, &store, &mut plain, &mut Work::default())?;
            let plain = Tensor::from_vec(plain, want.shape(), &dev)?;
            let gap = largest_gap(&plain, &want)?;
            assert!(
                gap < 1e-4,
                "{kv_heads}x{n_rep}, {kv_len} cached, plainly: off by {gap}"
            );
        }
        Ok(())
    }

    /// Attention over enough positions to dispatch is the same sum, to the
    /// bit, every time it is computed: no value depends on which thread took
    /// which run of positions.
    #[test]
    fn decode_attention_is_the_same_sum_every_time() -> Result<()> {
        let dev = Device::Cpu;
        let (kv_heads, n_rep, kv_len, d) = (2, 6, 700, 128);
        let q = Tensor::randn(0f32, 1.0, (1, kv_heads * n_rep, 1, d), &dev)?;
        let k = Tensor::randn(0f32, 1.0, (1, kv_heads, kv_len, d), &dev)?;
        let v = Tensor::randn(0f32, 1.0, (1, kv_heads, kv_len, d), &dev)?;
        let store = stored(&k, &v)?;
        let q = q.flatten_all()?.to_vec1::<f32>()?;
        let mut work = Work::default();
        let mut first = vec![0f32; q.len()];
        attend(&q, &store, &mut first, &mut work)?;
        for _ in 0..200 {
            let mut again = vec![0f32; q.len()];
            attend(&q, &store, &mut again, &mut work)?;
            assert_eq!(again, first);
        }
        Ok(())
    }

    /// A query that does not fit the cache, more than one query position, or
    /// a cache with nothing in it is refused rather than read out of bounds.
    #[test]
    fn decode_attention_refuses_what_does_not_fit() -> Result<()> {
        let dev = Device::Cpu;
        let k = Tensor::zeros((1, 2, 5, 8), DType::F32, &dev)?;
        let store = stored(&k, &k)?;
        let two = Tensor::zeros((1, 4, 2, 8), DType::F32, &dev)?;
        assert!(one(&two, &store).is_err(), "two query positions");
        let odd = Tensor::zeros((1, 3, 1, 8), DType::F32, &dev)?;
        assert!(one(&odd, &store).is_err(), "heads not a multiple");
        let wide = Tensor::zeros((1, 4, 1, 16), DType::F32, &dev)?;
        assert!(one(&wide, &store).is_err(), "head size differs");
        let fits = Tensor::zeros((1, 4, 1, 8), DType::F32, &dev)?;
        assert!(one(&fits, &KvStore::new(2, 8)).is_err(), "nothing cached");
        let mut short = vec![0f32; 8];
        let q = vec![0f32; 32];
        assert!(
            attend(&q, &store, &mut short, &mut Work::default()).is_err(),
            "no room for the result"
        );
        assert!(one(&fits, &store).is_ok());
        Ok(())
    }
}
