//! The plain-row arithmetic of [`super`] in 128-bit vectors, for the aarch64
//! CPUs that multiply through tiles: the MLP's gate, attention's scores and
//! weighted values, and an adapter's factors.
//!
//! Each kernel is its AVX-512 twin with every vector of sixteen taken as
//! four of four: the same steps on every value, fused the same way, and a
//! sum across a vector's values added in the order the twin adds it. So a
//! row's values are the values an x86-64 CPU gives, to the bit. No CPU runs
//! both, so each kernel is held to a mark of the twin's values over inputs
//! drawn from a seed (`super::marks`), and here to the arithmetic a value at
//! a time.

use super::{EXP_LEAST, EXP_MOST, LN2_HIGH, LN2_LOW, LOG2_E, SPAN, SUMS, TOGETHER};
use std::arch::aarch64::*;

/// Eight values, two fours.
type F8 = [float32x4_t; 2];
/// Sixteen values, four fours.
type F16 = [float32x4_t; 4];

/// `e` to each of four values: the steps of the twin's exponential, and its
/// values. The power of two is put back in two halves, as the AVX2 kernel
/// puts it: two to the 128th is no f32, though a value just under one times
/// it is, and both halves of any power here are.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn exp(x: float32x4_t) -> float32x4_t {
    // A value that is not a number comes back as one.
    let x = vminq_f32(vdupq_n_f32(EXP_MOST), vmaxq_f32(vdupq_n_f32(EXP_LEAST), x));
    let n = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(LOG2_E)));
    let r = vfmsq_f32(x, n, vdupq_n_f32(LN2_HIGH));
    let r = vfmsq_f32(r, n, vdupq_n_f32(LN2_LOW));
    let mut p = vdupq_n_f32(1.987_569_1e-4);
    p = vfmaq_f32(vdupq_n_f32(1.398_2e-3), p, r);
    p = vfmaq_f32(vdupq_n_f32(8.333_452e-3), p, r);
    p = vfmaq_f32(vdupq_n_f32(4.166_579_6e-2), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.666_666_5e-1), p, r);
    p = vfmaq_f32(vdupq_n_f32(0.5), p, r);
    let e = vaddq_f32(vfmaq_f32(r, p, vmulq_f32(r, r)), vdupq_n_f32(1.0));
    let whole = vcvtq_s32_f32(n);
    let first = vshrq_n_s32::<1>(whole);
    let second = vsubq_s32(whole, first);
    let bias = vdupq_n_s32(127);
    let by_first = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(first, bias)));
    let by_second = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(second, bias)));
    vmulq_f32(vmulq_f32(e, by_first), by_second)
}

/// `silu(gate) · up` for four values.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn gated(g: float32x4_t, u: float32x4_t) -> float32x4_t {
    let e = exp(vsubq_f32(vdupq_n_f32(0.0), g));
    vmulq_f32(vdivq_f32(g, vaddq_f32(vdupq_n_f32(1.0), e)), u)
}

/// `silu(gate) · up` into `out`, over as many values as the three share.
///
/// SAFETY (for the caller): the CPU has NEON, as every CPU this is compiled
/// for has.
#[target_feature(enable = "neon")]
pub(in crate::models::qwen) unsafe fn silu_times(gate: &[f32], up: &[f32], out: &mut [f32]) {
    let n = gate.len().min(up.len()).min(out.len());
    let mut i = 0;
    while i + 4 <= n {
        let v = gated(
            vld1q_f32(gate.as_ptr().add(i)),
            vld1q_f32(up.as_ptr().add(i)),
        );
        vst1q_f32(out.as_mut_ptr().add(i), v);
        i += 4;
    }
    if i < n {
        // The last step takes what is left, beside zeros.
        let (mut g, mut u, mut o) = ([0f32; 4], [0f32; 4], [0f32; 4]);
        g[..n - i].copy_from_slice(&gate[i..n]);
        u[..n - i].copy_from_slice(&up[i..n]);
        vst1q_f32(
            o.as_mut_ptr(),
            gated(vld1q_f32(g.as_ptr()), vld1q_f32(u.as_ptr())),
        );
        out[i..n].copy_from_slice(&o[..n - i]);
    }
}

/// The sum of sixteen values, four fours, added in the order the twin's own
/// reduction adds a vector's: the two eights, then the two fours, then the
/// two pairs, then the two.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn across(v: F16) -> f32 {
    let four = vaddq_f32(vaddq_f32(v[0], v[2]), vaddq_f32(v[1], v[3]));
    let two = vadd_f32(vget_low_f32(four), vget_high_f32(four));
    vget_lane_f32::<0>(two) + vget_lane_f32::<1>(two)
}

#[target_feature(enable = "neon")]
unsafe fn scores_of<const N: usize>(
    q: &[f32],
    keys: &[f32],
    scale: f32,
    out: *mut f32,
    stride: usize,
    most: &mut [f32],
) {
    // The twin's vectors of sixteen to a head.
    let dim = N * 16;
    let rep = q.len() / dim;
    for p in 0..keys.len() / dim {
        let key = keys.as_ptr().add(p * dim);
        for r in 0..rep {
            let query = q.as_ptr().add(r * dim);
            // The twin's two running sums of sixteen, each as four of four.
            let (mut a, mut b) = ([vdupq_n_f32(0.0); 4], [vdupq_n_f32(0.0); 4]);
            for j in (0..N).step_by(2) {
                for h in 0..4 {
                    let at = j * 16 + h * 4;
                    a[h] = vfmaq_f32(a[h], vld1q_f32(key.add(at)), vld1q_f32(query.add(at)));
                    b[h] = vfmaq_f32(
                        b[h],
                        vld1q_f32(key.add(at + 16)),
                        vld1q_f32(query.add(at + 16)),
                    );
                }
            }
            let score = across([
                vaddq_f32(a[0], b[0]),
                vaddq_f32(a[1], b[1]),
                vaddq_f32(a[2], b[2]),
                vaddq_f32(a[3], b[3]),
            ]) * scale;
            *out.add(r * stride + p) = score;
            if score > *most.get_unchecked(r) {
                *most.get_unchecked_mut(r) = score;
            }
        }
    }
}

/// The scores of the query heads in `q` against each position of `keys`, as
/// [`super::scores`] writes them on an x86-64 CPU.
///
/// SAFETY (for the caller): [`super::attends`] holds of `dim`, `most` has a
/// value for each head in `q`, and `out` has room at every place written
/// with no other thread writing there.
#[target_feature(enable = "neon")]
pub(in crate::models::qwen) unsafe fn scores(
    q: &[f32],
    keys: &[f32],
    dim: usize,
    scale: f32,
    out: *mut f32,
    stride: usize,
    most: &mut [f32],
) {
    match dim {
        32 => scores_of::<2>(q, keys, scale, out, stride, most),
        64 => scores_of::<4>(q, keys, scale, out, stride, most),
        _ => scores_of::<8>(q, keys, scale, out, stride, most),
    }
}

/// `K` vectors of four at `row` take a span's positions: each position's
/// same `K` vectors at `values`, `dim` values apart, times its weight, added
/// in the positions' order. The sums stay in registers while the span goes
/// by.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn weighted<const K: usize>(row: *mut f32, weights: &[f32], values: *const f32, dim: usize) {
    let mut sum = [vdupq_n_f32(0.0); K];
    for (k, s) in sum.iter_mut().enumerate() {
        *s = vld1q_f32(row.add(k * 4));
    }
    for (p, weight) in weights.iter().enumerate() {
        let by = vdupq_n_f32(*weight);
        let value = values.add(p * dim);
        for (k, s) in sum.iter_mut().enumerate() {
            *s = vfmaq_f32(*s, by, vld1q_f32(value.add(k * 4)));
        }
    }
    for (k, s) in sum.iter().enumerate() {
        vst1q_f32(row.add(k * 4), *s);
    }
}

#[target_feature(enable = "neon")]
unsafe fn weigh_of<const N: usize>(
    scores: *const f32,
    stride: usize,
    most: &[f32],
    values: &[f32],
    acc: &mut [f32],
    sums: &mut [f32],
) {
    let dim = N * 16;
    let positions = values.len() / dim;
    let mut weights = [0f32; SPAN];
    for (r, top) in most.iter().enumerate() {
        let top = vdupq_n_f32(*top);
        let row = acc.as_mut_ptr().add(r * dim);
        let mut total = 0f32;
        let mut at = 0;
        while at < positions {
            let n = SPAN.min(positions - at);
            let from = scores.add(r * stride + at);
            let mut i = 0;
            while i + 4 <= n {
                vst1q_f32(
                    weights.as_mut_ptr().add(i),
                    exp(vsubq_f32(vld1q_f32(from.add(i)), top)),
                );
                i += 4;
            }
            if i < n {
                // The last step takes what is left, beside zeros.
                let (mut some, mut done) = ([0f32; 4], [0f32; 4]);
                std::ptr::copy_nonoverlapping(from.add(i), some.as_mut_ptr(), n - i);
                vst1q_f32(
                    done.as_mut_ptr(),
                    exp(vsubq_f32(vld1q_f32(some.as_ptr()), top)),
                );
                weights[i..n].copy_from_slice(&done[..n - i]);
            }
            for weight in &weights[..n] {
                total += weight;
            }
            // A value's sum takes the positions in their order whatever
            // values are summed beside it, so the head goes by in runs of
            // as many vectors as stay in registers: sixteen, a head of 128
            // values in two.
            let (weights, value) = (&weights[..n], values.as_ptr().add(at * dim));
            match N {
                2 => weighted::<8>(row, weights, value, dim),
                4 => weighted::<16>(row, weights, value, dim),
                _ => {
                    weighted::<16>(row, weights, value, dim);
                    weighted::<16>(row.add(64), weights, value.add(64), dim);
                }
            }
            at += n;
        }
        *sums.get_unchecked_mut(r) += total;
    }
}

/// Each position of `values` weighted by `e` to its score less its head's
/// largest and added into `acc`, as [`super::weigh`] adds them on an x86-64
/// CPU.
///
/// SAFETY (for the caller): [`super::attends`] holds of `dim`, `acc` has
/// `dim` values and `sums` one for each head, and `scores` holds a score for
/// every head and position at `r * stride + p`.
#[target_feature(enable = "neon")]
pub(in crate::models::qwen) unsafe fn weigh(
    scores: *const f32,
    stride: usize,
    most: &[f32],
    values: &[f32],
    dim: usize,
    acc: &mut [f32],
    sums: &mut [f32],
) {
    match dim {
        32 => weigh_of::<2>(scores, stride, most, values, acc, sums),
        64 => weigh_of::<4>(scores, stride, most, values, acc, sums),
        _ => weigh_of::<8>(scores, stride, most, values, acc, sums),
    }
}

/// Eight factors at `a`, whole numbers of eight bits, as floats.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn eights(a: *const i8) -> F8 {
    let wide = vmovl_s8(vld1_s8(a));
    [
        vcvtq_f32_s32(vmovl_s16(vget_low_s16(wide))),
        vcvtq_f32_s32(vmovl_high_s16(wide)),
    ]
}

/// Sixteen factors at `a` as floats.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn sixteen(a: *const i8) -> F16 {
    let (low, high) = (eights(a), eights(a.add(8)));
    [low[0], low[1], high[0], high[1]]
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn summed(sums: &[F8; SUMS]) -> F8 {
    let mut out = [vdupq_n_f32(0.0); 2];
    for (h, out) in out.iter_mut().enumerate() {
        *out = vaddq_f32(
            vaddq_f32(
                vaddq_f32(sums[0][h], sums[1][h]),
                vaddq_f32(sums[2][h], sums[3][h]),
            ),
            vaddq_f32(
                vaddq_f32(sums[4][h], sums[5][h]),
                vaddq_f32(sums[6][h], sums[7][h]),
            ),
        );
    }
    out
}

/// Sixteen rank rows of an adapter's `A` over `n` columns, times those
/// columns of each of `rows` rows of input, as [`super::through_cols`]
/// writes them on an x86-64 CPU: the sixteen sums a row are two eights
/// here, each gone through on its own, a column still added to running sum
/// `c % 8` in the order of the columns.
///
/// SAFETY (for the caller): `a` holds `16 * n` factors; each row of `x`
/// holds `n` values; and each place in `out` is room for sixteen.
#[target_feature(enable = "neon")]
pub(in crate::models::qwen) unsafe fn through_cols(
    a: *const i8,
    n: usize,
    x: *const f32,
    stride: usize,
    rows: usize,
    out: *mut f32,
    out_stride: usize,
) {
    for half in 0..2 {
        let a = a.add(half * 8);
        let mut first = 0;
        while first < rows {
            let some = (rows - first).min(TOGETHER);
            let x = x.add(first * stride);
            let mut sums = [[[vdupq_n_f32(0.0); 2]; SUMS]; TOGETHER];
            let mut c = 0;
            while c + SUMS <= n {
                let mut cols = [[vdupq_n_f32(0.0); 2]; SUMS];
                for (j, col) in cols.iter_mut().enumerate() {
                    *col = eights(a.add((c + j) * 16));
                }
                for (i, sums) in sums[..some].iter_mut().enumerate() {
                    let x = x.add(i * stride + c);
                    for (j, sum) in sums.iter_mut().enumerate() {
                        let by = vdupq_n_f32(*x.add(j));
                        sum[0] = vfmaq_f32(sum[0], cols[j][0], by);
                        sum[1] = vfmaq_f32(sum[1], cols[j][1], by);
                    }
                }
                c += SUMS;
            }
            for j in 0..n - c {
                let col = eights(a.add((c + j) * 16));
                for (i, sums) in sums[..some].iter_mut().enumerate() {
                    let by = vdupq_n_f32(*x.add(i * stride + c + j));
                    sums[j][0] = vfmaq_f32(sums[j][0], col[0], by);
                    sums[j][1] = vfmaq_f32(sums[j][1], col[1], by);
                }
            }
            for (i, sums) in sums[..some].iter().enumerate() {
                let at = out.add((first + i) * out_stride + half * 8);
                let [low, high] = summed(sums);
                vst1q_f32(at, low);
                vst1q_f32(at.add(4), high);
            }
            first += some;
        }
    }
}

/// To `sixteens` sixteens of products of each of `rows` rows, an adapter's
/// `B` times that row's `A · x`, as [`super::add_sixteens`] adds them on an
/// x86-64 CPU: the ranks still added in their order into two sums, the even
/// and the odd, and the two to the products, sixteen ranks at a time.
///
/// One row makes a rank's factors floats as it comes to them. Several make
/// them floats once for every row there is, eight of a sixteen at a time.
///
/// SAFETY (for the caller): `b` holds `16 * rank * sixteens` factors; `t`
/// holds `rank` values for every row; and each row's place in `out` holds
/// `16 * sixteens` values that nothing else reads or writes meanwhile.
#[target_feature(enable = "neon")]
pub(in crate::models::qwen) unsafe fn add_sixteens(
    b: *const i8,
    rank: usize,
    sixteens: usize,
    t: *const f32,
    rows: usize,
    out: *mut f32,
    stride: usize,
) {
    let zero = vdupq_n_f32(0.0);
    for s in 0..sixteens {
        let b = b.add(s * rank * 16);
        let mut first = 0;
        while first < rank {
            let some = (rank - first).min(16);
            if rows == 1 {
                let (b, t) = (b.add(first * 16), t.add(first));
                let (mut even, mut odd) = ([zero; 4], [zero; 4]);
                let mut r = 0;
                while r + 2 <= some {
                    let (f, g) = (sixteen(b.add(r * 16)), sixteen(b.add((r + 1) * 16)));
                    let (by, next) = (vdupq_n_f32(*t.add(r)), vdupq_n_f32(*t.add(r + 1)));
                    for k in 0..4 {
                        even[k] = vfmaq_f32(even[k], f[k], by);
                        odd[k] = vfmaq_f32(odd[k], g[k], next);
                    }
                    r += 2;
                }
                if r < some {
                    let f = sixteen(b.add(r * 16));
                    let by = vdupq_n_f32(*t.add(r));
                    for k in 0..4 {
                        even[k] = vfmaq_f32(even[k], f[k], by);
                    }
                }
                for k in 0..4 {
                    let at = out.add(s * 16 + k * 4);
                    vst1q_f32(at, vaddq_f32(vld1q_f32(at), vaddq_f32(even[k], odd[k])));
                }
                first += some;
                continue;
            }
            for half in 0..2 {
                let mut factors = [[zero; 2]; 16];
                for (r, factor) in factors[..some].iter_mut().enumerate() {
                    *factor = eights(b.add((first + r) * 16 + half * 8));
                }
                for i in 0..rows {
                    let t = t.add(i * rank + first);
                    let (mut even, mut odd) = ([zero; 2], [zero; 2]);
                    let mut r = 0;
                    while r + 2 <= some {
                        let (by, next) = (vdupq_n_f32(*t.add(r)), vdupq_n_f32(*t.add(r + 1)));
                        even[0] = vfmaq_f32(even[0], factors[r][0], by);
                        even[1] = vfmaq_f32(even[1], factors[r][1], by);
                        odd[0] = vfmaq_f32(odd[0], factors[r + 1][0], next);
                        odd[1] = vfmaq_f32(odd[1], factors[r + 1][1], next);
                        r += 2;
                    }
                    if r < some {
                        let by = vdupq_n_f32(*t.add(r));
                        even[0] = vfmaq_f32(even[0], factors[r][0], by);
                        even[1] = vfmaq_f32(even[1], factors[r][1], by);
                    }
                    let at = out.add(i * stride + s * 16 + half * 8);
                    vst1q_f32(at, vaddq_f32(vld1q_f32(at), vaddq_f32(even[0], odd[0])));
                    vst1q_f32(
                        at.add(4),
                        vaddq_f32(vld1q_f32(at.add(4)), vaddq_f32(even[1], odd[1])),
                    );
                }
            }
            first += some;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Values from a seed, the same in every run, between -1 and 1.
    fn values(seed: u64, n: usize) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect()
    }

    /// The vector exponential is the scalar one to within two units in the
    /// last place across its whole range, is nothing far below it and past
    /// any value far above, and leaves what is not a number as it is.
    #[test]
    fn the_exponential_in_fours_is_the_scalar_one() {
        let mut worst = 0f64;
        let mut x = -87.0f32;
        while x < 88.5 {
            let xs: [f32; 4] = std::array::from_fn(|i| x + i as f32 * 0.003_7);
            let mut got = [0f32; 4];
            // SAFETY: every aarch64 CPU has NEON, and both are four values.
            unsafe { vst1q_f32(got.as_mut_ptr(), exp(vld1q_f32(xs.as_ptr()))) };
            for (g, v) in got.iter().zip(xs) {
                let want = f64::from(v).exp();
                worst = worst.max(((f64::from(*g) - want) / want).abs());
            }
            x += 0.0153;
        }
        assert!(worst < 2.5e-7, "off by {worst} of the value");
        let edges = [
            [-1000.0f32, 1000.0, f32::NAN, 0.0],
            [-0.0, f32::NEG_INFINITY, f32::INFINITY, 1.0],
        ];
        let mut got = [[0f32; 4]; 2];
        for (got, xs) in got.iter_mut().zip(&edges) {
            // SAFETY: as above.
            unsafe { vst1q_f32(got.as_mut_ptr(), exp(vld1q_f32(xs.as_ptr()))) };
        }
        assert!(
            got[0][0] < 1e-37 && got[1][1] < 1e-37,
            "far below is nothing"
        );
        assert!(
            got[0][1] > 1e38 && got[1][2] > 1e38,
            "far above is past any value"
        );
        assert!(got[0][2].is_nan());
        assert_eq!((got[0][3], got[1][0]), (1.0, 1.0));
        assert!((got[1][3] - std::f32::consts::E).abs() < 3e-7);
    }

    /// The gate in fours is the gate a value at a time to within its
    /// rounding, for widths on and off the vectors' multiple, and writes
    /// nothing past the end.
    #[test]
    fn the_gate_in_fours_is_the_gate_a_value_at_a_time() {
        for n in [0usize, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 100, 560] {
            let gate: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin() * 9.0).collect();
            let up: Vec<f32> = (0..n).map(|i| (i as f32 * 0.11).cos() * 3.0).collect();
            let mut got = vec![7f32; n + 3];
            // SAFETY: every aarch64 CPU has NEON.
            unsafe { silu_times(&gate, &up, &mut got[..n]) };
            for i in 0..n {
                let want = gate[i] / (1.0 + (-gate[i]).exp()) * up[i];
                assert!(
                    (got[i] - want).abs() <= 1e-6 * want.abs().max(1.0),
                    "{n} values, the {i}th: {} against {want}",
                    got[i]
                );
            }
            assert_eq!(
                &got[n..],
                &[7.0, 7.0, 7.0],
                "nothing past the end is written"
            );
        }
    }

    /// Attention's two passes in fours are the passes a value at a time to
    /// within the order of a sum: for every width of head, with one query
    /// head and with several, and for runs of positions on and off the
    /// vectors' multiple and longer than a span.
    #[test]
    fn attention_in_fours_is_attention_a_value_at_a_time() {
        for dim in [32usize, 64, 128] {
            for (rep, positions) in [
                (1usize, 1usize),
                (6, 3),
                (6, 7),
                (2, 16),
                (6, 129),
                (3, 300),
            ] {
                let q = values(dim as u64 + rep as u64, rep * dim);
                let keys = values(7 + positions as u64, positions * dim);
                let vals = values(11 + dim as u64, positions * dim);
                let scale = 1.0 / (dim as f32).sqrt();
                let mut scored = vec![0f32; rep * positions];
                let mut most = vec![f32::NEG_INFINITY; rep];
                let mut acc = vec![0.25f32; rep * dim];
                let mut sums = vec![0.5f32; rep];
                // SAFETY: every aarch64 CPU has NEON, and every place has
                // the room the kernels are told of.
                unsafe {
                    scores(
                        &q,
                        &keys,
                        dim,
                        scale,
                        scored.as_mut_ptr(),
                        positions,
                        &mut most,
                    );
                    weigh(
                        scored.as_ptr(),
                        positions,
                        &most,
                        &vals,
                        dim,
                        &mut acc,
                        &mut sums,
                    );
                }
                for r in 0..rep {
                    let mut top = f32::NEG_INFINITY;
                    for p in 0..positions {
                        let want: f32 = q[r * dim..(r + 1) * dim]
                            .iter()
                            .zip(&keys[p * dim..(p + 1) * dim])
                            .map(|(a, b)| a * b)
                            .sum::<f32>()
                            * scale;
                        let got = scored[r * positions + p];
                        assert!(
                            (got - want).abs() < 1e-4,
                            "{dim} wide, head {r}, position {p}: {got} against {want}"
                        );
                        top = top.max(got);
                    }
                    assert_eq!(most[r].to_bits(), top.to_bits());
                    let mut total = 0.5f64;
                    let mut row = vec![0.25f64; dim];
                    for p in 0..positions {
                        let weight = f64::from(scored[r * positions + p] - most[r]).exp();
                        total += weight;
                        for (o, v) in row.iter_mut().zip(&vals[p * dim..(p + 1) * dim]) {
                            *o += weight * f64::from(*v);
                        }
                    }
                    assert!((f64::from(sums[r]) - total).abs() < 1e-4 * total);
                    for (i, want) in row.iter().enumerate() {
                        let got = f64::from(acc[r * dim + i]);
                        assert!(
                            (got - want).abs() < 1e-4 * positions as f64,
                            "{dim} wide, head {r}, value {i}: {got} against {want}"
                        );
                    }
                }
            }
        }
    }

    /// An adapter's two products in fours are the products a factor at a
    /// time to within the order of a sum, and write nothing beside them:
    /// for one row and for more than are kept together, columns on and off
    /// the running sums' multiple, and ranks on and off sixteen.
    #[test]
    fn an_adapters_products_in_fours_are_the_ones_a_factor_at_a_time() {
        let factors = |seed: u64, n: usize| -> Vec<i8> {
            values(seed, n).iter().map(|v| (v * 127.0) as i8).collect()
        };
        for (n, rows) in [
            (1usize, 1usize),
            (8, 1),
            (13, 3),
            (64, 8),
            (100, 9),
            (256, 19),
        ] {
            let a = factors(3 + n as u64, 16 * n);
            let stride = n + 5;
            let x = values(5 + rows as u64, rows * stride);
            let out_stride = 16 + 3;
            let mut got = vec![9f32; rows * out_stride];
            // SAFETY: every aarch64 CPU has NEON, and the slices hold what
            // the kernel is told of.
            unsafe {
                through_cols(
                    a.as_ptr(),
                    n,
                    x.as_ptr(),
                    stride,
                    rows,
                    got.as_mut_ptr(),
                    out_stride,
                );
            }
            for i in 0..rows {
                for r in 0..16 {
                    let want: f64 = (0..n)
                        .map(|c| f64::from(a[c * 16 + r]) * f64::from(x[i * stride + c]))
                        .sum();
                    let sum = f64::from(got[i * out_stride + r]);
                    assert!(
                        (sum - want).abs() < 1e-2,
                        "{n} columns, row {i}, rank {r}: {sum} against {want}"
                    );
                }
                assert_eq!(&got[i * out_stride + 16..(i + 1) * out_stride], &[9.0; 3]);
            }
        }
        for (rank, sixteens, rows) in [
            (1usize, 1usize, 1usize),
            (16, 3, 1),
            (17, 2, 1),
            (48, 2, 1),
            (16, 2, 3),
            (17, 3, 2),
            (48, 2, 5),
        ] {
            let b = factors(17 + rank as u64, 16 * rank * sixteens);
            let t = values(19 + rows as u64, rows * rank);
            let stride = 16 * sixteens + 2;
            let mut got = vec![0.5f32; rows * stride];
            // SAFETY: as above.
            unsafe {
                add_sixteens(
                    b.as_ptr(),
                    rank,
                    sixteens,
                    t.as_ptr(),
                    rows,
                    got.as_mut_ptr(),
                    stride,
                );
            }
            for i in 0..rows {
                for s in 0..sixteens {
                    for k in 0..16 {
                        let want: f64 = 0.5
                            + (0..rank)
                                .map(|r| {
                                    f64::from(b[(s * rank + r) * 16 + k])
                                        * f64::from(t[i * rank + r])
                                })
                                .sum::<f64>();
                        let product = f64::from(got[i * stride + s * 16 + k]);
                        assert!(
                            (product - want).abs() < 1e-2,
                            "rank {rank}, row {i}, sixteen {s}, product {k}: {product} against {want}"
                        );
                    }
                }
                assert_eq!(
                    &got[i * stride + 16 * sixteens..(i + 1) * stride],
                    &[0.5; 2]
                );
            }
        }
    }
}
