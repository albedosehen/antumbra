//! A decoded token's arithmetic on plain f32 rows in 512-bit vectors, for the
//! CPUs that multiply through tiles ([`super::tiles::available`]).
//!
//! Between a token's products there is the MLP's gate, an exponential for
//! every value of its width, and attention, a score against every cached
//! position and a weighted sum of the cached values. A value at a time, or
//! eight at a time without a vector exponential, they are most of what one
//! thread does while the others wait for it. Each kernel here has a plain
//! twin it is held to.

use std::arch::x86_64::*;

const LOG2_E: f32 = std::f32::consts::LOG2_E;
/// ln 2 in two parts, the first exact in few bits, so `x - n · ln 2` loses
/// nothing for the `n` an f32 exponent can be.
const LN2_HIGH: f32 = 0.693_359_4;
const LN2_LOW: f32 = -2.121_944_4e-4;
/// e^x overflows an f32 past the first and is nothing before the second.
const EXP_MOST: f32 = 88.7;
const EXP_LEAST: f32 = -87.3;

/// `e` to each of sixteen values, to within two units in the last place: the
/// power of two nearest is taken off, what is left is a polynomial's, and the
/// power is put back (Cephes' `expf`, sixteen at a time).
#[target_feature(enable = "avx512f")]
unsafe fn exp(x: __m512) -> __m512 {
    // The bound is the first operand, so a value that is not a number comes
    // back as it is.
    let x = _mm512_min_ps(
        _mm512_set1_ps(EXP_MOST),
        _mm512_max_ps(_mm512_set1_ps(EXP_LEAST), x),
    );
    let n = _mm512_roundscale_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
        _mm512_mul_ps(x, _mm512_set1_ps(LOG2_E)),
    );
    let r = _mm512_fnmadd_ps(n, _mm512_set1_ps(LN2_HIGH), x);
    let r = _mm512_fnmadd_ps(n, _mm512_set1_ps(LN2_LOW), r);
    let mut p = _mm512_set1_ps(1.987_569_1e-4);
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.398_2e-3));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(8.333_452e-3));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(4.166_579_6e-2));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.666_666_5e-1));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(0.5));
    let e = _mm512_add_ps(
        _mm512_fmadd_ps(p, _mm512_mul_ps(r, r), r),
        _mm512_set1_ps(1.0),
    );
    _mm512_scalef_ps(e, n)
}

/// `silu(gate) · up` into `out`, over as many values as the three share.
///
/// SAFETY (for the caller): the CPU has AVX-512F.
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn silu_times(gate: &[f32], up: &[f32], out: &mut [f32]) {
    let n = gate.len().min(up.len()).min(out.len());
    let one = _mm512_set1_ps(1.0);
    let mut i = 0;
    while i < n {
        // The last step takes what is left, under a mask.
        let k: __mmask16 = if n - i >= 16 {
            !0
        } else {
            (1u16 << (n - i)) - 1
        };
        let g = _mm512_maskz_loadu_ps(k, gate.as_ptr().add(i));
        let u = _mm512_maskz_loadu_ps(k, up.as_ptr().add(i));
        let e = exp(_mm512_sub_ps(_mm512_setzero_ps(), g));
        let gated = _mm512_mul_ps(_mm512_div_ps(g, _mm512_add_ps(one, e)), u);
        _mm512_mask_storeu_ps(out.as_mut_ptr().add(i), k, gated);
        i += 16;
    }
}

/// Whether [`scores`] and [`weigh`] take heads of `dim` values.
pub(super) fn attends(dim: usize) -> bool {
    matches!(dim, 32 | 64 | 128)
}

#[target_feature(enable = "avx512f")]
unsafe fn scores_of<const N: usize>(
    q: &[f32],
    keys: &[f32],
    scale: f32,
    out: *mut f32,
    stride: usize,
    most: &mut [f32],
) {
    let dim = N * 16;
    let rep = q.len() / dim;
    for p in 0..keys.len() / dim {
        let mut key = [_mm512_setzero_ps(); N];
        for (j, k) in key.iter_mut().enumerate() {
            *k = _mm512_loadu_ps(keys.as_ptr().add(p * dim + j * 16));
        }
        for r in 0..rep {
            let query = q.as_ptr().add(r * dim);
            // Two running sums, so one step's wait is not the next one's.
            let (mut a, mut b) = (_mm512_setzero_ps(), _mm512_setzero_ps());
            for j in (0..N).step_by(2) {
                a = _mm512_fmadd_ps(key[j], _mm512_loadu_ps(query.add(j * 16)), a);
                b = _mm512_fmadd_ps(key[j + 1], _mm512_loadu_ps(query.add(j * 16 + 16)), b);
            }
            let score = _mm512_reduce_add_ps(_mm512_add_ps(a, b)) * scale;
            *out.add(r * stride + p) = score;
            if score > *most.get_unchecked(r) {
                *most.get_unchecked_mut(r) = score;
            }
        }
    }
}

/// The scores of the query heads in `q` (`dim` values each, one after
/// another) against each position of `keys` (`dim` values each): head `r`'s
/// against position `p`, times `scale`, is written to `out[r * stride + p]`,
/// and `most[r]` is raised to the largest of them.
///
/// SAFETY (for the caller): the CPU has AVX-512F, [`attends`] holds of
/// `dim`, `most` has a value for each head in `q`, and `out` has room at
/// every place named above with no other thread writing there.
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn scores(
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

/// Positions whose weights are worked out at a time.
const SPAN: usize = 128;

#[target_feature(enable = "avx512f")]
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
        let top = _mm512_set1_ps(*top);
        let row = acc.as_mut_ptr().add(r * dim);
        let mut sum = [_mm512_setzero_ps(); N];
        for (j, s) in sum.iter_mut().enumerate() {
            *s = _mm512_loadu_ps(row.add(j * 16));
        }
        let mut total = 0f32;
        let mut at = 0;
        while at < positions {
            let n = SPAN.min(positions - at);
            let mut i = 0;
            while i < n {
                let k: __mmask16 = if n - i >= 16 {
                    !0
                } else {
                    (1u16 << (n - i)) - 1
                };
                let score = _mm512_maskz_loadu_ps(k, scores.add(r * stride + at + i));
                _mm512_mask_storeu_ps(
                    weights.as_mut_ptr().add(i),
                    k,
                    exp(_mm512_sub_ps(score, top)),
                );
                i += 16;
            }
            for (p, weight) in weights[..n].iter().enumerate() {
                total += weight;
                let by = _mm512_set1_ps(*weight);
                let value = values.as_ptr().add((at + p) * dim);
                for (j, s) in sum.iter_mut().enumerate() {
                    *s = _mm512_fmadd_ps(by, _mm512_loadu_ps(value.add(j * 16)), *s);
                }
            }
            at += n;
        }
        for (j, s) in sum.iter().enumerate() {
            _mm512_storeu_ps(row.add(j * 16), *s);
        }
        *sums.get_unchecked_mut(r) += total;
    }
}

/// For each head `r` of `most.len()`: each position of `values` (`dim`
/// values each) weighted by `e` to its score less `most[r]`, added into
/// `acc[r * dim..][..dim]`, and the weights' sum added to `sums[r]`. The
/// scores are read where [`scores`] wrote them.
///
/// SAFETY (for the caller): the CPU has AVX-512F, [`attends`] holds of
/// `dim`, `acc` has `dim` values and `sums` one for each head, and `scores`
/// holds a score for every head and position at `r * stride + p`.
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn weigh(
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

#[cfg(test)]
mod tests {
    use super::*;

    fn wide() -> bool {
        is_x86_feature_detected!("avx512f")
    }

    /// The vector exponential is the scalar one to within two units in the
    /// last place across its whole range, is nothing far below it and
    /// endless far above, and leaves what is not a number as it is.
    #[test]
    fn the_exponential_in_vectors_is_the_scalar_one() {
        if !wide() {
            eprintln!("skipped: this CPU has no AVX-512");
            return;
        }
        let mut worst = 0f64;
        let mut x = -87.0f32;
        while x < 88.5 {
            let xs: [f32; 16] = std::array::from_fn(|i| x + i as f32 * 0.003_7);
            let mut got = [0f32; 16];
            // SAFETY: the test returned above without AVX-512.
            unsafe { _mm512_storeu_ps(got.as_mut_ptr(), exp(_mm512_loadu_ps(xs.as_ptr()))) };
            for (g, v) in got.iter().zip(xs) {
                let want = f64::from(v).exp();
                worst = worst.max(((f64::from(*g) - want) / want).abs());
            }
            x += 0.061;
        }
        assert!(worst < 2.5e-7, "off by {worst} of the value");
        let edge = [
            -1000.0f32,
            1000.0,
            f32::NAN,
            0.0,
            -0.0,
            f32::NEG_INFINITY,
            f32::INFINITY,
            1.0,
        ];
        let mut xs = [0f32; 16];
        xs[..8].copy_from_slice(&edge);
        let mut got = [0f32; 16];
        // SAFETY: as above.
        unsafe { _mm512_storeu_ps(got.as_mut_ptr(), exp(_mm512_loadu_ps(xs.as_ptr()))) };
        assert!(got[0] < 1e-37 && got[5] < 1e-37, "far below is nothing");
        assert!(
            got[1] > 1e38 && got[6] > 1e38,
            "far above is past any value"
        );
        assert!(got[2].is_nan());
        assert_eq!((got[3], got[4]), (1.0, 1.0));
        assert!((got[7] - std::f32::consts::E).abs() < 3e-7);
    }

    /// The gate in vectors is the gate a value at a time, for widths on and
    /// off the vectors' multiple.
    #[test]
    fn the_gate_in_vectors_is_the_gate_a_value_at_a_time() {
        if !wide() {
            return;
        }
        for n in [0usize, 1, 15, 16, 17, 100, 560] {
            let gate: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin() * 9.0).collect();
            let up: Vec<f32> = (0..n).map(|i| (i as f32 * 0.11).cos() * 3.0).collect();
            let mut got = vec![7f32; n + 3];
            // SAFETY: the test returned above without AVX-512.
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
}
