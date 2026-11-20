//! The plain-row arithmetic of [`super`] in 256-bit vectors, for the x86-64
//! CPUs that multiply through tiles with AVX2 and no AVX-512: the MLP's
//! gate, attention's scores and weighted values, and an adapter's factors.
//!
//! Each kernel is its AVX-512 twin with every vector of sixteen taken as two
//! of eight: the same steps on every value, fused the same way, and a sum
//! across a vector's values added in the order the twin adds it. So a row's
//! values are the same values on either CPU, to the bit, and each kernel is
//! held to its twin where a CPU has both.

use super::{EXP_LEAST, EXP_MOST, LN2_HIGH, LN2_LOW, LOG2_E, SPAN, SUMS, TOGETHER};
use std::arch::x86_64::*;

/// `e` to each of eight values: the steps of the twin's exponential, and its
/// values. The power of two is put back in two halves: two to the 128th is
/// no f32, though a value just under one times it is, and both halves of any
/// power here are.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp(x: __m256) -> __m256 {
    // The bound is the first operand, so a value that is not a number comes
    // back as it is.
    let x = _mm256_min_ps(
        _mm256_set1_ps(EXP_MOST),
        _mm256_max_ps(_mm256_set1_ps(EXP_LEAST), x),
    );
    let n = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm256_mul_ps(
        x,
        _mm256_set1_ps(LOG2_E),
    ));
    let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(LN2_HIGH), x);
    let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(LN2_LOW), r);
    let mut p = _mm256_set1_ps(1.987_569_1e-4);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.398_2e-3));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.333_452e-3));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.166_579_6e-2));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.666_666_5e-1));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5));
    let e = _mm256_add_ps(
        _mm256_fmadd_ps(p, _mm256_mul_ps(r, r), r),
        _mm256_set1_ps(1.0),
    );
    let whole = _mm256_cvtps_epi32(n);
    let first = _mm256_srai_epi32::<1>(whole);
    let second = _mm256_sub_epi32(whole, first);
    let bias = _mm256_set1_epi32(127);
    let by_first = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(first, bias)));
    let by_second = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(second, bias)));
    _mm256_mul_ps(_mm256_mul_ps(e, by_first), by_second)
}

/// `silu(gate) · up` for eight values.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn gated(g: __m256, u: __m256) -> __m256 {
    let e = exp(_mm256_sub_ps(_mm256_setzero_ps(), g));
    _mm256_mul_ps(_mm256_div_ps(g, _mm256_add_ps(_mm256_set1_ps(1.0), e)), u)
}

/// `silu(gate) · up` into `out`, over as many values as the three share.
///
/// SAFETY (for the caller): the CPU has AVX2 and FMA.
#[target_feature(enable = "avx2,fma")]
pub(in crate::models::qwen) unsafe fn silu_times(gate: &[f32], up: &[f32], out: &mut [f32]) {
    let n = gate.len().min(up.len()).min(out.len());
    let mut i = 0;
    while i + 8 <= n {
        let v = gated(
            _mm256_loadu_ps(gate.as_ptr().add(i)),
            _mm256_loadu_ps(up.as_ptr().add(i)),
        );
        _mm256_storeu_ps(out.as_mut_ptr().add(i), v);
        i += 8;
    }
    if i < n {
        // The last step takes what is left, beside zeros.
        let (mut g, mut u, mut o) = ([0f32; 8], [0f32; 8], [0f32; 8]);
        g[..n - i].copy_from_slice(&gate[i..n]);
        u[..n - i].copy_from_slice(&up[i..n]);
        _mm256_storeu_ps(
            o.as_mut_ptr(),
            gated(_mm256_loadu_ps(g.as_ptr()), _mm256_loadu_ps(u.as_ptr())),
        );
        out[i..n].copy_from_slice(&o[..n - i]);
    }
}

/// The sum of sixteen values, two eights, added in the order the twin's own
/// reduction adds a vector's: the two eights, then the two fours, then the
/// two pairs, then the two.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn across(low: __m256, high: __m256) -> f32 {
    let eight = _mm256_add_ps(low, high);
    let four = _mm_add_ps(
        _mm256_castps256_ps128(eight),
        _mm256_extractf128_ps::<1>(eight),
    );
    let two = _mm_add_ps(four, _mm_movehl_ps(four, four));
    _mm_cvtss_f32(two) + _mm_cvtss_f32(_mm_shuffle_ps::<0b01>(two, two))
}

#[target_feature(enable = "avx2,fma")]
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
            // The twin's two running sums of sixteen, each as two of eight.
            let (mut a, mut b) = ([_mm256_setzero_ps(); 2], [_mm256_setzero_ps(); 2]);
            for j in (0..N).step_by(2) {
                for h in 0..2 {
                    let at = j * 16 + h * 8;
                    a[h] = _mm256_fmadd_ps(
                        _mm256_loadu_ps(key.add(at)),
                        _mm256_loadu_ps(query.add(at)),
                        a[h],
                    );
                    b[h] = _mm256_fmadd_ps(
                        _mm256_loadu_ps(key.add(at + 16)),
                        _mm256_loadu_ps(query.add(at + 16)),
                        b[h],
                    );
                }
            }
            let score = across(_mm256_add_ps(a[0], b[0]), _mm256_add_ps(a[1], b[1])) * scale;
            *out.add(r * stride + p) = score;
            if score > *most.get_unchecked(r) {
                *most.get_unchecked_mut(r) = score;
            }
        }
    }
}

/// The scores of the query heads in `q` against each position of `keys`, as
/// [`super::scores`] writes them.
///
/// SAFETY (for the caller): the CPU has AVX2 and FMA, [`super::attends`]
/// holds of `dim`, `most` has a value for each head in `q`, and `out` has
/// room at every place written with no other thread writing there.
#[target_feature(enable = "avx2,fma")]
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

/// Vectors of eight whose running sums are kept at once while a span's
/// positions go by: a head of 128 values is two such runs.
const KEPT: usize = 8;

#[target_feature(enable = "avx2,fma")]
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
        let top = _mm256_set1_ps(*top);
        let row = acc.as_mut_ptr().add(r * dim);
        let mut total = 0f32;
        let mut at = 0;
        while at < positions {
            let n = SPAN.min(positions - at);
            let from = scores.add(r * stride + at);
            let mut i = 0;
            while i + 8 <= n {
                _mm256_storeu_ps(
                    weights.as_mut_ptr().add(i),
                    exp(_mm256_sub_ps(_mm256_loadu_ps(from.add(i)), top)),
                );
                i += 8;
            }
            if i < n {
                // The last step takes what is left, beside zeros.
                let (mut some, mut done) = ([0f32; 8], [0f32; 8]);
                std::ptr::copy_nonoverlapping(from.add(i), some.as_mut_ptr(), n - i);
                _mm256_storeu_ps(
                    done.as_mut_ptr(),
                    exp(_mm256_sub_ps(_mm256_loadu_ps(some.as_ptr()), top)),
                );
                weights[i..n].copy_from_slice(&done[..n - i]);
            }
            for weight in &weights[..n] {
                total += weight;
            }
            // A value's sum takes the positions in their order whatever
            // values are summed beside it, so the head goes by in runs of
            // as many vectors as stay in registers.
            let mut first = 0;
            while first < 2 * N {
                let kept = (2 * N - first).min(KEPT);
                let mut sum = [_mm256_setzero_ps(); KEPT];
                for (k, s) in sum[..kept].iter_mut().enumerate() {
                    *s = _mm256_loadu_ps(row.add((first + k) * 8));
                }
                for (p, weight) in weights[..n].iter().enumerate() {
                    let by = _mm256_set1_ps(*weight);
                    let value = values.as_ptr().add((at + p) * dim + first * 8);
                    for (k, s) in sum[..kept].iter_mut().enumerate() {
                        *s = _mm256_fmadd_ps(by, _mm256_loadu_ps(value.add(k * 8)), *s);
                    }
                }
                for (k, s) in sum[..kept].iter().enumerate() {
                    _mm256_storeu_ps(row.add((first + k) * 8), *s);
                }
                first += kept;
            }
            at += n;
        }
        *sums.get_unchecked_mut(r) += total;
    }
}

/// Each position of `values` weighted by `e` to its score less its head's
/// largest and added into `acc`, as [`super::weigh`] adds them.
///
/// SAFETY (for the caller): the CPU has AVX2 and FMA, [`super::attends`]
/// holds of `dim`, `acc` has `dim` values and `sums` one for each head, and
/// `scores` holds a score for every head and position at `r * stride + p`.
#[target_feature(enable = "avx2,fma")]
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
#[target_feature(enable = "avx2")]
unsafe fn eights(a: *const i8) -> __m256 {
    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(a as *const __m128i)))
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn summed(sums: &[__m256; SUMS]) -> __m256 {
    _mm256_add_ps(
        _mm256_add_ps(
            _mm256_add_ps(sums[0], sums[1]),
            _mm256_add_ps(sums[2], sums[3]),
        ),
        _mm256_add_ps(
            _mm256_add_ps(sums[4], sums[5]),
            _mm256_add_ps(sums[6], sums[7]),
        ),
    )
}

/// Sixteen rank rows of an adapter's `A` over `n` columns, times those
/// columns of each of `rows` rows of input, as [`super::through_cols`]
/// writes them: the sixteen sums a row are two eights here, each gone
/// through on its own, a column still added to running sum `c % 8` in the
/// order of the columns.
///
/// SAFETY (for the caller): the CPU has AVX2 and FMA; `a` holds `16 * n`
/// factors; each row of `x` holds `n` values; and each place in `out` is
/// room for sixteen.
#[target_feature(enable = "avx2,fma")]
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
            let mut sums = [[_mm256_setzero_ps(); SUMS]; TOGETHER];
            let mut c = 0;
            while c + SUMS <= n {
                let mut cols = [_mm256_setzero_ps(); SUMS];
                for (j, col) in cols.iter_mut().enumerate() {
                    *col = eights(a.add((c + j) * 16));
                }
                for (i, sums) in sums[..some].iter_mut().enumerate() {
                    let x = x.add(i * stride + c);
                    for (j, sum) in sums.iter_mut().enumerate() {
                        *sum = _mm256_fmadd_ps(cols[j], _mm256_set1_ps(*x.add(j)), *sum);
                    }
                }
                c += SUMS;
            }
            for j in 0..n - c {
                let col = eights(a.add((c + j) * 16));
                for (i, sums) in sums[..some].iter_mut().enumerate() {
                    sums[j] =
                        _mm256_fmadd_ps(col, _mm256_set1_ps(*x.add(i * stride + c + j)), sums[j]);
                }
            }
            for (i, sums) in sums[..some].iter().enumerate() {
                _mm256_storeu_ps(out.add((first + i) * out_stride + half * 8), summed(sums));
            }
            first += some;
        }
    }
}

/// To `sixteens` sixteens of products of each of `rows` rows, an adapter's
/// `B` times that row's `A · x`, as [`super::add_sixteens`] adds them: a
/// sixteen is two eights here, the ranks still added in their order into
/// two sums, the even and the odd, and the two to the products.
///
/// SAFETY (for the caller): the CPU has AVX2 and FMA; `b` holds
/// `16 * rank * sixteens` factors; `t` holds `rank` values for every row; and
/// each row's place in `out` holds `16 * sixteens` values that nothing else
/// reads or writes meanwhile.
#[target_feature(enable = "avx2,fma")]
pub(in crate::models::qwen) unsafe fn add_sixteens(
    b: *const i8,
    rank: usize,
    sixteens: usize,
    t: *const f32,
    rows: usize,
    out: *mut f32,
    stride: usize,
) {
    for s in 0..sixteens {
        let b = b.add(s * rank * 16);
        let mut first = 0;
        while first < rank {
            let some = (rank - first).min(16);
            for half in 0..2 {
                let mut factors = [_mm256_setzero_ps(); 16];
                for (r, factor) in factors[..some].iter_mut().enumerate() {
                    *factor = eights(b.add((first + r) * 16 + half * 8));
                }
                for i in 0..rows {
                    let t = t.add(i * rank + first);
                    let (mut even, mut odd) = (_mm256_setzero_ps(), _mm256_setzero_ps());
                    let mut r = 0;
                    while r + 2 <= some {
                        even = _mm256_fmadd_ps(factors[r], _mm256_set1_ps(*t.add(r)), even);
                        odd = _mm256_fmadd_ps(factors[r + 1], _mm256_set1_ps(*t.add(r + 1)), odd);
                        r += 2;
                    }
                    if r < some {
                        even = _mm256_fmadd_ps(factors[r], _mm256_set1_ps(*t.add(r)), even);
                    }
                    let at = out.add(i * stride + s * 16 + half * 8);
                    _mm256_storeu_ps(
                        at,
                        _mm256_add_ps(_mm256_loadu_ps(at), _mm256_add_ps(even, odd)),
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

    fn here() -> bool {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }

    fn twin() -> bool {
        is_x86_feature_detected!("avx512f")
    }

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

    fn same(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    /// The gate in eights is the gate a value at a time to within its
    /// rounding, for widths on and off the vectors' multiple, writes nothing
    /// past the end, and is the gate in sixteens to the bit where a CPU has
    /// both: across the exponential's whole range, and at its edges.
    #[test]
    fn the_gate_in_eights_is_the_gate_in_sixteens() {
        if !here() {
            eprintln!("skipped: this CPU has no AVX2");
            return;
        }
        for n in [0usize, 1, 7, 8, 9, 15, 16, 17, 100, 560] {
            let gate: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin() * 9.0).collect();
            let up: Vec<f32> = (0..n).map(|i| (i as f32 * 0.11).cos() * 3.0).collect();
            let mut got = vec![7f32; n + 3];
            // SAFETY: the test returned above without AVX2 and FMA.
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
        if !twin() {
            return;
        }
        // Every exponent the exponential takes, and what lies past them.
        let mut gate: Vec<f32> = (0..6000).map(|i| -90.0 + i as f32 * 0.030_3).collect();
        gate.extend([
            -1000.0,
            1000.0,
            f32::NAN,
            0.0,
            -0.0,
            f32::NEG_INFINITY,
            f32::INFINITY,
            88.7,
            -88.7,
            87.3,
            -87.3,
            1e-30,
            -1e-30,
        ]);
        let up = vec![1.0f32; gate.len()];
        let (mut eight, mut sixteen) = (vec![0f32; gate.len()], vec![0f32; gate.len()]);
        // SAFETY: both sets of features were found above.
        unsafe {
            silu_times(&gate, &up, &mut eight);
            super::super::silu_times(&gate, &up, &mut sixteen);
        }
        for (i, (a, b)) in eight.iter().zip(&sixteen).enumerate() {
            assert!(
                a.to_bits() == b.to_bits(),
                "the gate of {}: {a} in eights, {b} in sixteens",
                gate[i]
            );
        }
    }

    /// Attention's two passes in eights are the passes a value at a time to
    /// within the order of a sum, and the passes in sixteens to the bit
    /// where a CPU has both: for every width of head, with one query head
    /// and with several, and for runs of positions on and off the vectors'
    /// multiple and longer than a span.
    #[test]
    fn attention_in_eights_is_attention_in_sixteens() {
        if !here() {
            return;
        }
        for dim in [32usize, 64, 128] {
            for (rep, positions) in [(1usize, 1usize), (6, 7), (2, 16), (6, 129), (3, 300)] {
                let q = values(dim as u64 + rep as u64, rep * dim);
                let keys = values(7 + positions as u64, positions * dim);
                let vals = values(11 + dim as u64, positions * dim);
                let scale = 1.0 / (dim as f32).sqrt();
                let run = |eights: bool| {
                    let mut scored = vec![0f32; rep * positions];
                    let mut most = vec![f32::NEG_INFINITY; rep];
                    let mut acc = vec![0.25f32; rep * dim];
                    let mut sums = vec![0.5f32; rep];
                    // SAFETY: the features of whichever kernel runs were
                    // found before it was asked for, and every place has
                    // the room the kernels are told of.
                    unsafe {
                        if eights {
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
                        } else {
                            super::super::scores(
                                &q,
                                &keys,
                                dim,
                                scale,
                                scored.as_mut_ptr(),
                                positions,
                                &mut most,
                            );
                            super::super::weigh(
                                scored.as_ptr(),
                                positions,
                                &most,
                                &vals,
                                dim,
                                &mut acc,
                                &mut sums,
                            );
                        }
                    }
                    (scored, most, acc, sums)
                };
                let (scored, most, acc, sums) = run(true);
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
                if twin() {
                    let (scored16, most16, acc16, sums16) = run(false);
                    assert!(
                        same(&scored, &scored16),
                        "{dim} wide, {rep} heads, {positions} positions: scores"
                    );
                    assert!(same(&most, &most16) && same(&sums, &sums16));
                    assert!(
                        same(&acc, &acc16),
                        "{dim} wide, {rep} heads, {positions} positions: weighted values"
                    );
                }
            }
        }
    }

    /// An adapter's two products in eights are the products a factor at a
    /// time to within the order of a sum, and the products in sixteens to
    /// the bit where a CPU has both: for one row and for more than are kept
    /// together, columns on and off the running sums' multiple, and ranks
    /// on and off sixteen.
    #[test]
    fn an_adapters_products_in_eights_are_the_ones_in_sixteens() {
        if !here() {
            return;
        }
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
            let run = |eights: bool| {
                let mut out = vec![9f32; rows * out_stride];
                // SAFETY: the features of whichever kernel runs were found
                // before it was asked for, and the slices hold what the
                // kernels are told of.
                unsafe {
                    if eights {
                        through_cols(
                            a.as_ptr(),
                            n,
                            x.as_ptr(),
                            stride,
                            rows,
                            out.as_mut_ptr(),
                            out_stride,
                        );
                    } else {
                        super::super::through_cols(
                            a.as_ptr(),
                            n,
                            x.as_ptr(),
                            stride,
                            rows,
                            out.as_mut_ptr(),
                            out_stride,
                        );
                    }
                }
                out
            };
            let got = run(true);
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
            if twin() {
                assert!(same(&got, &run(false)), "{n} columns, {rows} rows");
            }
        }
        for (rank, sixteens, rows) in [(1usize, 1usize, 1usize), (16, 2, 3), (17, 3, 2), (48, 2, 5)]
        {
            let b = factors(17 + rank as u64, 16 * rank * sixteens);
            let t = values(19 + rows as u64, rows * rank);
            let stride = 16 * sixteens + 2;
            let run = |eights: bool| {
                let mut out = vec![0.5f32; rows * stride];
                // SAFETY: as above.
                unsafe {
                    if eights {
                        add_sixteens(
                            b.as_ptr(),
                            rank,
                            sixteens,
                            t.as_ptr(),
                            rows,
                            out.as_mut_ptr(),
                            stride,
                        );
                    } else {
                        super::super::add_sixteens(
                            b.as_ptr(),
                            rank,
                            sixteens,
                            t.as_ptr(),
                            rows,
                            out.as_mut_ptr(),
                            stride,
                        );
                    }
                }
                out
            };
            let got = run(true);
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
            if twin() {
                assert!(
                    same(&got, &run(false)),
                    "rank {rank}, {sixteens} sixteens, {rows} rows"
                );
            }
        }
    }
}
