//! The arithmetic on plain f32 rows in 512-bit vectors, for the CPUs that
//! multiply through tiles ([`super::tiles::available`]).
//!
//! Between a token's products there is the MLP's gate, an exponential for
//! every value of its width, and attention, a score against every cached
//! position and a weighted sum of the cached values. A value at a time, or
//! eight at a time without a vector exponential, they are most of what one
//! thread does while the others wait for it. An adapter's factors, held at
//! eight bits, are multiplied here as well. Each kernel here has a plain
//! twin it is held to.
//!
//! A CPU that multiplies through tiles with AVX2 alone takes the same steps
//! in 256-bit vectors (`wide_avx2.rs`), and an aarch64 CPU in 128-bit ones
//! (`wide_neon.rs`): each kernel there is its twin here with every vector of
//! sixteen as two of eight, or four of four, and gives its values to the
//! bit. So a row's values do not depend on which of the three a CPU has
//! (`marks`, below, holds each to one number).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[path = "wide_avx2.rs"]
pub(super) mod avx2;

#[cfg(target_arch = "aarch64")]
#[path = "wide_neon.rs"]
pub(super) mod neon;

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
#[cfg(target_arch = "x86_64")]
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
#[cfg(target_arch = "x86_64")]
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

#[cfg(target_arch = "x86_64")]
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
#[cfg(target_arch = "x86_64")]
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

#[cfg(target_arch = "x86_64")]
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
#[cfg(target_arch = "x86_64")]
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

/// Sixteen factors at `a`, whole numbers of eight bits, as floats.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn eights(a: *const i8) -> __m512 {
    _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(a as *const __m128i)))
}

/// Running sums a row's products with `A` are kept in, so one step does not
/// wait on the step before it.
const SUMS: usize = 8;

/// Rows of input whose running sums are kept at once.
const TOGETHER: usize = 8;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn summed(sums: &[__m512; SUMS]) -> __m512 {
    _mm512_add_ps(
        _mm512_add_ps(
            _mm512_add_ps(sums[0], sums[1]),
            _mm512_add_ps(sums[2], sums[3]),
        ),
        _mm512_add_ps(
            _mm512_add_ps(sums[4], sums[5]),
            _mm512_add_ps(sums[6], sums[7]),
        ),
    )
}

/// Sixteen rank rows of an adapter's `A` over `n` columns, times those
/// columns of each of `rows` rows of input. `a` holds a column at a time,
/// the sixteen rows' values of it at eight bits; row `i`'s values of the columns
/// are at `x + i * stride`, and its sixteen sums are written at
/// `out + i * out_stride`.
///
/// A column's vector times one value of the input adds to all sixteen sums
/// at once, so nothing is ever summed across a vector's lanes; and a column
/// is made floats once for every row there is. Each row's sums are the sums
/// it would have alone, to the bit: column `c` is added to running sum
/// `c % 8` in the order of the columns, whatever rows are beside it.
///
/// SAFETY (for the caller): the CPU has AVX-512F; `a` holds `16 * n`
/// factors; each row of `x` holds `n` values; and each place in `out` is
/// room for sixteen.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn through_cols(
    a: *const i8,
    n: usize,
    x: *const f32,
    stride: usize,
    rows: usize,
    out: *mut f32,
    out_stride: usize,
) {
    if rows == 1 {
        // One row keeps its sums in registers.
        let mut sums = [_mm512_setzero_ps(); SUMS];
        let mut c = 0;
        while c + SUMS <= n {
            for (j, sum) in sums.iter_mut().enumerate() {
                *sum = _mm512_fmadd_ps(
                    eights(a.add((c + j) * 16)),
                    _mm512_set1_ps(*x.add(c + j)),
                    *sum,
                );
            }
            c += SUMS;
        }
        for (j, sum) in sums[..n - c].iter_mut().enumerate() {
            *sum = _mm512_fmadd_ps(
                eights(a.add((c + j) * 16)),
                _mm512_set1_ps(*x.add(c + j)),
                *sum,
            );
        }
        _mm512_storeu_ps(out, summed(&sums));
        return;
    }
    let mut first = 0;
    while first < rows {
        let some = (rows - first).min(TOGETHER);
        let x = x.add(first * stride);
        let mut sums = [[_mm512_setzero_ps(); SUMS]; TOGETHER];
        let mut c = 0;
        while c + SUMS <= n {
            let mut cols = [_mm512_setzero_ps(); SUMS];
            for (j, col) in cols.iter_mut().enumerate() {
                *col = eights(a.add((c + j) * 16));
            }
            for (i, sums) in sums[..some].iter_mut().enumerate() {
                let x = x.add(i * stride + c);
                for (j, sum) in sums.iter_mut().enumerate() {
                    *sum = _mm512_fmadd_ps(cols[j], _mm512_set1_ps(*x.add(j)), *sum);
                }
            }
            c += SUMS;
        }
        for j in 0..n - c {
            let col = eights(a.add((c + j) * 16));
            for (i, sums) in sums[..some].iter_mut().enumerate() {
                sums[j] = _mm512_fmadd_ps(col, _mm512_set1_ps(*x.add(i * stride + c + j)), sums[j]);
            }
        }
        for (i, sums) in sums[..some].iter().enumerate() {
            _mm512_storeu_ps(out.add((first + i) * out_stride), summed(sums));
        }
        first += some;
    }
}

/// To `sixteens` sixteens of products of each of `rows` rows, an adapter's
/// `B` times that row's `A · x`. `b` holds a sixteen of products at a time,
/// for each of the `rank` ranks the sixteen's factors at eight bits; row `i`'s
/// `rank` values of `A · x` are at `t + i * rank`, and its products at
/// `out + i * stride`.
///
/// A rank's vector times that rank's value adds to all sixteen products at
/// once, and a sixteen's factors are made floats once for every row there
/// is. Each row's products are what it would get alone, to the bit: the
/// ranks are added in their order into two sums, the even and the odd, and
/// the two to the products, sixteen ranks at a time.
///
/// SAFETY (for the caller): the CPU has AVX-512F; `b` holds
/// `16 * rank * sixteens` factors; `t` holds `rank` values for every row; and
/// each row's place in `out` holds `16 * sixteens` values that nothing else
/// reads or writes meanwhile.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn add_sixteens(
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
            let mut factors = [_mm512_setzero_ps(); 16];
            for (r, factor) in factors[..some].iter_mut().enumerate() {
                *factor = eights(b.add((first + r) * 16));
            }
            for i in 0..rows {
                let t = t.add(i * rank + first);
                let (mut even, mut odd) = (_mm512_setzero_ps(), _mm512_setzero_ps());
                let mut r = 0;
                while r + 2 <= some {
                    even = _mm512_fmadd_ps(factors[r], _mm512_set1_ps(*t.add(r)), even);
                    odd = _mm512_fmadd_ps(factors[r + 1], _mm512_set1_ps(*t.add(r + 1)), odd);
                    r += 2;
                }
                if r < some {
                    even = _mm512_fmadd_ps(factors[r], _mm512_set1_ps(*t.add(r)), even);
                }
                let at = out.add(i * stride + s * 16);
                _mm512_storeu_ps(
                    at,
                    _mm512_add_ps(_mm512_loadu_ps(at), _mm512_add_ps(even, odd)),
                );
            }
            first += some;
        }
    }
}

/// What every set of these kernels is held to, since no CPU runs all three:
/// a kernel's values over inputs drawn from a seed, folded into one number,
/// which each set a CPU has must give. The numbers are the AVX-512 kernels'.
/// A set that gave another value anywhere, by a bit, would give another
/// number.
#[cfg(test)]
mod marks {
    /// One set of the kernels.
    struct Set {
        name: &'static str,
        silu_times: unsafe fn(&[f32], &[f32], &mut [f32]),
        scores: unsafe fn(&[f32], &[f32], usize, f32, *mut f32, usize, &mut [f32]),
        weigh: unsafe fn(*const f32, usize, &[f32], &[f32], usize, &mut [f32], &mut [f32]),
        through_cols: unsafe fn(*const i8, usize, *const f32, usize, usize, *mut f32, usize),
        add_sixteens: unsafe fn(*const i8, usize, usize, *const f32, usize, *mut f32, usize),
    }

    /// Every set this CPU can run.
    fn sets() -> Vec<Set> {
        let mut sets = Vec::new();
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                sets.push(Set {
                    name: "AVX-512",
                    silu_times: super::silu_times,
                    scores: super::scores,
                    weigh: super::weigh,
                    through_cols: super::through_cols,
                    add_sixteens: super::add_sixteens,
                });
            }
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                sets.push(Set {
                    name: "AVX2",
                    silu_times: super::avx2::silu_times,
                    scores: super::avx2::scores,
                    weigh: super::avx2::weigh,
                    through_cols: super::avx2::through_cols,
                    add_sixteens: super::avx2::add_sixteens,
                });
            }
        }
        #[cfg(target_arch = "aarch64")]
        sets.push(Set {
            name: "NEON",
            silu_times: super::neon::silu_times,
            scores: super::neon::scores,
            weigh: super::neon::weigh,
            through_cols: super::neon::through_cols,
            add_sixteens: super::neon::add_sixteens,
        });
        sets
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

    fn factors(seed: u64, n: usize) -> Vec<i8> {
        values(seed, n).iter().map(|v| (v * 127.0) as i8).collect()
    }

    const START: u64 = 0xcbf2_9ce4_8422_2325;

    /// `mark` with the bits of `values` folded in. What is not a number is
    /// folded in as one value, since which of them a CPU makes is its own.
    fn folded(mark: u64, values: &[f32]) -> u64 {
        values
            .iter()
            .map(|v| match v.is_nan() {
                true => f32::NAN.to_bits(),
                false => v.to_bits(),
            })
            .flat_map(u32::to_le_bytes)
            .fold(mark, |mark, byte| {
                (mark ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
            })
    }

    /// The gate across the exponential's whole range and past both ends of
    /// it, over a length that leaves every set a last step short of a
    /// vector.
    #[test]
    fn every_set_gives_the_gate_one_value() {
        let mut gate: Vec<f32> = (0..6000).map(|i| -90.0 + i as f32 * 0.030_3).collect();
        gate.extend([
            -1000.0,
            1000.0,
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
            0.5,
        ]);
        let up: Vec<f32> = values(29, gate.len()).iter().map(|v| v * 3.0).collect();
        for set in sets() {
            let mut out = vec![0f32; gate.len()];
            // SAFETY: `sets` names only the kernels this CPU has.
            unsafe { (set.silu_times)(&gate, &up, &mut out) };
            assert_eq!(folded(START, &out), GATE, "the gate through {}", set.name);
        }
    }

    /// Attention's two passes: every width of head, one query head and
    /// several, and runs of positions on and off the vectors' multiple and
    /// longer than a span.
    #[test]
    fn every_set_gives_attention_one_value() {
        for set in sets() {
            let mut mark = START;
            for dim in [32usize, 64, 128] {
                for (rep, positions) in [(1usize, 1usize), (6, 7), (2, 16), (6, 129), (3, 300)] {
                    let q = values(dim as u64 + rep as u64, rep * dim);
                    let keys = values(7 + positions as u64, positions * dim);
                    let vals = values(11 + dim as u64, positions * dim);
                    let scale = 1.0 / (dim as f32).sqrt();
                    let mut scored = vec![0f32; rep * positions];
                    let mut most = vec![f32::NEG_INFINITY; rep];
                    let mut acc = vec![0.25f32; rep * dim];
                    let mut sums = vec![0.5f32; rep];
                    // SAFETY: `sets` names only the kernels this CPU has,
                    // and every place has the room the kernels are told of.
                    unsafe {
                        (set.scores)(
                            &q,
                            &keys,
                            dim,
                            scale,
                            scored.as_mut_ptr(),
                            positions,
                            &mut most,
                        );
                        (set.weigh)(
                            scored.as_ptr(),
                            positions,
                            &most,
                            &vals,
                            dim,
                            &mut acc,
                            &mut sums,
                        );
                    }
                    for part in [&scored, &most, &acc, &sums] {
                        mark = folded(mark, part);
                    }
                }
            }
            assert_eq!(mark, ATTENTION, "attention through {}", set.name);
        }
    }

    /// An adapter's two products: one row and more than are kept together,
    /// columns on and off the running sums' multiple, and ranks on and off
    /// sixteen.
    #[test]
    fn every_set_gives_an_adapters_products_one_value() {
        for set in sets() {
            let mut mark = START;
            for (n, rows) in [
                (1usize, 1usize),
                (8, 1),
                (1536, 1),
                (13, 3),
                (64, 8),
                (100, 9),
                (256, 19),
            ] {
                let a = factors(3 + n as u64, 16 * n);
                let stride = n + 5;
                let x = values(5 + rows as u64, rows * stride);
                let out_stride = 16 + 3;
                let mut out = vec![9f32; rows * out_stride];
                // SAFETY: `sets` names only the kernels this CPU has, and
                // the slices hold what the kernel is told of.
                unsafe {
                    (set.through_cols)(
                        a.as_ptr(),
                        n,
                        x.as_ptr(),
                        stride,
                        rows,
                        out.as_mut_ptr(),
                        out_stride,
                    );
                }
                mark = folded(mark, &out);
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
                let mut out = vec![0.5f32; rows * stride];
                // SAFETY: as above.
                unsafe {
                    (set.add_sixteens)(
                        b.as_ptr(),
                        rank,
                        sixteens,
                        t.as_ptr(),
                        rows,
                        out.as_mut_ptr(),
                        stride,
                    );
                }
                mark = folded(mark, &out);
            }
            assert_eq!(mark, ADAPTER, "an adapter's products through {}", set.name);
        }
    }

    const GATE: u64 = 0x2e40_9c27_14f7_debb;
    const ATTENTION: u64 = 0x6702_f0c5_3153_87da;
    const ADAPTER: u64 = 0x03bd_9e20_c782_41c6;
}

#[cfg(all(test, target_arch = "x86_64"))]
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
