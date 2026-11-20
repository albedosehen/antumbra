//! The tile kernels in AVX2, for the x86-64 CPUs without AVX-512 VNNI:
//! Intel's from Haswell, its desktop and laptop CPUs since Alder Lake, and
//! AMD's before Zen 4. With FMA and F16C, which every one of them has.
//!
//! A tile's sixteen rows are two vectors of eight here, each 32-bit lane a
//! row, over the tiles the AVX-512 kernels read. There is no dot-product
//! instruction, so four bytes of a row times four input values is two steps:
//! unsigned bytes by signed bytes into 16-bit pairs, and the pairs into 32
//! bits. The pairs do not overflow: a stored value is at most 63, or for a
//! Q8_0 weight its size, at most 128, and an input value is at most 127 in
//! size, so a pair is at most 32,512.
//!
//! Each kernel sums in whole numbers until a block's last step and takes the
//! same float steps there as its AVX-512 twin, fused the same way, so a row's
//! product is the same value on either CPU. A CPU with AVX-512 VNNI and
//! without VBMI multiplies its six-bit tiles here.

use super::super::phases::{dispatch, Phase};
use super::{Row, TileQ4K, TileQ6, TileQ8, Tiled, Tiles, QK8_0, QK_K, ROWS};
use std::arch::x86_64::*;

/// Sixteen rows' values, eight to a vector.
type F16 = [__m256; 2];

/// Where one weight's products go: `rows.len()` rows of `width` values.
/// Each unit of work writes its own sixteens of every row and no other
/// unit writes them, which is what lets the threads share it.
struct Out {
    at: *mut f32,
    width: usize,
}

// SAFETY: the pointer is only written through, at places no two units of
// work share (see `multiply`), for as long as the vector it points into
// is borrowed.
unsafe impl Sync for Out {}

/// `acc` with, added to each row's 32 bits, the sum of the four products of
/// the unsigned bytes of `u` and the signed bytes of `s` that lie over it.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn dot(acc: __m256i, u: __m256i, s: __m256i) -> __m256i {
    let pairs = _mm256_maddubs_epi16(u, s);
    _mm256_add_epi32(acc, _mm256_madd_epi16(pairs, _mm256_set1_epi16(1)))
}

/// Four input values, repeated across eight rows.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn four(act: *const i8, at: usize) -> __m256i {
    _mm256_set1_epi32(std::ptr::read_unaligned(act.add(at) as *const i32))
}

/// Sixteen rows' f16 scales, each times `by`.
#[inline]
#[target_feature(enable = "avx2,f16c")]
unsafe fn scales_by(d: &[u16; ROWS], by: f32) -> F16 {
    let by = _mm256_set1_ps(by);
    [
        _mm256_mul_ps(
            _mm256_cvtph_ps(_mm_loadu_si128(d.as_ptr() as *const __m128i)),
            by,
        ),
        _mm256_mul_ps(
            _mm256_cvtph_ps(_mm_loadu_si128(d.as_ptr().add(8) as *const __m128i)),
            by,
        ),
    ]
}

/// The sum of a vector's eight whole numbers.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn summed(v: __m256i) -> i32 {
    let four = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
    let two = _mm_add_epi32(four, _mm_shuffle_epi32::<0b00_00_11_10>(four));
    let one = _mm_add_epi32(two, _mm_shuffle_epi32::<0b00_00_00_01>(two));
    _mm_cvtsi128_si32(one)
}

/// [`super::quantize_plainly`] eight values at a step, value for value the
/// same: the scale from the first value furthest from zero, each value
/// rounded half away from zero and held to an i8, and the sums 16 at a time.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn quantize(x: &[f32], row: &mut Row) {
    let kb = x.len() / QK_K;
    row.sized(kb);
    for i in 0..kb {
        quantize_block(
            x.as_ptr().add(i * QK_K),
            row.d.as_mut_ptr().add(i),
            row.qs.as_mut_ptr().add(i * QK_K),
            row.sums.as_mut_ptr().add(i * 16),
        );
    }
}

/// One block of 256 values at `xs` quantized: its scale to `d`, its values
/// to `qs` and their sixteen sums, 16 at a time, to `sums`, all of them
/// written whatever they held.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn quantize_block(xs: *const f32, d: *mut f32, qs: *mut i8, sums: *mut i32) {
    let size = _mm256_castsi256_ps(_mm256_set1_epi32(i32::MAX));
    let mut most = _mm256_setzero_ps();
    for j in 0..QK_K / 8 {
        most = _mm256_max_ps(most, _mm256_and_ps(_mm256_loadu_ps(xs.add(j * 8)), size));
    }
    let mut eight = [0f32; 8];
    _mm256_storeu_ps(eight.as_mut_ptr(), most);
    let amax = eight.iter().fold(0f32, |most, v| most.max(*v));
    if amax == 0.0 {
        // A block of nothing: no scale is taken from it.
        *d = 0.0;
        std::ptr::write_bytes(qs, 0, QK_K);
        std::ptr::write_bytes(sums, 0, 16);
        return;
    }
    // The first value that far from zero, with its sign.
    let reached = _mm256_set1_ps(amax);
    let mut max = amax;
    for j in 0..QK_K / 8 {
        let v = _mm256_and_ps(_mm256_loadu_ps(xs.add(j * 8)), size);
        let at = _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_EQ_OQ>(v, reached));
        if at != 0 {
            max = *xs.add(j * 8 + at.trailing_zeros() as usize);
            break;
        }
    }
    let sign = _mm256_castsi256_ps(_mm256_set1_epi32(i32::MIN));
    let (half, one) = (_mm256_set1_ps(0.5), _mm256_set1_ps(1.0));
    let (low, high) = (_mm256_set1_ps(-128.0), _mm256_set1_ps(127.0));
    let iscale = -127f32 / max;
    let by = _mm256_set1_ps(iscale);
    for j in 0..QK_K / 16 {
        let mut sum = _mm256_setzero_si256();
        for k in 0..2 {
            let t = _mm256_mul_ps(by, _mm256_loadu_ps(xs.add(j * 16 + k * 8)));
            // Half away from zero: toward zero, then one further where what
            // was cut off is a half or more.
            let cut = _mm256_round_ps::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(t);
            let off = _mm256_and_ps(_mm256_sub_ps(t, cut), size);
            let further = _mm256_or_ps(one, _mm256_and_ps(t, sign));
            let step = _mm256_and_ps(further, _mm256_cmp_ps::<_CMP_GE_OQ>(off, half));
            let rounded = _mm256_add_ps(cut, step);
            let held = _mm256_cvttps_epi32(_mm256_min_ps(_mm256_max_ps(rounded, low), high));
            sum = _mm256_add_epi32(sum, held);
            // Eight whole numbers to eight bytes: narrowed within each half
            // of the vector, and the two halves' four bytes set side by side.
            let words = _mm256_packs_epi32(held, held);
            let bytes = _mm256_packs_epi16(words, words);
            let at = qs.add(j * 16 + k * 8) as *mut i32;
            std::ptr::write_unaligned(at, _mm_cvtsi128_si32(_mm256_castsi256_si128(bytes)));
            std::ptr::write_unaligned(
                at.add(1),
                _mm_cvtsi128_si32(_mm256_extracti128_si256::<1>(bytes)),
            );
        }
        *sums.add(j) = summed(sum);
    }
    *d = 1.0 / iscale;
}

/// Sixteen rows' products with one block of 256 input values, added to
/// `acc`. A weight is `d * scale * q - dmin * min`, so a row's product is
/// each run of 32's `q · input` by its scale, less its min by the input's
/// own sum over the run: whole numbers until the block's last step, where
/// the two totals meet the row's two scales.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn q4k_block(tile: &TileQ4K, act: *const i8, d_act: f32, sums: *const i32, acc: F16) -> F16 {
    let mask = _mm256_set1_epi8(0x0F);
    let qs = tile.qs.as_ptr();
    let mut scaled = [_mm256_setzero_si256(); 2];
    for s in 0..8 {
        let (mut low, mut high) = ([_mm256_setzero_si256(); 2], [_mm256_setzero_si256(); 2]);
        for g in 0..4 {
            let group = qs.add((g * 8 + s) * ROWS * 4);
            let (first, second) = (four(act, s * 32 + g * 8), four(act, s * 32 + g * 8 + 4));
            for h in 0..2 {
                let w = _mm256_loadu_si256(group.add(h * 32) as *const __m256i);
                low[h] = dot(low[h], _mm256_and_si256(w, mask), first);
                high[h] = dot(
                    high[h],
                    _mm256_and_si256(_mm256_srli_epi16::<4>(w), mask),
                    second,
                );
            }
        }
        for h in 0..2 {
            let scale = _mm256_cvtepu8_epi32(_mm_loadl_epi64(
                tile.scales[s].as_ptr().add(h * 8) as *const __m128i
            ));
            scaled[h] = _mm256_add_epi32(
                scaled[h],
                _mm256_mullo_epi32(_mm256_add_epi32(low[h], high[h]), scale),
            );
        }
    }
    // Each row's mins by the input's sums, two runs to a step: the mins as
    // 16-bit pairs against the two sums as a pair.
    let mut mins = [_mm256_setzero_si256(); 2];
    for (pair, both) in tile.mins.iter().enumerate() {
        let first = *sums.add(4 * pair) + *sums.add(4 * pair + 1);
        let second = *sums.add(4 * pair + 2) + *sums.add(4 * pair + 3);
        // A run's sum is at most 32 values of 128: it fits 16 bits.
        let sum = _mm256_set1_epi32((first & 0xFFFF) | (second << 16));
        for (h, mins) in mins.iter_mut().enumerate() {
            let min =
                _mm256_cvtepu8_epi16(_mm_loadu_si128(both.as_ptr().add(h * 16) as *const __m128i));
            *mins = _mm256_add_epi32(*mins, _mm256_madd_epi16(min, sum));
        }
    }
    let d = scales_by(&tile.d, d_act);
    let dmin = scales_by(&tile.dmin, d_act);
    let mut out = acc;
    for h in 0..2 {
        let with = _mm256_fmadd_ps(_mm256_cvtepi32_ps(scaled[h]), d[h], out[h]);
        out[h] = _mm256_fnmadd_ps(_mm256_cvtepi32_ps(mins[h]), dmin[h], with);
    }
    out
}

/// A row's three bytes to the low three of its four, the fourth nothing:
/// for twelve bytes at the start of sixteen, and for twelve at their end.
const FRONT: [u8; 16] = [0, 1, 2, 0x80, 3, 4, 5, 0x80, 6, 7, 8, 0x80, 9, 10, 11, 0x80];
const BACK: [u8; 16] = [
    4, 5, 6, 0x80, 7, 8, 9, 0x80, 10, 11, 12, 0x80, 13, 14, 15, 0x80,
];

/// Eight rows' four six-bit values each, from half `h` of the group at
/// `bytes`: twenty-four bytes, read as two sixteens. The last twelve of a
/// group are read from four bytes back, so no load runs past the tile.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn sixes(bytes: *const u8, h: usize) -> __m256i {
    let front = _mm_loadu_si128(FRONT.as_ptr() as *const __m128i);
    let (four_rows, four_more) = if h == 0 {
        (
            _mm_shuffle_epi8(_mm_loadu_si128(bytes as *const __m128i), front),
            _mm_shuffle_epi8(_mm_loadu_si128(bytes.add(12) as *const __m128i), front),
        )
    } else {
        (
            _mm_shuffle_epi8(_mm_loadu_si128(bytes.add(24) as *const __m128i), front),
            _mm_shuffle_epi8(
                _mm_loadu_si128(bytes.add(32) as *const __m128i),
                _mm_loadu_si128(BACK.as_ptr() as *const __m128i),
            ),
        )
    };
    // Each row's 24 bits in 32: its four fields, each moved to a byte.
    let w = _mm256_set_m128i(four_more, four_rows);
    let first = _mm256_and_si256(w, _mm256_set1_epi32(0x3F));
    let second = _mm256_and_si256(_mm256_slli_epi32::<2>(w), _mm256_set1_epi32(0x3F00));
    let third = _mm256_and_si256(_mm256_slli_epi32::<4>(w), _mm256_set1_epi32(0x3F_0000));
    let fourth = _mm256_and_si256(_mm256_slli_epi32::<6>(w), _mm256_set1_epi32(0x3F00_0000));
    _mm256_or_si256(
        _mm256_or_si256(first, second),
        _mm256_or_si256(third, fourth),
    )
}

/// Sixteen rows' products with one block of 256 input values, added to
/// `acc`. A weight is `d * scale * (q - 32)`, so a row's product is each
/// run of 16's `q · input`, less 32 times the input's own sum over the run,
/// by its scale: whole numbers until the block's last step.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn q6k_block(tile: &TileQ6, act: *const i8, d_act: f32, sums: *const i32, acc: F16) -> F16 {
    let qs = tile.qs.as_ptr();
    let mut total = [_mm256_setzero_si256(); 2];
    for run in 0..16 {
        let mut isum = [_mm256_setzero_si256(); 2];
        for g in 0..4 {
            let group = qs.add((run * 4 + g) * ROWS * 3);
            let x = four(act, run * 16 + g * 4);
            for (h, isum) in isum.iter_mut().enumerate() {
                *isum = dot(*isum, sixes(group, h), x);
            }
        }
        let off = _mm256_set1_epi32(32 * *sums.add(run));
        for h in 0..2 {
            let scale = _mm256_cvtepi8_epi32(_mm_loadl_epi64(
                tile.scales[run].as_ptr().add(h * 8) as *const __m128i
            ));
            total[h] = _mm256_add_epi32(
                total[h],
                _mm256_mullo_epi32(_mm256_sub_epi32(isum[h], off), scale),
            );
        }
    }
    let d = scales_by(&tile.d, d_act);
    let mut out = acc;
    for h in 0..2 {
        out[h] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(total[h]), d[h], out[h]);
    }
    out
}

/// Sixteen rows' products with one block of 32 input values, added to
/// `acc`. The weights are stored 128 up; taking it back off makes them the
/// signed values they are, and a weight's size by the input given the
/// weight's sign is the product, so no correction by the input's sum is
/// needed.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn q8_block(tile: &TileQ8, act: *const i8, d_act: f32, acc: F16) -> F16 {
    let qs = tile.qs.as_ptr();
    let up = _mm256_set1_epi8(i8::MIN);
    let mut isum = [_mm256_setzero_si256(); 2];
    for g in 0..8 {
        let x = four(act, g * 4);
        for (h, isum) in isum.iter_mut().enumerate() {
            let stored = _mm256_loadu_si256(qs.add((g * ROWS + h * 8) * 4) as *const __m256i);
            let w = _mm256_xor_si256(stored, up);
            *isum = dot(*isum, _mm256_abs_epi8(w), _mm256_sign_epi8(x, w));
        }
    }
    let d = scales_by(&tile.d, d_act);
    let mut out = acc;
    for h in 0..2 {
        out[h] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(isum[h]), d[h], out[h]);
    }
    out
}

/// Rows whose running sums are kept at once while a sixteen's tiles go by.
const TOGETHER: usize = 32;

/// One block `b` of sixteen `t` of `w` against `row`, added to `acc`.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn block_of(w: &Tiled, t: usize, b: usize, row: &Row, acc: F16) -> F16 {
    match &w.tiles {
        Tiles::Q4K(tiles) => q4k_block(
            tiles.get_unchecked(t * (w.cols / QK_K) + b),
            row.qs.as_ptr().add(b * QK_K),
            *row.d.get_unchecked(b),
            row.sums.as_ptr().add(b * 16),
            acc,
        ),
        Tiles::Q6K(tiles) => q6k_block(
            tiles.get_unchecked(t * (w.cols / QK_K) + b),
            row.qs.as_ptr().add(b * QK_K),
            *row.d.get_unchecked(b),
            row.sums.as_ptr().add(b * 16),
            acc,
        ),
        Tiles::Q8_0(tiles) => q8_block(
            tiles.get_unchecked(t * (w.cols / QK8_0) + b),
            row.qs.as_ptr().add(b * QK8_0),
            *row.d.get_unchecked(b / 8),
            acc,
        ),
    }
}

/// Blocks to a row of `w`.
fn blocks(w: &Tiled) -> usize {
    match &w.tiles {
        Tiles::Q4K(_) | Tiles::Q6K(_) => w.cols / QK_K,
        Tiles::Q8_0(_) => w.cols / QK8_0,
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn store(at: *mut f32, sixteen: F16) {
    _mm256_storeu_ps(at, sixteen[0]);
    _mm256_storeu_ps(at.add(8), sixteen[1]);
}

/// The sixteens `first..last` of `w` against every row, written to `out`.
/// A sixteen's tiles go by once, each against every row (no more than
/// [`TOGETHER`] at once) before the next: a wide weight's sixteen is more
/// tiles than the nearest cache holds.
///
/// SAFETY (for the caller): the CPU has the features named, `out` has room
/// for `rows.len()` rows of `w.rows` values, and no other thread writes
/// these sixteens.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn sixteens_of(w: &Tiled, first: usize, last: usize, rows: &[Row], out: &Out) {
    let per_row = blocks(w);
    for t in first..last {
        if let [row] = rows {
            let mut acc = [_mm256_setzero_ps(); 2];
            for b in 0..per_row {
                acc = block_of(w, t, b, row, acc);
            }
            store(out.at.add(t * ROWS), acc);
            continue;
        }
        for (run, some) in rows.chunks(TOGETHER).enumerate() {
            let mut acc = [[_mm256_setzero_ps(); 2]; TOGETHER];
            for b in 0..per_row {
                for (acc, row) in acc.iter_mut().zip(some) {
                    *acc = block_of(w, t, b, row, *acc);
                }
            }
            for (i, acc) in acc[..some.len()].iter().enumerate() {
                store(
                    out.at.add((run * TOGETHER + i) * out.width + t * ROWS),
                    *acc,
                );
            }
        }
    }
}

/// [`sixteens_of`] for a caller that holds only where the products go.
///
/// SAFETY (for the caller): as [`sixteens_of`], with `out` room for
/// `rows.len()` rows of `w.rows` values.
#[target_feature(enable = "avx2,fma,f16c")]
pub(super) unsafe fn sixteens(w: &Tiled, first: usize, last: usize, rows: &[Row], out: *mut f32) {
    sixteens_of(
        w,
        first,
        last,
        rows,
        &Out {
            at: out,
            width: w.rows,
        },
    );
}

/// Every weight in `group` against every row, into `outs` (one slice a
/// weight, `rows.len()` rows of its width): one phase of one dispatch, its
/// units the group's sixteens, one weight's after another's.
pub(super) fn multiply(group: &[&Tiled], rows: &[Row], outs: &mut [&mut [f32]]) {
    let total: usize = group.iter().map(|w| w.rows / ROWS).sum();
    let outs: Vec<Out> = group
        .iter()
        .zip(outs.iter_mut())
        .map(|(w, out)| {
            debug_assert_eq!(out.len(), rows.len() * w.rows);
            Out {
                at: out.as_mut_ptr(),
                width: w.rows,
            }
        })
        .collect();
    let per_sixteen = group.first().map_or(1, |w| w.sixteen_bytes());
    // The sixteens `first..last` of the group, in the weights they fall in.
    let run = |first: usize, last: usize| {
        let mut before = 0;
        for (w, out) in group.iter().zip(&outs) {
            let sixteens = w.rows / ROWS;
            let (from, to) = (first.max(before), last.min(before + sixteens));
            if from < to {
                // SAFETY: a weight is only tiled where `available` found
                // the features; `out` was sized for every row of this
                // weight just above; and a unit is done by one thread.
                unsafe { sixteens_of(w, from - before, to - before, rows, out) };
            }
            before += sixteens;
        }
    };
    dispatch(
        &[Phase {
            units: total,
            most: (super::CLAIM / (per_sixteen * rows.len().max(1))).max(1),
            run: &run,
            then: None,
        }],
        false,
    );
}
