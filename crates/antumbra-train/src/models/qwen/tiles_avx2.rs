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
//! The second step is taken as seldom as the pairs allow. A four-bit value's
//! pair is at most 3,840, so a run of 32's eight steps are summed in their
//! sixteen bits, and a six-bit value's at most 16,128, so two are; and the
//! step that widens them multiplies them by the run's scale as it does, each
//! row's scale in both halves of its 32 bits. A Q8_0 weight's pairs have no
//! such room.
//!
//! Six-bit values lie in their tiles another way round for these kernels
//! ([`super::Sixes`]): the first of a row's four highest in its 24 bits,
//! where two multiplies move the four to a byte each.
//!
//! Rows multiplied together share that work. A tile is cut once, every value
//! to a byte of its own and every scale to the sixteen bits it is multiplied
//! in, and each row is then two steps a vector against what was cut. On one
//! thread of a laptop CPU, a row among sixteen takes a little over a third
//! of the time a row alone takes through a six-bit tile, and three quarters
//! through the others, whose values want little cutting. A cut is a store
//! for every vector of it, so a row or a few are multiplied as one row is
//! ([`cut_from`]).
//!
//! What a row then waits on is how many steps it is, twenty to a run of
//! sixteen six-bit values, on the three ports that do a vector's arithmetic;
//! not its loads, and not the multiplies alone. Reading the input as the
//! unsigned side of each product, so that what was cut is read from memory
//! as it is multiplied, and repeating a run's input from one load where four
//! values are four, were both tried and take as long or longer.
//!
//! Each kernel sums in whole numbers until a block's last step and takes the
//! same float steps there as its AVX-512 twin, fused the same way, so a row's
//! product is the same value on either CPU, and the same alone as among
//! other rows. A CPU with AVX-512 VNNI and without VBMI multiplies its
//! six-bit tiles here.

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

/// Where each of eight rows' bytes goes to be the high byte of both halves
/// of its row's 32 bits.
const TWICE: [u8; 32] = {
    let mut to = [0x80u8; 32];
    let mut row = 0;
    while row < 8 {
        to[row * 4 + 1] = row as u8;
        to[row * 4 + 3] = row as u8;
        row += 1;
    }
    to
};

/// Eight rows' bytes at `bytes`, each a signed sixteen bits in both halves
/// of its row's 32: what a row's pairs of products are multiplied by to be
/// summed and scaled in the one step.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn twice(bytes: *const u8) -> __m256i {
    let eight = _mm256_broadcastq_epi64(_mm_loadl_epi64(bytes as *const __m128i));
    let high = _mm256_shuffle_epi8(eight, _mm256_loadu_si256(TWICE.as_ptr() as *const __m256i));
    _mm256_srai_epi16::<8>(high)
}

/// A block's sixteen sums of sixteen input values at `sums`, two to each 32
/// bits: element `p` holds sums `2p` and `2p + 1`, the first lowest. A sum
/// of sixteen values is at most 2,048 in size, so it fits sixteen bits.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn paired(sums: *const i32) -> [i32; 8] {
    let packed = _mm256_packs_epi32(
        _mm256_loadu_si256(sums as *const __m256i),
        _mm256_loadu_si256(sums.add(8) as *const __m256i),
    );
    // Packed within each half of the vector: the quarters back in order.
    let ordered = _mm256_permute4x64_epi64::<0b11_01_10_00>(packed);
    let mut pairs = [0i32; 8];
    _mm256_storeu_si256(pairs.as_mut_ptr() as *mut __m256i, ordered);
    pairs
}

/// The sums of a block's eight runs of 32 input values, from its sixteen
/// sums of sixteen at `sums`, two to each 32 bits: element `p` holds runs
/// `2p` and `2p + 1`, the first lowest. A run's sum is at most 4,096 in
/// size.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn paired_runs(sums: *const i32) -> [i32; 4] {
    // Added within each half of the vector: runs 0, 1, 4, 5 and 2, 3, 6, 7.
    let runs = _mm256_hadd_epi32(
        _mm256_loadu_si256(sums as *const __m256i),
        _mm256_loadu_si256(sums.add(8) as *const __m256i),
    );
    let packed = _mm256_packs_epi32(runs, runs);
    let mut all = [0i32; 8];
    _mm256_storeu_si256(all.as_mut_ptr() as *mut __m256i, packed);
    [all[0], all[4], all[1], all[5]]
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

/// A Q4_K block's last steps: each row's scaled sums and its mins' meet its
/// two scales, and are added to `acc`.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn q4k_floats(
    tile: &TileQ4K,
    d_act: f32,
    scaled: [__m256i; 2],
    mins: [__m256i; 2],
    acc: F16,
) -> F16 {
    let d = scales_by(&tile.d, d_act);
    let dmin = scales_by(&tile.dmin, d_act);
    let mut out = acc;
    for h in 0..2 {
        let with = _mm256_fmadd_ps(_mm256_cvtepi32_ps(scaled[h]), d[h], out[h]);
        out[h] = _mm256_fnmadd_ps(_mm256_cvtepi32_ps(mins[h]), dmin[h], with);
    }
    out
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
        // A run's eight steps of pairs, summed in their sixteen bits.
        let mut pairs = [_mm256_setzero_si256(); 2];
        for g in 0..4 {
            let group = qs.add((g * 8 + s) * ROWS * 4);
            let (first, second) = (four(act, s * 32 + g * 8), four(act, s * 32 + g * 8 + 4));
            for (h, pairs) in pairs.iter_mut().enumerate() {
                let w = _mm256_loadu_si256(group.add(h * 32) as *const __m256i);
                let low = _mm256_maddubs_epi16(_mm256_and_si256(w, mask), first);
                let high =
                    _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16::<4>(w), mask), second);
                *pairs = _mm256_add_epi16(*pairs, _mm256_add_epi16(low, high));
            }
        }
        for h in 0..2 {
            let scale = twice(tile.scales[s].as_ptr().add(h * 8));
            scaled[h] = _mm256_add_epi32(scaled[h], _mm256_madd_epi16(pairs[h], scale));
        }
    }
    // Each row's mins by the input's sums, two runs to a step: the mins as
    // 16-bit pairs against the two sums as a pair.
    let runs = paired_runs(sums);
    let mut mins = [_mm256_setzero_si256(); 2];
    for (both, sum) in tile.mins.iter().zip(runs) {
        let sum = _mm256_set1_epi32(sum);
        for (h, mins) in mins.iter_mut().enumerate() {
            let min =
                _mm256_cvtepu8_epi16(_mm_loadu_si128(both.as_ptr().add(h * 16) as *const __m128i));
            *mins = _mm256_add_epi32(*mins, _mm256_madd_epi16(min, sum));
        }
    }
    q4k_floats(tile, d_act, scaled, mins, acc)
}

/// A Q4_K tile cut for rows multiplied together: every value in a byte of
/// its own, and each run's scales and mins in the sixteen bits they are
/// multiplied in.
struct CutQ4K {
    /// `[run of 32 (8)][half (2)][step (4)][low, high]`: eight rows' four
    /// values, in the order a row reads them.
    qs: [__m256i; QK_K / 2],
    /// `[run (8)][half (2)]`: each row's scale, in both halves of its 32
    /// bits.
    scales: [__m256i; 16],
    /// `[pair of runs (4)][half (2)]`: each row's two mins.
    mins: [__m256i; 8],
}

/// `tile` cut into the room at `cut`, every vector of which is written.
///
/// SAFETY (for the caller): the CPU has AVX2, and `cut` is room for a cut,
/// whatever it holds.
#[target_feature(enable = "avx2")]
unsafe fn q4k_cut(tile: &TileQ4K, cut: *mut CutQ4K) {
    let mask = _mm256_set1_epi8(0x0F);
    let qs = tile.qs.as_ptr();
    let to = std::ptr::addr_of_mut!((*cut).qs) as *mut __m256i;
    let scales = std::ptr::addr_of_mut!((*cut).scales) as *mut __m256i;
    let mins = std::ptr::addr_of_mut!((*cut).mins) as *mut __m256i;
    for s in 0..8 {
        for h in 0..2 {
            for g in 0..4 {
                let w =
                    _mm256_loadu_si256(qs.add((g * 8 + s) * ROWS * 4 + h * 32) as *const __m256i);
                let at = ((s * 2 + h) * 4 + g) * 2;
                to.add(at).write(_mm256_and_si256(w, mask));
                to.add(at + 1)
                    .write(_mm256_and_si256(_mm256_srli_epi16::<4>(w), mask));
            }
            scales
                .add(s * 2 + h)
                .write(twice(tile.scales[s].as_ptr().add(h * 8)));
        }
    }
    for (pair, both) in tile.mins.iter().enumerate() {
        for h in 0..2 {
            mins.add(pair * 2 + h)
                .write(_mm256_cvtepu8_epi16(_mm_loadu_si128(
                    both.as_ptr().add(h * 16) as *const __m128i,
                )));
        }
    }
}

/// [`q4k_block`] of block `b` of `row` against a tile already cut: the same
/// whole numbers, and the same float steps.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn q4k_row(cut: &CutQ4K, tile: &TileQ4K, row: &Row, b: usize, acc: F16) -> F16 {
    let act = row.qs.as_ptr().add(b * QK_K);
    let d_act = *row.d.get_unchecked(b);
    let sums = row.sums.as_ptr().add(b * 16);
    let mut scaled = [_mm256_setzero_si256(); 2];
    for s in 0..8 {
        let mut x = [_mm256_setzero_si256(); 8];
        for (i, x) in x.iter_mut().enumerate() {
            *x = four(act, s * 32 + i * 4);
        }
        for (h, scaled) in scaled.iter_mut().enumerate() {
            let q = cut.qs.as_ptr().add((s * 2 + h) * 8);
            let mut pairs = _mm256_maddubs_epi16(*q, x[0]);
            for (i, x) in x.iter().enumerate().skip(1) {
                pairs = _mm256_add_epi16(pairs, _mm256_maddubs_epi16(*q.add(i), *x));
            }
            let scale = *cut.scales.get_unchecked(s * 2 + h);
            *scaled = _mm256_add_epi32(*scaled, _mm256_madd_epi16(pairs, scale));
        }
    }
    let runs = paired_runs(sums);
    let mut mins = [_mm256_setzero_si256(); 2];
    for (pair, sum) in runs.into_iter().enumerate() {
        let sum = _mm256_set1_epi32(sum);
        for (h, mins) in mins.iter_mut().enumerate() {
            let min = *cut.mins.get_unchecked(pair * 2 + h);
            *mins = _mm256_add_epi32(*mins, _mm256_madd_epi16(min, sum));
        }
    }
    q4k_floats(tile, d_act, scaled, mins, acc)
}

/// Where each byte of eight rows' three goes to have its row's four values
/// cut out: a row's bytes as its second, its first, its third and its second
/// again. For 24 bytes loaded from four before them, twelve at the end of
/// the first sixteen and twelve at the start of the second; and for two
/// sixteens that each end in twelve.
const FRONT: [u8; 32] = [
    5, 4, 6, 5, 8, 7, 9, 8, 11, 10, 12, 11, 14, 13, 15, 14, 1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 10,
    9, 11, 10,
];
const BACK: [u8; 32] = [
    5, 4, 6, 5, 8, 7, 9, 8, 11, 10, 12, 11, 14, 13, 15, 14, 5, 4, 6, 5, 8, 7, 9, 8, 11, 10, 12, 11,
    14, 13, 15, 14,
];

/// A six-bit tile's values, through a pointer that may read the tile's other
/// fields: the first group's first half is loaded from four bytes before it.
#[inline]
fn sixes_of(tile: &TileQ6) -> *const u8 {
    // SAFETY: `qs` lies inside the tile the pointer was made from.
    unsafe { (tile as *const TileQ6 as *const u8).add(std::mem::offset_of!(TileQ6, qs)) }
}

/// The upper sixteen bits of each product of two unsigned sixteen bits.
/// Written out because the compiler takes a multiply by powers of two for
/// the shifts it stands for, has no instruction here that shifts each
/// sixteen bits by a count of its own, and makes seven instructions of the
/// one.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn upper_of_products(a: __m256i, b: __m256i) -> __m256i {
    let out: __m256i;
    std::arch::asm!(
        "vpmulhuw {out}, {a}, {b}",
        out = lateout(ymm_reg) out,
        a = in(ymm_reg) a,
        b = in(ymm_reg) b,
        options(pure, nomem, nostack, preserves_flags),
    );
    out
}

/// The lower sixteen bits of each product of two sixteen bits, written out
/// as [`upper_of_products`] is.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn lower_of_products(a: __m256i, b: __m256i) -> __m256i {
    let out: __m256i;
    std::arch::asm!(
        "vpmullw {out}, {a}, {b}",
        out = lateout(ymm_reg) out,
        a = in(ymm_reg) a,
        b = in(ymm_reg) b,
        options(pure, nomem, nostack, preserves_flags),
    );
    out
}

/// Eight rows' four six-bit values each, a byte each, from half `h` of the
/// group at `group`: 24 bytes, a row's four values in its three with the
/// first of them highest ([`super::Sixes::FirstHighest`]). With a row's
/// bytes set out as two sixteen-bit halves that each hold two of its values
/// whole, one multiply moves the upper value of each half down to its byte
/// and another moves the lower one up to its own.
///
/// SAFETY (for the caller): four bytes before the group are readable, as
/// they are behind [`sixes_of`].
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn sixes(group: *const u8, h: usize) -> __m256i {
    let w = if h == 0 {
        _mm256_shuffle_epi8(
            _mm256_loadu_si256(group.sub(4) as *const __m256i),
            _mm256_loadu_si256(FRONT.as_ptr() as *const __m256i),
        )
    } else {
        // The last twelve bytes are read as the end of a sixteen, so no
        // load runs past the tile.
        let both = _mm256_inserti128_si256::<1>(
            _mm256_castsi128_si256(_mm_loadu_si128(group.add(20) as *const __m128i)),
            _mm_loadu_si128(group.add(32) as *const __m128i),
        );
        _mm256_shuffle_epi8(both, _mm256_loadu_si256(BACK.as_ptr() as *const __m256i))
    };
    let upper = upper_of_products(
        _mm256_and_si256(w, _mm256_set1_epi32(0x0FC0_FC00)),
        _mm256_set1_epi32(0x0400_0040),
    );
    let lower = lower_of_products(
        _mm256_and_si256(w, _mm256_set1_epi32(0x003F_03F0)),
        _mm256_set1_epi32(0x0100_0010),
    );
    _mm256_or_si256(upper, lower)
}

/// A Q6_K block's last step: a weight is its stored value less 32, so 32
/// times the input's scaled sums come off the scaled products, and what is
/// left meets the row's scale and is added to `acc`.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn q6k_floats(
    tile: &TileQ6,
    d_act: f32,
    total: [__m256i; 2],
    offs: [__m256i; 2],
    acc: F16,
) -> F16 {
    let d = scales_by(&tile.d, d_act);
    let mut out = acc;
    for h in 0..2 {
        let whole = _mm256_sub_epi32(total[h], _mm256_slli_epi32::<5>(offs[h]));
        out[h] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(whole), d[h], out[h]);
    }
    out
}

/// Sixteen rows' products with one block of 256 input values, added to
/// `acc`. A weight is `d * scale * (q - 32)`, so a row's product is each
/// run of 16's `q · input`, less 32 times the input's own sum over the run,
/// by its scale: whole numbers until the block's last step.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn q6k_block(tile: &TileQ6, act: *const i8, d_act: f32, sums: *const i32, acc: F16) -> F16 {
    let qs = sixes_of(tile);
    let sums = paired(sums);
    let mut total = [_mm256_setzero_si256(); 2];
    let mut offs = [_mm256_setzero_si256(); 2];
    for (two, sum) in sums.into_iter().enumerate() {
        let mut scales = [[_mm256_setzero_si256(); 2]; 2];
        for (r, scales) in scales.iter_mut().enumerate() {
            let run = two * 2 + r;
            for (h, scale) in scales.iter_mut().enumerate() {
                *scale = twice(tile.scales[run].as_ptr().add(h * 8) as *const u8);
            }
            // Two groups' pairs fit sixteen bits together.
            for g in [run * 4, run * 4 + 2] {
                let (x, next) = (four(act, g * 4), four(act, g * 4 + 4));
                let group = qs.add(g * ROWS * 3);
                for (h, total) in total.iter_mut().enumerate() {
                    let pairs = _mm256_add_epi16(
                        _mm256_maddubs_epi16(sixes(group, h), x),
                        _mm256_maddubs_epi16(sixes(group.add(ROWS * 3), h), next),
                    );
                    *total = _mm256_add_epi32(*total, _mm256_madd_epi16(pairs, scales[h]));
                }
            }
        }
        // The two runs' sums of the input by their scales, in one step.
        let sum = _mm256_set1_epi32(sum);
        for (h, offs) in offs.iter_mut().enumerate() {
            let both = _mm256_blend_epi16::<0xAA>(scales[0][h], scales[1][h]);
            *offs = _mm256_add_epi32(*offs, _mm256_madd_epi16(both, sum));
        }
    }
    q6k_floats(tile, d_act, total, offs, acc)
}

/// A Q6_K tile cut for rows multiplied together: every value in a byte of
/// its own, and each run's scales in the sixteen bits they are multiplied
/// in.
struct CutQ6 {
    /// `[run of 16 (16)][half (2)][group of 4 values (4)]`: eight rows' four
    /// values, in the order a row reads them.
    qs: [__m256i; QK_K / 2],
    /// `[run (16)][half (2)]`: each row's scale, in both halves of its 32
    /// bits.
    scales: [__m256i; 32],
    /// `[pair of runs (8)][half (2)]`: each row's scales of the two runs.
    both: [__m256i; 16],
}

/// `tile` cut into the room at `cut`, every vector of which is written.
///
/// SAFETY (for the caller): the CPU has AVX2, and `cut` is room for a cut,
/// whatever it holds.
#[target_feature(enable = "avx2")]
unsafe fn q6k_cut(tile: &TileQ6, cut: *mut CutQ6) {
    let qs = sixes_of(tile);
    let to = std::ptr::addr_of_mut!((*cut).qs) as *mut __m256i;
    let scales = std::ptr::addr_of_mut!((*cut).scales) as *mut __m256i;
    let both = std::ptr::addr_of_mut!((*cut).both) as *mut __m256i;
    for run in 0..16 {
        for h in 0..2 {
            for g in 0..4 {
                to.add((run * 2 + h) * 4 + g)
                    .write(sixes(qs.add((run * 4 + g) * ROWS * 3), h));
            }
            scales
                .add(run * 2 + h)
                .write(twice(tile.scales[run].as_ptr().add(h * 8) as *const u8));
        }
    }
    for two in 0..8 {
        for h in 0..2 {
            both.add(two * 2 + h).write(_mm256_blend_epi16::<0xAA>(
                *scales.add(two * 4 + h),
                *scales.add(two * 4 + 2 + h),
            ));
        }
    }
}

/// [`q6k_block`] of block `b` of `row` against a tile already cut: the same
/// whole numbers, and the same float step.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn q6k_row(cut: &CutQ6, tile: &TileQ6, row: &Row, b: usize, acc: F16) -> F16 {
    let act = row.qs.as_ptr().add(b * QK_K);
    let d_act = *row.d.get_unchecked(b);
    let sums = row.sums.as_ptr().add(b * 16);
    let mut total = [_mm256_setzero_si256(); 2];
    for run in 0..16 {
        let mut x = [_mm256_setzero_si256(); 4];
        for (g, x) in x.iter_mut().enumerate() {
            *x = four(act, run * 16 + g * 4);
        }
        for (h, total) in total.iter_mut().enumerate() {
            let q = cut.qs.as_ptr().add((run * 2 + h) * 4);
            let scale = *cut.scales.get_unchecked(run * 2 + h);
            let first = _mm256_add_epi16(
                _mm256_maddubs_epi16(*q, x[0]),
                _mm256_maddubs_epi16(*q.add(1), x[1]),
            );
            let second = _mm256_add_epi16(
                _mm256_maddubs_epi16(*q.add(2), x[2]),
                _mm256_maddubs_epi16(*q.add(3), x[3]),
            );
            *total = _mm256_add_epi32(
                *total,
                _mm256_add_epi32(
                    _mm256_madd_epi16(first, scale),
                    _mm256_madd_epi16(second, scale),
                ),
            );
        }
    }
    let mut offs = [_mm256_setzero_si256(); 2];
    for (two, sum) in paired(sums).into_iter().enumerate() {
        let sum = _mm256_set1_epi32(sum);
        for (h, offs) in offs.iter_mut().enumerate() {
            let both = *cut.both.get_unchecked(two * 2 + h);
            *offs = _mm256_add_epi32(*offs, _mm256_madd_epi16(both, sum));
        }
    }
    q6k_floats(tile, d_act, total, offs, acc)
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

/// A Q8_0 tile cut for rows multiplied together:
/// `[group of 4 values (8)][half (2)][the weights, their sizes]`.
struct CutQ8 {
    qs: [__m256i; QK8_0],
}

/// `tile` cut into the room at `cut`, every vector of which is written.
///
/// SAFETY (for the caller): the CPU has AVX2, and `cut` is room for a cut,
/// whatever it holds.
#[target_feature(enable = "avx2")]
unsafe fn q8_cut(tile: &TileQ8, cut: *mut CutQ8) {
    let qs = tile.qs.as_ptr();
    let to = std::ptr::addr_of_mut!((*cut).qs) as *mut __m256i;
    let up = _mm256_set1_epi8(i8::MIN);
    for g in 0..8 {
        for h in 0..2 {
            let stored = _mm256_loadu_si256(qs.add((g * ROWS + h * 8) * 4) as *const __m256i);
            let w = _mm256_xor_si256(stored, up);
            to.add((g * 2 + h) * 2).write(w);
            to.add((g * 2 + h) * 2 + 1).write(_mm256_abs_epi8(w));
        }
    }
}

/// [`q8_block`] of block `b` of `row` against a tile already cut: the same
/// whole numbers, and the same float step.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn q8_row(cut: &CutQ8, tile: &TileQ8, row: &Row, b: usize, acc: F16) -> F16 {
    let act = row.qs.as_ptr().add(b * QK8_0);
    let d_act = *row.d.get_unchecked(b / 8);
    let mut isum = [_mm256_setzero_si256(); 2];
    for g in 0..8 {
        let x = four(act, g * 4);
        for (h, isum) in isum.iter_mut().enumerate() {
            let q = cut.qs.as_ptr().add((g * 2 + h) * 2);
            *isum = dot(*isum, *q.add(1), _mm256_sign_epi8(x, *q));
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

/// Rows from which a tile is cut once for them all. Fewer are each multiplied
/// as one row is, a sixteen's tiles read from the cache again for each of
/// them: to cut a tile is to store every vector of it, which two or three
/// rows win back only where the cutting is most of a row's work, as it is of
/// six-bit values and is not of the others.
fn cut_from(w: &Tiled) -> usize {
    match &w.tiles {
        Tiles::Q6K(_) => 2,
        Tiles::Q4K(_) => 4,
        Tiles::Q8_0(_) => 16,
    }
}

/// The sixteens `$first..$last` of `$tiles`, `$per_row` tiles to a sixteen,
/// against the rows `$rows`, no more than [`TOGETHER`] at once, written to
/// `$out`. `$cut` cuts a tile into room of the type `$room`, once for every
/// row, and `$row` adds one row's products with what was cut to that row's
/// running sums.
macro_rules! together {
    ($tiles:expr, $per_row:expr, $first:expr, $last:expr, $rows:expr, $out:expr, $room:ty, $cut:path, $row:path) => {{
        // Room for one tile's cutting: every tile's is written whole before
        // any of it is read.
        let mut cut = std::mem::MaybeUninit::<$room>::uninit();
        let mut acc = [[_mm256_setzero_ps(); 2]; TOGETHER];
        for t in $first..$last {
            for (run, some) in $rows.chunks(TOGETHER).enumerate() {
                acc[..some.len()].fill([_mm256_setzero_ps(); 2]);
                for b in 0..$per_row {
                    let tile = $tiles.get_unchecked(t * $per_row + b);
                    $cut(tile, cut.as_mut_ptr());
                    let cut = &*cut.as_ptr();
                    for (acc, row) in acc.iter_mut().zip(some) {
                        *acc = $row(cut, tile, row, b, *acc);
                    }
                }
                for (i, acc) in acc[..some.len()].iter().enumerate() {
                    store(
                        $out.at.add((run * TOGETHER + i) * $out.width + t * ROWS),
                        *acc,
                    );
                }
            }
        }
    }};
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
/// A row or a few multiply each tile as they cut it, a row after another.
/// From [`cut_from`] rows on, a sixteen's tiles go by once, each cut once and
/// multiplied by every row (no more than [`TOGETHER`] at once) before the
/// next: a wide weight's sixteen is more tiles than the nearest cache holds.
///
/// SAFETY (for the caller): the CPU has the features named, `out` has room
/// for `rows.len()` rows of `w.rows` values, and no other thread writes
/// these sixteens.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn sixteens_of(w: &Tiled, first: usize, last: usize, rows: &[Row], out: &Out) {
    let per_row = blocks(w);
    if rows.len() < cut_from(w) {
        for t in first..last {
            for (i, row) in rows.iter().enumerate() {
                let mut acc = [_mm256_setzero_ps(); 2];
                for b in 0..per_row {
                    acc = block_of(w, t, b, row, acc);
                }
                store(out.at.add(i * out.width + t * ROWS), acc);
            }
        }
        return;
    }
    match &w.tiles {
        Tiles::Q4K(tiles) => {
            together!(tiles, per_row, first, last, rows, out, CutQ4K, q4k_cut, q4k_row)
        }
        Tiles::Q6K(tiles) => {
            together!(tiles, per_row, first, last, rows, out, CutQ6, q6k_cut, q6k_row)
        }
        Tiles::Q8_0(tiles) => {
            together!(tiles, per_row, first, last, rows, out, CutQ8, q8_cut, q8_row)
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
