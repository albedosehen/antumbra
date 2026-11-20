//! The sketch's kernels in AVX2, for the x86-64 CPUs that cut neither a
//! six-bit tile nor a sketch out of its bytes with AVX-512 VBMI: eight rows
//! to a vector over a sketch whose codes have a nibble each
//! ([`super::Layout::Nibbles`]), so a group's sixteen rows are one load, a
//! mask and a shift. The products are the AVX-512 kernel's, to the bit: the
//! same whole numbers, and its float steps.

use super::{Places, Row, Sketch, BLOCK, LEVELS, NIBBLE_GROUP, ROWS};
use std::arch::x86_64::*;

/// Groups whose pairs of products are summed in their sixteen bits before
/// they are widened. A code is at most 7 and an input value at most 128 in
/// size, so a pair of products is at most 1,792 and sixteen groups' at most
/// 28,672.
const NARROW: usize = 16;

/// Four input values, repeated across eight rows.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn four(act: *const i8, at: usize) -> __m256i {
    _mm256_set1_epi32(std::ptr::read_unaligned(act.add(at) as *const i32))
}

/// Sixteen `t` of the sketch against `N` rows of input at once: each row's
/// sixteen products, eight to a vector, as it would have them alone. A
/// sketched weight is its row's scale times `2 * code + 1 - LEVELS`, so a
/// product is `2 * code · input` less `LEVELS - 1` times the input's own
/// sum, in whole numbers over each block of 256, then by the block's scale,
/// and by the row's at the end.
///
/// A group's codes are cut out of their bytes once for every row of input,
/// and a row's products with them are two steps a vector: unsigned bytes by
/// signed bytes into sixteen-bit pairs, which are summed as they are for
/// [`NARROW`] groups before they are widened.
///
/// SAFETY (for the caller): the CPU has the features named, `rows` holds
/// `N` rows of the sketch's columns, the sketch's codes have a nibble each,
/// and the sixteen is among the sketch's own.
#[target_feature(enable = "avx2,fma")]
unsafe fn sixteen_of<const N: usize>(sketch: &Sketch, t: usize, rows: &[Row]) -> [[__m256; 2]; N] {
    let nibble = _mm256_set1_epi8(0x0F);
    let ones = _mm256_set1_epi16(1);
    let per = sketch.cols / 4 * NIBBLE_GROUP;
    let mut codes = sketch.codes.as_ptr().add(t * per);
    let mut acc = [[_mm256_setzero_ps(); 2]; N];
    for b in 0..sketch.cols / BLOCK {
        let blocks: [(*const i8, f32, i32); N] = std::array::from_fn(|i| {
            let (act, d, total) = rows[i].block(b);
            (act.as_ptr(), d, total)
        });
        let mut sums = [[_mm256_setzero_si256(); 2]; N];
        for part in 0..BLOCK / 4 / NARROW {
            let mut pairs = [[_mm256_setzero_si256(); 2]; N];
            for g in part * NARROW..(part + 1) * NARROW {
                let packed = _mm256_loadu_si256(codes.add(g * NIBBLE_GROUP) as *const __m256i);
                // The first eight rows' codes, and the eight after them.
                let low = _mm256_and_si256(packed, nibble);
                let high = _mm256_and_si256(_mm256_srli_epi16::<4>(packed), nibble);
                for (pairs, (act, _, _)) in pairs.iter_mut().zip(&blocks) {
                    let x = four(*act, 4 * g);
                    pairs[0] = _mm256_add_epi16(pairs[0], _mm256_maddubs_epi16(low, x));
                    pairs[1] = _mm256_add_epi16(pairs[1], _mm256_maddubs_epi16(high, x));
                }
            }
            for (sums, pairs) in sums.iter_mut().zip(&pairs) {
                for (sum, pairs) in sums.iter_mut().zip(pairs) {
                    *sum = _mm256_add_epi32(*sum, _mm256_madd_epi16(*pairs, ones));
                }
            }
        }
        codes = codes.add(BLOCK / 4 * NIBBLE_GROUP);
        for ((acc, sums), (_, d, total)) in acc.iter_mut().zip(&sums).zip(&blocks) {
            let off = _mm256_set1_epi32((LEVELS as i32 - 1) * total);
            let by = _mm256_set1_ps(*d);
            for (acc, sum) in acc.iter_mut().zip(sums) {
                let centered = _mm256_sub_epi32(_mm256_slli_epi32::<1>(*sum), off);
                *acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(centered), by, *acc);
            }
        }
    }
    let scales = sketch.scales.as_ptr().add(t * ROWS);
    for acc in acc.iter_mut() {
        for (h, acc) in acc.iter_mut().enumerate() {
            *acc = _mm256_mul_ps(*acc, _mm256_loadu_ps(scales.add(h * 8)));
        }
    }
    acc
}

/// The largest of sixteen values.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn largest(sixteen: &[__m256; 2]) -> f32 {
    let eight = _mm256_max_ps(sixteen[0], sixteen[1]);
    let four = _mm_max_ps(
        _mm256_castps256_ps128(eight),
        _mm256_extractf128_ps::<1>(eight),
    );
    let two = _mm_max_ps(four, _mm_movehl_ps(four, four));
    let one = _mm_max_ss(two, _mm_shuffle_ps::<0b01>(two, two));
    _mm_cvtss_f32(one)
}

/// The sixteens `first..last` of the sketch against every one of `rows`:
/// each row's products of them, and the largest of each sixteen's, to its
/// own of `places`. The rows are taken four, two and one at a time, a
/// sixteen's codes cut out of their bytes once for each such run of them.
///
/// SAFETY (for the caller): the CPU has the features named, every row holds
/// the sketch's columns, the sketch's codes have a nibble each, each of
/// `places` has room for a product of every row of the head and a largest
/// for every sixteen, the sixteens are among the sketch's own, and no other
/// thread writes their places.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn sixteens_of_each(
    sketch: &Sketch,
    first: usize,
    last: usize,
    rows: &[Row],
    places: &[Places],
) {
    let all = rows.len().min(places.len());
    for t in first..last {
        let mut done = 0;
        while done < all {
            let mut out = [[_mm256_setzero_ps(); 2]; 4];
            let n = match all - done {
                4.. => {
                    out = sixteen_of::<4>(sketch, t, &rows[done..]);
                    4
                }
                2.. => {
                    out[..2].copy_from_slice(&sixteen_of::<2>(sketch, t, &rows[done..]));
                    2
                }
                _ => {
                    out[..1].copy_from_slice(&sixteen_of::<1>(sketch, t, &rows[done..]));
                    1
                }
            };
            for (v, place) in out[..n].iter().zip(&places[done..]) {
                _mm256_storeu_ps(place.products.add(t * ROWS), v[0]);
                _mm256_storeu_ps(place.products.add(t * ROWS + 8), v[1]);
                *place.tops.add(t) = largest(v);
            }
            done += n;
        }
    }
}

/// Which of a sixteen's rows could still be first: those whose sketched
/// product, and their reach for an input of this `size`, come to `reached`
/// or more.
///
/// SAFETY (for the caller): the CPU has AVX2 and FMA, and `products` and
/// `reach` each point at sixteen values.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn could_be_first(
    products: *const f32,
    reach: *const f32,
    size: f32,
    reached: f32,
) -> u16 {
    let (size, reached) = (_mm256_set1_ps(size), _mm256_set1_ps(reached));
    let mut could = 0u16;
    for h in 0..2 {
        let bound = _mm256_fmadd_ps(
            _mm256_loadu_ps(reach.add(h * 8)),
            size,
            _mm256_loadu_ps(products.add(h * 8)),
        );
        let mask = _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_GE_OQ>(bound, reached));
        could |= (mask as u16) << (h * 8);
    }
    could
}
