//! The sketch's kernels on aarch64: NEON with the dot-product instructions,
//! four rows to a vector over the sketch the AVX-512 kernels read.

use super::{Row, Sketch, BLOCK, GROUP, LEVELS, ROWS};
use std::arch::aarch64::*;

/// Which of a quarter's six bytes each of its sixteen three-bit fields is cut
/// from: a pair of rows' three bytes as `[b0, b0, b0, b1, b1, b1, b2, b2]`,
/// and beside them the byte a field that straddles two takes its high bits
/// from.
const FROM: [u8; 16] = [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5];
const AND_FROM: [u8; 16] = [
    0xFF, 0xFF, 1, 0xFF, 0xFF, 2, 0xFF, 0xFF, 0xFF, 0xFF, 4, 0xFF, 0xFF, 5, 0xFF, 0xFF,
];
/// How far each byte of the two moves: right for the first, left for the
/// second.
const DOWN: [i8; 16] = [0, -3, -6, -1, -4, -7, -2, -5, 0, -3, -6, -1, -4, -7, -2, -5];
const UP: [i8; 16] = [0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0];

/// `SDOT`, written out because its intrinsic is not yet in a stable Rust.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn dot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let mut acc = acc;
    std::arch::asm!(
        "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
        acc = inout(vreg) acc,
        a = in(vreg) a,
        b = in(vreg) b,
        options(pure, nomem, nostack, preserves_flags),
    );
    acc
}

/// The sixteens `first..last` of the sketch against the one row `row`: each
/// sixteen's products to `products`, and the largest of them to `tops`. A
/// sketched weight is its row's scale times `2 * code + 1 - LEVELS`, so a
/// row's product is `2 * code · input` less `LEVELS - 1` times the input's
/// own sum, in whole numbers over each block of 256, then by the block's
/// scale, and by the row's at the end: the steps of the AVX-512 twin, and
/// its values.
///
/// SAFETY (for the caller): the CPU has the features named, `row` holds the
/// sketch's columns, `products` has room for every row's product and `tops`
/// for every sixteen's, the sixteens are among the sketch's own, and no
/// other thread writes their places.
#[target_feature(enable = "neon,dotprod")]
pub(super) unsafe fn sixteens(
    sketch: &Sketch,
    first: usize,
    last: usize,
    row: &Row,
    products: *mut f32,
    tops: *mut f32,
) {
    let (from, and_from) = (vld1q_u8(FROM.as_ptr()), vld1q_u8(AND_FROM.as_ptr()));
    let (down, up) = (vld1q_s8(DOWN.as_ptr()), vld1q_s8(UP.as_ptr()));
    let code = vdupq_n_u8(LEVELS as u8 - 1);
    let blocks = sketch.cols / BLOCK;
    let per = sketch.cols / 4 * GROUP;
    for t in first..last {
        let mut codes = sketch.codes.as_ptr().add(t * per);
        let mut acc = [vdupq_n_f32(0.0); 4];
        for b in 0..blocks {
            let (act, d, total) = row.block(b);
            let act = act.as_ptr();
            let mut sums = [vdupq_n_s32(0); 4];
            for g in 0..BLOCK / 4 {
                let x = vreinterpretq_s8_s32(vld1q_dup_s32(act.add(4 * g) as *const i32));
                let group = codes.add(g * GROUP);
                for (k, sum) in sums.iter_mut().enumerate() {
                    // A quarter is six bytes of the sixteen loaded: what is
                    // past it is the next quarter's, or the slack kept
                    // after the last, and the tables read none of it.
                    let loaded = vld1q_u8(group.add(k * 6));
                    let low = vshlq_u8(vqtbl1q_u8(loaded, from), down);
                    let high = vshlq_u8(vqtbl1q_u8(loaded, and_from), up);
                    let q = vreinterpretq_s8_u8(vandq_u8(vorrq_u8(low, high), code));
                    *sum = dot(*sum, q, x);
                }
            }
            codes = codes.add(BLOCK / 4 * GROUP);
            let off = vdupq_n_s32((LEVELS as i32 - 1) * total);
            let by = vdupq_n_f32(d);
            for (acc, sum) in acc.iter_mut().zip(sums) {
                let centered = vsubq_s32(vshlq_n_s32::<1>(sum), off);
                *acc = vfmaq_f32(*acc, vcvtq_f32_s32(centered), by);
            }
        }
        let mut top = vdupq_n_f32(f32::NEG_INFINITY);
        for (k, acc) in acc.iter().enumerate() {
            let v = vmulq_f32(
                *acc,
                vld1q_f32(sketch.scales.as_ptr().add(t * ROWS + k * 4)),
            );
            vst1q_f32(products.add(t * ROWS + k * 4), v);
            top = vmaxq_f32(top, v);
        }
        *tops.add(t) = vmaxvq_f32(top);
    }
}

/// Which of a sixteen's rows could still be first: those whose sketched
/// product, and their reach for an input of this `size`, come to `reached`
/// or more.
///
/// SAFETY (for the caller): the CPU has NEON, and `products` and `reach`
/// each point at sixteen values.
#[target_feature(enable = "neon")]
pub(super) unsafe fn could_be_first(
    products: *const f32,
    reach: *const f32,
    size: f32,
    reached: f32,
) -> u16 {
    let (size, reached) = (vdupq_n_f32(size), vdupq_n_f32(reached));
    let bit = vld1q_u32([1u32, 2, 4, 8].as_ptr());
    let mut mask = 0u16;
    for k in 0..4 {
        let bound = vfmaq_f32(
            vld1q_f32(products.add(k * 4)),
            vld1q_f32(reach.add(k * 4)),
            size,
        );
        let could = vandq_u32(vcgeq_f32(bound, reached), bit);
        mask |= (vaddvq_u32(could) as u16) << (4 * k);
    }
    mask
}
