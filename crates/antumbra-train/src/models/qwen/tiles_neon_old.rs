//! The tile kernels on aarch64: NEON with the dot-product instructions.
//!
//! A tile's sixteen rows are four vectors of four here, where AVX-512 holds
//! them in one of sixteen. A tile is laid out a row to every four bytes of a
//! group, so a quarter of a group is one 128-bit load and one signed dot
//! product against the group's four input values. Every stored value fits a
//! signed byte once a Q8_0 weight's 128 is taken back off, so the one
//! signed-by-signed product NEON has serves all three forms.
//!
//! Each kernel sums in whole numbers until a block's last step and takes the
//! same float steps there as its AVX-512 twin, fused the same way, so a row's
//! product is the same value on either CPU.

use super::super::phases::{dispatch, Phase};
use super::{Row, TileQ4K, TileQ6, TileQ8, Tiled, Tiles, QK8_0, QK_K, ROWS};
use std::arch::aarch64::*;

/// Sixteen rows' values, four to a vector.
type F16 = [float32x4_t; 4];

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

/// Sixteen rows' f16 scales, each times `by`. `FCVTL` is written out because
/// the half-precision vector types are not yet in a stable Rust; the
/// instruction itself is every ARMv8 CPU's.
#[target_feature(enable = "neon")]
unsafe fn scales_by(d: &[u16; ROWS], by: f32) -> F16 {
    let by = vdupq_n_f32(by);
    let mut out = [by; 4];
    for (k, out) in out.iter_mut().enumerate() {
        let halves = vld1_u16(d.as_ptr().add(k * 4));
        let wide: float32x4_t;
        std::arch::asm!(
            "fcvtl {wide:v}.4s, {halves:v}.4h",
            wide = lateout(vreg) wide,
            halves = in(vreg) halves,
            options(pure, nomem, nostack, preserves_flags),
        );
        *out = vmulq_f32(wide, by);
    }
    out
}

#[target_feature(enable = "neon")]
unsafe fn zeros() -> F16 {
    [vdupq_n_f32(0.0); 4]
}

/// [`super::quantize_plainly`] four values at a step, value for value the
/// same: the scale from the first value furthest from zero, each value
/// rounded half away from zero and held to an i8, and the sums 16 at a time.
#[target_feature(enable = "neon")]
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
#[target_feature(enable = "neon")]
pub(super) unsafe fn quantize_block(xs: *const f32, d: *mut f32, qs: *mut i8, sums: *mut i32) {
    let mut most = vdupq_n_f32(0.0);
    for j in 0..QK_K / 4 {
        most = vmaxq_f32(most, vabsq_f32(vld1q_f32(xs.add(j * 4))));
    }
    let amax = vmaxvq_f32(most);
    if amax == 0.0 {
        // A block of nothing: no scale is taken from it.
        *d = 0.0;
        std::ptr::write_bytes(qs, 0, QK_K);
        std::ptr::write_bytes(sums, 0, 16);
        return;
    }
    // The first value that far from zero, with its sign.
    let reached = vdupq_n_f32(amax);
    let mut max = amax;
    'found: for j in 0..QK_K / 4 {
        let at = vceqq_f32(vabsq_f32(vld1q_f32(xs.add(j * 4))), reached);
        if vmaxvq_u32(at) != 0 {
            for lane in 0..4 {
                let v = *xs.add(j * 4 + lane);
                if v.abs() == amax {
                    max = v;
                    break 'found;
                }
            }
        }
    }
    let (low, high) = (vdupq_n_f32(-128.0), vdupq_n_f32(127.0));
    let iscale = -127f32 / max;
    let by = vdupq_n_f32(iscale);
    for j in 0..QK_K / 16 {
        let mut held = [vdupq_n_s32(0); 4];
        for (k, held) in held.iter_mut().enumerate() {
            let t = vmulq_f32(by, vld1q_f32(xs.add(j * 16 + k * 4)));
            // Half away from zero is the rounding NEON calls "to nearest
            // with ties away".
            *held = vcvtq_s32_f32(vminq_f32(vmaxq_f32(vrndaq_f32(t), low), high));
        }
        let lower = vcombine_s16(vqmovn_s32(held[0]), vqmovn_s32(held[1]));
        let upper = vcombine_s16(vqmovn_s32(held[2]), vqmovn_s32(held[3]));
        vst1q_s8(
            qs.add(j * 16),
            vcombine_s8(vqmovn_s16(lower), vqmovn_s16(upper)),
        );
        *sums.add(j) = vaddvq_s32(vaddq_s32(
            vaddq_s32(held[0], held[1]),
            vaddq_s32(held[2], held[3]),
        ));
    }
    *d = 1.0 / iscale;
}

/// `acc` with, added to each of its four, the sum of the four products of
/// the bytes of `a` and of `b` that lie over it: `SDOT`, written out because
/// its intrinsic is not yet in a stable Rust.
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

/// Four input values, repeated across four rows.
#[target_feature(enable = "neon")]
unsafe fn four(act: *const i8, at: usize) -> int8x16_t {
    vreinterpretq_s8_s32(vld1q_dup_s32(act.add(at) as *const i32))
}

/// Sixteen rows' unsigned bytes as whole numbers, four rows to a vector.
#[target_feature(enable = "neon")]
unsafe fn widened16(bytes: *const u8) -> [int32x4_t; 4] {
    let all = vld1q_u8(bytes);
    let (low, high) = (vmovl_u8(vget_low_u8(all)), vmovl_u8(vget_high_u8(all)));
    [
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(low))),
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(low))),
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(high))),
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(high))),
    ]
}

/// Sixteen rows' signed bytes as whole numbers, four rows to a vector.
#[target_feature(enable = "neon")]
unsafe fn widened16_signed(bytes: *const i8) -> [int32x4_t; 4] {
    let all = vld1q_s8(bytes);
    let (low, high) = (vmovl_s8(vget_low_s8(all)), vmovl_s8(vget_high_s8(all)));
    [
        vmovl_s16(vget_low_s16(low)),
        vmovl_s16(vget_high_s16(low)),
        vmovl_s16(vget_low_s16(high)),
        vmovl_s16(vget_high_s16(high)),
    ]
}

/// Sixteen rows' products with one block of 256 input values, added to
/// `acc`. A weight is `d * scale * q - dmin * min`, so a row's product is
/// each run of 32's `q · input` by its scale, less its min by the input's
/// own sum over the run: whole numbers until the block's last step, where
/// the two totals meet the row's two scales.
///
/// Four rows at a time, each run of 32 with a sum for its low values and one
/// for its high, so eight sums are on the go and none waits on its own.
#[target_feature(enable = "neon,dotprod")]
unsafe fn q4k_block(tile: &TileQ4K, act: *const i8, d_act: f32, sums: *const i32, acc: F16) -> F16 {
    let mask = vdupq_n_u8(0x0F);
    let qs = tile.qs.as_ptr();
    let mut scaled = [vdupq_n_s32(0); 4];
    for s in 0..8 {
        let (mut low, mut high) = ([vdupq_n_s32(0); 4], [vdupq_n_s32(0); 4]);
        for g in 0..4 {
            let group = qs.add((g * 8 + s) * ROWS * 4);
            let (first, second) = (four(act, s * 32 + g * 8), four(act, s * 32 + g * 8 + 4));
            for k in 0..4 {
                let w = vld1q_u8(group.add(k * 16));
                low[k] = dot(low[k], vreinterpretq_s8_u8(vandq_u8(w, mask)), first);
                high[k] = dot(high[k], vreinterpretq_s8_u8(vshrq_n_u8::<4>(w)), second);
            }
        }
        let scale = widened16(tile.scales[s].as_ptr());
        for k in 0..4 {
            scaled[k] = vmlaq_s32(scaled[k], vaddq_s32(low[k], high[k]), scale[k]);
        }
    }
    // Each row's mins by the input's sums, two runs to a step: a row's two
    // mins side by side in 32 bits.
    let mut mins = [vdupq_n_s32(0); 4];
    let low16 = vdupq_n_u32(0xFFFF);
    for (pair, both) in tile.mins.iter().enumerate() {
        let first = vdupq_n_s32(*sums.add(4 * pair) + *sums.add(4 * pair + 1));
        let second = vdupq_n_s32(*sums.add(4 * pair + 2) + *sums.add(4 * pair + 3));
        for (k, mins) in mins.iter_mut().enumerate() {
            let two = vreinterpretq_u32_u16(vmovl_u8(vld1_u8(both.as_ptr().add(k * 8))));
            *mins = vmlaq_s32(*mins, vreinterpretq_s32_u32(vandq_u32(two, low16)), first);
            *mins = vmlaq_s32(*mins, vreinterpretq_s32_u32(vshrq_n_u32::<16>(two)), second);
        }
    }
    let d = scales_by(&tile.d, d_act);
    let dmin = scales_by(&tile.dmin, d_act);
    let mut out = acc;
    for k in 0..4 {
        let with = vfmaq_f32(out[k], vcvtq_f32_s32(scaled[k]), d[k]);
        out[k] = vfmsq_f32(with, vcvtq_f32_s32(mins[k]), dmin[k]);
    }
    out
}

/// [`q4k_block`] for two rows of input at once, each row's products added to
/// its own of `acc`: to the bit what the block gives each row alone. A
/// prompt's rows are bound by this arithmetic, not by memory, and splitting
/// each byte of the tile into its two values is the same for every row: here
/// it is done once for the two.
#[target_feature(enable = "neon,dotprod")]
unsafe fn q4k_block2(tile: &TileQ4K, rows: [&Row; 2], block: usize, acc: [F16; 2]) -> [F16; 2] {
    let mask = vdupq_n_u8(0x0F);
    let qs = tile.qs.as_ptr();
    let act = rows.map(|row| row.qs.as_ptr().add(block * QK_K));
    let mut scaled = [[vdupq_n_s32(0); 4]; 2];
    for s in 0..8 {
        let mut low = [[vdupq_n_s32(0); 4]; 2];
        let mut high = [[vdupq_n_s32(0); 4]; 2];
        for g in 0..4 {
            let group = qs.add((g * 8 + s) * ROWS * 4);
            let first = act.map(|act| four(act, s * 32 + g * 8));
            let second = act.map(|act| four(act, s * 32 + g * 8 + 4));
            for k in 0..4 {
                let w = vld1q_u8(group.add(k * 16));
                let (l, h) = (
                    vreinterpretq_s8_u8(vandq_u8(w, mask)),
                    vreinterpretq_s8_u8(vshrq_n_u8::<4>(w)),
                );
                for n in 0..2 {
                    low[n][k] = dot(low[n][k], l, first[n]);
                    high[n][k] = dot(high[n][k], h, second[n]);
                }
            }
        }
        let scale = widened16(tile.scales[s].as_ptr());
        for n in 0..2 {
            for k in 0..4 {
                scaled[n][k] = vmlaq_s32(scaled[n][k], vaddq_s32(low[n][k], high[n][k]), scale[k]);
            }
        }
    }
    let low16 = vdupq_n_u32(0xFFFF);
    let mut out = acc;
    for n in 0..2 {
        let sums = rows[n].sums.as_ptr().add(block * 16);
        let mut mins = [vdupq_n_s32(0); 4];
        for (pair, both) in tile.mins.iter().enumerate() {
            let first = vdupq_n_s32(*sums.add(4 * pair) + *sums.add(4 * pair + 1));
            let second = vdupq_n_s32(*sums.add(4 * pair + 2) + *sums.add(4 * pair + 3));
            for (k, mins) in mins.iter_mut().enumerate() {
                let two = vreinterpretq_u32_u16(vmovl_u8(vld1_u8(both.as_ptr().add(k * 8))));
                *mins = vmlaq_s32(*mins, vreinterpretq_s32_u32(vandq_u32(two, low16)), first);
                *mins = vmlaq_s32(*mins, vreinterpretq_s32_u32(vshrq_n_u32::<16>(two)), second);
            }
        }
        let by = *rows[n].d.get_unchecked(block);
        let d = scales_by(&tile.d, by);
        let dmin = scales_by(&tile.dmin, by);
        for k in 0..4 {
            let with = vfmaq_f32(out[n][k], vcvtq_f32_s32(scaled[n][k]), d[k]);
            out[n][k] = vfmsq_f32(with, vcvtq_f32_s32(mins[k]), dmin[k]);
        }
    }
    out
}

/// Which of a quarter's twelve bytes each of its sixteen fields is cut from:
/// a row's three bytes as `[b0, b0, b1, b2]`, and beside them `[-, b1, b2,
/// -]`, a field straddling two bytes taking its low bits from the first and
/// its high bits from the second.
const FROM: [u8; 16] = [0, 0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10, 11];
const AND_FROM: [u8; 16] = [
    0xFF, 1, 2, 0xFF, 0xFF, 4, 5, 0xFF, 0xFF, 7, 8, 0xFF, 0xFF, 10, 11, 0xFF,
];
/// How far each byte of the two moves: right for the first, left for the
/// second.
const DOWN: [i8; 16] = [0, -6, -4, -2, 0, -6, -4, -2, 0, -6, -4, -2, 0, -6, -4, -2];
const UP: [i8; 16] = [0, 2, 4, 0, 0, 2, 4, 0, 0, 2, 4, 0, 0, 2, 4, 0];

/// [`FROM`] and [`AND_FROM`] for twelve bytes that are a load's last twelve.
const FROM_BACK: [u8; 16] = [4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11, 12, 13, 13, 14, 15];
const AND_FROM_BACK: [u8; 16] = [
    0xFF, 5, 6, 0xFF, 0xFF, 8, 9, 0xFF, 0xFF, 11, 12, 0xFF, 0xFF, 14, 15, 0xFF,
];

/// The tables and shifts that cut a quarter's six-bit values out.
struct Cut {
    from: [uint8x16_t; 2],
    and_from: [uint8x16_t; 2],
    down: int8x16_t,
    up: int8x16_t,
    six: uint8x16_t,
}

impl Cut {
    #[target_feature(enable = "neon")]
    unsafe fn new() -> Self {
        Self {
            from: [vld1q_u8(FROM.as_ptr()), vld1q_u8(FROM_BACK.as_ptr())],
            and_from: [
                vld1q_u8(AND_FROM.as_ptr()),
                vld1q_u8(AND_FROM_BACK.as_ptr()),
            ],
            down: vld1q_s8(DOWN.as_ptr()),
            up: vld1q_s8(UP.as_ptr()),
            six: vdupq_n_u8(0x3F),
        }
    }

    /// Four rows' four six-bit values each, from quarter `k` of the group at
    /// `bytes`: twelve bytes, read as sixteen. The last quarter of a group
    /// is read from four bytes back, so no load runs past the tile.
    #[target_feature(enable = "neon")]
    unsafe fn sixes(&self, bytes: *const u8, k: usize) -> int8x16_t {
        let (loaded, back) = if k == 3 {
            (vld1q_u8(bytes.add(32)), 1)
        } else {
            (vld1q_u8(bytes.add(k * 12)), 0)
        };
        let low = vshlq_u8(vqtbl1q_u8(loaded, self.from[back]), self.down);
        let high = vshlq_u8(vqtbl1q_u8(loaded, self.and_from[back]), self.up);
        vreinterpretq_s8_u8(vandq_u8(vorrq_u8(low, high), self.six))
    }
}

/// Sixteen rows' products with one block of 256 input values, added to
/// `acc`. A weight is `d * scale * (q - 32)`, so a row's product is each
/// run of 16's `q · input`, less 32 times the input's own sum over the run,
/// by its scale: whole numbers until the block's last step.
#[target_feature(enable = "neon,dotprod")]
unsafe fn q6k_block(tile: &TileQ6, act: *const i8, d_act: f32, sums: *const i32, acc: F16) -> F16 {
    let cut = Cut::new();
    let qs = tile.qs.as_ptr();
    let mut total = [vdupq_n_s32(0); 4];
    for run in 0..16 {
        let mut isum = [vdupq_n_s32(0); 4];
        for g in 0..4 {
            let group = qs.add((run * 4 + g) * ROWS * 3);
            let x = four(act, run * 16 + g * 4);
            for (k, isum) in isum.iter_mut().enumerate() {
                *isum = dot(*isum, cut.sixes(group, k), x);
            }
        }
        let off = vdupq_n_s32(32 * *sums.add(run));
        let scale = widened16_signed(tile.scales[run].as_ptr());
        for k in 0..4 {
            total[k] = vmlaq_s32(total[k], vsubq_s32(isum[k], off), scale[k]);
        }
    }
    let d = scales_by(&tile.d, d_act);
    let mut out = acc;
    for k in 0..4 {
        out[k] = vfmaq_f32(out[k], vcvtq_f32_s32(total[k]), d[k]);
    }
    out
}

/// [`q6k_block`] for four rows of input at once, each row's products added
/// to its own of `acc`: to the bit what the block gives each row alone.
/// Cutting a tile's six-bit values out of its bytes is as much work as
/// multiplying them, and it is the same for every row: here it is done once
/// for the four.
#[target_feature(enable = "neon,dotprod")]
unsafe fn q6k_block4(tile: &TileQ6, rows: [&Row; 4], block: usize, acc: [F16; 4]) -> [F16; 4] {
    let cut = Cut::new();
    let qs = tile.qs.as_ptr();
    let act = rows.map(|row| row.qs.as_ptr().add(block * QK_K));
    let sums = rows.map(|row| row.sums.as_ptr().add(block * 16));
    let mut total = [[vdupq_n_s32(0); 4]; 4];
    for run in 0..16 {
        let mut isum = [[vdupq_n_s32(0); 4]; 4];
        for g in 0..4 {
            let group = qs.add((run * 4 + g) * ROWS * 3);
            let x = act.map(|act| four(act, run * 16 + g * 4));
            for k in 0..4 {
                let q = cut.sixes(group, k);
                for (isum, x) in isum.iter_mut().zip(x) {
                    isum[k] = dot(isum[k], q, x);
                }
            }
        }
        let scale = widened16_signed(tile.scales[run].as_ptr());
        for n in 0..4 {
            let off = vdupq_n_s32(32 * *sums[n].add(run));
            for k in 0..4 {
                total[n][k] = vmlaq_s32(total[n][k], vsubq_s32(isum[n][k], off), scale[k]);
            }
        }
    }
    let mut out = acc;
    for n in 0..4 {
        let d = scales_by(&tile.d, *rows[n].d.get_unchecked(block));
        for k in 0..4 {
            out[n][k] = vfmaq_f32(out[n][k], vcvtq_f32_s32(total[n][k]), d[k]);
        }
    }
    out
}

/// Sixteen rows' products with one block of 32 input values, added to
/// `acc`. The weights are stored 128 up; taking it back off each byte makes
/// them the signed values they are, so the product needs no correction by
/// the input's sum.
#[target_feature(enable = "neon,dotprod")]
unsafe fn q8_block(tile: &TileQ8, act: *const i8, d_act: f32, acc: F16) -> F16 {
    let qs = tile.qs.as_ptr();
    let up = vdupq_n_u8(0x80);
    let mut isum = [vdupq_n_s32(0); 4];
    for g in 0..8 {
        let x = four(act, g * 4);
        for (k, isum) in isum.iter_mut().enumerate() {
            let w = vld1q_u8(qs.add((g * ROWS + k * 4) * 4));
            *isum = dot(*isum, vreinterpretq_s8_u8(veorq_u8(w, up)), x);
        }
    }
    let d = scales_by(&tile.d, d_act);
    let mut out = acc;
    for k in 0..4 {
        out[k] = vfmaq_f32(out[k], vcvtq_f32_s32(isum[k]), d[k]);
    }
    out
}

/// Rows whose running sums are kept at once while a sixteen's tiles go by.
const TOGETHER: usize = 32;

/// One block `b` of sixteen `t` of `w` against `row`, added to `acc`.
#[target_feature(enable = "neon,dotprod")]
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

#[target_feature(enable = "neon")]
unsafe fn store(at: *mut f32, sixteen: F16) {
    for (k, four) in sixteen.iter().enumerate() {
        vst1q_f32(at.add(k * 4), *four);
    }
}

/// The sixteens `first..last` of `w` against every row, written to `out`.
/// A sixteen's tiles go by once, each against every row (no more than
/// [`TOGETHER`] at once) before the next: a wide weight's sixteen is more
/// tiles than the nearest cache holds.
///
/// SAFETY (for the caller): the CPU has the features named, `out` has room
/// for `rows.len()` rows of `w.rows` values, and no other thread writes
/// these sixteens.
#[target_feature(enable = "neon,dotprod")]
unsafe fn sixteens_of(w: &Tiled, first: usize, last: usize, rows: &[Row], out: &Out) {
    let per_row = blocks(w);
    for t in first..last {
        if let [row] = rows {
            let mut acc = zeros();
            for b in 0..per_row {
                acc = block_of(w, t, b, row, acc);
            }
            store(out.at.add(t * ROWS), acc);
            continue;
        }
        for (run, some) in rows.chunks(TOGETHER).enumerate() {
            let mut acc = [zeros(); TOGETHER];
            for b in 0..per_row {
                // The rows that share a tile's unpacking go together, and
                // what is left of them a row at a time.
                let mut i = 0;
                match &w.tiles {
                    Tiles::Q4K(tiles) => {
                        let tile = tiles.get_unchecked(t * per_row + b);
                        while i + 2 <= some.len() {
                            let two =
                                q4k_block2(tile, [&some[i], &some[i + 1]], b, [acc[i], acc[i + 1]]);
                            acc[i..i + 2].copy_from_slice(&two);
                            i += 2;
                        }
                    }
                    Tiles::Q6K(tiles) => {
                        let tile = tiles.get_unchecked(t * per_row + b);
                        while i + 4 <= some.len() {
                            let four = q6k_block4(
                                tile,
                                [&some[i], &some[i + 1], &some[i + 2], &some[i + 3]],
                                b,
                                [acc[i], acc[i + 1], acc[i + 2], acc[i + 3]],
                            );
                            acc[i..i + 4].copy_from_slice(&four);
                            i += 4;
                        }
                    }
                    Tiles::Q8_0(_) => {}
                }
                for (acc, row) in acc[i..].iter_mut().zip(&some[i..]) {
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
#[target_feature(enable = "neon,dotprod")]
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
