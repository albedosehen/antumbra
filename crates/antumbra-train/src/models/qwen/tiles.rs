//! A served base's quantized weights as Antumbra multiplies them on a CPU
//! with AVX-512 VNNI, or with NEON's dot product: sixteen rows to a tile,
//! each SIMD lane a row.
//!
//! candle multiplies a quantized weight through tiles of its own, and on a
//! decoded token three things about that cost more than the arithmetic. A
//! Q4_K tile keeps its scales and mins as f32 already multiplied through,
//! 1024 bytes of a 3072-byte tile, so a token reads a third more than the
//! 2304 bytes the plain blocks take, and a decoded token is bound by how
//! fast the weights are read. Each projection quantizes its input again,
//! where q, k and v read one input and so do gate and up. And each is its
//! own dispatch to the thread pool. candle also keeps the plain blocks beside
//! its tiles, two copies of every weight.
//!
//! Here a Q4_K tile keeps the scales and mins as the bytes they are, beside
//! the rows' two f16 scales: 2368 bytes. The input is quantized once for
//! every weight that reads it, and those weights are one dispatch. The tiles
//! are the only copy: a weight is quantized and tiled a run of rows at a
//! time on every core, the plain blocks of each run let go as it is packed,
//! and a prompt's rows go through the same tiles.
//!
//! Each form a base is served in has a tile: Q4_K, Q8_0, and Q6_K, which is
//! also the head's form over a Q4_K base, at the size of its plain blocks.
//! Six-bit values are cut out of their bytes with AVX-512 VBMI, which the
//! first CPUs with VNNI lack; there, as on a CPU with AVX2 alone, they are
//! packed the other way round and cut out with two multiplies.
//!
//! A tile holds what the plain blocks hold, each SIMD lane a row, in the
//! order its kernel reads it, and the input is quantized the way candle
//! quantizes it. A row's product is summed in whole numbers until a block's
//! last step, so it is the product candle computes up to the order of a few
//! float sums. The kernels below are AVX-512's; `tiles_neon.rs` holds
//! aarch64's, four rows to a vector over the same tiles. On a CPU with
//! neither nothing is tiled and candle multiplies as before.

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{bail, DType, Device, Result, Storage, Tensor};

use super::slices::values;

/// Rows to a tile.
pub(super) const ROWS: usize = 16;
/// Weights to a Q4_K block, and to a block of quantized input.
const QK_K: usize = 256;
/// Weights to a Q8_0 block.
const QK8_0: usize = 32;
const Q4K_BLOCK: usize = 144;
const Q6K_BLOCK: usize = 210;
const Q8_0_BLOCK: usize = 34;

/// One Q4_K block of 256 weights for sixteen rows.
#[repr(C)]
#[derive(Clone)]
struct TileQ4K {
    /// Each row's scale and min scale, as f16 bits.
    d: [u16; ROWS],
    dmin: [u16; ROWS],
    /// Each run of 32's 6-bit scale, a row a byte.
    scales: [[u8; ROWS]; 8],
    /// Each run of 32's 6-bit min, two runs side by side:
    /// `[pair of runs (4)][row (16)][2 bytes]`.
    mins: [[u8; ROWS * 2]; 4],
    /// `[step (4)][run of 32 (8)][row (16)][4 bytes]`: at step `g` of run
    /// `s`, byte `j` holds value `32s + 8g + j` in its low nibble and
    /// `32s + 8g + j + 4` in its high one. A step of every run, then the next
    /// step of every run: the order the kernel reads them in, so a tile is
    /// read from end to end.
    qs: [u8; QK_K / 8 * ROWS * 4],
}

/// One Q8_0 block of 32 weights for sixteen rows.
#[repr(C)]
#[derive(Clone)]
struct TileQ8 {
    /// Each row's scale, as f16 bits.
    d: [u16; ROWS],
    /// `[group of 4 values (8)][row (16)][4 bytes]`, each value plus 128 so
    /// it is unsigned for the dot-product step.
    qs: [u8; QK8_0 * ROWS],
}

/// One Q6_K block of 256 weights for sixteen rows: as many bytes as the plain
/// blocks take.
#[repr(C)]
#[derive(Clone)]
struct TileQ6 {
    /// Each row's scale, as f16 bits.
    d: [u16; ROWS],
    /// Each run of 16's scale, a row a byte.
    scales: [[i8; ROWS]; 16],
    /// `[group of 4 values (64)][row (16)][3 bytes]`: a row's four 6-bit
    /// values in 24 bits, the first of them lowest or highest as the kernel
    /// that reads them cuts them out ([`Sixes`]). Each value is 32 more than
    /// the weight's, so it is unsigned for the dot-product step.
    qs: [u8; QK_K / 4 * ROWS * 3],
}

/// Which way round a row's four six-bit values lie in their three bytes of
/// a tile. The tiles are packed on the CPU that multiplies them, so they are
/// packed for the instructions it cuts them out with.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Sixes {
    /// The first value lowest in the 24 bits: what AVX-512 VBMI's shifts and
    /// NEON's tables cut out.
    FirstLowest,
    /// The first value highest, the bytes in that order: what AVX2 cuts out
    /// with two multiplies (`tiles_avx2.rs`).
    FirstHighest,
}

/// The way round this CPU's six-bit tiles are packed.
fn sixes() -> Sixes {
    #[cfg(target_arch = "x86_64")]
    if !wide_six_bit() {
        return Sixes::FirstHighest;
    }
    Sixes::FirstLowest
}

const _: () = assert!(std::mem::size_of::<TileQ6>() == 3360);
const _: () = assert!(std::mem::size_of::<TileQ4K>() == 2368);
const _: () = assert!(std::mem::size_of::<TileQ8>() == 544);

// Where nothing is tiled, nothing reads a tile.
#[cfg_attr(
    not(any(target_arch = "x86_64", target_arch = "aarch64")),
    allow(dead_code)
)]
enum Tiles {
    /// `[sixteen rows][block of 256]`.
    Q4K(Vec<TileQ4K>),
    /// `[sixteen rows][block of 256]`.
    Q6K(Vec<TileQ6>),
    /// `[sixteen rows][block of 32]`.
    Q8_0(Vec<TileQ8>),
}

/// One quantized weight, tiled: `rows` by `cols`.
pub(super) struct Tiled {
    tiles: Tiles,
    rows: usize,
    cols: usize,
}

/// Whether this CPU can multiply through the tiles.
pub(super) fn available() -> bool {
    crate::device::tiles()
}

/// Whether this CPU can multiply through six-bit tiles as well.
pub(super) fn six_bit() -> bool {
    crate::device::six_bit_tiles()
}

/// Whether this CPU multiplies through the AVX-512 kernels, and so takes the
/// plain-row steps of `wide` in vectors of sixteen as well: the other x86-64
/// CPUs that tile multiply through AVX2 and take those steps in vectors of
/// eight (`wide::avx2`).
#[cfg(target_arch = "x86_64")]
pub(super) fn wide() -> bool {
    crate::device::wide_tiles()
}

/// Whether this CPU multiplies six-bit tiles, and the head's sketch, through
/// AVX-512: with VBMI, which the first CPUs with VNNI lack.
#[cfg(target_arch = "x86_64")]
pub(super) fn wide_six_bit() -> bool {
    crate::device::wide_tiles() && is_x86_feature_detected!("avx512vbmi")
}

/// The 4-bit value `v` of a plain Q4_K block's 128 bytes: the low nibbles of
/// each 32 bytes are one run of 32 values and the high nibbles the next.
fn nibble(qs: &[u8], v: usize) -> u8 {
    let (chunk, within) = (v / 64, v % 64);
    if within < 32 {
        qs[32 * chunk + within] & 0xF
    } else {
        qs[32 * chunk + within - 32] >> 4
    }
}

const EMPTY_Q4K: TileQ4K = TileQ4K {
    d: [0; ROWS],
    dmin: [0; ROWS],
    scales: [[0; ROWS]; 8],
    mins: [[0; ROWS * 2]; 4],
    qs: [0; QK_K / 8 * ROWS * 4],
};

const EMPTY_Q6: TileQ6 = TileQ6 {
    d: [0; ROWS],
    scales: [[0; ROWS]; 16],
    qs: [0; QK_K / 4 * ROWS * 3],
};

/// The 6-bit value `i` of a plain Q6_K block: its low four bits from the
/// 128 bytes of `ql`, its high two from the 64 of `qh`. Each half of the
/// block is four runs of 32: the low nibbles of its first 32 bytes of `ql`,
/// the low nibbles of its next 32, then the high nibbles of each; a byte of
/// `qh` holds the high bits of one value from each run.
fn six(ql: &[u8], qh: &[u8], i: usize) -> u8 {
    let (half, within) = (i / 128, i % 128);
    let (run, l) = (within / 32, within % 32);
    let low = ql[half * 64 + l + 32 * (run & 1)];
    let low = if run < 2 { low & 0xF } else { low >> 4 };
    let high = (qh[half * 32 + l] >> (2 * run)) & 3;
    low | (high << 4)
}

/// Tile the plain Q6_K blocks of `tiles.len() / kb` sixteens of rows, `kb`
/// blocks a row.
fn pack_q6k(blocks: &[u8], kb: usize, tiles: &mut [TileQ6]) {
    let order = sixes();
    for (t, tile) in tiles.iter_mut().enumerate() {
        let (first_row, b) = (t / kb * ROWS, t % kb);
        for r in 0..ROWS {
            let blk = &blocks[((first_row + r) * kb + b) * Q6K_BLOCK..][..Q6K_BLOCK];
            let (ql, qh, scales) = (&blk[..128], &blk[128..192], &blk[192..208]);
            tile.d[r] = u16::from_le_bytes([blk[208], blk[209]]);
            for (s, scale) in scales.iter().enumerate() {
                tile.scales[s][r] = *scale as i8;
            }
            for g in 0..QK_K / 4 {
                let value = |j: usize| u32::from(six(ql, qh, 4 * g + j));
                let bytes = match order {
                    Sixes::FirstLowest => {
                        let four = (0..4).fold(0u32, |all, j| all | (value(j) << (6 * j)));
                        let [low, middle, high, _] = four.to_le_bytes();
                        [low, middle, high]
                    }
                    Sixes::FirstHighest => {
                        let four = (0..4).fold(0u32, |all, j| (all << 6) | value(j));
                        let [_, high, middle, low] = four.to_be_bytes();
                        [high, middle, low]
                    }
                };
                tile.qs[(g * ROWS + r) * 3..(g * ROWS + r) * 3 + 3].copy_from_slice(&bytes);
            }
        }
    }
}

const EMPTY_Q8: TileQ8 = TileQ8 {
    d: [0; ROWS],
    qs: [0; QK8_0 * ROWS],
};

/// Tile the plain Q4_K blocks of `tiles.len() / kb` sixteens of rows, `kb`
/// blocks a row.
fn pack_q4k(blocks: &[u8], kb: usize, tiles: &mut [TileQ4K]) {
    for (t, tile) in tiles.iter_mut().enumerate() {
        let (first_row, b) = (t / kb * ROWS, t % kb);
        for r in 0..ROWS {
            let blk = &blocks[((first_row + r) * kb + b) * Q4K_BLOCK..][..Q4K_BLOCK];
            tile.d[r] = u16::from_le_bytes([blk[0], blk[1]]);
            tile.dmin[r] = u16::from_le_bytes([blk[2], blk[3]]);
            // Eight 6-bit scales and eight 6-bit mins in twelve bytes.
            let sc = &blk[4..16];
            for s in 0..8 {
                let (scale, min) = if s < 4 {
                    (sc[s] & 63, sc[s + 4] & 63)
                } else {
                    (
                        (sc[s + 4] & 0xF) | ((sc[s - 4] >> 6) << 4),
                        (sc[s + 4] >> 4) | ((sc[s] >> 6) << 4),
                    )
                };
                tile.scales[s][r] = scale;
                tile.mins[s / 2][r * 2 + s % 2] = min;
            }
            let qs = &blk[16..];
            for (s, g) in (0..8).flat_map(|s| (0..4).map(move |g| (s, g))) {
                let first = 32 * s + 8 * g;
                for j in 0..4 {
                    tile.qs[((g * 8 + s) * ROWS + r) * 4 + j] =
                        nibble(qs, first + j) | (nibble(qs, first + j + 4) << 4);
                }
            }
        }
    }
}

/// Tile the plain Q8_0 blocks of `tiles.len() / nb` sixteens of rows, `nb`
/// blocks a row.
fn pack_q8_0(blocks: &[u8], nb: usize, tiles: &mut [TileQ8]) {
    for (t, tile) in tiles.iter_mut().enumerate() {
        let (first_row, b) = (t / nb * ROWS, t % nb);
        for r in 0..ROWS {
            let blk = &blocks[((first_row + r) * nb + b) * Q8_0_BLOCK..][..Q8_0_BLOCK];
            tile.d[r] = u16::from_le_bytes([blk[0], blk[1]]);
            for g in 0..QK8_0 / 4 {
                for j in 0..4 {
                    // An i8 moved up by 128: its bits with the top one turned.
                    tile.qs[(g * ROWS + r) * 4 + j] = blk[2 + g * 4 + j] ^ 0x80;
                }
            }
        }
    }
}

/// Whether a weight of `rows` by `cols` can be multiplied through tiles here:
/// its rows come in sixteens and its columns in blocks of 256, on a CPU that
/// has the instructions.
fn fits(rows: usize, cols: usize) -> bool {
    available() && rows != 0 && rows.is_multiple_of(ROWS) && cols != 0 && cols.is_multiple_of(QK_K)
}

/// Quantize `w` to `dtype` and tile it into `tiles` (`per_row` tiles for each
/// sixteen rows), a run of sixteens on each thread. A block is quantized from
/// its own values and no others, so the rows quantized in runs are the rows
/// quantized whole, and the weights are tiled by as many cores as there are:
/// on one, loading a 1.5B base to serve took a minute of quantizing.
fn quantize_in_runs<T: Send>(
    w: &Tensor,
    dtype: GgmlDType,
    block_bytes: usize,
    per_row: usize,
    tiles: &mut [T],
    pack: impl Fn(&[u8], &mut [T]) + Sync,
) -> Result<()> {
    let (rows, cols) = w.dims2()?;
    let w = w.to_dtype(DType::F32)?.contiguous()?;
    let (storage, layout) = w.storage_and_layout();
    let Storage::Cpu(storage) = &*storage else {
        bail!("a weight is quantized from host memory")
    };
    let values = values(storage, layout, "a weight to quantize")?;
    let threads = std::thread::available_parallelism().map_or(1, usize::from);
    // Rows a thread takes: sixteens, about as many for each.
    let per = rows.div_ceil(threads).div_ceil(ROWS) * ROWS;
    std::thread::scope(|scope| {
        let handles: Vec<_> = tiles
            .chunks_mut(per / ROWS * per_row)
            .enumerate()
            .map(|(i, part)| {
                let pack = &pack;
                scope.spawn(move || -> Result<()> {
                    let n = part.len() / per_row * ROWS;
                    // A tensor of its own: candle quantizes a view's
                    // storage from its start, whatever rows the view names.
                    let run = &values[i * per * cols..(i * per + n) * cols];
                    let run = Tensor::from_slice(run, (n, cols), &Device::Cpu)?;
                    let quantized = QTensor::quantize(&run, dtype)?;
                    let blocks = quantized.data()?;
                    if blocks.len() != n * per_row * block_bytes {
                        bail!(
                            "{n} rows of {dtype:?} came to {} bytes, not {per_row} blocks of {block_bytes} a row",
                            blocks.len()
                        )
                    }
                    pack(&blocks, part);
                    Ok(())
                })
            })
            .collect();
        for handle in handles {
            match handle.join() {
                Ok(done) => done?,
                Err(_) => bail!("a thread quantizing a weight panicked"),
            }
        }
        Ok(())
    })
}

impl Tiled {
    /// Tile `q`, when it can be multiplied through tiles here: a Q4_K or Q8_0
    /// weight in host memory whose rows come in sixteens and whose columns
    /// in blocks of 256, on a CPU that has the instructions. `None`
    /// otherwise. A weight quantized whole and then tiled: what
    /// [`Tiled::quantize`] is tested against.
    #[cfg(test)]
    pub(super) fn pack(q: &QTensor) -> Result<Option<Self>> {
        let (rows, cols) = match q.shape().dims() {
            [rows, cols] => (*rows, *cols),
            _ => return Ok(None),
        };
        if !fits(rows, cols) || !q.device().is_cpu() {
            return Ok(None);
        }
        let blocks = q.data()?;
        let tiles = match q.dtype() {
            GgmlDType::Q4K if blocks.len() == rows * (cols / QK_K) * Q4K_BLOCK => {
                let kb = cols / QK_K;
                let mut tiles = vec![EMPTY_Q4K; rows / ROWS * kb];
                pack_q4k(&blocks, kb, &mut tiles);
                Tiles::Q4K(tiles)
            }
            GgmlDType::Q6K if six_bit() && blocks.len() == rows * (cols / QK_K) * Q6K_BLOCK => {
                let kb = cols / QK_K;
                let mut tiles = vec![EMPTY_Q6; rows / ROWS * kb];
                pack_q6k(&blocks, kb, &mut tiles);
                Tiles::Q6K(tiles)
            }
            GgmlDType::Q8_0 if blocks.len() == rows * (cols / QK8_0) * Q8_0_BLOCK => {
                let nb = cols / QK8_0;
                let mut tiles = vec![EMPTY_Q8; rows / ROWS * nb];
                pack_q8_0(&blocks, nb, &mut tiles);
                Tiles::Q8_0(tiles)
            }
            _ => return Ok(None),
        };
        Ok(Some(Self { tiles, rows, cols }))
    }

    /// Quantize the dense `w` to `dtype` and tile it, on every core, when it
    /// can be multiplied through tiles here: what [`Tiled::pack`] makes of
    /// `w` quantized whole, without the plain blocks ever being held whole.
    /// `None` under the same conditions, and for a weight not in host memory.
    pub(super) fn quantize(w: &Tensor, dtype: GgmlDType) -> Result<Option<Self>> {
        let (rows, cols) = match w.dims() {
            [rows, cols] => (*rows, *cols),
            _ => return Ok(None),
        };
        if !fits(rows, cols) || !w.device().is_cpu() {
            return Ok(None);
        }
        let tiles = match dtype {
            GgmlDType::Q4K => {
                let kb = cols / QK_K;
                let mut tiles = vec![EMPTY_Q4K; rows / ROWS * kb];
                quantize_in_runs(w, dtype, Q4K_BLOCK, kb, &mut tiles, |blocks, part| {
                    pack_q4k(blocks, kb, part)
                })?;
                Tiles::Q4K(tiles)
            }
            GgmlDType::Q6K if six_bit() => {
                let kb = cols / QK_K;
                let mut tiles = vec![EMPTY_Q6; rows / ROWS * kb];
                quantize_in_runs(w, dtype, Q6K_BLOCK, kb, &mut tiles, |blocks, part| {
                    pack_q6k(blocks, kb, part)
                })?;
                Tiles::Q6K(tiles)
            }
            GgmlDType::Q8_0 => {
                let nb = cols / QK8_0;
                let mut tiles = vec![EMPTY_Q8; rows / ROWS * nb];
                quantize_in_runs(w, dtype, Q8_0_BLOCK, nb, &mut tiles, |blocks, part| {
                    pack_q8_0(blocks, nb, part)
                })?;
                Tiles::Q8_0(tiles)
            }
            _ => return Ok(None),
        };
        Ok(Some(Self { tiles, rows, cols }))
    }

    pub(super) fn rows(&self) -> usize {
        self.rows
    }

    pub(super) fn cols(&self) -> usize {
        self.cols
    }

    /// Sixteens of rows: the units a product is shared out in.
    pub(super) fn sixteens(&self) -> usize {
        self.rows / ROWS
    }

    /// Bytes of tiles one sixteen of rows is read from.
    pub(super) fn sixteen_bytes(&self) -> usize {
        match &self.tiles {
            Tiles::Q4K(_) => self.cols / QK_K * std::mem::size_of::<TileQ4K>(),
            Tiles::Q6K(_) => self.cols / QK_K * std::mem::size_of::<TileQ6>(),
            Tiles::Q8_0(_) => self.cols / QK8_0 * std::mem::size_of::<TileQ8>(),
        }
    }

    /// The sixteens `first..last` of this weight's rows times each of the
    /// rows `rows` were quantized from: row `i`'s products are written to
    /// `out[i * self.rows() + 16 * first..][..16 * (last - first)]`. A unit
    /// of work in a dispatch of the caller's.
    ///
    /// SAFETY (for the caller): each of `rows` holds this weight's `cols`
    /// values, `out` has room for its `rows` products for every one of
    /// them, the sixteens are among its own, and no other thread writes
    /// those products.
    pub(super) unsafe fn multiply_sixteens(
        &self,
        first: usize,
        last: usize,
        rows: &[Row],
        out: *mut f32,
    ) {
        debug_assert!(rows.iter().all(|r| r.qs.len() == self.cols) && last <= self.sixteens());
        kernels::sixteens(self, first, last, rows, out);
    }
}

/// Tile bytes a claim of a product's rows covers, about: enough that claiming
/// is nothing beside the multiplying, little enough that the last claims
/// even out.
pub(super) const CLAIM: usize = 48 << 10;

/// One row of input as the tiles read it: for each 256 values a scale, the
/// values as i8, and their sums 16 at a time. What candle quantizes a
/// product's input to (Q8_K), whatever the weight's form.
#[derive(Default)]
pub(super) struct Row {
    d: Vec<f32>,
    qs: Vec<i8>,
    sums: Vec<i32>,
}

impl Row {
    /// Room for `kb` blocks of 256 values. What it held is left: every block
    /// is written whole when it is quantized.
    fn sized(&mut self, kb: usize) {
        if self.d.len() != kb {
            self.d.resize(kb, 0.0);
            self.qs.resize(kb * QK_K, 0);
            self.sums.resize(kb * 16, 0);
        }
    }

    /// Whether this row was quantized from `cols` values.
    pub(super) fn holds(&self, cols: usize) -> bool {
        self.qs.len() == cols
    }

    /// Block `b` of the row as a kernel outside the tiles reads it: its 256
    /// values, its scale, and the sum of its values.
    // Off x86-64 there is no such kernel.
    #[cfg_attr(
        not(any(target_arch = "x86_64", target_arch = "aarch64")),
        allow(dead_code)
    )]
    pub(super) fn block(&self, b: usize) -> (&[i8], f32, i32) {
        (
            &self.qs[b * QK_K..(b + 1) * QK_K],
            self.d[b],
            self.sums[b * 16..(b + 1) * 16].iter().sum(),
        )
    }

    /// This row with room for `values` values, to be quantized a block of 256
    /// at a time by the units of a phase.
    pub(super) fn blocks(&mut self, values: usize) -> Blocks {
        let kb = values / QK_K;
        self.sized(kb);
        Blocks {
            d: self.d.as_mut_ptr(),
            qs: self.qs.as_mut_ptr(),
            sums: self.sums.as_mut_ptr(),
            kb,
        }
    }
}

/// A row being quantized a block at a time, each block by one unit of work.
pub(super) struct Blocks {
    d: *mut f32,
    qs: *mut i8,
    sums: *mut i32,
    kb: usize,
}

// SAFETY: a block is written by the one unit of work that quantizes it (see
// `Blocks::quantize`), and the row is read only once every block is written.
unsafe impl Sync for Blocks {}

impl Blocks {
    /// Quantize the 256 values `x` as block `block` of the row.
    ///
    /// SAFETY (for the caller): this CPU multiplies through tiles
    /// ([`available`]), the row this was made from is still borrowed and has
    /// this block, and no other thread writes the block.
    pub(super) unsafe fn quantize(&self, block: usize, x: &[f32; QK_K]) {
        debug_assert!(block < self.kb);
        kernels::quantize_block(
            x.as_ptr(),
            self.d.add(block),
            self.qs.add(block * QK_K),
            self.sums.add(block * 16),
        );
    }
}

fn quantize(x: &[f32]) -> Row {
    let mut row = Row::default();
    quantize_into(x, &mut row);
    row
}

/// Quantize `x` into `row`, whose room is kept from one row to the next: a
/// decoded token quantizes four rows a layer.
pub(super) fn quantize_into(x: &[f32], row: &mut Row) {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    if available() {
        // SAFETY: `available` found the features the kernel is built for.
        unsafe { kernels::quantize(x, row) };
        return;
    }
    *row = quantize_plainly(x);
}

/// Each weight in `group` times the one row `row` was quantized from, into
/// its own slice of `outs`: [`forward`] for one row that is already plain
/// values, with nothing made into a tensor.
pub(super) fn multiply(group: &[&Tiled], row: &Row, outs: &mut [&mut [f32]]) -> Result<()> {
    let fits = group.len() == outs.len()
        && group
            .iter()
            .zip(outs.iter())
            .all(|(w, out)| w.cols == row.qs.len() && w.rows == out.len());
    if !fits {
        bail!(
            "tiled weights: {} weights against a row of {} values and {} places for products",
            group.len(),
            row.qs.len(),
            outs.len()
        )
    }
    kernels::multiply(group, std::slice::from_ref(row), outs);
    Ok(())
}

/// [`quantize`] a value at a time: what the vectors are held to.
fn quantize_plainly(x: &[f32]) -> Row {
    let kb = x.len() / QK_K;
    let mut row = Row {
        d: vec![0.0; kb],
        qs: vec![0; kb * QK_K],
        sums: vec![0; kb * 16],
    };
    for i in 0..kb {
        let xs = &x[i * QK_K..(i + 1) * QK_K];
        // The value furthest from zero, with its sign.
        let (mut max, mut amax) = (0f32, 0f32);
        for &v in xs {
            if amax < v.abs() {
                amax = v.abs();
                max = v;
            }
        }
        if amax == 0.0 {
            continue;
        }
        let iscale = -127f32 / max;
        let qs = &mut row.qs[i * QK_K..(i + 1) * QK_K];
        for (q, v) in qs.iter_mut().zip(xs) {
            *q = (iscale * v).round().clamp(-128.0, 127.0) as i8;
        }
        for (s, sum) in row.sums[i * 16..(i + 1) * 16].iter_mut().enumerate() {
            *sum = qs[s * 16..(s + 1) * 16].iter().map(|q| i32::from(*q)).sum();
        }
        row.d[i] = 1.0 / iscale;
    }
    row
}

/// Every row of `xs` quantized: a decoded token's one row where it stands,
/// a prompt's spread over the thread pool.
fn quantize_rows(xs: &[f32], cols: usize) -> Vec<Row> {
    let n = xs.len() / cols;
    if n < 4 {
        return xs.chunks_exact(cols).map(quantize).collect();
    }
    let slots: Vec<std::sync::Mutex<Option<Row>>> =
        (0..n).map(|_| std::sync::Mutex::new(None)).collect();
    candle_core::utils::barrier_pool().execute_chunked(n, |range| {
        for i in range {
            let row = quantize(&xs[i * cols..(i + 1) * cols]);
            // Each slot is written by the one thread that took its row.
            *slots[i]
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(row);
        }
    });
    slots
        .into_iter()
        .zip(xs.chunks_exact(cols))
        .map(|(slot, x)| {
            slot.into_inner()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .unwrap_or_else(|| quantize(x))
        })
        .collect()
}

/// Each weight in `group` times the rows of `x`: for `x` of `(.., cols)`, one
/// `(.., rows)` for each weight. The rows are quantized once for all of
/// them, and all of them are one dispatch to the thread pool. No gradient: a
/// served base does not train.
pub(super) fn forward(group: &[&Tiled], x: &Tensor) -> Result<Vec<Tensor>> {
    let Some(cols) = group.first().map(|w| w.cols) else {
        return Ok(Vec::new());
    };
    if !x.device().is_cpu() || x.dtype() != DType::F32 {
        bail!("tiled weights multiply f32 rows in host memory")
    }
    let dims = x.dims().to_vec();
    if dims.last() != Some(&cols) || group.iter().any(|w| w.cols != cols) {
        bail!("tiled weights of {cols} columns against rows of {dims:?}")
    }
    let x = x.contiguous()?;
    let mut outs: Vec<Vec<f32>> = {
        let (storage, layout) = x.storage_and_layout();
        let Storage::Cpu(storage) = &*storage else {
            bail!("tiled weights multiply rows in host memory")
        };
        let xs = values(storage, layout, "tiled weights: the rows")?;
        let rows = quantize_rows(xs, cols);
        let mut outs: Vec<Vec<f32>> = group
            .iter()
            .map(|w| vec![0f32; rows.len() * w.rows])
            .collect();
        let mut places: Vec<&mut [f32]> = outs.iter_mut().map(Vec::as_mut_slice).collect();
        kernels::multiply(group, &rows, &mut places);
        outs
    };
    let mut shape = dims;
    group
        .iter()
        .zip(outs.drain(..))
        .map(|(w, out)| {
            if let Some(last) = shape.last_mut() {
                *last = w.rows;
            }
            Tensor::from_vec(out, shape.as_slice(), &Device::Cpu)
        })
        .collect()
}

#[cfg(target_arch = "aarch64")]
#[path = "tiles_neon.rs"]
mod kernels;

/// SCRATCH, not for merge: the kernels as the release branch has them, to
/// time beside these in one process.
#[cfg(all(test, target_arch = "aarch64"))]
#[allow(dead_code)]
#[path = "tiles_neon_old.rs"]
mod kernels_old;

/// SCRATCH: the same file a second time, so the probe says how far two
/// copies of one source lie apart, which is what a difference has to beat.
#[cfg(all(test, target_arch = "aarch64"))]
#[allow(dead_code)]
#[path = "tiles_neon_old.rs"]
mod kernels_old_again;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
mod kernels {
    use super::{Row, Tiled};

    /// Nothing is tiled on other CPUs, so nothing reaches this.
    pub(super) fn multiply(_: &[&Tiled], _: &[Row], _: &mut [&mut [f32]]) {}

    pub(super) unsafe fn sixteens(_: &Tiled, _: usize, _: usize, _: &[Row], _: *mut f32) {}

    pub(super) unsafe fn quantize_block(_: *const f32, _: *mut f32, _: *mut i8, _: *mut i32) {}
}

/// The x86-64 kernels: AVX-512's where the CPU has them for the form a
/// weight is in, and AVX2's otherwise.
#[cfg(target_arch = "x86_64")]
mod kernels {
    use super::{avx2, avx512, Row, Tiled, Tiles};

    /// Whether `w` is multiplied through AVX-512: six-bit tiles need VBMI as
    /// well, which the first CPUs with VNNI lack.
    fn wide(w: &Tiled) -> bool {
        match &w.tiles {
            Tiles::Q6K(_) => super::wide_six_bit(),
            _ => super::wide(),
        }
    }

    pub(super) fn multiply(group: &[&Tiled], rows: &[Row], outs: &mut [&mut [f32]]) {
        match group.iter().all(|w| wide(w)) {
            true => avx512::multiply(group, rows, outs),
            false => avx2::multiply(group, rows, outs),
        }
    }

    /// SAFETY (for the caller): this CPU tiles ([`super::available`]), and
    /// the rest as the kernel it is handed to asks.
    pub(super) unsafe fn sixteens(
        w: &Tiled,
        first: usize,
        last: usize,
        rows: &[Row],
        out: *mut f32,
    ) {
        match wide(w) {
            true => avx512::sixteens(w, first, last, rows, out),
            false => avx2::sixteens(w, first, last, rows, out),
        }
    }

    /// SAFETY (for the caller): this CPU tiles ([`super::available`]).
    pub(super) unsafe fn quantize(x: &[f32], row: &mut Row) {
        match super::wide() {
            true => avx512::quantize(x, row),
            false => avx2::quantize(x, row),
        }
    }

    /// SAFETY (for the caller): this CPU tiles ([`super::available`]), and
    /// the rest as the kernel it is handed to asks.
    pub(super) unsafe fn quantize_block(xs: *const f32, d: *mut f32, qs: *mut i8, sums: *mut i32) {
        match super::wide() {
            true => avx512::quantize_block(xs, d, qs, sums),
            false => avx2::quantize_block(xs, d, qs, sums),
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[path = "tiles_avx2.rs"]
mod avx2;

#[cfg(target_arch = "x86_64")]
mod avx512 {
    use super::super::phases::{dispatch, Phase};
    use super::{Row, TileQ4K, TileQ6, TileQ8, Tiled, Tiles, QK8_0, QK_K, ROWS};
    use std::arch::x86_64::*;

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

    /// [`super::quantize_plainly`] sixteen values at a step, value for value
    /// the same: the scale from the first value furthest from zero, each
    /// value rounded half away from zero and held to an i8, and the sums 32
    /// at a time.
    #[target_feature(enable = "avx512f,avx512bw")]
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

    /// One block of 256 values at `xs` quantized: its scale to `d`, its
    /// values to `qs` and their sixteen sums, 16 at a time, to `sums`, all of
    /// them written whatever they held.
    #[target_feature(enable = "avx512f,avx512bw")]
    pub(super) unsafe fn quantize_block(xs: *const f32, d: *mut f32, qs: *mut i8, sums: *mut i32) {
        let mut most = _mm512_setzero_ps();
        for j in 0..QK_K / 16 {
            most = _mm512_max_ps(most, _mm512_abs_ps(_mm512_loadu_ps(xs.add(j * 16))));
        }
        let amax = _mm512_reduce_max_ps(most);
        if amax == 0.0 {
            // A block of nothing: no scale is taken from it.
            *d = 0.0;
            std::ptr::write_bytes(qs, 0, QK_K);
            std::ptr::write_bytes(sums, 0, 16);
            return;
        }
        // The first value that far from zero, with its sign.
        let reached = _mm512_set1_ps(amax);
        let mut max = amax;
        for j in 0..QK_K / 16 {
            let v = _mm512_abs_ps(_mm512_loadu_ps(xs.add(j * 16)));
            let at = _mm512_cmp_ps_mask::<_CMP_EQ_OQ>(v, reached);
            if at != 0 {
                max = *xs.add(j * 16 + at.trailing_zeros() as usize);
                break;
            }
        }
        let sign = _mm512_set1_epi32(i32::MIN);
        let (half, one) = (_mm512_set1_ps(0.5), _mm512_set1_ps(1.0));
        let (low, high) = (_mm512_set1_ps(-128.0), _mm512_set1_ps(127.0));
        let iscale = -127f32 / max;
        let by = _mm512_set1_ps(iscale);
        for j in 0..QK_K / 16 {
            let t = _mm512_mul_ps(by, _mm512_loadu_ps(xs.add(j * 16)));
            // Half away from zero: toward zero, then one further where
            // what was cut off is a half or more.
            let cut = _mm512_roundscale_ps::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(t);
            let off = _mm512_abs_ps(_mm512_sub_ps(t, cut));
            let further = _mm512_castsi512_ps(_mm512_or_si512(
                _mm512_castps_si512(one),
                _mm512_and_si512(_mm512_castps_si512(t), sign),
            ));
            let rounded = _mm512_mask_add_ps(
                cut,
                _mm512_cmp_ps_mask::<_CMP_GE_OQ>(off, half),
                cut,
                further,
            );
            let held = _mm512_cvttps_epi32(_mm512_min_ps(_mm512_max_ps(rounded, low), high));
            _mm_storeu_si128(qs.add(j * 16) as *mut __m128i, _mm512_cvtepi32_epi8(held));
            *sums.add(j) = _mm512_reduce_add_epi32(held);
        }
        *d = 1.0 / iscale;
    }

    /// Four input values, repeated across the sixteen rows.
    #[target_feature(enable = "avx512f")]
    unsafe fn four(act: *const i8, at: usize) -> __m512i {
        _mm512_set1_epi32(std::ptr::read_unaligned(act.add(at) as *const i32))
    }

    /// Sixteen rows' products with one block of 256 input values, added to
    /// `acc`. A weight is `d * scale * q - dmin * min`, so a row's product is
    /// each run of 32's `q · input` by its scale, less its min by the
    /// input's own sum over the run: whole numbers until the block's last
    /// step, where the two totals meet the row's two scales.
    ///
    /// Each of the eight runs keeps a running sum of its own and the runs
    /// take a step in turn, so the wait for one step's sum is spent on the
    /// other seven: with two sums on the go the kernel waited on its own
    /// arithmetic for most of a tile, which is what bounds a prompt's rows,
    /// read from cache as they are.
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni,f16c")]
    unsafe fn q4k_block(
        tile: &TileQ4K,
        act: *const i8,
        d_act: f32,
        sums: *const i32,
        acc: __m512,
    ) -> __m512 {
        let mask = _mm512_set1_epi8(0x0F);
        let qs = tile.qs.as_ptr();
        let mut isum = [_mm512_setzero_si512(); 8];
        for g in 0..4 {
            for (s, isum) in isum.iter_mut().enumerate() {
                let w = _mm512_loadu_si512(qs.add((g * 8 + s) * ROWS * 4) as *const __m512i);
                *isum = _mm512_dpbusd_epi32(
                    *isum,
                    _mm512_and_si512(w, mask),
                    four(act, s * 32 + g * 8),
                );
                *isum = _mm512_dpbusd_epi32(
                    *isum,
                    _mm512_and_si512(_mm512_srli_epi16(w, 4), mask),
                    four(act, s * 32 + g * 8 + 4),
                );
            }
        }
        let mut scaled = _mm512_setzero_si512();
        for (s, isum) in isum.iter().enumerate() {
            let scale =
                _mm512_cvtepu8_epi32(_mm_loadu_si128(tile.scales[s].as_ptr() as *const __m128i));
            scaled = _mm512_add_epi32(scaled, _mm512_mullo_epi32(*isum, scale));
        }
        // Each row's mins by the input's sums, two runs to a step: the mins
        // as 16-bit pairs against the two sums as a pair.
        let mut mins = _mm512_setzero_si512();
        for (pair, both) in tile.mins.iter().enumerate() {
            let min = _mm512_cvtepu8_epi16(_mm256_loadu_si256(both.as_ptr() as *const __m256i));
            let first = *sums.add(4 * pair) + *sums.add(4 * pair + 1);
            let second = *sums.add(4 * pair + 2) + *sums.add(4 * pair + 3);
            // A run's sum is at most 32 values of 128: it fits 16 bits.
            let sum = _mm512_set1_epi32((first & 0xFFFF) | (second << 16));
            mins = _mm512_add_epi32(mins, _mm512_madd_epi16(min, sum));
        }
        let d = _mm512_cvtph_ps(_mm256_loadu_si256(tile.d.as_ptr() as *const __m256i));
        let dmin = _mm512_cvtph_ps(_mm256_loadu_si256(tile.dmin.as_ptr() as *const __m256i));
        let by = _mm512_set1_ps(d_act);
        let acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(scaled), _mm512_mul_ps(d, by), acc);
        _mm512_fnmadd_ps(_mm512_cvtepi32_ps(mins), _mm512_mul_ps(dmin, by), acc)
    }

    /// [`q4k_block`] for three rows of input at once, each row's products
    /// added to its own of `acc`: to the bit what the block gives each row
    /// alone. A prompt's rows are bound by this arithmetic, not by memory,
    /// and a third of it a row at a time is splitting each byte of the tile
    /// into its two values, which is the same for every row: here it is done
    /// once for the three.
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni,f16c")]
    unsafe fn q4k_block3(
        tile: &TileQ4K,
        rows: [&Row; 3],
        block: usize,
        acc: [__m512; 3],
    ) -> [__m512; 3] {
        let mask = _mm512_set1_epi8(0x0F);
        let qs = tile.qs.as_ptr();
        let act = rows.map(|row| row.qs.as_ptr().add(block * QK_K));
        let mut isum = [[_mm512_setzero_si512(); 8]; 3];
        for g in 0..4 {
            for s in 0..8 {
                let w = _mm512_loadu_si512(qs.add((g * 8 + s) * ROWS * 4) as *const __m512i);
                let low = _mm512_and_si512(w, mask);
                let high = _mm512_and_si512(_mm512_srli_epi16(w, 4), mask);
                for (isum, act) in isum.iter_mut().zip(act) {
                    let sum = &mut isum[s];
                    *sum = _mm512_dpbusd_epi32(*sum, low, four(act, s * 32 + g * 8));
                    *sum = _mm512_dpbusd_epi32(*sum, high, four(act, s * 32 + g * 8 + 4));
                }
            }
        }
        let mut scale = [_mm512_setzero_si512(); 8];
        for (s, scale) in scale.iter_mut().enumerate() {
            *scale =
                _mm512_cvtepu8_epi32(_mm_loadu_si128(tile.scales[s].as_ptr() as *const __m128i));
        }
        let mut min = [_mm512_setzero_si512(); 4];
        for (pair, min) in min.iter_mut().enumerate() {
            *min = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                tile.mins[pair].as_ptr() as *const __m256i
            ));
        }
        let d = _mm512_cvtph_ps(_mm256_loadu_si256(tile.d.as_ptr() as *const __m256i));
        let dmin = _mm512_cvtph_ps(_mm256_loadu_si256(tile.dmin.as_ptr() as *const __m256i));
        let mut out = acc;
        for ((out, isum), row) in out.iter_mut().zip(&isum).zip(rows) {
            let mut scaled = _mm512_setzero_si512();
            for (isum, scale) in isum.iter().zip(&scale) {
                scaled = _mm512_add_epi32(scaled, _mm512_mullo_epi32(*isum, *scale));
            }
            let sums = row.sums.as_ptr().add(block * 16);
            let mut mins = _mm512_setzero_si512();
            for (pair, min) in min.iter().enumerate() {
                let first = *sums.add(4 * pair) + *sums.add(4 * pair + 1);
                let second = *sums.add(4 * pair + 2) + *sums.add(4 * pair + 3);
                let sum = _mm512_set1_epi32((first & 0xFFFF) | (second << 16));
                mins = _mm512_add_epi32(mins, _mm512_madd_epi16(*min, sum));
            }
            let by = _mm512_set1_ps(*row.d.get_unchecked(block));
            let with = _mm512_fmadd_ps(_mm512_cvtepi32_ps(scaled), _mm512_mul_ps(d, by), *out);
            *out = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(mins), _mm512_mul_ps(dmin, by), with);
        }
        out
    }

    /// Where each byte of a group of 4 values goes for its fields to be cut
    /// out: a row's three bytes to the low three of its four, the last of
    /// them repeated above, where it is not read.
    const SPREAD: [u8; 64] = {
        let mut to = [0u8; 64];
        let mut r = 0;
        while r < ROWS {
            to[4 * r] = (3 * r) as u8;
            to[4 * r + 1] = (3 * r + 1) as u8;
            to[4 * r + 2] = (3 * r + 2) as u8;
            to[4 * r + 3] = (3 * r + 2) as u8;
            r += 1;
        }
        to
    };

    /// Sixteen rows' products with one block of 256 input values, added to
    /// `acc`. A weight is `d * scale * (q - 32)`, so a row's product is each
    /// run of 16's `q · input`, less 32 times the input's own sum over the
    /// run, by its scale: whole numbers until the block's last step.
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx512vbmi,f16c")]
    unsafe fn q6k_block(
        tile: &TileQ6,
        act: *const i8,
        d_act: f32,
        sums: *const i32,
        acc: __m512,
    ) -> __m512 {
        let spread = _mm512_loadu_si512(SPREAD.as_ptr() as *const __m512i);
        // Four 6-bit fields from each row's 24 bits, two rows to a 64-bit
        // lane: the bit each field starts at.
        let fields = _mm512_set1_epi64(i64::from_le_bytes([0, 6, 12, 18, 32, 38, 44, 50]));
        let six = _mm512_set1_epi8(0x3F);
        let qs = tile.qs.as_ptr();
        let mut total = _mm512_setzero_si512();
        for run in 0..16 {
            let mut isum = _mm512_setzero_si512();
            for g in 0..4 {
                // 48 bytes are a group; the mask keeps the load inside them.
                let packed = _mm512_maskz_loadu_epi8(
                    0x0000_FFFF_FFFF_FFFF,
                    qs.add((run * 4 + g) * ROWS * 3) as *const i8,
                );
                let q = _mm512_and_si512(
                    _mm512_multishift_epi64_epi8(fields, _mm512_permutexvar_epi8(spread, packed)),
                    six,
                );
                isum = _mm512_dpbusd_epi32(isum, q, four(act, run * 16 + g * 4));
            }
            let centered = _mm512_sub_epi32(isum, _mm512_set1_epi32(32 * *sums.add(run)));
            let scale =
                _mm512_cvtepi8_epi32(_mm_loadu_si128(tile.scales[run].as_ptr() as *const __m128i));
            total = _mm512_add_epi32(total, _mm512_mullo_epi32(centered, scale));
        }
        let d = _mm512_cvtph_ps(_mm256_loadu_si256(tile.d.as_ptr() as *const __m256i));
        _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(total),
            _mm512_mul_ps(d, _mm512_set1_ps(d_act)),
            acc,
        )
    }

    /// [`q6k_block`] for `N` rows of input at once, each row's products added
    /// to its own of `acc`: to the bit what the block gives each row alone.
    /// Cutting a tile's six-bit values out of its bytes is as much work as
    /// multiplying them, and it is the same for every row: here it is done
    /// once for them all.
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx512vbmi,f16c")]
    unsafe fn q6k_block_of<const N: usize>(
        tile: &TileQ6,
        rows: [&Row; N],
        block: usize,
        acc: [__m512; N],
    ) -> [__m512; N] {
        let spread = _mm512_loadu_si512(SPREAD.as_ptr() as *const __m512i);
        let fields = _mm512_set1_epi64(i64::from_le_bytes([0, 6, 12, 18, 32, 38, 44, 50]));
        let six = _mm512_set1_epi8(0x3F);
        let qs = tile.qs.as_ptr();
        let act = rows.map(|row| row.qs.as_ptr().add(block * QK_K));
        let sums = rows.map(|row| row.sums.as_ptr().add(block * 16));
        let mut total = [_mm512_setzero_si512(); N];
        for run in 0..16 {
            let mut isum = [_mm512_setzero_si512(); N];
            for g in 0..4 {
                // 48 bytes are a group; the mask keeps the load inside them.
                let packed = _mm512_maskz_loadu_epi8(
                    0x0000_FFFF_FFFF_FFFF,
                    qs.add((run * 4 + g) * ROWS * 3) as *const i8,
                );
                let q = _mm512_and_si512(
                    _mm512_multishift_epi64_epi8(fields, _mm512_permutexvar_epi8(spread, packed)),
                    six,
                );
                for (isum, act) in isum.iter_mut().zip(act) {
                    *isum = _mm512_dpbusd_epi32(*isum, q, four(act, run * 16 + g * 4));
                }
            }
            let scale =
                _mm512_cvtepi8_epi32(_mm_loadu_si128(tile.scales[run].as_ptr() as *const __m128i));
            for ((total, isum), sums) in total.iter_mut().zip(isum).zip(sums) {
                let centered = _mm512_sub_epi32(isum, _mm512_set1_epi32(32 * *sums.add(run)));
                *total = _mm512_add_epi32(*total, _mm512_mullo_epi32(centered, scale));
            }
        }
        let d = _mm512_cvtph_ps(_mm256_loadu_si256(tile.d.as_ptr() as *const __m256i));
        let mut out = acc;
        for ((out, total), row) in out.iter_mut().zip(total).zip(rows) {
            *out = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(total),
                _mm512_mul_ps(d, _mm512_set1_ps(*row.d.get_unchecked(block))),
                *out,
            );
        }
        out
    }

    /// [`q6k_block_of`] for the first `N` of `rows`, their products added
    /// to the first `N` of `acc`.
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx512vbmi,f16c")]
    unsafe fn q6k_block_some<const N: usize>(
        tile: &TileQ6,
        rows: &[Row],
        block: usize,
        acc: &mut [__m512],
    ) {
        let some: [&Row; N] = std::array::from_fn(|i| &rows[i]);
        let held: [__m512; N] = std::array::from_fn(|i| acc[i]);
        acc[..N].copy_from_slice(&q6k_block_of(tile, some, block, held));
    }

    /// Sixteen rows' products with one block of 32 input values, added to
    /// `acc`. The weights are stored 128 up, so the input's sum over the
    /// block, 128 times, comes back off.
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni,f16c")]
    unsafe fn q8_block(tile: &TileQ8, act: *const i8, d_act: f32, sum: i32, acc: __m512) -> __m512 {
        let qs = tile.qs.as_ptr();
        let mut isum_a = _mm512_setzero_si512();
        let mut isum_b = _mm512_setzero_si512();
        for g in 0..4 {
            let (ga, gb) = (g * 2, g * 2 + 1);
            isum_a = _mm512_dpbusd_epi32(
                isum_a,
                _mm512_loadu_si512(qs.add(ga * ROWS * 4) as *const __m512i),
                four(act, ga * 4),
            );
            isum_b = _mm512_dpbusd_epi32(
                isum_b,
                _mm512_loadu_si512(qs.add(gb * ROWS * 4) as *const __m512i),
                four(act, gb * 4),
            );
        }
        let f = _mm512_cvtepi32_ps(_mm512_sub_epi32(
            _mm512_add_epi32(isum_a, isum_b),
            _mm512_set1_epi32(128 * sum),
        ));
        let d = _mm512_cvtph_ps(_mm256_loadu_si256(tile.d.as_ptr() as *const __m256i));
        _mm512_fmadd_ps(f, _mm512_mul_ps(d, _mm512_set1_ps(d_act)), acc)
    }

    /// Rows whose running sums are kept at once while a sixteen's tiles go
    /// by.
    const TOGETHER: usize = 32;

    /// Rows a six-bit tile is cut once for: each has a sum and a total in
    /// hand through the block, and eight rows' are what the registers hold
    /// beside the tile's own vectors.
    const Q6K_TOGETHER: usize = 8;

    /// Sixteen `t` of `w` against the one row `row`.
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni,f16c")]
    unsafe fn sixteen_of(w: &Tiled, t: usize, row: &Row) -> __m512 {
        let mut acc = _mm512_setzero_ps();
        match &w.tiles {
            Tiles::Q4K(tiles) => {
                let kb = w.cols / QK_K;
                for (b, tile) in tiles[t * kb..(t + 1) * kb].iter().enumerate() {
                    acc = q4k_block(
                        tile,
                        row.qs.as_ptr().add(b * QK_K),
                        *row.d.get_unchecked(b),
                        row.sums.as_ptr().add(b * 16),
                        acc,
                    );
                }
            }
            Tiles::Q6K(tiles) => {
                let kb = w.cols / QK_K;
                for (b, tile) in tiles[t * kb..(t + 1) * kb].iter().enumerate() {
                    // Six-bit tiles are only made where `six_bit` found
                    // the one feature more this is built for.
                    acc = q6k_block(
                        tile,
                        row.qs.as_ptr().add(b * QK_K),
                        *row.d.get_unchecked(b),
                        row.sums.as_ptr().add(b * 16),
                        acc,
                    );
                }
            }
            Tiles::Q8_0(tiles) => {
                let nb = w.cols / QK8_0;
                for (b, tile) in tiles[t * nb..(t + 1) * nb].iter().enumerate() {
                    acc = q8_block(
                        tile,
                        row.qs.as_ptr().add(b * QK8_0),
                        *row.d.get_unchecked(b / 8),
                        *row.sums.get_unchecked(2 * b) + *row.sums.get_unchecked(2 * b + 1),
                        acc,
                    );
                }
            }
        }
        acc
    }

    /// Sixteen `t` of `w` against each of `rows` (no more than
    /// [`TOGETHER`]), into `acc`: each row's products as [`sixteen_of`]
    /// gives them, to the bit. The sixteen's tiles go by once, each against
    /// every row before the next: a wide weight's sixteen is more tiles than
    /// the nearest cache holds, and a row at a time read them all again for
    /// every row.
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni,f16c")]
    unsafe fn sixteen_of_each(w: &Tiled, t: usize, rows: &[Row], acc: &mut [__m512]) {
        match &w.tiles {
            Tiles::Q4K(tiles) => {
                let kb = w.cols / QK_K;
                for (b, tile) in tiles[t * kb..(t + 1) * kb].iter().enumerate() {
                    let mut i = 0;
                    while i + 3 <= rows.len() {
                        let three = q4k_block3(
                            tile,
                            [&rows[i], &rows[i + 1], &rows[i + 2]],
                            b,
                            [acc[i], acc[i + 1], acc[i + 2]],
                        );
                        acc[i..i + 3].copy_from_slice(&three);
                        i += 3;
                    }
                    for (acc, row) in acc[i..].iter_mut().zip(&rows[i..]) {
                        *acc = q4k_block(
                            tile,
                            row.qs.as_ptr().add(b * QK_K),
                            *row.d.get_unchecked(b),
                            row.sums.as_ptr().add(b * 16),
                            *acc,
                        );
                    }
                }
            }
            Tiles::Q6K(tiles) => {
                let kb = w.cols / QK_K;
                for (b, tile) in tiles[t * kb..(t + 1) * kb].iter().enumerate() {
                    // The rows share the cutting of the tile, as many to a
                    // cutting as have their sums in hand at once: as few
                    // cuttings as that allows, the rows spread evenly over
                    // them (nine rows are five and four, not eight and one
                    // alone).
                    let (mut rows, mut acc) = (rows, &mut *acc);
                    let mut cuttings = rows.len().div_ceil(Q6K_TOGETHER);
                    while !rows.is_empty() {
                        let n = rows.len().div_ceil(cuttings);
                        cuttings -= 1;
                        match n {
                            8 => q6k_block_some::<8>(tile, rows, b, acc),
                            7 => q6k_block_some::<7>(tile, rows, b, acc),
                            6 => q6k_block_some::<6>(tile, rows, b, acc),
                            5 => q6k_block_some::<5>(tile, rows, b, acc),
                            4 => q6k_block_some::<4>(tile, rows, b, acc),
                            3 => q6k_block_some::<3>(tile, rows, b, acc),
                            2 => q6k_block_some::<2>(tile, rows, b, acc),
                            _ => {
                                acc[0] = q6k_block(
                                    tile,
                                    rows[0].qs.as_ptr().add(b * QK_K),
                                    *rows[0].d.get_unchecked(b),
                                    rows[0].sums.as_ptr().add(b * 16),
                                    acc[0],
                                );
                            }
                        }
                        rows = &rows[n..];
                        acc = &mut acc[n..];
                    }
                }
            }
            Tiles::Q8_0(tiles) => {
                let nb = w.cols / QK8_0;
                for (b, tile) in tiles[t * nb..(t + 1) * nb].iter().enumerate() {
                    for (acc, row) in acc.iter_mut().zip(rows) {
                        *acc = q8_block(
                            tile,
                            row.qs.as_ptr().add(b * QK8_0),
                            *row.d.get_unchecked(b / 8),
                            *row.sums.get_unchecked(2 * b) + *row.sums.get_unchecked(2 * b + 1),
                            *acc,
                        );
                    }
                }
            }
        }
    }

    /// The sixteens `first..last` of `w` against every row, written to `out`.
    ///
    /// SAFETY (for the caller): the CPU has the features named, `out` has
    /// room for `rows.len()` rows of `w.rows` values, and no other thread
    /// writes these sixteens.
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni,f16c")]
    unsafe fn sixteens_of(w: &Tiled, first: usize, last: usize, rows: &[Row], out: &Out) {
        for t in first..last {
            if let [row] = rows {
                _mm512_storeu_ps(out.at.add(t * ROWS), sixteen_of(w, t, row));
                continue;
            }
            for (run, some) in rows.chunks(TOGETHER).enumerate() {
                let mut acc = [_mm512_setzero_ps(); TOGETHER];
                sixteen_of_each(w, t, some, &mut acc[..some.len()]);
                for (i, acc) in acc[..some.len()].iter().enumerate() {
                    _mm512_storeu_ps(
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
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni,f16c")]
    pub(super) unsafe fn sixteens(
        w: &Tiled,
        first: usize,
        last: usize,
        rows: &[Row],
        out: *mut f32,
    ) {
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
    /// weight, `rows.len()` rows of its width): one phase of one dispatch,
    /// its units the group's sixteens, one weight's after another's.
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::QMatMul;
    use candle_core::Module;

    fn largest_gap(a: &Tensor, b: &Tensor) -> Result<f32> {
        let peak = b.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        Ok((a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()? / peak)
    }

    /// The forms this CPU has tiles for.
    fn forms() -> Vec<GgmlDType> {
        let mut forms = vec![GgmlDType::Q4K, GgmlDType::Q8_0];
        if six_bit() {
            forms.push(GgmlDType::Q6K);
        }
        forms
    }

    /// `rows` by `cols` of weights at `dtype`: tiled, and as candle holds it.
    fn weight(rows: usize, cols: usize, dtype: GgmlDType) -> Result<(Option<Tiled>, QMatMul)> {
        let w = Tensor::randn(0f32, 0.05, (rows, cols), &Device::Cpu)?;
        let q = QTensor::quantize(&w, dtype)?;
        let tiled = Tiled::pack(&q)?;
        Ok((tiled, QMatMul::from_qtensor(q)?))
    }

    /// How near candle's own product a form's tiles come on this CPU. candle
    /// quantizes the input as the tiles do, 256 values to a scale, except for
    /// a Q8_0 weight on a CPU without AVX-512 VNNI, where it gives every 32
    /// values a scale of their own: there the two products are of two
    /// roundings of the input, a hundredth or two apart.
    /// [`tiles_multiply_what_they_hold`] holds the tiles to the bit there.
    fn as_candle(dtype: GgmlDType) -> f32 {
        #[cfg(target_arch = "x86_64")]
        let as_the_tiles = wide();
        #[cfg(not(target_arch = "x86_64"))]
        let as_the_tiles = false;
        if !as_the_tiles && dtype == GgmlDType::Q8_0 {
            3e-2
        } else {
            1e-5
        }
    }

    /// A weight multiplied through its tiles is the weight as quantized
    /// times the rows as quantized, on any CPU that tiles: the product of
    /// the values the tiles and the quantized rows stand for, to within the
    /// order of a few float sums.
    #[test]
    fn tiles_multiply_what_they_hold() -> Result<()> {
        if !available() {
            return Ok(());
        }
        for dtype in forms() {
            let w = Tensor::randn(0f32, 0.05, (48, 512), &Device::Cpu)?;
            let q = QTensor::quantize(&w, dtype)?;
            let held = q.dequantize(&Device::Cpu)?;
            let Some(tiled) = Tiled::pack(&q)? else {
                bail!("{dtype:?} was not tiled")
            };
            for n in [1usize, 5, 35] {
                let x = Tensor::randn(0f32, 1.0, (n, 512), &Device::Cpu)?;
                let mut read = Vec::with_capacity(n * 512);
                for row in x.to_vec2::<f32>()? {
                    let row = quantize_plainly(&row);
                    read.extend(
                        row.qs
                            .iter()
                            .enumerate()
                            .map(|(i, q)| row.d[i / QK_K] * f32::from(*q)),
                    );
                }
                let want = Tensor::from_vec(read, (n, 512), &Device::Cpu)?.matmul(&held.t()?)?;
                let got = forward(&[&tiled], &x)?;
                let gap = largest_gap(&got[0], &want)?;
                assert!(gap < 1e-5, "{dtype:?}, {n} rows: off by {gap} of the peak");
            }
        }
        Ok(())
    }

    /// A weight multiplied through its tiles is the product candle computes,
    /// at both forms, for one row and for a prompt's, and for weights that
    /// share their rows as for one alone.
    #[test]
    fn tiles_multiply_as_candle_does() -> Result<()> {
        if !available() {
            eprintln!("skipped: nothing is tiled on this CPU");
            return Ok(());
        }
        for dtype in forms() {
            let (a, candle_a) = weight(48, 512, dtype)?;
            let (b, candle_b) = weight(16, 512, dtype)?;
            let (Some(a), Some(b)) = (a, b) else {
                bail!("{dtype:?} was not tiled")
            };
            assert_eq!((a.rows(), b.rows()), (48, 16));
            for shape in [(1, 1, 512), (1, 5, 512), (2, 3, 512)] {
                let x = Tensor::randn(0f32, 1.0, shape, &Device::Cpu)?;
                let both = forward(&[&a, &b], &x)?;
                let [got_a, got_b] = both.as_slice() else {
                    bail!("two weights, {} products", both.len())
                };
                let (want_a, want_b) = (candle_a.forward(&x)?, candle_b.forward(&x)?);
                assert_eq!(got_a.dims(), want_a.dims());
                assert_eq!(got_b.dims(), want_b.dims());
                for (got, want) in [(got_a, &want_a), (got_b, &want_b)] {
                    let gap = largest_gap(got, want)?;
                    assert!(
                        gap < as_candle(dtype),
                        "{dtype:?} {shape:?}: off by {gap} of the peak"
                    );
                }
                let alone = forward(&[&b], &x)?;
                let [alone] = alone.as_slice() else {
                    bail!("one weight, {} products", alone.len())
                };
                assert_eq!(largest_gap(alone, got_b)?, 0.0, "alone as in a group");
            }
        }
        Ok(())
    }

    /// A row of nothing, and a block of nothing in a row, multiply to what
    /// candle gives: no scale is taken from a block with no value in it.
    #[test]
    fn a_block_of_zeros_in_the_rows_is_no_trouble() -> Result<()> {
        if !available() {
            return Ok(());
        }
        for dtype in forms() {
            let (Some(tiled), candle) = weight(32, 512, dtype)? else {
                bail!("{dtype:?} was not tiled")
            };
            let live = Tensor::randn(0f32, 1.0, (1, 1, 256), &Device::Cpu)?;
            let dead = Tensor::zeros((1, 1, 256), DType::F32, &Device::Cpu)?;
            let x = Tensor::cat(&[&dead, &live], 2)?;
            let got = forward(&[&tiled], &x)?;
            let gap = largest_gap(&got[0], &candle.forward(&x)?)?;
            assert!(
                gap < as_candle(dtype),
                "{dtype:?}: off by {gap} of the peak"
            );
            let none = Tensor::zeros((1, 1, 512), DType::F32, &Device::Cpu)?;
            let got = forward(&[&tiled], &none)?;
            let most = got[0].abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            assert_eq!(most, 0.0);
        }
        Ok(())
    }

    /// The vectors quantize a row to exactly what the plain loop does: its
    /// scales, its values and their sums, with values that round at a half,
    /// values past an i8, a block of zeros, and a negative extreme.
    #[test]
    fn rows_are_quantized_in_vectors_as_they_are_plainly() -> Result<()> {
        if !available() {
            return Ok(());
        }
        let mut x: Vec<f32> = Tensor::randn(0f32, 1.0, 1024, &Device::Cpu)?.to_vec1()?;
        // A block whose extreme is 127, so the scale is one and values land
        // on halves; and one whose extreme is negative.
        for (i, v) in x[..256].iter_mut().enumerate() {
            *v = (i as f32 - 128.0) / 2.0;
        }
        x[0] = 127.0;
        x[1] = -127.0;
        x[256..512].fill(0.0);
        x[600] = -9.5;
        let (got, want) = (quantize(&x), quantize_plainly(&x));
        assert_eq!(got.d, want.d);
        assert_eq!(got.qs, want.qs);
        assert_eq!(got.sums, want.sums);
        assert!(got.qs[..256].iter().any(|q| *q == -128 || *q == 127));
        Ok(())
    }

    /// A weight quantized and tiled in runs on every core multiplies to
    /// exactly what the same weight quantized whole and then tiled does, at
    /// both forms, with more sixteens than threads and with fewer.
    #[test]
    fn a_weight_quantized_in_runs_is_the_weight_quantized_whole() -> Result<()> {
        if !available() {
            return Ok(());
        }
        for dtype in forms() {
            for rows in [16usize, 48, 1024] {
                let w = Tensor::randn(0f32, 0.05, (rows, 512), &Device::Cpu)?;
                let (Some(whole), Some(runs)) = (
                    Tiled::pack(&QTensor::quantize(&w, dtype)?)?,
                    Tiled::quantize(&w, dtype)?,
                ) else {
                    bail!("{dtype:?} was not tiled")
                };
                let x = Tensor::randn(0f32, 1.0, (1, 3, 512), &Device::Cpu)?;
                let (want, got) = (forward(&[&whole], &x)?, forward(&[&runs], &x)?);
                assert_eq!(got[0].dims(), &[1, 3, rows]);
                assert_eq!(
                    got[0].flatten_all()?.to_vec1::<f32>()?,
                    want[0].flatten_all()?.to_vec1::<f32>()?,
                    "{dtype:?}, {rows} rows"
                );
            }
        }
        // What has no tile is left alone.
        let w = Tensor::randn(0f32, 0.05, (16, 512), &Device::Cpu)?;
        assert!(Tiled::quantize(&w, GgmlDType::Q5K)?.is_none());
        assert!(Tiled::quantize(&w.narrow(0, 0, 8)?, GgmlDType::Q4K)?.is_none());
        Ok(())
    }

    /// A row quantized a block at a time, as a phase's units do it, is the
    /// row quantized whole, into room that held another row before; and a
    /// weight multiplied a few sixteens at a time, as those units do it, is
    /// the weight multiplied whole.
    #[test]
    fn a_row_and_a_weight_taken_in_parts_are_the_whole() -> Result<()> {
        if !available() {
            return Ok(());
        }
        let x: Vec<f32> = Tensor::randn(0f32, 1.0, 1024, &Device::Cpu)?.to_vec1()?;
        let mut x = x;
        x[256..512].fill(0.0);
        let want = quantize(&x);
        let mut row = quantize(&vec![7.5f32; 1024]);
        let blocks = row.blocks(1024);
        for (block, values) in x.chunks_exact(QK_K).enumerate().rev() {
            let Ok(values) = <&[f32; QK_K]>::try_from(values) else {
                bail!("a block is 256 values")
            };
            // SAFETY: the row has four blocks, and nothing else writes them.
            unsafe { blocks.quantize(block, values) };
        }
        assert_eq!(
            (&row.d, &row.qs, &row.sums),
            (&want.d, &want.qs, &want.sums)
        );

        for dtype in forms() {
            let (Some(tiled), _) = weight(64, 1024, dtype)? else {
                bail!("{dtype:?} was not tiled")
            };
            assert_eq!(
                (tiled.rows(), tiled.cols(), tiled.sixteens()),
                (64, 1024, 4)
            );
            let mut whole = vec![0f32; 64];
            multiply(&[&tiled], &row, &mut [&mut whole])?;
            let mut parts = vec![0f32; 64];
            for (first, last) in [(3, 4), (0, 1), (1, 3)] {
                // SAFETY: the row holds the weight's 1024 values, `parts` its
                // 64 products, and nothing else writes them.
                unsafe {
                    tiled.multiply_sixteens(
                        first,
                        last,
                        std::slice::from_ref(&row),
                        parts.as_mut_ptr(),
                    )
                };
            }
            assert_eq!(parts, whole, "{dtype:?}");
            let mut short = vec![0f32; 48];
            assert!(multiply(&[&tiled], &row, &mut [&mut short]).is_err());
        }
        Ok(())
    }

    /// Rows multiplied together are each row multiplied alone, to the bit,
    /// at every form and for every number of rows that is taken another way:
    /// one, the few that share a tile's cutting, what is left over after
    /// them, and more than are kept together.
    #[test]
    fn rows_multiplied_together_are_each_multiplied_alone() -> Result<()> {
        if !available() {
            return Ok(());
        }
        for dtype in forms() {
            let (Some(tiled), _) = weight(48, 768, dtype)? else {
                bail!("{dtype:?} was not tiled")
            };
            let most = 2 * 32 + 3;
            let mut xs: Vec<f32> = Tensor::randn(0f32, 1.0, most * 768, &Device::Cpu)?.to_vec1()?;
            // A block of nothing in one row, and a row of nothing.
            xs[5 * 768 + 256..5 * 768 + 512].fill(0.0);
            xs[9 * 768..10 * 768].fill(0.0);
            let rows = quantize_rows(&xs, 768);
            let mut alone = vec![0f32; most * 48];
            for (row, out) in rows.iter().zip(alone.chunks_exact_mut(48)) {
                multiply(&[&tiled], row, &mut [out])?;
            }
            for n in [
                1usize, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 23, 24, 27, 31,
                32, 33, 34, 35, 48, most,
            ] {
                let mut together = vec![0f32; n * 48];
                // SAFETY: each row holds the weight's 768 values, `together`
                // every row's 48 products, and nothing else writes them.
                unsafe {
                    tiled.multiply_sixteens(0, tiled.sixteens(), &rows[..n], together.as_mut_ptr())
                };
                assert!(
                    together == alone[..n * 48],
                    "{dtype:?}: {n} rows together are not the rows alone"
                );
            }
        }
        Ok(())
    }

    /// Weights and input values at the ends of their ranges multiply to what
    /// they stand for, alone and among other rows. The kernels sum products
    /// in sixteen bits for as long as they fit, and the ends are where the
    /// sums are widest: every stored value its form's least or its most,
    /// against input values of 127 and of -127.
    #[test]
    fn values_at_their_extremes_multiply_as_any_do() -> Result<()> {
        if !available() {
            return Ok(());
        }
        let (rows, cols, n) = (32usize, 512usize, 35usize);
        // A run of sixteen weights: all one value; one, then fifteen of the
        // other sign just under it; and nothing, then fifteen of one value.
        let weights: [fn(usize) -> f32; 3] = [
            |_| 0.05,
            |i| if i % 16 == 0 { 0.05 } else { -0.0485 },
            |i| if i % 16 == 0 { 0.0 } else { 0.05 },
        ];
        // A block of 256 input values: all one value, which is quantized to
        // -127; one below nothing and the rest above, quantized to 127; and
        // the two signs turn about.
        let inputs: [fn(usize) -> f32; 3] = [
            |_| 1.0,
            |i| if i % 256 == 0 { -1.0 } else { 1.0 },
            |i| if i % 2 == 0 { -1.0 } else { 1.0 },
        ];
        for dtype in forms() {
            for (wi, weight) in weights.iter().enumerate() {
                let w: Vec<f32> = (0..rows * cols).map(|i| weight(i % cols)).collect();
                let w = Tensor::from_vec(w, (rows, cols), &Device::Cpu)?;
                let q = QTensor::quantize(&w, dtype)?;
                let held = q.dequantize(&Device::Cpu)?;
                let Some(tiled) = Tiled::pack(&q)? else {
                    bail!("{dtype:?} was not tiled")
                };
                for (xi, input) in inputs.iter().enumerate() {
                    let xs: Vec<f32> = (0..n * cols).map(|i| input(i % cols)).collect();
                    let given = quantize_rows(&xs, cols);
                    let read: Vec<f32> = given
                        .iter()
                        .flat_map(|row| {
                            row.qs
                                .iter()
                                .enumerate()
                                .map(|(i, q)| row.d[i / QK_K] * f32::from(*q))
                        })
                        .collect();
                    let want =
                        Tensor::from_vec(read, (n, cols), &Device::Cpu)?.matmul(&held.t()?)?;
                    let x = Tensor::from_vec(xs, (n, cols), &Device::Cpu)?;
                    let got = forward(&[&tiled], &x)?;
                    // Against the sizes summed, 26 a row, and not the sum: the
                    // signs that turn about leave next to nothing of it. A
                    // sum that left its sixteen bits would be out by far
                    // more than the order of a float sum.
                    let gap = (&got[0] - &want)?
                        .abs()?
                        .flatten_all()?
                        .max(0)?
                        .to_scalar::<f32>()?;
                    assert!(
                        gap < 1e-5 * 0.05 * cols as f32,
                        "{dtype:?}, weights {wi}, input {xi}: off by {gap}"
                    );
                    let together: Vec<f32> = got[0].flatten_all()?.to_vec1()?;
                    for (i, row) in given.iter().enumerate() {
                        let mut alone = vec![0f32; rows];
                        multiply(&[&tiled], row, &mut [&mut alone])?;
                        assert!(
                            alone == together[i * rows..(i + 1) * rows],
                            "{dtype:?}, weights {wi}, input {xi}: row {i} among {n} is not the row alone"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    /// SCRATCH, not for merge: rows of a six-bit weight through three sets
    /// of kernels in one process, on one thread, the best of the rounds kept
    /// for each and their products compared bit for bit. `release` is the
    /// release branch's file; `again` the same file a second time, which
    /// says how far apart two copies of one source lie; `branch` the file
    /// as the pull request has it.
    #[cfg(target_arch = "aarch64")]
    #[test]
    #[ignore]
    fn pace_of_rows_old_and_new() -> Result<()> {
        use std::time::Instant;
        type Kernel = unsafe fn(&Tiled, usize, usize, &[Row], *mut f32);
        let sets: [(&str, Kernel); 3] = [
            ("release", kernels_old::sixteens),
            ("again", kernels_old_again::sixteens),
            ("branch", kernels::sixteens),
        ];
        let small = std::env::var("PACE_SMALL").is_ok();
        let weights: &[(usize, usize, &str)] = if small {
            &[(64, 1536, "in the cache")]
        } else {
            &[
                (256, 1536, "in the cache"),
                (1536, 1536, "an attention weight"),
                (8960, 1536, "an MLP weight"),
            ]
        };
        for &(rows_w, cols, what) in weights {
            let (Some(tiled), _) = weight(rows_w, cols, GgmlDType::Q6K)? else {
                bail!("Q6K was not tiled")
            };
            let xs: Vec<f32> = Tensor::randn(0f32, 1.0, 32 * cols, &Device::Cpu)?.to_vec1()?;
            let rows = quantize_rows(&xs, cols);
            for n in [1usize, 2, 3, 4, 5, 6, 7, 8, 16, 32] {
                let mut outs: Vec<Vec<f32>> = sets.iter().map(|_| vec![0f32; n * rows_w]).collect();
                let mut once = |which: usize| {
                    let at = outs[which].as_mut_ptr();
                    let start = Instant::now();
                    // SAFETY: each row holds the weight's values, `at` has
                    // room for every row's products, and nothing else
                    // writes them.
                    unsafe { (sets[which].1)(&tiled, 0, tiled.sixteens(), &rows[..n], at) };
                    start.elapsed().as_secs_f64()
                };
                let first = (0..sets.len()).map(&mut once).fold(1e-9, f64::max);
                let reps = ((0.02 / first) as usize).clamp(1, 2000);
                let mut best = [f64::MAX; 3];
                for round in 0..(if small { 2 } else { 16 }) {
                    // Forwards and backwards by turns, so no set is always
                    // the first to run.
                    let mut order: Vec<usize> = (0..sets.len()).collect();
                    if round % 2 == 1 {
                        order.reverse();
                    }
                    for which in order {
                        let mut took = 0.0;
                        for _ in 0..reps {
                            took += once(which);
                        }
                        best[which] = best[which].min(took / reps as f64);
                    }
                }
                for (which, (name, _)) in sets.iter().enumerate().skip(1) {
                    assert!(
                        outs[which] == outs[0],
                        "{n} rows: {name} is not the release"
                    );
                }
                eprintln!(
                    "pace tiles Q6K {rows_w}x{cols} ({what}), {n} rows: release {:.1} us, again {:.3}, branch {:.3}",
                    best[0] * 1e6,
                    best[1] / best[0],
                    best[2] / best[0]
                );
            }
        }
        Ok(())
    }

    /// What cannot be tiled is left to candle, and rows that do not fit the
    /// weights are refused rather than read out of bounds.
    #[test]
    fn what_does_not_fit_is_not_tiled_or_multiplied() -> Result<()> {
        if !available() {
            return Ok(());
        }
        // Rows that are not sixteens, and a form with no tile.
        assert!(weight(8, 512, GgmlDType::Q4K)?.0.is_none());
        assert!(weight(16, 512, GgmlDType::Q5K)?.0.is_none());
        let (Some(wide), _) = weight(16, 512, GgmlDType::Q4K)? else {
            bail!("not tiled")
        };
        let (Some(narrow), _) = weight(16, 256, GgmlDType::Q4K)? else {
            bail!("not tiled")
        };
        let x = Tensor::zeros((1, 1, 512), DType::F32, &Device::Cpu)?;
        assert!(forward(&[&wide, &narrow], &x).is_err(), "widths differ");
        assert!(forward(&[&narrow], &x).is_err(), "rows too wide");
        assert!(forward(&[], &x)?.is_empty());
        Ok(())
    }
}
