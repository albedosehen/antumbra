//! The output head in two stages, for a token that is to be the head's first.
//!
//! The head has a row for every token of the vocabulary, and a decoded token
//! reads every one of them to learn a single thing: which row's product with
//! the last hidden row is the largest. Over a 1.5B base that is 191 MB of the
//! 1.3 GB a token reads at six bits, and a decoded token is bound by how fast
//! memory is read.
//!
//! Here every row is held once more as a sketch: its values at eight uniform
//! levels, three bits each, on a scale of the row's own, 88 MB for that head.
//! A token's products with the sketch are read from the sketch alone. How far
//! a sketched product may fall short of the row's own is known for each row,
//! the row's distance from its sketch times the size of the input, so the
//! rows that could still be first are few: those whose sketched product and
//! that much more reach the largest product found so far by reading a row
//! whole. Those rows are multiplied whole, from the dense rows the embedding
//! table keeps anyway, and the largest of those products is the token: about
//! forty rows of 151,936 at the median, 6 KB each.
//!
//! So the token is the dense head's first, where the quantized head's first
//! is another token at one or two positions in a hundred. What stands between
//! the two stages is a margin and not a proof: a row further from its sketch
//! than the margin allows, in the direction of the input, would be passed
//! over, and the token would be the best of the rows that were read. The
//! margin is twelve of a row's typical errors ([`BOUND`]). Over two experts'
//! 5,021 held-out positions the first token's product was never more than 5.4
//! of them above its sketched one, and the token was the dense rows' first at
//! every position.
//!
//! It is taken where a token is the largest logit and nothing else is asked
//! of the logits: greedy decoding with no penalty on, on a CPU that
//! multiplies six-bit tiles, whose instructions cut the sketch's values out
//! of their bytes as they cut a six-bit tile's. Sampling, penalties, and
//! everything that reads log-probabilities keep the whole head.
//!
//! The codes lie in their bytes as the CPU cuts them out ([`Layout`]): three
//! bits each where AVX-512 VBMI or NEON does, and a nibble each, 117 MB for
//! that head, where AVX2 does with a mask and a shift. They are the same
//! codes on the same scales either way, so the products are the same to the
//! bit, the same rows are read whole, and the token is the same token.
//!
//! On one laptop CPU a 1.5B base decodes 7 to 10% more tokens a second for
//! it: 29.9 where it decoded 27.9 at Q6_K, 38.7 where 36.1 at Q4_K, and 23.5
//! where 21.4 at Q8_0, medians of twelve rounds. With an expert on: 28.8
//! where 26.4, 37.3 where 34.5, and 21.7 where 20.7.

use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

use candle_core::{bail, DType, Result, Storage, Tensor};

use super::phases::{dispatch, Phase, Shared};
use super::slices::{dot, values};
use super::tiles::{self, Row, ROWS};

/// Bits a sketched weight is held in.
const BITS: usize = 3;
/// The levels those bits name: a sketched weight is its row's scale times
/// `2 * code + 1 - LEVELS`, the odd whole numbers either side of zero.
const LEVELS: usize = 1 << BITS;
/// Values to a block of quantized input ([`tiles::Row`]).
const BLOCK: usize = 256;
/// Bytes two rows' four values take: the fields of one 64-bit lane.
const PAIR: usize = BITS;
/// Bytes a group of four values takes for sixteen rows.
const GROUP: usize = ROWS / 2 * PAIR;
/// Bytes a group of four values takes for sixteen rows where a code has a
/// nibble to itself: a byte for two rows' codes of one value.
const NIBBLE_GROUP: usize = ROWS / 2 * 4;
/// Bytes past the last group that a load of a whole vector may reach into.
const SLACK: usize = 64;

/// How a sketch's codes lie in their bytes: the way the kernel that reads
/// them cuts them out. A sketch is made on the CPU that reads it.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Layout {
    /// `[sixteen rows][group of 4 values][pair of rows (8)][PAIR bytes]`: a
    /// pair's eight values in `8 * BITS` bits, the first row's four lowest
    /// and each row's first value lowest. What AVX-512 VBMI and NEON cut
    /// out.
    Fields,
    /// `[sixteen rows][group of 4 values][row of the first eight (8)][value
    /// (4)]`: a byte holds its row's code in its low nibble and, in its high
    /// one, the code of the row eight after it. What AVX2 cuts out.
    Nibbles,
}

impl Layout {
    /// The layout this CPU's kernel reads.
    fn here() -> Self {
        #[cfg(target_arch = "x86_64")]
        if !tiles::wide_six_bit() {
            return Self::Nibbles;
        }
        Self::Fields
    }

    /// Bytes a group of four values takes for sixteen rows.
    fn group(self) -> usize {
        match self {
            Self::Fields => GROUP,
            Self::Nibbles => NIBBLE_GROUP,
        }
    }
}

/// The steps tried for a row, in units of the root mean square of its
/// values: the one that leaves the least of the row outside its sketch is
/// kept. About the best uniform step for values that fall as a bell curve
/// does, and some either side for the rows that do not.
const STEPS: [f32; 5] = [0.48, 0.54, 0.60, 0.66, 0.72];

/// How many times a row's typical error its sketched product is allowed to
/// fall short by. A row's typical error for an input is its distance from
/// its sketch times the input's size over the root of its width: what the
/// product of the difference would come to if the input pointed nowhere in
/// particular. A real input does point somewhere, and the first token's
/// product was seen up to 5.4 typical errors above its sketched one; at
/// twelve a token reads a hundred or so rows whole on average, under a
/// megabyte beside the sketch's 88.
const BOUND: f32 = 12.0;

/// Sixteens whose largest sketched product is read whole first, to have a
/// product every other row must reach. One thread reads them while the rest
/// wait, a dense row from memory each, so they are few: the first of them is
/// the token itself at most positions, and a product a little short of the
/// best costs the second stage a few rows more, which every core shares.
const FIRST: usize = 8;

/// Sixteens a claim of the second stage covers. Most hold no row to read
/// whole and go by in a few instructions; the few that do are a dense row
/// from memory each, and a claim small enough spreads them over the cores.
const DECIDED_AT_ONCE: usize = 16;

/// The head's rows as a sketch: `rows` by `cols`.
// Where nothing is sketched, nothing reads a sketch.
#[cfg_attr(
    not(any(target_arch = "x86_64", target_arch = "aarch64")),
    allow(dead_code)
)]
pub(super) struct Sketch {
    /// Every row's codes, laid out as `layout` says. A sixteen's groups one
    /// after another: the order the kernel reads them in.
    codes: Vec<u8>,
    layout: Layout,
    /// Each row's scale.
    scales: Vec<f32>,
    /// How far each row's sketched product may fall short of its own, for an
    /// input of size one: [`BOUND`] typical errors.
    reach: Vec<f32>,
    rows: usize,
    cols: usize,
}

/// What a token's two stages are worked out in, kept from one token to the
/// next.
#[derive(Default)]
pub(super) struct Room {
    /// Every row's sketched product.
    products: Vec<f32>,
    /// Each sixteen's largest sketched product, then the largest product of
    /// the rows of it read whole.
    tops: Vec<f32>,
    /// Where in its sixteen that row is.
    lanes: Vec<u8>,
    /// Rows read whole for the last token.
    read: usize,
}

impl Room {
    /// How many dense rows the last token read whole.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn read(&self) -> usize {
        self.read
    }
}

/// The whole number a code stands for.
fn level(code: u8) -> i32 {
    2 * i32::from(code) + 1 - LEVELS as i32
}

/// The code of the level nearest `w`, the levels `unit` times the odd whole
/// numbers.
fn code_of(w: f32, unit: f32) -> u8 {
    // Halfway between two levels is a whole multiple of `2 * unit`; at and
    // past the outermost levels a value keeps the outermost code. A value
    // that is not a number takes the lowest.
    let at = (w / (2.0 * unit) + (LEVELS / 2) as f32)
        .max(0.0)
        .min(LEVELS as f32 - 0.5);
    at as u8
}

/// For `w` at the levels `unit` apart: the sum of its levels' squares, and of
/// its levels times its values.
fn fit(w: &[f32], unit: f32) -> (f64, f64) {
    const LANES: usize = 8;
    let (mut squares, mut products) = ([0f64; LANES], [0f64; LANES]);
    let chunks = w.chunks_exact(LANES);
    let rest = chunks.remainder();
    for values in chunks {
        for ((value, square), product) in values.iter().zip(&mut squares).zip(&mut products) {
            let l = f64::from(level(code_of(*value, unit)));
            *square += l * l;
            *product += l * f64::from(*value);
        }
    }
    let (mut ss, mut sw) = (squares.iter().sum::<f64>(), products.iter().sum::<f64>());
    for value in rest {
        let l = f64::from(level(code_of(*value, unit)));
        ss += l * l;
        sw += l * f64::from(*value);
    }
    (ss, sw)
}

/// One row sketched: its codes into `codes`, and what comes back is its
/// scale and its distance from the row those codes and that scale say.
///
/// The scale is not the one that brings the sketched row closest. It is the
/// one that leaves the difference at right angles to the row itself: a
/// sketch closest to its row is shorter than the row, so an input that points
/// along a row, as the last hidden row points along the row of the token it
/// means, has a sketched product that falls short every time, and the margin
/// would be spent on that. At right angles the difference has nothing along
/// the row, and what is left of it is as likely to add as to take away.
fn sketch_row(w: &[f32], codes: &mut [u8]) -> (f32, f32) {
    let ww: f64 = w.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
    if !ww.is_finite() {
        // A row that is not numbers has no sketch: it is read whole.
        codes.fill(0);
        return (0.0, f32::INFINITY);
    }
    if ww == 0.0 {
        codes.fill(0);
        return (0.0, 0.0);
    }
    let rms = (ww / w.len() as f64).sqrt() as f32;
    let mut best = (f64::INFINITY, 0f32);
    for step in STEPS {
        let unit = step * rms / 2.0;
        let (ss, sw) = fit(w, unit);
        // What of the row stays outside the best multiple of these levels.
        let left = ww - sw * sw / ss;
        if left < best.0 {
            best = (left, unit);
        }
    }
    let unit = best.1;
    for (code, value) in codes.iter_mut().zip(w) {
        *code = code_of(*value, unit);
    }
    let (ss, sw) = fit(w, unit);
    if sw <= 0.0 {
        // Levels that do not point the row's way: no scale brings them near.
        codes.fill(0);
        return (0.0, f32::INFINITY);
    }
    let scale = ww / sw;
    let off = (ww - 2.0 * scale * sw + scale * scale * ss).max(0.0).sqrt();
    (scale as f32, off as f32)
}

/// Sixteen rows' codes, a row after another in `held`, into the bytes of
/// their sixteen.
fn pack(layout: Layout, held: &[u8], cols: usize, codes: &mut [u8]) {
    match layout {
        Layout::Fields => {
            for (g, group) in codes.chunks_exact_mut(GROUP).enumerate() {
                for (p, pair) in group.chunks_exact_mut(PAIR).enumerate() {
                    let mut bits = 0u64;
                    for k in 0..4 {
                        let low = u64::from(held[2 * p * cols + 4 * g + k]);
                        let high = u64::from(held[(2 * p + 1) * cols + 4 * g + k]);
                        bits |= low << (BITS * k) | high << (BITS * (4 + k));
                    }
                    pair.copy_from_slice(&bits.to_le_bytes()[..PAIR]);
                }
            }
        }
        Layout::Nibbles => {
            for (g, group) in codes.chunks_exact_mut(NIBBLE_GROUP).enumerate() {
                for (r, row) in group.chunks_exact_mut(4).enumerate() {
                    for (k, byte) in row.iter_mut().enumerate() {
                        let low = held[r * cols + 4 * g + k];
                        let high = held[(r + ROWS / 2) * cols + 4 * g + k];
                        *byte = low | high << 4;
                    }
                }
            }
        }
    }
}

/// A run of sixteens sketched: `values` are their dense rows.
fn sketch_rows(
    layout: Layout,
    values: &[f32],
    cols: usize,
    codes: &mut [u8],
    scales: &mut [f32],
    reach: &mut [f32],
) {
    let per = cols / 4 * layout.group();
    let margin = BOUND / (cols as f32).sqrt();
    let mut held = vec![0u8; ROWS * cols];
    for (t, sixteen) in values.chunks_exact(ROWS * cols).enumerate() {
        for (r, row) in sixteen.chunks_exact(cols).enumerate() {
            let (scale, off) = sketch_row(row, &mut held[r * cols..(r + 1) * cols]);
            scales[t * ROWS + r] = scale;
            reach[t * ROWS + r] = off * margin;
        }
        pack(layout, &held, cols, &mut codes[t * per..(t + 1) * per]);
    }
}

/// A product and the token it is of, as one word two threads may pass.
fn word(product: f32, token: u32) -> u64 {
    (u64::from(product.to_bits()) << 32) | u64::from(token)
}

fn of_word(word: u64) -> (f32, u32) {
    (f32::from_bits((word >> 32) as u32), word as u32)
}

/// Where one input's sketched products go, a value for every row of the
/// head, and each sixteen's largest of them.
#[derive(Clone, Copy)]
struct Places {
    products: *mut f32,
    tops: *mut f32,
}

// SAFETY: the pointers are only written through, at the places of sixteens
// no two units of work share, for as long as the rooms they point into are
// borrowed.
unsafe impl Sync for Places {}

/// The sixteens `first..last` of the sketch against every one of `rows`,
/// each row's products of them and their largest to its own of `places`.
/// Off x86-64 one row goes through the one-row kernel, and two or more
/// through the kernel that takes up to four rows to a cutting of the codes.
///
/// SAFETY (for the caller): as the kernels' `sixteens`, for every row and
/// its places.
#[cfg(not(target_arch = "x86_64"))]
unsafe fn sixteens_of_each(
    sketch: &Sketch,
    first: usize,
    last: usize,
    rows: &[Row],
    places: &[Places],
) {
    match (rows, places) {
        ([row], [place]) => kernels::sixteens(sketch, first, last, row, place.products, place.tops),
        _ => kernels::sixteens_of_each(sketch, first, last, rows, places),
    }
}

/// The sixteens `first..last` of the sketch against every one of `rows`,
/// each row's products of them and their largest to its own of `places`,
/// through the kernel of the sketch's layout: AVX-512's, which takes up to
/// eight rows to a cutting of the codes, or AVX2's, which takes up to four.
///
/// SAFETY (for the caller): as the kernel's own, on a CPU [`available`]
/// found the features of.
#[cfg(target_arch = "x86_64")]
unsafe fn sixteens_of_each(
    sketch: &Sketch,
    first: usize,
    last: usize,
    rows: &[Row],
    places: &[Places],
) {
    match sketch.layout {
        Layout::Fields => avx512::sixteens_of_each(sketch, first, last, rows, places),
        Layout::Nibbles => avx2::sixteens_of_each(sketch, first, last, rows, places),
    }
}

/// Which of a sixteen's rows could still be first: those whose sketched
/// product, and their reach for an input of this `size`, come to `reached`
/// or more.
///
/// SAFETY (for the caller): `products` and `reach` each point at sixteen
/// values, on a CPU [`available`] found the features of.
unsafe fn could_be_first(
    sketch: &Sketch,
    products: *const f32,
    reach: *const f32,
    size: f32,
    reached: f32,
) -> u16 {
    #[cfg(target_arch = "x86_64")]
    {
        match sketch.layout {
            Layout::Fields => avx512::could_be_first(products, reach, size, reached),
            Layout::Nibbles => avx2::could_be_first(products, reach, size, reached),
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = sketch;
        kernels::could_be_first(products, reach, size, reached)
    }
}

/// Whether this CPU multiplies a sketch of the head: the CPUs that multiply
/// six-bit tiles, whose instructions cut the sketch's values out as well.
pub(super) fn available() -> bool {
    tiles::six_bit()
}

impl Sketch {
    /// The dense head `w` sketched, on every core, when its sketch can be
    /// multiplied here: plain f32 rows in host memory, in sixteens, their
    /// columns in blocks of 256, on a CPU that multiplies six-bit tiles.
    /// `None` otherwise.
    pub(super) fn of(w: &Tensor) -> Result<Option<Self>> {
        let (rows, cols) = match w.dims() {
            [rows, cols] => (*rows, *cols),
            _ => return Ok(None),
        };
        let fits = available()
            && rows != 0
            && rows.is_multiple_of(ROWS)
            && cols != 0
            && cols.is_multiple_of(BLOCK)
            && w.device().is_cpu()
            && w.dtype() == DType::F32
            && w.is_contiguous();
        if !fits {
            return Ok(None);
        }
        let (storage, held) = w.storage_and_layout();
        let Storage::Cpu(storage) = &*storage else {
            bail!("a head is sketched from host memory")
        };
        let values = values(storage, held, "a head to sketch")?;
        let layout = Layout::here();
        let (sixteens, per) = (rows / ROWS, cols / 4 * layout.group());
        let mut codes = vec![0u8; sixteens * per + SLACK];
        let mut scales = vec![0f32; rows];
        let mut reach = vec![0f32; rows];
        let threads = std::thread::available_parallelism().map_or(1, usize::from);
        // Sixteens a thread takes, about as many for each.
        let run = sixteens.div_ceil(threads).max(1);
        std::thread::scope(|scope| {
            let handles: Vec<_> = codes[..sixteens * per]
                .chunks_mut(run * per)
                .zip(scales.chunks_mut(run * ROWS))
                .zip(reach.chunks_mut(run * ROWS))
                .zip(values.chunks(run * ROWS * cols))
                .map(|(((codes, scales), reach), values)| {
                    scope.spawn(move || sketch_rows(layout, values, cols, codes, scales, reach))
                })
                .collect();
            for handle in handles {
                if handle.join().is_err() {
                    bail!("a thread sketching the head panicked")
                }
            }
            Ok(())
        })?;
        Ok(Some(Self {
            codes,
            layout,
            scales,
            reach,
            rows,
            cols,
        }))
    }

    /// Row `r` of sixteen `t`'s code for value `i`.
    #[cfg(test)]
    fn code(&self, t: usize, r: usize, i: usize) -> u8 {
        let group = (t * (self.cols / 4) + i / 4) * self.layout.group();
        match self.layout {
            Layout::Fields => {
                let at = group + r / 2 * PAIR;
                let mut bytes = [0u8; 8];
                bytes[..PAIR].copy_from_slice(&self.codes[at..at + PAIR]);
                let bits = u64::from_le_bytes(bytes);
                ((bits >> (BITS * (4 * (r % 2) + i % 4))) & (LEVELS as u64 - 1)) as u8
            }
            Layout::Nibbles => {
                let byte = self.codes[group + r % (ROWS / 2) * 4 + i % 4];
                (byte >> (4 * (r / (ROWS / 2)))) & 0xF
            }
        }
    }

    /// Sixteen `t`'s sketched products with `row`, a value at a time: what
    /// the kernel is held to, to the bit.
    #[cfg(test)]
    fn plainly(&self, t: usize, row: &Row) -> [f32; ROWS] {
        std::array::from_fn(|r| {
            let mut acc = 0f32;
            for b in 0..self.cols / BLOCK {
                let (act, d, sum) = row.block(b);
                let isum: i32 = act
                    .iter()
                    .enumerate()
                    .map(|(i, a)| i32::from(self.code(t, r, b * BLOCK + i)) * i32::from(*a))
                    .sum();
                let centered = 2 * isum - (LEVELS as i32 - 1) * sum;
                acc = (centered as f32).mul_add(d, acc);
            }
            acc * self.scales[t * ROWS + r]
        })
    }

    /// The token whose dense row's product with `x` is the largest, the
    /// first of them where several are: `row` is `x` quantized, and `dense`
    /// the rows this was sketched from, a row after another.
    pub(super) fn first(
        &self,
        row: &Row,
        x: &[f32],
        dense: &[f32],
        room: &mut Room,
    ) -> Result<u32> {
        let mut token = [0u32];
        self.first_of_each(
            std::slice::from_ref(row),
            x,
            dense,
            std::slice::from_mut(room),
            &mut token,
        )?;
        Ok(token[0])
    }

    /// [`Sketch::first`] for several inputs at once: `rows` are the rows of
    /// `xs` quantized, a room for each, and each one's token is written to
    /// `tokens`. Each token is the one its input gets alone. The sketch is
    /// read from memory once for all of them: a run of sixteens goes by for
    /// every input before the next run is read, its codes cut out of their
    /// bytes once for as many inputs as the kernel takes together, and the
    /// two stages are one dispatch whatever the number of inputs.
    pub(super) fn first_of_each(
        &self,
        rows: &[Row],
        xs: &[f32],
        dense: &[f32],
        rooms: &mut [Room],
        tokens: &mut [u32],
    ) -> Result<()> {
        let n = rows.len();
        if n == 0
            || xs.len() != n * self.cols
            || rooms.len() != n
            || tokens.len() != n
            || rows.iter().any(|row| !row.holds(self.cols))
            || dense.len() != self.rows * self.cols
        {
            bail!(
                "a sketch of {} by {} against {n} rows, {} values of them, {} rooms and {} dense values",
                self.rows,
                self.cols,
                xs.len(),
                rooms.len(),
                dense.len()
            )
        }
        let sixteens = self.rows / ROWS;
        /// One input's two stages.
        struct Of<'a> {
            x: &'a [f32],
            size: f32,
            products: Shared<f32>,
            tops: Shared<f32>,
            lanes: Shared<u8>,
            /// The largest product found by reading a row whole, and its
            /// token.
            found: AtomicU64,
            reached: AtomicU32,
            read: AtomicUsize,
        }
        let each: Vec<Of> = xs
            .chunks_exact(self.cols)
            .zip(rooms.iter_mut())
            .map(|(x, room)| {
                room.products.resize(self.rows, 0.0);
                room.tops.resize(sixteens, 0.0);
                room.lanes.resize(sixteens, 0);
                Of {
                    x,
                    size: dot(x, x).sqrt(),
                    products: Shared::of(&mut room.products),
                    tops: Shared::of(&mut room.tops),
                    lanes: Shared::of(&mut room.lanes),
                    found: AtomicU64::new(word(f32::NEG_INFINITY, 0)),
                    reached: AtomicU32::new(f32::NEG_INFINITY.to_bits()),
                    read: AtomicUsize::new(0),
                }
            })
            .collect();
        let whole =
            |of: &Of, token: usize| dot(&dense[token * self.cols..(token + 1) * self.cols], of.x);
        let places: Vec<Places> = each
            .iter()
            .map(|of| Places {
                products: of.products.at(),
                tops: of.tops.at(),
            })
            .collect();

        let sketched = |first: usize, last: usize| {
            // SAFETY: a sketch is only made where `available` found the
            // features the kernel is built for; each row holds this sketch's
            // columns, checked above; the rooms were sized just above; and a
            // sixteen is done by one thread.
            unsafe { sixteens_of_each(self, first, last, rows, &places) };
        };
        let first_read = || {
            for of in &each {
                // SAFETY: every sixteen's products and their largest were
                // written in the phase before, and nothing writes them until
                // this is done.
                let (products, tops) = unsafe { (of.products.all(), of.tops.all()) };
                // The sixteens with the largest sketched products.
                let mut held = [(f32::NEG_INFINITY, 0usize); FIRST];
                let n = FIRST.min(sixteens);
                let mut least = 0;
                for (t, top) in tops.iter().enumerate() {
                    // A largest that is not a number is no largest.
                    let top = if top.is_nan() {
                        f32::NEG_INFINITY
                    } else {
                        *top
                    };
                    if t < n {
                        held[t] = (top, t);
                    } else if top > held[least].0 {
                        held[least] = (top, t);
                    } else {
                        continue;
                    }
                    if t + 1 >= n {
                        least = (0..n).fold(0, |least, i| match held[i].0 < held[least].0 {
                            true => i,
                            false => least,
                        });
                    }
                }
                let mut best = (f32::NEG_INFINITY, 0u32);
                let mut whole_rows = 0;
                for (top, t) in &held[..n] {
                    let sixteen = &products[t * ROWS..(t + 1) * ROWS];
                    let Some(lane) = sixteen.iter().position(|p| p == top) else {
                        continue;
                    };
                    let token = t * ROWS + lane;
                    let product = whole(of, token);
                    whole_rows += 1;
                    if product > best.0 || (product == best.0 && (token as u32) < best.1) {
                        best = (product, token as u32);
                    }
                }
                of.read.fetch_add(whole_rows, Ordering::Relaxed);
                of.found.store(word(best.0, best.1), Ordering::Release);
                of.reached.store(best.0.to_bits(), Ordering::Release);
            }
        };
        let decided = |first: usize, last: usize| {
            for of in &each {
                let reached = f32::from_bits(of.reached.load(Ordering::Acquire));
                let mut whole_rows = 0;
                for t in first..last {
                    // SAFETY: the sketched products were all written two
                    // steps back and nothing writes them now; this sixteen's
                    // place among this input's tops and lanes is this unit's
                    // alone, and a unit is done by one thread.
                    let (products, top, lane) = unsafe {
                        (
                            of.products.at().add(t * ROWS),
                            of.tops.part(t, 1),
                            of.lanes.part(t, 1),
                        )
                    };
                    // SAFETY: as above, and `reach` has a value for every row.
                    let mut could = unsafe {
                        could_be_first(
                            self,
                            products,
                            self.reach.as_ptr().add(t * ROWS),
                            of.size,
                            reached,
                        )
                    };
                    let mut best = (f32::NEG_INFINITY, 0u8);
                    while could != 0 {
                        let r = could.trailing_zeros() as usize;
                        could &= could - 1;
                        let product = whole(of, t * ROWS + r);
                        whole_rows += 1;
                        if product > best.0 {
                            best = (product, r as u8);
                        }
                    }
                    top[0] = best.0;
                    lane[0] = best.1;
                }
                of.read.fetch_add(whole_rows, Ordering::Relaxed);
            }
        };
        let per_sixteen = self.cols / 4 * self.layout.group();
        dispatch(
            &[
                Phase {
                    units: sixteens,
                    most: (tiles::CLAIM / per_sixteen).max(1),
                    run: &sketched,
                    then: Some(&first_read),
                },
                Phase {
                    units: sixteens,
                    most: DECIDED_AT_ONCE,
                    run: &decided,
                    then: None,
                },
            ],
            false,
        );
        let found: Vec<(u64, usize)> = each
            .iter()
            .map(|of| {
                (
                    of.found.load(Ordering::Acquire),
                    of.read.load(Ordering::Relaxed),
                )
            })
            .collect();
        drop(each);
        for ((room, token), (found, read)) in rooms.iter_mut().zip(tokens).zip(found) {
            room.read = read;
            let (mut product, mut first) = of_word(found);
            for (t, (top, lane)) in room.tops.iter().zip(&room.lanes).enumerate() {
                let at = (t * ROWS + usize::from(*lane)) as u32;
                if *top > product || (*top == product && at < first) {
                    (product, first) = (*top, at);
                }
            }
            *token = first;
        }
        Ok(())
    }
}

#[cfg(target_arch = "aarch64")]
#[path = "sketch_neon.rs"]
mod kernels;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
mod kernels {
    use super::{Places, Row, Sketch};

    /// Nothing is sketched on other CPUs, so nothing reaches these.
    pub(super) unsafe fn sixteens(
        _: &Sketch,
        _: usize,
        _: usize,
        _: &Row,
        _: *mut f32,
        _: *mut f32,
    ) {
    }

    pub(super) unsafe fn sixteens_of_each(_: &Sketch, _: usize, _: usize, _: &[Row], _: &[Places]) {
    }

    pub(super) unsafe fn could_be_first(_: *const f32, _: *const f32, _: f32, _: f32) -> u16 {
        0
    }
}

#[cfg(target_arch = "x86_64")]
#[path = "sketch_avx2.rs"]
mod avx2;

#[cfg(target_arch = "x86_64")]
mod avx512 {
    use super::{Places, Row, Sketch, BITS, BLOCK, GROUP, LEVELS, PAIR, ROWS};
    use std::arch::x86_64::*;

    /// Where each byte of a group goes for its fields to be cut out: a pair
    /// of rows' bytes to the low bytes of its 64-bit lane, the last of them
    /// repeated above, where it is not read.
    const SPREAD: [u8; 64] = {
        let mut to = [0u8; 64];
        let mut i = 0;
        while i < 64 {
            let (pair, byte) = (i / 8, i % 8);
            to[i] = (pair * PAIR + if byte < PAIR { byte } else { PAIR - 1 }) as u8;
            i += 1;
        }
        to
    };

    /// The bit each of a lane's eight fields starts at.
    const FIELDS: i64 = {
        let mut at = [0u8; 8];
        let mut i = 0;
        while i < 8 {
            at[i] = (i * BITS) as u8;
            i += 1;
        }
        i64::from_le_bytes(at)
    };

    /// Four input values, repeated across the sixteen rows.
    #[target_feature(enable = "avx512f")]
    unsafe fn four(act: *const i8, at: usize) -> __m512i {
        _mm512_set1_epi32(std::ptr::read_unaligned(act.add(at) as *const i32))
    }

    /// Sixteen `t` of the sketch against `N` rows of input at once: each
    /// row's sixteen products, as it would have them alone, to the bit. A
    /// sketched weight is its row's scale times `2 * code + 1 - LEVELS`, so a
    /// product is `2 * code · input` less `LEVELS - 1` times the input's own
    /// sum, in whole numbers over each block of 256, then by the block's
    /// scale, and by the row's at the end.
    ///
    /// A group's codes are cut out of their bytes once for every row of
    /// input. Each row keeps `SUMS` running sums that take a group in turn,
    /// so the wait for one sum is spent on the others: with one row that is
    /// four sums, with eight rows the eight rows' own.
    ///
    /// SAFETY (for the caller): the CPU has the features named, `rows` holds
    /// `N` rows of the sketch's columns, and the sixteen is among the
    /// sketch's own.
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx512vbmi")]
    unsafe fn sixteen_of<const N: usize, const SUMS: usize>(
        sketch: &Sketch,
        t: usize,
        rows: &[Row],
    ) -> [__m512; N] {
        let spread = _mm512_loadu_si512(SPREAD.as_ptr() as *const __m512i);
        let fields = _mm512_set1_epi64(FIELDS);
        let code = _mm512_set1_epi8(LEVELS as i8 - 1);
        let per = sketch.cols / 4 * GROUP;
        let mut codes = sketch.codes.as_ptr().add(t * per);
        let mut acc = [_mm512_setzero_ps(); N];
        for b in 0..sketch.cols / BLOCK {
            let blocks: [(*const i8, f32, i32); N] = std::array::from_fn(|i| {
                let (act, d, total) = rows[i].block(b);
                (act.as_ptr(), d, total)
            });
            let mut sums = [[_mm512_setzero_si512(); SUMS]; N];
            for g in 0..BLOCK / 4 {
                // A group is fewer bytes than the vector loaded: what is
                // past it is the next group's, or the slack kept after the
                // last, and `spread` reads none of it.
                let packed = _mm512_loadu_si512(codes.add(g * GROUP) as *const __m512i);
                let q = _mm512_and_si512(
                    _mm512_multishift_epi64_epi8(fields, _mm512_permutexvar_epi8(spread, packed)),
                    code,
                );
                for (sums, (act, _, _)) in sums.iter_mut().zip(&blocks) {
                    let sum = &mut sums[g % SUMS];
                    *sum = _mm512_dpbusd_epi32(*sum, q, four(*act, 4 * g));
                }
            }
            codes = codes.add(BLOCK / 4 * GROUP);
            for ((acc, sums), (_, d, total)) in acc.iter_mut().zip(&sums).zip(&blocks) {
                let mut isum = sums[0];
                for sum in &sums[1..] {
                    isum = _mm512_add_epi32(isum, *sum);
                }
                let centered = _mm512_sub_epi32(
                    _mm512_slli_epi32::<1>(isum),
                    _mm512_set1_epi32((LEVELS as i32 - 1) * total),
                );
                *acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(centered), _mm512_set1_ps(*d), *acc);
            }
        }
        let scales = _mm512_loadu_ps(sketch.scales.as_ptr().add(t * ROWS));
        for acc in acc.iter_mut() {
            *acc = _mm512_mul_ps(*acc, scales);
        }
        acc
    }

    /// The sixteens `first..last` of the sketch against every one of
    /// `rows`: each row's products of them, and the largest of each
    /// sixteen's, to its own of `places`. The rows are taken eight, four,
    /// two and one at a time, a sixteen's codes cut out of their bytes once
    /// for each such run of them.
    ///
    /// SAFETY (for the caller): the CPU has the features named, every row
    /// holds the sketch's columns, each of `places` has room for a product
    /// of every row of the head and a largest for every sixteen, the
    /// sixteens are among the sketch's own, and no other thread writes
    /// their places.
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx512vbmi")]
    pub(super) unsafe fn sixteens_of_each(
        sketch: &Sketch,
        first: usize,
        last: usize,
        rows: &[Row],
        places: &[Places],
    ) {
        for t in first..last {
            let mut done = 0;
            while done < rows.len().min(places.len()) {
                let mut out = [_mm512_setzero_ps(); 8];
                let n = match rows.len().min(places.len()) - done {
                    8.. => {
                        out = sixteen_of::<8, 1>(sketch, t, &rows[done..]);
                        8
                    }
                    4.. => {
                        out[..4].copy_from_slice(&sixteen_of::<4, 2>(sketch, t, &rows[done..]));
                        4
                    }
                    2.. => {
                        out[..2].copy_from_slice(&sixteen_of::<2, 4>(sketch, t, &rows[done..]));
                        2
                    }
                    _ => {
                        out[..1].copy_from_slice(&sixteen_of::<1, 4>(sketch, t, &rows[done..]));
                        1
                    }
                };
                for (v, place) in out[..n].iter().zip(&places[done..]) {
                    _mm512_storeu_ps(place.products.add(t * ROWS), *v);
                    *place.tops.add(t) = _mm512_reduce_max_ps(*v);
                }
                done += n;
            }
        }
    }

    /// Which of a sixteen's rows could still be first: those whose sketched
    /// product, and their reach for an input of this `size`, come to
    /// `reached` or more.
    ///
    /// SAFETY (for the caller): the CPU has AVX-512F, and `products` and
    /// `reach` each point at sixteen values.
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn could_be_first(
        products: *const f32,
        reach: *const f32,
        size: f32,
        reached: f32,
    ) -> u16 {
        let bound = _mm512_fmadd_ps(
            _mm512_loadu_ps(reach),
            _mm512_set1_ps(size),
            _mm512_loadu_ps(products),
        );
        _mm512_cmp_ps_mask::<_CMP_GE_OQ>(bound, _mm512_set1_ps(reached))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use rand::rngs::StdRng;
    use rand::{RngExt as _, SeedableRng};

    /// Standard normals from a seed: the same in every run.
    fn normals(seed: u64, n: usize) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                // Box-Muller: two uniforms make one standard normal.
                let u1: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
                let u2: f64 = rng.random::<f64>();
                ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
            })
            .collect()
    }

    fn sketched(w: &[f32], rows: usize, cols: usize) -> Result<Option<Sketch>> {
        Sketch::of(&Tensor::from_slice(w, (rows, cols), &Device::Cpu)?)
    }

    fn quantized(x: &[f32]) -> Row {
        let mut row = Row::default();
        tiles::quantize_into(x, &mut row);
        row
    }

    /// The first place the largest of the dense rows' products is at.
    fn dense_first(w: &[f32], cols: usize, x: &[f32]) -> u32 {
        let mut best = (f32::NEG_INFINITY, 0u32);
        for (token, row) in w.chunks_exact(cols).enumerate() {
            let product = dot(row, x);
            if product > best.0 {
                best = (product, token as u32);
            }
        }
        best.1
    }

    /// A value takes the nearest level, the outermost past either end, and
    /// the levels are the odd whole numbers either side of zero.
    #[test]
    fn a_value_takes_the_nearest_level() -> Result<()> {
        let top = LEVELS as i32 - 1;
        assert_eq!(level(0), -top);
        assert_eq!(level(LEVELS as u8 - 1), top);
        for (value, want) in [
            (0.1f32, 1),
            (-0.1, -1),
            (1.9, 1),
            (2.1, 3),
            (-2.1, -3),
            (1e9, top),
            (-1e9, -top),
            (f32::NAN, -top),
        ] {
            assert_eq!(level(code_of(value, 1.0)), want, "{value}");
        }
        Ok(())
    }

    /// A row's sketch is the row again to within what its eight levels can
    /// say, the distance that comes back is the distance, and the difference
    /// is at right angles to the row.
    #[test]
    fn a_sketched_row_is_near_its_row_and_level_with_it() -> Result<()> {
        let w = normals(11, 1536);
        let mut codes = vec![0u8; w.len()];
        let (scale, off) = sketch_row(&w, &mut codes);
        let said: Vec<f32> = codes.iter().map(|c| scale * level(*c) as f32).collect();
        let gap: Vec<f32> = w.iter().zip(&said).map(|(a, b)| a - b).collect();
        let (ww, distance) = (dot(&w, &w).sqrt(), dot(&gap, &gap).sqrt());
        assert!(
            (distance - off).abs() < 1e-3 * ww,
            "{distance} against {off}"
        );
        // Eight uniform levels leave about a fifth of a bell curve's size out.
        assert!(off < 0.25 * ww, "{off} of {ww}");
        assert!(dot(&gap, &w).abs() < 1e-3 * ww * ww, "{}", dot(&gap, &w));

        // Rows of nothing, and of what is not a number.
        assert_eq!(sketch_row(&[0.0; 256], &mut [0u8; 256]), (0.0, 0.0));
        let mut broken = w.clone();
        broken[7] = f32::NAN;
        assert_eq!(sketch_row(&broken, &mut codes), (0.0, f32::INFINITY));
        Ok(())
    }

    /// The kernel's products are the ones summed a value at a time, to the
    /// bit, for every row of every sixteen: for one row of input, and for as
    /// many at once as take every width of a kernel that shares the cutting
    /// of the codes (eight, four, two and one).
    #[test]
    fn sketched_products_in_vectors_are_the_plain_ones() -> Result<()> {
        let (rows, cols) = (48, 512);
        let w = normals(3, rows * cols);
        let Some(sketch) = sketched(&w, rows, cols)? else {
            return Ok(());
        };
        for inputs in [1usize, 2, 3, 4, 5, 6, 7, 8, 15] {
            let given: Vec<Row> = (0..inputs)
                .map(|i| quantized(&normals(100 + (inputs * 31 + i) as u64, cols)))
                .collect();
            let mut products = vec![vec![0f32; rows]; inputs];
            let mut tops = vec![vec![0f32; rows / ROWS]; inputs];
            let places: Vec<Places> = products
                .iter_mut()
                .zip(tops.iter_mut())
                .map(|(products, tops)| Places {
                    products: products.as_mut_ptr(),
                    tops: tops.as_mut_ptr(),
                })
                .collect();
            // SAFETY: a sketch was made, so the CPU has the kernel's
            // features; every row holds the sketch's columns and each has
            // room of its own, sized for it.
            unsafe { sixteens_of_each(&sketch, 0, rows / ROWS, &given, &places) };
            for (i, row) in given.iter().enumerate() {
                for t in 0..rows / ROWS {
                    let plain = sketch.plainly(t, row);
                    let got = &products[i][t * ROWS..(t + 1) * ROWS];
                    assert!(
                        plain
                            .iter()
                            .zip(got)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "{inputs} rows, row {i}, sixteen {t}: {got:?} against {plain:?}"
                    );
                    let most = got.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    assert_eq!(tops[i][t].to_bits(), most.to_bits());
                }
            }
        }
        Ok(())
    }

    /// Inputs read in one pass each get the token they get alone, and each
    /// has as many rows read whole for it as it has alone.
    #[test]
    fn inputs_read_together_get_the_tokens_they_get_alone() -> Result<()> {
        let (rows, cols) = (4096, 256);
        let w = normals(5, rows * cols);
        let Some(sketch) = sketched(&w, rows, cols)? else {
            return Ok(());
        };
        for inputs in [2usize, 3, 5, 8, 13] {
            let mut xs = normals(300 + inputs as u64, inputs * cols);
            // Every other input along a row of the head, as a hidden row is.
            for (i, x) in xs
                .chunks_exact_mut(cols)
                .enumerate()
                .filter(|(i, _)| i % 2 == 0)
            {
                let token = (i * 977 + 31 * inputs) % rows;
                for (x, row) in x.iter_mut().zip(&w[token * cols..(token + 1) * cols]) {
                    *x = 0.3 * *x + row;
                }
            }
            let given: Vec<Row> = xs.chunks_exact(cols).map(quantized).collect();
            let (mut alone, mut read_alone) = (Vec::new(), Vec::new());
            let mut room = Room::default();
            for (row, x) in given.iter().zip(xs.chunks_exact(cols)) {
                alone.push(sketch.first(row, x, &w, &mut room)?);
                read_alone.push(room.read());
                assert_eq!(alone.last().copied(), Some(dense_first(&w, cols, x)));
            }
            let mut rooms: Vec<Room> = (0..inputs).map(|_| Room::default()).collect();
            let mut together = vec![0u32; inputs];
            sketch.first_of_each(&given, &xs, &w, &mut rooms, &mut together)?;
            assert_eq!(together, alone, "{inputs} inputs");
            let read: Vec<usize> = rooms.iter().map(Room::read).collect();
            assert_eq!(read, read_alone, "{inputs} inputs: rows read whole");
        }
        Ok(())
    }

    /// The token is the dense rows' first: for inputs that point nowhere in
    /// particular, where most rows have to be read whole, and for inputs
    /// that point along a row, where few do.
    #[test]
    fn the_first_token_is_the_dense_rows_first() -> Result<()> {
        let (rows, cols) = (4096, 256);
        let w = normals(5, rows * cols);
        let Some(sketch) = sketched(&w, rows, cols)? else {
            return Ok(());
        };
        let mut room = Room::default();
        for seed in 0..24u64 {
            let mut x = normals(200 + seed, cols);
            if seed % 2 == 1 {
                // Along one row, as a hidden row is along its token's.
                let token = (seed as usize * 977) % rows;
                for (x, row) in x.iter_mut().zip(&w[token * cols..(token + 1) * cols]) {
                    *x = 0.3 * *x + row;
                }
            }
            let token = sketch.first(&quantized(&x), &x, &w, &mut room)?;
            assert_eq!(token, dense_first(&w, cols, &x), "input {seed}");
            if seed % 2 == 1 {
                assert!(
                    room.read() < rows / 8,
                    "input {seed}: {} rows read whole",
                    room.read()
                );
            }
        }
        Ok(())
    }

    /// Where two rows are the same row, the first of them is the token, as
    /// it is for the whole head.
    #[test]
    fn of_equal_rows_the_first_is_the_token() -> Result<()> {
        let (rows, cols) = (64, 256);
        let mut w = normals(9, rows * cols);
        let x: Vec<f32> = w[40 * cols..41 * cols].to_vec();
        let (low, high) = w.split_at_mut(40 * cols);
        low[17 * cols..18 * cols].copy_from_slice(&high[..cols]);
        let Some(sketch) = sketched(&w, rows, cols)? else {
            return Ok(());
        };
        let mut room = Room::default();
        assert_eq!(sketch.first(&quantized(&x), &x, &w, &mut room)?, 17);
        Ok(())
    }

    /// What does not fit is not sketched, and a sketch refuses a row or
    /// dense rows that are not its own.
    #[test]
    fn what_does_not_fit_is_not_sketched_or_read() -> Result<()> {
        assert!(sketched(&normals(1, 15 * 256), 15, 256)?.is_none());
        assert!(sketched(&normals(1, 16 * 100), 16, 100)?.is_none());
        let w = normals(2, 16 * 256);
        let Some(sketch) = sketched(&w, 16, 256)? else {
            return Ok(());
        };
        let mut room = Room::default();
        let x = normals(3, 256);
        assert!(sketch
            .first(&quantized(&x), &x, &w[..256], &mut room)
            .is_err());
        let wide = normals(4, 512);
        assert!(sketch
            .first(&quantized(&wide), &wide, &w, &mut room)
            .is_err());
        Ok(())
    }
}
