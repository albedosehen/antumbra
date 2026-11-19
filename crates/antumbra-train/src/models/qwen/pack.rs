//! An adapter's factors as rows read on a CPU read them: at half precision, in
//! the order the kernels read them.
//!
//! A decoded token is bound by how fast memory is read, and an adapter's
//! factors are read whole for every token beside the base: 74 MB at rank 16
//! on a 1.5B base, 295 MB at a four-expert blend's rank, in f32. Half of that
//! is precision the token does not use. An expert is trained in bf16, whose
//! eight significant bits a half-precision float holds exactly; and a factor
//! rounded to half precision is off by one part in two thousand at most,
//! where the quantized base under it is off by one part in a hundred. So the
//! step that reads plain rows reads a copy of the factors at half precision.
//!
//! The copy is made when rows are first read after the factors change
//! (`LoraLinear::pack`), and whoever sets the factors lets it go
//! (`QwenLora::adapters_changed`). The trainer's factors stay as they are:
//! training and every GPU path read those.

/// Values to a block of a quantized row.
pub(super) const BLOCK: usize = 256;

/// The nearest half-precision float to `x`, ties to even, as its bits.
pub(super) fn to_half(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7F_FFFF;
    if exponent == 0xFF {
        // Endless, or not a number.
        return sign | 0x7C00 | if mantissa != 0 { 0x200 } else { 0 };
    }
    let exponent = exponent - 127 + 15;
    if exponent >= 0x1F {
        return sign | 0x7C00;
    }
    // What is cut off rounds the rest up when it is more than half a step,
    // or exactly half and the rest is odd.
    let round = |kept: u32, cut: u32, half: u32| {
        kept + u32::from(cut > half || (cut == half && kept & 1 == 1))
    };
    if exponent <= 0 {
        // Below the smallest normal half: a subnormal, or nothing.
        if exponent < -10 {
            return sign;
        }
        let whole = mantissa | 0x80_0000;
        let shift = (14 - exponent) as u32;
        let kept = round(whole >> shift, whole & ((1 << shift) - 1), 1 << (shift - 1));
        return sign | kept as u16;
    }
    // A carry out of the mantissa is the next exponent, and past the largest
    // exponent it is endless: both are what the bits then say.
    let kept = round(
        ((exponent as u32) << 10) | (mantissa >> 13),
        mantissa & 0x1FFF,
        0x1000,
    );
    sign | kept as u16
}

/// The float a half-precision float's bits stand for.
pub(super) fn from_half(half: u16) -> f32 {
    let sign = (u32::from(half) & 0x8000) << 16;
    let exponent = u32::from(half >> 10) & 0x1F;
    let mantissa = u32::from(half & 0x3FF);
    let bits = match (exponent, mantissa) {
        (0, 0) => sign,
        (0, _) => {
            // A subnormal: its leading one becomes the implied one.
            let shift = mantissa.leading_zeros() - 21;
            sign | ((113 - shift) << 23) | (((mantissa << shift) & 0x3FF) << 13)
        }
        (0x1F, _) => sign | 0x7F80_0000 | (mantissa << 13),
        _ => sign | ((exponent + 112) << 23) | (mantissa << 13),
    };
    f32::from_bits(bits)
}

/// `values` at half precision.
fn halved(values: &[f32]) -> Vec<u16> {
    let mut halves = vec![0u16; values.len()];
    #[cfg(target_arch = "x86_64")]
    if super::tiles::available() {
        // SAFETY: `available` found AVX-512F among the features it names.
        unsafe { super::wide::to_halves(values, &mut halves) };
        return halves;
    }
    for (half, value) in halves.iter_mut().zip(values) {
        *half = to_half(*value);
    }
    halves
}

/// Values a vector holds: a sixteen of products, or of the rank rows of `A`.
const SIXTEEN: usize = 16;

/// One projection's adapter at half precision: `scale · B · (A · x)` added
/// to the `rows` products of a row `x` of `cols` values.
///
/// Both factors are laid out so that a vector holds sixteen results at once
/// and nothing is ever summed across a vector's lanes. `A` is held a column
/// at a time, sixteen of its rank rows to a vector: a column's vector times
/// one value of the input adds to all sixteen of those rows' sums. `B` is
/// held a sixteen of products at a time, a vector for each rank: a rank's
/// vector times that rank's value of `A · x` adds to all sixteen products.
/// Held a row of `rank` at a time instead, every product is a sum across a
/// vector, which costs ten times the multiplying it follows.
pub(super) struct Packed {
    /// The ranks that are held: the factors' own, less those at the end
    /// that add nothing.
    pub(super) rank: usize,
    pub(super) scale: f32,
    cols: usize,
    rows: usize,
    /// `A`: for each sixteen of its rank rows (the last filled out with
    /// zeros), for each column, those rows' values of the column.
    a: Vec<u16>,
    /// `B`: for each sixteen of products (the last filled out with zeros),
    /// for each rank, those products' factors for it.
    b: Vec<u16>,
}

impl Packed {
    /// The factors `a` (`rank` rows of `cols`) and `b` (a row of `rank` for
    /// each product) at half precision. `None` when the factors do not fit
    /// one another.
    ///
    /// The ranks at the end that add nothing are left out: a rank adds
    /// nothing when its row of `A` or its column of `B` is all zeros. A
    /// model that serves blends holds factors for as many experts as a blend
    /// may have, and a blend of fewer leaves the rest of its rank at zero.
    /// Read whole, one expert alone would cost every token three experts'
    /// bytes, two of them zeros. What is held gives the products the whole
    /// would give.
    pub(super) fn of(a: &[f32], b: &[f32], rank: usize, cols: usize, scale: f32) -> Option<Self> {
        let fits = a.len() == rank * cols && (rank == 0 || b.len().is_multiple_of(rank));
        if !fits {
            return None;
        }
        let rows = match rank {
            0 => 0,
            rank => b.len() / rank,
        };
        let adds = |r: &usize| {
            a[r * cols..(r + 1) * cols].iter().any(|v| *v != 0.0)
                && b.iter().skip(*r).step_by(rank).any(|v| *v != 0.0)
        };
        let live = (0..rank).rev().find(adds).map_or(0, |r| r + 1);
        let mut by_columns = vec![0f32; live.div_ceil(SIXTEEN) * cols * SIXTEEN];
        for (sixteen, (held, rows)) in by_columns
            .chunks_exact_mut((cols * SIXTEEN).max(1))
            .zip(a[..live * cols].chunks((cols * SIXTEEN).max(1)))
            .enumerate()
        {
            let ranks = (live - sixteen * SIXTEEN).min(SIXTEEN);
            for (c, column) in held.chunks_exact_mut(SIXTEEN).enumerate() {
                for (lane, value) in column[..ranks].iter_mut().enumerate() {
                    *value = rows[lane * cols + c];
                }
            }
        }
        let mut by_sixteens = vec![0f32; rows.div_ceil(SIXTEEN) * live * SIXTEEN];
        for (held, products) in by_sixteens
            .chunks_exact_mut((live * SIXTEEN).max(1))
            .zip(b.chunks((rank * SIXTEEN).max(1)))
        {
            for (r, factors) in held.chunks_exact_mut(SIXTEEN).enumerate() {
                for (factor, product) in factors.iter_mut().zip(products.chunks_exact(rank)) {
                    *factor = product[r];
                }
            }
        }
        Some(Self {
            rank: live,
            scale,
            cols,
            rows,
            a: halved(&by_columns),
            b: halved(&by_sixteens),
        })
    }

    /// Products this adapter adds to.
    pub(super) fn rows(&self) -> usize {
        self.rows
    }

    pub(super) fn cols(&self) -> usize {
        self.cols
    }

    /// `A` over its columns `first..first + n`, times those columns of each
    /// of `rows` rows of input: row `i`'s values of them are
    /// `x[i * stride..][..n]`, and its `rank` sums are written to
    /// `out[i * rank..]`, before the scale. A row's whole `A · x` is these
    /// over all the columns, and it may be had a run of columns at a time
    /// and added up: the MLP's gated row is only ever held a block at a
    /// time. A row's sums are the sums it has alone, to the bit, whatever
    /// rows are beside it.
    pub(super) fn through(
        &self,
        first: usize,
        n: usize,
        x: &[f32],
        stride: usize,
        rows: usize,
        out: &mut [f32],
    ) {
        let rank = self.rank;
        let out = &mut out[..rows * rank];
        if rows == 0 || rank == 0 {
            return;
        }
        if n == 0 || self.cols == 0 {
            out.fill(0.0);
            return;
        }
        for (sixteen, held) in self.a.chunks_exact(self.cols * SIXTEEN).enumerate() {
            let a = &held[first * SIXTEEN..(first + n) * SIXTEEN];
            let (at, ranks) = (sixteen * SIXTEEN, (rank - sixteen * SIXTEEN).min(SIXTEEN));
            if ranks == SIXTEEN {
                sixteen_through(a, n, x, stride, rows, &mut out[at..], rank);
                continue;
            }
            // The last sixteen of a rank that is no multiple of sixteen:
            // through room for whole sixteens.
            let mut whole = vec![0f32; rows * SIXTEEN];
            sixteen_through(a, n, x, stride, rows, &mut whole, SIXTEEN);
            for (i, whole) in whole.chunks_exact(SIXTEEN).enumerate() {
                out[i * rank + at..(i + 1) * rank].copy_from_slice(&whole[..ranks]);
            }
        }
    }

    /// `B`'s rows `first..first + n`, each times a row's `scale · A · x`,
    /// added to that row's products of them, for each of `rows` rows of
    /// input: row `i`'s `rank` values are `t[i * rank..]`, and its products
    /// of those rows of `B` are at `out + i * stride`. `first` is a multiple
    /// of sixteen. A row's products are what it gets alone, to the bit.
    ///
    /// SAFETY (for the caller): each of the `rows` places in `out` holds `n`
    /// values, and nothing else reads or writes them meanwhile.
    pub(super) unsafe fn add(
        &self,
        first: usize,
        n: usize,
        t: &[f32],
        rows: usize,
        out: *mut f32,
        stride: usize,
    ) {
        let rank = self.rank;
        if rank == 0 || rows == 0 || n == 0 {
            return;
        }
        debug_assert!(first.is_multiple_of(SIXTEEN) && first + n <= self.rows);
        let (whole, t) = (n / SIXTEEN, &t[..rows * rank]);
        let b = &self.b
            [first / SIXTEEN * rank * SIXTEEN..(first + n).div_ceil(SIXTEEN) * rank * SIXTEEN];
        sixteens_added(b, rank, whole, t, rows, out, stride);
        let left = n - whole * SIXTEEN;
        if left > 0 {
            // The last few of a count that is no multiple of sixteen:
            // through room for a whole sixteen.
            let b = &b[whole * rank * SIXTEEN..];
            for (i, t) in t.chunks_exact(rank).enumerate() {
                let at = out.add(i * stride + whole * SIXTEEN);
                let mut last = [0f32; SIXTEEN];
                std::ptr::copy_nonoverlapping(at, last.as_mut_ptr(), left);
                sixteens_added(b, rank, 1, t, 1, last.as_mut_ptr(), 0);
                std::ptr::copy_nonoverlapping(last.as_ptr(), at, left);
            }
        }
    }
}

/// Sixteen rank rows of `A` over `n` columns (`a`: a column's sixteen values
/// at a time) times those columns of each of `rows` rows of `x`, `stride`
/// apart: row `i`'s sixteen sums are written to `out[i * out_stride..]`.
fn sixteen_through(
    a: &[u16],
    n: usize,
    x: &[f32],
    stride: usize,
    rows: usize,
    out: &mut [f32],
    out_stride: usize,
) {
    let (a, x) = (&a[..n * SIXTEEN], &x[..(rows - 1) * stride + n]);
    let out = &mut out[..(rows - 1) * out_stride + SIXTEEN];
    #[cfg(target_arch = "x86_64")]
    if super::tiles::available() {
        // SAFETY: `available` found AVX-512F among the features it names,
        // and the three slices were cut just above to what the kernel reads
        // and writes.
        unsafe {
            super::wide::through_cols(
                a.as_ptr(),
                n,
                x.as_ptr(),
                stride,
                rows,
                out.as_mut_ptr(),
                out_stride,
            )
        };
        return;
    }
    for i in 0..rows {
        let sums = &mut out[i * out_stride..i * out_stride + SIXTEEN];
        sums.fill(0.0);
        for (column, x) in a.chunks_exact(SIXTEEN).zip(&x[i * stride..]) {
            for (sum, a) in sums.iter_mut().zip(column) {
                *sum += from_half(*a) * x;
            }
        }
    }
}

/// To `sixteens` sixteens of products of each of `rows` rows at `out`,
/// `stride` apart: `b` (a sixteen at a time, the sixteen's factors for each
/// rank) times that row's `rank` values of `t`.
///
/// SAFETY (for the caller): each of the `rows` places in `out` holds
/// `16 * sixteens` values, and nothing else reads or writes them meanwhile.
unsafe fn sixteens_added(
    b: &[u16],
    rank: usize,
    sixteens: usize,
    t: &[f32],
    rows: usize,
    out: *mut f32,
    stride: usize,
) {
    let (b, t) = (&b[..sixteens * rank * SIXTEEN], &t[..rows * rank]);
    #[cfg(target_arch = "x86_64")]
    if super::tiles::available() {
        // `available` found AVX-512F among the features it names, and the
        // two slices were cut just above to what the kernel reads.
        super::wide::add_sixteens(b.as_ptr(), rank, sixteens, t.as_ptr(), rows, out, stride);
        return;
    }
    for (i, t) in t.chunks_exact(rank).enumerate() {
        for (s, b) in b.chunks_exact(rank * SIXTEEN).enumerate() {
            let products =
                std::slice::from_raw_parts_mut(out.add(i * stride + s * SIXTEEN), SIXTEEN);
            for (factors, t) in b.chunks_exact(SIXTEEN).zip(t) {
                for (product, factor) in products.iter_mut().zip(factors) {
                    *product += from_half(*factor) * t;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Values across a half's whole range, its edges among them.
    fn values() -> Vec<f32> {
        let mut values = vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            65504.0,
            65520.0,
            65519.99,
            1e9,
            -1e9,
            f32::INFINITY,
            f32::NEG_INFINITY,
            6.103_515_6e-5,
            6.1e-5,
            5.960_464_5e-8,
            2.980_232_2e-8,
            2.980_232_4e-8,
            1e-9,
            1.000_488_3,
            1.000_244_1,
            1.000_732_4,
            0.333_333_34,
        ];
        let mut x = 1.7e-8f32;
        while x < 7e4 {
            values.push(x);
            values.push(-x * 1.000_31);
            x *= 1.37;
        }
        values
    }

    /// A float goes to the half nearest it and comes back as that half's
    /// value: what a half holds exactly comes back as it was, and nothing
    /// comes back further off than half a step of the half it went to.
    #[test]
    fn a_float_goes_to_the_nearest_half_and_back() {
        for v in values() {
            let back = from_half(to_half(v));
            if v.abs() >= 65520.0 {
                assert!(
                    back.is_infinite() && back.signum() == v.signum(),
                    "{v} to {back}"
                );
                continue;
            }
            // A half's step: 2^-10 of its power of two, 2^-24 below the
            // smallest normal half.
            let step = (v.abs().max(6.103_515_6e-5).log2().floor() - 10.0).exp2();
            assert!((back - v).abs() <= step / 2.0, "{v} came back as {back}");
            assert_eq!(from_half(to_half(back)), back, "{back} is a half and stays");
        }
        assert!(from_half(to_half(f32::NAN)).is_nan());
        assert_eq!(to_half(1.0), 0x3C00);
        assert_eq!(to_half(-2.0), 0xC000);
        assert_eq!(from_half(0x0001), 5.960_464_5e-8);
        assert_eq!(from_half(0x7BFF), 65504.0);
        // A tie goes to the even half: 1 + 2^-11 is halfway between two.
        assert_eq!(to_half(1.000_488_3), 0x3C00);
        assert_eq!(to_half(1.001_464_8), 0x3C02);
    }

    /// The vectors halve as the plain function does, bit for bit.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn the_vectors_halve_as_the_plain_function_does() {
        if !super::super::tiles::available() {
            eprintln!("skipped: this CPU has no AVX-512");
            return;
        }
        let mut values = values();
        values.retain(|v| !v.is_nan());
        let mut wide = vec![0u16; values.len()];
        // SAFETY: the test returned above without AVX-512.
        unsafe { super::super::wide::to_halves(&values, &mut wide) };
        let plain: Vec<u16> = values.iter().map(|v| to_half(*v)).collect();
        assert_eq!(wide, plain);
    }

    /// The largest gap between two rows of values, over the peak of `want`.
    fn off(got: &[f32], want: &[f32]) -> f32 {
        let peak = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        got.iter()
            .zip(want)
            .fold(0f32, |m, (g, w)| m.max((g - w).abs()))
            / peak
    }

    /// A packed adapter gives `scale · B · (A · x)` as its factors at half
    /// precision do a value at a time, to the order of a float sum, for
    /// ranks and counts of products on and off a multiple of sixteen, with
    /// `A · x` had whole or a block of columns at a time. Rows of input read
    /// together get, to the bit, what each gets alone. And factors that do
    /// not fit are not packed.
    #[test]
    fn a_packed_adapter_is_the_adapter() -> candle_core::Result<()> {
        const INPUTS: usize = 11;
        // What the order of a float sum of a thousand terms can move it by.
        const CLOSE: f32 = 1e-4;
        for (rank, cols, rows) in [
            (8usize, 512usize, 48usize),
            (16, 256, 16),
            (5, 768, 33),
            (40, 512, 64),
            (16, 1300, 80),
        ] {
            // Factors a half holds, so the plain sums below are of the very
            // values the pack holds.
            let a: Vec<f32> = (0..rank * cols)
                .map(|i| from_half(to_half((i as f32 * 0.37).sin() * 0.05)))
                .collect();
            let b: Vec<f32> = (0..rows * rank)
                .map(|i| from_half(to_half((i as f32 * 0.23).cos() * 0.05)))
                .collect();
            let x: Vec<f32> = (0..INPUTS * cols)
                .map(|i| (i as f32 * 0.11).cos() * 3.0)
                .collect();
            let scale = 2.0f32;
            let Some(pack) = Packed::of(&a, &b, rank, cols, scale) else {
                candle_core::bail!("factors that fit were not packed")
            };
            assert_eq!((pack.rows(), pack.cols()), (rows, cols));

            // A value at a time.
            let want_t: Vec<f32> = x
                .chunks_exact(cols)
                .flat_map(|x| {
                    a.chunks_exact(cols)
                        .map(|row| row.iter().zip(x).map(|(a, x)| a * x).sum::<f32>() * scale)
                        .collect::<Vec<f32>>()
                })
                .collect();
            let want: Vec<f32> = want_t
                .chunks_exact(rank)
                .flat_map(|t| {
                    b.chunks_exact(rank)
                        .map(|row| 1.0 + row.iter().zip(t).map(|(b, t)| b * t).sum::<f32>())
                        .collect::<Vec<f32>>()
                })
                .collect();

            // `A · x` for every row at once, and for each alone.
            let mut together = vec![0f32; INPUTS * rank];
            pack.through(0, cols, &x, cols, INPUTS, &mut together);
            for (i, x) in x.chunks_exact(cols).enumerate() {
                let mut alone = vec![0f32; rank];
                pack.through(0, cols, x, cols, 1, &mut alone);
                assert_eq!(
                    alone,
                    &together[i * rank..(i + 1) * rank],
                    "rank {rank} over {cols}: row {i} through A"
                );
            }
            for t in together.iter_mut() {
                *t *= scale;
            }
            assert!(
                off(&together, &want_t) < CLOSE,
                "rank {rank} over {cols}: A · x is off by {} of the peak",
                off(&together, &want_t)
            );

            // The same a block of columns at a time, where the columns come
            // in blocks.
            if cols.is_multiple_of(BLOCK) {
                let mut gathered = vec![0f32; INPUTS * rank];
                for block in 0..cols / BLOCK {
                    let mut part = vec![0f32; INPUTS * rank];
                    pack.through(
                        block * BLOCK,
                        BLOCK,
                        &x[block * BLOCK..],
                        cols,
                        INPUTS,
                        &mut part,
                    );
                    for (g, p) in gathered.iter_mut().zip(part) {
                        *g += p;
                    }
                }
                for t in gathered.iter_mut() {
                    *t *= scale;
                }
                assert!(
                    off(&gathered, &want_t) < CLOSE,
                    "rank {rank} over {cols}, by blocks: off by {} of the peak",
                    off(&gathered, &want_t)
                );
            }

            // The products of every row at once, in two parts as the units
            // of a phase add them, and of each alone.
            let cut = (rows / 3).next_multiple_of(SIXTEEN).min(rows);
            let mut got = vec![1f32; INPUTS * rows];
            // SAFETY: `got` holds `rows` products for each of the inputs,
            // one input's after another's, and nothing else touches it.
            unsafe {
                pack.add(0, cut, &together, INPUTS, got.as_mut_ptr(), rows);
                pack.add(
                    cut,
                    rows - cut,
                    &together,
                    INPUTS,
                    got.as_mut_ptr().add(cut),
                    rows,
                );
            }
            for (i, t) in together.chunks_exact(rank).enumerate() {
                let mut alone = vec![1f32; rows];
                // SAFETY: `alone` holds one input's `rows` products.
                unsafe { pack.add(0, rows, t, 1, alone.as_mut_ptr(), rows) };
                assert_eq!(
                    alone,
                    &got[i * rows..(i + 1) * rows],
                    "rank {rank}, {rows} products: row {i} through B"
                );
            }
            assert!(
                off(&got, &want) < CLOSE,
                "rank {rank}, {rows} products: off by {} of the peak",
                off(&got, &want)
            );
        }
        assert!(
            Packed::of(&[0.0; 7], &[0.0; 8], 2, 4, 1.0).is_none(),
            "A too short"
        );
        assert!(
            Packed::of(&[0.0; 8], &[0.0; 7], 2, 4, 1.0).is_none(),
            "B not rows of rank"
        );
        let Some(none) = Packed::of(&[], &[], 0, 512, 1.0) else {
            candle_core::bail!("an adapter of no rank packs to nothing")
        };
        let mut out = [3f32; 4];
        // SAFETY: an adapter of no rank adds nothing and reads nothing.
        unsafe { none.add(0, 4, &[], 1, out.as_mut_ptr(), 4) };
        none.through(0, 512, &[0.0; 512], 512, 1, &mut []);
        assert_eq!((none.rows(), out), (0, [3.0; 4]));
        Ok(())
    }

    /// The ranks at the end that add nothing are not packed: one expert's
    /// factors in room for three pack to that expert's own, value for value,
    /// so they give its products. A rank adds nothing when its row of `A` or
    /// its column of `B` is all zeros, and one in the middle is kept.
    #[test]
    fn ranks_that_add_nothing_are_left_out() -> candle_core::Result<()> {
        let (rank, cols, rows, room) = (16usize, 512usize, 40usize, 48usize);
        let a: Vec<f32> = (0..rank * cols)
            .map(|i| 0.01 + (i as f32 * 0.37).sin().abs() * 0.05)
            .collect();
        let b: Vec<f32> = (0..rows * rank)
            .map(|i| 0.01 + (i as f32 * 0.23).cos().abs() * 0.05)
            .collect();
        let mut wide_a = a.clone();
        wide_a.resize(room * cols, 0.0);
        let wide_b: Vec<f32> = b
            .chunks_exact(rank)
            .flat_map(|row| {
                row.iter()
                    .copied()
                    .chain(std::iter::repeat_n(0.0, room - rank))
            })
            .collect();
        let (Some(own), Some(wide)) = (
            Packed::of(&a, &b, rank, cols, 2.0),
            Packed::of(&wide_a, &wide_b, room, cols, 2.0),
        ) else {
            candle_core::bail!("factors that fit were not packed")
        };
        assert_eq!((wide.rank, wide.rows(), wide.cols()), (rank, rows, cols));
        assert_eq!((&wide.a, &wide.b), (&own.a, &own.b));

        let ranks = |a: &[f32], b: &[f32]| Packed::of(a, b, rank, cols, 2.0).map(|p| p.rank);
        // The last rank's row of `A` at nothing, then its column of `B`.
        let mut without = a.clone();
        without[(rank - 1) * cols..].fill(0.0);
        assert_eq!(ranks(&without, &b), Some(rank - 1));
        let mut without = b.clone();
        for row in without.chunks_exact_mut(rank) {
            row[rank - 1] = 0.0;
            row[rank - 2] = 0.0;
        }
        assert_eq!(ranks(&a, &without), Some(rank - 2));
        // A rank in the middle that adds nothing is held all the same.
        let mut without = a.clone();
        without[3 * cols..4 * cols].fill(0.0);
        assert_eq!(ranks(&without, &b), Some(rank));
        // A fresh adapter, its `B` all zeros, has nothing to hold.
        assert_eq!(ranks(&a, &vec![0.0; rows * rank]), Some(0));
        Ok(())
    }
}
