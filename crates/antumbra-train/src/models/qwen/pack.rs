//! An adapter's factors as rows read on a CPU read them: at eight bits, in
//! the order the kernels read them.
//!
//! A decoded token is bound by how fast memory is read, and an adapter's
//! factors are read whole for every token beside the base: 74 MB at rank 16
//! on a 1.5B base, 222 MB for a blend of three experts, in f32. Most of that
//! is precision the token does not use. The base under an adapter is read at
//! eight bits or fewer, and what an adapter adds to a product is small beside
//! the base's own. So the step that reads plain rows reads a copy of the
//! factors at eight bits: each rank's row of `A`, and each rank's column of
//! `B`, as whole numbers of 127ths of its largest value, a quarter of the
//! f32 bytes.
//!
//! What that does to an expert is not to be told from nothing. One expert's
//! log-probabilities over 2078 held-out positions, read a token at a time
//! through this copy, shift from the dense base's by 0.103, 0.141 and 0.421
//! nats on average over a base at Q8_0, Q6_K and Q4_K; through the copy at
//! half precision that this one replaced, by 0.099, 0.146 and 0.418. The
//! order a float sum is added in moves the same reading by as much. Rounded
//! to six bits the factors read the same again; at four the shift is real
//! (0.125, 0.160 and 0.421 where 0.098, 0.144 and 0.413), and at two the
//! expert is gone.
//!
//! It buys less than its bytes say: a blend of three experts decodes 28.2
//! tokens a second at Q6_K on one laptop CPU where it decoded 27.4, and one
//! expert decodes as it did.
//!
//! The copy is made when rows are first read after the factors change
//! (`LoraLinear::pack`), and whoever sets the factors lets it go
//! (`QwenLora::adapters_changed`). The trainer's factors stay as they are:
//! training, every GPU path and the tensor path on a CPU read those.

/// Values to a block of a quantized row.
pub(super) const BLOCK: usize = 256;

/// The whole number the largest value of a rank's row of `A`, or of its
/// column of `B`, is held as.
const MOST: f32 = 127.0;

/// The largest size among `values`. What is not a number is passed over.
fn peak(values: impl Iterator<Item = f32>) -> f32 {
    values.fold(0f32, |most, v| most.max(v.abs()))
}

/// `value` as a whole number of steps, where `steps` of them make one: the
/// nearest, a half going away from zero. `steps` is [`MOST`] over the peak of
/// the value's row or column, so no value is past `MOST` steps.
fn held(value: f32, steps: f32) -> i8 {
    let v = value * steps;
    // The cast cuts toward zero, holds to an i8 and makes nothing of what is
    // not a number.
    (v + 0.5f32.copysign(v)) as i8
}

/// How many steps make one for a row or column whose largest value is
/// `peak`: none where there is nothing in it, or nothing that is a number.
fn steps(peak: f32) -> f32 {
    match peak > 0.0 && peak.is_finite() {
        true => MOST / peak,
        false => 0.0,
    }
}

/// Values a vector holds: a sixteen of products, or of the rank rows of `A`.
const SIXTEEN: usize = 16;

/// One projection's adapter at eight bits: `scale · B · (A · x)` added to
/// the `rows` products of a row `x` of `cols` values.
///
/// Both factors are laid out so that a vector holds sixteen results at once
/// and nothing is ever summed across a vector's lanes. `A` is held a column
/// at a time, sixteen of its rank rows to a vector: a column's vector times
/// one value of the input adds to all sixteen of those rows' sums. `B` is
/// held a sixteen of products at a time, a vector for each rank: a rank's
/// vector times that rank's value of `A · x` adds to all sixteen products.
/// Held a row of `rank` at a time instead, every product is a sum across a
/// vector, which costs ten times the multiplying it follows.
///
/// A factor is a whole number of steps, and a rank has two steps: one for
/// its row of `A`, one for its column of `B`. Neither is multiplied in where
/// the factors are read. A rank's sum through `A` is multiplied once by the
/// adapter's scale and both of the rank's steps ([`Packed::scaled`]), and
/// what comes of that is what `B`'s whole numbers are multiplied by.
pub(super) struct Packed {
    /// The ranks that are held: the factors' own, less those at the end
    /// that add nothing.
    pub(super) rank: usize,
    /// For each rank that is held, the adapter's scale times the step of
    /// its row of `A` times the step of its column of `B`.
    scales: Vec<f32>,
    cols: usize,
    rows: usize,
    /// `A`: for each sixteen of its rank rows (the last filled out with
    /// zeros), for each column, those rows' values of the column.
    a: Vec<i8>,
    /// `B`: for each sixteen of products (the last filled out with zeros),
    /// for each rank, those products' factors for it.
    b: Vec<i8>,
}

impl Packed {
    /// The factors `a` (`rank` rows of `cols`) and `b` (a row of `rank` for
    /// each product) at eight bits. `None` when the factors do not fit one
    /// another.
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
        // How many steps make one, for each held rank's row of `A` and for
        // its column of `B`.
        let steps_a: Vec<f32> = (0..live)
            .map(|r| steps(peak(a[r * cols..(r + 1) * cols].iter().copied())))
            .collect();
        let mut peaks_b = vec![0f32; live];
        for product in b.chunks_exact(rank.max(1)) {
            for (most, v) in peaks_b.iter_mut().zip(product) {
                *most = most.max(v.abs());
            }
        }
        let steps_b: Vec<f32> = peaks_b.into_iter().map(steps).collect();
        let mut by_columns = vec![0i8; live.div_ceil(SIXTEEN) * cols * SIXTEEN];
        for (sixteen, (held_columns, rows)) in by_columns
            .chunks_exact_mut((cols * SIXTEEN).max(1))
            .zip(a[..live * cols].chunks((cols * SIXTEEN).max(1)))
            .enumerate()
        {
            let steps = &steps_a[sixteen * SIXTEEN..];
            let ranks = steps.len().min(SIXTEEN);
            for (c, column) in held_columns.chunks_exact_mut(SIXTEEN).enumerate() {
                for (lane, value) in column[..ranks].iter_mut().enumerate() {
                    *value = held(rows[lane * cols + c], steps[lane]);
                }
            }
        }
        let mut by_sixteens = vec![0i8; rows.div_ceil(SIXTEEN) * live * SIXTEEN];
        for (held_factors, products) in by_sixteens
            .chunks_exact_mut((live * SIXTEEN).max(1))
            .zip(b.chunks((rank * SIXTEEN).max(1)))
        {
            for (r, (factors, steps)) in held_factors
                .chunks_exact_mut(SIXTEEN)
                .zip(&steps_b)
                .enumerate()
            {
                for (factor, product) in factors.iter_mut().zip(products.chunks_exact(rank)) {
                    *factor = held(product[r], *steps);
                }
            }
        }
        // A step is one over how many of them make one; a rank with nothing
        // in it has no step, and adds nothing.
        let step = |steps: f32| match steps > 0.0 {
            true => 1.0 / steps,
            false => 0.0,
        };
        let scales = steps_a
            .iter()
            .zip(&steps_b)
            .map(|(a, b)| scale * step(*a) * step(*b))
            .collect();
        Some(Self {
            rank: live,
            scales,
            cols,
            rows,
            a: by_columns,
            b: by_sixteens,
        })
    }

    /// Products this adapter adds to.
    pub(super) fn rows(&self) -> usize {
        self.rows
    }

    pub(super) fn cols(&self) -> usize {
        self.cols
    }

    /// For each rank that is held, what its sum through `A` is multiplied
    /// by before `B` reads it.
    #[cfg(test)]
    pub(super) fn scales(&self) -> &[f32] {
        &self.scales
    }

    /// Rows of `rank` sums through `A`, a row after another, each by its
    /// rank's scale: what `B`'s whole numbers are multiplied by.
    pub(super) fn scaled(&self, sums: &mut [f32]) {
        for row in sums.chunks_exact_mut(self.rank.max(1)) {
            for (sum, scale) in row.iter_mut().zip(&self.scales) {
                *sum *= scale;
            }
        }
    }

    /// `A` over its columns `first..first + n`, times those columns of each
    /// of `rows` rows of input: row `i`'s values of them are
    /// `x[i * stride..][..n]`, and its `rank` sums are written to
    /// `out[i * rank..]`, in `A`'s whole numbers: before the scale. A row's
    /// whole `A · x` is these over all the columns, and it may be had a run
    /// of columns at a time and added up: the MLP's gated row is only ever
    /// held a block at a time. A row's sums are the sums it has alone, to
    /// the bit, whatever rows are beside it.
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

    /// `B`'s rows `first..first + n`, each times a row's scaled sums through
    /// `A` ([`Packed::scaled`]), added to that row's products of them, for
    /// each of `rows` rows of input: row `i`'s `rank` values are
    /// `t[i * rank..]`, and its products of those rows of `B` are at
    /// `out + i * stride`. `first` is a multiple of sixteen. A row's
    /// products are what it gets alone, to the bit.
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
    a: &[i8],
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
                *sum += f32::from(*a) * x;
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
    b: &[i8],
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
                    *product += f32::from(*factor) * t;
                }
            }
        }
    }
}

/// `values` as a pack holds them, in their places: each rank's row of an `A`
/// (`a_side`), or its column of a `B`, at whole numbers of its own step. For
/// a test that holds the rows read here to the tensor path, which reads the
/// factors themselves: on factors a pack holds as they are, the two differ
/// by their arithmetic alone.
#[cfg(test)]
pub(super) fn as_held(values: &mut [f32], rank: usize, a_side: bool) {
    if rank == 0 {
        return;
    }
    let other = values.len() / rank;
    for r in 0..rank {
        let at = |i: usize| match a_side {
            true => r * other + i,
            false => i * rank + r,
        };
        let steps = steps(peak((0..other).map(|i| values[at(i)])));
        for i in 0..other {
            values[at(i)] = match steps > 0.0 {
                true => f32::from(held(values[at(i)], steps)) / steps,
                false => 0.0,
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A value is held as the nearest whole number of steps, a half going
    /// away from zero, the largest of its row as 127; and a row of nothing,
    /// or with nothing that is a number for a peak, is held as nothing.
    #[test]
    fn a_factor_is_held_at_the_nearest_step() -> candle_core::Result<()> {
        let row = [0.5f32, -0.5, 0.25, 0.0, 0.001, -0.002, 0.498];
        let steps_of = steps(peak(row.iter().copied()));
        assert_eq!(steps_of, 254.0);
        let got: Vec<i8> = row.iter().map(|v| held(*v, steps_of)).collect();
        // 0.25 is 63.5 steps, a half that goes away from zero; 0.001 is a
        // quarter of a step and 0.002 just over a half.
        assert_eq!(got, [127, -127, 64, 0, 0, -1, 126]);
        assert_eq!(held(f32::NAN, steps_of), 0);
        assert_eq!(steps(0.0), 0.0);
        assert_eq!(steps(f32::INFINITY), 0.0);
        assert_eq!(held(3.0, 0.0), 0);
        assert_eq!(peak([1.0, f32::NAN, -2.0].into_iter()), 2.0);
        Ok(())
    }

    /// Factors put as a pack holds them are within half a step of what they
    /// were, a row's largest as it was; and they are held as they are:
    /// putting them so again changes nothing but a last bit. A column of a
    /// `B` is put as a row of an `A` is.
    #[test]
    fn factors_as_held_stay_as_they_are() -> candle_core::Result<()> {
        let (rank, cols) = (5usize, 300usize);
        let trained: Vec<f32> = (0..rank * cols)
            .map(|i| (i as f32 * 0.37).sin() * 0.05)
            .collect();
        let mut a = trained.clone();
        as_held(&mut a, rank, true);
        for (held, trained) in a.chunks_exact(cols).zip(trained.chunks_exact(cols)) {
            let most = peak(trained.iter().copied());
            assert!((peak(held.iter().copied()) - most).abs() <= 1e-6 * most);
            for (h, t) in held.iter().zip(trained) {
                assert!((h - t).abs() <= most / 254.0 * 1.001, "{t} held as {h}");
            }
        }
        let mut again = a.clone();
        as_held(&mut again, rank, true);
        for (once, twice) in a.iter().zip(&again) {
            assert!(
                (once - twice).abs() <= 1e-6 * once.abs(),
                "{once} then {twice}"
            );
        }
        // The same values as columns: `cols` rows of `rank`.
        let by_columns = |rows: &[f32]| -> Vec<f32> {
            (0..cols * rank)
                .map(|i| rows[i % rank * cols + i / rank])
                .collect()
        };
        let mut b = by_columns(&trained);
        as_held(&mut b, rank, false);
        assert_eq!(b, by_columns(&a));
        Ok(())
    }

    /// The largest gap between two rows of values, over the peak of `want`.
    fn off(got: &[f32], want: &[f32]) -> f32 {
        let peak = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        got.iter()
            .zip(want)
            .fold(0f32, |m, (g, w)| m.max((g - w).abs()))
            / peak
    }

    /// A packed adapter gives `scale · B · (A · x)` as the factors it holds
    /// do a value at a time, to the order of a float sum, for ranks and
    /// counts of products on and off a multiple of sixteen, with `A · x` had
    /// whole or a block of columns at a time. Rows of input read together
    /// get, to the bit, what each gets alone. Factors as they were trained
    /// pack to the whole numbers of the factors as held. And factors that do
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
            let trained_a: Vec<f32> = (0..rank * cols)
                .map(|i| (i as f32 * 0.37).sin() * 0.05)
                .collect();
            let trained_b: Vec<f32> = (0..rows * rank)
                .map(|i| (i as f32 * 0.23).cos() * 0.05)
                .collect();
            // Factors as a pack holds them, so the plain sums below are of
            // the very values the pack holds.
            let (mut a, mut b) = (trained_a.clone(), trained_b.clone());
            as_held(&mut a, rank, true);
            as_held(&mut b, rank, false);
            let x: Vec<f32> = (0..INPUTS * cols)
                .map(|i| (i as f32 * 0.11).cos() * 3.0)
                .collect();
            let scale = 2.0f32;
            let Some(pack) = Packed::of(&a, &b, rank, cols, scale) else {
                candle_core::bail!("factors that fit were not packed")
            };
            assert_eq!((pack.rows(), pack.cols()), (rows, cols));

            // A value at a time.
            let want: Vec<f32> = x
                .chunks_exact(cols)
                .flat_map(|x| {
                    let t: Vec<f32> = a
                        .chunks_exact(cols)
                        .map(|row| row.iter().zip(x).map(|(a, x)| a * x).sum::<f32>() * scale)
                        .collect();
                    b.chunks_exact(rank)
                        .map(|row| 1.0 + row.iter().zip(&t).map(|(b, t)| b * t).sum::<f32>())
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
                assert!(
                    off(&gathered, &together) < CLOSE,
                    "rank {rank} over {cols}, by blocks: off by {} of the peak",
                    off(&gathered, &together)
                );
            }
            pack.scaled(&mut together);

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

            // The factors as they were trained pack to the same whole
            // numbers: what they add is what the factors as held add, each
            // of which is within half a step of the trained one.
            let Some(of_trained) = Packed::of(&trained_a, &trained_b, rank, cols, scale) else {
                candle_core::bail!("factors that fit were not packed")
            };
            assert_eq!((&of_trained.a, &of_trained.b), (&pack.a, &pack.b));
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
        none.scaled(&mut []);
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
        assert_eq!(wide.scales(), own.scales());

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
        // A rank in the middle that adds nothing is held all the same, and
        // adds nothing: its scale is nothing.
        let mut without = a.clone();
        without[3 * cols..4 * cols].fill(0.0);
        let Some(held) = Packed::of(&without, &b, rank, cols, 2.0) else {
            candle_core::bail!("factors that fit were not packed")
        };
        assert_eq!((held.rank, held.scales()[3]), (rank, 0.0));
        assert!(held.scales()[2] > 0.0);
        // A fresh adapter, its `B` all zeros, has nothing to hold.
        assert_eq!(ranks(&a, &vec![0.0; rows * rank]), Some(0));
        Ok(())
    }
}
