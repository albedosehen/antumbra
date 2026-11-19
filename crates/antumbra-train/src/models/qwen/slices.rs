//! What the one-pass CPU ops share: a tensor's values as a slice, and the dot
//! product they are built from.

use candle_core::{bail, CpuStorage, Layout, Result};

/// A contiguous f32 tensor's values.
pub(super) fn values<'a>(
    storage: &'a CpuStorage,
    layout: &Layout,
    what: &str,
) -> Result<&'a [f32]> {
    let Some((start, end)) = layout.contiguous_offsets() else {
        bail!("{what} is not contiguous")
    };
    Ok(&storage.as_slice::<f32>()?[start..end])
}

/// Sixteen running sums folded pairwise: summed one after another they would
/// cost more than the products they hold, wherever a dot product is short.
fn fold(lanes: [f32; 16]) -> f32 {
    let eight: [f32; 8] = std::array::from_fn(|i| lanes[i] + lanes[i + 8]);
    let four: [f32; 4] = std::array::from_fn(|i| eight[i] + eight[i + 4]);
    (four[0] + four[2]) + (four[1] + four[3])
}

/// The dot product, sixteen lanes at a time so it vectorizes without being
/// allowed to reorder a single running sum.
pub(super) fn dot(a: &[f32], b: &[f32]) -> f32 {
    let (chunks_a, chunks_b) = (a.chunks_exact(16), b.chunks_exact(16));
    let tail: f32 = chunks_a
        .remainder()
        .iter()
        .zip(chunks_b.remainder())
        .map(|(x, y)| x * y)
        .sum();
    let mut lanes = [0f32; 16];
    for (xs, ys) in chunks_a.zip(chunks_b) {
        for ((lane, x), y) in lanes.iter_mut().zip(xs).zip(ys) {
            *lane += x * y;
        }
    }
    fold(lanes) + tail
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The dot product is the plain sum of products, for lengths on and off
    /// the lanes' multiple and for nothing at all.
    #[test]
    fn the_dot_product_is_the_sum_of_products() {
        for n in [0usize, 1, 15, 16, 17, 64, 100, 1536] {
            let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
            let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.11).cos()).collect();
            let want: f64 = a.iter().zip(&b).map(|(x, y)| *x as f64 * *y as f64).sum();
            let got = dot(&a, &b) as f64;
            assert!(
                (got - want).abs() < 1e-4,
                "{n} values: {got} against {want}"
            );
        }
    }
}
