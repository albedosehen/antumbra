//! Pure vector utilities over `Vec<f32>` — no GPU, no tensor backend.
//!
//! The one operation here is the crux of the **Matryoshka** embedder path: a
//! Matryoshka-trained model (BGE-M3, multilingual-e5, …) emits a long vector
//! (e.g. 1024-d) whose leading prefixes are themselves usable embeddings. The
//! HNSW index is fixed-dimension (`EMBED_DIM`), so to store a long Matryoshka
//! vector we take its first `EMBED_DIM` components and **re-normalize** them
//! back to unit length — a truncated prefix of a unit vector is no longer unit,
//! and cosine recall assumes unit vectors. This keeps the stored vector exactly
//! index-wide while letting an operator point at a richer generalist model.
//!
//! Kept as a plain `Vec<f32>` helper (rather than a Candle tensor op) so it is
//! reusable by both the HTTP embedder and the offline benchmark harness without
//! pulling in a tensor backend, and trivially unit-testable.

/// Below this L2 norm a vector is treated as the zero vector: dividing by it
/// would yield NaNs/infinities, so we return it unscaled instead. `1e-12` is far
/// under any meaningful embedding magnitude yet safely above `f32` underflow.
const NORM_EPSILON: f32 = 1e-12;

/// Truncate `v` to its first `target` components, then L2-renormalize to unit
/// length.
///
/// This is the Matryoshka prefix operation: storing the renormalized
/// `target`-dim prefix of a longer embedding into a fixed-dimension index.
///
/// Semantics:
/// - The prefix is `min(v.len(), target)` components. If `v` is already shorter
///   than `target` this is a no-op truncation (renormalize-only); the helper
///   does **not** error or pad — enforcing the *expected* source length is the
///   caller's job (see `HttpEmbedder`, which checks the endpoint's dimension
///   before calling this), so this stays a pure, total function.
/// - Renormalization divides by the L2 norm, except for a (near-)zero vector,
///   which is returned as-is to avoid `NaN` (an all-zero embedding is degenerate
///   but must not poison the index).
///
/// The result therefore has length `min(v.len(), target)` and, for any
/// non-degenerate input, unit L2 norm.
pub fn truncate_renormalize(mut v: Vec<f32>, target: usize) -> Vec<f32> {
    v.truncate(target);
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > NORM_EPSILON {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    /// L2 norm helper for assertions.
    fn norm(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    #[test]
    fn truncates_then_renormalizes_to_unit_length() {
        // A known 4-vector truncated to its first 2 components: [3, 4] → /5.
        let out = truncate_renormalize(vec![3.0, 4.0, 100.0, -7.0], 2);
        assert_eq!(out.len(), 2);
        assert!((out[0] - 0.6).abs() < 1e-6, "{out:?}");
        assert!((out[1] - 0.8).abs() < 1e-6, "{out:?}");
        assert!((norm(&out) - 1.0).abs() < 1e-6, "unit norm: {}", norm(&out));
    }

    #[test]
    fn zero_vector_does_not_nan() {
        let out = truncate_renormalize(vec![0.0; 8], 4);
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|x| *x == 0.0), "stays zero, no NaN: {out:?}");
    }

    #[test]
    fn truncating_to_own_length_is_renormalize_only() {
        // target == len: no components dropped, just rescaled to unit norm.
        let v = vec![1.0, 2.0, 2.0]; // norm 3
        let out = truncate_renormalize(v.clone(), v.len());
        assert_eq!(out.len(), v.len());
        assert!((out[0] - 1.0 / 3.0).abs() < 1e-6, "{out:?}");
        assert!((norm(&out) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn target_longer_than_input_truncates_to_min() {
        // A shorter-than-target vector is taken whole (min(len, target)) and
        // renormalized; the helper itself never pads or errors.
        let out = truncate_renormalize(vec![0.0, 3.0], 10);
        assert_eq!(out.len(), 2);
        assert!((out[1] - 1.0).abs() < 1e-6, "{out:?}");
    }
}
