//! How a nearest-neighbour recall asks for its neighbours.
//!
//! Two numbers, and both of them exist because the index-backed KNN
//! operator behaves differently from the exhaustive one it replaced.
//!
//! The exhaustive form (`<|k,COSINE|>`) compares every row, so a
//! `WHERE` beside it narrows the population the comparison runs over
//! and `k` means k *matching* rows. The index-backed form
//! (`<|k,ef|>`) walks an HNSW graph that knows nothing about tenants
//! or tombstones: it hands back its k nearest, and every other
//! predicate is a residual filter applied to those. Ask for k and
//! filter afterwards and a tenant holding a small share of a large
//! table gets a short answer, or an empty one.
//!
//! So a filtered recall over-fetches and truncates, and an unfiltered
//! one does not need to. That distinction is per-call-site and is
//! stated at each of them.

/// The candidate pool a filtered recall pulls before its residual
/// filters run, so truncating to `k` afterwards still has `k` left.
///
/// The multiplier is the same one hybrid recall already used to give
/// Reciprocal Rank Fusion room to reorder; the ceiling is what keeps a
/// large `k` from turning one recall into a table read by another
/// name. A pool that still comes back short is a genuinely sparse
/// tenant, not a sizing bug.
pub fn candidate_pool(k: usize) -> usize {
    k.saturating_mul(POOL_MULTIPLIER).clamp(MIN_POOL, MAX_POOL)
}

/// The search effort (`ef`) an index-backed KNN asks for.
///
/// `ef` bounds the candidate list the graph walk keeps. Below `k` the
/// search cannot return `k` neighbours at all; well above it, recall
/// approaches exhaustive at a cost roughly linear in `ef`. Antumbra's
/// vector tables are working sets rather than corpora — a workspace's
/// memories, one population of experts — so the generous end of the
/// usual range is cheap here and the floor matters more than the
/// ceiling: a small `k` is the common case and is exactly where too
/// small an `ef` quietly loses neighbours.
pub fn search_effort(k: usize) -> i64 {
    let effort = k
        .saturating_mul(EFFORT_MULTIPLIER)
        .clamp(MIN_EFFORT, MAX_EFFORT);
    i64::try_from(effort).unwrap_or(MAX_EFFORT as i64)
}

const POOL_MULTIPLIER: usize = 5;
const MIN_POOL: usize = 20;
const MAX_POOL: usize = 200;

const EFFORT_MULTIPLIER: usize = 4;
const MIN_EFFORT: usize = 64;
const MAX_EFFORT: usize = 512;

#[cfg(test)]
mod tests {
    use super::*;

    /// The pool is never smaller than what was asked for, which is the
    /// property the truncation downstream depends on.
    #[test]
    fn the_pool_is_never_narrower_than_k() {
        for k in [1, 5, 20, 41, 199, 200, 1_000] {
            assert!(candidate_pool(k) >= k.min(MAX_POOL), "k = {k}");
        }
    }

    #[test]
    fn the_pool_holds_its_floor_and_its_ceiling() {
        assert_eq!(candidate_pool(1), MIN_POOL);
        assert_eq!(candidate_pool(10), 50);
        assert_eq!(candidate_pool(10_000), MAX_POOL);
    }

    /// An `ef` below `k` cannot return `k` neighbours, so the floor has
    /// to hold for every `k` the ceiling still covers.
    #[test]
    fn the_effort_is_never_below_k() {
        for k in [1, 16, 64, 128, 512] {
            assert!(search_effort(k) >= k as i64, "k = {k}");
        }
        // Past the ceiling the effort stops growing, which is the
        // deliberate trade: an enormous k is a paging problem, not a
        // recall-quality one.
        assert_eq!(search_effort(10_000), MAX_EFFORT as i64);
    }

    /// Saturating arithmetic, not a panic, on a k no caller should
    /// ever send.
    #[test]
    fn an_absurd_k_saturates() {
        assert_eq!(candidate_pool(usize::MAX), MAX_POOL);
        assert_eq!(search_effort(usize::MAX), MAX_EFFORT as i64);
    }
}
