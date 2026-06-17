//! Reciprocal Rank Fusion (RRF) for hybrid retrieval.
//!
//! Hybrid recall runs two retrievers over the same corpus -- dense (HNSW vector
//! similarity) and sparse (BM25 full-text) -- each returning a *ranked* list of
//! ids. RRF merges them by rank rather than raw score, so the two incomparable
//! score scales (cosine similarity vs BM25) never need calibrating: an item's
//! fused score is the sum of `1 / (k + rank)` over the lists it appears in (rank
//! is 1-based; `k` damps how much low ranks contribute -- 60 is the Cormack et
//! al. 2009 default). This is the model-agnostic, no-re-embed core of the
//! retrieval upgrade.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

/// The conventional RRF damping constant (Cormack et al., 2009).
pub const DEFAULT_RRF_K: f64 = 60.0;

/// Fuse several ranked id lists into a single ranking by Reciprocal Rank Fusion.
///
/// Each input is an ordered list of ids, most relevant first. The result is the
/// union of all ids ordered by descending fused score (`Σ 1 / (k + rank)`), with
/// ties broken lexicographically by id so the order is deterministic. `k` is the
/// damping constant (see [`DEFAULT_RRF_K`]); a larger `k` flattens the weight a
/// top rank carries over a lower one.
///
/// Duplicate ids *within* one ranking are scored at their first (best) position;
/// the same id appearing across rankings accumulates, which is the whole point --
/// an item both retrievers rank highly rises above one only a single retriever
/// found.
pub fn rrf_fuse(rankings: &[Vec<String>], k: f64) -> Vec<String> {
    let mut scores: HashMap<&str, f64> = HashMap::new();
    for ranking in rankings {
        // Within one ranking, only the best (first) occurrence of an id counts.
        let mut seen: HashSet<&str> = HashSet::new();
        for (i, id) in ranking.iter().enumerate() {
            if !seen.insert(id.as_str()) {
                continue;
            }
            let rank = (i + 1) as f64;
            *scores.entry(id.as_str()).or_insert(0.0) += 1.0 / (k + rank);
        }
    }
    let mut fused: Vec<(&str, f64)> = scores.into_iter().collect();
    fused.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.0.cmp(b.0))
    });
    fused.into_iter().map(|(id, _)| id.to_string()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(ids: &[&str]) -> Vec<String> {
        ids.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn empty_input_is_empty() {
        assert!(rrf_fuse(&[], DEFAULT_RRF_K).is_empty());
        assert!(rrf_fuse(&[v(&[])], DEFAULT_RRF_K).is_empty());
    }

    #[test]
    fn single_ranking_preserves_order() {
        assert_eq!(
            rrf_fuse(&[v(&["a", "b", "c"])], DEFAULT_RRF_K),
            v(&["a", "b", "c"])
        );
    }

    #[test]
    fn item_in_both_lists_outranks_single_list_item() {
        // `x` is found by both retrievers; `y` only by the second. Even though
        // `y` is rank 1 in its list, `x` (rank 1 + rank 2) accumulates more.
        let dense = v(&["x"]);
        let sparse = v(&["y", "x"]);
        let fused = rrf_fuse(&[dense, sparse], DEFAULT_RRF_K);
        assert_eq!(fused, v(&["x", "y"]));
    }

    #[test]
    fn ties_break_lexicographically_for_determinism() {
        // `a` and `b` swap ranks across the two lists -> identical fused score.
        // The tie-break orders them by id, so the output is stable.
        let a = v(&["a", "b", "c"]);
        let b = v(&["b", "a", "d"]);
        let fused = rrf_fuse(&[a, b], DEFAULT_RRF_K);
        assert_eq!(&fused[..2], &v(&["a", "b"])[..], "tied top pair, id order");
        assert_eq!(&fused[2..], &v(&["c", "d"])[..], "tied tail, id order");
    }

    #[test]
    fn union_includes_every_id() {
        let fused = rrf_fuse(&[v(&["a", "b"]), v(&["c", "b"])], DEFAULT_RRF_K);
        assert_eq!(fused.len(), 3, "a, b, c -- b deduped: {fused:?}");
        assert_eq!(fused[0], "b", "b is the only item in both lists");
    }

    #[test]
    fn smaller_k_sharpens_top_rank_advantage() {
        // The fused score of a rank-1 item is 1/(k+1); a smaller k makes that
        // larger, widening the gap to lower ranks. Sanity-check the formula.
        let top_small = 1.0 / (1.0 + 1.0);
        let top_default = 1.0 / (DEFAULT_RRF_K + 1.0);
        assert!(top_small > top_default);
    }
}
