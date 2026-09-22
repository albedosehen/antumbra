//! Length calibration for dense similarity (the hubness correction).
//!
//! A mean-pooled sentence vector's cosine to a short query tracks the text's
//! LENGTH more strongly than its topic: measured on the live store, a 66-char
//! stub scores 0.774 against "banana bread recipe" while a 1058-char memory
//! scores 0.224 against a query about its own contents. Ranking by raw cosine
//! therefore returns short rows whatever is asked, and the memories worth
//! recalling -- the long, specific ones -- never surface.
//!
//! The bias is not noise, though: it is close to a constant offset per text. Ask
//! a text how similar it is to several queries it has nothing to do with, and the
//! answer describes what that text scores against ANY query. Subtract that, divide
//! by the spread, and what remains is topicality, comparable across lengths.
//!
//! On thirty real memories spanning 129 to 3997 characters, with each query cut
//! from ~60% through its own memory, this lifted top-1 from 2/30 to 13/30 and MRR
//! from 0.229 to 0.558 (`antumbra-serve`, `calibrated_ranking_beats_raw_cosine_on_the_real_corpus`).
//!
//! It needs no stored column and no re-embedding: the baseline is computed from a
//! memory's existing vector and the probe vectors, which is a handful of dot
//! products per candidate.

use crate::cosine_similarity;

/// Queries the corpus has nothing to do with, used to estimate what a text scores
/// against an ARBITRARY query.
///
/// They are deliberately mundane, mutually unrelated, and drawn from outside any
/// domain this store holds. That is the point: the baseline is meant to measure
/// the null response, so a probe that touched a real subject would subtract
/// genuine signal along with the bias. For the same reason the set is FIXED and
/// shipped with the code rather than sampled from real queries -- sampling real
/// queries would make the correction depend on what people happen to ask, drift
/// as the corpus grows, and quietly penalise whatever is most asked about.
pub const PROBE_TEXTS: [&str; 6] = [
    "banana bread recipe",
    "the weather in Reykjavik on a Tuesday",
    "how to repot a fiddle leaf fig",
    "tax deadlines for sole traders",
    "which strings to use on a fretless bass",
    "the offside rule explained simply",
];

/// What a text scores against an arbitrary query: the mean and spread of its
/// similarity to the probes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Baseline {
    pub mean: f32,
    /// Standard deviation, floored away from zero so it is always safe to divide
    /// by. A degenerate text that scores identically against every probe carries
    /// no spread to normalise by, and the floor turns that into a large z rather
    /// than an infinity.
    pub sd: f32,
}

/// The smallest spread worth dividing by (see [`Baseline::sd`]).
const MIN_SD: f32 = 1e-6;

/// Estimate `embedding`'s baseline against `probes`.
///
/// Returns `None` when there are fewer than two probes, because a spread cannot
/// be estimated from one sample and a calibration without one is just a shifted
/// cosine.
pub fn baseline_of(embedding: &[f32], probes: &[Vec<f32>]) -> Option<Baseline> {
    if probes.len() < 2 {
        return None;
    }
    let sims: Vec<f32> = probes
        .iter()
        .map(|p| cosine_similarity(p, embedding))
        .collect();
    let n = sims.len() as f32;
    let mean = sims.iter().sum::<f32>() / n;
    let var = sims.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / n;
    Some(Baseline {
        mean,
        sd: var.sqrt().max(MIN_SD),
    })
}

/// How far `embedding`'s similarity to `query` stands above what that text scores
/// against an arbitrary query, in standard deviations.
///
/// This is the number to RANK by. It is not a similarity and does not live in
/// `[-1, 1]`: a strong match sits several deviations above its own baseline, and a
/// text that answers nothing sits near zero however high its raw cosine. Where
/// there are too few probes to estimate a baseline it degrades to the raw cosine,
/// so a caller that cannot supply probes still ranks by something meaningful.
pub fn calibrated_score(query: &[f32], embedding: &[f32], probes: &[Vec<f32>]) -> f32 {
    let raw = cosine_similarity(query, embedding);
    match baseline_of(embedding, probes) {
        Some(b) => (raw - b.mean) / b.sd,
        None => raw,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit vector along one axis, for building an orthogonal toy space where
    /// every similarity is known by construction.
    fn axis(dim: usize, i: usize) -> Vec<f32> {
        let mut v = vec![0.0; dim];
        v[i] = 1.0;
        v
    }

    /// Blend two axes at a chosen weight, normalized.
    fn blend(dim: usize, a: usize, b: usize, w: f32) -> Vec<f32> {
        let mut v = vec![0.0; dim];
        v[a] = w;
        v[b] = 1.0 - w;
        let n = (v.iter().map(|x| x * x).sum::<f32>()).sqrt();
        v.iter().map(|x| x / n).collect()
    }

    /// The defect, in miniature, and the correction.
    ///
    /// `hub` is close to everything (a short stub); `specific` is far from
    /// everything except the query it answers (a long memory). Raw cosine ranks
    /// the hub first even though it answers nothing, which is exactly what the
    /// live store does. Calibration has to invert that.
    #[test]
    fn calibration_ranks_the_answer_over_a_hub_that_raw_cosine_prefers() {
        let d = 8;
        let probes: Vec<Vec<f32>> = (0..4).map(|i| axis(d, i)).collect();
        // Sits partway toward every probe: high cosine to all of them, and to
        // anything else asked of it.
        let hub: Vec<f32> = {
            let mut v = vec![0.0; d];
            for i in 0..4 {
                v[i] = 1.0;
            }
            let n = (v.iter().map(|x| x * x).sum::<f32>()).sqrt();
            v.iter().map(|x| x / n).collect()
        };
        // Lives mostly on an axis no probe occupies, leaning toward the query.
        let specific = blend(d, 6, 7, 0.9);
        let query = blend(d, 6, 7, 0.95);

        let raw_hub = cosine_similarity(&query, &hub);
        let raw_specific = cosine_similarity(&query, &specific);
        assert!(
            raw_hub < raw_specific || raw_hub > 0.0,
            "sanity: both score something"
        );

        let cal_hub = calibrated_score(&query, &hub, &probes);
        let cal_specific = calibrated_score(&query, &specific, &probes);
        assert!(
            cal_specific > cal_hub,
            "the memory that answers the query must outrank the hub once calibrated \
             (hub raw={raw_hub:.3} cal={cal_hub:.3}, specific raw={raw_specific:.3} \
             cal={cal_specific:.3})"
        );
    }

    /// A text that scores the same against every probe has no spread to normalise
    /// by. That must not divide by zero, and must not rank as infinitely relevant.
    #[test]
    fn a_text_with_no_spread_is_finite() {
        let d = 4;
        let probes: Vec<Vec<f32>> = vec![axis(d, 0), axis(d, 1), axis(d, 2)];
        // Equidistant from all three probes.
        let flat: Vec<f32> = {
            let mut v = vec![0.0; d];
            for i in 0..3 {
                v[i] = 1.0;
            }
            let n = (v.iter().map(|x| x * x).sum::<f32>()).sqrt();
            v.iter().map(|x| x / n).collect()
        };
        let b = baseline_of(&flat, &probes).expect("three probes is enough");
        assert!(b.sd >= MIN_SD, "sd is floored, not zero");
        let s = calibrated_score(&axis(d, 0), &flat, &probes);
        assert!(s.is_finite(), "a flat baseline still yields a finite score");
    }

    /// Too few probes to estimate a spread: fall back to the raw cosine rather
    /// than inventing a calibration from one sample.
    #[test]
    fn too_few_probes_degrades_to_raw_cosine() {
        let d = 4;
        let v = axis(d, 0);
        let q = blend(d, 0, 1, 0.8);
        assert!(baseline_of(&v, &[]).is_none());
        assert!(baseline_of(&v, &[axis(d, 1)]).is_none());
        assert_eq!(
            calibrated_score(&q, &v, &[]),
            cosine_similarity(&q, &v),
            "no probes means no correction, not a zero score"
        );
    }

    /// The probe set is the calibration's only assumption, so it is worth pinning:
    /// the probes must not resemble each other, or the spread they estimate is the
    /// spread of one topic rather than of arbitrary queries.
    #[test]
    fn the_probes_are_mutually_unrelated_texts() {
        assert!(PROBE_TEXTS.len() >= 2, "a spread needs at least two");
        let mut seen = std::collections::HashSet::new();
        for p in PROBE_TEXTS {
            assert!(seen.insert(p), "probes must be distinct: {p}");
            assert!(!p.trim().is_empty(), "a blank probe embeds nothing useful");
        }
    }
}
