//! Turning a ranking score into a calibrated probability (ADR-0023 B-2, ADR-0024 D-2).
//!
//! **This exists because the record had ruled it out, and the ruling was based
//! on a weaker measurement than the one that later contradicted it.**
//!
//! ADR-0023 B-2 says the deployed cross-encoder is reliable for ORDERING
//! candidates within one query and unreliable as an absolute magnitude ACROSS
//! queries, and concludes that no fixed threshold over it can carry a relevance
//! floor. That conclusion came from ten memories against a handful of varied
//! nonsense queries, where the same memory scored 3.7e-5 against one and 7.9e-4
//! against another — a spread of twenty-one times, with a genuine match sitting
//! inside it.
//!
//! The D-2 control then measured the same signal over 800 balanced pairs drawn
//! from 400 DISTINCT queries and found one fixed threshold reaching 0.802
//! accuracy and 0.785 F1. Both measurements are real; the second is far larger,
//! and it says the across-query spread costs accuracy rather than destroying
//! the signal.
//!
//! A threshold answers "which side". A floor that means the same thing for every
//! query needs "how likely", which is a different object: a monotone map from
//! score to probability, FITTED against labels. That is Platt scaling, and the
//! record never tried it before concluding the signal could not carry a floor.
//!
//! **Why this is an admissible `TypedDecider` and not a shortcut.** The port's
//! contract demands an implementation trained against a strictly proper scoring
//! rule over outcomes a VERIFIER produced. Both halves hold here: the labels
//! come from `scripts/d2-labels.sh`, whose verifier is span provenance with the
//! span excised, derived from no model; and the fit minimises log loss, which is
//! strictly proper, so reporting an honest probability is the only way to score
//! well. Fitting on a model's own past answers would violate ADR-0022's anchor
//! invariant — fitting on a deterministic verifier's does not.
//!
//! The fit is on `log10(score)` rather than the score. Reranker scores span
//! orders of magnitude (3.7e-5 to 7.9e-4 in the measurement above), so a
//! logistic in the raw score would spend its whole dynamic range on the top of
//! the distribution and read every small score as equally hopeless.

/// A fitted logistic over log-scores: `P(relevant) = sigmoid(a * log10(s) + b)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Platt {
    pub a: f32,
    pub b: f32,
}

/// Scores at or below this are treated as `log10(FLOOR_SCORE)` rather than
/// `-inf`. A reranker can emit a true zero, and a zero has no logarithm.
const FLOOR_SCORE: f32 = 1e-12;

fn log10_score(s: f32) -> f32 {
    s.max(FLOOR_SCORE).log10()
}

fn sigmoid(z: f32) -> f32 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

impl Platt {
    /// The probability this score maps to.
    pub fn probability(&self, score: f32) -> f32 {
        sigmoid(self.a * log10_score(score) + self.b)
    }

    /// Fit `(a, b)` on `(score, relevant)` pairs by minimising LOG LOSS, the
    /// strictly proper rule [`TypedDecider`](crate::ports::TypedDecider)
    /// requires.
    ///
    /// Plain gradient descent rather than Newton's method: two parameters over a
    /// convex objective, so the simple thing converges and there is no Hessian to
    /// get wrong. The gradient of log loss through a sigmoid is `(p - y) * x`,
    /// which is why the loop below has no exponentials in it.
    pub fn fit(samples: &[(f32, bool)], steps: usize, lr: f32) -> Self {
        // Start from a slope that is positive (a higher score means more likely
        // relevant, which is the one thing known about this signal in advance)
        // and an intercept centred on the mean log-score, so the initial
        // probabilities sit near 0.5 rather than saturated.
        let mean_x = if samples.is_empty() {
            0.0
        } else {
            samples.iter().map(|(s, _)| log10_score(*s)).sum::<f32>() / samples.len() as f32
        };
        let (mut a, mut b) = (1.0f32, -mean_x);
        let n = samples.len().max(1) as f32;
        for _ in 0..steps {
            let (mut ga, mut gb) = (0.0f32, 0.0f32);
            for (s, y) in samples {
                let x = log10_score(*s);
                let p = sigmoid(a * x + b);
                let d = p - if *y { 1.0 } else { 0.0 };
                ga += d * x;
                gb += d;
            }
            a -= lr * ga / n;
            b -= lr * gb / n;
        }
        Self { a, b }
    }
}

/// How far a set of probabilities is from meaning what it says, as expected
/// calibration error over `bins` equal-width buckets.
///
/// ADR-0024 Validation 4 asks for this **sliced**, not as a global average,
/// following ADR-0022's insistence that a global number hides a broken slice.
/// The slicing is the caller's job — pass one slice at a time.
pub fn expected_calibration_error(samples: &[(f32, bool)], bins: usize) -> f32 {
    if samples.is_empty() || bins == 0 {
        return 0.0;
    }
    let mut sum_p = vec![0.0f32; bins];
    let mut sum_y = vec![0.0f32; bins];
    let mut count = vec![0.0f32; bins];
    for (p, y) in samples {
        let idx = ((p * bins as f32) as usize).min(bins - 1);
        sum_p[idx] += p;
        sum_y[idx] += if *y { 1.0 } else { 0.0 };
        count[idx] += 1.0;
    }
    let n = samples.len() as f32;
    (0..bins)
        .filter(|i| count[*i] > 0.0)
        .map(|i| (count[i] / n) * ((sum_p[i] / count[i]) - (sum_y[i] / count[i])).abs())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A separable signal is fitted to probabilities on the right sides of 0.5.
    #[test]
    fn a_separable_signal_calibrates() {
        // The two bands must NOT overlap, or the probe points sit on the
        // boundary where 0.5 is the correct answer and the assertion is testing
        // the fixture rather than the fit.
        let mut samples = Vec::new();
        for i in 0..100 {
            let t = i as f32 / 99.0;
            samples.push((1e-6 * (1.0 + 9.0 * t), false)); // 1e-6 .. 1e-5
            samples.push((1e-3 * (1.0 + 9.0 * t), true)); // 1e-3 .. 1e-2
        }
        let p = Platt::fit(&samples, 4000, 2.0);
        assert!(
            p.probability(5e-3) > 0.5,
            "a score inside the positive band reads as probably relevant, got {}",
            p.probability(5e-3)
        );
        assert!(
            p.probability(5e-6) < 0.5,
            "a score inside the negative band reads as probably not, got {}",
            p.probability(5e-6)
        );
    }

    /// The slope comes out POSITIVE, which is the one thing known about this
    /// signal in advance: a higher rerank score means more likely relevant. A
    /// negative slope would mean the fit had inverted the signal, which is the
    /// failure that would silently empty every recall.
    #[test]
    fn the_fitted_slope_is_positive() {
        let samples: Vec<(f32, bool)> = (0..200)
            .map(|i| {
                let relevant = i % 2 == 0;
                (if relevant { 5e-4 } else { 2e-5 }, relevant)
            })
            .collect();
        let p = Platt::fit(&samples, 4000, 2.0);
        assert!(p.a > 0.0, "slope must be positive, got {}", p.a);
    }

    /// Perfectly calibrated predictions have no calibration error, and confident
    /// wrong ones have a lot. The metric has to separate those or it says
    /// nothing.
    #[test]
    fn calibration_error_separates_honest_from_overconfident() {
        // Half the samples at p=1.0 and true, half at p=0.0 and false.
        let honest: Vec<(f32, bool)> = (0..100)
            .map(|i| if i % 2 == 0 { (1.0, true) } else { (0.0, false) })
            .collect();
        assert!(expected_calibration_error(&honest, 10) < 0.01);

        // Confidently wrong about everything.
        let wrong: Vec<(f32, bool)> = (0..100)
            .map(|i| if i % 2 == 0 { (1.0, false) } else { (0.0, true) })
            .collect();
        assert!(
            expected_calibration_error(&wrong, 10) > 0.9,
            "confident and wrong must score near 1.0"
        );
    }

    /// ADR-0023 B-2's open question, answered against the real signal: can the
    /// deployed cross-encoder carry a relevance floor once it is CALIBRATED
    /// rather than thresholded?
    ///
    /// Needs no GPU and no model — it reads the scores
    /// `scripts/d2-relevance-baseline.sh` writes with
    /// `ANTUMBRA_D2_SCORES_OUT`, so the whole calibration question is answerable
    /// on any machine:
    ///
    ///   ANTUMBRA_D2_SCORES=/path/to/scores.txt \
    ///     cargo test -p antumbra-train -- --ignored --nocapture calibrating_the_reranker
    ///
    /// **The fit is on one half and every number is reported on the other.** The
    /// rows come in pairs, a positive and its hard negative from the same memory,
    /// so the split is taken at a pair boundary — splitting mid-pair would put
    /// one memory's two rows on both sides, which is the leak the D-2 head test
    /// avoids by splitting on memories.
    #[test]
    #[ignore = "needs ANTUMBRA_D2_SCORES, the output of d2-relevance-baseline.sh"]
    fn calibrating_the_reranker() {
        let Ok(path) = std::env::var("ANTUMBRA_D2_SCORES") else {
            println!("ANTUMBRA_D2_SCORES unset -- skipped");
            return;
        };
        let raw = std::fs::read_to_string(&path).expect("read scores");
        let rows: Vec<(f32, bool)> = raw
            .lines()
            .filter_map(|l| {
                let mut it = l.split_whitespace();
                let s: f32 = it.next()?.parse().ok()?;
                let y = it.next()? == "1";
                Some((s, y))
            })
            .collect();
        assert!(rows.len() >= 100, "need a real set, got {}", rows.len());

        let cut = (rows.len() / 2) & !1; // even, so no pair straddles the split
        let (train, test) = rows.split_at(cut);
        let platt = Platt::fit(train, 20_000, 1.0);
        println!(
            "\n  {} rows: {} fit / {} held out\n  P(relevant) = sigmoid({:.3} * log10(score) + {:.3})",
            rows.len(),
            train.len(),
            test.len(),
            platt.a,
            platt.b
        );
        assert!(platt.a > 0.0, "a higher rerank score must read as MORE likely");

        let probs: Vec<(f32, bool)> = test
            .iter()
            .map(|(s, y)| (platt.probability(*s), *y))
            .collect();
        let (mut tp, mut fp, mut fern, mut tn) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for (p, y) in &probs {
            match (*p >= 0.5, *y) {
                (true, true) => tp += 1.0,
                (true, false) => fp += 1.0,
                (false, true) => fern += 1.0,
                (false, false) => tn += 1.0,
            }
        }
        let prec = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
        let rec = if tp + fern > 0.0 { tp / (tp + fern) } else { 0.0 };
        let f1 = if prec + rec > 0.0 {
            2.0 * prec * rec / (prec + rec)
        } else {
            0.0
        };
        println!(
            "  at the natural 0.5 floor: acc={:.3} prec={:.3} rec={:.3} F1={:.3}",
            (tp + tn) / probs.len() as f32,
            prec,
            rec,
            f1
        );
        println!(
            "  expected calibration error (10 bins): {:.4}",
            expected_calibration_error(&probs, 10)
        );

        // The floor is a DEFAULT someone has to choose, and 0.5 is only the
        // natural reading of a probability, not the right operating point. A
        // memory system that hides a relevant memory fails worse than one that
        // shows a weak one, because the caller can see and discard the second
        // and cannot know about the first. So the curve decides, not the round
        // number.
        println!("  the floor is a choice -- precision/recall across it:");
        for floor in [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] {
            let (mut tp, mut fp, mut fern) = (0.0f32, 0.0f32, 0.0f32);
            for (p, y) in &probs {
                match (*p >= floor, *y) {
                    (true, true) => tp += 1.0,
                    (true, false) => fp += 1.0,
                    (false, true) => fern += 1.0,
                    (false, false) => {}
                }
            }
            let prec = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
            let rec = if tp + fern > 0.0 { tp / (tp + fern) } else { 0.0 };
            let f1 = if prec + rec > 0.0 {
                2.0 * prec * rec / (prec + rec)
            } else {
                0.0
            };
            println!(
                "    floor {floor:.2}: prec={prec:.3} rec={rec:.3} F1={f1:.3}  \
                 (drops {:.1}% of genuine answers)",
                100.0 * fern / (tp + fern).max(1.0)
            );
        }

        // Validation 4 wants ECE SLICED, because a global average hides a broken
        // slice. Every question here is a `Noul`, so the slice that can differ is
        // the confidence band itself: a floor is read near the middle, and a
        // model well calibrated only where it is certain is useless there.
        println!("  sliced by confidence band (the floor is read in the middle):");
        for (lo, hi) in [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)] {
            let slice: Vec<(f32, bool)> = probs
                .iter()
                .copied()
                .filter(|(p, _)| *p >= lo && *p < hi)
                .collect();
            if slice.is_empty() {
                println!("    {lo:.2}-{hi:.2}: empty");
                continue;
            }
            let actual = slice.iter().filter(|(_, y)| *y).count() as f32 / slice.len() as f32;
            let mean_p = slice.iter().map(|(p, _)| *p).sum::<f32>() / slice.len() as f32;
            println!(
                "    {lo:.2}-{hi:.2}: n={:<4} predicted {:.3}  actual {:.3}  gap {:.3}",
                slice.len(),
                mean_p,
                actual,
                (mean_p - actual).abs()
            );
        }
    }

    /// A score of zero has no logarithm, and a reranker can emit one. The floor
    /// keeps it finite rather than poisoning the fit with a NaN.
    #[test]
    fn a_zero_score_does_not_poison_the_fit() {
        let samples = vec![(0.0f32, false), (1e-3, true), (0.0, false), (2e-3, true)];
        let p = Platt::fit(&samples, 1000, 1.0);
        assert!(p.a.is_finite() && p.b.is_finite());
        assert!(p.probability(0.0).is_finite());
    }
}
