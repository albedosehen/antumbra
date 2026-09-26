//! The critic's influence, bounded by arithmetic (ADR-0022 S-2).
//!
//! The verifier partitions a group of samples into passed and failed, and the
//! critic may only reorder samples inside a part. [`shaped_advantages`]
//! normalizes the critic's scores separately inside each part and scales them
//! by the rank correlation between the critic and the verifier over the group,
//! so:
//! - a critic that tracks the verifier shapes advantage within each part;
//! - one that has stopped tracking it decays to nothing;
//! - one that has inverted has its influence flip sign.
//!
//! The critic's term is clamped to a quarter of the gap between the parts'
//! verifier advantages, so no critic score, however extreme, lifts a failed
//! sample over a passed one. No threshold fires and no operator has to notice.
//!
//! The rest of the module is instruments the record asks for: calibration and
//! agreement sliced so that a broken slice is not averaged away, and the
//! exogenous floor, the share of a critic's training labels that must come
//! fresh from the verifier.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::ports::CriticScore;

/// A trace's score from its steps: the weakest one, not the sum, since a sum
/// pays for verbose vacuous steps. `None` for a trace with no scored step.
pub fn weakest_step(scores: &[CriticScore]) -> Option<f32> {
    scores.iter().map(|s| s.value).reduce(f32::min)
}

/// Ranks of `xs` from 1, ties sharing the mean of the ranks they span.
fn ranks(xs: &[f32]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..xs.len()).collect();
    order.sort_by(|&a, &b| xs[a].total_cmp(&xs[b]));
    let mut out = vec![0.0; xs.len()];
    let mut i = 0;
    while i < order.len() {
        let mut j = i;
        while j + 1 < order.len() && xs[order[j + 1]] == xs[order[i]] {
            j += 1;
        }
        let rank = (i + j) as f64 / 2.0 + 1.0;
        for &k in &order[i..=j] {
            out[k] = rank;
        }
        i = j + 1;
    }
    out
}

/// Spearman's rank correlation of two series, or `None` when either is
/// constant (or they differ in length, or are shorter than two).
pub fn spearman(xs: &[f32], ys: &[f32]) -> Option<f32> {
    if xs.len() != ys.len() || xs.len() < 2 {
        return None;
    }
    let (rx, ry) = (ranks(xs), ranks(ys));
    let n = rx.len() as f64;
    let (mx, my) = (rx.iter().sum::<f64>() / n, ry.iter().sum::<f64>() / n);
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (a, b) in rx.iter().zip(&ry) {
        sxy += (a - mx) * (b - my);
        sxx += (a - mx).powi(2);
        syy += (b - my).powi(2);
    }
    if sxx <= f64::EPSILON || syy <= f64::EPSILON {
        return None;
    }
    Some((sxy / (sxx * syy).sqrt()) as f32)
}

/// The critic's scores z-normalized inside each part the verifier made, zero
/// where a part has fewer than two samples or no spread.
fn within_parts(passed: &[bool], critic: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; critic.len()];
    for part in [true, false] {
        let idx: Vec<usize> = (0..critic.len()).filter(|&i| passed[i] == part).collect();
        if idx.len() < 2 {
            continue;
        }
        let n = idx.len() as f32;
        let mean = idx.iter().map(|&i| critic[i]).sum::<f32>() / n;
        let var = idx.iter().map(|&i| (critic[i] - mean).powi(2)).sum::<f32>() / n;
        let std = var.sqrt();
        if std < 1e-6 {
            continue;
        }
        for &i in &idx {
            out[i] = (critic[i] - mean) / std;
        }
    }
    out
}

/// The advantages of one group: the verifier's, group-normalized, plus the
/// critic's shaping inside each part.
///
/// `weight` scales the critic's term before the clamp, so 0 is verifier-only
/// reward. With no spread in the verifier's verdicts there is nothing to
/// partition and nothing to correlate with, so the critic adds nothing and
/// the group stays flat: the critic cannot create signal the verifier has not.
pub fn shaped_advantages(passed: &[bool], critic: &[f32], weight: f32) -> Vec<f32> {
    let rewards: Vec<f32> = passed.iter().map(|&p| if p { 1.0 } else { 0.0 }).collect();
    let base = normalized(&rewards);
    if critic.len() != passed.len() || weight == 0.0 {
        return base;
    }
    let Some(rho) = spearman(critic, &rewards) else {
        return base;
    };
    let pass_adv = passed.iter().zip(&base).find(|(p, _)| **p).map(|(_, a)| *a);
    let fail_adv = passed
        .iter()
        .zip(&base)
        .find(|(p, _)| !**p)
        .map(|(_, a)| *a);
    let (Some(up), Some(down)) = (pass_adv, fail_adv) else {
        return base;
    };
    let bound = (up - down) / 4.0;
    let z = within_parts(passed, critic);
    base.iter()
        .zip(&z)
        .map(|(b, zi)| b + (weight * rho * zi).clamp(-bound, bound))
        .collect()
}

/// A group's rewards, mean-centered and scaled to unit deviation; all zero
/// when they have no spread.
fn normalized(rewards: &[f32]) -> Vec<f32> {
    let n = rewards.len();
    if n == 0 {
        return Vec::new();
    }
    let mean = rewards.iter().sum::<f32>() / n as f32;
    let var = rewards.iter().map(|r| (r - mean).powi(2)).sum::<f32>() / n as f32;
    let std = var.sqrt();
    if std < 1e-6 {
        return vec![0.0; n];
    }
    rewards.iter().map(|r| (r - mean) / std).collect()
}

/// One critic score against the verifier's verdict on the same artifact, with
/// the slice it belongs to (a language, a task family, a step depth).
#[derive(Debug, Clone, PartialEq)]
pub struct Scored {
    pub slice: String,
    /// The critic's predicted probability that the artifact passes.
    pub predicted: f32,
    pub passed: bool,
}

/// How well the critic matched the verifier in one slice.
#[derive(Debug, Clone, PartialEq)]
pub struct SliceCalibration {
    pub slice: String,
    pub n: u32,
    /// Expected calibration error over equal-width bins.
    pub ece: f32,
    /// The share of verdicts it called on the right side of one half.
    pub agreement: f32,
}

/// Calibration and agreement per slice, in slice order, so a broken slice is
/// reported as itself rather than averaged into a healthy total.
pub fn calibration_by_slice(scored: &[Scored], bins: usize) -> Vec<SliceCalibration> {
    let bins = bins.max(1);
    let mut by: BTreeMap<&str, Vec<&Scored>> = BTreeMap::new();
    for s in scored {
        by.entry(s.slice.as_str()).or_default().push(s);
    }
    by.into_iter()
        .map(|(slice, items)| {
            let n = items.len();
            let mut sums = vec![(0.0f32, 0.0f32, 0u32); bins];
            let mut agree = 0u32;
            for s in &items {
                let p = s.predicted.clamp(0.0, 1.0);
                let b = ((p * bins as f32) as usize).min(bins - 1);
                let outcome = if s.passed { 1.0 } else { 0.0 };
                sums[b].0 += p;
                sums[b].1 += outcome;
                sums[b].2 += 1;
                agree += u32::from((p >= 0.5) == s.passed);
            }
            let ece = sums
                .iter()
                .filter(|(_, _, c)| *c > 0)
                .map(|(p, o, c)| (*c as f32 / n as f32) * ((p - o) / *c as f32).abs())
                .sum();
            SliceCalibration {
                slice: slice.to_string(),
                n: u32::try_from(n).unwrap_or(u32::MAX),
                ece,
                agreement: agree as f32 / n as f32,
            }
        })
        .collect()
}

/// A monotone map from a critic's score to the pass rate it stands for,
/// refitted each generation on fresh verdicts (ADR-0022 S-2).
///
/// The record asks for quantile regression rather than temperature scaling.
/// For a verdict that is 0 or 1 every conditional quantile is 0 or 1, so the
/// map is fitted by isotonic regression instead, which keeps what the record
/// wanted from quantile regression: no parametric form, so a critic
/// overconfident in one range and not another is corrected where it is wrong,
/// not scaled uniformly.
#[derive(Debug, Clone, PartialEq)]
pub struct Isotonic {
    /// Block boundaries: the highest score in each block, ascending.
    upper: Vec<f32>,
    /// Each block's pass rate, non-decreasing.
    rate: Vec<f32>,
}

impl Isotonic {
    /// Fit by pool-adjacent-violators. `None` with nothing to fit.
    pub fn fit(scores: &[f32], passed: &[bool]) -> Option<Self> {
        if scores.is_empty() || scores.len() != passed.len() {
            return None;
        }
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]));
        // (sum of outcomes, count, highest score) per block.
        let mut blocks: Vec<(f32, f32, f32)> = Vec::new();
        for i in order {
            blocks.push((if passed[i] { 1.0 } else { 0.0 }, 1.0, scores[i]));
            while blocks.len() > 1 {
                let (s2, n2, u2) = blocks[blocks.len() - 1];
                let (s1, n1, _) = blocks[blocks.len() - 2];
                if s1 / n1 <= s2 / n2 {
                    break;
                }
                blocks.pop();
                let last = blocks.len() - 1;
                blocks[last] = (s1 + s2, n1 + n2, u2);
            }
        }
        Some(Isotonic {
            upper: blocks.iter().map(|b| b.2).collect(),
            rate: blocks.iter().map(|b| b.0 / b.1).collect(),
        })
    }

    /// The pass rate a score stands for: its block's, the last block's past
    /// the highest score seen.
    pub fn apply(&self, score: f32) -> f32 {
        let i = self.upper.partition_point(|&u| u < score);
        self.rate[i.min(self.rate.len() - 1)]
    }
}

/// How far two critics agree: the rank correlation of their scores on the
/// same completions. The record keeps a twin, trained on another seed and
/// slice, purely as this instrument, because agreement falls under
/// optimization pressure before headline fitness turns over.
pub fn agreement(a: &[f32], b: &[f32]) -> Option<f32> {
    spearman(a, b)
}

/// How a critic read during one generation's training (ADR-0022 S-2): on the
/// answers it scored to shape advantage, against the verifier's verdicts on
/// the same answers, and against its twin. The record asks for calibration to
/// be re-earned every generation, because the policy the critic judges changes
/// every generation, and for the twin's agreement to be watched, because it
/// falls under optimization pressure before fitness turns over. Shaping reads
/// the critic by rank, which a monotone recalibration does not change, so the
/// recalibration here is an instrument, not a correction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CriticWatch {
    /// Answers scored.
    pub n: u32,
    /// Rank correlation with the verifier's verdicts.
    pub correlation: Option<f32>,
    /// Calibration error of the raw scores.
    pub ece: Option<f32>,
    /// Calibration error after an isotonic map fitted on every other answer
    /// and read on the rest.
    pub recalibrated_ece: Option<f32>,
    /// Rank agreement with the twin on the same answers, when there is one.
    #[serde(default)]
    pub twin_agreement: Option<f32>,
}

/// Calibration error over `scored` as one slice, with ten bins.
fn ece(scored: &[Scored]) -> Option<f32> {
    let all: Vec<Scored> = scored
        .iter()
        .map(|s| Scored {
            slice: "all".into(),
            ..s.clone()
        })
        .collect();
    calibration_by_slice(&all, 10).first().map(|c| c.ece)
}

/// Read a critic's `scores` against the verifier's `passed` on the same
/// answers, and against `twin`'s scores on them when given.
pub fn watch(scores: &[f32], passed: &[bool], twin: Option<&[f32]>) -> CriticWatch {
    let n = scores.len().min(passed.len());
    let scored: Vec<Scored> = (0..n)
        .map(|i| Scored {
            slice: "all".into(),
            predicted: scores[i],
            passed: passed[i],
        })
        .collect();
    let verdicts: Vec<f32> = passed[..n]
        .iter()
        .map(|&p| if p { 1.0 } else { 0.0 })
        .collect();
    let (mut fit, mut held) = (Vec::new(), Vec::new());
    for (i, s) in scored.iter().enumerate() {
        if i % 2 == 0 {
            fit.push(s);
        } else {
            held.push(s);
        }
    }
    let recalibrated_ece = Isotonic::fit(
        &fit.iter().map(|s| s.predicted).collect::<Vec<_>>(),
        &fit.iter().map(|s| s.passed).collect::<Vec<_>>(),
    )
    .and_then(|map| {
        let mapped: Vec<Scored> = held
            .iter()
            .map(|s| Scored {
                predicted: map.apply(s.predicted),
                ..(*s).clone()
            })
            .collect();
        ece(&mapped)
    });
    CriticWatch {
        n: u32::try_from(n).unwrap_or(u32::MAX),
        correlation: spearman(&scores[..n], &verdicts),
        ece: ece(&scored),
        recalibrated_ece,
        twin_agreement: twin
            .filter(|t| t.len() >= n)
            .and_then(|t| agreement(&scores[..n], &t[..n])),
    }
}

/// The exogenous floor: how many labels derived from the critic (critic-scored
/// or critic-selected) a training set with `fresh` fresh verifier labels may
/// hold, so that the fresh share is at least `floor`. A floor of 1 admits none;
/// a floor of 0 or less is no floor, and admits every one.
pub fn derived_allowed(fresh: usize, floor: f64) -> usize {
    if floor <= 0.0 {
        return usize::MAX;
    }
    if floor >= 1.0 {
        return 0;
    }
    ((fresh as f64) * (1.0 - floor) / floor).floor() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    fn group() -> Vec<bool> {
        vec![true, true, true, false, false, false]
    }

    #[test]
    fn a_watch_reads_the_critic_against_the_verifier_and_its_twin() {
        let scores = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1];
        let passed = [true, true, true, false, false, false];
        let agreeing = [0.95, 0.85, 0.6, 0.4, 0.25, 0.05];
        let w = watch(&scores, &passed, Some(&agreeing));
        assert_eq!(w.n, 6);
        assert!(w.correlation.unwrap() > 0.8);
        assert_eq!(w.twin_agreement, Some(1.0));
        assert!(w.ece.is_some() && w.recalibrated_ece.is_some());
        // A twin that ranks the answers the other way round disagrees.
        let inverted: Vec<f32> = scores.iter().map(|s| 1.0 - s).collect();
        assert_eq!(
            watch(&scores, &passed, Some(&inverted)).twin_agreement,
            Some(-1.0)
        );
        assert_eq!(watch(&scores, &passed, None).twin_agreement, None);
        // Nothing scored reads nothing.
        let empty = watch(&[], &[], None);
        assert_eq!((empty.n, empty.correlation), (0, None));
    }

    #[test]
    fn spearman_handles_ties_and_constants() {
        assert_eq!(spearman(&[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0]), Some(1.0));
        assert_eq!(spearman(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]), Some(-1.0));
        assert_eq!(spearman(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0]), None);
        assert_eq!(spearman(&[1.0], &[1.0]), None);
        let tied = spearman(&[0.9, 0.8, 0.2, 0.1], &[1.0, 1.0, 0.0, 0.0]).unwrap();
        assert!((tied - 0.894_427).abs() < 1e-5, "{tied}");
    }

    #[test]
    fn a_critic_never_lifts_a_failed_sample_over_a_passed_one() {
        let passed = group();
        for critic in [
            vec![0.9, 0.8, 0.7, 0.3, 0.2, 0.1],
            vec![0.0, 0.0, 0.1, 1.0, 1.0, 0.9],
            vec![1e6, -1e6, 0.0, 1e6, -1e6, 3.0],
        ] {
            for weight in [0.5, 1.0, 100.0] {
                let adv = shaped_advantages(&passed, &critic, weight);
                let worst_pass = adv[..3].iter().cloned().fold(f32::INFINITY, f32::min);
                let best_fail = adv[3..].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                assert!(worst_pass > best_fail, "{critic:?} x{weight}: {adv:?}");
            }
        }
    }

    #[test]
    fn a_tracking_critic_shapes_within_parts_and_an_inverted_one_flips() {
        let passed = group();
        let base = shaped_advantages(&passed, &[0.0; 6], 1.0);
        // Tracks the verifier across the group, and orders each part.
        let tracking = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1];
        let adv = shaped_advantages(&passed, &tracking, 1.0);
        assert!(adv[0] > adv[1] && adv[1] > adv[2], "{adv:?}");
        assert!(adv[3] > adv[4] && adv[4] > adv[5], "{adv:?}");
        // The same order inside each part, but inverted across the group.
        let inverted = [0.3, 0.2, 0.1, 0.9, 0.8, 0.7];
        let flipped = shaped_advantages(&passed, &inverted, 1.0);
        assert!(
            flipped[0] < flipped[1] && flipped[1] < flipped[2],
            "{flipped:?}"
        );
        // Each part's mean is untouched: the critic only reorders inside it.
        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        assert!((mean(&adv[..3]) - base[0]).abs() < 1e-5);
        assert!((mean(&adv[3..]) - base[3]).abs() < 1e-5);
    }

    #[test]
    fn a_critic_that_says_nothing_about_the_verifier_adds_nothing() {
        let passed = group();
        let base = shaped_advantages(&passed, &[0.0; 6], 1.0);
        // Constant critic: no correlation to scale by.
        assert_eq!(shaped_advantages(&passed, &[0.5; 6], 1.0), base);
        // Zero weight is verifier-only reward.
        assert_eq!(
            shaped_advantages(&passed, &[0.9, 0.8, 0.7, 0.3, 0.2, 0.1], 0.0),
            base
        );
        // A group the verifier did not split stays flat whatever the critic says.
        let flat = shaped_advantages(&[true; 4], &[0.1, 0.9, 0.5, 0.3], 1.0);
        assert_eq!(flat, vec![0.0; 4]);
    }

    #[test]
    fn a_trace_scores_as_its_weakest_step() {
        let step = |i: u32, value: f32| CriticScore {
            step_idx: i,
            dimension: "critic".into(),
            value,
        };
        assert_eq!(
            weakest_step(&[step(0, 0.9), step(1, 0.2), step(2, 0.7)]),
            Some(0.2)
        );
        assert_eq!(weakest_step(&[]), None);
    }

    #[test]
    fn calibration_is_reported_per_slice() {
        let s = |slice: &str, predicted: f32, passed: bool| Scored {
            slice: slice.into(),
            predicted,
            passed,
        };
        let scored = vec![
            s("python", 0.9, true),
            s("python", 0.1, false),
            s("rust", 0.9, false),
            s("rust", 0.8, false),
        ];
        let report = calibration_by_slice(&scored, 10);
        assert_eq!(report.len(), 2);
        assert_eq!(report[0].slice, "python");
        assert!((report[0].ece - 0.1).abs() < 1e-6);
        assert_eq!(report[0].agreement, 1.0);
        // The broken slice is visible as itself.
        assert!((report[1].ece - 0.85).abs() < 1e-6);
        assert_eq!(report[1].agreement, 0.0);
    }

    #[test]
    fn isotonic_recalibration_is_monotone_and_corrects_overconfidence() {
        // Scores high across the board, but only the top ones pass.
        let scores = [0.9, 0.91, 0.92, 0.95, 0.97, 0.99];
        let passed = [false, false, true, false, true, true];
        let map = Isotonic::fit(&scores, &passed).unwrap();
        let fitted: Vec<f32> = scores.iter().map(|&s| map.apply(s)).collect();
        assert!(fitted.windows(2).all(|w| w[0] <= w[1]), "{fitted:?}");
        assert_eq!(map.apply(0.9), 0.0);
        assert_eq!(map.apply(0.99), 1.0);
        // The pooled middle takes its pool's rate.
        assert!((map.apply(0.93) - 0.5).abs() < 1e-6, "{fitted:?}");
        // Beyond what was seen, the ends hold.
        assert_eq!(map.apply(0.1), 0.0);
        assert_eq!(map.apply(2.0), 1.0);
        assert!(Isotonic::fit(&[], &[]).is_none());
    }

    #[test]
    fn twins_agree_by_rank() {
        assert_eq!(agreement(&[0.1, 0.5, 0.9], &[0.2, 0.3, 0.8]), Some(1.0));
        assert_eq!(agreement(&[0.1, 0.5, 0.9], &[0.8, 0.3, 0.2]), Some(-1.0));
    }

    #[test]
    fn the_floor_caps_derived_labels_and_never_decays() {
        assert_eq!(derived_allowed(30, 0.25), 90);
        assert_eq!(derived_allowed(30, 0.5), 30);
        assert_eq!(derived_allowed(30, 1.0), 0);
        assert_eq!(derived_allowed(0, 0.5), 0);
        assert_eq!(derived_allowed(7, 0.0), usize::MAX);
    }
}
