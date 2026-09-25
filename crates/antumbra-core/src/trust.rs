//! The trust protocol (ADR-0022 S-4): what a verifier must show before it may
//! grant reward, and keep showing afterwards.
//!
//! A verifier is measured on **cases**: artifacts whose outcome is known from
//! something the loop did not produce. An authored verifier's verdict, an
//! authored reference solution, or construction: code built to solve nothing,
//! and artifacts of a task that no artifact can satisfy. There is no anchor
//! for a synthesized verifier's verdict, trusted or not, so a label can never
//! come from the thing being measured or from anything like it.
//!
//! Every case is run several times, and [`Tally::judge`] reads the runs in
//! this order:
//! - **determinism:** a case whose verdict changed between runs rejects the
//!   verifier outright;
//! - **the adversarial holdout:** one pass on a task that cannot be satisfied
//!   is proof of a shortcut, not a near miss, and rejects it outright too;
//! - **the paired holdout:** the one-sided upper confidence bound on its
//!   false-positive rate, the rate at which it passes what anchored truth
//!   fails, must be under the policy's. The bound, not the observed rate, so
//!   a verifier seen on few failures cannot pass on luck;
//! - **usefulness:** it must accept enough of the known-good artifacts. A
//!   false negative only wastes compute, so this is a floor, not a bound.
//!
//! The asymmetry is the design: a false positive teaches the population
//! something untrue and cannot be taken back once it has trained.

use chrono::{DateTime, TimeDelta, Utc};
use serde::{Deserialize, Serialize};

use crate::ids::VerifierId;
use crate::verifier::{TrustState, VerifierOrigin};

/// What a case's artifact is known to be.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Label {
    /// Right: a verifier should pass it.
    Good,
    /// Wrong: a verifier should fail it.
    Bad,
    /// Of a task no artifact can satisfy: a pass is a shortcut.
    Impossible,
    /// Built to be wrong (a mutant of the reference, a forgery) and failed by
    /// anchored truth. A pass is a shortcut here too: the artifact was chosen
    /// to be wrong, so passing it is not a rare miss the bound may forgive.
    Adversarial,
}

/// Where a case's label comes from. Nothing the loop synthesized can be one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Anchor {
    /// An authored verifier's verdict.
    Verifier { id: VerifierId },
    /// An authored reference solution: right by authorship.
    Reference,
    /// Built so it cannot be right.
    Constructed,
}

/// One artifact a verifier is measured on.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Case {
    /// Names the case in a report: a flaky case, or a shortcut.
    pub id: String,
    pub task: String,
    /// The completion the verifier judges.
    pub completion: String,
    pub label: Label,
    pub anchor: Anchor,
}

/// How much a verifier must show. The defaults ask for a false-positive rate
/// under 10% at 95% confidence, which takes at least 29 known-bad cases with
/// none passed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrustPolicy {
    /// Runs of every case, for the determinism gate.
    pub repeats: u32,
    /// Of the one-sided bound on the false-positive rate.
    pub confidence: f64,
    /// The bound must be at or under this.
    pub max_false_positive: f64,
    /// The share of known-good cases it must accept.
    pub min_accepted: f64,
    /// How long a sound measurement keeps a verifier trusted.
    pub ttl: TimeDelta,
}

impl Default for TrustPolicy {
    fn default() -> Self {
        TrustPolicy {
            repeats: 3,
            confidence: 0.95,
            max_false_positive: 0.10,
            min_accepted: 0.5,
            ttl: TimeDelta::days(7),
        }
    }
}

/// What a measurement found, from the first gate that failed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TrustVerdict {
    /// Every gate passed.
    Sound,
    /// These cases changed verdict between runs of the same artifact.
    Flaky { cases: Vec<String> },
    /// It passed these cases of tasks no artifact can satisfy.
    Shortcut { cases: Vec<String> },
    /// Too few anchored cases to bound anything: no known-good or known-bad
    /// cases, or too few known-bad ones, none passed, to bound the rate.
    Unmeasured { reason: String },
    /// It passed known-bad cases, and its false-positive rate may be as high
    /// as `upper`, over `max`.
    FalsePositives { upper: f64, max: f64 },
    /// It accepted `accepted` of the known-good cases, under `min`.
    TooStrict { accepted: f64, min: f64 },
}

impl TrustVerdict {
    pub fn is_sound(&self) -> bool {
        matches!(self, TrustVerdict::Sound)
    }

    /// Whether it rejects a verifier outright, however much evidence follows.
    pub fn is_rejection(&self) -> bool {
        matches!(
            self,
            TrustVerdict::Flaky { .. } | TrustVerdict::Shortcut { .. }
        )
    }

    /// Whether it shows the verifier granting reward it should not.
    pub fn is_unsound(&self) -> bool {
        self.is_rejection() || matches!(self, TrustVerdict::FalsePositives { .. })
    }
}

/// The runs of a measurement, counted.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Tally {
    pub repeats: u32,
    pub flaky: Vec<String>,
    pub good: u32,
    pub good_passed: u32,
    pub bad: u32,
    pub bad_passed: u32,
    pub impossible: u32,
    pub adversarial: u32,
    pub shortcuts: Vec<String>,
}

impl Tally {
    /// Count one case's runs. A case whose runs disagree is flaky and counts
    /// toward nothing else.
    pub fn add(&mut self, case: &Case, runs: &[bool]) {
        let Some(&first) = runs.first() else {
            return;
        };
        if runs.iter().any(|&r| r != first) {
            self.flaky.push(case.id.clone());
            return;
        }
        match case.label {
            Label::Good => {
                self.good += 1;
                self.good_passed += u32::from(first);
            }
            Label::Bad => {
                self.bad += 1;
                self.bad_passed += u32::from(first);
            }
            Label::Impossible | Label::Adversarial => {
                if case.label == Label::Impossible {
                    self.impossible += 1;
                } else {
                    self.adversarial += 1;
                }
                if first {
                    self.shortcuts.push(case.id.clone());
                }
            }
        }
    }

    /// Judge the tally under `policy`, gates in order.
    pub fn judge(
        &self,
        verifier: &VerifierId,
        at: DateTime<Utc>,
        policy: &TrustPolicy,
    ) -> TrustMeasurement {
        let upper = upper_bound(self.bad_passed, self.bad, policy.confidence);
        let accepted = if self.good == 0 {
            0.0
        } else {
            f64::from(self.good_passed) / f64::from(self.good)
        };
        let verdict = if !self.flaky.is_empty() {
            TrustVerdict::Flaky {
                cases: self.flaky.clone(),
            }
        } else if !self.shortcuts.is_empty() {
            TrustVerdict::Shortcut {
                cases: self.shortcuts.clone(),
            }
        } else if self.bad == 0 || self.good == 0 {
            TrustVerdict::Unmeasured {
                reason: format!(
                    "{} known-good and {} known-bad case(s): a bound needs both",
                    self.good, self.bad
                ),
            }
        } else if upper > policy.max_false_positive && self.bad_passed == 0 {
            // Nothing wrong was passed, but too little was tried to bound
            // the rate: missing evidence, not evidence against it.
            TrustVerdict::Unmeasured {
                reason: format!(
                    "none of {} known-bad case(s) passed, too few to bound the rate under {}",
                    self.bad, policy.max_false_positive
                ),
            }
        } else if upper > policy.max_false_positive {
            TrustVerdict::FalsePositives {
                upper,
                max: policy.max_false_positive,
            }
        } else if accepted < policy.min_accepted {
            TrustVerdict::TooStrict {
                accepted,
                min: policy.min_accepted,
            }
        } else {
            TrustVerdict::Sound
        };
        TrustMeasurement {
            verifier: verifier.clone(),
            at,
            valid_until: at + policy.ttl,
            repeats: self.repeats,
            good: self.good,
            good_passed: self.good_passed,
            bad: self.bad,
            bad_passed: self.bad_passed,
            impossible: self.impossible,
            adversarial: self.adversarial,
            false_positive_upper: upper,
            confidence: policy.confidence,
            verdict,
        }
    }
}

/// One measurement of a verifier, as recorded.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrustMeasurement {
    pub verifier: VerifierId,
    pub at: DateTime<Utc>,
    /// Until when a sound measurement keeps the verifier trusted.
    pub valid_until: DateTime<Utc>,
    pub repeats: u32,
    pub good: u32,
    pub good_passed: u32,
    pub bad: u32,
    pub bad_passed: u32,
    pub impossible: u32,
    /// Deliberately wrong artifacts it was run on, none of which it may pass.
    #[serde(default)]
    pub adversarial: u32,
    /// The one-sided upper bound on the false-positive rate.
    pub false_positive_upper: f64,
    pub confidence: f64,
    pub verdict: TrustVerdict,
}

/// The one-sided upper confidence bound on a rate seen `k` times in `n`
/// trials (Clopper-Pearson): the largest rate under which `k` or fewer is
/// still at least `1 - confidence` likely. With no trials nothing is bounded,
/// and the bound is 1.
pub fn upper_bound(k: u32, n: u32, confidence: f64) -> f64 {
    if n == 0 || k >= n {
        return 1.0;
    }
    let alpha = 1.0 - confidence;
    // The chance of k or fewer falls as the rate rises: bisect for alpha.
    let (mut lo, mut hi) = (f64::from(k) / f64::from(n), 1.0);
    for _ in 0..64 {
        let mid = (lo + hi) / 2.0;
        if binomial_cdf(k, n, mid) > alpha {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    hi
}

/// P(X <= k) for X ~ Binomial(n, p), summed in log space.
fn binomial_cdf(k: u32, n: u32, p: f64) -> f64 {
    if p <= 0.0 {
        return 1.0;
    }
    if p >= 1.0 {
        return if k >= n { 1.0 } else { 0.0 };
    }
    let (lp, lq) = (p.ln(), (1.0 - p).ln());
    let mut log_choose = 0.0f64;
    let mut sum = 0.0;
    for i in 0..=k {
        if i > 0 {
            log_choose += f64::from(n - i + 1).ln() - f64::from(i).ln();
        }
        sum += (log_choose + f64::from(i) * lp + f64::from(n - i) * lq).exp();
    }
    sum.min(1.0)
}

/// Where a measurement moves a verifier in `state`, or `None` when it stays.
///
/// A proposed verifier is trusted by a sound measurement and revoked by an
/// outright rejection. Any other finding leaves it proposed, since more
/// evidence may yet bound it. A trusted verifier found unsound is quarantined
/// at once. One that was merely unmeasured, or too strict, keeps its state,
/// but only a sound measurement renews its time to live.
pub fn after_measurement(state: TrustState, verdict: &TrustVerdict) -> Option<TrustState> {
    match state {
        TrustState::Proposed if verdict.is_sound() => Some(TrustState::Trusted),
        TrustState::Proposed if verdict.is_rejection() => Some(TrustState::Revoked),
        TrustState::Trusted if verdict.is_unsound() => Some(TrustState::Quarantined),
        _ => None,
    }
}

/// Whether a verifier may grant reward at `now`. An authored one may while it
/// is trusted. A synthesized one also needs a sound measurement still inside
/// its time to live: trust that is not re-measured lapses.
pub fn grants_reward(
    origin: VerifierOrigin,
    state: TrustState,
    last_sound: Option<&TrustMeasurement>,
    now: DateTime<Utc>,
) -> bool {
    if state != TrustState::Trusted {
        return false;
    }
    match origin {
        VerifierOrigin::Authored => true,
        VerifierOrigin::Synthesized => {
            last_sound.is_some_and(|m| m.verdict.is_sound() && now < m.valid_until)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case(id: &str, label: Label) -> Case {
        Case {
            id: id.into(),
            task: "t".into(),
            completion: String::new(),
            label,
            anchor: Anchor::Constructed,
        }
    }

    fn tally(good: (u32, u32), bad: (u32, u32)) -> Tally {
        Tally {
            repeats: 3,
            good: good.0,
            good_passed: good.1,
            bad: bad.0,
            bad_passed: bad.1,
            ..Tally::default()
        }
    }

    fn judge(t: &Tally) -> TrustVerdict {
        t.judge(
            &VerifierId::new("verifier:v"),
            Utc::now(),
            &TrustPolicy::default(),
        )
        .verdict
    }

    #[test]
    fn the_bound_matches_the_closed_form_with_no_failures() {
        for n in [1u32, 10, 29, 59, 300] {
            let exact = 1.0 - 0.05f64.powf(1.0 / f64::from(n));
            assert!((upper_bound(0, n, 0.95) - exact).abs() < 1e-9, "n={n}");
        }
        assert_eq!(upper_bound(0, 0, 0.95), 1.0);
        assert_eq!(upper_bound(5, 5, 0.95), 1.0);
    }

    #[test]
    fn the_bound_matches_clopper_pearson_with_failures() {
        // Reference values: the 0.95 quantile of Beta(k + 1, n - k).
        let cases = [
            (1u32, 10u32, 0.394_163),
            (2, 50, 0.120_614),
            (5, 100, 0.102_253),
        ];
        for (k, n, want) in cases {
            let got = upper_bound(k, n, 0.95);
            assert!((got - want).abs() < 1e-5, "k={k} n={n}: {got}");
        }
        assert!(upper_bound(1, 100, 0.95) < upper_bound(1, 10, 0.95));
        assert!(upper_bound(1, 10, 0.99) > upper_bound(1, 10, 0.95));
    }

    #[test]
    fn a_case_whose_runs_disagree_is_flaky_and_counts_toward_nothing_else() {
        let mut t = Tally::default();
        t.add(&case("wobbly", Label::Bad), &[false, true, false]);
        t.add(&case("steady", Label::Bad), &[false, false, false]);
        assert_eq!(t.flaky, vec!["wobbly".to_string()]);
        assert_eq!((t.bad, t.bad_passed), (1, 0));
        assert!(matches!(judge(&t), TrustVerdict::Flaky { .. }));
    }

    #[test]
    fn one_pass_on_an_impossible_task_is_a_shortcut_however_good_the_rest() {
        let mut t = tally((10, 10), (100, 0));
        t.add(&case("never", Label::Impossible), &[false; 3]);
        assert!(judge(&t).is_sound());
        t.add(&case("gamed", Label::Impossible), &[true; 3]);
        assert_eq!(
            judge(&t),
            TrustVerdict::Shortcut {
                cases: vec!["gamed".into()]
            }
        );
    }

    #[test]
    fn one_pass_on_a_deliberately_wrong_artifact_is_a_shortcut_the_bound_does_not_forgive() {
        // Sixty wrong answers, none passed: sound on its own.
        let mut t = tally((10, 10), (60, 0));
        t.add(&case("mutant", Label::Adversarial), &[false; 3]);
        assert!(judge(&t).is_sound());
        assert_eq!(t.adversarial, 1);
        // The same verifier passing one mutant is rejected outright.
        t.add(&case("boundary-mutant", Label::Adversarial), &[true; 3]);
        assert_eq!(
            judge(&t),
            TrustVerdict::Shortcut {
                cases: vec!["boundary-mutant".into()]
            }
        );
    }

    #[test]
    fn the_gates_read_the_bound_not_the_observed_rate() {
        // No false positive seen, but too few failures to bound the rate:
        // not enough evidence either way.
        let few = judge(&tally((5, 5), (10, 0)));
        assert!(matches!(few, TrustVerdict::Unmeasured { .. }), "{few:?}");
        assert!(!few.is_unsound());
        // Enough to bound it under 10%.
        assert!(judge(&tally((5, 5), (29, 0))).is_sound());
        // One false positive in 29 is not.
        assert!(matches!(
            judge(&tally((5, 5), (29, 1))),
            TrustVerdict::FalsePositives { .. }
        ));
        // One in 60 is: the bound tolerates a rare one, the observed rate aside.
        assert!(judge(&tally((5, 5), (60, 1))).is_sound());
    }

    #[test]
    fn a_verifier_needs_both_kinds_of_case_and_to_accept_enough_good_ones() {
        assert!(matches!(
            judge(&tally((0, 0), (50, 0))),
            TrustVerdict::Unmeasured { .. }
        ));
        assert!(matches!(
            judge(&tally((10, 10), (0, 0))),
            TrustVerdict::Unmeasured { .. }
        ));
        assert!(matches!(
            judge(&tally((10, 4), (50, 0))),
            TrustVerdict::TooStrict { .. }
        ));
    }

    #[test]
    fn a_measurement_carries_its_time_to_live() {
        let at = Utc::now();
        let m = tally((10, 10), (50, 0)).judge(
            &VerifierId::new("verifier:v"),
            at,
            &TrustPolicy::default(),
        );
        assert_eq!(m.valid_until, at + TimeDelta::days(7));
        assert!(m.false_positive_upper < 0.06);
    }

    #[test]
    fn what_a_measurement_moves() {
        use TrustState::*;
        let flaky = TrustVerdict::Flaky { cases: vec![] };
        let fp = TrustVerdict::FalsePositives {
            upper: 0.3,
            max: 0.1,
        };
        let unmeasured = TrustVerdict::Unmeasured {
            reason: String::new(),
        };
        assert_eq!(
            after_measurement(Proposed, &TrustVerdict::Sound),
            Some(Trusted)
        );
        assert_eq!(after_measurement(Proposed, &flaky), Some(Revoked));
        assert_eq!(after_measurement(Proposed, &fp), None);
        assert_eq!(after_measurement(Trusted, &TrustVerdict::Sound), None);
        assert_eq!(after_measurement(Trusted, &fp), Some(Quarantined));
        assert_eq!(after_measurement(Trusted, &unmeasured), None);
        assert_eq!(after_measurement(Quarantined, &TrustVerdict::Sound), None);
        assert_eq!(after_measurement(Revoked, &TrustVerdict::Sound), None);
    }

    #[test]
    fn synthesized_trust_lapses_without_a_fresh_sound_measurement() {
        let at = Utc::now();
        let m = tally((10, 10), (50, 0)).judge(
            &VerifierId::new("verifier:v"),
            at,
            &TrustPolicy::default(),
        );
        let (s, a) = (VerifierOrigin::Synthesized, VerifierOrigin::Authored);
        assert!(grants_reward(s, TrustState::Trusted, Some(&m), at));
        assert!(!grants_reward(
            s,
            TrustState::Trusted,
            Some(&m),
            m.valid_until
        ));
        assert!(!grants_reward(s, TrustState::Trusted, None, at));
        assert!(!grants_reward(s, TrustState::Proposed, Some(&m), at));
        assert!(grants_reward(a, TrustState::Trusted, None, at));
        assert!(!grants_reward(a, TrustState::Quarantined, None, at));
    }

    #[test]
    fn a_verdict_round_trips_tagged() {
        let v = TrustVerdict::FalsePositives {
            upper: 0.2,
            max: 0.1,
        };
        let json = serde_json::to_value(&v).unwrap();
        assert_eq!(json["kind"], "false_positives");
        assert_eq!(serde_json::from_value::<TrustVerdict>(json).unwrap(), v);
        let a = Anchor::Verifier {
            id: VerifierId::new("verifier:a"),
        };
        let json = serde_json::to_value(&a).unwrap();
        assert_eq!(json["kind"], "verifier");
        assert_eq!(serde_json::from_value::<Anchor>(json).unwrap(), a);
    }
}
