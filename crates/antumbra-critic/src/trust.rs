//! Running the trust protocol: every case a verifier applies to
//! goes through it several times, and the runs are counted into a [`Tally`]
//! and judged under the policy.
//!
//! [`challenge`] is the protocol's decisive test. A deliberately wrong artifact
//! that anchored truth fails must be failed by every trusted verifier that
//! checks it, so in a challenge a known-bad case counts as one no artifact
//! can satisfy, and one pass is a shortcut.

use chrono::{DateTime, Utc};

use antumbra_core::ports::{Verifier, VerifyRequest};
use antumbra_core::{
    Case, Label, Result, RunId, Tally, TrustMeasurement, TrustPolicy, VerifierRecord,
};

/// Measure `record` on the cases it applies to.
pub async fn measure(
    verifier: &dyn Verifier,
    record: &VerifierRecord,
    cases: &[Case],
    policy: &TrustPolicy,
    now: DateTime<Utc>,
) -> Result<TrustMeasurement> {
    Ok(tally(verifier, record, cases, policy)
        .await?
        .judge(&record.id, now, policy))
}

/// Run `record` on the cases it applies to and count the runs, unjudged, so
/// a caller can fold in earlier tallies before judging.
pub async fn tally(
    verifier: &dyn Verifier,
    record: &VerifierRecord,
    cases: &[Case],
    policy: &TrustPolicy,
) -> Result<Tally> {
    let repeats = policy.repeats.max(1);
    let run_id = RunId::new(format!("trust:{}", record.id));
    let mut tally = Tally {
        repeats,
        ..Tally::default()
    };
    for case in cases.iter().filter(|c| record.applies_to(&c.task)) {
        let mut runs = Vec::with_capacity(repeats as usize);
        for step_idx in 0..repeats {
            let req = VerifyRequest {
                run_id: run_id.clone(),
                step_idx,
                dimension: "trust".into(),
                artifact: serde_json::json!({
                    "task": case.task,
                    "completion": case.completion,
                    "verify": record.spec,
                }),
            };
            runs.push(verifier.verify(&req).await?.passed);
        }
        tally.add(case, &runs);
    }
    Ok(tally)
}

/// The decisive test for one verifier: `record` on the known-bad and
/// impossible cases it applies to, with any pass a shortcut. The known-good
/// cases play no part.
pub async fn challenge(
    verifier: &dyn Verifier,
    record: &VerifierRecord,
    cases: &[Case],
    policy: &TrustPolicy,
    now: DateTime<Utc>,
) -> Result<TrustMeasurement> {
    let wrong: Vec<Case> = cases
        .iter()
        .filter(|c| c.label != Label::Good)
        .map(|c| Case {
            label: if c.label == Label::Impossible {
                Label::Impossible
            } else {
                Label::Adversarial
            },
            ..c.clone()
        })
        .collect();
    measure(verifier, record, &wrong, policy, now).await
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use async_trait::async_trait;

    use antumbra_core::ports::VerifierVerdict;
    use antumbra_core::{Anchor, TrustVerdict, VerifierOrigin, VerifierTier};

    use super::*;
    use crate::CommandVerifier;

    fn record(spec: serde_json::Value, task: Option<&str>) -> VerifierRecord {
        VerifierRecord::new(
            "strings",
            task.map(str::to_string),
            VerifierTier::Reducible,
            VerifierOrigin::Synthesized,
            spec,
            Utc::now(),
        )
    }

    fn case(id: &str, task: &str, completion: &str, label: Label) -> Case {
        Case {
            id: id.into(),
            task: task.into(),
            completion: completion.into(),
            label,
            anchor: Anchor::Reference,
        }
    }

    /// A reference, 29 wrong answers, and an artifact for a task nothing can
    /// satisfy.
    fn cases() -> Vec<Case> {
        let mut all = vec![case("ref", "t", "def swap(s): return s[::-1]", Label::Good)];
        for i in 0..29 {
            all.push(case(
                &format!("bad{i}"),
                "t",
                &format!("def swap(s): return {i}"),
                Label::Bad,
            ));
        }
        all.push(case(
            "imp",
            "t",
            "def swap(s): return None",
            Label::Impossible,
        ));
        all
    }

    #[tokio::test]
    async fn a_verifier_that_reads_the_answer_is_sound() {
        let exact = record(serde_json::json!({ "contains_all": ["s[::-1]"] }), None);
        let m = measure(
            &CommandVerifier,
            &exact,
            &cases(),
            &TrustPolicy::default(),
            Utc::now(),
        )
        .await
        .unwrap();
        assert_eq!(m.verdict, TrustVerdict::Sound, "{m:?}");
        assert_eq!((m.good, m.good_passed, m.bad, m.bad_passed), (1, 1, 29, 0));
        assert_eq!(m.impossible, 1);
    }

    #[tokio::test]
    async fn a_verifier_that_checks_only_the_signature_passes_what_it_should_not() {
        let weak = record(serde_json::json!({ "contains_all": ["def swap"] }), None);
        let m = measure(
            &CommandVerifier,
            &weak,
            &cases(),
            &TrustPolicy::default(),
            Utc::now(),
        )
        .await
        .unwrap();
        // The impossible task's artifact has the signature too: a shortcut.
        assert_eq!(
            m.verdict,
            TrustVerdict::Shortcut {
                cases: vec!["imp".into()]
            }
        );
        assert_eq!(m.bad_passed, 29);
    }

    #[tokio::test]
    async fn a_verifier_is_measured_only_on_the_task_it_checks() {
        let mut all = cases();
        all.push(case("elsewhere", "u", "anything", Label::Bad));
        let exact = record(
            serde_json::json!({ "contains_all": ["s[::-1]"] }),
            Some("t"),
        );
        let m = measure(
            &CommandVerifier,
            &exact,
            &all,
            &TrustPolicy::default(),
            Utc::now(),
        )
        .await
        .unwrap();
        assert_eq!(m.bad, 29);
    }

    /// Passes every other time it is asked.
    struct Coin(AtomicU32);

    #[async_trait]
    impl Verifier for Coin {
        async fn verify(&self, _req: &VerifyRequest) -> Result<VerifierVerdict> {
            let n = self.0.fetch_add(1, Ordering::SeqCst);
            Ok(VerifierVerdict {
                passed: n.is_multiple_of(2),
                value: 0.0,
            })
        }
    }

    #[tokio::test]
    async fn a_verifier_that_disagrees_with_itself_is_flaky() {
        let any = record(serde_json::json!({}), None);
        let m = measure(
            &Coin(AtomicU32::new(0)),
            &any,
            &cases(),
            &TrustPolicy::default(),
            Utc::now(),
        )
        .await
        .unwrap();
        assert!(matches!(m.verdict, TrustVerdict::Flaky { .. }), "{m:?}");
    }

    #[tokio::test]
    async fn a_challenge_allows_no_pass_on_a_deliberately_wrong_artifact() {
        let policy = TrustPolicy::default();
        let exact = record(serde_json::json!({ "contains_all": ["s[::-1]"] }), None);
        let clean = challenge(&CommandVerifier, &exact, &cases(), &policy, Utc::now())
            .await
            .unwrap();
        assert!(!clean.verdict.is_unsound(), "{clean:?}");
        assert_eq!((clean.impossible, clean.adversarial), (1, 29));
        // One wrong artifact it passes is enough.
        let mut wrong = cases();
        wrong.push(case(
            "sneaky",
            "t",
            "def swap(s): s[::-1]; return s",
            Label::Bad,
        ));
        let caught = challenge(&CommandVerifier, &exact, &wrong, &policy, Utc::now())
            .await
            .unwrap();
        assert_eq!(
            caught.verdict,
            TrustVerdict::Shortcut {
                cases: vec!["sneaky".into()]
            }
        );
    }
}
