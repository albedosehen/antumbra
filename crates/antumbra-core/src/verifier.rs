//! Verifiers as governed records (ADR-0022 S-4): the system may propose a
//! check, and only measurement against ground truth it did not produce makes
//! it trusted.
//!
//! A verifier is data, a spec `CommandVerifier` executes, addressed by the
//! hash of its content. So the check that was measured is exactly the check
//! that grants reward, and a changed check is a new verifier that starts with
//! no power. An authored verifier is the ground and is trusted by authorship.
//! A synthesized one moves through four states:
//! - **proposed:** measured beside authored truth, granting nothing;
//! - **trusted:** grants reward while its last sound measurement is younger
//!   than its time to live;
//! - **quarantined:** a measurement found it unsound; it grants nothing, and
//!   what it granted stays on record;
//! - **revoked:** out of use, its rows retained.
//!
//! The only way into trusted is a measurement (see [`crate::trust`]). A person
//! may quarantine or revoke a verifier, never promote one.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{AntumbraError, Result};
use crate::ids::VerifierId;
use crate::trust::TrustVerdict;

/// What decides a verifier's verdict. There is no tier for oracles derived
/// from the implementation: generated tests that assert observed behaviour
/// are refused outright, since a check that learns what the code does cannot
/// say what it should do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VerifierTier {
    /// Something frozen decides: a compiler, a type checker, a schema, an
    /// exit status, a differential check against a reference. The proposer
    /// only wires it up.
    Reducible,
    /// A metamorphic relation or property check: a verdict without anyone
    /// stating the expected output.
    Partial,
}

impl VerifierTier {
    pub fn as_str(self) -> &'static str {
        match self {
            VerifierTier::Reducible => "reducible",
            VerifierTier::Partial => "partial",
        }
    }
}

impl std::str::FromStr for VerifierTier {
    type Err = AntumbraError;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "reducible" => Ok(VerifierTier::Reducible),
            "partial" => Ok(VerifierTier::Partial),
            "derived" => Err(AntumbraError::rejected(
                "oracles derived from the implementation are refused (ADR-0022 S-4): \
                 a check that learns what the code does cannot say what it should do",
            )),
            other => Err(AntumbraError::rejected(format!(
                "unknown verifier tier `{other}` (use reducible or partial)"
            ))),
        }
    }
}

/// Who wrote a verifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VerifierOrigin {
    /// A person: the ground, trusted by authorship.
    Authored,
    /// The system proposed it: trusted only by measurement.
    Synthesized,
}

impl VerifierOrigin {
    pub fn as_str(self) -> &'static str {
        match self {
            VerifierOrigin::Authored => "authored",
            VerifierOrigin::Synthesized => "synthesized",
        }
    }
}

/// A verifier as the namespace keeps it. Everything but its origin and who
/// proposed it is part of its address, so none of it can change in place.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerifierRecord {
    pub id: VerifierId,
    /// Where it is measured and where it may grant reward: a region of tasks.
    pub domain: String,
    /// The one task it checks, or every task in its domain when `None`.
    #[serde(default)]
    pub task: Option<String>,
    pub tier: VerifierTier,
    pub origin: VerifierOrigin,
    /// What `CommandVerifier` runs.
    pub spec: serde_json::Value,
    /// Who or what proposed it, for the audit trail.
    #[serde(default)]
    pub proposed_by: Option<String>,
    pub created_at: DateTime<Utc>,
}

impl VerifierRecord {
    pub fn new(
        domain: impl Into<String>,
        task: Option<String>,
        tier: VerifierTier,
        origin: VerifierOrigin,
        spec: serde_json::Value,
        now: DateTime<Utc>,
    ) -> Self {
        let domain = domain.into();
        let id = verifier_address(&domain, task.as_deref(), tier, &spec);
        VerifierRecord {
            id,
            domain,
            task,
            tier,
            origin,
            spec,
            proposed_by: None,
            created_at: now,
        }
    }

    /// Whether its address still matches its content. A mismatch means the
    /// stored check is not the one that was measured.
    pub fn is_intact(&self) -> bool {
        verifier_address(&self.domain, self.task.as_deref(), self.tier, &self.spec) == self.id
    }

    /// Whether it checks `task`.
    pub fn applies_to(&self, task: &str) -> bool {
        self.task.as_deref().is_none_or(|t| t == task)
    }
}

/// The content address of a verifier: the hash of its domain, task, tier and
/// spec in canonical form.
pub fn verifier_address(
    domain: &str,
    task: Option<&str>,
    tier: VerifierTier,
    spec: &serde_json::Value,
) -> VerifierId {
    let content = serde_json::json!({
        "domain": domain,
        "task": task,
        "tier": tier.as_str(),
        "spec": spec,
    });
    let digest = Sha256::digest(canonical_json(&content).as_bytes());
    let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    VerifierId::new(format!("verifier:{hex}"))
}

/// The canonical text of a JSON value: object keys sorted at every depth and
/// no whitespace, so the same content always has the same address whatever
/// order it was written in.
pub fn canonical_json(value: &serde_json::Value) -> String {
    let mut out = String::new();
    write_canonical(value, &mut out);
    out
}

fn write_canonical(value: &serde_json::Value, out: &mut String) {
    match value {
        serde_json::Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (i, key) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::Value::String(key.clone()).to_string());
                out.push(':');
                write_canonical(&map[key], out);
            }
            out.push('}');
        }
        serde_json::Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(item, out);
            }
            out.push(']');
        }
        scalar => out.push_str(&scalar.to_string()),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TrustState {
    Proposed,
    Trusted,
    Quarantined,
    Revoked,
}

impl TrustState {
    pub fn as_str(self) -> &'static str {
        match self {
            TrustState::Proposed => "proposed",
            TrustState::Trusted => "trusted",
            TrustState::Quarantined => "quarantined",
            TrustState::Revoked => "revoked",
        }
    }

    /// Where a verifier starts: an authored one is trusted by authorship, a
    /// synthesized one with no power.
    pub fn initial(origin: VerifierOrigin) -> TrustState {
        match origin {
            VerifierOrigin::Authored => TrustState::Trusted,
            VerifierOrigin::Synthesized => TrustState::Proposed,
        }
    }

    /// The states reachable in one step. Quarantine is not undone: the same
    /// content is the same check, and a repaired check is a new verifier.
    pub fn allowed_next(self) -> &'static [TrustState] {
        use TrustState::*;
        match self {
            Proposed => &[Trusted, Revoked],
            Trusted => &[Quarantined, Revoked],
            Quarantined => &[Revoked],
            Revoked => &[],
        }
    }

    /// Guarded transition: refuses a move the state machine forbids, and
    /// trust from anything but a sound measurement.
    pub fn transition(self, to: TrustState, cause: &TrustCause) -> Result<TrustState> {
        if !self.allowed_next().contains(&to) {
            return Err(AntumbraError::InvalidTransition {
                entity: "verifier",
                from: self.as_str().into(),
                to: to.as_str().into(),
            });
        }
        let sound = matches!(
            cause,
            TrustCause::Measured {
                verdict: TrustVerdict::Sound,
                ..
            }
        );
        if to == TrustState::Trusted && !sound {
            return Err(AntumbraError::rejected(
                "only a sound measurement trusts a verifier (ADR-0022 S-4)",
            ));
        }
        Ok(to)
    }
}

/// Why a verifier changed state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum TrustCause {
    /// The measurement taken at `at` moved it: the only cause that trusts.
    Measured {
        at: DateTime<Utc>,
        verdict: TrustVerdict,
    },
    /// A person asked.
    Operator {
        #[serde(default)]
        note: Option<String>,
    },
}

/// One change of a verifier's state, as recorded.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerifierTransition {
    pub verifier: VerifierId,
    pub from: TrustState,
    pub to: TrustState,
    pub cause: TrustCause,
    pub at: DateTime<Utc>,
}

/// The verifier a task's `verify` names, when it names one from the
/// namespace (`{"verifier": "verifier:..."}`) rather than carrying a spec.
pub fn named_verifier(spec: &serde_json::Value) -> Option<VerifierId> {
    spec.get("verifier")
        .and_then(|v| v.as_str())
        .map(VerifierId::new)
}

/// The reward a named verifier granted that a run trained on: every pass of
/// its that became training data. So when it is quarantined, what it taught
/// can be found.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifierGrant {
    pub verifier: VerifierId,
    pub passes: u32,
}

/// Count one trained-on pass toward the verifier `spec` names, if it names
/// one. A spec written into the task is authored and counts toward nothing.
pub fn count_grant(grants: &mut Vec<VerifierGrant>, spec: &serde_json::Value) {
    let Some(id) = named_verifier(spec) else {
        return;
    };
    match grants.iter_mut().find(|g| g.verifier == id) {
        Some(g) => g.passes += 1,
        None => grants.push(VerifierGrant {
            verifier: id,
            passes: 1,
        }),
    }
}

/// An answer a named verifier judged in training, on a task training learns
/// from. The loop rechecks these against the tasks' authored anchors, so a
/// trusted verifier is measured again on the answers of the policy it is
/// rewarding, not only on the cases it was promoted on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JudgedSample {
    pub verifier: VerifierId,
    pub task: String,
    pub completion: String,
    /// Its verdict as training saw it.
    pub passed: bool,
    /// Whether the pass became training data.
    pub rewarded: bool,
}

impl JudgedSample {
    /// The sample to keep when `spec` names a verifier, rewarded when it
    /// passed. `None` for a spec written into the task: it is authored, and
    /// there is nothing to recheck.
    pub fn named(
        spec: &serde_json::Value,
        task: &str,
        completion: &str,
        passed: bool,
    ) -> Option<Self> {
        named_verifier(spec).map(|verifier| JudgedSample {
            verifier,
            task: task.to_string(),
            completion: completion.to_string(),
            passed,
            rewarded: passed,
        })
    }
}

/// The state a history leaves a verifier in: the last move's, or where its
/// origin starts it when it has none.
pub fn current_trust(origin: VerifierOrigin, history: &[VerifierTransition]) -> TrustState {
    history.last().map_or(TrustState::initial(origin), |t| t.to)
}

#[cfg(test)]
mod tests {
    use super::*;
    use TrustState::*;

    fn spec() -> serde_json::Value {
        serde_json::json!({ "program": "python", "args": ["-c", "pass"], "extract_code": true })
    }

    fn sound() -> TrustCause {
        TrustCause::Measured {
            at: Utc::now(),
            verdict: TrustVerdict::Sound,
        }
    }

    fn operator() -> TrustCause {
        TrustCause::Operator { note: None }
    }

    #[test]
    fn the_address_is_the_content_in_any_key_order() {
        let a = serde_json::json!({ "program": "sh", "args": ["-c", "exit 0"] });
        let b: serde_json::Value =
            serde_json::from_str(r#"{"args":["-c","exit 0"],"program":"sh"}"#).unwrap();
        let tier = VerifierTier::Reducible;
        assert_eq!(
            verifier_address("strings", None, tier, &a),
            verifier_address("strings", None, tier, &b)
        );
        let id = verifier_address("strings", None, tier, &a);
        assert!(id.as_str().starts_with("verifier:"));
        assert_eq!(id.as_str().len(), "verifier:".len() + 64);
        // Every part of it is part of the address.
        for other in [
            verifier_address("lists", None, tier, &a),
            verifier_address("strings", Some("strings/t1"), tier, &a),
            verifier_address("strings", None, VerifierTier::Partial, &a),
            verifier_address(
                "strings",
                None,
                tier,
                &serde_json::json!({ "program": "sh" }),
            ),
        ] {
            assert_ne!(other, id);
        }
    }

    #[test]
    fn canonical_json_sorts_keys_at_every_depth() {
        let v: serde_json::Value =
            serde_json::from_str(r#"{"b":{"y":1,"x":[{"q":2,"p":"s"}]},"a":null}"#).unwrap();
        assert_eq!(
            canonical_json(&v),
            r#"{"a":null,"b":{"x":[{"p":"s","q":2}],"y":1}}"#
        );
    }

    #[test]
    fn a_record_whose_content_changed_is_not_intact() {
        let mut r = VerifierRecord::new(
            "strings",
            Some("strings/t1".into()),
            VerifierTier::Reducible,
            VerifierOrigin::Synthesized,
            spec(),
            Utc::now(),
        );
        assert!(r.is_intact());
        assert!(r.applies_to("strings/t1") && !r.applies_to("strings/t2"));
        r.spec = serde_json::json!({ "program": "true" });
        assert!(!r.is_intact());
    }

    #[test]
    fn derived_oracles_are_refused_by_name() {
        assert_eq!(
            "reducible".parse::<VerifierTier>().unwrap(),
            VerifierTier::Reducible
        );
        let refused = "derived".parse::<VerifierTier>().unwrap_err();
        assert!(refused.to_string().contains("refused"), "{refused}");
        assert!("oracle".parse::<VerifierTier>().is_err());
    }

    #[test]
    fn only_a_sound_measurement_trusts() {
        assert_eq!(Proposed.transition(Trusted, &sound()).unwrap(), Trusted);
        assert!(Proposed.transition(Trusted, &operator()).is_err());
        let flaky = TrustCause::Measured {
            at: Utc::now(),
            verdict: TrustVerdict::Flaky {
                cases: vec!["c".into()],
            },
        };
        assert!(Proposed.transition(Trusted, &flaky).is_err());
    }

    #[test]
    fn quarantine_is_not_undone_and_revocation_is_final() {
        assert_eq!(
            Trusted.transition(Quarantined, &operator()).unwrap(),
            Quarantined
        );
        assert!(Quarantined.transition(Trusted, &sound()).is_err());
        assert_eq!(
            Quarantined.transition(Revoked, &operator()).unwrap(),
            Revoked
        );
        for to in [Proposed, Trusted, Quarantined, Revoked] {
            assert!(Revoked.transition(to, &sound()).is_err(), "{to:?}");
        }
    }

    #[test]
    fn grants_count_only_toward_a_named_verifier() {
        let named = serde_json::json!({ "verifier": "verifier:a" });
        let other = serde_json::json!({ "verifier": "verifier:b" });
        let inline = serde_json::json!({ "program": "python" });
        let mut grants = Vec::new();
        for spec in [&named, &inline, &named, &other, &serde_json::Value::Null] {
            count_grant(&mut grants, spec);
        }
        assert_eq!(
            grants,
            vec![
                VerifierGrant {
                    verifier: VerifierId::new("verifier:a"),
                    passes: 2
                },
                VerifierGrant {
                    verifier: VerifierId::new("verifier:b"),
                    passes: 1
                },
            ]
        );
        assert_eq!(named_verifier(&inline), None);
    }

    #[test]
    fn only_a_named_verifiers_verdict_is_kept_for_the_recheck() {
        let named = serde_json::json!({ "verifier": "verifier:a" });
        let kept = JudgedSample::named(&named, "t1", "answer", true).expect("named");
        assert_eq!(kept.verifier, VerifierId::new("verifier:a"));
        assert_eq!(
            (kept.task.as_str(), kept.completion.as_str()),
            ("t1", "answer")
        );
        assert!(kept.passed && kept.rewarded);
        let failed = JudgedSample::named(&named, "t1", "answer", false).expect("named");
        assert!(!failed.rewarded);
        let inline = serde_json::json!({ "program": "python" });
        assert_eq!(JudgedSample::named(&inline, "t1", "answer", true), None);
    }

    #[test]
    fn authored_starts_trusted_and_synthesized_with_no_power() {
        assert_eq!(current_trust(VerifierOrigin::Authored, &[]), Trusted);
        assert_eq!(current_trust(VerifierOrigin::Synthesized, &[]), Proposed);
        let moved = VerifierTransition {
            verifier: VerifierId::new("verifier:x"),
            from: Proposed,
            to: Trusted,
            cause: sound(),
            at: Utc::now(),
        };
        assert_eq!(
            current_trust(VerifierOrigin::Synthesized, &[moved]),
            Trusted
        );
    }
}
