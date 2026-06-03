//! # antumbra-boundary — ADR-0004 (the keystone)
//!
//! Counterfactual scope search. The behavior **B** is held fixed; the context
//! is varied along candidate governing dimensions; the frozen population is
//! re-probed (via the [`AcceptabilityProbe`] port) until B's acceptability
//! *flips*. The payload is the governing dimension and the nearest in-scope
//! context **C'** — *where and why* the rule applies — never a negated goal.
//!
//! This is job #1 of ADR-0004. Building the persisted [`FailureBoundary`] from
//! a finding (job #2's store side) is [`finding_to_boundary`].

use chrono::{DateTime, Utc};
use serde_json::Value;

use antumbra_core::ports::AcceptabilityProbe;
use antumbra_core::{BoundaryId, FailureBoundary, Generation, Grain, Result};

/// A recovered scope: the governing feature and the nearest context where the
/// behavior becomes acceptable.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryFinding {
    pub behavior: String,
    pub governing_feature: String,
    /// C — where the behavior was judged incorrect.
    pub fail_context: Value,
    /// C' — the nearest context where it is acceptable.
    pub near_ok_context: Value,
}

/// A candidate governing dimension and the alternative values to probe.
pub type Candidate = (String, Vec<Value>);

/// Hold `behavior` fixed and vary each candidate feature over its alternatives
/// until the probe judges the behavior acceptable. Returns the first recovered
/// scope, or `None` if no single-feature change flips acceptability.
///
/// `fail_context` must be a JSON object (the context C where B is incorrect).
pub async fn find_scope(
    behavior: &str,
    fail_context: &Value,
    candidates: &[Candidate],
    probe: &dyn AcceptabilityProbe,
) -> Result<Option<BoundaryFinding>> {
    for (feature, values) in candidates {
        for value in values {
            let mut candidate = fail_context.clone();
            candidate[feature.as_str()] = value.clone();
            if probe.acceptable(behavior, &candidate).await? {
                return Ok(Some(BoundaryFinding {
                    behavior: behavior.to_string(),
                    governing_feature: feature.clone(),
                    fail_context: fail_context.clone(),
                    near_ok_context: candidate,
                }));
            }
        }
    }
    Ok(None)
}

/// Search over **whole candidate contexts** (each carrying its own `verify`),
/// rather than single-feature swaps. This is what maps a *specific expert's*
/// competence boundary: the expert acts the fixed behavior, succeeds on the
/// in-scope task and fails on the out-of-scope one, and the first context it
/// finds acceptable is C'. `governing_feature` labels what differs.
pub async fn find_scope_over_contexts(
    behavior: &str,
    governing_feature: &str,
    fail_context: &Value,
    candidate_contexts: &[Value],
    probe: &dyn AcceptabilityProbe,
) -> Result<Option<BoundaryFinding>> {
    for context in candidate_contexts {
        if probe.acceptable(behavior, context).await? {
            return Ok(Some(BoundaryFinding {
                behavior: behavior.to_string(),
                governing_feature: governing_feature.to_string(),
                fail_context: fail_context.clone(),
                near_ok_context: context.clone(),
            }));
        }
    }
    Ok(None)
}

/// Promote a finding to a persistable [`FailureBoundary`]. Confidence and the
/// embedded `context_vec` are supplied by the caller (the loop owns the
/// embedder); the boundary is only actionable because C' was recovered.
pub fn finding_to_boundary(
    id: BoundaryId,
    finding: &BoundaryFinding,
    grain: Grain,
    confidence: f32,
    context_vec: Option<Vec<f32>>,
    generation: Generation,
    now: DateTime<Utc>,
) -> FailureBoundary {
    FailureBoundary {
        id,
        behavior: finding.behavior.clone(),
        fail_context: finding.fail_context.clone(),
        near_ok_context: Some(finding.near_ok_context.clone()),
        governing_features: vec![finding.governing_feature.clone()],
        grain: Some(grain),
        context_vec,
        confidence,
        generation,
        created_at: now,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antumbra_core::testing::FeatureProbe;

    #[tokio::test]
    async fn recovers_the_governing_feature_and_c_prime() {
        // `npm install` is wrong in a Deno repo (C), right in a node repo (C').
        let probe = FeatureProbe {
            feature: "runtime".into(),
            ok_value: serde_json::json!("node"),
        };
        let fail = serde_json::json!({ "runtime": "deno", "task": "install deps" });
        let candidates = vec![("runtime".to_string(), vec![serde_json::json!("node")])];

        let finding = find_scope("npm install", &fail, &candidates, &probe)
            .await
            .unwrap()
            .expect("scope recovered");

        assert_eq!(finding.governing_feature, "runtime");
        assert_eq!(
            finding.near_ok_context["runtime"],
            serde_json::json!("node")
        );

        let boundary = finding_to_boundary(
            BoundaryId::new("b:1"),
            &finding,
            Grain::Project,
            0.8,
            None,
            Generation::ZERO,
            Utc::now(),
        );
        assert!(boundary.is_actionable());
        assert_eq!(boundary.governing_features, vec!["runtime".to_string()]);
    }

    #[tokio::test]
    async fn returns_none_when_no_change_flips_acceptability() {
        let probe = FeatureProbe {
            feature: "runtime".into(),
            ok_value: serde_json::json!("node"),
        };
        let fail = serde_json::json!({ "runtime": "deno" });
        // we only offer values that never satisfy the probe
        let candidates = vec![("runtime".to_string(), vec![serde_json::json!("bun")])];
        let finding = find_scope("npm install", &fail, &candidates, &probe)
            .await
            .unwrap();
        assert!(finding.is_none());
    }

    #[tokio::test]
    async fn over_contexts_returns_first_acceptable_context() {
        // Maps an expert's competence: it fails the "multiply" task and passes
        // the "add" task; the recovered C' is the first acceptable context.
        let probe = FeatureProbe {
            feature: "op".into(),
            ok_value: serde_json::json!("add"),
        };
        let fail = serde_json::json!({ "op": "multiply" });
        let candidates = vec![
            serde_json::json!({ "op": "reverse" }),
            serde_json::json!({ "op": "add" }),
        ];
        let finding =
            find_scope_over_contexts("implement the op", "op", &fail, &candidates, &probe)
                .await
                .unwrap()
                .expect("an in-scope context exists");
        assert_eq!(finding.governing_feature, "op");
        assert_eq!(finding.near_ok_context["op"], serde_json::json!("add"));
        assert_eq!(finding.fail_context["op"], serde_json::json!("multiply"));
    }
}
