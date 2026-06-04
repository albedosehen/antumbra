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

use std::collections::BTreeSet;

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

/// The set of a key's values across the given contexts, stringified for set ops.
fn value_set(contexts: &[&Value], key: &str) -> BTreeSet<String> {
    contexts
        .iter()
        .filter_map(|c| c.get(key))
        .map(|v| v.to_string())
        .collect()
}

/// **Discover** the governing feature, rather than being told it. Probe every
/// context, partition into pass/fail, then find the context key whose value
/// alone separates the two (its passing-values are disjoint from its
/// failing-values) — preferring the simplest such key (fewest distinct values).
/// Returns `None` when no boundary exists (all pass or all fail) or no single
/// feature explains the split. This is the autonomous half of ADR-0004: the
/// system names *its own* governing feature from evaluated behavior.
pub async fn discover_boundary(
    behavior: &str,
    contexts: &[Value],
    probe: &dyn AcceptabilityProbe,
) -> Result<Option<BoundaryFinding>> {
    let mut passing: Vec<&Value> = Vec::new();
    let mut failing: Vec<&Value> = Vec::new();
    for context in contexts {
        if probe.acceptable(behavior, context).await? {
            passing.push(context);
        } else {
            failing.push(context);
        }
    }
    if passing.is_empty() || failing.is_empty() {
        return Ok(None);
    }

    let mut keys: BTreeSet<String> = BTreeSet::new();
    for context in contexts {
        if let Some(obj) = context.as_object() {
            keys.extend(obj.keys().filter(|k| *k != "verify").cloned());
        }
    }

    let mut best: Option<(String, usize)> = None;
    for key in &keys {
        let pv = value_set(&passing, key);
        let fv = value_set(&failing, key);
        if pv.is_empty() || fv.is_empty() || !pv.is_disjoint(&fv) {
            continue;
        }
        let distinct = pv.union(&fv).count();
        let better = match &best {
            None => true,
            Some((_, d)) => distinct < *d,
        };
        if better {
            best = Some((key.clone(), distinct));
        }
    }

    let Some((governing_feature, _)) = best else {
        return Ok(None);
    };
    Ok(Some(BoundaryFinding {
        behavior: behavior.to_string(),
        governing_feature,
        fail_context: failing[0].clone(),
        near_ok_context: passing[0].clone(),
    }))
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
    ok_context_vec: Option<Vec<f32>>,
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
        ok_context_vec,
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

    #[tokio::test]
    async fn discover_infers_the_governing_feature_from_pass_fail() {
        // The probe is told nothing; it infers that `op` (not the incidental
        // `note`) governs, from which contexts pass vs fail.
        let probe = FeatureProbe {
            feature: "op".into(),
            ok_value: serde_json::json!("add"),
        };
        let contexts = vec![
            serde_json::json!({ "op": "add", "note": "x" }),
            serde_json::json!({ "op": "multiply", "note": "y" }),
            serde_json::json!({ "op": "reverse", "note": "x" }),
        ];
        let finding = discover_boundary("implement the op", &contexts, &probe)
            .await
            .unwrap()
            .expect("a governing feature is discoverable");
        assert_eq!(finding.governing_feature, "op");
        assert_eq!(finding.near_ok_context["op"], serde_json::json!("add"));
    }

    #[tokio::test]
    async fn discover_returns_none_when_no_single_feature_separates() {
        // Acceptable iff a==1 AND b==1, so neither `a` nor `b` alone splits
        // pass from fail -> no single governing feature can be named.
        struct AndProbe;
        #[async_trait::async_trait]
        impl antumbra_core::ports::AcceptabilityProbe for AndProbe {
            async fn acceptable(&self, _behavior: &str, ctx: &Value) -> Result<bool> {
                Ok(ctx["a"] == serde_json::json!(1) && ctx["b"] == serde_json::json!(1))
            }
        }
        let contexts = vec![
            serde_json::json!({ "a": 1, "b": 1 }),
            serde_json::json!({ "a": 1, "b": 0 }),
            serde_json::json!({ "a": 0, "b": 1 }),
        ];
        assert!(discover_boundary("do it", &contexts, &AndProbe)
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn discover_returns_none_when_everything_passes() {
        let probe = FeatureProbe {
            feature: "op".into(),
            ok_value: serde_json::json!("add"),
        };
        let contexts = vec![
            serde_json::json!({ "op": "add" }),
            serde_json::json!({ "op": "add", "x": 1 }),
        ];
        assert!(discover_boundary("do it", &contexts, &probe)
            .await
            .unwrap()
            .is_none());
    }
}
