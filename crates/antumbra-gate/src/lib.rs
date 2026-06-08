//! # antumbra-gate: the boundary-conditioned gate (v0 heuristic)
//!
//! A learned, boundary-conditioned latent mixer is the north-star gate. v0
//! ships the honest fallback we call "coverage routing (still
//! useful)": score each expert by capability similarity to the task, subtract
//! the in-scope inhibition any counterfactual boundary imposes, take the top-k,
//! and **escalate** when nothing clears the in-scope bar. The learned mixer
//! plugs in behind the same `route` signature later.
//!
//! ## Out-of-scope detection is relative, not absolute
//!
//! Routing among in-scope experts is just nearest-capability. Deciding that a
//! task is out of *every* expert's scope is the harder, keystone half (the
//! counterfactual boundary of competence) and is an out-of-distribution problem. Validation showed an **absolute**
//! similarity floor cannot do it: sentence-embedding cosine for short texts is
//! compressed into a high band (~0.6-0.9 for everything), so an out-of-scope
//! task still clears any usable absolute bar. The fix, grounded in the OOD
//! literature, is a **relative** score that cancels the non-discriminative,
//! shared direction:
//!
//! - Relative Mahalanobis Distance (arXiv:2106.09022) shows near-OOD fails
//!   because non-discriminative dimensions make in- and out-of-domain look
//!   equidistant; the cure is to cancel that shared background. RMD does it with
//!   a class-agnostic Gaussian (covariance whitening), infeasible here, where
//!   one vector per expert cannot estimate a covariance.
//! - The valid cosine-space realization is the **difference of the two nearest
//!   prototypes**: `coverage = cos(task, e₁) − cos(task, e₂)` (top-1 minus
//!   top-2). The shared "generic-code" direction contributes near-equally to
//!   both terms and cancels, leaving only discriminative signal, exactly RMD's
//!   intent. (Subtracting the *centroid* instead does **not** work: the centroid
//!   absorbs the shared direction, so `cos(task, centroid) ≈ cos(task, e₁)` and
//!   coverage collapses to ~0 for in- and out-of-scope alike. This was measured,
//!   not assumed.) Deep-kNN OOD (arXiv:2204.06507) is the same family: distance
//!   to nearest in-distribution prototypes on L2-normalized features (our
//!   capability vectors already are), non-parametric.
//! - Selective prediction (arXiv:1705.08500): escalation is abstention; the
//!   `coverage_threshold` is the risk-coverage knob, calibrated per deployment.
//!
//! A confident counterfactual boundary is subtracted from coverage, so it can force
//! escalation independently. Known v0 limitation: a task served *equally well*
//! by two experts has a small margin and will escalate; the prototype-margin
//! conflates "ambiguous between in-scope experts" with "out of scope". Escalating
//! such cases is safe for coverage routing; the north-star heterogeneous composed
//! model, which blends adapters, dissolves it.

use antumbra_core::{Expert, ExpertId, FailureBoundary};

#[derive(Debug, Clone, Copy)]
pub struct GateConfig {
    /// Abstention threshold on **relative coverage** (the top expert's
    /// similarity above the population background). Below it the gate escalates.
    /// Relative scoring (RMD, arXiv:2106.09022) defeats the compressed near-OOD
    /// cosine band an absolute floor cannot; the value is the selective-prediction
    /// risk-coverage knob (arXiv:1705.08500), calibrated per deployment.
    pub coverage_threshold: f32,
    /// Cosine-similarity radius inside which a boundary's penalty applies.
    pub inhibition_radius: f32,
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            coverage_threshold: 0.08,
            inhibition_radius: 0.5,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScoredExpert {
    pub id: ExpertId,
    pub score: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GateDecision {
    /// The selected in-scope experts (top-k), empty when escalating.
    pub chosen: Vec<ExpertId>,
    /// True when the boundary/coverage says out-of-scope: consult the flagship.
    pub escalate: bool,
    /// All scored experts, best first (for inspection / the TUI).
    pub ranked: Vec<ScoredExpert>,
    /// Relative coverage of the best expert: the top-1-minus-top-2 capability
    /// margin, less any boundary inhibition. This is the abstention signal
    /// compared against `coverage_threshold`; surfaced for inspection.
    pub coverage: f32,
}

/// Score and select experts for a task embedding.
///
/// Ranking is `score(e) = cosine(e.capability_vec, task) -
/// max_boundary_inhibition(task)` (experts without a capability vector are
/// skipped). The escalation decision uses **relative coverage**: the
/// top-1-minus-top-2 capability margin, less boundary inhibition, so it is not
/// fooled by the compressed near-OOD cosine band (see module docs).
pub fn route(
    task_vec: &[f32],
    experts: &[Expert],
    boundaries: &[FailureBoundary],
    k: usize,
    cfg: &GateConfig,
) -> GateDecision {
    let inhibition = boundaries
        .iter()
        .map(|b| b.inhibition_for(task_vec, cfg.inhibition_radius))
        .fold(0.0f32, f32::max);

    // Raw capability similarities, best first. The margin is taken on these (not
    // the inhibited score) so a uniform inhibition does not cancel out of it.
    let mut sims: Vec<(ExpertId, f32)> = experts
        .iter()
        .filter_map(|e| {
            e.capability_similarity(task_vec)
                .map(|sim| (e.id.clone(), sim))
        })
        .collect();
    sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Relative coverage: prototype margin (top-1 minus top-2), which cancels the
    // shared background; with a single expert it degrades to the absolute score.
    // Boundary inhibition is then subtracted so a confident boundary can escalate.
    let coverage_base = match sims.len() {
        0 => f32::MIN,
        1 => sims[0].1,
        _ => sims[0].1 - sims[1].1,
    };
    let coverage = coverage_base - inhibition;
    let escalate = sims.is_empty() || coverage < cfg.coverage_threshold;

    let ranked: Vec<ScoredExpert> = sims
        .iter()
        .map(|(id, sim)| ScoredExpert {
            id: id.clone(),
            score: sim - inhibition,
        })
        .collect();
    let chosen = if escalate {
        Vec::new()
    } else {
        ranked.iter().take(k).map(|s| s.id.clone()).collect()
    };

    GateDecision {
        chosen,
        escalate,
        ranked,
        coverage,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antumbra_core::{BoundaryId, Generation, Grain};
    use chrono::Utc;

    fn expert(key: &str, vec: Vec<f32>) -> Expert {
        Expert {
            id: ExpertId::new(key),
            name: key.into(),
            base_model: "base".into(),
            artifact_uri: "mem://x".into(),
            capability_card: serde_json::Value::Null,
            capability_vec: Some(vec),
            fitness: 0.5,
            frozen_at: None,
            generation: Generation::ZERO,
            owner: None,
            compartment: None,
            created_at: Utc::now(),
        }
    }

    fn boundary(ctx: Vec<f32>, confidence: f32) -> FailureBoundary {
        FailureBoundary {
            id: BoundaryId::new("b:1"),
            behavior: "do thing".into(),
            fail_context: serde_json::json!({}),
            near_ok_context: Some(serde_json::json!({})),
            governing_features: vec!["f".into()],
            grain: Some(Grain::Project),
            context_vec: Some(ctx),
            ok_context_vec: None,
            confidence,
            generation: Generation::ZERO,
            created_at: Utc::now(),
        }
    }

    #[test]
    fn picks_nearest_expert_in_scope() {
        let experts = vec![expert("a", vec![1.0, 0.0]), expert("b", vec![0.0, 1.0])];
        let decision = route(&[0.9, 0.1], &experts, &[], 1, &GateConfig::default());
        assert!(!decision.escalate);
        assert_eq!(decision.chosen, vec![ExpertId::new("a")]);
    }

    #[test]
    fn escalates_when_best_below_threshold() {
        let experts = vec![expert("a", vec![1.0, 0.0])];
        let cfg = GateConfig {
            coverage_threshold: 0.95,
            ..GateConfig::default()
        };
        // One expert -> absolute fallback; task is off-axis -> below the bar.
        let decision = route(&[0.3, 0.7], &experts, &[], 1, &cfg);
        assert!(decision.escalate);
        assert!(decision.chosen.is_empty());
    }

    #[test]
    fn boundary_inhibition_can_force_escalation() {
        let experts = vec![expert("a", vec![1.0, 0.0])];
        // Without a boundary the expert is a perfect match (score 1.0).
        let cfg = GateConfig {
            coverage_threshold: 0.1,
            ..GateConfig::default()
        };
        let clear = route(&[1.0, 0.0], &experts, &[], 1, &cfg);
        assert!(!clear.escalate);

        // A confident boundary right on the task cancels the similarity
        // (1.0 - 1.0 = 0.0), dropping below the bar -> escalate.
        let bounds = vec![boundary(vec![1.0, 0.0], 1.0)];
        let decision = route(&[1.0, 0.0], &experts, &bounds, 1, &cfg);
        assert!(decision.escalate);
    }

    #[test]
    fn relative_coverage_escalates_generic_query_an_absolute_floor_would_keep() {
        // Two orthogonal specialists. A task aligned with one is clearly in
        // scope; a task equidistant from both is generic/out-of-scope despite a
        // high *absolute* cosine (0.707 to each) that any usable floor admits.
        let experts = vec![expert("a", vec![1.0, 0.0]), expert("b", vec![0.0, 1.0])];
        let cfg = GateConfig {
            coverage_threshold: 0.1,
            ..GateConfig::default()
        };

        let focused = route(&[1.0, 0.0], &experts, &[], 1, &cfg);
        assert!(!focused.escalate);
        assert_eq!(focused.chosen, vec![ExpertId::new("a")]);
        assert!(focused.coverage > 0.1);

        // Equidistant task: cos to each is 0.707, so the top-1-minus-top-2
        // margin is ~0 -> escalate. An absolute 0.5 floor on the raw 0.707
        // score would have wrongly admitted it.
        let generic = route(&[1.0, 1.0], &experts, &[], 1, &cfg);
        assert!(generic.escalate);
        assert!(generic.coverage < 0.1);
    }

    #[test]
    fn escalates_with_an_empty_population() {
        let decision = route(&[1.0, 0.0], &[], &[], 1, &GateConfig::default());
        assert!(decision.escalate);
        assert!(decision.chosen.is_empty());
        assert!(decision.ranked.is_empty());
    }

    #[test]
    fn selects_top_k_best_first_when_in_scope() {
        let experts = vec![
            expert("a", vec![1.0, 0.0, 0.0]),
            expert("b", vec![0.0, 1.0, 0.0]),
            expert("c", vec![0.0, 0.0, 1.0]),
        ];
        let decision = route(&[1.0, 0.0, 0.0], &experts, &[], 2, &GateConfig::default());
        assert!(!decision.escalate);
        assert_eq!(decision.chosen.len(), 2);
        assert_eq!(decision.chosen[0], ExpertId::new("a"));
    }
}
