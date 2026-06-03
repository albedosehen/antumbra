//! # antumbra-gate — ADR-0005 (v0 heuristic)
//!
//! A learned, boundary-conditioned latent mixer is the north-star gate. v0
//! ships the honest fallback that ADR-0005 names "coverage routing (still
//! useful)": score each expert by capability similarity to the task, subtract
//! the in-scope inhibition any boundary imposes (ADR-0004), take the top-k —
//! and **escalate** when nothing clears the in-scope bar. The learned mixer
//! plugs in behind the same `route` signature later.

use antumbra_core::{Expert, ExpertId, FailureBoundary};

#[derive(Debug, Clone, Copy)]
pub struct GateConfig {
    /// Minimum score for an expert to count as in-scope; below this the gate
    /// escalates rather than guessing.
    pub in_scope_threshold: f32,
    /// Cosine-similarity radius inside which a boundary's penalty applies.
    pub inhibition_radius: f32,
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            in_scope_threshold: 0.0,
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
}

/// Score and select experts for a task embedding.
///
/// `score(e) = cosine(e.capability_vec, task) - max_boundary_inhibition(task)`.
/// Experts without a capability vector are skipped (not yet routable).
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

    let mut ranked: Vec<ScoredExpert> = experts
        .iter()
        .filter_map(|e| {
            e.capability_similarity(task_vec).map(|sim| ScoredExpert {
                id: e.id.clone(),
                score: sim - inhibition,
            })
        })
        .collect();
    ranked.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let best = ranked.first().map_or(f32::MIN, |s| s.score);
    let escalate = ranked.is_empty() || best < cfg.in_scope_threshold;
    let chosen = if escalate {
        Vec::new()
    } else {
        ranked.iter().take(k).map(|s| s.id.clone()).collect()
    };

    GateDecision {
        chosen,
        escalate,
        ranked,
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
            in_scope_threshold: 0.95,
            ..GateConfig::default()
        };
        // task is fairly off-axis from the only expert -> below the high bar.
        let decision = route(&[0.3, 0.7], &experts, &[], 1, &cfg);
        assert!(decision.escalate);
        assert!(decision.chosen.is_empty());
    }

    #[test]
    fn boundary_inhibition_can_force_escalation() {
        let experts = vec![expert("a", vec![1.0, 0.0])];
        // Without a boundary the expert is a perfect match (score 1.0).
        let cfg = GateConfig {
            in_scope_threshold: 0.1,
            ..GateConfig::default()
        };
        let clear = route(&[1.0, 0.0], &experts, &[], 1, &cfg);
        assert!(!clear.escalate);

        // A confident boundary right on the task cancels the similarity
        // (1.0 - 1.0 = 0.0), dropping below the in-scope bar -> escalate.
        let bounds = vec![boundary(vec![1.0, 0.0], 1.0)];
        let decision = route(&[1.0, 0.0], &experts, &bounds, 1, &cfg);
        assert!(decision.escalate);
    }
}
