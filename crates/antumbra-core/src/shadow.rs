//! The trainable penumbra: short-lived shadows that explore around the frozen
//! core and then graduate (deepen into umbra) or prune. ADR-0002.
//!
//! The status field is a real state machine; [`ShadowStatus::transition`] is the
//! single guarded entry point, so illegal moves (e.g. resurrecting a pruned
//! shadow) are unrepresentable rather than merely discouraged.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::{AntumbraError, Result};
use crate::ids::{ExpertId, Generation, ShadowId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ShadowStatus {
    /// Allocated, adapter not yet attached.
    Spawning,
    /// Adapter attached; acting on the corpus.
    Exploring,
    /// Environment verifies + critic densifies (ADR-0003).
    Scoring,
    /// Fitness cleared threshold; will refreeze into an expert (ADR-0001).
    Graduated,
    /// Stalled or collapsed; discarded, boundary logged (ADR-0004).
    Pruned,
}

impl ShadowStatus {
    /// Terminal states never transition again.
    pub fn is_terminal(self) -> bool {
        matches!(self, ShadowStatus::Graduated | ShadowStatus::Pruned)
    }

    /// Lowercase wire form (matches the serde representation), for building
    /// typed store filters without re-serializing.
    pub fn as_str(self) -> &'static str {
        match self {
            ShadowStatus::Spawning => "spawning",
            ShadowStatus::Exploring => "exploring",
            ShadowStatus::Scoring => "scoring",
            ShadowStatus::Graduated => "graduated",
            ShadowStatus::Pruned => "pruned",
        }
    }

    /// The states reachable in one step from `self`, mirroring the ADR-0002
    /// lifecycle diagram exactly.
    pub fn allowed_next(self) -> &'static [ShadowStatus] {
        use ShadowStatus::*;
        match self {
            Spawning => &[Exploring, Pruned],
            // scoring loops back to exploring to keep training on verified
            // outcomes, or settles to a terminal state.
            Exploring => &[Scoring, Pruned],
            Scoring => &[Exploring, Graduated, Pruned],
            Graduated | Pruned => &[],
        }
    }

    /// Guarded transition: the only sanctioned way to change a shadow's status.
    pub fn transition(self, to: ShadowStatus) -> Result<ShadowStatus> {
        if self.allowed_next().contains(&to) {
            Ok(to)
        } else {
            Err(AntumbraError::InvalidTransition {
                entity: "shadow",
                from: format!("{self:?}"),
                to: format!("{to:?}"),
            })
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Shadow {
    pub id: ShadowId,
    /// The expert this shadow explores around, if it descends from one.
    #[serde(default)]
    pub parent_expert: Option<ExpertId>,
    /// LoRA checkpoint under training; `None` until attached.
    #[serde(default)]
    pub adapter_uri: Option<String>,
    pub status: ShadowStatus,
    pub generation: Generation,
    /// Per-step fitness history; the anti-collapse signal lives here (ADR-0002).
    #[serde(default)]
    pub reward_curve: Vec<f32>,
    pub created_at: DateTime<Utc>,
}

impl Shadow {
    pub fn spawn(
        id: ShadowId,
        generation: Generation,
        parent: Option<ExpertId>,
        now: DateTime<Utc>,
    ) -> Self {
        Shadow {
            id,
            parent_expert: parent,
            adapter_uri: None,
            status: ShadowStatus::Spawning,
            generation,
            reward_curve: Vec::new(),
            created_at: now,
        }
    }

    /// Apply a guarded status change in place.
    pub fn advance_to(&mut self, to: ShadowStatus) -> Result<()> {
        self.status = self.status.transition(to)?;
        Ok(())
    }

    /// Latest fitness reading, or `0.0` before any scoring.
    pub fn current_fitness(&self) -> f32 {
        self.reward_curve.last().copied().unwrap_or(0.0)
    }

    /// Collapse guard (ADR-0002): a shadow whose reward never rises above the
    /// floor has degenerated to empty/trivial output and should be pruned.
    pub fn has_collapsed(&self, floor: f32) -> bool {
        !self.reward_curve.is_empty() && self.reward_curve.iter().all(|&r| r <= floor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lifecycle_follows_adr0002() {
        let s = ShadowStatus::Spawning;
        let s = s.transition(ShadowStatus::Exploring).unwrap();
        let s = s.transition(ShadowStatus::Scoring).unwrap();
        // can loop back to keep training
        let s2 = s.transition(ShadowStatus::Exploring).unwrap();
        assert_eq!(s2, ShadowStatus::Exploring);
        // or graduate
        let g = s.transition(ShadowStatus::Graduated).unwrap();
        assert!(g.is_terminal());
    }

    #[test]
    fn illegal_transitions_are_rejected() {
        assert!(ShadowStatus::Spawning
            .transition(ShadowStatus::Graduated)
            .is_err());
        assert!(ShadowStatus::Pruned
            .transition(ShadowStatus::Exploring)
            .is_err());
        assert!(ShadowStatus::Graduated
            .transition(ShadowStatus::Scoring)
            .is_err());
    }

    #[test]
    fn collapse_detection() {
        let mut s = Shadow::spawn(
            ShadowId::new("shadow:1"),
            Generation::ZERO,
            None,
            Utc::now(),
        );
        assert!(!s.has_collapsed(0.01));
        s.reward_curve = vec![0.0, 0.0, 0.0];
        assert!(s.has_collapsed(0.01));
        s.reward_curve = vec![0.0, 0.2, 0.5];
        assert!(!s.has_collapsed(0.01));
        assert_eq!(s.current_fitness(), 0.5);
    }
}
