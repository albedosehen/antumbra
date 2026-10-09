//! An expert's place in the population: active, dormant,
//! archived, deleted. Experts stay frozen forever; what moves is
//! whether the gate may route to one, and whether its weights are kept.
//!
//! The governing rule is **staleness demotes, only redundancy deletes.** A
//! stale but unique expert goes dormant and stays recoverable, because drift
//! reverses. Deletion is the one irreversible move, and the one
//! [`TransitionCause`] that permits it is another expert covering this one.
//!
//! A status change is a row of its own, appended and never rewritten, so an
//! expert's record is never touched by its own retirement and the history of
//! every move, with its cause, is the audit trail. An expert with no
//! transitions is active.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::{AntumbraError, Result};
use crate::ids::{ExpertId, Generation, VerifierId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExpertStatus {
    /// In the population the gate routes over.
    Active,
    /// Masked from the gate, weights on disk: still served when named.
    Dormant,
    /// Out of routing and serving, weights kept: revivable.
    Archived,
    /// Gone. The only irreversible state, reached only through redundancy.
    Deleted,
}

impl ExpertStatus {
    /// Lowercase wire form (matches the serde representation).
    pub fn as_str(self) -> &'static str {
        match self {
            ExpertStatus::Active => "active",
            ExpertStatus::Dormant => "dormant",
            ExpertStatus::Archived => "archived",
            ExpertStatus::Deleted => "deleted",
        }
    }

    /// Whether the gate may route to an expert in this state.
    pub fn is_routable(self) -> bool {
        self == ExpertStatus::Active
    }

    /// Whether an expert in this state is served when asked for by name.
    pub fn is_servable(self) -> bool {
        matches!(self, ExpertStatus::Active | ExpertStatus::Dormant)
    }

    /// Whether its weights are still kept, so the byte-identity tripwire
    /// still holds them to their freeze.
    pub fn keeps_weights(self) -> bool {
        self != ExpertStatus::Deleted
    }

    /// The states reachable in one step. Every state but deletion can be
    /// revived to active; archiving is reachable from active so a merge can
    /// archive the experts it replaces in one move.
    pub fn allowed_next(self) -> &'static [ExpertStatus] {
        use ExpertStatus::*;
        match self {
            Active => &[Dormant, Archived],
            Dormant => &[Active, Archived, Deleted],
            Archived => &[Active, Deleted],
            Deleted => &[],
        }
    }

    /// Guarded transition: the only sanctioned way to change an expert's
    /// status. Refuses a move the state machine forbids, and a deletion for
    /// any cause but redundancy.
    pub fn transition(self, to: ExpertStatus, cause: &TransitionCause) -> Result<ExpertStatus> {
        let invalid = || AntumbraError::InvalidTransition {
            entity: "expert",
            from: self.as_str().into(),
            to: to.as_str().into(),
        };
        if !self.allowed_next().contains(&to) {
            return Err(invalid());
        }
        if to == ExpertStatus::Deleted && !matches!(cause, TransitionCause::Redundant { .. }) {
            return Err(AntumbraError::rejected(format!(
                "only redundancy deletes an expert: {} to deleted needs another expert that covers it",
                self.as_str()
            )));
        }
        Ok(to)
    }
}

/// Why an expert changed state: the evidence the record keeps with the move.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum TransitionCause {
    /// A person asked, through the CLI or the console.
    Operator {
        #[serde(default)]
        note: Option<String>,
    },
    /// Another expert covers this one: the only cause that may delete.
    Redundant { of: ExpertId },
    /// The loop confirmed staleness: in each of these consecutive measured
    /// generations, the expert's leave-one-out contribution on the tasks
    /// routed to it was at or below the floor. The evidence it was demoted on.
    Stale {
        generations: Vec<Generation>,
        contributions: Vec<f32>,
    },
    /// A verifier it trained under was quarantined or revoked:
    /// what that verifier taught is kept out of routing, serving and any
    /// training downstream, not merely scored low.
    Quarantined { verifier: VerifierId },
}

/// One change of an expert's status, as recorded.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpertTransition {
    pub expert: ExpertId,
    pub from: ExpertStatus,
    pub to: ExpertStatus,
    pub cause: TransitionCause,
    /// The generation that decided it, when a run did.
    #[serde(default)]
    pub generation: Option<Generation>,
    pub at: DateTime<Utc>,
}

/// The status a history leaves an expert in: the last move's, or active when
/// it has none. `history` is in the order the moves were made.
pub fn current_status(history: &[ExpertTransition]) -> ExpertStatus {
    history.last().map_or(ExpertStatus::Active, |t| t.to)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ExpertStatus::*;

    fn operator() -> TransitionCause {
        TransitionCause::Operator { note: None }
    }

    fn redundant() -> TransitionCause {
        TransitionCause::Redundant {
            of: ExpertId::new("expert:other"),
        }
    }

    #[test]
    fn staleness_demotes_and_every_demotion_but_deletion_revives() {
        assert_eq!(Active.transition(Dormant, &operator()).unwrap(), Dormant);
        assert_eq!(Dormant.transition(Archived, &operator()).unwrap(), Archived);
        assert_eq!(Dormant.transition(Active, &operator()).unwrap(), Active);
        assert_eq!(Archived.transition(Active, &operator()).unwrap(), Active);
        assert_eq!(Active.transition(Archived, &operator()).unwrap(), Archived);
    }

    #[test]
    fn only_redundancy_deletes() {
        let stale = TransitionCause::Stale {
            generations: vec![Generation(2), Generation(4), Generation(6)],
            contributions: vec![0.0, -0.1, 0.0],
        };
        for from in [Dormant, Archived] {
            let refused = from.transition(Deleted, &operator()).unwrap_err();
            assert!(refused.is_rejection(), "{refused}");
            assert!(
                from.transition(Deleted, &stale).is_err(),
                "staleness demotes"
            );
            assert_eq!(from.transition(Deleted, &redundant()).unwrap(), Deleted);
        }
        assert_eq!(Active.transition(Dormant, &stale).unwrap(), Dormant);
    }

    #[test]
    fn an_active_expert_is_demoted_before_it_is_deleted_and_deletion_is_final() {
        assert!(matches!(
            Active.transition(Deleted, &redundant()),
            Err(AntumbraError::InvalidTransition { .. })
        ));
        for to in [Active, Dormant, Archived, Deleted] {
            assert!(Deleted.transition(to, &redundant()).is_err(), "{to:?}");
        }
        assert!(Active.transition(Active, &operator()).is_err());
    }

    #[test]
    fn what_each_state_allows() {
        assert!(Active.is_routable() && Active.is_servable());
        assert!(!Dormant.is_routable() && Dormant.is_servable());
        assert!(!Archived.is_routable() && !Archived.is_servable() && Archived.keeps_weights());
        assert!(!Deleted.is_servable() && !Deleted.keeps_weights());
    }

    #[test]
    fn an_expert_with_no_history_is_active_and_otherwise_where_its_last_move_left_it() {
        assert_eq!(current_status(&[]), Active);
        let moved = |from, to| ExpertTransition {
            expert: ExpertId::new("expert:a"),
            from,
            to,
            cause: operator(),
            generation: None,
            at: Utc::now(),
        };
        assert_eq!(
            current_status(&[moved(Active, Dormant), moved(Dormant, Archived)]),
            Archived
        );
    }

    #[test]
    fn a_transition_round_trips_with_its_cause_tagged() {
        let t = ExpertTransition {
            expert: ExpertId::new("expert:a"),
            from: Dormant,
            to: Deleted,
            cause: redundant(),
            generation: Some(Generation(4)),
            at: Utc::now(),
        };
        let json = serde_json::to_value(&t).unwrap();
        assert_eq!(json["cause"]["kind"], "redundant");
        assert_eq!(json["to"], "deleted");
        let back: ExpertTransition = serde_json::from_value(json).unwrap();
        assert_eq!(back, t);
    }
}
