//! What the operator console (`antumbra tui`) shows and does, in the shape the
//! server sends it: one [`Snapshot`] per read, and the [`Action`]s an operator
//! may take.
//!
//! A snapshot has two parts. The workspace's part (the experts the token can
//! see, its memories and their edges) comes for every token. The substrate's
//! part (shadows, boundaries, the loop, evaluations, the router) comes only for
//! an operator's: none of it belongs to one workspace, and a boundary carries
//! the context of whatever work produced it.

use serde::{Deserialize, Serialize};

use crate::evaluation::EvaluationRun;
use crate::generational::GenerationHead;
use crate::ids::{BoundaryId, ExpertId, RunId, ShadowId};
use crate::memory::{Memory, MemoryEdge};
use crate::router::LearnedRouter;
use crate::usage::{MemoryPolicy, MemoryUse};
use crate::{Expert, FailureBoundary, Shadow};

/// Everything the console shows, read in one go.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Snapshot {
    pub experts: Vec<Expert>,
    /// Memories without their embeddings, grouped by network and strongest
    /// first within each.
    pub memories: Vec<Memory>,
    /// The edges between the memories listed; an edge whose end is gone is
    /// left out.
    pub edges: Vec<MemoryEdge>,
    /// What the system has seen of each memory's use, for those it has seen.
    /// Empty from a server older than 0.2.5.
    #[serde(default)]
    pub usage: Vec<MemoryUse>,
    /// The workspace's knobs, which the strength of every memory is read
    /// with.
    #[serde(default)]
    pub policy: MemoryPolicy,
    pub boundaries: Vec<FailureBoundary>,
    /// Newest first.
    pub shadows: Vec<Shadow>,
    pub loop_heads: Vec<GenerationHead>,
    /// The first loop has a halt requested that it has not yet reached.
    pub loop_halt_pending: bool,
    pub evals: Vec<EvaluationRun>,
    pub router: Option<LearnedRouter>,
    /// The token that read it may take [`Action`]s, and the substrate's part
    /// is filled.
    pub operator: bool,
}

/// An operator's action on the substrate, from the console.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum Action {
    /// Replace the operator's workspace's knobs (clamped into range).
    SetMemoryPolicy {
        policy: MemoryPolicy,
    },
    PruneShadow {
        shadow: ShadowId,
    },
    GraduateShadow {
        shadow: ShadowId,
    },
    FreezeExpert {
        expert: ExpertId,
    },
    ThawExpert {
        expert: ExpertId,
    },
    DeleteBoundary {
        boundary: BoundaryId,
    },
    /// Stop the loop at its next generation.
    HaltLoop {
        run: RunId,
    },
    CancelHalt {
        run: RunId,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_action_reads_as_its_name_and_target() {
        let wire = serde_json::json!({ "action": "freeze_expert", "expert": "expert:a" });
        let action: Action = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(
            action,
            Action::FreezeExpert {
                expert: ExpertId::new("expert:a")
            }
        );
        assert_eq!(serde_json::to_value(&action).unwrap(), wire);
    }

    #[test]
    fn a_snapshot_round_trips() {
        let now = chrono::Utc::now();
        let snapshot = Snapshot {
            memories: vec![crate::Memory::new(
                "memory:1",
                "ws:1",
                crate::MemoryNetwork::World,
                "a fact",
                0.6,
                now,
            )],
            operator: true,
            ..Snapshot::default()
        };
        let back: Snapshot =
            serde_json::from_str(&serde_json::to_string(&snapshot).unwrap()).unwrap();
        assert_eq!(back.memories, snapshot.memories);
        assert!(back.operator);
    }

    #[test]
    fn a_snapshot_from_an_older_server_reads_with_the_default_knobs() {
        let back: Snapshot = serde_json::from_value(serde_json::json!({
            "experts": [], "memories": [], "edges": [], "boundaries": [], "shadows": [],
            "loop_heads": [], "loop_halt_pending": false, "evals": [], "router": null,
            "operator": false
        }))
        .unwrap();
        assert!(back.usage.is_empty());
        assert_eq!(back.policy, MemoryPolicy::default());
    }

    #[test]
    fn setting_the_knobs_reads_as_its_name_and_policy() {
        let wire = serde_json::json!({
            "action": "set_memory_policy",
            "policy": { "auto": false, "strength_weight": 0.5 }
        });
        let Action::SetMemoryPolicy { policy } = serde_json::from_value(wire).unwrap() else {
            panic!("not a policy action");
        };
        assert!(!policy.auto);
        assert_eq!(policy.strength_weight, 0.5);
        assert_eq!(policy.max_links, MemoryPolicy::default().max_links);
    }
}
