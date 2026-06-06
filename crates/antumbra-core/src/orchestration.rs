//! Durable orchestration runs (ADR-0005) — multi-step composition whose status
//! field IS the checkpoint, so a crash resumes mid-task. The generational loop
//! (ADR-0008) rides the same durable-state engine.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::{AntumbraError, Result};
use crate::ids::{ExpertId, RunId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OrchestrationStatus {
    Routing,
    Executing,
    Scoring,
    Deciding,
    Done,
    Failed,
}

impl OrchestrationStatus {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            OrchestrationStatus::Done | OrchestrationStatus::Failed
        )
    }

    pub fn allowed_next(self) -> &'static [OrchestrationStatus] {
        use OrchestrationStatus::*;
        match self {
            Routing => &[Executing, Failed],
            Executing => &[Scoring, Failed],
            Scoring => &[Deciding, Failed],
            Deciding => &[Routing, Done, Failed],
            Done | Failed => &[],
        }
    }

    pub fn transition(self, to: OrchestrationStatus) -> Result<OrchestrationStatus> {
        if self.allowed_next().contains(&to) {
            Ok(to)
        } else {
            Err(AntumbraError::InvalidTransition {
                entity: "orchestration_run",
                from: format!("{self:?}"),
                to: format!("{to:?}"),
            })
        }
    }
}

/// How the gate composed the chosen experts for this run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ComposeStrategy {
    Parallel,
    Cascade,
    Vote,
    Refine,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestrationRun {
    pub id: RunId,
    pub task_id: String,
    #[serde(default)]
    pub round: u32,
    pub status: OrchestrationStatus,
    #[serde(default)]
    pub chosen_experts: Vec<ExpertId>,
    #[serde(default)]
    pub compose_strategy: Option<ComposeStrategy>,
    pub updated_at: DateTime<Utc>,
}

impl OrchestrationRun {
    pub fn start(id: RunId, task_id: impl Into<String>, now: DateTime<Utc>) -> Self {
        OrchestrationRun {
            id,
            task_id: task_id.into(),
            round: 0,
            status: OrchestrationStatus::Routing,
            chosen_experts: Vec::new(),
            compose_strategy: None,
            updated_at: now,
        }
    }

    pub fn advance_to(&mut self, to: OrchestrationStatus, now: DateTime<Utc>) -> Result<()> {
        let from = self.status;
        self.status = self.status.transition(to)?;
        // Looping back to route again (Deciding -> Routing) begins a new round, so
        // `round` actually counts the compose/refine iterations it is meant to
        // (it was previously fixed at 0 -- the loop edge never advanced it).
        if from == OrchestrationStatus::Deciding && to == OrchestrationStatus::Routing {
            self.round += 1;
        }
        self.updated_at = now;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn happy_path_completes() {
        let mut run = OrchestrationRun::start(RunId::new("run:1"), "task:42", Utc::now());
        for to in [
            OrchestrationStatus::Executing,
            OrchestrationStatus::Scoring,
            OrchestrationStatus::Deciding,
            OrchestrationStatus::Done,
        ] {
            run.advance_to(to, Utc::now()).unwrap();
        }
        assert!(run.status.is_terminal());
    }

    #[test]
    fn deciding_can_loop_for_another_round() {
        assert!(OrchestrationStatus::Deciding
            .transition(OrchestrationStatus::Routing)
            .is_ok());
    }

    #[test]
    fn looping_back_to_routing_advances_the_round() {
        let mut run = OrchestrationRun::start(RunId::new("run:1"), "task:1", Utc::now());
        assert_eq!(run.round, 0);
        for to in [
            OrchestrationStatus::Executing,
            OrchestrationStatus::Scoring,
            OrchestrationStatus::Deciding,
            OrchestrationStatus::Routing, // loop back: round 1
        ] {
            run.advance_to(to, Utc::now()).unwrap();
        }
        assert_eq!(run.round, 1, "the Deciding -> Routing loop begins a new round");
        // A second lap bumps it again; non-loop transitions leave it alone.
        for to in [
            OrchestrationStatus::Executing,
            OrchestrationStatus::Scoring,
            OrchestrationStatus::Deciding,
            OrchestrationStatus::Routing,
        ] {
            run.advance_to(to, Utc::now()).unwrap();
        }
        assert_eq!(run.round, 2);
    }

    #[test]
    fn cannot_skip_states() {
        assert!(OrchestrationStatus::Routing
            .transition(OrchestrationStatus::Done)
            .is_err());
    }
}
