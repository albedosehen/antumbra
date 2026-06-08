//! The durable generational loop as a resumable state machine.
//!
//! `grow -> explore -> score -> decide -> consolidate -> grow | paused`.
//! The state is persisted; killing the process mid-run and restarting resumes
//! from the last stored [`LoopState`], because the state value *is* the
//! checkpoint.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::{AntumbraError, Result};
use crate::ids::{Generation, RunId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LoopState {
    /// Decide where the population is weak and spawn shadows there.
    Grow,
    /// Shadows train (DIY candle QLoRA via the trainer port).
    Explore,
    /// Verifiers + critic produce reward signals.
    Score,
    /// Winners graduate into experts; losers are pruned and their boundary logged.
    Decide,
    /// Update fitness, merge/decay the stores, reindex, checkpoint.
    Consolidate,
    /// Fully resumable rest state.
    Paused,
}

impl LoopState {
    /// The canonical forward step for each state (the solid edges of the
    /// durable generational loop). `Consolidate` advances to the next generation's
    /// `Grow`; `Paused` resumes into `Grow`.
    pub fn forward(self) -> LoopState {
        use LoopState::*;
        match self {
            Grow => Explore,
            Explore => Score,
            Score => Decide,
            Decide => Consolidate,
            Consolidate => Grow,
            Paused => Grow,
        }
    }

    pub fn allowed_next(self) -> &'static [LoopState] {
        use LoopState::*;
        match self {
            Grow => &[Explore, Paused],
            Explore => &[Score, Paused],
            Score => &[Decide, Paused],
            Decide => &[Consolidate, Paused],
            Consolidate => &[Grow, Paused],
            Paused => &[Grow],
        }
    }

    pub fn transition(self, to: LoopState) -> Result<LoopState> {
        if self.allowed_next().contains(&to) {
            Ok(to)
        } else {
            Err(AntumbraError::InvalidTransition {
                entity: "generation",
                from: format!("{self:?}"),
                to: format!("{to:?}"),
            })
        }
    }

    /// Whether moving from `Consolidate` to `Grow` crosses a generation
    /// boundary (used to bump the generation counter exactly once per cycle).
    pub fn crosses_generation(self, to: LoopState) -> bool {
        self == LoopState::Consolidate && to == LoopState::Grow
    }
}

/// The persisted head of the loop. One row; updating it in a transaction is the
/// checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationHead {
    pub run_id: RunId,
    pub generation: Generation,
    pub state: LoopState,
    pub updated_at: DateTime<Utc>,
}

impl GenerationHead {
    pub fn new(run_id: RunId, now: DateTime<Utc>) -> Self {
        GenerationHead {
            run_id,
            generation: Generation::ZERO,
            state: LoopState::Grow,
            updated_at: now,
        }
    }

    /// Guarded advance that bumps the generation when the cycle wraps.
    pub fn advance_to(&mut self, to: LoopState, now: DateTime<Utc>) -> Result<()> {
        if self.state.crosses_generation(to) {
            self.generation = self.generation.next();
        }
        self.state = self.state.transition(to)?;
        self.updated_at = now;
        Ok(())
    }

    pub fn pause(&mut self, now: DateTime<Utc>) -> Result<()> {
        self.advance_to(LoopState::Paused, now)
    }
}

/// An out-of-band control signal for a running loop, written by an operator and
/// polled by the runner at each generation boundary. Absent = [`LoopCommand::Run`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LoopCommand {
    /// Keep running (the default when no control row is set).
    #[default]
    Run,
    /// Halt gracefully at the next generation boundary: checkpoint the head as
    /// `Paused` and exit. Re-running the loop resumes from that checkpoint.
    Halt,
}

impl LoopCommand {
    pub fn as_str(self) -> &'static str {
        match self {
            LoopCommand::Run => "run",
            LoopCommand::Halt => "halt",
        }
    }
}

/// The persisted control row for a run (one per run id). Writing it is how an
/// operator cooperatively stops a running loop without killing the process; the
/// runner consumes it at the next generation boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopControl {
    pub run_id: RunId,
    pub command: LoopCommand,
    pub updated_at: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_full_cycle_bumps_generation_once() {
        let mut head = GenerationHead::new(RunId::new("run:loop"), Utc::now());
        assert_eq!(head.generation, Generation::ZERO);
        let mut s = head.state;
        for _ in 0..5 {
            let to = s.forward();
            head.advance_to(to, Utc::now()).unwrap();
            s = to;
        }
        // grow->explore->score->decide->consolidate->grow == one generation
        assert_eq!(head.state, LoopState::Grow);
        assert_eq!(head.generation, Generation(1));
    }

    #[test]
    fn pause_and_resume_is_lossless() {
        let mut head = GenerationHead::new(RunId::new("run:loop"), Utc::now());
        head.advance_to(LoopState::Explore, Utc::now()).unwrap();
        head.pause(Utc::now()).unwrap();
        assert_eq!(head.state, LoopState::Paused);
        // resume
        head.advance_to(LoopState::Grow, Utc::now()).unwrap();
        assert_eq!(head.state, LoopState::Grow);
        // pausing did not spuriously bump the generation
        assert_eq!(head.generation, Generation::ZERO);
    }

    #[test]
    fn illegal_jumps_rejected() {
        assert!(LoopState::Grow.transition(LoopState::Decide).is_err());
    }
}
