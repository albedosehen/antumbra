//! What the grow step decides on and what it decided: which
//! region of the corpus the next generation learns from.
//!
//! A region is a skill. Its evidence is a census taken with each contribution
//! measurement: how often the routed population is accepted there, by the
//! authored verifiers the grow step cannot influence. The decision keeps every
//! candidate it weighed, so the curriculum is auditable like everything else.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::ids::{ExpertId, Generation, RunId};

/// How the population fares on one region, in one measured generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegionCensus {
    pub run_id: RunId,
    pub generation: Generation,
    pub region: String,
    /// Live tasks of the region the population was scored on.
    pub tasks: u32,
    /// The routed population's mean pass rate there, the base model standing
    /// in where the gate escalates: the region's acceptability.
    pub acceptability: f32,
    /// The mean embedding of the region's tasks, for how alike two regions
    /// are.
    pub centroid: Vec<f32>,
    pub at: DateTime<Utc>,
}

/// One region as the grow step weighed it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegionCandidate {
    pub region: String,
    pub acceptability: f32,
    /// Whether it passed the gate: the population is non-trivially above zero
    /// there, so it is neither impossible nor unreachable.
    pub admitted: bool,
    /// `p(1-p)` on its acceptability: highest where the population succeeds
    /// about half the time.
    pub learnability: f32,
    /// How much of its learnability is given up for being like the regions
    /// chosen recently.
    pub penalty: f32,
    pub score: f32,
}

/// What the grow step decided for one generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GrowRecord {
    pub run_id: RunId,
    pub generation: Generation,
    /// The census it decided on. `None` when there was none yet, and the
    /// generation learned from every visible task.
    #[serde(default)]
    pub census_generation: Option<Generation>,
    /// The region chosen. `None` when no region passed the gate.
    #[serde(default)]
    pub chosen: Option<String>,
    #[serde(default)]
    pub candidates: Vec<RegionCandidate>,
    /// Tasks the shadow learns from: the chosen region's, and those sampled
    /// from the whole visible slice.
    pub focus: u32,
    /// Of those, the ones sampled unfiltered.
    pub unfiltered: u32,
    /// What the previous choice realized: its region's acceptability now,
    /// less what it was when chosen. The credit the policy learns from.
    #[serde(default)]
    pub credit: Option<f32>,
    /// The expert the shadow started from: the one serving the chosen region.
    /// `None` when it started fresh.
    #[serde(default)]
    pub warm_from: Option<ExpertId>,
    pub at: DateTime<Utc>,
}
