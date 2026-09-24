//! What one expert adds to the population it routes in (ADR-0022 S-5): its
//! leave-one-out contribution, measured in one generation.
//!
//! Mask the expert, route the live tasks again, and score both ways: the
//! difference is what the expert is worth there. Routing share alone cannot
//! tell an expert that is unused from one that is useless, and only the
//! second is a candidate for demotion, so the record keeps both: how many
//! tasks came to the expert, and, of those, how much better it did than
//! whatever the gate would have picked without it.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::ids::{ExpertId, Generation, RunId};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContributionRecord {
    pub expert: ExpertId,
    pub run_id: RunId,
    pub generation: Generation,
    /// Live tasks the whole population routed to it: its routing share.
    pub routed: u32,
    /// Live tasks measured, across the population.
    pub tasks: u32,
    /// Its mean score on the tasks routed to it. `None` when none were.
    #[serde(default)]
    pub with: Option<f32>,
    /// The population's mean score on the same tasks with it masked: the next
    /// expert the gate picks, or the base model when none covers a task.
    #[serde(default)]
    pub without: Option<f32>,
    /// Evaluations per task on each side, under the same seeds, so the two
    /// means are a paired comparison.
    pub seeds: u32,
    /// How close the inputs routed to it sit to its capability vector: their
    /// mean cosine similarity. A label-free reading of drift in what it is
    /// asked to do. `None` when none were routed to it.
    #[serde(default)]
    pub affinity: Option<f32>,
    pub at: DateTime<Utc>,
}

impl ContributionRecord {
    /// What the expert added on the tasks routed to it. `None` when it was not
    /// routed to: unused, which is not the same as useless.
    pub fn delta(&self) -> Option<f32> {
        Some(self.with? - self.without?)
    }

    pub fn is_unused(&self) -> bool {
        self.routed == 0
    }

    /// Its share of the live tasks.
    pub fn share(&self) -> f32 {
        self.routed as f32 / self.tasks.max(1) as f32
    }
}

/// The whole population against its single best expert, on the same live
/// tasks under the same seeds (ADR-0022 S-5): the population's score with
/// every task routed as the gate routes it, against the best any one of its
/// experts scores with every task sent to it alone. Kept whether or not it
/// flatters the architecture. If the gap collapses, the honest answer is
/// fewer and broader experts, not a better retirement policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BaselineRecord {
    pub run_id: RunId,
    pub generation: Generation,
    /// Live tasks both sides were scored on.
    pub tasks: u32,
    /// The routed population's mean score, the base model standing in for a
    /// task the gate escalates.
    pub population: f32,
    /// The expert that scored best alone, and its mean score.
    #[serde(default)]
    pub best: Option<ExpertId>,
    #[serde(default)]
    pub best_alone: Option<f32>,
    pub seeds: u32,
    pub at: DateTime<Utc>,
}

impl BaselineRecord {
    /// What routing across the population adds over its best single expert.
    pub fn delta(&self) -> Option<f32> {
        Some(self.population - self.best_alone?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(routed: u32, with: Option<f32>, without: Option<f32>) -> ContributionRecord {
        ContributionRecord {
            expert: ExpertId::new("expert:e"),
            run_id: RunId::new("run"),
            generation: Generation(1),
            routed,
            tasks: 8,
            with,
            without,
            seeds: 2,
            affinity: None,
            at: Utc::now(),
        }
    }

    #[test]
    fn unused_is_not_useless() {
        let unused = record(0, None, None);
        assert!(unused.is_unused() && unused.delta().is_none());
        let useless = record(4, Some(0.5), Some(0.5));
        assert!(!useless.is_unused());
        assert_eq!(useless.delta(), Some(0.0));
        assert_eq!(useless.share(), 0.5);
        assert_eq!(record(2, Some(0.75), Some(0.25)).delta(), Some(0.5));
    }
}
