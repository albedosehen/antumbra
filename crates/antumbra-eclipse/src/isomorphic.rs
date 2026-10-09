//! Isomorphic re-verification: run a graduation candidate again under a
//! transform that changes the task's surface and not its meaning.
//!
//! Renaming, reordering, a consistent substitution of literals. Genuine
//! competence is invariant to all of it. The documented failure this catches is
//! a learner that has enumerated instance-level answers satisfying an
//! extensional check: it passes the task it was measured on and fails its own
//! twin, because what it learned was the instance and not the task.
//!
//! A divergence is disqualifying in both directions. Passing the original and
//! failing the transform is the classic shape, but failing the original and
//! passing the transform is the same fact from the other side: the candidate's
//! answer depends on something the specification does not.

use serde::{Deserialize, Serialize};

/// A task and the same task wearing a different surface.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Pair {
    pub task_id: String,
    /// What the transform did: `rename`, `reorder`, `literals`. Carried so a
    /// divergence names the transform it was found under, which is what makes
    /// it reproducible.
    pub transform: String,
    pub original_passed: bool,
    pub transformed_passed: bool,
}

impl Pair {
    pub fn new(
        task_id: impl Into<String>,
        transform: impl Into<String>,
        original_passed: bool,
        transformed_passed: bool,
    ) -> Self {
        Pair {
            task_id: task_id.into(),
            transform: transform.into(),
            original_passed,
            transformed_passed,
        }
    }

    /// Whether the result survived the transform. Two passes or two failures
    /// are both invariant: a candidate that fails both has not demonstrated
    /// competence, but it has not demonstrated a shortcut either, and this
    /// instrument only measures the second.
    pub fn invariant(&self) -> bool {
        self.original_passed == self.transformed_passed
    }
}

/// What re-verification found.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Reverification {
    pub pairs_checked: u32,
    /// Every pair whose result moved under a transform that did not change the
    /// task. Named, with their transforms, because the point of this
    /// instrument is that the failures are reproducible.
    pub diverged: Vec<Pair>,
}

impl Reverification {
    pub fn of(pairs: &[Pair]) -> Self {
        Reverification {
            pairs_checked: pairs.len() as u32,
            diverged: pairs.iter().filter(|p| !p.invariant()).cloned().collect(),
        }
    }

    /// Whether the candidate may graduate on this evidence.
    ///
    /// A candidate that diverged on any pair may not. Nor may one that was
    /// never re-verified: *every* graduation candidate owes this check, so
    /// an empty run is a re-verification that did not happen rather than one
    /// that found nothing, and reading it as a pass would make the instrument
    /// disappear the moment the caller forgot to run it.
    pub fn clears(&self) -> bool {
        self.pairs_checked > 0 && self.diverged.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_result_that_moves_under_a_rename_disqualifies_the_candidate() {
        let found = Reverification::of(&[
            Pair::new("task:1", "rename", true, true),
            Pair::new("task:2", "rename", true, false),
            Pair::new("task:3", "reorder", true, true),
        ]);
        assert!(!found.clears());
        assert_eq!(
            found
                .diverged
                .iter()
                .map(|p| (p.task_id.as_str(), p.transform.as_str()))
                .collect::<Vec<_>>(),
            vec![("task:2", "rename")],
            "a divergence names the transform it was found under"
        );
    }

    #[test]
    fn a_divergence_counts_in_both_directions() {
        // Failing the original and passing its twin is the same fact from the
        // other side: the answer turns on something the task does not say.
        let found = Reverification::of(&[Pair::new("task:1", "literals", false, true)]);
        assert!(!found.clears());
        assert_eq!(found.diverged.len(), 1);
    }

    #[test]
    fn failing_both_is_incompetence_and_not_a_shortcut() {
        let found = Reverification::of(&[
            Pair::new("task:1", "rename", false, false),
            Pair::new("task:2", "rename", true, true),
        ]);
        assert!(found.clears(), "this instrument measures shortcuts only");
        assert_eq!(found.pairs_checked, 2);
    }

    #[test]
    fn a_re_verification_that_did_not_run_is_not_a_pass() {
        let never = Reverification::of(&[]);
        assert!(
            !never.clears(),
            "an instrument that vanishes when unused is not an instrument"
        );
        assert_eq!(never.pairs_checked, 0);
        assert!(never.diverged.is_empty());
    }
}
