//! Where the partition becomes real: which of a corpus's tasks a run learns
//! from, and which it only measures (ADR-0022).
//!
//! The instruments read a generation through a partition of its tasks. That
//! reading means something only if the run respected the partition: a held-out
//! task that was trained on is not held out, and an audit task that fed the
//! fitness graduation reads has been touched by a decision. So the split happens
//! here, before the first sample, and each training path receives the two sets
//! separately rather than one list to be labeled afterwards.

use antumbra_core::slice::Holdout;
use antumbra_core::{AntumbraError, Result};

use crate::model::CorpusTask;

/// The tasks a run learns from, and the tasks it measures without learning
/// from.
#[derive(Debug, Clone, Default)]
pub struct Split {
    /// Learned from, and the only tasks the run's fitness is computed over.
    pub learn: Vec<CorpusTask>,
    /// Measured in the final round against the same adapter, never learned
    /// from and never counted in fitness: the held-out slice, and the audit
    /// slice when it is due.
    pub withheld: Vec<CorpusTask>,
}

/// Split a corpus under a holdout. With none, every task is learned from and
/// nothing is withheld, which is what every training path did before the
/// partition existed.
///
/// Refuses a holdout that leaves nothing to learn from. A small corpus can hash
/// entirely into the withheld slices -- both tasks of the shipped arithmetic
/// corpus do under the default partition -- and a run over no tasks would save
/// an untrained adapter and report it as a generation.
pub fn split(tasks: Vec<CorpusTask>, holdout: Option<&Holdout>) -> Result<Split> {
    // An impossible task can only be passed by a shortcut, so a winner of one
    // is a shortcut, and training on it would teach the shortcut. It is never
    // learned from, whether or not anything else is held out.
    let (impossible, tasks): (Vec<CorpusTask>, Vec<CorpusTask>) =
        tasks.into_iter().partition(|t| t.impossible);
    let Some(holdout) = holdout else {
        return Ok(Split {
            learn: tasks,
            withheld: Vec::new(),
        });
    };
    let total = tasks.len();
    let (learn, rest): (Vec<CorpusTask>, Vec<CorpusTask>) =
        tasks.into_iter().partition(|t| holdout.learns_from(&t.id));
    if learn.is_empty() && total > 0 {
        return Err(AntumbraError::other(format!(
            "the partition (seed {}) leaves none of the {total} task(s) to learn from; \
             a corpus this small cannot hold anything out, so train it without a holdout",
            holdout.partition.seed
        )));
    }
    // Impossible tasks are measured on every run, not on the audit schedule: a
    // single pass fails the generation, so the alarm cannot wait k generations.
    let withheld = rest
        .into_iter()
        .filter(|t| holdout.measures(&t.id))
        .chain(impossible)
        .collect();
    Ok(Split { learn, withheld })
}

#[cfg(test)]
mod tests {
    use super::*;
    use antumbra_core::slice::Partition;

    fn task(id: &str) -> CorpusTask {
        CorpusTask::new(id, format!("prompt for {id}"))
    }

    fn ids(tasks: &[CorpusTask]) -> Vec<&str> {
        tasks.iter().map(|t| t.id.as_str()).collect()
    }

    fn holdout(audit: bool) -> Holdout {
        Holdout {
            partition: Partition::default(),
            audit,
        }
    }

    #[test]
    fn with_no_holdout_everything_is_learned_and_nothing_withheld() -> Result<()> {
        let s = split(vec![task("task:0"), task("task:1"), task("task:19")], None)?;
        assert_eq!(ids(&s.learn), ["task:0", "task:1", "task:19"]);
        assert!(s.withheld.is_empty());
        Ok(())
    }

    #[test]
    fn a_withheld_task_is_never_in_the_learn_set() -> Result<()> {
        // task:0 is visible, task:1 held out, task:19 audited (pinned in core).
        let corpus = || vec![task("task:0"), task("task:1"), task("task:19")];
        let due = split(corpus(), Some(&holdout(true)))?;
        assert_eq!(ids(&due.learn), ["task:0"]);
        assert_eq!(ids(&due.withheld), ["task:1", "task:19"]);
        // Off schedule the audit task is neither learned nor measured: not
        // measuring it is how the audit slice stays read every k generations.
        let not_due = split(corpus(), Some(&holdout(false)))?;
        assert_eq!(ids(&not_due.learn), ["task:0"]);
        assert_eq!(ids(&not_due.withheld), ["task:1"]);
        Ok(())
    }

    #[test]
    fn an_impossible_task_is_never_learned_and_always_measured() -> Result<()> {
        let corpus = || vec![task("task:0"), task("task:0-unsatisfiable").impossible()];
        // No holdout: nothing is measured apart, and the impossible task is
        // still not learned from.
        let bare = split(corpus(), None)?;
        assert_eq!(ids(&bare.learn), ["task:0"]);
        assert!(bare.withheld.is_empty());
        // Under a holdout it is measured whether or not the audit is due.
        for audit in [true, false] {
            let held = split(corpus(), Some(&holdout(audit)))?;
            assert_eq!(ids(&held.learn), ["task:0"]);
            assert_eq!(ids(&held.withheld), ["task:0-unsatisfiable"]);
        }
        Ok(())
    }

    #[test]
    fn impossible_tasks_do_not_count_as_something_to_learn() {
        // Held-out tasks plus impossible ones, and nothing visible: still refused.
        let refused = split(
            vec![task("add"), task("multiply"), task("x").impossible()],
            Some(&holdout(true)),
        );
        assert!(refused.is_err());
    }

    #[test]
    fn a_corpus_that_hashes_entirely_out_of_view_is_refused() {
        // The shipped arithmetic corpus: both ids fall in the held-out slice.
        let refused = split(vec![task("add"), task("multiply")], Some(&holdout(true)));
        assert!(
            refused.is_err(),
            "training on nothing must not look like a run"
        );
        // An empty corpus is not refused: there was nothing to withhold.
        assert!(split(Vec::new(), Some(&holdout(true))).is_ok());
    }
}
