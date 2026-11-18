//! A standing expert's training runs.
//!
//! The server trains a standing expert again whenever its behaviors change,
//! and records each training as a candidate. One expert therefore has several
//! candidates, all of them graduated, and only the newest of those is the
//! adapter it answers with: the earlier ones were replaced. This finds the
//! runs that belong to one expert, so a list can tell them apart and a chart
//! can show how the expert has done from one training to the next.

use antumbra_core::{behavior, Shadow, ShadowId, ShadowStatus};

use super::App;

/// What the runs of one standing expert share in their ids: everything before
/// the last part, which names the set of behaviors that run trained. `None`
/// for a candidate that is not a standing expert's, such as the loop's.
fn expert_of(id: &ShadowId) -> Option<&str> {
    behavior::shadow_scope(id)?;
    id.as_str().rsplit_once(':').map(|(expert, _set)| expert)
}

impl App {
    /// Every run of the standing expert that `s` is a run of, oldest first,
    /// `s` among them. Empty when `s` is not a standing expert's run.
    pub fn runs_with(&self, s: &Shadow) -> Vec<&Shadow> {
        let Some(expert) = expert_of(&s.id) else {
            return Vec::new();
        };
        let mut runs: Vec<&Shadow> = self
            .shadows
            .iter()
            .filter(|run| expert_of(&run.id) == Some(expert))
            .collect();
        runs.sort_by_key(|run| run.created_at);
        runs
    }

    /// The run that took `s`'s place: the newest graduated run of the same
    /// standing expert, when `s` graduated before it. A run that did not
    /// graduate replaced nothing, and the newest that did is the one in use.
    pub fn replaced_by(&self, s: &Shadow) -> Option<&Shadow> {
        if s.status != ShadowStatus::Graduated {
            return None;
        }
        let expert = expert_of(&s.id)?;
        self.shadows
            .iter()
            .filter(|run| {
                run.status == ShadowStatus::Graduated
                    && run.created_at > s.created_at
                    && expert_of(&run.id) == Some(expert)
            })
            .max_by_key(|run| run.created_at)
    }
}
