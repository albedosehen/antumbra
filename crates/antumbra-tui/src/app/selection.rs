//! What is selected, kept through a read of the server.
//!
//! Each list remembers its selection as a row. A read replaces the lists, and
//! the rows need not come back as they were: the experts arrive in the
//! store's order and are sorted here, the memories are ranked by a strength
//! that moves with time and use, and a new candidate goes in above the rest.
//! A row number carried across a read would then name something else, and the
//! selection would be seen to jump every few seconds. So before a read is
//! applied the selection is noted by what it is ([`Kept`]), and afterwards
//! each list's row is found again.
//!
//! A child module of `app`, so these are the same inherent methods on the same
//! `App`.

use chrono::{DateTime, Utc};

use super::App;

/// What is selected in each list, named by what it is and not by its row.
pub(super) struct Kept {
    expert: Option<String>,
    candidate: Option<String>,
    boundary: Option<String>,
    memory: Option<String>,
    /// An evaluation is one run's check of one subject against one task, at
    /// one time.
    eval: Option<(String, String, String, DateTime<Utc>)>,
    run: Option<String>,
    /// A skill is counted per workspace.
    skill: Option<(String, String)>,
}

/// The row of the first of `items` that `is` the one wanted; where it is
/// gone, `row` brought within the list, so the selection stays near where it
/// was.
fn found<T>(items: &[T], row: usize, is: impl Fn(&T) -> bool) -> usize {
    items
        .iter()
        .position(is)
        .unwrap_or_else(|| row.min(items.len().saturating_sub(1)))
}

impl App {
    /// Note what is selected in every list, before a read replaces them.
    pub(super) fn kept(&self) -> Kept {
        Kept {
            expert: self.selected_expert().map(|e| e.id.as_str().to_string()),
            candidate: self.selected_shadow().map(|s| s.id.as_str().to_string()),
            boundary: self.selected_boundary().map(|b| b.id.as_str().to_string()),
            memory: self.selected_memory().map(|m| m.id.as_str().to_string()),
            eval: self.evals.get(self.selected_eval).map(|e| {
                (
                    e.run_id.as_str().to_string(),
                    e.subject_id.clone(),
                    e.corpus_task_id.clone(),
                    e.created_at,
                )
            }),
            run: self
                .selected_loop_head()
                .map(|h| h.run_id.as_str().to_string()),
            skill: self
                .sovereign
                .skills
                .get(self.selected_skill)
                .map(|s| (s.skill.clone(), s.workspace.clone())),
        }
    }

    /// Put every list's selection back on what it was on, wherever a read
    /// has moved it. A selection whose item is gone keeps its row, brought
    /// within the list.
    pub(super) fn keep(&mut self, kept: Kept) {
        self.selected = found(&self.experts, self.selected, |e| {
            Some(e.id.as_str()) == kept.expert.as_deref()
        });
        self.selected_shadow = found(&self.shadows, self.selected_shadow, |s| {
            Some(s.id.as_str()) == kept.candidate.as_deref()
        });
        self.selected_boundary = found(&self.boundaries, self.selected_boundary, |b| {
            Some(b.id.as_str()) == kept.boundary.as_deref()
        });
        self.selected_memory = found(&self.memories, self.selected_memory, |m| {
            Some(m.id.as_str()) == kept.memory.as_deref()
        });
        self.selected_eval = found(&self.evals, self.selected_eval, |e| {
            kept.eval.as_ref().is_some_and(|(run, subject, task, at)| {
                e.run_id.as_str() == run
                    && &e.subject_id == subject
                    && &e.corpus_task_id == task
                    && e.created_at == *at
            })
        });
        self.selected_loop = found(&self.loop_heads, self.selected_loop, |h| {
            Some(h.run_id.as_str()) == kept.run.as_deref()
        });
        self.selected_skill = found(&self.sovereign.skills, self.selected_skill, |s| {
            kept.skill
                .as_ref()
                .is_some_and(|(skill, workspace)| &s.skill == skill && &s.workspace == workspace)
        });
    }
}
