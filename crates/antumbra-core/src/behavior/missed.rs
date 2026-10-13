//! What the expert answered instead: for a behavior its last training did
//! not learn, one held-out task and the trained expert's answer to it, which
//! failed the behavior's check.
//!
//! The training note says how far short a behavior fell; this says how. An
//! answer that does the right thing in words the check does not expect points
//! at the check, one that does something else entirely points at the
//! examples, and one that does nothing in particular says the rule needs more
//! of them. Kept beside the note, replaced at each training, and cleared when
//! a training learns the behavior.

use serde::{Deserialize, Serialize};

const MISSED: &str = "behavior-missed:";

/// A task or an answer is kept to this many characters: enough to see what
/// the expert did, without a second copy of a long completion in the
/// behavior.
pub const MISSED_CHARS: usize = 400;

/// One held-out task a behavior was missed on, and the expert's answer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Missed {
    pub task: String,
    pub answer: String,
}

impl Missed {
    /// `task` and the expert's `answer`, each cut to [`MISSED_CHARS`].
    pub fn new(task: &str, answer: &str) -> Self {
        Self {
            task: clip(task.trim()),
            answer: clip(answer.trim()),
        }
    }
}

fn clip(text: &str) -> String {
    match text.char_indices().nth(MISSED_CHARS) {
        Some((end, _)) => format!("{}…", &text[..end]),
        None => text.to_string(),
    }
}

/// Note the task and answer `missed` was missed on, replacing the note
/// before; `None` clears it, for a behavior its last training learned.
pub fn mark(evidence: &mut Vec<String>, missed: Option<&Missed>) {
    evidence.retain(|e| !e.starts_with(MISSED));
    if let Some(json) = missed.and_then(|m| serde_json::to_string(m).ok()) {
        evidence.push(format!("{MISSED}{json}"));
    }
}

/// The task and answer a behavior was last missed on, if its last training
/// missed it.
pub fn of(evidence: &[String]) -> Option<Missed> {
    evidence
        .iter()
        .rev()
        .find_map(|e| e.strip_prefix(MISSED))
        .and_then(|json| serde_json::from_str(json).ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_miss_is_noted_and_read_back() {
        let mut evidence = vec!["behavior-status:accepted".to_string()];
        let m = Missed::new("Push the branch.", "git push --force");
        mark(&mut evidence, Some(&m));
        assert_eq!(of(&evidence), Some(m));
        assert_eq!(evidence[0], "behavior-status:accepted", "the rest is kept");
    }

    #[test]
    fn a_new_miss_replaces_the_last_and_a_learned_training_clears_it() {
        let mut evidence = Vec::new();
        mark(&mut evidence, Some(&Missed::new("a", "first")));
        mark(&mut evidence, Some(&Missed::new("b", "second")));
        assert_eq!(evidence.len(), 1);
        assert_eq!(of(&evidence).map(|m| m.answer), Some("second".into()));
        mark(&mut evidence, None);
        assert!(evidence.is_empty());
        assert_eq!(of(&evidence), None);
    }

    /// An answer that runs on is cut at a character, never inside one, and
    /// says it was cut; a short one is kept as it was, trimmed.
    #[test]
    fn a_long_answer_is_cut_at_a_character_boundary() {
        let long = "é".repeat(MISSED_CHARS + 50);
        let m = Missed::new("t", &long);
        assert_eq!(m.answer.chars().count(), MISSED_CHARS + 1);
        assert!(m.answer.ends_with('…'));
        assert_eq!(Missed::new(" t ", "  short \n").answer, "short");
        assert_eq!(Missed::new(" t ", "x").task, "t");
    }

    /// An answer's newlines and quotes stay inside the one evidence line.
    #[test]
    fn an_answer_with_newlines_and_quotes_stays_one_line() {
        let mut evidence = Vec::new();
        let m = Missed::new("t", "line one\n\"quoted\" line two");
        mark(&mut evidence, Some(&m));
        assert!(!evidence[0].contains('\n'));
        assert_eq!(of(&evidence), Some(m));
    }

    #[test]
    fn evidence_without_a_miss_reads_none() {
        assert_eq!(of(&["behavior-status:trained".to_string()]), None);
        assert_eq!(of(&["behavior-missed:not json".to_string()]), None);
    }
}
