//! Candidates for behaviors among the memories a store already holds: the
//! ones that may state how to act, picked for the agent to judge.
//!
//! Experts learn behavior and never facts, and a behavior is recorded by the
//! agent that learned it, with a check (`record_behavior`). The rules written
//! down before behaviors were records sit in session notes and corrections,
//! so bringing them in is a pass over the store: this picks what is worth a
//! look, and the agent that reads one decides whether it states a rule a
//! program can check, records it, and cites the memory as a source. A cited
//! memory is not picked again.
//!
//! What is picked: every opinion memory, since judgments, preferences and
//! corrections are filed there and most of them say how to act; and any other
//! memory whose text carries a rule's wording ([`CUES`]). Precision is the
//! agent's: a cue in a fact is read and passed over. What is left out without
//! a look: behaviors and handoffs, which are records of their own; dependency
//! claims; volatile memories, which are counters and state; and memories a
//! behavior already cites.

use std::collections::HashSet;

use chrono::{DateTime, Utc};

use crate::behavior::State;
use crate::depgraph::Edge;
use crate::handoff;
use crate::provenance::GitProvenance;
use crate::{Memory, MemoryNetwork};

/// How much of a memory is shown: enough to judge whether it states a rule.
/// The first measurement of a store judged each memory from this much.
pub const EXCERPT_CHARS: usize = 400;

/// Words and phrases a rule is stated with, matched as whole words in the
/// lowercased text.
pub const CUES: &[&str] = &[
    "never",
    "always",
    "do not",
    "don't",
    "must",
    "should",
    "instead of",
    "rather than",
    "from now on",
    "rule",
    "convention",
    "prefer",
    "correction",
    "corrected",
];

/// Why an opinion memory is picked whatever its wording.
pub const OPINION: &str = "opinion";

/// A memory worth the agent's look, and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub network: MemoryNetwork,
    /// The repository its git anchor names, when it has one: the scope a
    /// behavior drawn from it would most likely have.
    pub repo: Option<String>,
    /// The first [`EXCERPT_CHARS`] characters of its content.
    pub excerpt: String,
    /// Why it was picked: [`OPINION`], and each of [`CUES`] found.
    pub reasons: Vec<&'static str>,
}

/// Whether `text` has `phrase` as a whole word or phrase, not inside another
/// word: `rule` is not found in `ruler`. `text` is already lowercased, and
/// `phrase` is ASCII, so each end of a find is a character boundary.
fn has_phrase(text: &str, phrase: &str) -> bool {
    let is_word = |c: char| c.is_alphanumeric() || c == '_';
    let mut from = 0;
    while let Some(i) = text[from..].find(phrase) {
        let start = from + i;
        let end = start + phrase.len();
        let before = text[..start]
            .chars()
            .next_back()
            .is_none_or(|c| !is_word(c));
        let after = text[end..].chars().next().is_none_or(|c| !is_word(c));
        if before && after {
            return true;
        }
        from = end;
    }
    false
}

/// The cues in `content`, in [`CUES`] order.
pub fn cues(content: &str) -> Vec<&'static str> {
    let text = content.to_lowercase();
    CUES.iter()
        .copied()
        .filter(|cue| has_phrase(&text, cue))
        .collect()
}

/// The first [`EXCERPT_CHARS`] characters of `content`, trimmed.
pub fn excerpt(content: &str) -> String {
    content.trim().chars().take(EXCERPT_CHARS).collect()
}

/// Whether `m` is a record of its own rather than a memory to read: a
/// behavior, a handoff, or a dependency claim.
fn is_record(m: &Memory) -> bool {
    State::of(&m.evidence).is_some()
        || Edge::of(m).is_some()
        || m.compartment.as_ref().is_some_and(handoff::is_compartment)
}

/// `m` as a candidate, when it is one: not volatile, not a record of its own,
/// not among `cited` (the memories behaviors already cite), and picked by its
/// network or its wording.
pub fn candidate(m: &Memory, cited: &HashSet<String>) -> Option<Candidate> {
    if m.volatile || is_record(m) || cited.contains(m.id.as_str()) {
        return None;
    }
    let mut reasons = Vec::new();
    if m.network == MemoryNetwork::Opinion {
        reasons.push(OPINION);
    }
    reasons.extend(cues(&m.content));
    if reasons.is_empty() {
        return None;
    }
    Some(Candidate {
        id: m.id.as_str().to_string(),
        created_at: m.created_at,
        network: m.network,
        repo: GitProvenance::from_evidence(&m.evidence).map(|g| g.repo),
        excerpt: excerpt(&m.content),
        reasons,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::behavior::{self, Status};
    use crate::ids::{TenantId, UserId};

    fn tenant() -> TenantId {
        TenantId::new("ws:test")
    }

    fn memory(id: &str, network: MemoryNetwork, content: &str) -> Memory {
        Memory::new(id, tenant(), network, content, 0.8, Utc::now())
    }

    fn nobody() -> HashSet<String> {
        HashSet::new()
    }

    #[test]
    fn a_cue_is_found_as_a_whole_word_or_phrase() {
        assert_eq!(cues("Never stack pull requests."), vec!["never"]);
        assert_eq!(cues("Don't ask; proceed."), vec!["don't"]);
        assert_eq!(
            cues("Use `-layout` instead of `-lineprinter`."),
            vec!["instead of"]
        );
        assert_eq!(
            cues("From now on the rule is: always fmt."),
            vec!["always", "from now on", "rule"]
        );
        assert!(cues("The ruler is 30 cm, and the Rulebook is thick.").is_empty());
        assert!(cues("surql-rs 0.28 shipped; 71 memories cluster on it.").is_empty());
    }

    #[test]
    fn an_opinion_is_picked_and_a_fact_without_a_rules_wording_is_not() {
        let opinion = memory("memory:o", MemoryNetwork::Opinion, "I like short PRs.");
        let picked = candidate(&opinion, &nobody()).expect("picked");
        assert_eq!(picked.reasons, vec![OPINION]);
        assert_eq!(picked.excerpt, "I like short PRs.");

        let rule = memory(
            "memory:w",
            MemoryNetwork::World,
            "Branches are named feat/{issue}-{slug}, never feature/.",
        );
        assert_eq!(candidate(&rule, &nobody()).unwrap().reasons, vec!["never"]);

        let fact = memory(
            "memory:f",
            MemoryNetwork::Bank,
            "Release 0.28 of surql-rs shipped on Tuesday.",
        );
        assert!(candidate(&fact, &nobody()).is_none());
    }

    #[test]
    fn records_volatile_memories_and_cited_memories_are_passed_over() {
        let mut behavior = memory("memory:b", MemoryNetwork::Opinion, "Never stack.");
        behavior
            .evidence
            .push(behavior::status_evidence(Status::Proposed));
        assert!(candidate(&behavior, &nobody()).is_none(), "a behavior");

        let handoff = memory(
            "memory:h",
            MemoryNetwork::Opinion,
            "Rerun this on the GPU box.",
        )
        .in_compartment(handoff::compartment_id(&tenant(), &UserId::new("user:a")));
        assert!(candidate(&handoff, &nobody()).is_none(), "a handoff");

        let counter = memory("memory:v", MemoryNetwork::Opinion, "[skill-use:x] 3").volatile(true);
        assert!(candidate(&counter, &nobody()).is_none(), "volatile");

        let rule = memory(
            "memory:r",
            MemoryNetwork::World,
            "Always run cargo fmt first.",
        );
        let cited: HashSet<String> = ["memory:r".to_string()].into_iter().collect();
        assert!(candidate(&rule, &cited).is_none(), "cited by a behavior");
        assert!(candidate(&rule, &nobody()).is_some());
    }

    #[test]
    fn a_candidate_carries_its_repository_and_a_bounded_excerpt() {
        let long = "Never ".repeat(200);
        let mut m = memory("memory:l", MemoryNetwork::Bank, &long);
        m.evidence.push(
            GitProvenance::new("github.com/acme/api", "0123456789abcdef")
                .on_branch("main")
                .to_evidence(),
        );
        let c = candidate(&m, &nobody()).expect("picked");
        assert_eq!(c.repo.as_deref(), Some("github.com/acme/api"));
        assert_eq!(c.excerpt.chars().count(), EXCERPT_CHARS);
        assert_eq!(c.network, MemoryNetwork::Bank);
    }
}
