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
//! Four signals pick a memory, and each is named in the candidate's reasons:
//! it is an opinion (judgments, preferences and corrections are filed there,
//! and most of them say how to act); it opens with a category tag about how
//! to act, such as `[feedback]` or `[convention]`; it carries a phrase that
//! records a correction or a preference, such as "standing preference" or
//! "pushed back"; or one of its sentences is short and opens like a rule, with
//! a negation or an imperative verb ("never stack a pull request", "use the
//! host alias"). Those sentences come back with the candidate, since the rule
//! is often an aside deep inside a long note.
//!
//! Measured on a store of 7,000 memories: a cue word anywhere picked 68% of
//! the store, because long technical notes say "must" and "never" constantly;
//! a sentence opening with any imperative verb picked 54%, because labels and
//! list items open with verbs too ("Deploy: kuskokwim at 42bf9b7"); the shape
//! below picks 31%, of which 13% is the opinion network. Precision is the
//! agent's: a fact that opens like a rule is read and passed over. The
//! ignored test `measure_on_a_sweep` repeats the measurement on a dump. What is left out without a look: behaviors and handoffs,
//! which are records of their own; dependency claims; volatile memories, which
//! are counters and state; and memories a behavior already cites.

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

/// The longest sentence read as a rule. A rule is a sentence; a paragraph
/// that happens to open with "never" is a narrative.
pub const SENTENCE_CHARS: usize = 160;

/// The fewest words a sentence has to have to be read as a rule: a verb
/// needs an object.
const SENTENCE_WORDS: usize = 3;

/// How many matching sentences a candidate carries.
pub const MAX_HITS: usize = 3;

/// How many reasons a candidate carries.
pub const MAX_REASONS: usize = 8;

/// Why an opinion memory is picked whatever its wording.
pub const OPINION: &str = "opinion";

/// Words in a leading bracketed tag that mark a memory as about how to act:
/// `[feedback] ...`, `[User preference pattern] ...`.
pub const TAGS: &[&str] = &[
    "feedback",
    "convention",
    "correction",
    "preference",
    "rule",
    "standing",
    "workflow",
    "decision",
    "process",
    "directive",
    "lesson",
];

/// Phrases that record a correction or a preference wherever they appear,
/// matched as whole words in the lowercased text.
pub const PHRASES: &[&str] = &[
    "standing rule",
    "standing preference",
    "make it a rule",
    "the rule is",
    "from now on",
    "going forward",
    "directive",
    "pushed back",
    "user preference",
    "user feedback",
    "user correction",
    "user corrected",
    "corrected me",
    "corrected my",
];

/// Words a sentence opens with that state a rule on their own: a negation,
/// a preference, or a verb that is itself a refusal.
pub const STRONG_OPENERS: &[&str] = &[
    "never", "always", "do not", "don't", "prefer", "avoid", "stop", "skip",
];

/// Imperative verbs a sentence opens with when it states a procedure. A verb
/// alone is also how a label or a list item begins ("Deploy: kuskokwim at
/// 42bf9b7"), so one of these counts only when the sentence also carries one
/// of [`CONDITIONS`]. Verbs that are as often a noun or a state ("open",
/// "merge", "set", "name") are left out: "Open worktrees until merged" is a
/// status, not a rule.
pub const VERBS: &[&str] = &[
    "use", "run", "keep", "write", "deploy", "commit", "push", "treat", "ask", "check", "read",
    "recall", "leave", "cite", "record", "verify", "wait", "proceed", "add", "remove", "store",
    "save",
];

/// What a rule has that a label does not: a scope, a condition, or a
/// contrast.
pub const CONDITIONS: &[&str] = &[
    "not", "never", "always", "instead", "rather", "before", "after", "every", "only", "when",
    "whenever", "unless", "until", "any", "all", "first", "yourself",
];

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
    /// The sentences that open like a rule, up to [`MAX_HITS`].
    pub hits: Vec<String>,
    /// Why it was picked, up to [`MAX_REASONS`]: [`OPINION`], `tag:<word>`,
    /// `says:<phrase>`, `opens:<word>`.
    pub reasons: Vec<String>,
}

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Whether `text` has `phrase` as a whole word or phrase, not inside another
/// word: `rule` is not found in `ruler`. `text` is already lowercased, and
/// `phrase` is ASCII, so each end of a find is a character boundary.
fn has_phrase(text: &str, phrase: &str) -> bool {
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

/// The [`PHRASES`] in `content`, in their order.
pub fn phrases(content: &str) -> Vec<&'static str> {
    let text = content.to_lowercase();
    PHRASES
        .iter()
        .copied()
        .filter(|p| has_phrase(&text, p))
        .collect()
}

/// The tag word a memory opens with, when its leading `[...]` names one of
/// [`TAGS`]: `[feedback]`, `[CORRECTION to ...]`, `[User preference pattern]`.
pub fn tag(content: &str) -> Option<&'static str> {
    let rest = content.trim_start().strip_prefix('[')?;
    let end = rest.find(']')?;
    if end > 120 {
        return None;
    }
    let inside = rest[..end].to_lowercase();
    TAGS.iter().copied().find(|t| has_phrase(&inside, t))
}

/// The opener a sentence states a rule with, when it does: after any bullet,
/// number or quote, it begins with one of [`STRONG_OPENERS`], or with one of
/// [`VERBS`] and carries one of [`CONDITIONS`]; it runs to at most
/// [`SENTENCE_CHARS`] and has at least [`SENTENCE_WORDS`] words; and the
/// opener is not a label (`name = ...`).
pub fn opener(sentence: &str) -> Option<&'static str> {
    let body = sentence
        .trim()
        .trim_start_matches(|c: char| !c.is_alphabetic());
    let words = body
        .split_whitespace()
        .filter(|w| w.chars().any(char::is_alphabetic))
        .count();
    if body.chars().count() > SENTENCE_CHARS || words < SENTENCE_WORDS {
        return None;
    }
    let lower = body.to_lowercase();
    // A hyphen joins a word: `read-only` does not open with the verb `read`.
    let opens_with = |o: &str| {
        lower.starts_with(o)
            && lower[o.len()..]
                .chars()
                .next()
                .is_none_or(|c| !is_word(c) && c != '-')
            && !lower[o.len()..].trim_start().starts_with('=')
    };
    if let Some(o) = STRONG_OPENERS.iter().copied().find(|o| opens_with(o)) {
        return Some(o);
    }
    let verb = VERBS.iter().copied().find(|o| opens_with(o))?;
    let rest = &lower[verb.len()..];
    CONDITIONS
        .iter()
        .any(|c| has_phrase(rest, c))
        .then_some(verb)
}

/// The sentences of `content` that open like a rule, each with its opener,
/// in order. Sentences end at a period, an exclamation or question mark, a
/// colon, a semicolon, or a line break: "Shon said: never stack pull
/// requests" carries the rule after its colon, and "Deploy: kuskokwim at
/// 42bf9b7" is a label before its own.
pub fn rule_sentences(content: &str) -> Vec<(&'static str, String)> {
    content
        .split(['.', '!', '?', ':', ';', '\n'])
        .filter_map(|s| {
            let s = s.trim().trim_start_matches(|c: char| !c.is_alphabetic());
            opener(s).map(|o| (o, s.to_string()))
        })
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
/// network, a tag, a phrase, or a sentence that opens like a rule.
pub fn candidate(m: &Memory, cited: &HashSet<String>) -> Option<Candidate> {
    if m.volatile || is_record(m) || cited.contains(m.id.as_str()) {
        return None;
    }
    let mut reasons: Vec<String> = Vec::new();
    if m.network == MemoryNetwork::Opinion {
        reasons.push(OPINION.to_string());
    }
    if let Some(t) = tag(&m.content) {
        reasons.push(format!("tag:{t}"));
    }
    reasons.extend(phrases(&m.content).into_iter().map(|p| format!("says:{p}")));
    let sentences = rule_sentences(&m.content);
    let mut openers: Vec<&str> = Vec::new();
    for (o, _) in &sentences {
        if !openers.contains(o) {
            openers.push(o);
        }
    }
    reasons.extend(openers.into_iter().map(|o| format!("opens:{o}")));
    if reasons.is_empty() {
        return None;
    }
    reasons.truncate(MAX_REASONS);
    Some(Candidate {
        id: m.id.as_str().to_string(),
        created_at: m.created_at,
        network: m.network,
        repo: GitProvenance::from_evidence(&m.evidence).map(|g| g.repo),
        excerpt: excerpt(&m.content),
        hits: sentences
            .into_iter()
            .map(|(_, s)| s)
            .take(MAX_HITS)
            .collect(),
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
    fn a_sentence_opens_like_a_rule_with_a_negation_or_a_conditioned_verb() {
        assert_eq!(
            opener("Never stack a pull request on an unmerged branch"),
            Some("never")
        );
        assert_eq!(opener("- Don't ask before agreed work"), Some("don't"));
        assert_eq!(opener("3. Use the host alias, not an address"), Some("use"));
        assert_eq!(
            opener("\"Do not chain git commands with &&\""),
            Some("do not")
        );
        assert_eq!(
            opener("Run text sweeps before adding literals"),
            Some("run")
        );
        assert_eq!(opener("Deploy it yourself after the merge"), Some("deploy"));
        assert_eq!(opener("Avoid dumping the whole profile"), Some("avoid"));
        assert_eq!(
            opener("The dispatcher never reads it"),
            None,
            "a negation mid-sentence"
        );
        assert_eq!(opener("Rhai never widens int to float"), None);
        assert_eq!(opener("Never again"), None, "too short to carry an object");
        assert_eq!(opener("Users of the API"), None, "`use` inside a word");
        assert_eq!(
            opener("Check the index applied in prd"),
            None,
            "a verb without a condition"
        );
        assert_eq!(
            opener("Open worktrees until merged"),
            None,
            "a state as often as a verb"
        );
        assert_eq!(opener("use = a tinker mends pots"), None, "a label");
        assert_eq!(
            opener("read-only-do-it, dps and naming not learned"),
            None,
            "a hyphenated word"
        );
        assert_eq!(opener("Stop p95 1"), None, "numbers are not words");
        let long = format!("Never {}", "x ".repeat(100));
        assert_eq!(opener(&long), None, "a paragraph is not a rule");
    }

    #[test]
    fn rule_sentences_are_cut_at_sentence_ends_and_keep_their_order() {
        let note = "Deploy: kuskokwim at 42bf9b7. Shon said: never stack pull requests; \
                    always open them against main. The gate never fires.\n\
                    - Use `pdftotext -layout` for every PDF";
        let hits = rule_sentences(note);
        let openers: Vec<&str> = hits.iter().map(|(o, _)| *o).collect();
        assert_eq!(openers, ["never", "always", "use"]);
        assert_eq!(hits[0].1, "never stack pull requests");
        assert_eq!(hits[1].1, "always open them against main");
        assert_eq!(hits[2].1, "Use `pdftotext -layout` for every PDF");
    }

    #[test]
    fn a_tag_and_a_phrase_pick_a_memory_whatever_its_shape() {
        assert_eq!(
            tag("[feedback] the invitation email should be branded"),
            Some("feedback")
        );
        assert_eq!(
            tag("  [CORRECTION 2026-06-18 to prior memories] they were wrong"),
            Some("correction")
        );
        assert_eq!(
            tag("[User preference pattern, confirmed three times] ..."),
            Some("preference")
        );
        assert_eq!(tag("[Project] kushtakas status"), None);
        assert_eq!(
            tag("[data-plane-schema-cli - DP-668 master specifications] done"),
            None
        );
        assert_eq!(tag("no bracket here"), None);

        assert_eq!(
            phrases("SHON DIRECTIVE: the email should be branded."),
            vec!["directive"]
        );
        assert_eq!(
            phrases("User pushed back on the animal trio."),
            vec!["pushed back"]
        );
        assert_eq!(
            phrases("The ruler is 30 cm; it corrects nothing."),
            Vec::<&str>::new()
        );
    }

    #[test]
    fn an_opinion_is_picked_and_a_fact_without_a_rules_shape_is_not() {
        let opinion = memory("memory:o", MemoryNetwork::Opinion, "I like short PRs.");
        let picked = candidate(&opinion, &nobody()).expect("picked");
        assert_eq!(picked.reasons, [OPINION]);
        assert!(picked.hits.is_empty());
        assert_eq!(picked.excerpt, "I like short PRs.");

        let rule = memory(
            "memory:w",
            MemoryNetwork::World,
            "Branch naming in this repository. Never name a branch feature/anything; \
             use feat/{issue}-{slug} instead.",
        );
        let picked = candidate(&rule, &nobody()).expect("picked");
        assert_eq!(picked.reasons, ["opens:never", "opens:use"]);
        assert_eq!(
            picked.hits,
            [
                "Never name a branch feature/anything",
                "use feat/{issue}-{slug} instead"
            ]
        );

        let fact = memory(
            "memory:f",
            MemoryNetwork::Bank,
            "Release 0.28 of surql-rs shipped on Tuesday. The gate must never fire twice, \
             and the dispatcher never reads the claim.",
        );
        assert!(
            candidate(&fact, &nobody()).is_none(),
            "cue words mid-sentence are facts"
        );
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
        let long = format!("[feedback] {}", "word ".repeat(200));
        let mut m = memory("memory:l", MemoryNetwork::Bank, &long);
        m.evidence.push(
            GitProvenance::new("github.com/acme/api", "0123456789abcdef")
                .on_branch("main")
                .to_evidence(),
        );
        let c = candidate(&m, &nobody()).expect("picked");
        assert_eq!(c.reasons, ["tag:feedback"]);
        assert_eq!(c.repo.as_deref(), Some("github.com/acme/api"));
        assert_eq!(c.excerpt.chars().count(), EXCERPT_CHARS);
        assert_eq!(c.network, MemoryNetwork::Bank);
    }

    /// Runs the filter over a dump of a real store and prints what it picks,
    /// so a change to the signals is measured before it ships. Ignored because
    /// it reads files: point `ANTUMBRA_SWEEP_DIR` at a directory of
    /// `page-*.json` files, each `{"memories": [{id, network, content, ...}]}`
    /// as `list_memories` returns them, and run with `--ignored --nocapture`.
    #[test]
    #[ignore = "reads a store dump; set ANTUMBRA_SWEEP_DIR"]
    fn measure_on_a_sweep() {
        let Ok(dir) = std::env::var("ANTUMBRA_SWEEP_DIR") else {
            println!("ANTUMBRA_SWEEP_DIR unset -- skipped");
            return;
        };
        let mut rows = Vec::new();
        for entry in std::fs::read_dir(&dir).expect("the sweep directory") {
            let path = entry.expect("an entry").path();
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            if !(name.starts_with("page-") && name.ends_with(".json")) {
                continue;
            }
            let text = std::fs::read_to_string(&path).expect("a page");
            let page: serde_json::Value = serde_json::from_str(&text).expect("json");
            for v in page["memories"].as_array().cloned().unwrap_or_default() {
                rows.push(v);
            }
        }
        let mut total = 0usize;
        let mut picked = 0usize;
        let mut by_network = std::collections::BTreeMap::new();
        let mut by_signal = std::collections::BTreeMap::new();
        let mut by_reason: std::collections::BTreeMap<String, usize> =
            std::collections::BTreeMap::new();
        let mut samples = Vec::new();
        for v in &rows {
            let id = v["id"].as_str().unwrap_or_default();
            let content = v["content"].as_str().unwrap_or_default();
            if id.starts_with("memory:dep-") {
                continue;
            }
            let network = match v["network"].as_str() {
                Some("opinion") => MemoryNetwork::Opinion,
                Some("bank") => MemoryNetwork::Bank,
                _ => MemoryNetwork::World,
            };
            let mut m = memory(id, network, content);
            if content.contains("```behavior") || content.contains("```behaviour") {
                m.evidence.push(behavior::status_evidence(Status::Proposed));
                continue;
            }
            total += 1;
            if let Some(c) = candidate(&m, &nobody()) {
                picked += 1;
                *by_network.entry(network.as_str()).or_insert(0usize) += 1;
                for r in &c.reasons {
                    let kind = r.split(':').next().unwrap_or(r).to_string();
                    *by_signal.entry(kind).or_insert(0usize) += 1;
                    *by_reason.entry(r.clone()).or_insert(0usize) += 1;
                }
                if samples.len() < 40 && !c.hits.is_empty() {
                    samples.push(format!("{id} [{}] {}", c.reasons.join(","), c.hits[0]));
                }
            }
        }
        println!(
            "memories {total}, candidates {picked} ({:.1}%)",
            100.0 * picked as f64 / total.max(1) as f64
        );
        println!("by network: {by_network:?}");
        println!("memories with each signal: {by_signal:?}");
        let mut reasons: Vec<(String, usize)> = by_reason.into_iter().collect();
        reasons.sort_by_key(|r| std::cmp::Reverse(r.1));
        println!("each reason: {:?}", &reasons[..reasons.len().min(40)]);
        for s in samples {
            println!("  {s}");
        }
    }
}
