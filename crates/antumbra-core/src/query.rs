//! The parts of a recall query that asks for more than one thing.
//!
//! A prompt like "move the tokens out of the config and record the dependency
//! edges" asks about two subjects. Its embedding is a blend of both and lands
//! near neither memory, and its words split between them, so the first stage
//! of recall can miss both while the reranker, reading the whole prompt, scores
//! each of them highly. Measured on 2026-10-04: neither memory reached the 100
//! candidates the reranker sees, and the reranker scored them 0.965 and 0.854
//! against that prompt. Retrieving for each part as well as for the whole puts
//! them in front of it.
//!
//! The split is lexical and cautious. A sentence is a part; within a sentence,
//! a part ends at "and", "then" or "also" only when both sides are long enough
//! to be a request of their own, so "salt and pepper" stays whole. A prompt
//! that asks for one thing has no parts, and recall stays as it was.

/// The most parts one query is split into, so a long paste costs a bounded
/// number of extra retrievals.
const MAX_PARTS: usize = 4;
/// The fewest words a part needs.
const MIN_WORDS: usize = 3;
/// The words a request is joined at.
const JOINS: [&str; 3] = ["and", "then", "also"];

/// The parts of `query`, when it asks for more than one thing; empty when it
/// asks for one.
pub fn parts(query: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for sentence in sentences(query) {
        for clause in clauses(&sentence) {
            let lower = clause.to_lowercase();
            if clause.split_whitespace().count() >= MIN_WORDS
                && !out.iter().any(|p| p.to_lowercase() == lower)
            {
                out.push(clause);
            }
        }
    }
    if out.len() < 2 {
        return Vec::new();
    }
    out.truncate(MAX_PARTS);
    out
}

/// Lines, then sentences: a `.`, `?`, `!` or `;` ends one only when a space
/// or the end follows it, so `~/.claude.json` and `2.1.289` stay whole.
fn sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in text.lines() {
        let mut current = String::new();
        let mut chars = line.chars().peekable();
        while let Some(c) = chars.next() {
            current.push(c);
            let ends = matches!(c, '.' | '?' | '!' | ';')
                && chars.peek().is_none_or(|next| next.is_whitespace());
            if ends {
                push_trimmed(&mut out, &current);
                current.clear();
            }
        }
        push_trimmed(&mut out, &current);
    }
    out
}

fn push_trimmed(out: &mut Vec<String>, text: &str) {
    let text = text.trim().trim_end_matches(['.', '?', '!', ';']).trim();
    if !text.is_empty() {
        out.push(text.to_string());
    }
}

/// A sentence cut at each join with enough words on both sides.
fn clauses(sentence: &str) -> Vec<String> {
    let words: Vec<&str> = sentence.split_whitespace().collect();
    let mut out = Vec::new();
    let mut start = 0;
    for (i, word) in words.iter().enumerate() {
        let bare = word
            .trim_matches(|c: char| !c.is_alphanumeric())
            .to_lowercase();
        let before = i - start;
        let after = words.len() - i - 1;
        if JOINS.contains(&bare.as_str()) && before >= MIN_WORDS && after >= MIN_WORDS {
            out.push(clean(&words[start..i]));
            start = i + 1;
        }
    }
    out.push(clean(&words[start..]));
    out
}

/// Words back into a clause, without the comma a join leaves behind.
fn clean(words: &[&str]) -> String {
    words.join(" ").trim_end_matches(',').trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_prompt_that_asks_for_two_things_is_two_parts() {
        assert_eq!(
            parts("move the tokens out of ~/.claude.json and record the 23 dependency edges on kushkokwim"),
            [
                "move the tokens out of ~/.claude.json",
                "record the 23 dependency edges on kushkokwim"
            ]
        );
    }

    #[test]
    fn a_prompt_that_asks_for_one_thing_has_no_parts() {
        for one in [
            "Proceed on antumbra work",
            "ok what do you need from me exactly",
            "add salt and pepper to the soup",
            "deploy 2.1.289 to ~/.claude.json now",
        ] {
            assert!(parts(one).is_empty(), "{one}: {:?}", parts(one));
        }
    }

    /// Sentences and lines are parts; a dot inside a name or a version is not
    /// the end of a sentence.
    #[test]
    fn sentences_and_lines_are_parts() {
        assert_eq!(
            parts(
                "Check the deploy on kuskokwim. Then roll the shaman pin\nand tell me what changed"
            ),
            [
                "Check the deploy on kuskokwim",
                "Then roll the shaman pin",
                "and tell me what changed"
            ]
        );
        assert_eq!(
            parts("the version is 2.1.289 now; the config is ~/.claude.json there"),
            [
                "the version is 2.1.289 now",
                "the config is ~/.claude.json there"
            ]
        );
    }

    #[test]
    fn a_long_paste_is_cut_to_a_few_distinct_parts() {
        let text = "one two three. four five six. one two three. seven eight nine. ten eleven twelve. thirteen fourteen fifteen.";
        let got = parts(text);
        assert_eq!(got.len(), MAX_PARTS);
        assert_eq!(got[2], "seven eight nine", "a repeated part counts once");
    }
}
