//! Handoffs: work a session leaves for a session on another of the
//! user's machines.
//!
//! A handoff is a memory in the user's own `handoff` compartment. Two evidence
//! entries carry its state, the same way a git anchor rides on a memory: who it
//! is for (one of the user's machines by host name, or [`ANY`]), and, once a
//! session has dealt with it, who marked it done and when. A session starting
//! on the target machine is told what is waiting until one marks it done; after
//! that it is no longer announced but stays readable, as history.
//!
//! Asynchronous and one direction: a session leaves a note, a later session
//! reads it. No session drives another, which is what separates this from the
//! remote control that sovereign mode gives up.

use chrono::{DateTime, Utc};

use crate::ids::{CompartmentId, TenantId, UserId};
use crate::memory::Memory;

/// Addressed to whichever of the user's machines starts a session next.
pub const ANY: &str = "any";

const FOR: &str = "handoff-for:";
const DONE: &str = "handoff-done:";

/// The longest title a handoff is announced under.
const TITLE_CHARS: usize = 80;

/// The name of a user's handoff compartment, and the last part of its id.
pub const COMPARTMENT_NAME: &str = "handoff";

/// The user's handoff compartment, one per `(tenant, user)`, named the way the
/// default compartment is.
pub fn compartment_id(tenant: &TenantId, user: &UserId) -> CompartmentId {
    CompartmentId::new(format!(
        "comp:{}:{}:{COMPARTMENT_NAME}",
        tenant.as_str(),
        user.as_str()
    ))
}

/// Whether `compartment` is a user's handoff compartment, whichever user's.
pub fn is_compartment(compartment: &CompartmentId) -> bool {
    compartment
        .as_str()
        .ends_with(&format!(":{COMPARTMENT_NAME}"))
}

/// A host name as handoffs compare them: trimmed and lowercased, so `Kuskokwim`
/// and `kuskokwim ` are the same machine. Empty means [`ANY`].
pub fn normalize_host(host: &str) -> String {
    let h = host.trim().to_lowercase();
    if h.is_empty() {
        ANY.to_string()
    } else {
        h
    }
}

/// The evidence entry saying who a handoff is for.
pub fn for_evidence(host: &str) -> String {
    format!("{FOR}{}", normalize_host(host))
}

/// The evidence entry saying a handoff was dealt with, by which machine, when.
pub fn done_evidence(host: &str, at: DateTime<Utc>) -> String {
    format!("{DONE}{}@{}", at.to_rfc3339(), normalize_host(host))
}

/// Where a handoff stands, read back from its evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HandoffState {
    /// The machine it is for, or [`ANY`].
    pub for_host: String,
    /// When it was marked done and by which machine. `None` while it waits.
    pub done: Option<(DateTime<Utc>, String)>,
}

impl HandoffState {
    /// Read a memory's evidence. `None` when it carries no addressee, which is
    /// to say it is not a handoff.
    pub fn of(evidence: &[String]) -> Option<Self> {
        let for_host = evidence
            .iter()
            .find_map(|e| e.strip_prefix(FOR))
            .map(normalize_host)?;
        let done = evidence.iter().find_map(|e| {
            let (at, host) = e.strip_prefix(DONE)?.rsplit_once('@')?;
            let at = DateTime::parse_from_rfc3339(at).ok()?.with_timezone(&Utc);
            Some((at, host.to_string()))
        });
        Some(Self { for_host, done })
    }

    /// Whether a session on `host` should hear about it: addressed to that
    /// machine or to any, and not yet done.
    pub fn waits_for(&self, host: &str) -> bool {
        self.done.is_none() && (self.for_host == ANY || self.for_host == normalize_host(host))
    }
}

/// What a handoff is announced under: its first non-empty line, cut at a word
/// boundary to fit one line of the session-start block.
pub fn title(content: &str) -> String {
    let line = content
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("(empty)");
    if line.chars().count() <= TITLE_CHARS {
        return line.to_string();
    }
    let cut: String = line.chars().take(TITLE_CHARS).collect();
    let cut = cut.rsplit_once(' ').map_or(cut.as_str(), |(head, _)| head);
    format!("{}...", cut.trim_end())
}

/// How long ago, as a session-start line says it: minutes, hours or days.
pub fn age(then: DateTime<Utc>, now: DateTime<Utc>) -> String {
    let minutes = (now - then).num_minutes().max(0);
    match minutes {
        0..=59 => format!("{minutes}m ago"),
        60..=2879 => format!("{}h ago", minutes / 60),
        _ => format!("{}d ago", minutes / 1440),
    }
}

/// The handoffs waiting for `host` among `memories`, newest first.
pub fn waiting_for<'a>(memories: &'a [Memory], host: &str) -> Vec<&'a Memory> {
    let mut waiting: Vec<&Memory> = memories
        .iter()
        .filter(|m| HandoffState::of(&m.evidence).is_some_and(|s| s.waits_for(host)))
        .collect();
    waiting.sort_by_key(|m| std::cmp::Reverse(m.created_at));
    waiting
}

/// The session-start block's lines about handoffs: a count and one line per
/// handoff, newest first, and how to read and close one. `None` when nothing
/// waits, so the block says nothing rather than "0 handoffs".
pub fn announcement(waiting: &[&Memory], host: &str, now: DateTime<Utc>) -> Option<String> {
    if waiting.is_empty() {
        return None;
    }
    let host = normalize_host(host);
    let mut out = format!(
        "{} handoff{} waiting for this machine ({host}):",
        waiting.len(),
        if waiting.len() == 1 { "" } else { "s" }
    );
    for m in waiting {
        let from = m.author_host.as_deref().unwrap_or("unknown host");
        out.push_str(&format!(
            "\n- {} (from {from}, {}; id {})",
            title(&m.content),
            age(m.created_at, now),
            m.id.as_str()
        ));
    }
    out.push_str(&format!(
        "\nRead them in full with the handoffs tool (host \"{host}\", full true); once one is dealt with, complete_handoff with its id so it stops being announced."
    ));
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryNetwork;
    use chrono::Duration;

    fn handoff(id: &str, content: &str, evidence: &[String], at: DateTime<Utc>) -> Memory {
        let mut m = Memory::new(
            id,
            TenantId::new("ws:t"),
            MemoryNetwork::Bank,
            content,
            1.0,
            at,
        )
        .with_evidence(evidence.to_vec());
        m.author_host = Some("windows".into());
        m
    }

    #[test]
    fn hosts_compare_trimmed_and_lowercased_and_empty_is_any() {
        assert_eq!(normalize_host(" Kuskokwim "), "kuskokwim");
        assert_eq!(normalize_host(""), ANY);
        assert_eq!(for_evidence("Shaman"), "handoff-for:shaman");
    }

    #[test]
    fn state_reads_back_from_evidence() {
        let now = Utc::now();
        let waiting = HandoffState::of(&[for_evidence("kuskokwim")]).unwrap();
        assert!(waiting.waits_for("KUSKOKWIM") && !waiting.waits_for("windows"));
        let any = HandoffState::of(&[for_evidence(ANY)]).unwrap();
        assert!(any.waits_for("windows") && any.waits_for("kuskokwim"));
        let done = HandoffState::of(&[for_evidence(ANY), done_evidence("windows", now)]).unwrap();
        assert!(!done.waits_for("windows"));
        let (at, by) = done.done.unwrap();
        assert_eq!(by, "windows");
        assert_eq!(at.timestamp(), now.timestamp());
        assert_eq!(HandoffState::of(&["git:x".into()]), None, "not a handoff");
    }

    #[test]
    fn a_title_is_the_first_line_cut_at_a_word() {
        assert_eq!(
            title("\n  Deploy the dashboard \nthen check it"),
            "Deploy the dashboard"
        );
        let long = "word ".repeat(40);
        let t = title(&long);
        assert!(
            t.ends_with("...") && t.chars().count() <= TITLE_CHARS + 3,
            "{t}"
        );
        assert!(!t.contains("wor..."), "cut at a word: {t}");
        assert_eq!(title("   "), "(empty)");
    }

    #[test]
    fn ages_read_in_minutes_hours_or_days() {
        let now = Utc::now();
        assert_eq!(age(now - Duration::minutes(5), now), "5m ago");
        assert_eq!(age(now - Duration::hours(3), now), "3h ago");
        assert_eq!(age(now - Duration::days(4), now), "4d ago");
    }

    /// Only handoffs for this machine or any, not done, newest first; nothing
    /// waiting announces nothing.
    #[test]
    fn the_announcement_lists_what_waits_here_newest_first() {
        let now = Utc::now();
        let memories = vec![
            handoff(
                "memory:a",
                "Old note for here",
                &[for_evidence("kuskokwim")],
                now - Duration::hours(5),
            ),
            handoff(
                "memory:b",
                "Newer note for any",
                &[for_evidence(ANY)],
                now - Duration::hours(1),
            ),
            handoff(
                "memory:c",
                "For the other box",
                &[for_evidence("shaman")],
                now,
            ),
            handoff(
                "memory:d",
                "Already done",
                &[for_evidence("kuskokwim"), done_evidence("kuskokwim", now)],
                now,
            ),
            handoff("memory:e", "Not a handoff", &[], now),
        ];
        let waiting = waiting_for(&memories, "Kuskokwim");
        let ids: Vec<&str> = waiting.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, ["memory:b", "memory:a"]);
        let text = announcement(&waiting, "Kuskokwim", now).unwrap();
        assert!(
            text.starts_with("2 handoffs waiting for this machine (kuskokwim):"),
            "{text}"
        );
        assert!(
            text.contains("- Newer note for any (from windows, 1h ago; id memory:b)"),
            "{text}"
        );
        assert!(text.contains("complete_handoff"), "{text}");
        assert!(text.len() < 600, "a few hundred characters: {}", text.len());
        // A handoff for any machine waits everywhere; nothing waiting says
        // nothing at all.
        let elsewhere = waiting_for(&memories, "nowhere");
        assert_eq!(elsewhere.len(), 1);
        assert_eq!(elsewhere[0].id.as_str(), "memory:b");
        assert_eq!(announcement(&[], "nowhere", now), None);
    }
}
