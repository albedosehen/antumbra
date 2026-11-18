//! The command palette's vocabulary: the actions it can run and a small
//! hand-rolled fuzzy matcher (subsequence + gap scoring) that ranks them against
//! the typed query. A key ([`crate::keys`]) and the palette both resolve to an
//! [`Action`], which is carried out in one place ([`crate::drive`]).

use crate::app::{Focus, LayoutMode, Page};

/// Something the console can do, asked for by a key or chosen from the
/// palette. What a few of them mean depends on the view they are asked in:
/// [`Action::Move`] moves the selection, or scrolls an overlay.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Action {
    // The console itself.
    Quit,
    Reload,
    Help,
    Palette,
    Filter,
    Events,
    Connect,
    Knobs,
    Gate,
    Ask,
    /// Let the Services page's walk go further.
    Hops,
    // What is on screen.
    Page(Page),
    /// The next page (`1`) or the one before (`-1`), wrapping.
    PageStep(i32),
    Focus(Focus),
    /// The next list on the population page.
    FocusNext,
    Layout(LayoutMode),
    LayoutNext,
    CycleSort,
    CycleTheme,
    FpsUp,
    FpsDown,
    FollowMonitor,
    // The selection, or an overlay's scroll.
    /// Down (`1`) or up (`-1`) one.
    Move(i32),
    First,
    Last,
    /// Down (`1`) or up (`-1`) ten.
    PageMove(i32),
    /// Enter: open what is selected, run the highlighted command, jump to the
    /// highlighted match, route the task, or apply the knobs.
    Submit,
    // Acting on the selection, each behind a confirm.
    /// The `x` key: what it does follows from the page and the list.
    Act,
    GraduateShadow,
    FreezeExpert,
    CancelHalt,
    // An overlay.
    Close,
    /// A character typed into the palette, the filter or the ask.
    Type(char),
    Backspace,
    /// A knob one step down (`-1`) or up (`1`).
    Turn(i32),
    /// Go ahead with the staged action.
    Confirm,
    /// Drop the staged action.
    Decline,
}

/// A palette entry: what it does, in plain words that are also what a query
/// is matched against, the key that does the same from the main view (empty
/// for one that has none), and the action it runs.
pub struct Command {
    pub label: &'static str,
    pub key: &'static str,
    pub action: Action,
}

const fn command(label: &'static str, key: &'static str, action: Action) -> Command {
    Command { label, key, action }
}

/// The palette's full command list, in default (unfiltered) order. Each key
/// is the one [`crate::keys`] binds to the same action, which a test holds
/// it to.
pub const COMMANDS: &[Command] = &[
    command("Go to Population", "1", Action::Page(Page::Population)),
    command("Go to Memory", "2", Action::Page(Page::Memory)),
    command("Go to Training", "3", Action::Page(Page::Loop)),
    command("Go to Evaluations", "4", Action::Page(Page::Evals)),
    command("Go to Agent", "5", Action::Page(Page::Sovereign)),
    command("Go to Services", "6", Action::Page(Page::Services)),
    command("Show the experts", "", Action::Focus(Focus::Experts)),
    command("Show the candidates", "", Action::Focus(Focus::Shadows)),
    command("Show the boundaries", "", Action::Focus(Focus::Boundaries)),
    command("Ask which expert would take a task", "a", Action::Ask),
    command("Router: how tasks are routed", "", Action::Gate),
    command("Find in this list", "/", Action::Filter),
    command(
        "Freeze or thaw the selected expert",
        "",
        Action::FreezeExpert,
    ),
    command(
        "Graduate the selected candidate",
        "",
        Action::GraduateShadow,
    ),
    command("Cancel the selected run's halt", "", Action::CancelHalt),
    command("Memory knobs", "K", Action::Knobs),
    command("Connect an agent", "c", Action::Connect),
    command(
        "History: what changed and what was done",
        "e",
        Action::Events,
    ),
    command(
        "Layout: list, one list beside its detail",
        "",
        Action::Layout(LayoutMode::List),
    ),
    command(
        "Layout: dashboard, all three lists at once",
        "",
        Action::Layout(LayoutMode::Dashboard),
    ),
    command(
        "Layout: graph, the experts alone",
        "",
        Action::Layout(LayoutMode::Graph),
    ),
    command(
        "Layout: table, the experts as a table",
        "",
        Action::Layout(LayoutMode::Table),
    ),
    command(
        "Sort the experts by the next column",
        "s",
        Action::CycleSort,
    ),
    command("Theme: change to the next", "t", Action::CycleTheme),
    command("Frame rate: follow the monitor", "0", Action::FollowMonitor),
    command("Frame rate: raise the cap", "+", Action::FpsUp),
    command("Frame rate: lower the cap", "-", Action::FpsDown),
    command("Reload now", "r", Action::Reload),
    command(
        "Help: every key, and what each word means",
        "?",
        Action::Help,
    ),
    command("Quit", "q", Action::Quit),
];

/// Case-insensitive subsequence score (lower is better): rewards an early first
/// match and few gaps between matched characters. `None` if `query` is not a
/// subsequence of `label`; an empty query matches everything at score 0.
pub fn fuzzy_score(label: &str, query: &str) -> Option<i32> {
    score_chars(label, &lowered(query))
}

/// A query as [`score_chars`] takes it: its characters, lowercased. Made once
/// for a query that is then scored against many labels.
pub fn lowered(query: &str) -> Vec<char> {
    query.chars().flat_map(char::to_lowercase).collect()
}

/// [`fuzzy_score`] for a query already [`lowered`]. It reads `label` a
/// character at a time and stops at the last character of the query, copying
/// nothing, so scoring a query against every memory of a large store costs
/// no more than reading them.
pub fn score_chars(label: &str, needle: &[char]) -> Option<i32> {
    let Some(mut want) = needle.first().copied() else {
        return Some(0);
    };
    let mut matched = 0usize;
    let mut first = 0i32;
    let mut prev: Option<usize> = None;
    let mut gaps = 0i32;
    for (i, c) in label.chars().flat_map(char::to_lowercase).enumerate() {
        if c != want {
            continue;
        }
        if matched == 0 {
            first = i32::try_from(i).unwrap_or(i32::MAX);
        }
        if prev.is_some_and(|p| i != p + 1) {
            gaps += 1;
        }
        prev = Some(i);
        matched += 1;
        match needle.get(matched) {
            Some(&next) => want = next,
            None => return Some(first.saturating_add(gaps * 3)),
        }
    }
    None
}

/// The commands matching `query`, best match first; ties keep registry order.
pub fn matches(query: &str) -> Vec<&'static Command> {
    let mut scored: Vec<(&'static Command, i32)> = COMMANDS
        .iter()
        .filter_map(|c| fuzzy_score(c.label, query).map(|s| (c, s)))
        .collect();
    scored.sort_by_key(|&(_, score)| score);
    scored.into_iter().map(|(c, _)| c).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fuzzy_matches_subsequences_case_insensitively() -> anyhow::Result<()> {
        assert!(fuzzy_score("Theme: change to the next", "theme").is_some());
        assert!(fuzzy_score("Theme: change to the next", "chg").is_some());
        assert!(fuzzy_score("Reload now", "RLD").is_some());
        // Out-of-order characters are not a subsequence.
        assert!(fuzzy_score("reload", "dler").is_none());
        // An empty query matches everything.
        assert_eq!(fuzzy_score("anything", ""), Some(0));
        Ok(())
    }

    #[test]
    fn a_lowered_query_scores_as_the_query_does() -> anyhow::Result<()> {
        let labels = [
            "Theme: change to the next",
            "prefer deno over node for new scripts",
            "İstanbul ĞÜŞ: letters whose lowercase is longer than they are",
            "",
        ];
        for label in labels {
            for query in ["", "the", "DENO", "node", "zzz", "ist", "ğüş"] {
                assert_eq!(
                    score_chars(label, &lowered(query)),
                    fuzzy_score(label, query),
                    "`{query}` in `{label}`"
                );
            }
        }
        // It stops at the query's last character: what follows is not read
        // for a match, and cannot change the score.
        let needle = lowered("deno");
        assert_eq!(
            score_chars("deno", &needle),
            score_chars(&format!("deno{}", " and more".repeat(1000)), &needle)
        );
        assert_eq!(score_chars("prefer deno", &needle), Some(7));
        assert_eq!(score_chars("d e n o", &needle), Some(9), "three gaps");
        Ok(())
    }

    #[test]
    fn closer_matches_score_lower() -> anyhow::Result<()> {
        // A contiguous match beats a gappy one of the same query.
        let Some(contiguous) = fuzzy_score("focus", "foc") else {
            anyhow::bail!("`foc` did not match `focus`");
        };
        let Some(gappy) = fuzzy_score("f o c", "foc") else {
            anyhow::bail!("`foc` did not match `f o c`");
        };
        assert!(contiguous < gappy, "{contiguous} should rank above {gappy}");
        Ok(())
    }

    #[test]
    fn matches_filters_and_ranks() -> anyhow::Result<()> {
        // The commands that say `frame` outright score best and lead the list
        // (a looser subsequence elsewhere may also match, but ranks lower).
        let frame = matches("frame");
        assert!(frame.len() >= 3, "the frame-rate commands surface");
        assert!(
            frame.iter().take(3).all(|c| c.label.contains("Frame rate")),
            "the frame-rate commands rank first: {:?}",
            frame.iter().map(|c| c.label).collect::<Vec<_>>()
        );
        // A distinctive query narrows to one command.
        let quit = matches("quit");
        assert_eq!(quit.len(), 1);
        assert_eq!(quit[0].action, Action::Quit);
        // An empty query returns the whole registry in order.
        assert_eq!(matches("").len(), COMMANDS.len());
        Ok(())
    }

    #[test]
    fn every_command_says_what_it_does_in_plain_words() -> anyhow::Result<()> {
        for command in COMMANDS {
            let lower = command.label.to_lowercase();
            for internal in ["umbra", "shadow", "substrate", "gate"] {
                assert!(
                    !lower.contains(internal),
                    "`{}` says `{internal}`",
                    command.label
                );
            }
            assert!(
                command.label.chars().next().is_some_and(char::is_uppercase),
                "`{}` starts a sentence",
                command.label
            );
        }
        // No two commands read alike.
        let labels: std::collections::HashSet<&str> = COMMANDS.iter().map(|c| c.label).collect();
        assert_eq!(labels.len(), COMMANDS.len());
        Ok(())
    }
}
