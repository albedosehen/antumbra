//! The command palette's vocabulary: the actions it can run and a small
//! hand-rolled fuzzy matcher (subsequence + gap scoring) that ranks them against
//! the typed query. Keybindings and the palette both resolve to an [`Action`] the
//! loop applies in one place.

use crate::app::{Focus, LayoutMode, Page};

/// Something the console can do, triggered by a key or chosen from the palette.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Action {
    Reload,
    CycleTheme,
    CycleSort,
    Page(Page),
    Focus(Focus),
    Layout(LayoutMode),
    FpsUp,
    FpsDown,
    FollowMonitor,
    Ask,
    Events,
    GraduateShadow,
    FreezeExpert,
    Help,
    Quit,
}

/// A palette entry: the searchable label and the action it runs.
pub struct Command {
    pub label: &'static str,
    pub action: Action,
}

/// The palette's full command list, in default (unfiltered) order.
pub const COMMANDS: &[Command] = &[
    Command {
        label: "page · population",
        action: Action::Page(Page::Population),
    },
    Command {
        label: "page · memory networks",
        action: Action::Page(Page::Memory),
    },
    Command {
        label: "page · generational loop",
        action: Action::Page(Page::Loop),
    },
    Command {
        label: "page · evaluations",
        action: Action::Page(Page::Evals),
    },
    Command {
        label: "focus umbra · experts",
        action: Action::Focus(Focus::Experts),
    },
    Command {
        label: "focus penumbra · shadows",
        action: Action::Focus(Focus::Shadows),
    },
    Command {
        label: "focus antumbra · boundaries",
        action: Action::Focus(Focus::Boundaries),
    },
    Command {
        label: "theme · cycle palette",
        action: Action::CycleTheme,
    },
    Command {
        label: "layout · dashboard (all regions)",
        action: Action::Layout(LayoutMode::Dashboard),
    },
    Command {
        label: "layout · focused (single detail)",
        action: Action::Layout(LayoutMode::Focused),
    },
    Command {
        label: "layout · graph (full width)",
        action: Action::Layout(LayoutMode::Graph),
    },
    Command {
        label: "layout · table (sortable experts)",
        action: Action::Layout(LayoutMode::Table),
    },
    Command {
        label: "sort · cycle the table column",
        action: Action::CycleSort,
    },
    Command {
        label: "fps · follow active monitor",
        action: Action::FollowMonitor,
    },
    Command {
        label: "fps · increase cap",
        action: Action::FpsUp,
    },
    Command {
        label: "fps · decrease cap",
        action: Action::FpsDown,
    },
    Command {
        label: "ask · route a task through the gate",
        action: Action::Ask,
    },
    Command {
        label: "events · live store changes",
        action: Action::Events,
    },
    Command {
        label: "graduate the selected shadow",
        action: Action::GraduateShadow,
    },
    Command {
        label: "freeze · thaw the selected expert",
        action: Action::FreezeExpert,
    },
    Command {
        label: "reload from the store",
        action: Action::Reload,
    },
    Command {
        label: "help · keybindings",
        action: Action::Help,
    },
    Command {
        label: "quit",
        action: Action::Quit,
    },
];

/// Case-insensitive subsequence score (lower is better): rewards an early first
/// match and few gaps between matched characters. `None` if `query` is not a
/// subsequence of `label`; an empty query matches everything at score 0.
pub fn fuzzy_score(label: &str, query: &str) -> Option<i32> {
    if query.is_empty() {
        return Some(0);
    }
    let haystack: Vec<char> = label.to_lowercase().chars().collect();
    let mut needle = query.to_lowercase().chars().collect::<Vec<_>>().into_iter();
    let mut want = needle.next();
    let mut first: Option<i32> = None;
    let mut prev: Option<usize> = None;
    let mut gaps = 0i32;
    for (i, &c) in haystack.iter().enumerate() {
        match want {
            Some(qc) if qc == c => {
                if first.is_none() {
                    first = Some(i as i32);
                }
                if let Some(p) = prev {
                    if i != p + 1 {
                        gaps += 1;
                    }
                }
                prev = Some(i);
                want = needle.next();
            }
            _ => {}
        }
    }
    match want {
        Some(_) => None,
        None => Some(first.unwrap_or(0) + gaps * 3),
    }
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
    fn fuzzy_matches_subsequences_case_insensitively() {
        assert!(fuzzy_score("theme · cycle palette", "theme").is_some());
        assert!(fuzzy_score("theme · cycle palette", "cyc").is_some());
        assert!(fuzzy_score("reload from the store", "RLD").is_some());
        // Out-of-order characters are not a subsequence.
        assert!(fuzzy_score("reload", "dler").is_none());
        // An empty query matches everything.
        assert_eq!(fuzzy_score("anything", ""), Some(0));
    }

    #[test]
    fn closer_matches_score_lower() {
        // A contiguous match beats a gappy one of the same query.
        let contiguous = fuzzy_score("focus", "foc").unwrap();
        let gappy = fuzzy_score("f o c", "foc").unwrap();
        assert!(contiguous < gappy, "{contiguous} should rank above {gappy}");
    }

    #[test]
    fn matches_filters_and_ranks() {
        // The literal `fps` commands score best and lead the list (a fuzzy
        // subsequence like "focus penumbra · shadows" may also match, but ranks
        // lower).
        let fps = matches("fps");
        assert!(fps.len() >= 3, "the fps commands surface for 'fps'");
        assert!(
            fps.iter().take(3).all(|c| c.label.contains("fps")),
            "the literal fps commands rank first: {:?}",
            fps.iter().map(|c| c.label).collect::<Vec<_>>()
        );
        // A distinctive query narrows to one command.
        let quit = matches("quit");
        assert_eq!(quit.len(), 1);
        assert_eq!(quit[0].action, Action::Quit);
        // An empty query returns the whole registry in order.
        assert_eq!(matches("").len(), COMMANDS.len());
    }
}
