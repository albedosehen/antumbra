//! The console's words, in one place: what each thing on screen is called,
//! and the sentence that says what it is.
//!
//! They are the dashboard's words, so the two name one thing one way, and the
//! plain ones of `docs/antumbra-explained.md`. The project's internal names
//! for the same things (umbra, penumbra, antumbra, shadow, gate, substrate)
//! stay in the code and are not drawn. An id keeps its stored form, so a
//! candidate's still reads `shadow:…`.
//!
//! [`caption`] is the sentence under the tabs; [`GLOSSARY`] is the list the
//! help ends with.

use antumbra_core::Expert;

use crate::app::{App, Focus, Page};

/// A word the console uses, and what it means in a line or two.
pub struct Word {
    pub term: &'static str,
    /// The meaning, a line an entry; each fits the help's width.
    pub means: &'static [&'static str],
}

/// Every word the console uses that a newcomer would not know, in the order
/// they are met. The help lists them after the keys.
pub const GLOSSARY: &[Word] = &[
    Word {
        term: "expert",
        means: &[
            "a small model trained from your own verified work,",
            "to do one kind of task",
        ],
    },
    Word {
        term: "standing",
        means: &[
            "a standing expert is not sent tasks: it shapes every",
            "answer in its scope, a repository or everywhere",
        ],
    },
    Word {
        term: "fitness",
        means: &["how well an expert has done on checked work, 0 to 1"],
    },
    Word {
        term: "generation",
        means: &["which round of the training loop made it"],
    },
    Word {
        term: "frozen",
        means: &[
            "sealed, so later training cannot change",
            "what the expert learned",
        ],
    },
    Word {
        term: "candidate",
        means: &[
            "an expert in training. It graduates into an expert,",
            "or is pruned",
        ],
    },
    Word {
        term: "boundary",
        means: &[
            "where an expert was found to be wrong. In force,",
            "it keeps such tasks away from that expert",
        ],
    },
    Word {
        term: "router",
        means: &[
            "decides which expert takes a task, or that none does",
            "and your agent should",
        ],
    },
    Word {
        term: "memory",
        means: &[
            "what your agents stored: world is facts, bank is",
            "experiences, opinion is judgments",
        ],
    },
    Word {
        term: "training loop",
        means: &[
            "the cycle that finds weak spots, trains candidates,",
            "checks them and admits the winners",
        ],
    },
    Word {
        term: "evaluation",
        means: &[
            "one check of an expert or a candidate against a task.",
            "A frozen expert must keep passing",
        ],
    },
    Word {
        term: "operator",
        means: &[
            "a token that may act: prune, freeze, halt. Any other",
            "reads its own workspace and acts on nothing",
        ],
    },
];

/// The sentence under the tabs: what the page, or the list in front on the
/// population page, is.
pub fn caption(app: &App) -> &'static str {
    match app.page {
        Page::Population => match app.focus {
            Focus::Experts => {
                "Experts: small models trained from your own verified work, each for one kind of task."
            }
            Focus::Shadows => {
                "Candidates: experts in training. One that passes its checks graduates; one that does not is pruned."
            }
            Focus::Boundaries => {
                "Boundaries: where an expert was found to be wrong. One in force keeps such tasks away from it."
            }
        },
        Page::Memory => {
            "Memory: what your agents stored. World is facts, bank is experiences, opinion is judgments."
        }
        Page::Loop => {
            "Training: the loop that finds weak spots, trains candidates, checks them and admits the winners."
        }
        Page::Evals => {
            "Evaluations: each check of an expert or a candidate against a task. A frozen expert must keep passing."
        }
        Page::Sovereign => {
            "Agent: the rules your coding agent is given in place of its vendor's feature flags, and the skills it uses."
        }
        Page::Services => {
            "Services: what depends on what. Pick a service to see what a change to it reaches, and what says so."
        }
    }
}

/// What a list says when it is empty because the token may not read it.
/// Candidates, boundaries, the training loop, evaluations and the router
/// belong to the server, not to one workspace, so only an operator's token
/// reads them; without this a read-only console would look like a server
/// where nothing had ever run.
pub const OPERATOR_ONLY: &str = "Only an operator's token reads this: it belongs to the server, not to one workspace. Yours reads your workspace's experts and memory. An operator's is minted with `antumbra-mcp --mint-token --operator`.";

/// What a read-only console says when asked to act.
pub const READ_ONLY: &str =
    "Read-only: acting needs an operator's token (antumbra-mcp --mint-token --operator).";

/// A scope in brief: a repository's, such as `github.com/acme/orders`, by the
/// repository's name, the last part of its path; any other, such as
/// `everywhere`, as it is.
pub fn scope_name(scope: &str) -> &str {
    scope
        .rsplit('/')
        .next()
        .filter(|name| !name.is_empty())
        .unwrap_or(scope)
}

/// An expert's name as the console shows it. A standing expert's stored name
/// is its id, some forty characters of prefix before the one part that tells
/// it from the next, so a list of them is a column of identical rows; it is
/// named by its scope instead, the repository's name or `everywhere`. Any
/// other expert has a name of its own.
pub fn expert_name(e: &Expert) -> String {
    match e.standing_scope() {
        Some(scope) => scope_name(scope).to_string(),
        None => e.name.clone(),
    }
}

/// What kind of expert `e` is, in a word: `standing`, which shapes every
/// answer in its scope, or `routed`, which is sent the tasks it covers.
pub fn expert_kind(e: &Expert) -> &'static str {
    if e.standing_scope().is_some() {
        "standing"
    } else {
        "routed"
    }
}

/// A memory's kind, as its network and what that holds.
pub fn network(network: &str) -> &'static str {
    match network {
        "world" => "world · a fact",
        "bank" => "bank · an experience",
        "opinion" => "opinion · a judgment",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The widest a glossary line may be: the help's inner width, less the
    /// term's column.
    const ROOM: usize = 56;

    #[test]
    fn every_glossary_line_fits_the_help() -> anyhow::Result<()> {
        for word in GLOSSARY {
            assert!(!word.means.is_empty(), "{} means nothing", word.term);
            assert!(word.term.chars().count() <= 14, "{} is too wide", word.term);
            for line in word.means {
                assert!(
                    line.chars().count() <= ROOM,
                    "{}: `{line}` is {} columns",
                    word.term,
                    line.chars().count()
                );
            }
        }
        Ok(())
    }

    #[test]
    fn no_word_the_console_draws_is_an_internal_one() -> anyhow::Result<()> {
        // The glossary, the captions and the shared phrases are all plain.
        let mut drawn: Vec<&str> = GLOSSARY
            .iter()
            .flat_map(|w| std::iter::once(w.term).chain(w.means.iter().copied()))
            .collect();
        drawn.push(OPERATOR_ONLY);
        drawn.push(READ_ONLY);
        for text in drawn {
            // The product's own name, as the command a line names, is not
            // one of them.
            let lower = text.to_lowercase().replace("antumbra", "");
            for internal in ["umbra", "shadow", "substrate", " gate"] {
                assert!(!lower.contains(internal), "`{text}` says `{internal}`");
            }
        }
        Ok(())
    }
}
