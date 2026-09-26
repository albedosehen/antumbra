//! The knowledge diff (ADR-0019): what a pull request touches among the
//! memories and documents anchored to its repository, posted as a check run so
//! the knowledge is in front of the reviewer at review time.
//!
//! Three things tie knowledge to a change:
//! - **a memory's path:** anchored to a file the change adds to, modifies,
//!   renames or removes;
//! - **a document's file:** a knowledge document is titled by the repository
//!   and path it was ingested from, so the change touches the one ingested
//!   from a file it changes;
//! - **a memory's branch:** anchored to the branch the pull request merges.
//!   Those are re-anchored to the base when it merges, and orphaned if the
//!   branch is deleted without merging.
//!
//! The diff says nothing about what a change contradicts: that takes judging
//! meaning, and a check run that guessed would teach reviewers to ignore it.

use std::collections::BTreeMap;

use antumbra_core::{normalize_repo, GitProvenance, Memory};

use crate::api::ChangedFile;

/// Paths the output lists before it says how many more there are.
const MAX_PATHS: usize = 40;
/// Memories listed under one path, or on the branch.
const MAX_PER_GROUP: usize = 15;
/// How much of a memory's text a line shows.
const SUMMARY_CHARS: usize = 140;

/// One memory the diff names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Named {
    pub id: String,
    /// Its text on one line, cut short.
    pub summary: String,
}

/// What a change touches at one path.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AtPath {
    /// The knowledge document ingested from the file, by title, when there is
    /// one.
    pub document: Option<String>,
    /// The memories anchored to the file.
    pub memories: Vec<Named>,
}

/// What a pull request touches.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KnowledgeDiff {
    /// By changed path, in path order.
    pub touched: BTreeMap<String, AtPath>,
    /// Memories anchored to the head branch.
    pub on_branch: Vec<Named>,
}

/// A check run's output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckOutput {
    pub title: String,
    pub summary: String,
    pub text: String,
}

fn named(m: &Memory) -> Named {
    let line = m.content.split_whitespace().collect::<Vec<_>>().join(" ");
    let summary = if line.chars().count() > SUMMARY_CHARS {
        let cut: String = line.chars().take(SUMMARY_CHARS - 1).collect();
        format!("{cut}…")
    } else {
        line
    };
    Named {
        id: m.id.as_str().to_string(),
        summary,
    }
}

impl KnowledgeDiff {
    /// The diff for a pull request on `repo` from `head_branch`, which changed
    /// `changed`, over the workspace's `memories` and its knowledge documents'
    /// `titles`.
    pub fn of(
        repo: &str,
        head_branch: &str,
        changed: &[ChangedFile],
        memories: &[Memory],
        titles: &[String],
    ) -> Self {
        let repo = normalize_repo(repo);
        let mut paths: Vec<&str> = changed.iter().map(|f| f.path.as_str()).collect();
        paths.extend(changed.iter().filter_map(|f| f.previous_path.as_deref()));
        let mut diff = KnowledgeDiff::default();
        for path in &paths {
            let title = format!("{repo}:{path}");
            if titles.iter().any(|t| normalize_repo(t) == title) {
                diff.touched
                    .entry((*path).to_string())
                    .or_default()
                    .document = Some(title);
            }
        }
        for m in memories {
            let Some(anchor) = GitProvenance::from_evidence(&m.evidence) else {
                continue;
            };
            if normalize_repo(&anchor.repo) != repo {
                continue;
            }
            if let Some(path) = anchor.path.as_deref().filter(|p| paths.contains(p)) {
                diff.touched
                    .entry(path.to_string())
                    .or_default()
                    .memories
                    .push(named(m));
            }
            if anchor.branch.as_deref() == Some(head_branch) {
                diff.on_branch.push(named(m));
            }
        }
        diff
    }

    pub fn is_empty(&self) -> bool {
        self.touched.is_empty() && self.on_branch.is_empty()
    }

    fn touched_count(&self) -> usize {
        self.touched
            .values()
            .map(|at| at.memories.len() + usize::from(at.document.is_some()))
            .sum()
    }

    /// The check run's output. `head` and `base` are the pull request's
    /// branches.
    pub fn render(&self, head: &str, base: &str) -> CheckOutput {
        let (touched, on_branch) = (self.touched_count(), self.on_branch.len());
        let title = if self.is_empty() {
            "No anchored knowledge touches this change".to_string()
        } else {
            let n = touched + on_branch;
            format!(
                "{n} anchored memor{} and document{} touch this change",
                if n == 1 { "y" } else { "ies" },
                if n == 1 { "" } else { "s" }
            )
        };
        let summary = format!(
            "- **{touched}** memories and documents anchored to **{}** of the changed paths\n\
             - **{on_branch}** anchored to `{head}`: re-anchored to `{base}` when this merges, \
             orphaned if the branch is deleted without merging",
            self.touched.len()
        );
        let mut text = String::new();
        if !self.touched.is_empty() {
            text.push_str("## Anchored to the changed paths\n");
            for (path, at) in self.touched.iter().take(MAX_PATHS) {
                text.push_str(&format!("\n### `{path}`\n\n"));
                if let Some(title) = &at.document {
                    text.push_str(&format!("- the knowledge document `{title}`\n"));
                }
                push_group(&mut text, &at.memories);
            }
            if self.touched.len() > MAX_PATHS {
                text.push_str(&format!(
                    "\n…and {} more path(s).\n",
                    self.touched.len() - MAX_PATHS
                ));
            }
        }
        if !self.on_branch.is_empty() {
            text.push_str(&format!("\n## Anchored to `{head}`\n\n"));
            push_group(&mut text, &self.on_branch);
        }
        CheckOutput {
            title,
            summary,
            text,
        }
    }
}

fn push_group(text: &mut String, memories: &[Named]) {
    for m in memories.iter().take(MAX_PER_GROUP) {
        text.push_str(&format!("- {} (`{}`)\n", m.summary, m.id));
    }
    if memories.len() > MAX_PER_GROUP {
        text.push_str(&format!("- …and {} more\n", memories.len() - MAX_PER_GROUP));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antumbra_core::MemoryNetwork;
    use chrono::Utc;

    fn memory(id: &str, content: &str, anchor: Option<GitProvenance>) -> Memory {
        let evidence = anchor.map(|a| vec![a.to_evidence()]).unwrap_or_default();
        Memory::new(
            id,
            "ws:acme",
            MemoryNetwork::World,
            content,
            0.7,
            Utc::now(),
        )
        .with_evidence(evidence)
    }

    fn changed(path: &str, status: &str, previous: Option<&str>) -> ChangedFile {
        ChangedFile {
            path: path.into(),
            status: status.into(),
            previous_path: previous.map(str::to_string),
        }
    }

    fn at(path: Option<&str>, branch: Option<&str>) -> GitProvenance {
        let mut p = GitProvenance::new("github.com/acme/orders", "abc1234");
        if let Some(path) = path {
            p = p.at_path(path);
        }
        if let Some(branch) = branch {
            p = p.on_branch(branch);
        }
        p
    }

    #[test]
    fn knowledge_is_named_by_the_path_or_branch_that_ties_it_to_the_change() {
        let memories = vec![
            memory(
                "memory:retry",
                "retries  back off\nexponentially",
                Some(at(Some("src/retry.rs"), Some("main"))),
            ),
            memory(
                "memory:old",
                "the config loader lives here",
                Some(at(Some("src/config_old.rs"), None)),
            ),
            memory(
                "memory:wip",
                "the new queue drops duplicates",
                Some(at(None, Some("feat/queue"))),
            ),
            memory(
                "memory:elsewhere",
                "unrelated",
                Some(at(Some("src/other.rs"), Some("main"))),
            ),
            memory(
                "memory:other-repo",
                "same path, other repository",
                Some(
                    GitProvenance::new("github.com/acme/billing", "def5678")
                        .at_path("src/retry.rs"),
                ),
            ),
            memory("memory:unanchored", "no anchor", None),
        ];
        let files = vec![
            changed("src/retry.rs", "modified", None),
            changed("src/config.rs", "renamed", Some("src/config_old.rs")),
            changed("docs/retry.md", "modified", None),
        ];
        let titles = vec![
            "github.com/acme/orders:docs/retry.md".to_string(),
            "github.com/acme/billing:docs/retry.md".to_string(),
        ];
        let diff = KnowledgeDiff::of(
            "github.com/acme/orders",
            "feat/queue",
            &files,
            &memories,
            &titles,
        );
        let touched: Vec<(&str, Option<&str>, Vec<&str>)> = diff
            .touched
            .iter()
            .map(|(p, at)| {
                (
                    p.as_str(),
                    at.document.as_deref(),
                    at.memories.iter().map(|m| m.id.as_str()).collect(),
                )
            })
            .collect();
        assert_eq!(
            touched,
            vec![
                (
                    "docs/retry.md",
                    Some("github.com/acme/orders:docs/retry.md"),
                    vec![]
                ),
                ("src/config_old.rs", None, vec!["memory:old"]),
                ("src/retry.rs", None, vec!["memory:retry"]),
            ]
        );
        assert_eq!(diff.on_branch.len(), 1);
        assert_eq!(diff.on_branch[0].id, "memory:wip");
        assert_eq!(
            diff.touched["src/retry.rs"].memories[0].summary,
            "retries back off exponentially"
        );
    }

    #[test]
    fn the_output_counts_both_and_says_what_merging_does() {
        let memories = vec![
            memory("memory:a", "a", Some(at(Some("src/a.rs"), None))),
            memory("memory:b", "b", Some(at(None, Some("feat/x")))),
        ];
        let diff = KnowledgeDiff::of(
            "github.com/acme/orders",
            "feat/x",
            &[changed("src/a.rs", "modified", None)],
            &memories,
            &["github.com/acme/orders:src/a.rs".to_string()],
        );
        let out = diff.render("feat/x", "main");
        assert_eq!(
            out.title,
            "3 anchored memories and documents touch this change"
        );
        assert!(out
            .summary
            .contains("**2** memories and documents anchored to **1**"));
        assert!(out
            .summary
            .contains("re-anchored to `main` when this merges"));
        assert!(out.text.contains("### `src/a.rs`"));
        assert!(out
            .text
            .contains("- the knowledge document `github.com/acme/orders:src/a.rs`"));
        assert!(out.text.contains("- a (`memory:a`)"));
        assert!(out.text.contains("## Anchored to `feat/x`"));

        let none = KnowledgeDiff::default().render("feat/x", "main");
        assert_eq!(none.title, "No anchored knowledge touches this change");
        assert!(none.text.is_empty());
    }

    #[test]
    fn long_lists_are_cut_and_say_how_much_was_left_out() {
        let memories: Vec<Memory> = (0..20)
            .map(|i| {
                memory(
                    &format!("memory:{i}"),
                    &"x".repeat(300),
                    Some(at(Some("src/a.rs"), None)),
                )
            })
            .collect();
        let diff = KnowledgeDiff::of(
            "github.com/acme/orders",
            "feat/x",
            &[changed("src/a.rs", "modified", None)],
            &memories,
            &[],
        );
        let out = diff.render("feat/x", "main");
        assert!(out.text.contains("- …and 5 more"));
        assert!(diff.touched["src/a.rs"].memories[0].summary.chars().count() <= SUMMARY_CHARS);
    }
}
