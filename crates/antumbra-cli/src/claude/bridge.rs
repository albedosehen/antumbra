//! The bridge: how an `AGENTS.md` gets read again in sovereign mode (ADR-0021).
//!
//! The first plan was to supply the file through the session-start hook. A
//! hook's context is capped at 10,000 characters, and past the cap the agent is
//! handed a file path and a 2,000-character preview it is never asked to open:
//! a real instruction file (the first one measured was 11,468 characters) would
//! have been cut to a fifth, silently. Hook context is also a system reminder and
//! not project instructions, does not survive compaction, and does not reach
//! subagents.
//!
//! A `CLAUDE.local.md` that imports the file has none of those problems. It is
//! read natively, with no cap short of the agent's own 4 MiB, in the working
//! directory, above it, and in subdirectories as files there are read; it is
//! re-read from disk after compaction; subagents load it like any project
//! instruction. It is also the file the vendor designates for instructions that
//! are *not* committed, so with one line in `.git/info/exclude` (itself
//! untracked) the bridge changes nothing a repository tracks, which makes it safe
//! in a checkout the user does not own.
//!
//! The planner below is a function of what is on disk, supplied as a `read`
//! closure, so it is table-tested without touching any.

use std::path::{Path, PathBuf};

/// Marks a `CLAUDE.local.md` as ours: written by this command, safe to delete,
/// and transparent when deciding what the agent would natively have read.
pub const MARKER: &str =
    "<!-- antumbra: bridge to AGENTS.md (ADR-0021). Untracked; safe to delete. -->";

/// The file a bridge is written to, beside the `AGENTS.md` it imports.
pub const BRIDGE_FILE: &str = "CLAUDE.local.md";

/// The files that are a project's own instructions. One of them in scope means
/// the agent was never going to read `AGENTS.md` there, flags or no flags.
const CLAUDE_FILES: [&str; 3] = ["CLAUDE.md", ".claude/CLAUDE.md", "CLAUDE.local.md"];

/// Where an `AGENTS.md` may sit in a directory, as an import path from it.
const AGENTS_FILES: [&str; 2] = ["AGENTS.md", ".claude/AGENTS.md"];

/// The whole text of a bridge that imports `import` (a path relative to it).
pub fn bridge_text(import: &str) -> String {
    format!("{MARKER}\n@{import}\n")
}

/// Whether `text` is a bridge this command wrote.
pub fn is_bridge(text: &str) -> bool {
    text.contains(MARKER)
}

/// What to do about one directory's `AGENTS.md`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    /// Write a bridge here.
    Create { bridge: PathBuf, import: String },
    /// A bridge we wrote is already here.
    Present { bridge: PathBuf },
    /// Leave it: the agent would not have read this `AGENTS.md` natively either,
    /// because the project has instructions of its own in `because`. Bridging it
    /// would make sovereign mode behave differently from the default it mirrors.
    Native { agents: PathBuf, because: PathBuf },
}

/// A directory's own instruction file, ignoring a bridge we wrote.
fn own_instructions(dir: &Path, read: &dyn Fn(&Path) -> Option<String>) -> Option<PathBuf> {
    CLAUDE_FILES
        .iter()
        .map(|name| dir.join(name))
        .find(|path| read(path).is_some_and(|text| !is_bridge(&text)))
}

/// The `AGENTS.md` in `dir`, as (its path, the import path from `dir`).
fn agents_in(
    dir: &Path,
    read: &dyn Fn(&Path) -> Option<String>,
) -> Option<(PathBuf, &'static str)> {
    AGENTS_FILES
        .iter()
        .map(|name| (dir.join(name), *name))
        .find(|(path, _)| read(path).is_some())
}

fn bridge_or_create(dir: &Path, import: &str, read: &dyn Fn(&Path) -> Option<String>) -> Action {
    let bridge = dir.join(BRIDGE_FILE);
    match read(&bridge) {
        Some(text) if is_bridge(&text) => Action::Present { bridge },
        _ => Action::Create {
            bridge,
            import: import.to_string(),
        },
    }
}

/// Decide, for a session started in `start` inside the repository at `root`, what
/// each `AGENTS.md` needs. `below` is every directory under `start` that holds
/// one (the edge finds them; the planner only judges).
///
/// This mirrors the agent's documented default. At or above the working
/// directory it reads every `AGENTS.md`, provided no `CLAUDE.md` of any kind
/// sits at or above the working directory. In a subdirectory it reads that
/// directory's `AGENTS.md`, provided that directory has no `CLAUDE.md` of its
/// own. Nothing above the repository is touched.
pub fn plan(
    root: &Path,
    start: &Path,
    below: &[PathBuf],
    read: &dyn Fn(&Path) -> Option<String>,
) -> Vec<Action> {
    let chain: Vec<&Path> = start
        .ancestors()
        .take_while(|dir| dir.starts_with(root))
        .collect();
    let chain_instructions = chain.iter().find_map(|dir| own_instructions(dir, read));

    let above = chain.iter().filter_map(|dir| {
        let (agents, import) = agents_in(dir, read)?;
        Some(match &chain_instructions {
            Some(because) => Action::Native {
                agents,
                because: because.clone(),
            },
            None => bridge_or_create(dir, import, read),
        })
    });
    let under = below
        .iter()
        .filter(|dir| dir.starts_with(start) && dir.as_path() != start)
        .filter_map(|dir| {
            let (agents, import) = agents_in(dir, read)?;
            Some(match own_instructions(dir, read) {
                Some(because) => Action::Native { agents, because },
                None => bridge_or_create(dir, import, read),
            })
        });
    above.chain(under).collect()
}

/// Where the working directory stands, for the doctor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Instructions {
    /// An `AGENTS.md` the agent would have read natively and now will not.
    Unread(PathBuf),
    /// Every `AGENTS.md` in scope is imported by a bridge.
    Bridged(PathBuf),
    /// Nothing here depends on the lost feature.
    NotApplicable,
}

/// Judge the chain at and above `start` (what loads at launch).
pub fn standing(root: &Path, start: &Path, read: &dyn Fn(&Path) -> Option<String>) -> Instructions {
    let actions = plan(root, start, &[], read);
    let unread = actions.iter().find_map(|action| match action {
        Action::Create { bridge, import } => bridge.parent().map(|dir| dir.join(import)),
        _ => None,
    });
    let bridged = actions.iter().find_map(|action| match action {
        Action::Present { bridge } => Some(bridge.clone()),
        _ => None,
    });
    match (unread, bridged) {
        (Some(path), _) => Instructions::Unread(path),
        (None, Some(path)) => Instructions::Bridged(path),
        (None, None) => Instructions::NotApplicable,
    }
}

/// Directories never worth searching for an `AGENTS.md`.
const SKIPPED_DIRS: [&str; 7] = [
    ".git",
    "node_modules",
    "target",
    "vendor",
    "dist",
    "build",
    ".venv",
];

/// How deep below the working directory to look. Instruction files live near the
/// top of a tree; this bounds a walk over a very large one.
const MAX_DEPTH: usize = 6;

/// Every directory under `start` that holds an `AGENTS.md`. Touches the disk.
pub fn directories_with_agents(start: &Path) -> Vec<PathBuf> {
    fn walk(dir: &Path, depth: usize, found: &mut Vec<PathBuf>) {
        if AGENTS_FILES.iter().any(|name| dir.join(name).is_file()) {
            found.push(dir.to_path_buf());
        }
        if depth == 0 {
            return;
        }
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let skipped = path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| SKIPPED_DIRS.contains(&name));
            if path.is_dir() && !skipped {
                walk(&path, depth - 1, found);
            }
        }
    }
    let mut found = Vec::new();
    walk(start, MAX_DEPTH, &mut found);
    found.sort();
    found
}

/// The line that keeps bridges out of `git status`, and whether `exclude`
/// already carries it.
pub const EXCLUDE_LINE: &str = "CLAUDE.local.md";

pub fn excludes_bridges(exclude: &str) -> bool {
    exclude.lines().any(|line| {
        let line = line.trim();
        line == EXCLUDE_LINE || line == "/CLAUDE.local.md" || line == "**/CLAUDE.local.md"
    })
}

/// `exclude` with the line added, keeping whatever was there.
pub fn with_exclude_line(exclude: &str) -> String {
    let separator = if exclude.is_empty() || exclude.ends_with('\n') {
        ""
    } else {
        "\n"
    };
    format!(
        "{exclude}{separator}# antumbra: bridges to AGENTS.md are local, never committed (ADR-0021)\n{EXCLUDE_LINE}\n"
    )
}

/// Whether a `CLAUDE.local.md` is a bridge and nothing else: the marker, import
/// lines, and blanks. Only such a file is ever deleted. One the user has added to
/// is theirs now, and is left alone.
pub fn is_only_a_bridge(text: &str) -> bool {
    is_bridge(text)
        && text
            .lines()
            .map(str::trim)
            .all(|line| line.is_empty() || line == MARKER || line.starts_with('@'))
}

fn git(project: &Path, args: &[&str]) -> Option<String> {
    std::process::Command::new("git")
        .arg("-C")
        .arg(project)
        .args(args)
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Keep bridges out of `git status` without touching anything the repository
/// tracks: `.git/info/exclude` is local to the clone. Nothing to do when the
/// repository already ignores the file, or when this is not a repository.
fn exclude_bridges(project: &Path, dry_run: bool) -> anyhow::Result<Option<String>> {
    let ignored = std::process::Command::new("git")
        .arg("-C")
        .arg(project)
        .args(["check-ignore", "-q", BRIDGE_FILE])
        .status()
        .is_ok_and(|status| status.success());
    if ignored {
        return Ok(None);
    }
    let Some(relative) = git(project, &["rev-parse", "--git-path", "info/exclude"]) else {
        return Ok(None);
    };
    let path = project.join(relative);
    let current = std::fs::read_to_string(&path).unwrap_or_default();
    if excludes_bridges(&current) {
        return Ok(None);
    }
    if !dry_run {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(&path, with_exclude_line(&current))?;
    }
    Ok(Some(format!(
        "exclude  {} now lists {EXCLUDE_LINE} (local to this clone, not tracked)",
        path.display()
    )))
}

/// Bridge every `AGENTS.md` the agent would have read for a session started in
/// `project`, and say what was done, one line each. Writes only `CLAUDE.local.md`
/// files and the clone's own exclude file; with `dry_run`, writes nothing.
pub fn write_bridges(root: &Path, project: &Path, dry_run: bool) -> anyhow::Result<Vec<String>> {
    let read = |path: &Path| std::fs::read_to_string(path).ok();
    let actions = plan(root, project, &directories_with_agents(project), &read);
    let mut said = Vec::new();
    for action in &actions {
        said.push(match action {
            Action::Create { bridge, import } => {
                if !dry_run {
                    std::fs::write(bridge, bridge_text(import))?;
                }
                format!("bridge   {} imports {import}", bridge.display())
            }
            Action::Present { bridge } => format!("present  {}", bridge.display()),
            Action::Native { agents, because } => format!(
                "skipped  {}: the agent would not have read it either, since {} is the project's own instructions",
                agents.display(),
                because.display()
            ),
        });
    }
    let wrote = actions
        .iter()
        .any(|a| matches!(a, Action::Create { .. } | Action::Present { .. }));
    if wrote {
        said.extend(exclude_bridges(project, dry_run)?);
    }
    if actions.is_empty() {
        said.push("nothing to bridge: no AGENTS.md at, above, or below this directory".into());
    }
    Ok(said)
}

/// Take the bridges back out. Deletes a `CLAUDE.local.md` only when it is a
/// bridge and nothing else.
pub fn remove_bridges(root: &Path, project: &Path, dry_run: bool) -> anyhow::Result<Vec<String>> {
    let candidates = project
        .ancestors()
        .take_while(|dir| dir.starts_with(root))
        .map(Path::to_path_buf)
        .chain(directories_with_agents(project))
        .map(|dir| dir.join(BRIDGE_FILE));
    let mut seen = std::collections::BTreeSet::new();
    let mut said = Vec::new();
    for bridge in candidates.filter(|path| seen.insert(path.clone())) {
        let Ok(text) = std::fs::read_to_string(&bridge) else {
            continue;
        };
        if is_only_a_bridge(&text) {
            if !dry_run {
                std::fs::remove_file(&bridge)?;
            }
            said.push(format!("removed  {}", bridge.display()));
        } else if is_bridge(&text) {
            said.push(format!(
                "kept     {}: it holds more than the bridge now, so it is yours",
                bridge.display()
            ));
        }
    }
    if said.is_empty() {
        said.push("no bridges here".into());
    }
    Ok(said)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    type Disk = BTreeMap<PathBuf, String>;

    fn disk(files: &[(&str, &str)]) -> Disk {
        files
            .iter()
            .map(|(path, text)| (PathBuf::from(path), text.to_string()))
            .collect()
    }

    fn planned(files: &[(&str, &str)], start: &str, below: &[&str]) -> Vec<Action> {
        let disk = disk(files);
        let below: Vec<PathBuf> = below.iter().map(PathBuf::from).collect();
        plan(Path::new("/repo"), Path::new(start), &below, &|path| {
            disk.get(path).cloned()
        })
    }

    fn create(bridge: &str, import: &str) -> Action {
        Action::Create {
            bridge: PathBuf::from(bridge),
            import: import.to_string(),
        }
    }

    fn native(agents: &str, because: &str) -> Action {
        Action::Native {
            agents: PathBuf::from(agents),
            because: PathBuf::from(because),
        }
    }

    /// The agent's documented default, row by row.
    #[test]
    fn it_bridges_exactly_what_the_agent_would_have_read() {
        for (files, start, below, expected) in [
            // An AGENTS.md alone: bridged beside itself.
            (
                vec![("/repo/AGENTS.md", "rules")],
                "/repo",
                vec![],
                vec![create("/repo/CLAUDE.local.md", "AGENTS.md")],
            ),
            // Started in a subdirectory: every one at or above is read.
            (
                vec![
                    ("/repo/AGENTS.md", "a"),
                    ("/repo/crates/core/AGENTS.md", "b"),
                ],
                "/repo/crates/core",
                vec![],
                vec![
                    create("/repo/crates/core/CLAUDE.local.md", "AGENTS.md"),
                    create("/repo/CLAUDE.local.md", "AGENTS.md"),
                ],
            ),
            // The `.claude/` form imports by that path.
            (
                vec![("/repo/.claude/AGENTS.md", "a")],
                "/repo",
                vec![],
                vec![create("/repo/CLAUDE.local.md", ".claude/AGENTS.md")],
            ),
            // A CLAUDE.md at or above: the agent never read AGENTS.md here.
            (
                vec![("/repo/AGENTS.md", "a"), ("/repo/CLAUDE.md", "mine")],
                "/repo",
                vec![],
                vec![native("/repo/AGENTS.md", "/repo/CLAUDE.md")],
            ),
            // So does the user's own CLAUDE.local.md, which is left exactly as it is.
            (
                vec![
                    ("/repo/AGENTS.md", "a"),
                    ("/repo/CLAUDE.local.md", "my notes"),
                ],
                "/repo",
                vec![],
                vec![native("/repo/AGENTS.md", "/repo/CLAUDE.local.md")],
            ),
            // A subdirectory's file, judged by that directory alone.
            (
                vec![("/repo/AGENTS.md", "a"), ("/repo/web/AGENTS.md", "b")],
                "/repo",
                vec!["/repo/web"],
                vec![
                    create("/repo/CLAUDE.local.md", "AGENTS.md"),
                    create("/repo/web/CLAUDE.local.md", "AGENTS.md"),
                ],
            ),
            (
                vec![
                    ("/repo/web/AGENTS.md", "b"),
                    ("/repo/web/CLAUDE.md", "mine"),
                ],
                "/repo",
                vec!["/repo/web"],
                vec![native("/repo/web/AGENTS.md", "/repo/web/CLAUDE.md")],
            ),
            // Names the agent never reads are not instructions.
            (
                vec![
                    ("/repo/AGENTS.local.md", "a"),
                    ("/repo/.agents/AGENTS.md", "b"),
                ],
                "/repo",
                vec![],
                vec![],
            ),
            (vec![], "/repo", vec![], vec![]),
        ] {
            assert_eq!(planned(&files, start, &below), expected, "{files:?}");
        }
    }

    /// A bridge we wrote must not count as the project's own instructions, or the
    /// second run would conclude the first one's AGENTS.md was never read.
    #[test]
    fn a_bridge_is_transparent_so_a_second_run_changes_nothing() {
        let bridge = bridge_text("AGENTS.md");
        let files = [
            ("/repo/AGENTS.md", "a"),
            ("/repo/CLAUDE.local.md", bridge.as_str()),
            ("/repo/web/AGENTS.md", "b"),
        ];
        assert_eq!(
            planned(&files, "/repo", &["/repo/web"]),
            vec![
                Action::Present {
                    bridge: PathBuf::from("/repo/CLAUDE.local.md")
                },
                create("/repo/web/CLAUDE.local.md", "AGENTS.md"),
            ]
        );
    }

    #[test]
    fn nothing_above_the_repository_is_touched() {
        let files = [("/AGENTS.md", "machine-wide"), ("/repo/AGENTS.md", "a")];
        assert_eq!(
            planned(&files, "/repo", &[]),
            vec![create("/repo/CLAUDE.local.md", "AGENTS.md")]
        );
    }

    #[test]
    fn the_doctor_sees_unread_bridged_and_not_applicable() {
        let bridge = bridge_text("AGENTS.md");
        for (files, expected) in [
            (
                vec![("/repo/AGENTS.md", "a")],
                Instructions::Unread(PathBuf::from("/repo/AGENTS.md")),
            ),
            (
                vec![
                    ("/repo/AGENTS.md", "a"),
                    ("/repo/CLAUDE.local.md", bridge.as_str()),
                ],
                Instructions::Bridged(PathBuf::from("/repo/CLAUDE.local.md")),
            ),
            (
                vec![("/repo/AGENTS.md", "a"), ("/repo/CLAUDE.md", "mine")],
                Instructions::NotApplicable,
            ),
            (vec![], Instructions::NotApplicable),
        ] {
            let disk = disk(&files);
            assert_eq!(
                standing(Path::new("/repo"), Path::new("/repo"), &|path| disk
                    .get(path)
                    .cloned()),
                expected,
                "{files:?}"
            );
        }
    }

    #[test]
    fn the_bridge_is_an_import_and_says_what_it_is() {
        let text = bridge_text(".claude/AGENTS.md");
        assert!(is_bridge(&text));
        assert!(
            text.lines().any(|line| line == "@.claude/AGENTS.md"),
            "{text}"
        );
        assert!(!is_bridge("my own notes\n@AGENTS.md\n"));
    }

    /// Removal must never cost the user something they wrote.
    #[test]
    fn only_an_untouched_bridge_is_ever_deleted() {
        assert!(is_only_a_bridge(&bridge_text("AGENTS.md")));
        assert!(is_only_a_bridge(&format!(
            "{}\n\n",
            bridge_text("AGENTS.md")
        )));
        let extended = format!("{}- and my own note\n", bridge_text("AGENTS.md"));
        assert!(is_bridge(&extended) && !is_only_a_bridge(&extended));
        assert!(!is_only_a_bridge("my own notes\n@AGENTS.md\n"));
    }

    #[test]
    fn the_exclude_line_is_added_once_and_keeps_what_was_there() {
        assert!(!excludes_bridges(
            "# git ls-files --others --exclude-from=.git/info/exclude\n"
        ));
        for already in [
            "CLAUDE.local.md\n",
            "/CLAUDE.local.md",
            "*.log\n**/CLAUDE.local.md\n",
        ] {
            assert!(excludes_bridges(already), "{already:?}");
        }
        let kept = with_exclude_line("*.log");
        assert!(kept.starts_with("*.log\n"), "{kept:?}");
        assert!(excludes_bridges(&kept));
        assert!(excludes_bridges(&with_exclude_line("")));
    }
}
