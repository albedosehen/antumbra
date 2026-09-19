//! Facts a repository's history already states, derived from `git log` with no
//! parser: who works where (ownership per top-level area), which files change
//! most (hotspots), and which files change together (co-change). Each fact is
//! stored as a `world` memory whose evidence names the commit range it was
//! computed over, so it is self-invalidating: a later session sees the anchor
//! and knows how old the fact is, instead of trusting a symbol table that a
//! cron job may or may not have refreshed.
//!
//! The git reading is one subprocess; everything after it is a pure function
//! over the log text, which is what the tests exercise.

use std::collections::{BTreeMap, HashMap};
use std::process::{Command, Stdio};

use antumbra_core::GitProvenance;

/// One commit's touch: who, and which files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Touch {
    pub sha: String,
    pub author: String,
    pub files: Vec<String>,
}

/// What the history says, over a window.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Facts {
    /// Commits in the window.
    pub commits: usize,
    /// Oldest and newest commit in the window (`None` when the log is empty).
    pub range: Option<(String, String)>,
    /// Per top-level area, the authors by commit count (top few), most first.
    pub ownership: Vec<(String, Vec<(String, usize)>)>,
    /// Files by commit count, most first.
    pub hotspots: Vec<(String, usize)>,
    /// File pairs by the number of commits that changed both, most first.
    pub co_changes: Vec<((String, String), usize)>,
}

/// Commits with more files than this are treated as sweeps (a rename, a
/// reformat), not evidence that the files belong together.
const CO_CHANGE_MAX_FILES: usize = 20;
/// How many authors to keep per area.
const AUTHORS_PER_AREA: usize = 3;

/// The raw log for the last `days` days: one `<sha>\t<author>` line per commit,
/// followed by its numstat lines. `None` outside a repository or without git.
pub fn read_log(days: u32) -> Option<String> {
    let out = Command::new("git")
        .args([
            "log",
            &format!("--since={days}.days"),
            "--no-merges",
            "--numstat",
            "--format=%H%x09%an",
        ])
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Parse the `--numstat --format=%H%x09%an` log into touches. Tolerant of blank
/// lines and of binary files (numstat prints `-` for their counts).
pub fn parse_log(text: &str) -> Vec<Touch> {
    let mut touches: Vec<Touch> = Vec::new();
    for line in text.lines() {
        let line = line.trim_end();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        match fields.as_slice() {
            [sha, author] if sha.len() >= 7 && sha.chars().all(|c| c.is_ascii_hexdigit()) => {
                touches.push(Touch {
                    sha: sha.to_string(),
                    author: author.trim().to_string(),
                    files: Vec::new(),
                });
            }
            [_added, _deleted, path] => {
                if let Some(current) = touches.last_mut() {
                    current.files.push(path.trim().to_string());
                }
            }
            _ => {}
        }
    }
    touches
}

/// The top-level area a path belongs to: its first component, or the file
/// itself at the root.
fn area_of(path: &str) -> &str {
    path.split('/').next().unwrap_or(path)
}

/// Derive the facts, keeping the `top` most significant hotspots and pairs.
pub fn derive(touches: &[Touch], top: usize) -> Facts {
    let range = match (touches.last(), touches.first()) {
        (Some(oldest), Some(newest)) => Some((oldest.sha.clone(), newest.sha.clone())),
        _ => None,
    };
    let mut by_area: BTreeMap<&str, HashMap<&str, usize>> = BTreeMap::new();
    let mut by_file: HashMap<&str, usize> = HashMap::new();
    let mut by_pair: HashMap<(&str, &str), usize> = HashMap::new();
    for t in touches {
        let mut areas_seen: Vec<&str> = Vec::new();
        for f in &t.files {
            *by_file.entry(f.as_str()).or_default() += 1;
            let area = area_of(f);
            if !areas_seen.contains(&area) {
                areas_seen.push(area);
                *by_area
                    .entry(area)
                    .or_default()
                    .entry(t.author.as_str())
                    .or_default() += 1;
            }
        }
        if (2..=CO_CHANGE_MAX_FILES).contains(&t.files.len()) {
            let mut sorted: Vec<&str> = t.files.iter().map(String::as_str).collect();
            sorted.sort_unstable();
            sorted.dedup();
            for (i, a) in sorted.iter().enumerate() {
                for b in &sorted[i + 1..] {
                    *by_pair.entry((a, b)).or_default() += 1;
                }
            }
        }
    }
    let ownership = by_area
        .into_iter()
        .map(|(area, authors)| {
            let mut ranked: Vec<(String, usize)> = authors
                .into_iter()
                .map(|(a, n)| (a.to_string(), n))
                .collect();
            ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            ranked.truncate(AUTHORS_PER_AREA);
            (area.to_string(), ranked)
        })
        .collect();
    let mut hotspots: Vec<(String, usize)> = by_file
        .into_iter()
        .map(|(f, n)| (f.to_string(), n))
        .collect();
    hotspots.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    hotspots.truncate(top);
    let mut co_changes: Vec<((String, String), usize)> = by_pair
        .into_iter()
        .filter(|(_, n)| *n >= 2)
        .map(|((a, b), n)| ((a.to_string(), b.to_string()), n))
        .collect();
    co_changes.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    co_changes.truncate(top);
    Facts {
        commits: touches.len(),
        range,
        ownership,
        hotspots,
        co_changes,
    }
}

/// The facts as the sentences a memory holds. Each names its window so the
/// content stands on its own when recalled.
pub fn render(facts: &Facts, days: u32) -> Vec<String> {
    let window = format!("in the last {days} days");
    let mut out = Vec::new();
    for (area, authors) in &facts.ownership {
        let who: Vec<String> = authors.iter().map(|(a, n)| format!("{a} ({n})")).collect();
        out.push(format!(
            "Ownership: in `{area}`, the commits {window} were by {}.",
            who.join(", ")
        ));
    }
    for (file, n) in &facts.hotspots {
        out.push(format!(
            "Hotspot: `{file}` changed in {n} commit{} {window}.",
            if *n == 1 { "" } else { "s" }
        ));
    }
    for ((a, b), n) in &facts.co_changes {
        out.push(format!(
            "Co-change: `{a}` and `{b}` changed together in {n} commits {window}."
        ));
    }
    out
}

/// The evidence every fact carries: the anchor at HEAD plus the exact commit
/// range the facts were computed over.
pub fn evidence(anchor: Option<&GitProvenance>, facts: &Facts, days: u32) -> Vec<String> {
    let mut ev = Vec::new();
    if let Some(a) = anchor {
        ev.push(a.to_evidence());
    }
    if let Some((oldest, newest)) = &facts.range {
        ev.push(format!(
            "range:{oldest}..{newest} ({days} days, {} commits)",
            facts.commits
        ));
    }
    ev
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOG: &str = "\
aaaaaaa1\tAda
3\t1\tcrates/mcp/src/server.rs
1\t0\tcrates/mcp/src/params.rs

bbbbbbb2\tBen
5\t5\tcrates/mcp/src/server.rs
-\t-\tdocs/logo.png

ccccccc3\tAda
2\t2\tcrates/mcp/src/server.rs
1\t1\tcrates/mcp/src/params.rs
0\t1\tREADME.md

ddddddd4\tAda
1\t1\tdocs/adr/0018.md
";

    #[test]
    fn parse_reads_commits_and_their_files() {
        let touches = parse_log(LOG);
        assert_eq!(touches.len(), 4);
        assert_eq!(touches[0].author, "Ada");
        assert_eq!(
            touches[0].files,
            vec!["crates/mcp/src/server.rs", "crates/mcp/src/params.rs"]
        );
        assert_eq!(
            touches[1].files,
            vec!["crates/mcp/src/server.rs", "docs/logo.png"]
        );
        assert!(parse_log("").is_empty());
        assert!(parse_log("not a log\n\tat all\n").is_empty());
    }

    #[test]
    fn derive_ranks_ownership_hotspots_and_pairs() {
        let facts = derive(&parse_log(LOG), 5);
        assert_eq!(facts.commits, 4);
        assert_eq!(
            facts.range,
            Some(("ddddddd4".to_string(), "aaaaaaa1".to_string())),
            "oldest..newest, as git log lists newest first"
        );
        let crates = facts
            .ownership
            .iter()
            .find(|(area, _)| area == "crates")
            .map(|(_, a)| a.clone())
            .unwrap();
        assert_eq!(crates, vec![("Ada".to_string(), 2), ("Ben".to_string(), 1)]);
        assert_eq!(
            facts.hotspots[0],
            ("crates/mcp/src/server.rs".to_string(), 3)
        );
        assert_eq!(
            facts.co_changes[0],
            (
                (
                    "crates/mcp/src/params.rs".to_string(),
                    "crates/mcp/src/server.rs".to_string()
                ),
                2
            )
        );
        assert!(
            facts.co_changes.iter().all(|(_, n)| *n >= 2),
            "a pair seen once is not a pattern"
        );
    }

    #[test]
    fn render_and_evidence_name_the_window() {
        let facts = derive(&parse_log(LOG), 1);
        let lines = render(&facts, 30);
        assert!(
            lines.iter().any(|l| l.starts_with(
                "Ownership: in `crates`, the commits in the last 30 days were by Ada (2), Ben (1)."
            )),
            "{lines:?}"
        );
        assert!(
            lines.iter().any(|l| l
                == "Hotspot: `crates/mcp/src/server.rs` changed in 3 commits in the last 30 days."),
            "{lines:?}"
        );
        assert!(lines.iter().any(|l| l.starts_with("Co-change: `crates/mcp/src/params.rs` and `crates/mcp/src/server.rs` changed together in 2 commits")), "{lines:?}");
        let anchor = GitProvenance::new("github.com/o/r", "aaaaaaa1").on_branch("main");
        let ev = evidence(Some(&anchor), &facts, 30);
        assert_eq!(ev[0], "git:github.com/o/r@aaaaaaa1#main");
        assert_eq!(ev[1], "range:ddddddd4..aaaaaaa1 (30 days, 4 commits)");
        assert!(evidence(None, &Facts::default(), 30).is_empty());
    }
}
