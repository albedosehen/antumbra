//! Where the operator is right now, in git terms: the anchor the CLI stamps on
//! what it ingests, so a recalled chunk names the commit it was produced at.
//!
//! Read with plain `git` subprocesses (no libgit2), fail-open: outside a
//! repository, or without `git` on the path, there is simply no anchor.

use std::path::Path;
use std::process::{Command, Stdio};

use antumbra_core::{repo_slug_from_remote, GitProvenance};

/// The current repository's anchor: origin slug, HEAD commit, and the checked-out
/// branch (absent on a detached HEAD). `None` when the working directory is not
/// inside a git repository, `git` is missing, or `origin` has no usable URL.
pub fn detect() -> Option<GitProvenance> {
    detect_in(Path::new("."))
}

/// The anchor of the working tree at `dir`, as [`detect`] reads the current one.
pub fn detect_in(dir: &Path) -> Option<GitProvenance> {
    let remote = git_in(dir, &["remote", "get-url", "origin"])?;
    let repo = repo_slug_from_remote(&remote)?;
    let commit = git_in(dir, &["rev-parse", "HEAD"])?;
    let mut anchor = GitProvenance::new(repo, commit);
    if let Some(branch) =
        git_in(dir, &["rev-parse", "--abbrev-ref", "HEAD"]).filter(|b| b != "HEAD")
    {
        anchor = anchor.on_branch(branch);
    }
    anchor.is_valid().then_some(anchor)
}

/// The files git tracks in the working tree at `dir`, relative to its root and
/// `/`-separated. Empty outside a repository.
pub fn tracked_files(dir: &Path) -> Vec<String> {
    git_in(dir, &["ls-files", "--full-name"])
        .map(|out| out.lines().map(str::to_string).collect())
        .unwrap_or_default()
}

/// One `git` invocation in `dir`, its trimmed stdout, or `None` on any failure.
fn git_in(dir: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!text.is_empty()).then_some(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The crate's tests run inside this repository, so detection yields a
    /// well-formed anchor for it; outside git (a tarball checkout) it yields
    /// nothing rather than something wrong.
    #[test]
    fn detects_a_well_formed_anchor_or_nothing() {
        match detect() {
            Some(p) => {
                assert!(p.is_valid());
                assert!(p.repo.contains('/'), "{}", p.repo);
                assert!(p.commit.len() >= 7 && p.commit.chars().all(|c| c.is_ascii_hexdigit()));
                assert_eq!(GitProvenance::parse(&p.to_evidence()), Some(p));
            }
            None => assert!(
                git_in(Path::new("."), &["rev-parse", "HEAD"]).is_none()
                    || git_in(Path::new("."), &["remote", "get-url", "origin"]).is_none()
            ),
        }
    }

    /// From any directory, the tree's own anchor and files: `tracked_files`
    /// names them from the repository root.
    #[test]
    fn reads_a_tree_from_outside_it() {
        let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        if detect_in(crate_dir).is_none() {
            return;
        }
        assert_eq!(
            detect_in(crate_dir).map(|a| a.repo),
            detect().map(|a| a.repo)
        );
        let files = tracked_files(crate_dir);
        assert!(
            files.iter().any(|f| f == "crates/antumbra-cli/Cargo.toml"),
            "{files:?}"
        );
        assert!(tracked_files(&std::env::temp_dir().join("not-a-repo-anywhere")).is_empty());
    }

    #[test]
    fn a_failing_git_command_is_none() {
        assert_eq!(
            git_in(
                Path::new("."),
                &["rev-parse", "--verify", "definitely-not-a-ref-anywhere"]
            ),
            None
        );
    }
}
