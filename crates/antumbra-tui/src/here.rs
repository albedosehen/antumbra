//! Where the console was started, as git sees it, so the Memory page can say
//! how each memory's anchor relates to it: the same scope recall tags a hit
//! with (ADR-0018). Outside a repository, or without git, every memory's scope
//! is unknown, as it is for a recall that names no context.

use std::path::Path;
use std::process::Command;

use antumbra_core::{repo_slug_from_remote, GitContext};

/// One line of `git <args>` run in `dir`, or `None` when git is missing, fails,
/// or prints nothing.
fn git(dir: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let line = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!line.is_empty()).then_some(line)
}

/// The repository and branch `dir` is in. A detached head names no branch.
pub fn context(dir: &Path) -> GitContext {
    let repo = git(dir, &["config", "--get", "remote.origin.url"])
        .and_then(|url| repo_slug_from_remote(&url));
    // Works on a branch with no commits yet, and fails on a detached head.
    let branch = git(dir, &["symbolic-ref", "--short", "HEAD"]);
    GitContext { repo, branch }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("antumbra-here-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_clone_is_named_by_its_origin_and_branch() {
        let dir = scratch("clone");
        let run = |args: &[&str]| {
            Command::new("git")
                .arg("-C")
                .arg(&dir)
                .args(args)
                .output()
                .unwrap()
        };
        run(&["init", "-q", "-b", "main"]);
        run(&["remote", "add", "origin", "git@github.com:acme/orders.git"]);
        let here = context(&dir);
        assert_eq!(here.repo.as_deref(), Some("github.com/acme/orders"));
        assert_eq!(here.branch.as_deref(), Some("main"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_directory_with_no_remote_names_no_repository() {
        let dir = scratch("bare");
        Command::new("git")
            .arg("-C")
            .arg(&dir)
            .args(["init", "-q"])
            .output()
            .unwrap();
        assert_eq!(context(&dir).repo, None);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
