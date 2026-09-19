//! Git provenance: where a memory about code came from, precisely enough to
//! judge later whether it still applies.
//!
//! A memory about code is only as good as its anchor. Static extraction keeps
//! that anchor by re-extracting and pruning symbol tables; Antumbra keeps it as
//! **provenance on the memory itself** and evaluates freshness at recall, where
//! git is. The hook that boots a session can ask `git merge-base --is-ancestor`
//! whether the commit a memory names is still on HEAD, and `git show-ref`
//! whether its branch still exists, and tag the memory live, not-on-head, or
//! orphaned. Nothing is re-extracted and nothing is garbage-collected: the
//! anchor travels with the memory, and a stale one is *visible* rather than
//! silently wrong.
//!
//! The wire form is one evidence entry:
//!
//! ```text
//! git:<repo>@<commit>[#<branch>][:<path>]
//! ```
//!
//! `repo` is a slug (`github.com/org/name`), never a URL, and may not contain
//! `@`; [`repo_slug_from_remote`] normalizes the ssh and https spellings of a
//! remote into it so one repository never splits in two. `commit` is 7 to 40
//! hex digits. `branch` (which git forbids from containing `:`) and `path` (the
//! rest after the first `:` past the commit) are optional.

use std::fmt;

use serde::{Deserialize, Serialize};

/// The prefix that marks an evidence entry as git provenance.
pub const GIT_EVIDENCE_PREFIX: &str = "git:";

/// The git anchor of a memory: repository, commit, and optionally the branch and
/// path it was learned at.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GitProvenance {
    /// Repository slug, `host/org/name`.
    pub repo: String,
    /// The commit the memory was learned at (7 to 40 hex digits).
    pub commit: String,
    /// The branch checked out at the time, when known.
    pub branch: Option<String>,
    /// The file the memory is about, when it is about one file.
    pub path: Option<String>,
}

impl GitProvenance {
    /// A provenance at `repo` and `commit`, with no branch or path.
    pub fn new(repo: impl Into<String>, commit: impl Into<String>) -> Self {
        Self {
            repo: repo.into(),
            commit: commit.into(),
            branch: None,
            path: None,
        }
    }

    /// Record the branch the memory was learned on.
    pub fn on_branch(mut self, branch: impl Into<String>) -> Self {
        self.branch = Some(branch.into());
        self
    }

    /// Record the file the memory is about.
    pub fn at_path(mut self, path: impl Into<String>) -> Self {
        self.path = Some(path.into());
        self
    }

    /// Whether the fields form a valid evidence entry: a non-empty slug without
    /// `@`, a 7 to 40 digit hex commit, and no `:` in the branch.
    pub fn is_valid(&self) -> bool {
        let repo_ok = !self.repo.trim().is_empty() && !self.repo.contains('@');
        let commit_ok = (7..=40).contains(&self.commit.len())
            && self.commit.chars().all(|c| c.is_ascii_hexdigit());
        let branch_ok = self
            .branch
            .as_deref()
            .is_none_or(|b| !b.is_empty() && !b.contains(':'));
        repo_ok && commit_ok && branch_ok
    }

    /// Parse the evidence form. `None` when the entry is not git provenance or is
    /// malformed, so a free-text evidence note that happens to start with `git:`
    /// is ignored rather than misread.
    pub fn parse(evidence: &str) -> Option<Self> {
        let rest = evidence.strip_prefix(GIT_EVIDENCE_PREFIX)?;
        let (repo, rest) = rest.split_once('@')?;
        let commit_len = rest.chars().take_while(|c| c.is_ascii_hexdigit()).count();
        let (commit, rest) = rest.split_at(commit_len);
        let (branch, path) = match rest.chars().next() {
            None => (None, None),
            Some('#') => {
                let rest = &rest[1..];
                match rest.split_once(':') {
                    Some((branch, path)) => (Some(branch), Some(path)),
                    None => (Some(rest), None),
                }
            }
            Some(':') => (None, Some(&rest[1..])),
            Some(_) => return None,
        };
        let parsed = Self {
            repo: repo.to_string(),
            commit: commit.to_string(),
            branch: branch.map(str::to_string),
            path: path.filter(|p| !p.is_empty()).map(str::to_string),
        };
        parsed.is_valid().then_some(parsed)
    }

    /// The evidence entry.
    pub fn to_evidence(&self) -> String {
        let mut out = format!("{GIT_EVIDENCE_PREFIX}{}@{}", self.repo, self.commit);
        if let Some(branch) = &self.branch {
            out.push('#');
            out.push_str(branch);
        }
        if let Some(path) = &self.path {
            out.push(':');
            out.push_str(path);
        }
        out
    }

    /// The first git provenance among a memory's evidence entries.
    pub fn from_evidence(evidence: &[String]) -> Option<Self> {
        evidence.iter().find_map(|e| Self::parse(e))
    }
}

impl fmt::Display for GitProvenance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_evidence())
    }
}

/// Where the caller is right now, so recall can judge each memory's scope.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct GitContext {
    /// The caller's repository slug, when inside a repository.
    pub repo: Option<String>,
    /// The caller's checked-out branch, when known.
    pub branch: Option<String>,
}

/// How a memory's provenance relates to the caller's context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Scope {
    /// Same repository, and the same branch or no branch named.
    InScope,
    /// Same repository, a different branch: it may describe work that never
    /// reached this branch, or that this branch has since superseded.
    OtherBranch,
    /// A different repository.
    OtherRepo,
    /// The memory carries no git provenance, or the caller gave no context.
    Unknown,
}

impl Scope {
    /// Whether recall should demote the memory below in-scope ones.
    pub fn is_inhibited(self) -> bool {
        matches!(self, Scope::OtherBranch | Scope::OtherRepo)
    }

    /// The wire name (`in_scope`, `other_branch`, `other_repo`, `unknown`).
    pub fn as_str(self) -> &'static str {
        match self {
            Scope::InScope => "in_scope",
            Scope::OtherBranch => "other_branch",
            Scope::OtherRepo => "other_repo",
            Scope::Unknown => "unknown",
        }
    }
}

/// Judge a memory's provenance against the caller's context. This is the
/// branch-as-governing-feature rule: a memory scoped to one branch is inhibited
/// on another, the same way a repo-scoped convention is inhibited one repo over.
pub fn scope_of(provenance: Option<&GitProvenance>, ctx: &GitContext) -> Scope {
    let (Some(p), Some(repo)) = (provenance, ctx.repo.as_deref()) else {
        return Scope::Unknown;
    };
    if normalize_repo(&p.repo) != normalize_repo(repo) {
        return Scope::OtherRepo;
    }
    match (&p.branch, &ctx.branch) {
        (Some(theirs), Some(ours)) if theirs != ours => Scope::OtherBranch,
        _ => Scope::InScope,
    }
}

/// Canonical repo slug for comparison: lowercase, no trailing `.git` or `/`, so
/// the two spellings of one remote never split a repository in two.
pub fn normalize_repo(repo: &str) -> String {
    let mut s = repo.trim().to_ascii_lowercase();
    while s.ends_with('/') {
        s.pop();
    }
    if let Some(stripped) = s.strip_suffix(".git") {
        s = stripped.to_string();
    }
    s
}

/// Turn a git remote URL into the slug form (`host/org/name`). Handles
/// `git@host:org/name.git`, `ssh://git@host/org/name.git`,
/// `https://host/org/name.git`, and a slug given as-is. `None` for an empty or
/// unrecognizable remote (a local path, say).
pub fn repo_slug_from_remote(remote: &str) -> Option<String> {
    let remote = remote.trim();
    if remote.is_empty() {
        return None;
    }
    // Scheme form: `ssh://user@host/org/name.git`, `https://host/org/name`.
    let body = match remote.split_once("://") {
        Some((_, body)) => body.to_string(),
        // scp form `user@host:org/name.git` becomes `host/org/name.git`.
        None => match remote.split_once('@') {
            Some((_, host_and_path)) => host_and_path.replacen(':', "/", 1),
            None => remote.to_string(),
        },
    };
    // Drop any `user@` left in a scheme form.
    let body = body.rsplit_once('@').map_or(body.as_str(), |(_, b)| b);
    let slug = normalize_repo(body);
    // A slug is at least `host/name`; anything without a `/` (a local path on
    // Windows, a bare word) is not one.
    (slug.contains('/') && !slug.starts_with('/') && !slug.contains('\\')).then_some(slug)
}

/// Order recalled items so out-of-scope ones come after in-scope and unknown
/// ones, without hiding them: a same-repo other-branch memory may still be
/// right, it is just not the first thing to act on. Stable, so the recall
/// ranking survives within each group.
pub fn demote_out_of_scope<T>(items: Vec<T>, scope_of_item: impl Fn(&T) -> Scope) -> Vec<T> {
    let (kept, demoted): (Vec<T>, Vec<T>) = items
        .into_iter()
        .partition(|item| !scope_of_item(item).is_inhibited());
    kept.into_iter().chain(demoted).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evidence_round_trips_every_shape() {
        let cases = [
            GitProvenance::new("github.com/oneiriq/antumbra", "b697da7"),
            GitProvenance::new("github.com/oneiriq/antumbra", "b697da7").on_branch("main"),
            GitProvenance::new("github.com/oneiriq/antumbra", "b697da7")
                .on_branch("feat/provenance-and-hardening")
                .at_path("crates/antumbra-core/src/provenance.rs"),
            GitProvenance::new("github.com/oneiriq/antumbra", "b697da7")
                .at_path("docs/adr/0004-inhibitory-boundaries.md"),
        ];
        for p in cases {
            let wire = p.to_evidence();
            assert!(wire.starts_with("git:github.com/oneiriq/antumbra@b697da7"));
            assert_eq!(GitProvenance::parse(&wire), Some(p.clone()), "{wire}");
            assert_eq!(p.to_string(), wire);
        }
    }

    #[test]
    fn parse_reads_branch_and_path_separately() {
        let p =
            GitProvenance::parse("git:github.com/o/r@0123abcdef#release/1.2:src/lib.rs").unwrap();
        assert_eq!(p.branch.as_deref(), Some("release/1.2"));
        assert_eq!(p.path.as_deref(), Some("src/lib.rs"));
        let p = GitProvenance::parse("git:github.com/o/r@0123abcdef:src/lib.rs").unwrap();
        assert_eq!(p.branch, None);
        assert_eq!(p.path.as_deref(), Some("src/lib.rs"));
        let p = GitProvenance::parse("git:github.com/o/r@0123abcdef#main:").unwrap();
        assert_eq!(p.path, None, "an empty path is no path");
    }

    #[test]
    fn malformed_or_foreign_evidence_is_not_provenance() {
        for bad in [
            "https://example.com/a-link",
            "git:the deno convention, see the README",
            "git:github.com/o/r@",
            "git:github.com/o/r@12345",
            "git:github.com/o/r@xyz1234",
            "git:@0123abcdef",
            "git:github.com/o/r@0123abcdef?main",
            "git:github.com/o/r@0123abcdef#",
        ] {
            assert_eq!(GitProvenance::parse(bad), None, "{bad}");
        }
        assert!(!GitProvenance::new("user@host", "0123abcdef").is_valid());
        assert!(!GitProvenance::new("github.com/o/r", "0123abcdef")
            .on_branch("a:b")
            .is_valid());
    }

    #[test]
    fn from_evidence_finds_the_first_git_entry() {
        let evidence = vec![
            "test_orders.py passed".to_string(),
            "git:not really".to_string(),
            "git:github.com/o/r@0123abcdef#main".to_string(),
            "git:github.com/o/r@fedcba9876#other".to_string(),
        ];
        let p = GitProvenance::from_evidence(&evidence).unwrap();
        assert_eq!(p.commit, "0123abcdef");
        assert_eq!(GitProvenance::from_evidence(&["nothing".to_string()]), None);
    }

    #[test]
    fn scope_follows_repo_then_branch() {
        let here = GitContext {
            repo: Some("github.com/o/r".into()),
            branch: Some("main".into()),
        };
        let on_main = GitProvenance::new("GitHub.com/O/R.git", "0123abcdef").on_branch("main");
        let on_feat = GitProvenance::new("github.com/o/r", "0123abcdef").on_branch("feat/x");
        let no_branch = GitProvenance::new("github.com/o/r", "0123abcdef");
        let elsewhere = GitProvenance::new("github.com/o/other", "0123abcdef").on_branch("main");
        assert_eq!(scope_of(Some(&on_main), &here), Scope::InScope);
        assert_eq!(scope_of(Some(&on_feat), &here), Scope::OtherBranch);
        assert_eq!(scope_of(Some(&no_branch), &here), Scope::InScope);
        assert_eq!(scope_of(Some(&elsewhere), &here), Scope::OtherRepo);
        assert_eq!(scope_of(None, &here), Scope::Unknown);
        assert_eq!(
            scope_of(Some(&on_main), &GitContext::default()),
            Scope::Unknown
        );
        let no_branch_ctx = GitContext {
            repo: Some("github.com/o/r".into()),
            branch: None,
        };
        assert_eq!(scope_of(Some(&on_feat), &no_branch_ctx), Scope::InScope);
        assert!(Scope::OtherBranch.is_inhibited() && Scope::OtherRepo.is_inhibited());
        assert!(!Scope::InScope.is_inhibited() && !Scope::Unknown.is_inhibited());
        assert_eq!(Scope::OtherBranch.as_str(), "other_branch");
    }

    #[test]
    fn remotes_normalize_to_one_slug() {
        for remote in [
            "git@github.com:Oneiriq/antumbra.git",
            "ssh://git@github.com/Oneiriq/antumbra.git",
            "https://github.com/Oneiriq/antumbra.git",
            "https://github.com/Oneiriq/antumbra",
            "https://token@github.com/Oneiriq/antumbra.git",
            "github.com/oneiriq/antumbra",
            "  github.com/Oneiriq/antumbra/ ",
        ] {
            assert_eq!(
                repo_slug_from_remote(remote).as_deref(),
                Some("github.com/oneiriq/antumbra"),
                "{remote}"
            );
        }
        assert_eq!(repo_slug_from_remote(""), None);
        assert_eq!(repo_slug_from_remote("antumbra"), None);
        assert_eq!(repo_slug_from_remote("/srv/git/antumbra.git"), None);
        assert_eq!(repo_slug_from_remote("C:\\repos\\antumbra"), None);
    }

    #[test]
    fn demotion_is_stable_and_hides_nothing() {
        let items = vec![
            ("a", Scope::OtherBranch),
            ("b", Scope::InScope),
            ("c", Scope::Unknown),
            ("d", Scope::OtherRepo),
            ("e", Scope::InScope),
        ];
        let ordered: Vec<&str> = demote_out_of_scope(items, |(_, s)| *s)
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        assert_eq!(ordered, vec!["b", "c", "e", "a", "d"]);
    }
}
