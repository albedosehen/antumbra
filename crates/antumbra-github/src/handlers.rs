//! What each event does to memories, as pure functions: memories in, the
//! changed ones out. The rules:
//!
//! - A **merge** re-anchors every memory whose current anchor sits on the
//!   merged branch to the merge commit on the base branch (path kept, the old
//!   anchor kept behind it as history). This is the squash-merge gap of
//!   ADR-0018 closed at its source event: the merged commits are not ancestors
//!   of the base branch, so without this the memories would read not-on-head
//!   forever.
//! - A **branch deletion** marks every memory whose current anchor sits on that
//!   branch orphaned. A memory re-anchored by an earlier merge sits on the base
//!   branch and is untouched, which is why GitHub's delete-after-merge is safe.
//! - A **merged pull request** becomes a memory of its own, anchored to the
//!   merge commit with the pull request as evidence, under a deterministic id
//!   so a redelivery revises rather than duplicates it.

use chrono::{DateTime, Utc};

use antumbra_core::{
    mark_orphaned, normalize_repo, reanchor, BranchOrphan, CompartmentId, GitProvenance, Memory,
    MemoryNetwork, TenantId,
};

use crate::event::PullRequestEvent;

/// The user the integration writes as, provisioned in each mapped workspace.
pub const SYSTEM_USER: &str = "user:github";
/// The host stamped on memories the integration writes.
pub const SYSTEM_HOST: &str = "github";

/// A merge, as the handlers see it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Merge {
    /// Repository slug, `host/org/name`.
    pub repo: String,
    /// The branch that was merged.
    pub head_branch: String,
    /// The branch it was merged into.
    pub base_branch: String,
    /// The commit the merge produced on the base branch.
    pub merge_commit: String,
}

impl Merge {
    /// The merge a `pull_request` delivery describes; `None` unless it is a
    /// merge with a merge commit and a recognizable repository.
    pub fn from_event(event: &PullRequestEvent) -> Option<Self> {
        if !event.is_merge() {
            return None;
        }
        let pr = &event.pull_request;
        Some(Self {
            repo: event.repository.slug()?,
            head_branch: pr.head.name.clone(),
            base_branch: pr.base.name.clone(),
            merge_commit: pr.merge_commit_sha.clone()?,
        })
    }

    /// The anchor a memory moves to: the merge commit on the base branch,
    /// keeping the memory's own path.
    fn target(&self, from: &GitProvenance) -> GitProvenance {
        let mut to = GitProvenance::new(from.repo.clone(), self.merge_commit.clone())
            .on_branch(&self.base_branch);
        if let Some(path) = &from.path {
            to = to.at_path(path);
        }
        to
    }
}

/// Re-anchor the memories whose current anchor sits on the merged branch.
/// Returns only the memories that changed, stamped `updated_at = now`.
pub fn reanchor_merged(memories: Vec<Memory>, merge: &Merge, now: DateTime<Utc>) -> Vec<Memory> {
    memories
        .into_iter()
        .filter_map(|mut m| {
            let anchor = GitProvenance::from_evidence(&m.evidence)?;
            let on_merged_branch = normalize_repo(&anchor.repo) == normalize_repo(&merge.repo)
                && anchor.branch.as_deref() == Some(merge.head_branch.as_str());
            if !on_merged_branch || !reanchor(&mut m.evidence, &merge.target(&anchor)) {
                return None;
            }
            m.updated_at = now;
            Some(m)
        })
        .collect()
}

/// Mark orphaned the memories whose current anchor sits on the deleted branch.
/// Returns only the memories that changed, stamped `updated_at = now`.
pub fn orphan_branch(
    memories: Vec<Memory>,
    repo: &str,
    branch: &str,
    now: DateTime<Utc>,
) -> Vec<Memory> {
    let orphan = BranchOrphan::new(repo, branch, now);
    memories
        .into_iter()
        .filter_map(|mut m| {
            if !mark_orphaned(&mut m.evidence, &orphan) {
                return None;
            }
            m.updated_at = now;
            Some(m)
        })
        .collect()
}

/// The memory a merged pull request becomes: an experience (`bank`) anchored
/// to the merge commit, with the pull request URL as its second evidence
/// entry. The id is derived from the repository and number, so a redelivered
/// event upserts the same memory. The caller embeds the content.
pub fn pull_request_memory(
    tenant: &TenantId,
    compartment: &CompartmentId,
    event: &PullRequestEvent,
    merge: &Merge,
    now: DateTime<Utc>,
) -> Memory {
    let pr = &event.pull_request;
    let by = pr.user.as_ref().map_or("unknown", |u| u.login.as_str());
    let merged_by = pr
        .merged_by
        .as_ref()
        .map_or("unknown", |u| u.login.as_str());
    let count = |n: Option<u64>| n.map_or("?".to_string(), |n| n.to_string());
    let content = format!(
        "Pull request #{} merged into {} in {}: {}. Opened by {}, merged by {}; {} commits, {} files \
         changed (+{} / -{}) from branch {}.",
        pr.number,
        merge.base_branch,
        merge.repo,
        pr.title.trim(),
        by,
        merged_by,
        count(pr.commits),
        count(pr.changed_files),
        count(pr.additions),
        count(pr.deletions),
        merge.head_branch,
    );
    let anchor = GitProvenance::new(merge.repo.clone(), merge.merge_commit.clone())
        .on_branch(&merge.base_branch);
    Memory::new(
        pull_request_memory_id(&merge.repo, pr.number),
        tenant.clone(),
        MemoryNetwork::Bank,
        content,
        0.8,
        now,
    )
    .in_compartment(compartment.clone())
    .by(SYSTEM_USER, SYSTEM_HOST)
    .with_evidence(vec![anchor.to_evidence(), pr.html_url.clone()])
}

/// `memory:github-pr-<slug>-<number>`, the slug reduced to `[a-z0-9-]`.
pub fn pull_request_memory_id(repo: &str, number: u64) -> String {
    let slug: String = normalize_repo(repo)
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    format!("memory:github-pr-{slug}-{number}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{Actor, GitRef, PullRequest, Repository};
    use antumbra_core::orphan_of;

    const REPO: &str = "github.com/acme/orders";

    fn memory(id: &str, evidence: &[&str]) -> Memory {
        Memory::new(id, "ws:acme", MemoryNetwork::World, id, 0.6, Utc::now())
            .with_evidence(evidence.iter().map(|e| e.to_string()).collect())
    }

    fn merge() -> Merge {
        Merge {
            repo: REPO.into(),
            head_branch: "feat/outbox".into(),
            base_branch: "main".into(),
            merge_commit: "fedcba9876543210".into(),
        }
    }

    fn event() -> PullRequestEvent {
        PullRequestEvent {
            action: "closed".into(),
            pull_request: PullRequest {
                number: 42,
                title: " Move orders to the outbox pattern ".into(),
                html_url: "https://github.com/Acme/Orders/pull/42".into(),
                merged: true,
                merge_commit_sha: Some("fedcba9876543210".into()),
                base: GitRef {
                    name: "main".into(),
                    sha: "1111111".into(),
                },
                head: GitRef {
                    name: "feat/outbox".into(),
                    sha: "2222222".into(),
                },
                user: Some(Actor {
                    login: "shon".into(),
                }),
                merged_by: None,
                merged_at: None,
                commits: Some(3),
                additions: Some(120),
                deletions: None,
                changed_files: Some(5),
            },
            repository: Repository {
                full_name: "Acme/Orders".into(),
                html_url: Some("https://github.com/Acme/Orders".into()),
                default_branch: Some("main".into()),
            },
        }
    }

    #[test]
    fn a_merge_reanchors_only_the_merged_branch_in_that_repo() {
        let now = Utc::now();
        let memories = vec![
            memory(
                "memory:a",
                &["git:GitHub.com/Acme/Orders@2222222#feat/outbox:src/orders.rs"],
            ),
            memory("memory:b", &["git:github.com/acme/orders@1111111#main"]),
            memory(
                "memory:c",
                &["git:github.com/acme/billing@2222222#feat/outbox"],
            ),
            memory("memory:d", &["a note with no anchor"]),
            memory(
                "memory:e",
                &["git:github.com/acme/orders@3333333#feat/outbox"],
            ),
        ];
        let changed = reanchor_merged(memories, &merge(), now);
        let ids: Vec<&str> = changed.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, vec!["memory:a", "memory:e"]);
        let a = &changed[0];
        let anchor = GitProvenance::from_evidence(&a.evidence).unwrap();
        assert_eq!(anchor.commit, "fedcba9876543210");
        assert_eq!(anchor.branch.as_deref(), Some("main"));
        assert_eq!(
            anchor.path.as_deref(),
            Some("src/orders.rs"),
            "the path survives"
        );
        assert_eq!(
            anchor.repo, "GitHub.com/Acme/Orders",
            "the memory's own spelling is kept"
        );
        assert_eq!(
            a.evidence.len(),
            2,
            "the old anchor stays behind as history"
        );
        assert_eq!(a.updated_at, now);
        // Redelivery: everything already sits at the merge commit.
        assert!(reanchor_merged(changed, &merge(), now).is_empty());
    }

    #[test]
    fn a_branch_delete_orphans_only_that_branch() {
        let now = Utc::now();
        let memories = vec![
            memory(
                "memory:a",
                &["git:github.com/acme/orders@2222222#feat/outbox"],
            ),
            memory("memory:b", &["git:github.com/acme/orders@1111111#main"]),
            memory("memory:c", &["git:github.com/acme/orders@2222222"]),
        ];
        let changed = orphan_branch(memories, REPO, "feat/outbox", now);
        assert_eq!(changed.len(), 1);
        assert_eq!(changed[0].id.as_str(), "memory:a");
        assert_eq!(
            orphan_of(&changed[0].evidence).map(|o| o.branch),
            Some("feat/outbox".into())
        );
        assert!(
            orphan_branch(changed, REPO, "feat/outbox", now).is_empty(),
            "already marked"
        );
    }

    #[test]
    fn merge_then_delete_leaves_the_reanchored_memory_alone() {
        let now = Utc::now();
        let memories = vec![memory(
            "memory:a",
            &["git:github.com/acme/orders@2222222#feat/outbox"],
        )];
        let merged = reanchor_merged(memories, &merge(), now);
        assert_eq!(merged.len(), 1);
        assert!(
            orphan_branch(merged, REPO, "feat/outbox", now).is_empty(),
            "GitHub's delete-after-merge must not orphan what the merge just re-anchored"
        );
    }

    #[test]
    fn the_pull_request_memory_is_anchored_deterministic_and_readable() {
        let now = Utc::now();
        let tenant = TenantId::new("ws:acme");
        let compartment = CompartmentId::new("comp:ws:acme:user:github:default");
        let m = pull_request_memory(&tenant, &compartment, &event(), &merge(), now);
        assert_eq!(m.id.as_str(), "memory:github-pr-github-com-acme-orders-42");
        assert_eq!(m.network, MemoryNetwork::Bank);
        assert_eq!(m.author.as_ref().map(|u| u.as_str()), Some(SYSTEM_USER));
        assert_eq!(m.author_host.as_deref(), Some(SYSTEM_HOST));
        assert_eq!(m.compartment.as_ref(), Some(&compartment));
        assert_eq!(
            m.content,
            "Pull request #42 merged into main in github.com/acme/orders: Move orders to the outbox \
             pattern. Opened by shon, merged by unknown; 3 commits, 5 files changed (+120 / -?) from \
             branch feat/outbox."
        );
        let anchor = GitProvenance::from_evidence(&m.evidence).unwrap();
        assert_eq!(anchor.commit, "fedcba9876543210");
        assert_eq!(anchor.branch.as_deref(), Some("main"));
        assert_eq!(m.evidence[1], "https://github.com/Acme/Orders/pull/42");
        assert_eq!(
            pull_request_memory_id("GitHub.com/Acme/Orders.git", 7),
            "memory:github-pr-github-com-acme-orders-7"
        );
    }

    #[test]
    fn merge_from_event_needs_a_merge_commit_and_a_slug() {
        assert_eq!(Merge::from_event(&event()), Some(merge()));
        let mut unmerged = event();
        unmerged.pull_request.merged = false;
        assert_eq!(Merge::from_event(&unmerged), None);
        let mut no_sha = event();
        no_sha.pull_request.merge_commit_sha = None;
        assert_eq!(Merge::from_event(&no_sha), None);
        let mut no_slug = event();
        no_slug.repository.html_url = None;
        no_slug.repository.full_name = "orders".into();
        assert_eq!(Merge::from_event(&no_slug), None);
    }
}
