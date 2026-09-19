//! The delivery shapes Antumbra reads. Only the fields a handler needs are
//! modeled; everything else in a GitHub payload is ignored on parse, so a new
//! field on GitHub's side never breaks the receiver.

use serde::Deserialize;

use antumbra_core::{normalize_repo, repo_slug_from_remote};

/// The repository a delivery is about.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Repository {
    /// `org/name`.
    pub full_name: String,
    /// The browsable URL, which also names the host (github.com, or an
    /// Enterprise Server host).
    #[serde(default)]
    pub html_url: Option<String>,
    #[serde(default)]
    pub default_branch: Option<String>,
}

impl Repository {
    /// The repository slug (`host/org/name`) memories anchor to: from the
    /// browsable URL when GitHub sent one (so an Enterprise Server host is
    /// kept), else `github.com/<full_name>`.
    pub fn slug(&self) -> Option<String> {
        self.html_url
            .as_deref()
            .and_then(repo_slug_from_remote)
            .or_else(|| {
                self.full_name
                    .contains('/')
                    .then(|| normalize_repo(&format!("github.com/{}", self.full_name)))
            })
    }
}

/// The App installation a delivery came through: what an API call
/// authenticates as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct Installation {
    pub id: u64,
}

/// A repository named by an installation delivery (no URL there).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct RepoRef {
    /// `org/name`.
    pub full_name: String,
}

/// A user on GitHub.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Actor {
    pub login: String,
}

/// A branch reference on a pull request: its name and the commit it pointed at.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct GitRef {
    #[serde(rename = "ref")]
    pub name: String,
    pub sha: String,
}

/// The pull request inside a `pull_request` delivery.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct PullRequest {
    pub number: u64,
    pub title: String,
    pub html_url: String,
    #[serde(default)]
    pub merged: bool,
    /// The commit the merge produced (a merge commit, or the squashed or
    /// rebased head), set once the pull request is merged.
    #[serde(default)]
    pub merge_commit_sha: Option<String>,
    pub base: GitRef,
    pub head: GitRef,
    #[serde(default)]
    pub user: Option<Actor>,
    #[serde(default)]
    pub merged_by: Option<Actor>,
    #[serde(default)]
    pub merged_at: Option<String>,
    #[serde(default)]
    pub commits: Option<u64>,
    #[serde(default)]
    pub additions: Option<u64>,
    #[serde(default)]
    pub deletions: Option<u64>,
    #[serde(default)]
    pub changed_files: Option<u64>,
}

/// A `pull_request` delivery.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct PullRequestEvent {
    pub action: String,
    pub pull_request: PullRequest,
    pub repository: Repository,
    /// Present on deliveries that came through an App installation.
    #[serde(default)]
    pub installation: Option<Installation>,
}

impl PullRequestEvent {
    /// Whether this delivery is the pull request being merged (GitHub sends
    /// `closed` for both a merge and a plain close; `merged` tells them apart).
    pub fn is_merge(&self) -> bool {
        self.action == "closed" && self.pull_request.merged
    }
}

/// A `delete` delivery: a branch or tag was deleted.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct DeleteEvent {
    /// The deleted ref's short name (`feat/x`, not `refs/heads/feat/x`).
    #[serde(rename = "ref")]
    pub name: String,
    /// `branch` or `tag`.
    pub ref_type: String,
    pub repository: Repository,
    #[serde(default)]
    pub installation: Option<Installation>,
}

impl DeleteEvent {
    pub fn is_branch(&self) -> bool {
        self.ref_type == "branch"
    }
}

/// An `installation` or `installation_repositories` delivery: the App was
/// installed, or repositories were added to an installation. Either way the
/// named repositories are new to Antumbra.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstallationEvent {
    /// `created` / `deleted` for the App; `added` / `removed` for repositories.
    pub action: String,
    pub installation: Installation,
    /// The repositories this delivery adds.
    pub repositories: Vec<RepoRef>,
}

impl InstallationEvent {
    /// Whether the delivery brings repositories in (the cold start), as
    /// opposed to taking them away.
    pub fn is_cold_start(&self) -> bool {
        matches!(self.action.as_str(), "created" | "added") && !self.repositories.is_empty()
    }
}

/// The two installation payloads share a shape apart from the field naming
/// the repositories.
#[derive(Deserialize)]
struct InstallationRaw {
    action: String,
    installation: Installation,
    #[serde(default)]
    repositories: Vec<RepoRef>,
    #[serde(default)]
    repositories_added: Vec<RepoRef>,
}

/// A delivery Antumbra has a handler for, or knows it does not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    /// GitHub's hook-registration test; answered, never acted on.
    Ping,
    PullRequest(Box<PullRequestEvent>),
    Delete(DeleteEvent),
    Installation(InstallationEvent),
    /// Any other event kind, by name: acknowledged so GitHub does not retry.
    Other(String),
}

#[derive(Debug, thiserror::Error)]
pub enum EventError {
    #[error("malformed {kind} payload: {source}")]
    Malformed {
        kind: String,
        #[source]
        source: serde_json::Error,
    },
}

/// Parse a delivery from its event kind (the `X-GitHub-Event` header) and raw
/// JSON body.
pub fn parse(kind: &str, body: &[u8]) -> Result<Event, EventError> {
    let malformed = |source| EventError::Malformed {
        kind: kind.to_string(),
        source,
    };
    match kind {
        "ping" => Ok(Event::Ping),
        "pull_request" => serde_json::from_slice(body)
            .map(|e| Event::PullRequest(Box::new(e)))
            .map_err(malformed),
        "delete" => serde_json::from_slice(body)
            .map(Event::Delete)
            .map_err(malformed),
        "installation" | "installation_repositories" => serde_json::from_slice(body)
            .map(|raw: InstallationRaw| {
                let repositories = if raw.repositories_added.is_empty() {
                    raw.repositories
                } else {
                    raw.repositories_added
                };
                Event::Installation(InstallationEvent {
                    action: raw.action,
                    installation: raw.installation,
                    repositories,
                })
            })
            .map_err(malformed),
        other => Ok(Event::Other(other.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn merged_pr() -> serde_json::Value {
        json!({
            "action": "closed",
            "number": 42,
            "pull_request": {
                "number": 42,
                "title": "Move orders to the outbox pattern",
                "html_url": "https://github.com/Acme/Orders/pull/42",
                "merged": true,
                "merge_commit_sha": "fedcba9876543210fedcba9876543210fedcba98",
                "base": { "ref": "main", "sha": "1111111111111111111111111111111111111111" },
                "head": { "ref": "feat/outbox", "sha": "2222222222222222222222222222222222222222" },
                "user": { "login": "shon" },
                "merged_by": { "login": "reviewer" },
                "merged_at": "2026-09-19T10:00:00Z",
                "commits": 3, "additions": 120, "deletions": 40, "changed_files": 5,
                "an_unmodeled_field": { "nested": true }
            },
            "repository": {
                "full_name": "Acme/Orders",
                "html_url": "https://github.com/Acme/Orders",
                "default_branch": "main"
            },
            "installation": { "id": 77, "node_id": "MDIz" }
        })
    }

    #[test]
    fn a_merged_pull_request_parses_with_its_slug_and_installation() {
        let body = serde_json::to_vec(&merged_pr()).unwrap();
        let Event::PullRequest(e) = parse("pull_request", &body).unwrap() else {
            panic!("expected a pull_request event");
        };
        assert!(e.is_merge());
        assert_eq!(e.pull_request.number, 42);
        assert_eq!(e.pull_request.head.name, "feat/outbox");
        assert_eq!(e.pull_request.base.name, "main");
        assert_eq!(
            e.pull_request.merge_commit_sha.as_deref(),
            Some("fedcba9876543210fedcba9876543210fedcba98")
        );
        assert_eq!(
            e.repository.slug().as_deref(),
            Some("github.com/acme/orders")
        );
        assert_eq!(e.installation, Some(Installation { id: 77 }));
    }

    #[test]
    fn a_closed_but_unmerged_pull_request_is_not_a_merge() {
        let mut v = merged_pr();
        v["pull_request"]["merged"] = json!(false);
        v["pull_request"]["merge_commit_sha"] = serde_json::Value::Null;
        v.as_object_mut().unwrap().remove("installation");
        let body = serde_json::to_vec(&v).unwrap();
        let Event::PullRequest(e) = parse("pull_request", &body).unwrap() else {
            panic!("expected a pull_request event");
        };
        assert!(!e.is_merge());
        assert_eq!(e.installation, None, "a plain webhook has no installation");
        let mut v = merged_pr();
        v["action"] = json!("opened");
        let body = serde_json::to_vec(&v).unwrap();
        let Event::PullRequest(e) = parse("pull_request", &body).unwrap() else {
            panic!("expected a pull_request event");
        };
        assert!(!e.is_merge());
    }

    #[test]
    fn the_slug_keeps_an_enterprise_host_and_falls_back_to_github() {
        let ghes = Repository {
            full_name: "Acme/Orders".into(),
            html_url: Some("https://git.acme.internal/Acme/Orders".into()),
            default_branch: None,
        };
        assert_eq!(
            ghes.slug().as_deref(),
            Some("git.acme.internal/acme/orders")
        );
        let bare = Repository {
            full_name: "Acme/Orders".into(),
            html_url: None,
            default_branch: None,
        };
        assert_eq!(bare.slug().as_deref(), Some("github.com/acme/orders"));
        let broken = Repository {
            full_name: "orders".into(),
            html_url: None,
            default_branch: None,
        };
        assert_eq!(broken.slug(), None);
    }

    #[test]
    fn delete_ping_and_unknown_events_parse() {
        let body = serde_json::to_vec(&json!({
            "ref": "feat/outbox",
            "ref_type": "branch",
            "pusher_type": "user",
            "repository": { "full_name": "Acme/Orders", "html_url": "https://github.com/Acme/Orders" }
        }))
        .unwrap();
        let Event::Delete(e) = parse("delete", &body).unwrap() else {
            panic!("expected a delete event");
        };
        assert!(e.is_branch());
        assert_eq!(e.name, "feat/outbox");
        let body = serde_json::to_vec(&json!({
            "ref": "v1.2.0", "ref_type": "tag",
            "repository": { "full_name": "Acme/Orders" }
        }))
        .unwrap();
        let Event::Delete(e) = parse("delete", &body).unwrap() else {
            panic!("expected a delete event");
        };
        assert!(!e.is_branch());
        assert_eq!(parse("ping", b"{\"zen\":\"...\"}").unwrap(), Event::Ping);
        assert_eq!(
            parse("issues", b"{}").unwrap(),
            Event::Other("issues".to_string())
        );
    }

    #[test]
    fn installation_deliveries_name_the_repositories_they_add() {
        let body = serde_json::to_vec(&json!({
            "action": "created",
            "installation": { "id": 77, "account": { "login": "Acme" } },
            "repositories": [
                { "id": 1, "full_name": "Acme/Orders", "private": true },
                { "id": 2, "full_name": "Acme/Billing", "private": true }
            ]
        }))
        .unwrap();
        let Event::Installation(e) = parse("installation", &body).unwrap() else {
            panic!("expected an installation event");
        };
        assert!(e.is_cold_start());
        assert_eq!(e.installation.id, 77);
        assert_eq!(
            e.repositories
                .iter()
                .map(|r| r.full_name.as_str())
                .collect::<Vec<_>>(),
            vec!["Acme/Orders", "Acme/Billing"]
        );
        let body = serde_json::to_vec(&json!({
            "action": "added",
            "installation": { "id": 77 },
            "repositories_added": [{ "full_name": "Acme/Shipping" }],
            "repositories_removed": []
        }))
        .unwrap();
        let Event::Installation(e) = parse("installation_repositories", &body).unwrap() else {
            panic!("expected an installation event");
        };
        assert!(e.is_cold_start());
        assert_eq!(e.repositories[0].full_name, "Acme/Shipping");
        let body = serde_json::to_vec(&json!({
            "action": "deleted",
            "installation": { "id": 77 },
            "repositories": [{ "full_name": "Acme/Orders" }]
        }))
        .unwrap();
        let Event::Installation(e) = parse("installation", &body).unwrap() else {
            panic!("expected an installation event");
        };
        assert!(!e.is_cold_start(), "an uninstall adds nothing");
    }

    #[test]
    fn a_malformed_payload_names_its_kind() {
        let err = parse("pull_request", b"{\"action\":\"closed\"}").unwrap_err();
        assert!(
            err.to_string()
                .starts_with("malformed pull_request payload"),
            "{err}"
        );
        let err = parse("delete", b"not json").unwrap_err();
        assert!(
            err.to_string().starts_with("malformed delete payload"),
            "{err}"
        );
        let err = parse("installation", b"{\"action\":\"created\"}").unwrap_err();
        assert!(
            err.to_string()
                .starts_with("malformed installation payload"),
            "{err}"
        );
    }
}
