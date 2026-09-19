//! `POST /github/webhook`: the GitHub App's deliveries (ADR-0019). The route
//! is the I/O half of `antumbra-github`: verify the HMAC signature, parse the
//! event, find the repository's workspace, load that workspace's memories,
//! apply the pure handler, write back what changed.
//!
//! Every write runs in **owner mode** under the auth lock, the way provisioning
//! and the live-propagation watcher do: a merge or a branch deletion touches
//! memories across every user and compartment of the workspace, which no
//! single record session could see. The integration writes as the system user
//! `user:github`, provisioned in the workspace on first contact.
//!
//! Unauthorized is the only refusal a caller can provoke: a delivery that does
//! not verify learns nothing else. Anything verified but not acted on (a ping,
//! an event kind without a handler, a repository the map does not name, a pull
//! request that closed without merging) is acknowledged with the reason, so
//! GitHub does not retry it.

use std::sync::Arc;

use anyhow::{bail, Context, Result};
use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use chrono::Utc;

use antumbra_core::UserId;
use antumbra_github::{
    orphan_branch, parse, pull_request_memory, reanchor_merged, verify, DeleteEvent, Event, Merge,
    PullRequestEvent, RepoMap, DELIVERY_HEADER, EVENT_HEADER, SIGNATURE_HEADER, SYSTEM_USER,
};
use antumbra_store::repo::memory;

use super::{bad_request, internal_error, HttpState};

/// Deliveries larger than this are refused before verification; a pull request
/// payload is tens of kilobytes.
const MAX_BODY: usize = 2 * 1024 * 1024;

/// The receiver's configuration: the App's webhook secret and the
/// repository-to-workspace map.
pub struct GithubConfig {
    secret: Vec<u8>,
    repos: RepoMap,
}

// Hand-written so the secret never reaches a log or a panic message.
impl std::fmt::Debug for GithubConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GithubConfig")
            .field("secret", &"<redacted>")
            .field("repos", &self.repos)
            .finish()
    }
}

impl GithubConfig {
    pub fn new(secret: impl Into<Vec<u8>>, repos: RepoMap) -> Self {
        Self {
            secret: secret.into(),
            repos,
        }
    }

    /// From the server flags: `None` when nothing is configured; an error when
    /// only half is (a secret with no workspace to write into, or a map with
    /// no secret to verify against). The map file is read once, here.
    pub fn from_flags(
        secret: Option<String>,
        tenant: Option<String>,
        repos_file: Option<&std::path::Path>,
    ) -> Result<Option<Self>> {
        let repos = match (tenant, repos_file) {
            (Some(tenant), None) => Some(RepoMap::all(tenant.trim())),
            (None, Some(path)) => {
                let text = std::fs::read_to_string(path).with_context(|| {
                    format!("cannot read the GitHub repository map {}", path.display())
                })?;
                Some(RepoMap::from_json(&text).with_context(|| {
                    format!("the GitHub repository map {} is invalid", path.display())
                })?)
            }
            (Some(_), Some(_)) => bail!("pass --github-tenant or --github-repos, not both"),
            (None, None) => None,
        };
        match (secret, repos) {
            (None, None) => Ok(None),
            (Some(secret), Some(repos)) => Ok(Some(Self::new(secret, repos))),
            (Some(_), None) => bail!(
                "--github-webhook-secret needs --github-tenant or --github-repos: which workspace \
                 do a repository's memories live in?"
            ),
            (None, Some(_)) => bail!(
                "--github-tenant / --github-repos need --github-webhook-secret: deliveries must be \
                 verified before they can touch memories"
            ),
        }
    }
}

/// What a delivery did, returned to GitHub (and visible in its delivery log).
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Outcome {
    pub event: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repo: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,
    /// Memories moved from the merged branch to the merge commit.
    #[serde(default)]
    pub reanchored: usize,
    /// Memories marked orphaned by a branch deletion.
    #[serde(default)]
    pub orphaned: usize,
    /// Memories written (the pull request memory).
    #[serde(default)]
    pub stored: Vec<String>,
    /// Why nothing was done, when nothing was.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ignored: Option<String>,
}

impl Outcome {
    fn ignored(event: &str, repo: Option<String>, why: impl Into<String>) -> Self {
        Self {
            event: event.to_string(),
            repo,
            tenant: None,
            reanchored: 0,
            orphaned: 0,
            stored: Vec::new(),
            ignored: Some(why.into()),
        }
    }
}

pub(super) async fn handle(State(state): State<Arc<HttpState>>, req: Request<Body>) -> Response {
    let Some(config) = state.github.as_ref() else {
        return (StatusCode::NOT_FOUND, "github integration not configured").into_response();
    };
    let (parts, body) = req.into_parts();
    let header = |name: &str| {
        parts
            .headers
            .get(name)
            .and_then(|h| h.to_str().ok())
            .map(str::to_owned)
    };
    let signature = header(SIGNATURE_HEADER);
    let kind = header(EVENT_HEADER).unwrap_or_default();
    let delivery = header(DELIVERY_HEADER).unwrap_or_else(|| "-".into());
    let body = match axum::body::to_bytes(body, MAX_BODY).await {
        Ok(b) => b,
        Err(_) => return bad_request("delivery body unreadable or too large"),
    };
    // Verify before reading anything else: an unsigned delivery learns nothing.
    if let Err(e) = verify(&config.secret, &body, signature.as_deref()) {
        eprintln!("antumbra-mcp: rejected github delivery {delivery}: {e}");
        return (
            StatusCode::UNAUTHORIZED,
            "invalid or missing webhook signature",
        )
            .into_response();
    }
    let event = match parse(&kind, &body) {
        Ok(e) => e,
        Err(e) => return bad_request(&e.to_string()),
    };
    match apply(&state, &config.repos, event).await {
        Ok(outcome) => {
            eprintln!(
                "antumbra-mcp: github delivery {delivery} ({}): reanchored {}, orphaned {}, stored {}{}",
                outcome.event,
                outcome.reanchored,
                outcome.orphaned,
                outcome.stored.len(),
                outcome
                    .ignored
                    .as_deref()
                    .map_or(String::new(), |why| format!("; ignored: {why}"))
            );
            Json(outcome).into_response()
        }
        Err(e) => {
            eprintln!("antumbra-mcp: github delivery {delivery} ({kind}) failed: {e:#}");
            internal_error()
        }
    }
}

async fn apply(state: &HttpState, repos: &RepoMap, event: Event) -> Result<Outcome> {
    match event {
        Event::Ping => Ok(Outcome::ignored("ping", None, "pong")),
        Event::Other(kind) => Ok(Outcome::ignored(&kind, None, "no handler for this event")),
        Event::PullRequest(event) => apply_pull_request(state, repos, &event).await,
        Event::Delete(event) => apply_delete(state, repos, &event).await,
    }
}

async fn apply_pull_request(
    state: &HttpState,
    repos: &RepoMap,
    event: &PullRequestEvent,
) -> Result<Outcome> {
    const KIND: &str = "pull_request";
    let repo = event.repository.slug();
    if !event.is_merge() {
        return Ok(Outcome::ignored(
            KIND,
            repo,
            format!("action '{}' is not a merge", event.action),
        ));
    }
    let Some(merge) = Merge::from_event(event) else {
        return Ok(Outcome::ignored(
            KIND,
            repo,
            "merged without a merge commit, or the repository has no slug",
        ));
    };
    let Some(tenant) = repos.tenant_for(&merge.repo) else {
        return Ok(Outcome::ignored(
            KIND,
            repo,
            "repository is not mapped to a workspace",
        ));
    };

    // Owner mode across the workspace, serialized with every other owner-side
    // write; the section is one list, the changed rows, and one insert.
    let _guard = state.auth.lock().await;
    state.store.signin_root().await?;
    let compartment =
        crate::provision_identity(&state.store, tenant, &UserId::new(SYSTEM_USER)).await?;
    let now = Utc::now();
    let changed = reanchor_merged(memory::list(&state.store, tenant).await?, &merge, now);
    for m in &changed {
        memory::upsert(&state.store, m).await?;
    }
    let mut remembered = pull_request_memory(tenant, &compartment, event, &merge, now);
    let embedding = state
        .embedder_for(tenant)
        .await
        .embed(&remembered.content)
        .await
        .context("embedding the pull request memory")?;
    remembered = remembered.with_embedding(embedding);
    memory::upsert(&state.store, &remembered).await?;
    Ok(Outcome {
        event: KIND.into(),
        repo: Some(merge.repo),
        tenant: Some(tenant.as_str().to_string()),
        reanchored: changed.len(),
        orphaned: 0,
        stored: vec![remembered.id.as_str().to_string()],
        ignored: None,
    })
}

async fn apply_delete(state: &HttpState, repos: &RepoMap, event: &DeleteEvent) -> Result<Outcome> {
    const KIND: &str = "delete";
    let Some(repo) = event.repository.slug() else {
        return Ok(Outcome::ignored(KIND, None, "the repository has no slug"));
    };
    if !event.is_branch() {
        return Ok(Outcome::ignored(
            KIND,
            Some(repo),
            format!("a deleted {} orphans nothing", event.ref_type),
        ));
    }
    let Some(tenant) = repos.tenant_for(&repo) else {
        return Ok(Outcome::ignored(
            KIND,
            Some(repo),
            "repository is not mapped to a workspace",
        ));
    };
    let _guard = state.auth.lock().await;
    state.store.signin_root().await?;
    let now = Utc::now();
    let changed = orphan_branch(
        memory::list(&state.store, tenant).await?,
        &repo,
        &event.name,
        now,
    );
    for m in &changed {
        memory::upsert(&state.store, m).await?;
    }
    Ok(Outcome {
        event: KIND.into(),
        repo: Some(repo),
        tenant: Some(tenant.as_str().to_string()),
        reanchored: 0,
        orphaned: changed.len(),
        stored: Vec::new(),
        ignored: None,
    })
}

#[cfg(test)]
mod tests {
    use super::super::{router, Bounded, Serving, MAX_EMBEDDERS, MAX_SESSIONS};
    use super::*;
    use antumbra_core::testing::FixedEmbedder;
    use antumbra_core::{orphan_of, GitProvenance, Memory, MemoryNetwork, TenantId};
    use antumbra_github::sign;
    use antumbra_store::{Store, EMBED_DIM};
    use axum::http::header;
    use serde_json::json;
    use tokio::sync::Mutex;
    use tower::ServiceExt;

    const SECRET: &[u8] = b"webhook-test-secret";
    const TENANT: &str = "ws:acme";
    const REPO: &str = "github.com/acme/orders";
    const MERGE_SHA: &str = "fedcba9876543210fedcba9876543210fedcba98";

    async fn state(repos: Option<RepoMap>) -> Arc<HttpState> {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        Arc::new(HttpState {
            serving: Serving::Shared(store.clone()),
            store,
            host: "test".into(),
            verifier: crate::auth::JwtVerifier::hs256(b"test-secret"),
            embedder: Arc::new(FixedEmbedder::new(EMBED_DIM)),
            embedders: Mutex::new(Bounded::new(MAX_EMBEDDERS)),
            auth: Mutex::new(()),
            auto_propose: None,
            auto_consolidate: false,
            serve: None,
            reranker: None,
            copal: None,
            profile: None,
            github: repos.map(|r| Arc::new(GithubConfig::new(SECRET, r))),
            sessions: Mutex::new(Bounded::new(MAX_SESSIONS)),
            consolidating: Arc::new(Mutex::new(std::collections::HashSet::new())),
            registry: crate::notify::PeerRegistry::new(),
        })
    }

    fn delivery(kind: &str, body: &[u8], signature: Option<String>) -> Request<Body> {
        let mut b = Request::builder()
            .method("POST")
            .uri("/github/webhook")
            .header(header::CONTENT_TYPE, "application/json")
            .header(EVENT_HEADER, kind)
            .header(DELIVERY_HEADER, "d-1");
        if let Some(s) = signature {
            b = b.header(SIGNATURE_HEADER, s);
        }
        b.body(Body::from(body.to_vec())).unwrap()
    }

    fn signed(kind: &str, payload: &serde_json::Value) -> Request<Body> {
        let body = serde_json::to_vec(payload).unwrap();
        let signature = sign(SECRET, &body);
        delivery(kind, &body, Some(signature))
    }

    async fn outcome(resp: Response) -> Outcome {
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    fn merged_pr() -> serde_json::Value {
        json!({
            "action": "closed",
            "pull_request": {
                "number": 42,
                "title": "Move orders to the outbox pattern",
                "html_url": "https://github.com/Acme/Orders/pull/42",
                "merged": true,
                "merge_commit_sha": MERGE_SHA,
                "base": { "ref": "main", "sha": "1111111111111111111111111111111111111111" },
                "head": { "ref": "feat/outbox", "sha": "2222222222222222222222222222222222222222" },
                "user": { "login": "shon" },
                "commits": 3, "additions": 120, "deletions": 40, "changed_files": 5
            },
            "repository": { "full_name": "Acme/Orders", "html_url": "https://github.com/Acme/Orders" }
        })
    }

    fn branch_delete(branch: &str) -> serde_json::Value {
        json!({
            "ref": branch,
            "ref_type": "branch",
            "repository": { "full_name": "Acme/Orders", "html_url": "https://github.com/Acme/Orders" }
        })
    }

    async fn seed(state: &HttpState, id: &str, evidence: &str) {
        let m = Memory::new(
            id,
            TENANT,
            MemoryNetwork::World,
            format!("about {evidence}"),
            0.6,
            Utc::now(),
        )
        .with_evidence(vec![evidence.to_string()]);
        memory::upsert(&state.store, &m).await.unwrap();
    }

    async fn evidence_of(state: &HttpState, id: &str) -> Vec<String> {
        memory::get(
            &state.store,
            &TenantId::new(TENANT),
            &antumbra_core::MemoryId::new(id),
        )
        .await
        .unwrap()
        .expect("the memory exists")
        .evidence
    }

    #[tokio::test]
    async fn an_unsigned_or_missigned_delivery_is_unauthorized() {
        let st = state(Some(RepoMap::all(TENANT))).await;
        let body = serde_json::to_vec(&merged_pr()).unwrap();
        let resp = router(st.clone())
            .oneshot(delivery("pull_request", &body, None))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        let forged = sign(b"another secret", &body);
        let resp = router(st.clone())
            .oneshot(delivery("pull_request", &body, Some(forged)))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        // A tampered body no longer matches its signature.
        let signature = sign(SECRET, &body);
        let mut tampered = merged_pr();
        tampered["pull_request"]["number"] = json!(43);
        let resp = router(st)
            .oneshot(delivery(
                "pull_request",
                &serde_json::to_vec(&tampered).unwrap(),
                Some(signature),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn an_unconfigured_receiver_is_not_found() {
        let st = state(None).await;
        let resp = router(st)
            .oneshot(signed("ping", &json!({"zen": "x"})))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn a_merge_reanchors_the_branch_and_remembers_the_pull_request() {
        let st = state(Some(RepoMap::all(TENANT))).await;
        seed(
            &st,
            "memory:feat",
            &format!("git:{REPO}@2222222#feat/outbox:src/orders.rs"),
        )
        .await;
        seed(&st, "memory:main", &format!("git:{REPO}@1111111#main")).await;
        seed(
            &st,
            "memory:elsewhere",
            "git:github.com/acme/billing@2222222#feat/outbox",
        )
        .await;

        let out = outcome(
            router(st.clone())
                .oneshot(signed("pull_request", &merged_pr()))
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(out.event, "pull_request");
        assert_eq!(out.repo.as_deref(), Some(REPO));
        assert_eq!(out.tenant.as_deref(), Some(TENANT));
        assert_eq!(out.reanchored, 1);
        assert_eq!(
            out.stored,
            vec!["memory:github-pr-github-com-acme-orders-42"]
        );
        assert_eq!(out.ignored, None);

        let feat = evidence_of(&st, "memory:feat").await;
        let anchor = GitProvenance::from_evidence(&feat).unwrap();
        assert_eq!(anchor.commit, MERGE_SHA);
        assert_eq!(anchor.branch.as_deref(), Some("main"));
        assert_eq!(anchor.path.as_deref(), Some("src/orders.rs"));
        assert_eq!(feat.len(), 2, "the old anchor is kept as history");
        assert_eq!(evidence_of(&st, "memory:main").await.len(), 1);
        assert_eq!(evidence_of(&st, "memory:elsewhere").await.len(), 1);

        let pr = memory::get(
            &st.store,
            &TenantId::new(TENANT),
            &antumbra_core::MemoryId::new("memory:github-pr-github-com-acme-orders-42"),
        )
        .await
        .unwrap()
        .expect("the pull request memory");
        assert!(pr.content.contains("Pull request #42 merged into main"));
        assert!(pr.embedding.is_some(), "embedded so recall can find it");
        assert_eq!(pr.author.as_ref().map(|u| u.as_str()), Some(SYSTEM_USER));
        assert_eq!(pr.evidence[1], "https://github.com/Acme/Orders/pull/42");

        // Redelivery: nothing left to move, the same memory revised in place.
        let again = outcome(
            router(st.clone())
                .oneshot(signed("pull_request", &merged_pr()))
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(again.reanchored, 0);
        assert_eq!(again.stored.len(), 1);
        assert_eq!(evidence_of(&st, "memory:feat").await.len(), 2);
    }

    #[tokio::test]
    async fn a_branch_delete_orphans_its_memories_but_not_reanchored_ones() {
        let st = state(Some(RepoMap::all(TENANT))).await;
        seed(
            &st,
            "memory:feat",
            &format!("git:{REPO}@2222222#feat/outbox"),
        )
        .await;
        seed(
            &st,
            "memory:other",
            &format!("git:{REPO}@3333333#feat/other"),
        )
        .await;

        // Merge first (GitHub's delete-after-merge order), then delete.
        outcome(
            router(st.clone())
                .oneshot(signed("pull_request", &merged_pr()))
                .await
                .unwrap(),
        )
        .await;
        let out = outcome(
            router(st.clone())
                .oneshot(signed("delete", &branch_delete("feat/outbox")))
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(out.event, "delete");
        assert_eq!(out.orphaned, 0, "the merge already moved it to main");
        assert_eq!(orphan_of(&evidence_of(&st, "memory:feat").await), None);

        let out = outcome(
            router(st.clone())
                .oneshot(signed("delete", &branch_delete("feat/other")))
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(out.orphaned, 1);
        let other = evidence_of(&st, "memory:other").await;
        assert_eq!(
            orphan_of(&other).map(|o| o.branch),
            Some("feat/other".to_string())
        );
        // Redelivery is a no-op.
        let out = outcome(
            router(st.clone())
                .oneshot(signed("delete", &branch_delete("feat/other")))
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(out.orphaned, 0);
    }

    #[tokio::test]
    async fn pings_unhandled_events_and_unmapped_repositories_are_acknowledged() {
        let repos = RepoMap::from_json(r#"{"github.com/acme/billing": "ws:acme"}"#).unwrap();
        let st = state(Some(repos)).await;
        let out = outcome(
            router(st.clone())
                .oneshot(signed(
                    "ping",
                    &json!({"zen": "Keep it logically awesome."}),
                ))
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(out.ignored.as_deref(), Some("pong"));
        let out = outcome(
            router(st.clone())
                .oneshot(signed("issues", &json!({"action": "opened"})))
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(out.event, "issues");
        assert!(out.ignored.is_some());
        // The orders repository is not in the map: acknowledged, untouched.
        let out = outcome(
            router(st.clone())
                .oneshot(signed("pull_request", &merged_pr()))
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(out.reanchored, 0);
        assert!(out.stored.is_empty());
        assert_eq!(
            out.ignored.as_deref(),
            Some("repository is not mapped to a workspace")
        );
        // A close without a merge, and a tag deletion, do nothing.
        let mut closed = merged_pr();
        closed["pull_request"]["merged"] = json!(false);
        let out = outcome(
            router(st.clone())
                .oneshot(signed("pull_request", &closed))
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(
            out.ignored.as_deref(),
            Some("action 'closed' is not a merge")
        );
        let mut tag = branch_delete("v1");
        tag["ref_type"] = json!("tag");
        let out = outcome(router(st).oneshot(signed("delete", &tag)).await.unwrap()).await;
        assert_eq!(
            out.ignored.as_deref(),
            Some("a deleted tag orphans nothing")
        );
        // Garbage that verifies is still the sender's fault.
        let body = b"not json".to_vec();
        let signature = sign(SECRET, &body);
        let st = state(Some(RepoMap::all(TENANT))).await;
        let resp = router(st)
            .oneshot(delivery("pull_request", &body, Some(signature)))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn from_flags_requires_both_halves() {
        assert!(GithubConfig::from_flags(None, None, None)
            .unwrap()
            .is_none());
        let err = GithubConfig::from_flags(Some("s".into()), None, None).unwrap_err();
        assert!(err.to_string().contains("--github-tenant"), "{err}");
        let err = GithubConfig::from_flags(None, Some("ws:x".into()), None).unwrap_err();
        assert!(err.to_string().contains("--github-webhook-secret"), "{err}");
        let cfg = GithubConfig::from_flags(Some("s".into()), Some(" ws:x ".into()), None)
            .unwrap()
            .unwrap();
        assert_eq!(
            cfg.repos.tenant_for("github.com/a/b").map(|t| t.as_str()),
            Some("ws:x")
        );
        let map =
            std::env::temp_dir().join(format!("antumbra-github-repos-{}.json", std::process::id()));
        std::fs::write(&map, r#"{"github.com/acme/orders": "ws:acme"}"#).unwrap();
        let cfg = GithubConfig::from_flags(Some("s".into()), None, Some(&map))
            .unwrap()
            .unwrap();
        assert_eq!(cfg.repos.mapped(), Some(1));
        let err = GithubConfig::from_flags(Some("s".into()), Some("ws:x".into()), Some(&map))
            .unwrap_err();
        assert!(err.to_string().contains("not both"), "{err}");
        std::fs::remove_file(map).ok();
        let missing = std::env::temp_dir().join("antumbra-github-repos-missing.json");
        let err = GithubConfig::from_flags(Some("s".into()), None, Some(&missing)).unwrap_err();
        assert!(err.to_string().contains("cannot read"), "{err}");
    }
}
