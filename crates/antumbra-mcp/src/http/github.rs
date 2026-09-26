//! `POST /github/webhook`: the GitHub App's deliveries (ADR-0019). The route
//! is the I/O half of `antumbra-github`: verify the HMAC signature, parse the
//! event, find the repository's workspace, load that workspace's memories,
//! apply the pure handler, write back what changed, and, when the App can
//! read repository contents, ingest the documents a merge changed or a new
//! installation brought in ([`ingest`]).
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
//! GitHub does not retry it. Document ingest runs after the response, because
//! GitHub gives a receiver ten seconds and an ingest can take longer; the
//! delivery reports how many documents were queued and the log reports what
//! each one did.

mod ingest;
mod knowledge;
#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use chrono::Utc;
use tokio::sync::Mutex;

use antumbra_core::UserId;
use antumbra_github::{
    orphan_branch, parse, pull_request_memory, reanchor_merged, verify, AppCredentials,
    DeleteEvent, Event, GithubApi, InstallationEvent, InstallationToken, Merge, PullRequestEvent,
    RepoMap, DEFAULT_API_URL, DELIVERY_HEADER, EVENT_HEADER, SIGNATURE_HEADER, SYSTEM_USER,
};
use antumbra_store::repo::memory;

use super::{bad_request, internal_error, HttpState};

/// Deliveries larger than this are refused before verification; a pull request
/// payload is tens of kilobytes.
const MAX_BODY: usize = 2 * 1024 * 1024;

/// The receiver's configuration: the App's webhook secret, the
/// repository-to-workspace map, and, when the App may read contents, its
/// credentials and the API to read them through.
pub struct GithubConfig {
    secret: Vec<u8>,
    repos: RepoMap,
    app: Option<AppCredentials>,
    api: GithubApi,
    /// Installation tokens, minted on demand and reused until they are about
    /// to expire.
    tokens: Mutex<HashMap<u64, InstallationToken>>,
    /// Post the knowledge diff on pull requests (off by default).
    knowledge_diff: bool,
}

// Hand-written so the secret never reaches a log or a panic message.
impl std::fmt::Debug for GithubConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GithubConfig")
            .field("secret", &"<redacted>")
            .field("repos", &self.repos)
            .field("app", &self.app)
            .field("api", &self.api.base())
            .finish()
    }
}

impl GithubConfig {
    /// Verify with `secret`, land repositories per `repos`; no App
    /// credentials, so re-anchor and orphan only.
    pub fn new(secret: impl Into<Vec<u8>>, repos: RepoMap) -> Self {
        Self {
            secret: secret.into(),
            repos,
            app: None,
            api: GithubApi::new(DEFAULT_API_URL),
            tokens: Mutex::new(HashMap::new()),
            knowledge_diff: false,
        }
    }

    /// Post the knowledge diff as a check run on every pull request opened or
    /// pushed to in a mapped repository. Needs the App, with the Checks
    /// permission (write).
    pub fn with_knowledge_diff(mut self, on: bool) -> Self {
        self.knowledge_diff = on;
        self
    }

    /// Let the receiver read repository contents as the App.
    pub fn with_app(mut self, app: AppCredentials) -> Self {
        self.app = Some(app);
        self
    }

    /// Read through this API (an Enterprise Server base, or a test seam).
    pub fn with_api(mut self, api: GithubApi) -> Self {
        self.api = api;
        self
    }

    /// From the server flags: `None` when nothing is configured; an error when
    /// only half is (a secret with no workspace to write into, a map with no
    /// secret to verify against, an App id without its key). Files are read
    /// once, here.
    pub fn from_flags(
        secret: Option<String>,
        tenant: Option<String>,
        repos_file: Option<&std::path::Path>,
        app_id: Option<String>,
        app_key_file: Option<&std::path::Path>,
        api_url: &str,
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
        let app = match (app_id, app_key_file) {
            (Some(id), Some(path)) => {
                let pem = std::fs::read(path).with_context(|| {
                    format!("cannot read the GitHub App key {}", path.display())
                })?;
                Some(AppCredentials::new(id, &pem)?)
            }
            (None, None) => None,
            _ => bail!("--github-app-id and --github-app-key-file go together"),
        };
        let mut config = match (secret, repos) {
            (None, None) if app.is_none() => return Ok(None),
            (Some(secret), Some(repos)) => Self::new(secret, repos),
            (Some(_), None) => bail!(
                "--github-webhook-secret needs --github-tenant or --github-repos: which workspace \
                 do a repository's memories live in?"
            ),
            (None, _) => bail!(
                "--github-tenant / --github-repos / --github-app-id need --github-webhook-secret: \
                 deliveries must be verified before they can touch anything"
            ),
        };
        if let Some(app) = app {
            config = config.with_app(app);
        }
        Ok(Some(config.with_api(GithubApi::new(api_url))))
    }

    pub(super) fn api(&self) -> &GithubApi {
        &self.api
    }

    /// Whether the receiver can read repository contents.
    pub fn reads_contents(&self) -> bool {
        self.app.is_some()
    }

    /// A token for `installation_id`: cached while fresh, minted otherwise.
    /// `None` when no App credentials are configured.
    pub(super) async fn installation_token(&self, installation_id: u64) -> Result<Option<String>> {
        let Some(app) = &self.app else {
            return Ok(None);
        };
        let now = Utc::now();
        let mut tokens = self.tokens.lock().await;
        if let Some(cached) = tokens.get(&installation_id) {
            if cached.is_fresh(now) {
                return Ok(Some(cached.token.clone()));
            }
        }
        let minted = self
            .api
            .installation_token(app, installation_id, now)
            .await
            .with_context(|| format!("minting a token for installation {installation_id}"))?;
        let token = minted.token.clone();
        tokens.insert(installation_id, minted);
        Ok(Some(token))
    }
}

/// What a delivery did, returned to GitHub (and visible in its delivery log).
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
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
    /// Documents queued for ingest after this response (the log reports each).
    #[serde(default)]
    pub ingest_queued: usize,
    /// Whether the knowledge diff was queued, to post as a check run after
    /// this response.
    #[serde(default)]
    pub knowledge_diff_queued: bool,
    /// Why nothing (or not everything) was done.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ignored: Option<String>,
}

impl Outcome {
    fn ignored(event: &str, repo: Option<String>, why: impl Into<String>) -> Self {
        Self {
            event: event.to_string(),
            repo,
            ignored: Some(why.into()),
            ..Self::default()
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
    match apply(&state, event).await {
        Ok(outcome) => {
            eprintln!(
                "antumbra-mcp: github delivery {delivery} ({}): reanchored {}, orphaned {}, stored {}, \
                 ingest queued {}{}",
                outcome.event,
                outcome.reanchored,
                outcome.orphaned,
                outcome.stored.len(),
                outcome.ingest_queued,
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

async fn apply(state: &Arc<HttpState>, event: Event) -> Result<Outcome> {
    match event {
        Event::Ping => Ok(Outcome::ignored("ping", None, "pong")),
        Event::Other(kind) => Ok(Outcome::ignored(&kind, None, "no handler for this event")),
        Event::PullRequest(event) => apply_pull_request(state, &event).await,
        Event::Delete(event) => apply_delete(state, &event).await,
        Event::Installation(event) => apply_installation(state, &event).await,
    }
}

fn config(state: &HttpState) -> &GithubConfig {
    state
        .github
        .as_ref()
        .expect("the route is only reachable when the integration is configured")
}

async fn apply_pull_request(state: &Arc<HttpState>, event: &PullRequestEvent) -> Result<Outcome> {
    const KIND: &str = "pull_request";
    let cfg = config(state);
    let repo = event.repository.slug();
    if knowledge::answers(event) {
        return Ok(knowledge::queue(state, event));
    }
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
    let Some(tenant) = cfg.repos.tenant_for(&merge.repo) else {
        return Ok(Outcome::ignored(
            KIND,
            repo,
            "repository is not mapped to a workspace",
        ));
    };

    // Owner mode across the workspace, serialized with every other owner-side
    // write; the section is one list, the changed rows, and one insert.
    let (changed, remembered) = {
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
        (changed, remembered)
    };

    // The documents the merge changed, ingested after the response.
    let plan = ingest::plan_for_merge(cfg, event, &merge, tenant).await?;
    let ingest_queued = plan.as_ref().map_or(0, |p| p.paths.len());
    if let Some(plan) = plan {
        ingest::spawn(state.clone(), plan);
    }
    Ok(Outcome {
        event: KIND.into(),
        repo: Some(merge.repo),
        tenant: Some(tenant.as_str().to_string()),
        reanchored: changed.len(),
        stored: vec![remembered.id.as_str().to_string()],
        ingest_queued,
        ..Outcome::default()
    })
}

async fn apply_delete(state: &Arc<HttpState>, event: &DeleteEvent) -> Result<Outcome> {
    const KIND: &str = "delete";
    let cfg = config(state);
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
    let Some(tenant) = cfg.repos.tenant_for(&repo) else {
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
        orphaned: changed.len(),
        ..Outcome::default()
    })
}

/// The cold start: every mapped repository the installation brings in has its
/// knowledge documents ingested at the head of its default branch.
async fn apply_installation(state: &Arc<HttpState>, event: &InstallationEvent) -> Result<Outcome> {
    const KIND: &str = "installation";
    let cfg = config(state);
    if !event.is_cold_start() {
        return Ok(Outcome::ignored(
            KIND,
            None,
            format!("action '{}' adds no repositories", event.action),
        ));
    }
    if !cfg.reads_contents() {
        return Ok(Outcome::ignored(
            KIND,
            None,
            "no App credentials configured, so repository contents cannot be read",
        ));
    }
    let mut queued = 0;
    let mut unmapped = Vec::new();
    let mut problems = Vec::new();
    for repo in &event.repositories {
        let slug = cfg.api.repo_slug(&repo.full_name);
        let Some(tenant) = cfg.repos.tenant_for(&slug) else {
            unmapped.push(slug);
            continue;
        };
        match ingest::plan_for_repository(cfg, &repo.full_name, event.installation.id, tenant).await
        {
            Ok(plan) => {
                queued += plan.paths.len();
                ingest::spawn(state.clone(), plan);
            }
            // One repository that cannot be read (empty, archived, a permission
            // gap) must not fail the others' cold start.
            Err(e) => {
                eprintln!(
                    "antumbra-mcp: github cold start of {} failed: {e:#}",
                    repo.full_name
                );
                problems.push(format!("{}: {e:#}", repo.full_name));
            }
        }
    }
    let mut notes = Vec::new();
    if !unmapped.is_empty() {
        notes.push(format!(
            "not mapped to a workspace: {}",
            unmapped.join(", ")
        ));
    }
    if !problems.is_empty() {
        notes.push(format!("could not plan: {}", problems.join("; ")));
    }
    Ok(Outcome {
        event: KIND.into(),
        ingest_queued: queued,
        ignored: (!notes.is_empty()).then(|| notes.join(". ")),
        ..Outcome::default()
    })
}
