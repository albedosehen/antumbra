//! The slice of the GitHub REST API the integration reads, as the App: an
//! installation token, a merged pull request's changed files, a file's
//! contents at a commit, a branch head, and a repository's tree. Calls are
//! blocking `ureq` requests run off the async runtime (`spawn_blocking`); the
//! transport is a trait so a test can answer from canned routes.

use std::sync::Arc;

use chrono::{DateTime, Duration, Utc};
use serde::Deserialize;
use serde_json::Value;

use antumbra_core::normalize_repo;

use crate::app::{AppCredentials, AppError};
use crate::knowledge::CheckOutput;

/// GitHub's public API. An Enterprise Server host is `https://<host>/api/v3`.
pub const DEFAULT_API_URL: &str = "https://api.github.com";
const USER_AGENT: &str = "antumbra-github";
const API_VERSION: &str = "2022-11-28";
const ACCEPT_JSON: &str = "application/vnd.github+json";
const ACCEPT_RAW: &str = "application/vnd.github.raw+json";
/// A page of changed files, and how many pages are read before giving up on
/// a pull request that touched thousands of files.
const FILES_PER_PAGE: usize = 100;
const MAX_FILE_PAGES: usize = 10;
/// Default per-request budget, overridable with `ANTUMBRA_GITHUB_TIMEOUT_SECS`.
const TIMEOUT_SECS: u64 = 20;

#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("GitHub API call failed: {0}")]
    Transport(String),
    #[error("GitHub API {url} answered {status}: {body}")]
    Status {
        url: String,
        status: u16,
        body: String,
    },
    #[error("GitHub API response was not the expected shape: {0}")]
    Shape(String),
    #[error(transparent)]
    App(#[from] AppError),
    #[error("the GitHub API worker was canceled")]
    Canceled,
}

/// A raw response: status and body, before any interpretation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpResponse {
    pub status: u16,
    pub body: Vec<u8>,
}

/// The wire, as a trait so tests can answer from canned routes. Non-2xx
/// statuses come back as responses, not errors; the caller decides.
pub trait GithubTransport: Send + Sync {
    fn get(&self, url: &str, bearer: &str, accept: &str) -> Result<HttpResponse, ApiError>;
    fn post_json(&self, url: &str, bearer: &str, body: &Value) -> Result<HttpResponse, ApiError>;
}

/// An installation's token: what every repository read authenticates with.
#[derive(Clone, PartialEq, Eq, Deserialize)]
pub struct InstallationToken {
    pub token: String,
    pub expires_at: DateTime<Utc>,
}

// Hand-written so the token never reaches a log or a panic message.
impl std::fmt::Debug for InstallationToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InstallationToken")
            .field("token", &"<redacted>")
            .field("expires_at", &self.expires_at)
            .finish()
    }
}

impl InstallationToken {
    /// Usable for a little while yet: a token about to expire is replaced
    /// before a long ingest starts on it.
    pub fn is_fresh(&self, now: DateTime<Utc>) -> bool {
        self.expires_at - now > Duration::minutes(2)
    }
}

/// One file a pull request changed.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ChangedFile {
    #[serde(rename = "filename")]
    pub path: String,
    /// `added`, `modified`, `renamed`, `removed`, ...
    pub status: String,
    /// Where a `renamed` file used to live.
    #[serde(default, rename = "previous_filename")]
    pub previous_path: Option<String>,
}

impl ChangedFile {
    /// Whether the file still exists after the change.
    pub fn is_present(&self) -> bool {
        self.status != "removed"
    }

    /// The path this change took out of the tree, if any: the file itself when
    /// it was removed, its old path when it was renamed. A document ingested
    /// from that path describes nothing that exists any more.
    pub fn vacated_path(&self) -> Option<&str> {
        match self.status.as_str() {
            "removed" => Some(self.path.as_str()),
            "renamed" => self.previous_path.as_deref(),
            _ => None,
        }
    }
}

/// A repository tree's file paths.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Tree {
    pub paths: Vec<String>,
    /// GitHub truncates very large trees; the listing is then incomplete.
    pub truncated: bool,
}

/// The API client.
pub struct GithubApi {
    base: String,
    transport: Arc<dyn GithubTransport>,
}

impl GithubApi {
    /// Against `base` (see [`DEFAULT_API_URL`]) over the real wire.
    pub fn new(base: &str) -> Self {
        Self::with_transport(base, Arc::new(UreqTransport::new()))
    }

    /// As [`Self::new`] with an explicit transport: the test seam.
    pub fn with_transport(base: &str, transport: Arc<dyn GithubTransport>) -> Self {
        Self {
            base: base.trim().trim_end_matches('/').to_string(),
            transport,
        }
    }

    pub fn base(&self) -> &str {
        &self.base
    }

    /// The host repositories are browsed on: `github.com` for the public API,
    /// the Enterprise Server host otherwise.
    pub fn web_host(&self) -> String {
        let host = self
            .base
            .split_once("://")
            .map_or(self.base.as_str(), |(_, rest)| rest)
            .split('/')
            .next()
            .unwrap_or_default()
            .to_ascii_lowercase();
        if host == "api.github.com" {
            "github.com".to_string()
        } else {
            host
        }
    }

    /// The slug memories anchor to for `owner/name`.
    pub fn repo_slug(&self, full_name: &str) -> String {
        normalize_repo(&format!("{}/{full_name}", self.web_host()))
    }

    /// The browsable URL of `path` at `commit`.
    pub fn blob_url(&self, full_name: &str, commit: &str, path: &str) -> String {
        format!(
            "https://{}/{full_name}/blob/{commit}/{path}",
            self.web_host()
        )
    }

    /// Exchange the App's JWT for an installation token.
    pub async fn installation_token(
        &self,
        app: &AppCredentials,
        installation_id: u64,
        now: DateTime<Utc>,
    ) -> Result<InstallationToken, ApiError> {
        let jwt = app.jwt(now)?;
        let url = format!(
            "{}/app/installations/{installation_id}/access_tokens",
            self.base
        );
        let resp = self
            .call(move |t| t.post_json(&url, &jwt, &Value::Object(Default::default())))
            .await?;
        expect_success(&resp, "installation token")?;
        serde_json::from_slice(&resp.body).map_err(|e| ApiError::Shape(e.to_string()))
    }

    /// Post a completed, neutral check run named `name` on `head_sha`: an
    /// informational report that never blocks a merge. Needs the App's Checks
    /// permission (write).
    pub async fn create_check_run(
        &self,
        token: &str,
        full_name: &str,
        head_sha: &str,
        name: &str,
        output: &CheckOutput,
    ) -> Result<(), ApiError> {
        let url = format!("{}/repos/{full_name}/check-runs", self.base);
        let body = serde_json::json!({
            "name": name,
            "head_sha": head_sha,
            "status": "completed",
            "conclusion": "neutral",
            "output": {
                "title": output.title,
                "summary": output.summary,
                "text": output.text,
            },
        });
        let token = token.to_string();
        let resp = self.call(move |t| t.post_json(&url, &token, &body)).await?;
        expect_success(&resp, "check run")
    }

    /// Every file a pull request changed (paged; capped at
    /// `FILES_PER_PAGE * MAX_FILE_PAGES`).
    pub async fn pull_request_files(
        &self,
        token: &str,
        full_name: &str,
        number: u64,
    ) -> Result<Vec<ChangedFile>, ApiError> {
        let mut files = Vec::new();
        for page in 1..=MAX_FILE_PAGES {
            let url = format!(
                "{}/repos/{full_name}/pulls/{number}/files?per_page={FILES_PER_PAGE}&page={page}",
                self.base
            );
            let resp = self.get(url.clone(), token, ACCEPT_JSON).await?;
            expect_success(&resp, "pull request files")?;
            let batch: Vec<ChangedFile> =
                serde_json::from_slice(&resp.body).map_err(|e| ApiError::Shape(e.to_string()))?;
            let full = batch.len() >= FILES_PER_PAGE;
            files.extend(batch);
            if !full {
                break;
            }
        }
        Ok(files)
    }

    /// The text of `path` at `git_ref`; `None` when the file is absent there
    /// or is not UTF-8 text.
    pub async fn file_text(
        &self,
        token: &str,
        full_name: &str,
        path: &str,
        git_ref: &str,
    ) -> Result<Option<String>, ApiError> {
        let url = format!(
            "{}/repos/{full_name}/contents/{}?ref={git_ref}",
            self.base,
            encode_path(path)
        );
        let resp = self.get(url, token, ACCEPT_RAW).await?;
        if resp.status == 404 {
            return Ok(None);
        }
        expect_success(&resp, "file contents")?;
        Ok(String::from_utf8(resp.body).ok())
    }

    /// The repository's default branch.
    pub async fn default_branch(&self, token: &str, full_name: &str) -> Result<String, ApiError> {
        let url = format!("{}/repos/{full_name}", self.base);
        let resp = self.get(url, token, ACCEPT_JSON).await?;
        expect_success(&resp, "repository")?;
        let v: Value =
            serde_json::from_slice(&resp.body).map_err(|e| ApiError::Shape(e.to_string()))?;
        v.get("default_branch")
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| ApiError::Shape("repository without default_branch".into()))
    }

    /// The commit a branch points at.
    pub async fn branch_head(
        &self,
        token: &str,
        full_name: &str,
        branch: &str,
    ) -> Result<String, ApiError> {
        let url = format!(
            "{}/repos/{full_name}/branches/{}",
            self.base,
            encode_path(branch)
        );
        let resp = self.get(url, token, ACCEPT_JSON).await?;
        expect_success(&resp, "branch")?;
        let v: Value =
            serde_json::from_slice(&resp.body).map_err(|e| ApiError::Shape(e.to_string()))?;
        v.pointer("/commit/sha")
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| ApiError::Shape("branch without commit.sha".into()))
    }

    /// Every file path in the tree at `commit`.
    pub async fn tree_paths(
        &self,
        token: &str,
        full_name: &str,
        commit: &str,
    ) -> Result<Tree, ApiError> {
        let url = format!(
            "{}/repos/{full_name}/git/trees/{commit}?recursive=1",
            self.base
        );
        let resp = self.get(url, token, ACCEPT_JSON).await?;
        expect_success(&resp, "tree")?;
        #[derive(Deserialize)]
        struct Entry {
            path: String,
            #[serde(rename = "type")]
            kind: String,
        }
        #[derive(Deserialize)]
        struct Listing {
            #[serde(default)]
            tree: Vec<Entry>,
            #[serde(default)]
            truncated: bool,
        }
        let listing: Listing =
            serde_json::from_slice(&resp.body).map_err(|e| ApiError::Shape(e.to_string()))?;
        Ok(Tree {
            paths: listing
                .tree
                .into_iter()
                .filter(|e| e.kind == "blob")
                .map(|e| e.path)
                .collect(),
            truncated: listing.truncated,
        })
    }

    async fn get(&self, url: String, token: &str, accept: &str) -> Result<HttpResponse, ApiError> {
        let token = token.to_string();
        let accept = accept.to_string();
        self.call(move |t| t.get(&url, &token, &accept)).await
    }

    /// Run one blocking transport call off the async runtime.
    async fn call<T, F>(&self, f: F) -> Result<T, ApiError>
    where
        T: Send + 'static,
        F: FnOnce(&dyn GithubTransport) -> Result<T, ApiError> + Send + 'static,
    {
        let transport = self.transport.clone();
        tokio::task::spawn_blocking(move || f(transport.as_ref()))
            .await
            .map_err(|_| ApiError::Canceled)?
    }
}

fn expect_success(resp: &HttpResponse, what: &str) -> Result<(), ApiError> {
    if (200..300).contains(&resp.status) {
        return Ok(());
    }
    let body = String::from_utf8_lossy(&resp.body);
    Err(ApiError::Status {
        url: what.to_string(),
        status: resp.status,
        body: body.chars().take(300).collect(),
    })
}

/// Percent-encode a repository path for a URL, keeping `/` (GitHub reads
/// the path segments literally) and the unreserved characters.
fn encode_path(path: &str) -> String {
    let mut out = String::with_capacity(path.len());
    for b in path.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b'/' => {
                out.push(b as char);
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// The production transport: blocking `ureq` over an agent with a bounded
/// global timeout; statuses are returned, never turned into errors, so a 404
/// can mean "absent" to the caller.
struct UreqTransport {
    agent: ureq::Agent,
}

impl UreqTransport {
    fn new() -> Self {
        let secs = std::env::var("ANTUMBRA_GITHUB_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .filter(|&s| s > 0)
            .unwrap_or(TIMEOUT_SECS);
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(std::time::Duration::from_secs(secs)))
            .http_status_as_error(false)
            .build()
            .into();
        Self { agent }
    }
}

fn read(mut resp: ureq::http::Response<ureq::Body>, url: &str) -> Result<HttpResponse, ApiError> {
    let status = resp.status().as_u16();
    let body = resp
        .body_mut()
        .read_to_vec()
        .map_err(|e| ApiError::Transport(format!("reading {url}: {e}")))?;
    Ok(HttpResponse { status, body })
}

impl GithubTransport for UreqTransport {
    fn get(&self, url: &str, bearer: &str, accept: &str) -> Result<HttpResponse, ApiError> {
        let resp = self
            .agent
            .get(url)
            .header("authorization", &format!("Bearer {bearer}"))
            .header("accept", accept)
            .header("user-agent", USER_AGENT)
            .header("x-github-api-version", API_VERSION)
            .call()
            .map_err(|e| ApiError::Transport(format!("GET {url}: {e}")))?;
        read(resp, url)
    }

    fn post_json(&self, url: &str, bearer: &str, body: &Value) -> Result<HttpResponse, ApiError> {
        let resp = self
            .agent
            .post(url)
            .header("authorization", &format!("Bearer {bearer}"))
            .header("accept", ACCEPT_JSON)
            .header("user-agent", USER_AGENT)
            .header("x-github-api-version", API_VERSION)
            .send_json(body)
            .map_err(|e| ApiError::Transport(format!("POST {url}: {e}")))?;
        read(resp, url)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{FakeTransport, APP_PRIVATE_KEY_PEM};
    use serde_json::json;

    const FULL: &str = "Acme/Orders";

    fn api(fake: FakeTransport) -> (GithubApi, Arc<FakeTransport>) {
        let fake = Arc::new(fake);
        (
            GithubApi::with_transport("https://api.github.com/", fake.clone()),
            fake,
        )
    }

    #[test]
    fn hosts_slugs_and_paths_are_derived_from_the_base() {
        let public = GithubApi::with_transport(DEFAULT_API_URL, Arc::new(FakeTransport::new()));
        assert_eq!(public.base(), "https://api.github.com");
        assert_eq!(public.web_host(), "github.com");
        assert_eq!(public.repo_slug("Acme/Orders"), "github.com/acme/orders");
        assert_eq!(
            public.blob_url("Acme/Orders", "abc1234", "docs/x.md"),
            "https://github.com/Acme/Orders/blob/abc1234/docs/x.md"
        );
        let ghes = GithubApi::with_transport(
            "https://Git.Acme.internal/api/v3",
            Arc::new(FakeTransport::new()),
        );
        assert_eq!(ghes.web_host(), "git.acme.internal");
        assert_eq!(
            ghes.repo_slug("Acme/Orders"),
            "git.acme.internal/acme/orders"
        );
        assert_eq!(encode_path("docs/a b#c.md"), "docs/a%20b%23c.md");
        assert_eq!(encode_path("feat/x-y_z.1"), "feat/x-y_z.1");
    }

    #[tokio::test]
    async fn an_installation_token_is_minted_with_the_app_jwt() {
        let (api, fake) = api(FakeTransport::new().json(
            "POST",
            "https://api.github.com/app/installations/77/access_tokens",
            201,
            &json!({"token": "ghs_x", "expires_at": "2099-01-01T00:00:00Z"}),
        ));
        let app = AppCredentials::new("1", APP_PRIVATE_KEY_PEM).unwrap();
        let tok = api.installation_token(&app, 77, Utc::now()).await.unwrap();
        assert_eq!(tok.token, "ghs_x");
        assert!(tok.is_fresh(Utc::now()));
        assert!(
            !format!("{tok:?}").contains("ghs_x"),
            "the token never prints"
        );
        let calls = fake.calls();
        assert_eq!(calls.len(), 1);
        assert!(calls[0].starts_with("POST https://api.github.com/app/installations/77"));
        // The bearer on that call was the App JWT, not an installation token.
        assert!(fake.bearers()[0].starts_with("eyJ"), "{:?}", fake.bearers());
    }

    #[tokio::test]
    async fn a_check_run_is_posted_with_the_installation_token() {
        let url = format!("https://api.github.com/repos/{FULL}/check-runs");
        let output = CheckOutput {
            title: "t".into(),
            summary: "s".into(),
            text: String::new(),
        };
        let (posting, fake) = api(FakeTransport::new().json("POST", &url, 201, &json!({"id": 1})));
        posting
            .create_check_run("ghs_x", FULL, "abc", "Antumbra knowledge diff", &output)
            .await
            .unwrap();
        assert_eq!(fake.calls(), vec![format!("POST {url}")]);
        assert_eq!(fake.bearers(), vec!["ghs_x".to_string()]);

        // Without the Checks permission GitHub answers 403, and that is an error.
        let (refusing, _) =
            api(FakeTransport::new().json("POST", &url, 403, &json!({"message": "no"})));
        let refused = refusing
            .create_check_run("ghs_x", FULL, "abc", "Antumbra knowledge diff", &output)
            .await;
        assert!(matches!(refused, Err(ApiError::Status { status: 403, .. })));
    }

    #[tokio::test]
    async fn pull_request_files_page_until_a_short_page() {
        let full_page: Vec<Value> = (0..FILES_PER_PAGE)
            .map(|i| json!({"filename": format!("f{i}.md"), "status": "modified"}))
            .collect();
        let (api, _) = api(FakeTransport::new()
            .json(
                "GET",
                &format!("https://api.github.com/repos/{FULL}/pulls/9/files?per_page=100&page=1"),
                200,
                &Value::Array(full_page),
            )
            .json(
                "GET",
                &format!("https://api.github.com/repos/{FULL}/pulls/9/files?per_page=100&page=2"),
                200,
                &json!([{"filename": "last.md", "status": "removed"}]),
            ));
        let files = api.pull_request_files("t", FULL, 9).await.unwrap();
        assert_eq!(files.len(), FILES_PER_PAGE + 1);
        assert!(files[0].is_present());
        assert!(!files[FILES_PER_PAGE].is_present());
    }

    #[tokio::test]
    async fn file_text_reads_raw_and_treats_404_and_binary_as_absent() {
        let (api, _) = api(FakeTransport::new()
            .route(
                "GET",
                &format!("https://api.github.com/repos/{FULL}/contents/docs/a%20b.md?ref=abc"),
                200,
                "# A B\n",
            )
            .route(
                "GET",
                &format!("https://api.github.com/repos/{FULL}/contents/logo.png?ref=abc"),
                200,
                vec![0xFF, 0xFE, 0x00],
            )
            .route(
                "GET",
                &format!("https://api.github.com/repos/{FULL}/contents/secret.md?ref=abc"),
                403,
                "{\"message\":\"forbidden\"}",
            ));
        assert_eq!(
            api.file_text("t", FULL, "docs/a b.md", "abc")
                .await
                .unwrap()
                .as_deref(),
            Some("# A B\n")
        );
        assert_eq!(
            api.file_text("t", FULL, "logo.png", "abc").await.unwrap(),
            None
        );
        assert_eq!(
            api.file_text("t", FULL, "gone.md", "abc").await.unwrap(),
            None
        );
        let err = api
            .file_text("t", FULL, "secret.md", "abc")
            .await
            .unwrap_err();
        assert!(matches!(err, ApiError::Status { status: 403, .. }), "{err}");
    }

    #[tokio::test]
    async fn default_branch_head_and_tree_are_read() {
        let (api, _) = api(FakeTransport::new()
            .json(
                "GET",
                &format!("https://api.github.com/repos/{FULL}"),
                200,
                &json!({"default_branch": "main"}),
            )
            .json(
                "GET",
                &format!("https://api.github.com/repos/{FULL}/branches/release/1.2"),
                200,
                &json!({"commit": {"sha": "1111111"}}),
            )
            .json(
                "GET",
                &format!("https://api.github.com/repos/{FULL}/git/trees/1111111?recursive=1"),
                200,
                &json!({"tree": [
                        {"path": "README.md", "type": "blob"},
                        {"path": "docs", "type": "tree"},
                        {"path": "docs/x.md", "type": "blob"}
                    ], "truncated": true}),
            ));
        assert_eq!(api.default_branch("t", FULL).await.unwrap(), "main");
        assert_eq!(
            api.branch_head("t", FULL, "release/1.2").await.unwrap(),
            "1111111"
        );
        let tree = api.tree_paths("t", FULL, "1111111").await.unwrap();
        assert_eq!(
            tree.paths,
            vec!["README.md".to_string(), "docs/x.md".to_string()]
        );
        assert!(tree.truncated);
        let err = api.default_branch("t", "Acme/Missing").await.unwrap_err();
        assert!(matches!(err, ApiError::Status { status: 404, .. }), "{err}");
    }
}
