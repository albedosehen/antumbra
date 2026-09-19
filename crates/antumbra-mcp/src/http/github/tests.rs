//! The webhook receiver over the real router: the signature boundary, the
//! merge and delete handlers against the store, the acknowledgements, and the
//! App-driven document ingest with a canned GitHub.

use super::super::{router, Bounded, Serving, MAX_EMBEDDERS, MAX_SESSIONS};
use super::ingest::{self, IngestPlan};
use super::*;
use antumbra_core::ports::Embedder;
use antumbra_core::testing::FixedEmbedder;
use antumbra_core::{orphan_of, GitProvenance, Memory, MemoryNetwork, TenantId};
use antumbra_github::sign;
use antumbra_github::testing::{FakeTransport, APP_PRIVATE_KEY_PEM};
use antumbra_store::repo::document;
use antumbra_store::{Store, EMBED_DIM};
use axum::http::header;
use serde_json::json;
use tower::ServiceExt;

const SECRET: &[u8] = b"webhook-test-secret";
const TENANT: &str = "ws:acme";
const REPO: &str = "github.com/acme/orders";
const FULL: &str = "Acme/Orders";
const MERGE_SHA: &str = "fedcba9876543210fedcba9876543210fedcba98";
const HEAD_SHA: &str = "1111111111111111111111111111111111111111";
const API: &str = "https://api.github.com";

async fn state(github: Option<GithubConfig>) -> Arc<HttpState> {
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
        github: github.map(Arc::new),
        sessions: Mutex::new(Bounded::new(MAX_SESSIONS)),
        consolidating: Arc::new(Mutex::new(std::collections::HashSet::new())),
        registry: crate::notify::PeerRegistry::new(),
    })
}

fn plain() -> GithubConfig {
    GithubConfig::new(SECRET, RepoMap::all(TENANT))
}

fn with_app(fake: FakeTransport) -> GithubConfig {
    plain()
        .with_app(AppCredentials::new("123", APP_PRIVATE_KEY_PEM).unwrap())
        .with_api(GithubApi::with_transport(API, Arc::new(fake)))
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

async fn deliver(st: &Arc<HttpState>, kind: &str, payload: &serde_json::Value) -> Outcome {
    outcome(
        router(st.clone())
            .oneshot(signed(kind, payload))
            .await
            .unwrap(),
    )
    .await
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
            "base": { "ref": "main", "sha": HEAD_SHA },
            "head": { "ref": "feat/outbox", "sha": "2222222222222222222222222222222222222222" },
            "user": { "login": "shon" },
            "commits": 3, "additions": 120, "deletions": 40, "changed_files": 5
        },
        "repository": { "full_name": FULL, "html_url": "https://github.com/Acme/Orders" },
        "installation": { "id": 77 }
    })
}

fn branch_delete(branch: &str) -> serde_json::Value {
    json!({
        "ref": branch,
        "ref_type": "branch",
        "repository": { "full_name": FULL, "html_url": "https://github.com/Acme/Orders" }
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

/// Wait for the detached ingest task to land `titles`, or fail.
async fn wait_for_titles(state: &HttpState, titles: &[&str]) -> Vec<String> {
    let tenant = TenantId::new(TENANT);
    for _ in 0..100 {
        let have = document::list_titles(&state.store, &tenant).await.unwrap();
        if titles.iter().all(|t| have.iter().any(|h| h == t)) {
            return have;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    panic!(
        "titles {titles:?} never appeared; have {:?}",
        document::list_titles(&state.store, &tenant).await.unwrap()
    );
}

#[tokio::test]
async fn an_unsigned_or_missigned_delivery_is_unauthorized() {
    let st = state(Some(plain())).await;
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
    let st = state(Some(plain())).await;
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

    let out = deliver(&st, "pull_request", &merged_pr()).await;
    assert_eq!(out.event, "pull_request");
    assert_eq!(out.repo.as_deref(), Some(REPO));
    assert_eq!(out.tenant.as_deref(), Some(TENANT));
    assert_eq!(out.reanchored, 1);
    assert_eq!(
        out.stored,
        vec!["memory:github-pr-github-com-acme-orders-42"]
    );
    assert_eq!(
        out.ingest_queued, 0,
        "no App credentials, so nothing is read"
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
    let again = deliver(&st, "pull_request", &merged_pr()).await;
    assert_eq!(again.reanchored, 0);
    assert_eq!(again.stored.len(), 1);
    assert_eq!(evidence_of(&st, "memory:feat").await.len(), 2);
}

#[tokio::test]
async fn a_branch_delete_orphans_its_memories_but_not_reanchored_ones() {
    let st = state(Some(plain())).await;
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
    deliver(&st, "pull_request", &merged_pr()).await;
    let out = deliver(&st, "delete", &branch_delete("feat/outbox")).await;
    assert_eq!(out.event, "delete");
    assert_eq!(out.orphaned, 0, "the merge already moved it to main");
    assert_eq!(orphan_of(&evidence_of(&st, "memory:feat").await), None);

    let out = deliver(&st, "delete", &branch_delete("feat/other")).await;
    assert_eq!(out.orphaned, 1);
    let other = evidence_of(&st, "memory:other").await;
    assert_eq!(
        orphan_of(&other).map(|o| o.branch),
        Some("feat/other".to_string())
    );
    // Redelivery is a no-op.
    let out = deliver(&st, "delete", &branch_delete("feat/other")).await;
    assert_eq!(out.orphaned, 0);
}

#[tokio::test]
async fn pings_unhandled_events_and_unmapped_repositories_are_acknowledged() {
    let repos = RepoMap::from_json(r#"{"github.com/acme/billing": "ws:acme"}"#).unwrap();
    let st = state(Some(GithubConfig::new(SECRET, repos))).await;
    let out = deliver(&st, "ping", &json!({"zen": "Keep it logically awesome."})).await;
    assert_eq!(out.ignored.as_deref(), Some("pong"));
    let out = deliver(&st, "issues", &json!({"action": "opened"})).await;
    assert_eq!(out.event, "issues");
    assert!(out.ignored.is_some());
    // The orders repository is not in the map: acknowledged, untouched.
    let out = deliver(&st, "pull_request", &merged_pr()).await;
    assert_eq!(out.reanchored, 0);
    assert!(out.stored.is_empty());
    assert_eq!(
        out.ignored.as_deref(),
        Some("repository is not mapped to a workspace")
    );
    // A close without a merge, and a tag deletion, do nothing.
    let mut closed = merged_pr();
    closed["pull_request"]["merged"] = json!(false);
    let out = deliver(&st, "pull_request", &closed).await;
    assert_eq!(
        out.ignored.as_deref(),
        Some("action 'closed' is not a merge")
    );
    let mut tag = branch_delete("v1");
    tag["ref_type"] = json!("tag");
    let out = deliver(&st, "delete", &tag).await;
    assert_eq!(
        out.ignored.as_deref(),
        Some("a deleted tag orphans nothing")
    );
    // An installation without App credentials cannot read anything.
    let out = deliver(
        &st,
        "installation",
        &json!({"action": "created", "installation": {"id": 77},
                "repositories": [{"full_name": "Acme/Billing"}]}),
    )
    .await;
    assert_eq!(out.ingest_queued, 0);
    assert!(out
        .ignored
        .as_deref()
        .unwrap()
        .contains("no App credentials"));
    // Garbage that verifies is still the sender's fault.
    let body = b"not json".to_vec();
    let signature = sign(SECRET, &body);
    let st = state(Some(plain())).await;
    let resp = router(st)
        .oneshot(delivery("pull_request", &body, Some(signature)))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

fn token_route(fake: FakeTransport) -> FakeTransport {
    fake.json(
        "POST",
        &format!("{API}/app/installations/77/access_tokens"),
        201,
        &json!({"token": "ghs_test", "expires_at": "2099-01-01T00:00:00Z"}),
    )
}

/// With App credentials, a merge reads the changed knowledge documents at
/// the merge commit and ingests them, anchored, after the response.
#[tokio::test]
async fn a_merge_ingests_the_changed_documents_at_the_merge_commit() {
    let fake = token_route(FakeTransport::new())
        .json(
            "GET",
            &format!("{API}/repos/{FULL}/pulls/42/files?per_page=100&page=1"),
            200,
            &json!([
                {"filename": "docs/orders.md", "status": "modified"},
                {"filename": "src/main.rs", "status": "modified"},
                {"filename": "README.md", "status": "removed"},
                {"filename": "docs/gone.md", "status": "added"}
            ]),
        )
        .route(
            "GET",
            &format!("{API}/repos/{FULL}/contents/docs/orders.md?ref={MERGE_SHA}"),
            200,
            "# Orders\n\nOrders are written through an outbox.\n",
        );
    let st = state(Some(with_app(fake))).await;

    let out = deliver(&st, "pull_request", &merged_pr()).await;
    assert_eq!(out.ingest_queued, 2, "two documents survived the filter");
    let titles = wait_for_titles(&st, &["docs/orders.md"]).await;
    assert_eq!(
        titles,
        vec!["docs/orders.md".to_string()],
        "the absent one was skipped"
    );

    let embedder = FixedEmbedder::new(EMBED_DIM);
    let hits = document::recall(
        &st.store,
        &TenantId::new(TENANT),
        &embedder.embed("outbox").await.unwrap(),
        5,
    )
    .await
    .unwrap();
    assert_eq!(hits.len(), 1);
    let source = hits[0].source.as_deref().unwrap();
    assert_eq!(
        source,
        format!(
            "https://github.com/{FULL}/blob/{MERGE_SHA}/docs/orders.md @ git:{REPO}@{MERGE_SHA}#main:docs/orders.md"
        )
    );
    assert!(hits[0].content.contains("outbox"));
}

/// Installing the App on a repository cold-starts it: every knowledge
/// document at the head of the default branch is ingested.
#[tokio::test]
async fn an_installation_cold_starts_the_repository() {
    let fake = token_route(FakeTransport::new())
        .json(
            "GET",
            &format!("{API}/repos/{FULL}"),
            200,
            &json!({"default_branch": "main"}),
        )
        .json(
            "GET",
            &format!("{API}/repos/{FULL}/branches/main"),
            200,
            &json!({"commit": {"sha": HEAD_SHA}}),
        )
        .json(
            "GET",
            &format!("{API}/repos/{FULL}/git/trees/{HEAD_SHA}?recursive=1"),
            200,
            &json!({"tree": [
                {"path": "README.md", "type": "blob"},
                {"path": "src/lib.rs", "type": "blob"},
                {"path": "docs", "type": "tree"},
                {"path": "docs/adr/0001-outbox.md", "type": "blob"},
                {"path": "LICENSE", "type": "blob"}
            ], "truncated": false}),
        )
        .route(
            "GET",
            &format!("{API}/repos/{FULL}/contents/README.md?ref={HEAD_SHA}"),
            200,
            "# Orders service\n",
        )
        .route(
            "GET",
            &format!("{API}/repos/{FULL}/contents/docs/adr/0001-outbox.md?ref={HEAD_SHA}"),
            200,
            "# ADR 1: the outbox\n",
        );
    // An explicit map: the second repository the installation names is not
    // in it, and an unmapped repository is reported, never read.
    let repos = RepoMap::from_json(r#"{"github.com/acme/orders": "ws:acme"}"#).unwrap();
    let cfg = GithubConfig::new(SECRET, repos)
        .with_app(AppCredentials::new("123", APP_PRIVATE_KEY_PEM).unwrap())
        .with_api(GithubApi::with_transport(API, Arc::new(fake)));
    let st = state(Some(cfg)).await;
    let out = deliver(
        &st,
        "installation",
        &json!({"action": "created", "installation": {"id": 77},
                "repositories": [{"full_name": FULL}, {"full_name": "Other/Unmapped"}]}),
    )
    .await;
    assert_eq!(out.event, "installation");
    assert_eq!(out.ingest_queued, 2);
    assert_eq!(
        out.ignored.as_deref(),
        Some("not mapped to a workspace: github.com/other/unmapped")
    );
    let titles = wait_for_titles(&st, &["README.md", "docs/adr/0001-outbox.md"]).await;
    assert_eq!(titles.len(), 2);
    let embedder = FixedEmbedder::new(EMBED_DIM);
    let hits = document::recall(
        &st.store,
        &TenantId::new(TENANT),
        &embedder.embed("ADR outbox").await.unwrap(),
        5,
    )
    .await
    .unwrap();
    assert!(hits
        .iter()
        .any(|h| h.source.as_deref().unwrap().ends_with(&format!(
            "git:{REPO}@{HEAD_SHA}#main:docs/adr/0001-outbox.md"
        ))));
}

/// The planner and the runner, directly: a size cap and a read error are
/// per-document outcomes, and the token is minted once per installation.
#[tokio::test]
async fn the_runner_reports_per_document_and_reuses_the_token() {
    let big = "x".repeat(antumbra_github::MAX_DOCUMENT_BYTES + 1);
    let fake = token_route(FakeTransport::new())
        .route(
            "GET",
            &format!("{API}/repos/{FULL}/contents/docs/ok.md?ref={HEAD_SHA}"),
            200,
            "fine\n",
        )
        .route(
            "GET",
            &format!("{API}/repos/{FULL}/contents/docs/big.md?ref={HEAD_SHA}"),
            200,
            big,
        )
        .route(
            "GET",
            &format!("{API}/repos/{FULL}/contents/docs/forbidden.md?ref={HEAD_SHA}"),
            403,
            "{\"message\":\"forbidden\"}",
        );
    let fake = Arc::new(fake);
    let cfg = plain()
        .with_app(AppCredentials::new("123", APP_PRIVATE_KEY_PEM).unwrap())
        .with_api(GithubApi::with_transport(API, fake.clone()));
    let st = state(Some(cfg)).await;
    let cfg = st.github.as_ref().unwrap();
    let token = cfg.installation_token(77).await.unwrap().unwrap();
    assert_eq!(token, "ghs_test");
    assert_eq!(
        cfg.installation_token(77).await.unwrap().unwrap(),
        "ghs_test"
    );
    assert_eq!(
        fake.calls()
            .iter()
            .filter(|c| c.starts_with("POST"))
            .count(),
        1,
        "the second token came from the cache"
    );
    let plan = IngestPlan {
        tenant: TenantId::new(TENANT),
        full_name: FULL.into(),
        slug: REPO.into(),
        commit: HEAD_SHA.into(),
        branch: "main".into(),
        paths: vec![
            "docs/ok.md".into(),
            "docs/big.md".into(),
            "docs/forbidden.md".into(),
            "docs/missing.md".into(),
        ],
        truncated: false,
        token,
    };
    let report = ingest::run(&st, plan).await;
    assert_eq!(report.ingested, vec!["docs/ok.md".to_string()]);
    assert_eq!(
        report.skipped,
        vec!["docs/big.md".to_string(), "docs/missing.md".to_string()]
    );
    assert_eq!(report.failed.len(), 1);
    assert_eq!(report.failed[0].0, "docs/forbidden.md");
    assert!(report.failed[0].1.contains("403"), "{}", report.failed[0].1);
}

#[test]
fn from_flags_requires_matching_halves() {
    let none = |secret: Option<&str>, tenant: Option<&str>, app: Option<&str>| {
        GithubConfig::from_flags(
            secret.map(str::to_string),
            tenant.map(str::to_string),
            None,
            app.map(str::to_string),
            None,
            DEFAULT_API_URL,
        )
    };
    assert!(none(None, None, None).unwrap().is_none());
    let err = none(Some("s"), None, None).unwrap_err();
    assert!(err.to_string().contains("--github-tenant"), "{err}");
    let err = none(None, Some("ws:x"), None).unwrap_err();
    assert!(err.to_string().contains("--github-webhook-secret"), "{err}");
    let err = none(Some("s"), Some("ws:x"), Some("123")).unwrap_err();
    assert!(err.to_string().contains("go together"), "{err}");
    let cfg = none(Some("s"), Some(" ws:x "), None).unwrap().unwrap();
    assert!(!cfg.reads_contents());
    assert_eq!(
        cfg.repos.tenant_for("github.com/a/b").map(|t| t.as_str()),
        Some("ws:x")
    );

    let dir = std::env::temp_dir();
    let map = dir.join(format!("antumbra-github-repos-{}.json", std::process::id()));
    std::fs::write(&map, r#"{"github.com/acme/orders": "ws:acme"}"#).unwrap();
    let key = dir.join(format!("antumbra-github-key-{}.pem", std::process::id()));
    std::fs::write(&key, APP_PRIVATE_KEY_PEM).unwrap();
    let cfg = GithubConfig::from_flags(
        Some("s".into()),
        None,
        Some(&map),
        Some("123".into()),
        Some(&key),
        "https://ghe.acme.internal/api/v3/",
    )
    .unwrap()
    .unwrap();
    assert!(cfg.reads_contents());
    assert_eq!(cfg.repos.mapped(), Some(1));
    assert_eq!(cfg.api().web_host(), "ghe.acme.internal");
    assert!(!format!("{cfg:?}").contains("PRIVATE"), "{cfg:?}");
    let err = GithubConfig::from_flags(
        Some("s".into()),
        Some("ws:x".into()),
        Some(&map),
        None,
        None,
        DEFAULT_API_URL,
    )
    .unwrap_err();
    assert!(err.to_string().contains("not both"), "{err}");
    std::fs::remove_file(map).ok();
    std::fs::remove_file(key).ok();
    let missing = dir.join("antumbra-github-repos-missing.json");
    let err = GithubConfig::from_flags(
        Some("s".into()),
        None,
        Some(&missing),
        None,
        None,
        DEFAULT_API_URL,
    )
    .unwrap_err();
    assert!(err.to_string().contains("cannot read"), "{err}");
}
