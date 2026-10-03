//! The declared source through the App: an installation reads every
//! repository's manifests and records the edges among them; a merge into the
//! default branch that changes a manifest reads that repository again,
//! retracting what it no longer declares and bringing back what it declares
//! again; a merge anywhere else reads nothing.

use super::*;
use antumbra_core::depgraph::{Edge, Source};

const WEB: &str = "Acme/Web";
const ORDERS_HEAD: &str = "3333333333333333333333333333333333333333";
const WEB_HEAD: &str = "4444444444444444444444444444444444444444";
const DROPPED: &str = "5555555555555555555555555555555555555555";
const RESTORED: &str = "6666666666666666666666666666666666666666";
const ELSEWHERE: &str = "7777777777777777777777777777777777777777";

const ORDERS_CARGO: &str = "[package]\nname = \"acme-orders\"\n";
const WEB_CARGO: &str = "[package]\nname = \"acme-web\"\n\n[dependencies]\nacme-orders = \"1\"\n";
const WEB_CARGO_WITHOUT: &str = "[package]\nname = \"acme-web\"\n";

/// One repository's tree at `sha`, and each file's contents there.
fn tree(fake: FakeTransport, full: &str, sha: &str, files: &[(&str, &str)]) -> FakeTransport {
    let listing: Vec<serde_json::Value> = files
        .iter()
        .map(|(path, _)| json!({"path": path, "type": "blob"}))
        .collect();
    let mut fake = fake.json(
        "GET",
        &format!("{API}/repos/{full}/git/trees/{sha}?recursive=1"),
        200,
        &json!({"tree": listing, "truncated": false}),
    );
    for (path, text) in files {
        fake = fake.route(
            "GET",
            &format!("{API}/repos/{full}/contents/{path}?ref={sha}"),
            200,
            *text,
        );
    }
    fake
}

/// A repository's cold start: its default branch, that branch's head, and
/// the tree there.
fn cold(fake: FakeTransport, full: &str, sha: &str, files: &[(&str, &str)]) -> FakeTransport {
    let fake = fake
        .json(
            "GET",
            &format!("{API}/repos/{full}"),
            200,
            &json!({"default_branch": "main"}),
        )
        .json(
            "GET",
            &format!("{API}/repos/{full}/branches/main"),
            200,
            &json!({"commit": {"sha": sha}}),
        );
    tree(fake, full, sha, files)
}

/// A merge into the web repository's `base`, changing its Cargo.toml.
fn merge(fake: FakeTransport, number: u64) -> FakeTransport {
    fake.json(
        "GET",
        &format!("{API}/repos/{WEB}/pulls/{number}/files?per_page=100&page=1"),
        200,
        &json!([{"filename": "Cargo.toml", "status": "modified"}]),
    )
}

fn merged(number: u64, sha: &str, base: &str) -> serde_json::Value {
    json!({
        "action": "closed",
        "pull_request": {
            "number": number,
            "title": "Dependencies",
            "html_url": format!("https://github.com/{WEB}/pull/{number}"),
            "merged": true,
            "merge_commit_sha": sha,
            "base": { "ref": base, "sha": WEB_HEAD },
            "head": { "ref": "deps", "sha": "8888888888888888888888888888888888888888" },
            "user": { "login": "shon" },
            "commits": 1, "additions": 1, "deletions": 1, "changed_files": 1
        },
        "repository": {
            "full_name": WEB,
            "html_url": format!("https://github.com/{WEB}"),
            "default_branch": "main"
        },
        "installation": { "id": 77 }
    })
}

/// The declared edges in the workspace: from, to, and the file's anchor.
async fn declared(st: &HttpState) -> Vec<(String, String, Option<String>)> {
    let mut edges: Vec<_> = memory::with_any_evidence(
        &st.store,
        &TenantId::new(TENANT),
        &[Source::Declared.evidence_entry()],
    )
    .await
    .unwrap()
    .iter()
    .filter_map(Edge::of)
    .map(|e| (e.from, e.to, e.anchor.map(|a| a.to_evidence())))
    .collect();
    edges.sort();
    edges
}

/// Wait for the detached reading to leave `want` declared edges, or fail.
async fn wait_for_declared(st: &HttpState, want: usize) -> Vec<(String, String, Option<String>)> {
    for _ in 0..100 {
        let have = declared(st).await;
        if have.len() == want {
            return have;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    panic!(
        "never {want} declared edge(s); have {:?}",
        declared(st).await
    );
}

#[tokio::test]
async fn the_app_keeps_the_declared_edges_among_its_repositories() {
    let fake = token_route(FakeTransport::new());
    let fake = cold(
        fake,
        FULL,
        ORDERS_HEAD,
        &[("Cargo.toml", ORDERS_CARGO), ("README.md", "# Orders\n")],
    );
    let fake = cold(
        fake,
        WEB,
        WEB_HEAD,
        &[
            ("Cargo.toml", WEB_CARGO),
            // A fixture is not the repository's own manifest: never fetched.
            (
                "tests/fixtures/Cargo.toml",
                "[package]\nname = \"acme-orders\"\n",
            ),
        ],
    );
    let fake = tree(
        merge(fake, 43),
        WEB,
        DROPPED,
        &[("Cargo.toml", WEB_CARGO_WITHOUT)],
    );
    let fake = tree(merge(fake, 44), WEB, RESTORED, &[("Cargo.toml", WEB_CARGO)]);
    let fake = Arc::new(merge(fake, 45));
    let cfg = plain()
        .with_app(AppCredentials::new("123", APP_PRIVATE_KEY_PEM).unwrap())
        .with_api(GithubApi::with_transport(API, fake.clone()));
    let st = state(Some(cfg)).await;

    deliver(
        &st,
        "installation",
        &json!({"action": "created", "installation": {"id": 77},
                "repositories": [{"full_name": FULL}, {"full_name": WEB}]}),
    )
    .await;
    let installed = wait_for_declared(&st, 1).await;
    assert_eq!(
        installed,
        [(
            "github.com/acme/web".to_string(),
            REPO.to_string(),
            Some(format!(
                "git:github.com/acme/web@{WEB_HEAD}#main:Cargo.toml"
            ))
        )],
        "the edge is anchored to the declaring file at the commit it was read"
    );
    assert!(
        !fake.calls().iter().any(|c| c.contains("fixtures")),
        "{:?}",
        fake.calls()
    );

    // The dependency is dropped on the default branch: retracted.
    deliver(&st, "pull_request", &merged(43, DROPPED, "main")).await;
    assert!(wait_for_declared(&st, 0).await.is_empty());

    // Declared again: back, anchored to the new commit.
    deliver(&st, "pull_request", &merged(44, RESTORED, "main")).await;
    let restored = wait_for_declared(&st, 1).await;
    assert_eq!(
        restored[0].2.as_deref(),
        Some(format!("git:github.com/acme/web@{RESTORED}#main:Cargo.toml").as_str())
    );

    // A merge into another branch is not the repository's reading.
    deliver(&st, "pull_request", &merged(45, ELSEWHERE, "release")).await;
    assert!(
        !fake.calls().iter().any(|c| c.contains(ELSEWHERE)),
        "{:?}",
        fake.calls()
    );
}
