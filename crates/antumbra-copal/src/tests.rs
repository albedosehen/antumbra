use super::*;
use std::sync::Mutex;

/// What one recorded transport call was: enough to assert the request
/// shaping (URL, credential, body) without a socket.
#[derive(Debug, Clone)]
enum Call {
    Post {
        url: String,
        credential: CopalCredential,
        body: Value,
    },
    Put {
        url: String,
        credential: CopalCredential,
        content_type: String,
        digest: String,
        body: Vec<u8>,
    },
}

impl Call {
    fn credential(&self) -> &CopalCredential {
        match self {
            Call::Post { credential, .. } | Call::Put { credential, .. } => credential,
        }
    }
}

fn tenant(name: &str) -> CopalCredential {
    CopalCredential::Tenant(name.into())
}

fn bearer(key: &str) -> CopalCredential {
    CopalCredential::Bearer(key.into())
}

/// Returns canned create/upload responses (or errors) and records every
/// call, so both the parsing and the create-then-upload ordering are
/// tested offline.
struct FakeTransport {
    create: std::result::Result<Value, String>,
    upload: std::result::Result<Value, String>,
    calls: Mutex<Vec<Call>>,
}

impl FakeTransport {
    fn happy() -> Self {
        Self {
            create: Ok(json!({ "id": "01k6f2x9q3w8e5r7t1y4z6a8b0", "state": "draft" })),
            upload: Ok(
                json!({ "digest": "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08", "state": "ready" }),
            ),
            calls: Mutex::new(Vec::new()),
        }
    }

    fn calls(&self) -> Vec<Call> {
        self.calls.lock().unwrap().clone()
    }
}

impl CopalTransport for FakeTransport {
    fn post_json(&self, url: &str, credential: &CopalCredential, body: &Value) -> Result<Value> {
        self.calls.lock().unwrap().push(Call::Post {
            url: url.into(),
            credential: credential.clone(),
            body: body.clone(),
        });
        self.create.clone().map_err(AntumbraError::other)
    }

    fn put_bytes(
        &self,
        url: &str,
        credential: &CopalCredential,
        content_type: &str,
        digest: &str,
        body: &[u8],
    ) -> Result<Value> {
        self.calls.lock().unwrap().push(Call::Put {
            url: url.into(),
            credential: credential.clone(),
            content_type: content_type.into(),
            digest: digest.into(),
            body: body.to_vec(),
        });
        self.upload.clone().map_err(AntumbraError::other)
    }
}

/// A shared-tenancy archive (every workspace under the "antumbra" copal
/// tenant), the shape copal's `keys` auth mode forces.
fn archive(addr: &str, transport: Arc<FakeTransport>) -> CopalArchive {
    CopalArchive::with_transport(addr, CopalTenancy::Shared("antumbra".into()), transport)
}

/// A per-workspace-tenancy archive: each workspace presents itself as the
/// copal tenant (copal's header auth mode).
fn per_workspace(transport: Arc<FakeTransport>) -> CopalArchive {
    CopalArchive::with_transport("127.0.0.1:9010", CopalTenancy::PerWorkspace, transport)
}

#[tokio::test]
async fn create_then_upload_returns_the_provenance() {
    let t = Arc::new(FakeTransport::happy());
    let got = archive("127.0.0.1:9010", t.clone())
        .archive_document(
            "ws:one",
            "Onboarding Guide",
            Some("guide.md"),
            "the original text",
        )
        .await
        .unwrap();
    assert_eq!(got.file_id, "01k6f2x9q3w8e5r7t1y4z6a8b0");
    assert_eq!(
        got.digest,
        "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
    );

    // Create first, upload second, both under the CONFIGURED copal tenant
    // header -- while the idempotency key derives from the WORKSPACE.
    let calls = t.calls();
    assert_eq!(calls.len(), 2);
    let Call::Post {
        url,
        credential,
        body,
    } = &calls[0]
    else {
        panic!("first call is the create, got {calls:?}");
    };
    assert_eq!(url, "http://127.0.0.1:9010/v1/files");
    assert_eq!(credential, &tenant("antumbra"));
    assert_eq!(body["content_type"], "text/plain");
    assert_eq!(
        body["idempotency_key"],
        json!(idempotency_key(provenance_hash(
            "ws:one",
            "Onboarding Guide"
        )))
    );
    assert_eq!(body["metadata"]["title"], "Onboarding Guide");
    assert_eq!(body["metadata"]["source"], "guide.md");
    assert_eq!(body["metadata"]["workspace"], "ws:one");
    let path = body["path"].as_str().unwrap();
    assert!(
        path.starts_with("antumbra/onboarding-guide-") && path.ends_with(".txt"),
        "a readable slugged path: {path}"
    );

    let Call::Put {
        url,
        credential,
        content_type,
        digest,
        body,
    } = &calls[1]
    else {
        panic!("second call is the upload, got {calls:?}");
    };
    // The upload declares the bytes' sha256, for copal to verify.
    assert_eq!(digest, &content_digest(b"the original text"));
    assert_eq!(
        url,
        "http://127.0.0.1:9010/v1/files/01k6f2x9q3w8e5r7t1y4z6a8b0/content"
    );
    assert_eq!(credential, &tenant("antumbra"));
    assert_eq!(content_type, "text/plain");
    assert_eq!(body, b"the original text");
}

#[tokio::test]
async fn source_is_omitted_from_metadata_when_unstated() {
    let t = Arc::new(FakeTransport::happy());
    archive("127.0.0.1:9010", t.clone())
        .archive_document("ws:one", "Untitled", None, "text")
        .await
        .unwrap();
    let Call::Post { body, .. } = &t.calls()[0] else {
        panic!("create first");
    };
    assert!(body["metadata"].get("source").is_none());
}

#[test]
fn tenant_names_render_into_copal_grammar_injectively() {
    // Already-clean names pass through byte-identical.
    assert_eq!(copal_tenant_name("antumbra"), "antumbra");
    assert_eq!(copal_tenant_name("ws-shon_2"), "ws-shon_2");
    // A colon (every real workspace) sanitizes and carries the original's
    // hash, so it can never collide with a literally-clean lookalike.
    let mapped = copal_tenant_name("ws:shon");
    assert!(mapped.starts_with("ws-shon-"), "{mapped}");
    assert_ne!(mapped, "ws-shon");
    assert_ne!(copal_tenant_name("ws:shon"), copal_tenant_name("ws.shon"));
    // Deterministic: the same workspace always lands in the same tenant.
    assert_eq!(copal_tenant_name("ws:shon"), copal_tenant_name("ws:shon"));
    // Everything emitted sits inside copal's grammar.
    assert!(mapped
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-'));
}

#[test]
fn the_idempotency_key_is_stable_per_workspace_and_title() {
    // Same (workspace, title) => same key, so a re-ingest revisions the
    // same copal file; either changing breaks the replay.
    let k = |ws, title| idempotency_key(provenance_hash(ws, title));
    assert_eq!(k("ws:one", "guide"), k("ws:one", "guide"));
    assert_ne!(k("ws:one", "guide"), k("ws:one", "other"));
    assert_ne!(k("ws:one", "guide"), k("ws:two", "guide"));
}

#[tokio::test]
async fn two_workspaces_with_the_same_title_get_distinct_keys_and_paths() {
    // The multi-workspace HTTP surface shares one copal tenant, so the
    // workspace must be what keeps two same-titled documents apart: same
    // header tenant on the wire, different idempotency keys and paths --
    // neither workspace can revision the other's document of record.
    let t = Arc::new(FakeTransport::happy());
    let a = archive("127.0.0.1:9010", t.clone());
    a.archive_document("ws:one", "Guide", None, "one's text")
        .await
        .unwrap();
    a.archive_document("ws:two", "Guide", None, "two's text")
        .await
        .unwrap();
    let calls = t.calls();
    let (Call::Post { body: b1, .. }, Call::Post { body: b2, .. }) = (&calls[0], &calls[2]) else {
        panic!("creates at 0 and 2, got {calls:?}");
    };
    assert_eq!(calls[0].credential(), &tenant("antumbra"));
    assert_eq!(calls[2].credential(), &tenant("antumbra"));
    assert_ne!(b1["idempotency_key"], b2["idempotency_key"]);
    assert_ne!(b1["path"], b2["path"]);
}

#[tokio::test]
async fn per_workspace_tenancy_presents_each_workspace_as_the_copal_tenant() {
    // Full tenancy: the workspace rides the x-copal-tenant header on BOTH
    // calls, so each workspace gets its own quotas, listings, and search
    // scope at copal's boundary -- and the key/path derivation is the same
    // one shared tenancy uses, so moving a deployment between the modes
    // never re-identifies a document.
    let t = Arc::new(FakeTransport::happy());
    let a = per_workspace(t.clone());
    a.archive_document("ws:one", "Guide", None, "one's text")
        .await
        .unwrap();
    a.archive_document("ws:two", "Guide", None, "two's text")
        .await
        .unwrap();
    let calls = t.calls();
    assert_eq!(calls.len(), 4);
    // The header carries the GRAMMAR-SAFE rendering of each workspace
    // (copal's TenantId is [A-Za-z0-9_-]; the colon in every real
    // workspace name would be a 400 verbatim), still one distinct copal
    // tenant per workspace.
    let one = copal_tenant_name("ws:one");
    let two = copal_tenant_name("ws:two");
    assert_ne!(one, two);
    let credentials: Vec<&CopalCredential> = calls.iter().map(Call::credential).collect();
    assert_eq!(
        credentials,
        [&tenant(&one), &tenant(&one), &tenant(&two), &tenant(&two)]
    );
    let (Call::Post { body: b1, .. }, Call::Post { body: b2, .. }) = (&calls[0], &calls[2]) else {
        panic!("creates at 0 and 2, got {calls:?}");
    };
    assert_ne!(b1["idempotency_key"], b2["idempotency_key"]);
    assert_ne!(b1["path"], b2["path"]);
}

#[tokio::test]
async fn a_shared_key_authenticates_every_workspace_with_the_one_bearer() {
    // Keys auth mode, shared tenancy: no tenant header at all -- the ck1
    // key IS the tenant, on both the create and the upload.
    let t = Arc::new(FakeTransport::happy());
    let a = CopalArchive::with_transport(
        "127.0.0.1:9010",
        CopalTenancy::SharedKey("ck1.kid.secret".into()),
        t.clone(),
    );
    a.archive_document("ws:one", "Guide", None, "text")
        .await
        .unwrap();
    let credentials: Vec<CopalCredential> =
        t.calls().iter().map(|c| c.credential().clone()).collect();
    assert_eq!(
        credentials,
        [bearer("ck1.kid.secret"), bearer("ck1.kid.secret")]
    );
}

#[tokio::test]
async fn per_workspace_keys_authenticate_each_workspace_with_its_own_bearer() {
    // The future-proof shape: copal in keys auth mode AND per-workspace
    // tenancy. Each workspace's calls carry its own ck1 credential (the
    // tenant is the key's), and the key/path derivation is unchanged, so
    // a header-mode deployment upgrading to keys re-identifies nothing.
    let t = Arc::new(FakeTransport::happy());
    let keys: HashMap<String, String> = [
        ("ws:one".to_string(), "ck1.one.secret".to_string()),
        ("ws:two".to_string(), "ck1.two.secret".to_string()),
    ]
    .into();
    let a = CopalArchive::with_transport(
        "127.0.0.1:9010",
        CopalTenancy::PerWorkspaceKeys(keys),
        t.clone(),
    );
    a.archive_document("ws:one", "Guide", None, "one's text")
        .await
        .unwrap();
    a.archive_document("ws:two", "Guide", None, "two's text")
        .await
        .unwrap();
    let calls = t.calls();
    let credentials: Vec<&CopalCredential> = calls.iter().map(Call::credential).collect();
    assert_eq!(
        credentials,
        [
            &bearer("ck1.one.secret"),
            &bearer("ck1.one.secret"),
            &bearer("ck1.two.secret"),
            &bearer("ck1.two.secret")
        ]
    );
    let (Call::Post { body: b1, .. }, Call::Post { body: b2, .. }) = (&calls[0], &calls[2]) else {
        panic!("creates at 0 and 2, got {calls:?}");
    };
    assert_ne!(b1["idempotency_key"], b2["idempotency_key"]);
    assert_ne!(b1["path"], b2["path"]);
}

#[tokio::test]
async fn a_workspace_without_a_configured_key_fails_closed() {
    // A keyless workspace refuses before any byte moves: archiving into a
    // tenant that is not the workspace's own would be worse than failing.
    let t = Arc::new(FakeTransport::happy());
    let keys: HashMap<String, String> =
        [("ws:one".to_string(), "ck1.one.secret".to_string())].into();
    let a = CopalArchive::with_transport(
        "127.0.0.1:9010",
        CopalTenancy::PerWorkspaceKeys(keys),
        t.clone(),
    );
    let err = a
        .archive_document("ws:two", "Guide", None, "text")
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("ws:two"),
        "the error names the keyless workspace: {err}"
    );
    assert!(
        !err.to_string().contains("secret"),
        "and never a credential: {err}"
    );
    assert!(t.calls().is_empty(), "refused before any transport call");
}

#[test]
fn a_debug_rendering_never_contains_the_bearer_secret() {
    // The credential type appears in recorded calls and error chains; its
    // Debug must redact the key (the HttpEmbedder no-leak rule).
    let rendered = format!("{:?}", bearer("ck1.kid.secret"));
    assert!(!rendered.contains("secret"), "{rendered}");
    assert!(rendered.contains("redacted"), "{rendered}");
    // The tenant variant stays legible -- it is an identity, not a secret.
    assert_eq!(format!("{:?}", tenant("ws:one")), "Tenant(\"ws:one\")");
}

#[tokio::test]
async fn per_workspace_tenancy_refuses_a_workspace_that_cannot_be_a_header() {
    // The workspace becomes a header value in this mode; a value the HTTP
    // layer would reject (or split) refuses before any byte moves.
    let t = Arc::new(FakeTransport::happy());
    let a = per_workspace(t.clone());
    for bad in ["", "ws one", "ws\r\nx-evil: 1"] {
        assert!(
            a.archive_document(bad, "guide", None, "text")
                .await
                .is_err(),
            "{bad:?} must refuse"
        );
    }
    assert!(t.calls().is_empty(), "refused before any transport call");
}

#[test]
fn a_bare_host_port_gets_http_and_a_full_url_is_used_as_is() {
    assert_eq!(normalize_base("127.0.0.1:9010"), "http://127.0.0.1:9010");
    assert_eq!(
        normalize_base("https://copal.example/"),
        "https://copal.example"
    );
    assert_eq!(
        normalize_base("http://copal.example:9010"),
        "http://copal.example:9010"
    );
}

#[test]
fn a_slugless_title_still_gets_a_path() {
    // A title with no ASCII alphanumerics falls back to the bare hash.
    let p = document_path("——", 0xdead);
    assert!(p.starts_with("antumbra/") && p.ends_with(".txt"), "{p}");
}

#[tokio::test]
async fn a_failed_create_surfaces_and_nothing_is_uploaded() {
    let t = Arc::new(FakeTransport {
        create: Err("connection refused".into()),
        ..FakeTransport::happy()
    });
    assert!(archive("127.0.0.1:9010", t.clone())
        .archive_document("ws:one", "guide", None, "text")
        .await
        .is_err());
    assert_eq!(t.calls().len(), 1, "no upload after a failed create");
}

#[tokio::test]
async fn a_failed_upload_surfaces() {
    let t = Arc::new(FakeTransport {
        upload: Err("http status: 500".into()),
        ..FakeTransport::happy()
    });
    assert!(archive("127.0.0.1:9010", t)
        .archive_document("ws:one", "guide", None, "text")
        .await
        .is_err());
}

#[tokio::test]
async fn a_create_response_without_an_id_is_an_error() {
    let t = Arc::new(FakeTransport {
        create: Ok(json!({ "unexpected": true })),
        ..FakeTransport::happy()
    });
    assert!(archive("127.0.0.1:9010", t.clone())
        .archive_document("ws:one", "guide", None, "text")
        .await
        .is_err());
    assert_eq!(t.calls().len(), 1, "no upload without a file id");
}

#[tokio::test]
async fn an_upload_response_without_a_digest_is_an_error() {
    let t = Arc::new(FakeTransport {
        upload: Ok(json!({ "state": "ready" })),
        ..FakeTransport::happy()
    });
    assert!(archive("127.0.0.1:9010", t)
        .archive_document("ws:one", "guide", None, "text")
        .await
        .is_err());
}

#[test]
fn a_content_digest_is_copal_s_sha256_hex() {
    assert_eq!(
        content_digest(b"test"),
        "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
    );
}

#[tokio::test]
async fn a_replay_already_holding_these_bytes_is_not_uploaded_again() {
    let held = content_digest(b"the original text");
    let t = Arc::new(FakeTransport {
        create: Ok(json!({ "id": "01k6f2x9q3w8e5r7t1y4z6a8b0", "state": "ready", "digest": held })),
        ..FakeTransport::happy()
    });
    let got = archive("127.0.0.1:9010", t.clone())
        .archive_document("ws:one", "Guide", None, "the original text")
        .await
        .unwrap();
    assert_eq!(got.digest, held);
    assert_eq!(
        t.calls().len(),
        1,
        "the create only: copal holds these bytes"
    );

    // Changed content is uploaded as a new version.
    let t = Arc::new(FakeTransport {
        create: Ok(json!({ "id": "01k6f2x9q3w8e5r7t1y4z6a8b0", "state": "ready", "digest": held })),
        ..FakeTransport::happy()
    });
    archive("127.0.0.1:9010", t.clone())
        .archive_document("ws:one", "Guide", None, "the revised text")
        .await
        .unwrap();
    assert_eq!(t.calls().len(), 2);
}
