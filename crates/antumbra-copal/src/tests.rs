use super::*;
use anyhow::{bail, Context};
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

    /// Every call it took, in order. The fake only ever pushes to the list,
    /// so a poisoned lock holds a whole list and is read as is.
    fn calls(&self) -> Vec<Call> {
        self.calls
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

impl CopalTransport for FakeTransport {
    fn post_json(&self, url: &str, credential: &CopalCredential, body: &Value) -> Result<Value> {
        // Read as `calls` is: a push cannot leave the list half-written.
        self.calls
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(Call::Post {
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
        // Read as `calls` is: a push cannot leave the list half-written.
        self.calls
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(Call::Put {
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
async fn create_then_upload_returns_the_provenance() -> anyhow::Result<()> {
    let t = Arc::new(FakeTransport::happy());
    let got = archive("127.0.0.1:9010", t.clone())
        .archive_document(
            "ws:one",
            "Onboarding Guide",
            Some("guide.md"),
            "the original text",
        )
        .await?;
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
        bail!("first call is the create, got {calls:?}");
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
    let path = body["path"]
        .as_str()
        .context("the create body has no `path`")?;
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
        bail!("second call is the upload, got {calls:?}");
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
    Ok(())
}

#[tokio::test]
async fn source_is_omitted_from_metadata_when_unstated() -> anyhow::Result<()> {
    let t = Arc::new(FakeTransport::happy());
    archive("127.0.0.1:9010", t.clone())
        .archive_document("ws:one", "Untitled", None, "text")
        .await?;
    let Call::Post { body, .. } = &t.calls()[0] else {
        bail!("create first");
    };
    assert!(body["metadata"].get("source").is_none());
    Ok(())
}

#[test]
fn tenant_names_render_into_copal_grammar_injectively() -> anyhow::Result<()> {
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
    Ok(())
}

#[test]
fn the_idempotency_key_is_stable_per_workspace_and_title() -> anyhow::Result<()> {
    // Same (workspace, title) => same key, so a re-ingest revisions the
    // same copal file; either changing breaks the replay.
    let k = |ws, title| idempotency_key(provenance_hash(ws, title));
    assert_eq!(k("ws:one", "guide"), k("ws:one", "guide"));
    assert_ne!(k("ws:one", "guide"), k("ws:one", "other"));
    assert_ne!(k("ws:one", "guide"), k("ws:two", "guide"));
    Ok(())
}

#[tokio::test]
async fn two_workspaces_with_the_same_title_get_distinct_keys_and_paths() -> anyhow::Result<()> {
    // The multi-workspace HTTP surface shares one copal tenant, so the
    // workspace must be what keeps two same-titled documents apart: same
    // header tenant on the wire, different idempotency keys and paths --
    // neither workspace can revision the other's document of record.
    let t = Arc::new(FakeTransport::happy());
    let a = archive("127.0.0.1:9010", t.clone());
    a.archive_document("ws:one", "Guide", None, "one's text")
        .await?;
    a.archive_document("ws:two", "Guide", None, "two's text")
        .await?;
    let calls = t.calls();
    let (Call::Post { body: b1, .. }, Call::Post { body: b2, .. }) = (&calls[0], &calls[2]) else {
        bail!("creates at 0 and 2, got {calls:?}");
    };
    assert_eq!(calls[0].credential(), &tenant("antumbra"));
    assert_eq!(calls[2].credential(), &tenant("antumbra"));
    assert_ne!(b1["idempotency_key"], b2["idempotency_key"]);
    assert_ne!(b1["path"], b2["path"]);
    Ok(())
}

#[tokio::test]
async fn per_workspace_tenancy_presents_each_workspace_as_the_copal_tenant() -> anyhow::Result<()> {
    // Full tenancy: the workspace rides the x-copal-tenant header on BOTH
    // calls, so each workspace gets its own quotas, listings, and search
    // scope at copal's boundary -- and the key/path derivation is the same
    // one shared tenancy uses, so moving a deployment between the modes
    // never re-identifies a document.
    let t = Arc::new(FakeTransport::happy());
    let a = per_workspace(t.clone());
    a.archive_document("ws:one", "Guide", None, "one's text")
        .await?;
    a.archive_document("ws:two", "Guide", None, "two's text")
        .await?;
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
        bail!("creates at 0 and 2, got {calls:?}");
    };
    assert_ne!(b1["idempotency_key"], b2["idempotency_key"]);
    assert_ne!(b1["path"], b2["path"]);
    Ok(())
}

#[tokio::test]
async fn a_shared_key_authenticates_every_workspace_with_the_one_bearer() -> anyhow::Result<()> {
    // Keys auth mode, shared tenancy: no tenant header at all -- the ck1
    // key IS the tenant, on both the create and the upload.
    let t = Arc::new(FakeTransport::happy());
    let a = CopalArchive::with_transport(
        "127.0.0.1:9010",
        CopalTenancy::SharedKey("ck1.kid.secret".into()),
        t.clone(),
    );
    a.archive_document("ws:one", "Guide", None, "text").await?;
    let credentials: Vec<CopalCredential> =
        t.calls().iter().map(|c| c.credential().clone()).collect();
    assert_eq!(
        credentials,
        [bearer("ck1.kid.secret"), bearer("ck1.kid.secret")]
    );
    Ok(())
}

#[tokio::test]
async fn per_workspace_keys_authenticate_each_workspace_with_its_own_bearer() -> anyhow::Result<()>
{
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
        .await?;
    a.archive_document("ws:two", "Guide", None, "two's text")
        .await?;
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
        bail!("creates at 0 and 2, got {calls:?}");
    };
    assert_ne!(b1["idempotency_key"], b2["idempotency_key"]);
    assert_ne!(b1["path"], b2["path"]);
    Ok(())
}

#[tokio::test]
async fn a_workspace_without_a_configured_key_fails_closed() -> anyhow::Result<()> {
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
    let Err(err) = a.archive_document("ws:two", "Guide", None, "text").await else {
        bail!("a keyless workspace was archived");
    };
    assert!(
        err.to_string().contains("ws:two"),
        "the error names the keyless workspace: {err}"
    );
    assert!(
        !err.to_string().contains("secret"),
        "and never a credential: {err}"
    );
    assert!(t.calls().is_empty(), "refused before any transport call");
    Ok(())
}

#[test]
fn a_debug_rendering_never_contains_the_bearer_secret() -> anyhow::Result<()> {
    // The credential type appears in recorded calls and error chains; its
    // Debug must redact the key (the HttpEmbedder no-leak rule).
    let rendered = format!("{:?}", bearer("ck1.kid.secret"));
    assert!(!rendered.contains("secret"), "{rendered}");
    assert!(rendered.contains("redacted"), "{rendered}");
    // The tenant variant stays legible -- it is an identity, not a secret.
    assert_eq!(format!("{:?}", tenant("ws:one")), "Tenant(\"ws:one\")");
    Ok(())
}

#[tokio::test]
async fn per_workspace_tenancy_refuses_a_workspace_that_cannot_be_a_header() -> anyhow::Result<()> {
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
    Ok(())
}

#[test]
fn a_bare_host_port_gets_http_and_a_full_url_is_used_as_is() -> anyhow::Result<()> {
    assert_eq!(normalize_base("127.0.0.1:9010"), "http://127.0.0.1:9010");
    assert_eq!(normalize_base("localhost:9010"), "http://localhost:9010");
    assert_eq!(normalize_base("[::1]:9010"), "http://[::1]:9010");
    assert_eq!(
        normalize_base("https://copal.example/"),
        "https://copal.example"
    );
    assert_eq!(
        normalize_base("http://copal.example:9010"),
        "http://copal.example:9010"
    );
    Ok(())
}

/// A bare address that is not this machine is reached over TLS unless told
/// otherwise, and an explicit `http://` to one is refused without the flag.
#[test]
fn a_bare_remote_address_defaults_to_https() -> std::result::Result<(), String> {
    assert_eq!(
        normalize_base("archive.example:9000"),
        "https://archive.example:9000"
    );
    let over_tls =
        CopalArchive::from_flags(Some("archive.example:9000"), None, None, None, None, false)
            .map_err(|e| e.to_string())?;
    if over_tls.map(|a| a.base.clone()) != Some("https://archive.example:9000".into()) {
        return Err("a bare remote address did not become https".into());
    }
    match CopalArchive::from_flags(
        Some("http://archive.example:9000"),
        None,
        None,
        None,
        None,
        false,
    ) {
        Err(CopalConfigError::Cleartext(addr)) if addr == "http://archive.example:9000" => {}
        Err(other) => return Err(format!("the wrong refusal: {other}")),
        Ok(_) => return Err("cleartext across a network was not refused".into()),
    }
    let allowed = CopalArchive::from_flags(
        Some("http://archive.example:9000"),
        None,
        None,
        None,
        None,
        true,
    )
    .map_err(|e| e.to_string())?;
    if allowed.is_none() {
        return Err("--copal-allow-http did not allow it".into());
    }
    let local =
        CopalArchive::from_flags(Some("http://127.0.0.1:9010"), None, None, None, None, false)
            .map_err(|e| e.to_string())?;
    if local.is_none() {
        return Err("plain HTTP to this machine needs no flag".into());
    }
    Ok(())
}

/// Header-mode tenancies tell the archive the tenant in a header it trusts;
/// the keyed ones prove it.
#[test]
fn header_mode_tenancies_trust_headers_and_keyed_ones_do_not() {
    let at = |tenancy| CopalArchive::new("127.0.0.1:9010", tenancy);
    assert!(at(CopalTenancy::PerWorkspace).trusts_headers());
    assert!(at(CopalTenancy::Shared("one".into())).trusts_headers());
    assert!(!at(CopalTenancy::SharedKey("ck1_x".into())).trusts_headers());
    assert!(!at(CopalTenancy::PerWorkspaceKeys(HashMap::from([(
        "ws:a".to_string(),
        "ck1_a".to_string()
    )])))
    .trusts_headers());
}

#[test]
fn a_slugless_title_still_gets_a_path() -> anyhow::Result<()> {
    // A title with no ASCII alphanumerics falls back to the bare hash.
    let p = document_path("——", 0xdead);
    assert!(p.starts_with("antumbra/") && p.ends_with(".txt"), "{p}");
    Ok(())
}

#[tokio::test]
async fn a_failed_create_surfaces_and_nothing_is_uploaded() -> anyhow::Result<()> {
    let t = Arc::new(FakeTransport {
        create: Err("connection refused".into()),
        ..FakeTransport::happy()
    });
    assert!(archive("127.0.0.1:9010", t.clone())
        .archive_document("ws:one", "guide", None, "text")
        .await
        .is_err());
    assert_eq!(t.calls().len(), 1, "no upload after a failed create");
    Ok(())
}

#[tokio::test]
async fn a_failed_upload_surfaces() -> anyhow::Result<()> {
    let t = Arc::new(FakeTransport {
        upload: Err("http status: 500".into()),
        ..FakeTransport::happy()
    });
    assert!(archive("127.0.0.1:9010", t)
        .archive_document("ws:one", "guide", None, "text")
        .await
        .is_err());
    Ok(())
}

#[tokio::test]
async fn a_create_response_without_an_id_is_an_error() -> anyhow::Result<()> {
    let t = Arc::new(FakeTransport {
        create: Ok(json!({ "unexpected": true })),
        ..FakeTransport::happy()
    });
    assert!(archive("127.0.0.1:9010", t.clone())
        .archive_document("ws:one", "guide", None, "text")
        .await
        .is_err());
    assert_eq!(t.calls().len(), 1, "no upload without a file id");
    Ok(())
}

#[tokio::test]
async fn an_upload_response_without_a_digest_is_an_error() -> anyhow::Result<()> {
    let t = Arc::new(FakeTransport {
        upload: Ok(json!({ "state": "ready" })),
        ..FakeTransport::happy()
    });
    assert!(archive("127.0.0.1:9010", t)
        .archive_document("ws:one", "guide", None, "text")
        .await
        .is_err());
    Ok(())
}

#[test]
fn a_content_digest_is_copal_s_sha256_hex() -> anyhow::Result<()> {
    assert_eq!(
        content_digest(b"test"),
        "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
    );
    Ok(())
}

#[tokio::test]
async fn a_replay_already_holding_these_bytes_is_not_uploaded_again() -> anyhow::Result<()> {
    let held = content_digest(b"the original text");
    let t = Arc::new(FakeTransport {
        create: Ok(json!({ "id": "01k6f2x9q3w8e5r7t1y4z6a8b0", "state": "ready", "digest": held })),
        ..FakeTransport::happy()
    });
    let got = archive("127.0.0.1:9010", t.clone())
        .archive_document("ws:one", "Guide", None, "the original text")
        .await?;
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
        .await?;
    assert_eq!(t.calls().len(), 2);
    Ok(())
}

/// A shared-tenancy archive that knows where relics are displayed.
fn relic_archive(transport: Arc<FakeTransport>) -> CopalArchive {
    archive("127.0.0.1:9010", transport).with_render_origin(Some("http://10.0.0.130:8083/".into()))
}

#[tokio::test]
async fn a_relic_version_is_its_own_html_file() -> anyhow::Result<()> {
    let t = Arc::new(FakeTransport::happy());
    let got = relic_archive(t.clone())
        .put_relic("ws:one", "ceiling", 2, "The Ceiling", "<p>page</p>", true)
        .await?;
    assert_eq!(got.file_id, "01k6f2x9q3w8e5r7t1y4z6a8b0");

    let calls = t.calls();
    let Call::Post { url, body, .. } = &calls[0] else {
        bail!("first call is the create, got {calls:?}");
    };
    assert_eq!(url, "http://127.0.0.1:9010/v1/files");
    assert_eq!(body["path"], "relics/ceiling/v2.html");
    assert_eq!(body["content_type"], "text/html; charset=utf-8");
    assert_eq!(body["access"], "public");
    assert_eq!(body["metadata"]["slug"], "ceiling");
    assert_eq!(body["metadata"]["version"], 2);
    let Call::Put {
        content_type, body, ..
    } = &calls[1]
    else {
        bail!("second call is the upload, got {calls:?}");
    };
    assert_eq!(content_type, "text/html; charset=utf-8");
    assert_eq!(body, b"<p>page</p>");
    Ok(())
}

#[tokio::test]
async fn a_retried_publish_replays_its_version_and_the_next_one_does_not() -> anyhow::Result<()> {
    let key = |calls: &[Call]| -> anyhow::Result<String> {
        match &calls[0] {
            Call::Post { body, .. } => body["idempotency_key"]
                .as_str()
                .map(str::to_string)
                .context("the create has no idempotency key"),
            other => bail!("{other:?}"),
        }
    };
    let publish = |workspace: &'static str, number: u32| async move {
        let t = Arc::new(FakeTransport::happy());
        relic_archive(t.clone())
            .put_relic(workspace, "ceiling", number, "t", "a", false)
            .await?;
        key(&t.calls())
    };
    let first = publish("ws:one", 1).await?;
    assert_eq!(first, publish("ws:one", 1).await?, "a retry replays");
    assert_ne!(
        first,
        publish("ws:one", 2).await?,
        "the next version is new"
    );
    assert_ne!(
        first,
        publish("ws:two", 1).await?,
        "another workspace is new"
    );
    assert!(first.starts_with("antumbra-relic-"));
    Ok(())
}

#[tokio::test]
async fn access_changes_post_the_level() -> anyhow::Result<()> {
    let t = Arc::new(FakeTransport::happy());
    relic_archive(t.clone())
        .set_access("ws:one", "01abc", false)
        .await?;
    let calls = t.calls();
    let Call::Post {
        url,
        body,
        credential,
    } = &calls[0]
    else {
        bail!("{calls:?}");
    };
    assert_eq!(url, "http://127.0.0.1:9010/v1/files/01abc/access");
    assert_eq!(body, &json!({ "access": "private" }));
    assert_eq!(credential, &tenant("antumbra"));
    Ok(())
}

#[tokio::test]
async fn render_links_must_land_on_the_render_origin() -> anyhow::Result<()> {
    let t = Arc::new(FakeTransport {
        create: Ok(json!({
            "url": "http://10.0.0.130:8083/v1/render/cg1.abc.def",
            "expires_at": "2026-10-24T20:00:00Z",
        })),
        ..FakeTransport::happy()
    });
    let link = relic_archive(t.clone())
        .render_url("ws:one", "01abc", 300)
        .await?;
    assert_eq!(link.url, "http://10.0.0.130:8083/v1/render/cg1.abc.def");
    assert_eq!(link.expires_at.as_deref(), Some("2026-10-24T20:00:00Z"));
    let calls = t.calls();
    let Call::Post { url, body, .. } = &calls[0] else {
        bail!("{calls:?}");
    };
    assert_eq!(url, "http://127.0.0.1:9010/v1/files/01abc/render-url");
    assert_eq!(body, &json!({ "ttl_secs": 300 }));

    // Anything else is refused, so the dashboard never frames it.
    let elsewhere = Arc::new(FakeTransport {
        create: Ok(json!({ "url": "http://10.0.0.130:8083.evil.example/v1/render/x" })),
        ..FakeTransport::happy()
    });
    assert!(relic_archive(elsewhere)
        .render_url("ws:one", "01abc", 300)
        .await
        .is_err());
    // And with no render origin there is nothing to ask for.
    assert!(archive("127.0.0.1:9010", Arc::new(FakeTransport::happy()))
        .render_url("ws:one", "01abc", 300)
        .await
        .is_err());
    Ok(())
}

#[test]
fn a_public_file_has_a_permanent_address() -> anyhow::Result<()> {
    let t = Arc::new(FakeTransport::happy());
    assert_eq!(
        relic_archive(t.clone()).public_url("01abc").as_deref(),
        Some("http://10.0.0.130:8083/v1/public/01abc")
    );
    assert_eq!(archive("127.0.0.1:9010", t).public_url("01abc"), None);
    Ok(())
}

#[test]
fn a_render_origin_is_an_origin_and_nothing_more() -> anyhow::Result<()> {
    let built = CopalArchive::from_flags(
        Some("127.0.0.1:9010"),
        Some("antumbra".into()),
        None,
        None,
        Some("http://10.0.0.130:8083/".into()),
        false,
    )?
    .context("an address and a tenant configured no archive")?;
    assert_eq!(built.render_origin(), Some("http://10.0.0.130:8083"));
    for bad in [
        "10.0.0.130:8083",
        "http://h/path",
        "https://h; frame-src *",
        "ftp://h",
    ] {
        let refused = CopalArchive::from_flags(
            Some("127.0.0.1:9010"),
            None,
            None,
            None,
            Some(bad.into()),
            false,
        );
        assert!(
            matches!(refused, Err(CopalConfigError::RenderOrigin(_))),
            "{bad:?}"
        );
    }
    Ok(())
}

/// Answers every read with `content` and records what was asked for.
struct ReadingTransport {
    content: Vec<u8>,
    reads: Mutex<Vec<(String, CopalCredential, u64)>>,
}

impl ReadingTransport {
    /// Every read it was asked for, in order. The fake only ever pushes to
    /// the list, so a poisoned lock holds a whole list and is read as is.
    fn reads(&self) -> Vec<(String, CopalCredential, u64)> {
        self.reads
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

impl CopalTransport for ReadingTransport {
    fn post_json(&self, _url: &str, _credential: &CopalCredential, _body: &Value) -> Result<Value> {
        Err(AntumbraError::other("a read never posts"))
    }

    fn put_bytes(
        &self,
        _url: &str,
        _credential: &CopalCredential,
        _content_type: &str,
        _digest: &str,
        _body: &[u8],
    ) -> Result<Value> {
        Err(AntumbraError::other("a read never uploads"))
    }

    fn get_bytes(&self, url: &str, credential: &CopalCredential, limit: u64) -> Result<Vec<u8>> {
        // Read as `reads` is: a push cannot leave the list half-written.
        self.reads
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push((url.into(), credential.clone(), limit));
        Ok(self.content.clone())
    }
}

#[tokio::test]
async fn a_file_is_read_back_as_its_workspace_under_a_size_limit() -> anyhow::Result<()> {
    let t = Arc::new(ReadingTransport {
        content: b"<h1>kept</h1>".to_vec(),
        reads: Mutex::new(Vec::new()),
    });
    let keyed = CopalArchive::with_transport(
        "http://copal:8080/",
        CopalTenancy::PerWorkspaceKeys([("ws:one".to_string(), "ck1.one".to_string())].into()),
        t.clone(),
    );
    let bytes = keyed.read_file("ws:one", "01relicfile").await?;
    assert_eq!(bytes, b"<h1>kept</h1>");
    assert_eq!(
        t.reads(),
        vec![(
            "http://copal:8080/v1/files/01relicfile/content".to_string(),
            bearer("ck1.one"),
            MAX_READ_BYTES,
        )]
    );
    // A workspace with no key reads nothing, closed, like every other call.
    assert!(keyed.read_file("ws:other", "01relicfile").await.is_err());
    assert_eq!(t.reads().len(), 1);
    Ok(())
}

#[tokio::test]
async fn a_transport_that_only_archives_refuses_to_read() -> anyhow::Result<()> {
    let Err(refused) = archive("127.0.0.1:9010", Arc::new(FakeTransport::happy()))
        .read_file("ws:one", "01relicfile")
        .await
    else {
        bail!("a transport that only archives read a file");
    };
    assert!(
        refused.to_string().contains("cannot read files"),
        "{refused}"
    );
    Ok(())
}
