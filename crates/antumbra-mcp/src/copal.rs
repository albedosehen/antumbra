//! The copal document-of-record archive: on ingest, a knowledge document's
//! ORIGINAL content is uploaded to a copal file service (content-addressed,
//! versioned, sealed at rest) before its chunks are stored, and every chunk
//! carries the archived file's id + digest back to it. Without this, ingest
//! keeps only the chunks -- recall works, but the original bytes are gone.
//!
//! Two calls against copal's REST face: `POST {base}/v1/files` creates (or,
//! replayed, returns) the file record, then `PUT {base}/v1/files/{id}/content`
//! uploads the raw bytes and answers with the content digest. The create
//! carries an idempotency key derived from (workspace, title), so re-ingesting
//! the same title *revisions* the same copal file instead of littering a new
//! record per ingest.
//!
//! Tenancy is a deployment choice ([`CopalTenancy`]), two axes with two
//! answers each. *Which copal tenant?* Per-workspace (each antumbra workspace
//! is a full copal tenant: its own quotas, listings, and search scopes, with
//! cross-tenant reads refusing at copal's own boundary) or shared (every
//! workspace under one). *How does a call prove it?* Copal's `header` auth
//! mode trusts `x-copal-tenant` as the identity; its deployed `keys` mode
//! binds the tenant to a `ck1` bearer credential and ignores what a header
//! claims. The four shapes:
//!
//! - [`CopalTenancy::PerWorkspace`] (the default): the workspace rides the
//!   `x-copal-tenant` header. Header auth mode.
//! - [`CopalTenancy::Shared`] (`--copal-tenant`): one configured tenant rides
//!   the header. Header auth mode.
//! - [`CopalTenancy::PerWorkspaceKeys`] (`--copal-keys`, a JSON file mapping
//!   workspace to `ck1` key): each workspace authenticates with its own
//!   credential, so per-workspace tenancy survives copal's keys-mode upgrade.
//!   A workspace with no key FAILS its ingest (fail closed, like every other
//!   fault here) rather than silently landing in someone else's tenant.
//! - [`CopalTenancy::SharedKey`] (`--copal-key`): one credential, one tenant
//!   (the key's own). Keys auth mode.
//!
//! In EVERY mode the idempotency key and archived path derive from the
//! *(workspace, title)* pair, so two workspaces ingesting the same title can
//! never revision each other's document of record -- under a shared tenant
//! that derivation is the isolation; under per-workspace tenancy copal's
//! boundary isolates again above it -- and a deployment moving between modes
//! never re-identifies a document.
//!
//! The irreducible network calls are isolated behind [`CopalTransport`] (the
//! same seam the embedder uses), so the request shaping, response parsing, and
//! ordering are mock-tested offline. The address, tenancy, and credentials are
//! **operator-configured**; the only request-derived value is the workspace
//! tenant, which comes from the session's verified identity (never from tool
//! arguments) and is refused as a header unless it is a plain printable-ASCII
//! token. Keys ride the `Authorization` header (never the URL, never logged;
//! [`CopalCredential`]'s `Debug` redacts them), and the URL is never
//! request-derived, so this is not an SSRF sink.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::{json, Value};

use antumbra_core::{AntumbraError, Result};

/// How one request proves its copal tenant: the `x-copal-tenant` header
/// (copal's `header` auth mode, where the header IS the identity) or a `ck1`
/// bearer key (its deployed `keys` mode, where the tenant comes out of the
/// credential and no header can name one).
#[derive(Clone, PartialEq, Eq)]
pub enum CopalCredential {
    /// `x-copal-tenant: <tenant>` -- header auth mode.
    Tenant(String),
    /// `Authorization: Bearer <ck1 key>` -- keys auth mode.
    Bearer(String),
}

/// Redacts the bearer secret: a credential in a panic message, an error chain,
/// or a debug log must never be a credential leak (the `HttpEmbedder` rule,
/// enforced here by construction instead of by omitting `Debug`, because the
/// tests assert on recorded credentials).
impl std::fmt::Debug for CopalCredential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CopalCredential::Tenant(t) => f.debug_tuple("Tenant").field(t).finish(),
            CopalCredential::Bearer(_) => f.debug_tuple("Bearer").field(&"<redacted>").finish(),
        }
    }
}

/// The two network operations an archive performs: POST a JSON body (create the
/// file record) and PUT raw bytes (upload the content), each authenticated by a
/// [`CopalCredential`] and returning the parsed JSON response. Behind a trait
/// so the archive's logic is testable without a socket.
pub trait CopalTransport: Send + Sync {
    fn post_json(&self, url: &str, credential: &CopalCredential, body: &Value) -> Result<Value>;
    fn put_bytes(
        &self,
        url: &str,
        credential: &CopalCredential,
        content_type: &str,
        body: &[u8],
    ) -> Result<Value>;
}

/// Default per-request budget for the copal endpoint, overridable with
/// `ANTUMBRA_COPAL_TIMEOUT_SECS` (the embedder's `EMBED_TIMEOUT_SECS`
/// pattern). Without a bound a hung archive pins the ingest forever -- and the
/// ingest deliberately FAILS when a configured archive is unreachable, so the
/// bound is what turns "hangs" into "fails fast with a clear error".
const COPAL_TIMEOUT_SECS: u64 = 30;

/// The production transport: blocking `ureq` calls over an agent with a bounded
/// global timeout, so a dead copal fails fast instead of hanging the worker.
struct UreqTransport {
    agent: ureq::Agent,
}

impl UreqTransport {
    fn new() -> Self {
        let secs = std::env::var("ANTUMBRA_COPAL_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .filter(|&s| s > 0)
            .unwrap_or(COPAL_TIMEOUT_SECS);
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(std::time::Duration::from_secs(secs)))
            .build()
            .into();
        Self { agent }
    }
}

impl CopalTransport for UreqTransport {
    fn post_json(&self, url: &str, credential: &CopalCredential, body: &Value) -> Result<Value> {
        let req = self.agent.post(url);
        let mut resp = authed(req, credential)
            .send_json(body)
            .map_err(|e| AntumbraError::other(format!("copal POST {url} failed: {e}")))?;
        resp.body_mut()
            .read_json::<Value>()
            .map_err(|e| AntumbraError::other(format!("copal response was not JSON: {e}")))
    }

    fn put_bytes(
        &self,
        url: &str,
        credential: &CopalCredential,
        content_type: &str,
        body: &[u8],
    ) -> Result<Value> {
        let req = self.agent.put(url).header("content-type", content_type);
        let mut resp = authed(req, credential)
            .send(body)
            .map_err(|e| AntumbraError::other(format!("copal PUT {url} failed: {e}")))?;
        resp.body_mut()
            .read_json::<Value>()
            .map_err(|e| AntumbraError::other(format!("copal response was not JSON: {e}")))
    }
}

/// Apply `credential` to a request: the tenant header (copal header auth mode)
/// or the bearer key (keys mode). The key rides the `Authorization` header,
/// never the URL.
fn authed<B>(
    req: ureq::RequestBuilder<B>,
    credential: &CopalCredential,
) -> ureq::RequestBuilder<B> {
    match credential {
        CopalCredential::Tenant(t) => req.header("x-copal-tenant", t),
        CopalCredential::Bearer(k) => req.header("authorization", &format!("Bearer {k}")),
    }
}

/// What the archive hands back for a stored document: enough for every chunk
/// to name its document of record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchivedDocument {
    /// The copal file id the original content lives under.
    pub file_id: String,
    /// The content digest copal computed for the uploaded bytes.
    pub digest: String,
}

/// Whose copal tenant an archived document lands in, and how a call proves it
/// (see the module docs for the four shapes). The archive identity
/// (idempotency key + path) derives from the workspace in every mode; tenancy
/// decides only the boundary the file lives inside and the credential on the
/// wire. No `Debug`: two variants carry `ck1` secrets.
pub enum CopalTenancy {
    /// Each antumbra workspace IS its own copal tenant: the workspace tenant
    /// rides the `x-copal-tenant` header, so quotas, listings, and search
    /// scope per workspace and copal's own tenant boundary isolates them.
    /// Needs copal's header auth mode (the header is the trusted identity
    /// there).
    PerWorkspace,
    /// Every workspace lands under this one configured copal tenant, named by
    /// the header. Header auth mode.
    Shared(String),
    /// Each antumbra workspace authenticates with its own `ck1` bearer key
    /// (the map is workspace -> key), so per-workspace tenancy holds under
    /// copal's deployed `keys` auth mode, where the tenant is bound to the
    /// credential. Loaded once at startup (`--copal-keys`); restart to pick up
    /// keys minted afterward, like the serving engine picks up experts. A
    /// workspace absent from the map fails its ingest, closed.
    PerWorkspaceKeys(HashMap<String, String>),
    /// One `ck1` bearer key for every workspace: shared tenancy under keys
    /// auth mode -- the tenant is whichever one the key is bound to.
    SharedKey(String),
}

/// Archives ingested documents to a copal file service (the document of
/// record). One instance per server, operator-configured; absent, ingest
/// behaves exactly as before it existed.
pub struct CopalArchive {
    /// The normalized URL base (`http://host:port` or the full base as given).
    base: String,
    /// Which copal tenant receives each document (see [`CopalTenancy`]).
    tenancy: CopalTenancy,
    transport: Arc<dyn CopalTransport>,
}

impl CopalArchive {
    /// Point at `addr` -- a bare `host:port` (given `http://`) or a full URL
    /// base (used as-is) -- landing documents per `tenancy`.
    pub fn new(addr: &str, tenancy: CopalTenancy) -> Self {
        Self::with_transport(addr, tenancy, Arc::new(UreqTransport::new()))
    }

    /// As [`Self::new`] with an explicit transport (the test seam).
    pub(crate) fn with_transport(
        addr: &str,
        tenancy: CopalTenancy,
        transport: Arc<dyn CopalTransport>,
    ) -> Self {
        Self {
            base: normalize_base(addr),
            tenancy,
            transport,
        }
    }

    /// Upload `content` as the document of record for (`workspace`, `title`,
    /// `source`): create the file record (idempotent on (workspace, title), so
    /// a re-ingest revisions the same file and two workspaces sharing a title
    /// never revision each other's), then PUT the raw bytes. Returns the file
    /// id + content digest every chunk of the document is stamped with.
    /// `workspace` is the ingesting antumbra tenant; the archive's
    /// [`CopalTenancy`] decides how (and as whom) the calls authenticate.
    ///
    /// Any failure is an error, never a partial success -- the caller fails
    /// the ingest rather than storing chunks whose original was silently
    /// dropped. That includes a workspace with no configured key under
    /// [`CopalTenancy::PerWorkspaceKeys`]: refusing beats archiving into a
    /// tenant that is not the workspace's own.
    pub async fn archive_document(
        &self,
        workspace: &str,
        title: &str,
        source: Option<&str>,
        content: &str,
    ) -> Result<ArchivedDocument> {
        let credential = match &self.tenancy {
            CopalTenancy::Shared(t) => CopalCredential::Tenant(t.clone()),
            CopalTenancy::PerWorkspace => {
                // The workspace becomes an HTTP header value here. Session
                // identities are verified upstream, but a header is a syntax,
                // not just a trust question: refuse anything that is not a
                // plain printable-ASCII token rather than hand the HTTP layer
                // a value it would reject (or worse, split) mid-ingest.
                if workspace.is_empty() || !workspace.bytes().all(|b| b.is_ascii_graphic()) {
                    return Err(AntumbraError::other(format!(
                        "workspace {workspace:?} cannot be presented as a copal tenant"
                    )));
                }
                CopalCredential::Tenant(copal_tenant_name(workspace))
            }
            CopalTenancy::SharedKey(key) => CopalCredential::Bearer(key.clone()),
            CopalTenancy::PerWorkspaceKeys(keys) => {
                let key = keys.get(workspace).ok_or_else(|| {
                    AntumbraError::other(format!(
                        "no copal key is configured for workspace {workspace} \
                         (mint one and add it to --copal-keys, then restart)"
                    ))
                })?;
                CopalCredential::Bearer(key.clone())
            }
        };
        let base = self.base.clone();
        let transport = self.transport.clone();
        let workspace = workspace.to_string();
        let title = title.to_string();
        let source = source.map(str::to_string);
        let content = content.to_string();
        // `ureq` is blocking; run both calls off the async runtime so they
        // never stall a worker (the same offload the embedder uses).
        tokio::task::spawn_blocking(move || {
            let hash = provenance_hash(&workspace, &title);
            // The workspace rides the metadata in both modes: under shared
            // tenancy it is the human-readable answer to "whose document is
            // this?" (which the hash in the path only implies), and under
            // per-workspace tenancy it keeps the file self-describing even
            // when exported past copal's tenant boundary.
            let mut body = json!({
                "path": document_path(&title, hash),
                "content_type": "text/plain",
                "idempotency_key": idempotency_key(hash),
                "metadata": { "title": title, "workspace": workspace },
            });
            if let Some(src) = source {
                body["metadata"]["source"] = json!(src);
            }
            let created = transport.post_json(&format!("{base}/v1/files"), &credential, &body)?;
            let file_id = created
                .get("id")
                .and_then(Value::as_str)
                .ok_or_else(|| AntumbraError::other("copal create response missing `id`"))?;
            let uploaded = transport.put_bytes(
                &format!("{base}/v1/files/{file_id}/content"),
                &credential,
                "text/plain",
                content.as_bytes(),
            )?;
            let digest = uploaded
                .get("digest")
                .and_then(Value::as_str)
                .ok_or_else(|| AntumbraError::other("copal upload response missing `digest`"))?;
            Ok(ArchivedDocument {
                file_id: file_id.to_string(),
                digest: digest.to_string(),
            })
        })
        .await
        .map_err(|e| AntumbraError::other(format!("copal archive task panicked: {e}")))?
    }
}

/// Mirror how copal itself normalizes an embedder address: a full URL base is
/// used as-is (trailing slash trimmed), a bare `host:port` gets `http://`.
fn normalize_base(addr: &str) -> String {
    let addr = addr.trim_end_matches('/');
    if addr.starts_with("http://") || addr.starts_with("https://") {
        addr.to_string()
    } else {
        format!("http://{addr}")
    }
}

/// A stable 64-bit FNV-1a over (workspace, title) -- the identity of a
/// document within the archive. No extra deps (the `next_id` reasoning); a
/// collision merely makes two titles revision one copal file, and the chunks
/// still point at whatever id copal actually returned. NUL-separated: neither
/// a workspace tenant nor a meaningful title carries `\0`, so the pair is
/// unambiguous. When the workspace IS the configured copal tenant (stdio with
/// matching names, or any single-workspace deployment), this is byte-identical
/// to the original (tenant, title) derivation, so existing archived files keep
/// revisioning under their old keys and paths.
/// A workspace rendered into copal's tenant grammar, which is
/// `[A-Za-z0-9_-]` (its TenantId refuses anything else with a 400 -- found
/// live: every real antumbra workspace is `ws:<name>`, and the colon killed
/// the first per-workspace ingest outright). Names already inside the
/// grammar pass through untouched. Otherwise every foreign character
/// becomes `-` and the result carries a short hash of the ORIGINAL, so the
/// mapping stays injective: `ws:shon` -> `ws-shon-<8 hex>`, which can never
/// collide with a workspace literally named `ws-shon`. The exact original
/// still rides every create as `metadata.workspace`, so copal-side listings
/// stay attributable to the real workspace name.
fn copal_tenant_name(workspace: &str) -> String {
    let clean = workspace
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-');
    if clean {
        return workspace.to_string();
    }
    let sanitized: String = workspace
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '-'
            }
        })
        .collect();
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in workspace.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{sanitized}-{:08x}", (h >> 32) as u32)
}

fn provenance_hash(workspace: &str, title: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in workspace.bytes().chain([0u8]).chain(title.bytes()) {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The create's idempotency key: stable per (workspace, title), so copal
/// replays the original record and the upload revisions it.
fn idempotency_key(hash: u64) -> String {
    format!("antumbra-doc-{hash:016x}")
}

/// A readable, collision-safe path for the archived document: a slug of the
/// title (for humans listing the tenant's files) plus the (workspace, title)
/// hash (so two titles sharing a slug -- or two workspaces sharing a title --
/// still land on distinct paths).
fn document_path(title: &str, hash: u64) -> String {
    let mut slug = String::new();
    for c in title.chars() {
        if c.is_ascii_alphanumeric() {
            slug.push(c.to_ascii_lowercase());
        } else if !slug.ends_with('-') && !slug.is_empty() {
            slug.push('-');
        }
        if slug.len() >= 64 {
            break;
        }
    }
    let slug = slug.trim_end_matches('-');
    if slug.is_empty() {
        format!("antumbra/{hash:016x}.txt")
    } else {
        format!("antumbra/{slug}-{hash:016x}.txt")
    }
}

#[cfg(test)]
mod tests {
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
                create: Ok(json!({ "id": "file:01J", "state": "draft" })),
                upload: Ok(json!({ "digest": "sha256:abc", "state": "ready" })),
                calls: Mutex::new(Vec::new()),
            }
        }

        fn calls(&self) -> Vec<Call> {
            self.calls.lock().unwrap().clone()
        }
    }

    impl CopalTransport for FakeTransport {
        fn post_json(
            &self,
            url: &str,
            credential: &CopalCredential,
            body: &Value,
        ) -> Result<Value> {
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
            body: &[u8],
        ) -> Result<Value> {
            self.calls.lock().unwrap().push(Call::Put {
                url: url.into(),
                credential: credential.clone(),
                content_type: content_type.into(),
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
        assert_eq!(got.file_id, "file:01J");
        assert_eq!(got.digest, "sha256:abc");

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
            body,
        } = &calls[1]
        else {
            panic!("second call is the upload, got {calls:?}");
        };
        assert_eq!(url, "http://127.0.0.1:9010/v1/files/file:01J/content");
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
        let (Call::Post { body: b1, .. }, Call::Post { body: b2, .. }) = (&calls[0], &calls[2])
        else {
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
        let (Call::Post { body: b1, .. }, Call::Post { body: b2, .. }) = (&calls[0], &calls[2])
        else {
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
        let (Call::Post { body: b1, .. }, Call::Post { body: b2, .. }) = (&calls[0], &calls[2])
        else {
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
}
