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
//! record per ingest. The upload declares the bytes' sha256 in
//! `x-copal-digest`, so copal refuses bytes that are not the ones hashed here,
//! and a replayed record that already holds exactly these bytes is not
//! uploaded again, since copal would add a version for identical content.
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

mod transport;

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
/// file record) and PUT raw bytes (upload the content, declaring their sha256
/// `digest` for copal to verify), each authenticated by a [`CopalCredential`]
/// and returning the parsed JSON response. Behind a trait so the archive's
/// logic is testable without a socket.
pub trait CopalTransport: Send + Sync {
    fn post_json(&self, url: &str, credential: &CopalCredential, body: &Value) -> Result<Value>;
    fn put_bytes(
        &self,
        url: &str,
        credential: &CopalCredential,
        content_type: &str,
        digest: &str,
        body: &[u8],
    ) -> Result<Value>;
}

/// The sha256 of `bytes` as copal writes a content digest: 64 lowercase hex
/// characters.
pub fn content_digest(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
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
        Self::with_transport(addr, tenancy, Arc::new(transport::UreqTransport::new()))
    }

    /// As [`Self::new`] with an explicit transport: the test seam, and how any
    /// crate that ingests fakes copal in its own tests.
    pub fn with_transport(
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
            let digest = content_digest(content.as_bytes());
            if created.get("digest").and_then(Value::as_str) == Some(digest.as_str()) {
                return Ok(ArchivedDocument {
                    file_id: file_id.to_string(),
                    digest,
                });
            }
            let uploaded = transport.put_bytes(
                &format!("{base}/v1/files/{file_id}/content"),
                &credential,
                "text/plain",
                &digest,
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

/// Why the operator's copal flags could not become an archive. Every message
/// names the flag and the workspace, never a credential.
#[derive(Debug, thiserror::Error)]
pub enum CopalConfigError {
    #[error(
        "--copal-tenant, --copal-key, and --copal-keys each pick a copal tenancy; pass at most one"
    )]
    ConflictingTenancy,
    #[error("cannot read --copal-keys {path}: {reason}")]
    KeysUnreadable { path: String, reason: String },
    #[error("--copal-keys {path} is not a JSON object of workspace -> key: {reason}")]
    KeysNotAnObject { path: String, reason: String },
    #[error("--copal-keys {path} maps no workspaces")]
    KeysEmpty { path: String },
    #[error(
        "--copal-keys {path}: the key for workspace {workspace} is not a plain ASCII credential"
    )]
    KeyNotAscii { path: String, workspace: String },
}

impl CopalArchive {
    /// Resolve an archive from the operator's flags, the same way for every
    /// binary that ingests (the MCP server and the CLI): `None` when no
    /// address is configured, so ingest keeps only the chunks. The tenancy is
    /// exactly one of `tenant` (header auth, shared), `key` (keys auth,
    /// shared), `keys_file` (keys auth, per-workspace), or none of them
    /// (header auth, per-workspace, the default). Naming two is a
    /// contradiction refused up front, not a precedence resolved in silence.
    pub fn from_flags(
        addr: Option<&str>,
        tenant: Option<String>,
        key: Option<String>,
        keys_file: Option<&std::path::Path>,
    ) -> std::result::Result<Option<Arc<Self>>, CopalConfigError> {
        let Some(addr) = addr else {
            return Ok(None);
        };
        let tenancy = match (tenant, key, keys_file) {
            (None, None, None) => CopalTenancy::PerWorkspace,
            (Some(t), None, None) => CopalTenancy::Shared(t),
            (None, Some(k), None) => CopalTenancy::SharedKey(k),
            (None, None, Some(path)) => CopalTenancy::PerWorkspaceKeys(load_keys(path)?),
            _ => return Err(CopalConfigError::ConflictingTenancy),
        };
        Ok(Some(Arc::new(Self::new(addr, tenancy))))
    }
}

/// Load and validate the workspace -> `ck1` key map for per-workspace keys
/// tenancy: a JSON object of strings, non-empty, every key a plain
/// printable-ASCII credential (a real `ck1` token is; anything else is a
/// mangled file worth stopping over up front rather than at some tenant's
/// first ingest). Errors name the workspace, never the credential.
pub fn load_keys(
    path: &std::path::Path,
) -> std::result::Result<HashMap<String, String>, CopalConfigError> {
    let shown = path.display().to_string();
    let raw = std::fs::read_to_string(path).map_err(|e| CopalConfigError::KeysUnreadable {
        path: shown.clone(),
        reason: e.to_string(),
    })?;
    let keys: HashMap<String, String> =
        serde_json::from_str(&raw).map_err(|e| CopalConfigError::KeysNotAnObject {
            path: shown.clone(),
            reason: e.to_string(),
        })?;
    if keys.is_empty() {
        return Err(CopalConfigError::KeysEmpty { path: shown });
    }
    for (workspace, key) in &keys {
        if key.is_empty() || !key.bytes().all(|b| b.is_ascii_graphic()) {
            return Err(CopalConfigError::KeyNotAscii {
                path: shown,
                workspace: workspace.clone(),
            });
        }
    }
    Ok(keys)
}

#[cfg(test)]
mod tests;
