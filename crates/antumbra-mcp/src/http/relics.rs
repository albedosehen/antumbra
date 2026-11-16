//! The relics pages: the gallery (`/relics`) and one relic's viewer
//! (`/relic/<slug>`), beside the dashboard and built the same way.
//!
//! Neither page renders a relic itself. A relic is an agent-written page with
//! scripts of its own, so it is only ever displayed in an `<iframe>` from
//! copal's render origin, where it runs sandboxed in an opaque origin; this
//! server's origin, where the token lives, never parses it. The pages' policy
//! allows framing that one origin and nothing else.
//!
//! Like the dashboard, the pages carry no data. Everything they show comes from
//! the relic tools through `POST /mcp/call` with the signed-in token, except
//! for a public relic: `GET /relic/<slug>/public` answers without a token, for
//! a public relic only, with what its viewer needs and nothing about who owns
//! it. A private relic and a missing one are the same 404 there.
//!
//! `GET /relic/<slug>/export?format=html|md|pdf` hands the current version
//! over as a file to keep: the HTML as it was published, Markdown made from
//! it, or a PDF printed by the operator's renderer. It answers whoever may
//! view the relic (the token's identity, or anyone for a public one), reads
//! the bytes from copal on the server, and sends them as an attachment under a
//! `sandbox` policy, so this origin never runs what a relic holds.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::rejection::ExtensionRejection;
use axum::extract::{ConnectInfo, Path, Query, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Redirect, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::Mutex;

use antumbra_core::relic::{html_to_markdown, is_valid_slug};
use antumbra_core::{Relic, Visibility};
use antumbra_store::repo::relic as store_relic;

use antumbra_core::limit;

use super::{Bounded, HttpState, Serving};
use crate::auth::Identity;

const GALLERY: &str = include_str!("dashboard/relics.html");
const VIEWER: &str = include_str!("dashboard/relic.html");

/// What one client may ask of the public lookup. A page load reads it once,
/// so thirty at once and two a second after that is a great deal of reloading.
pub(super) const PUBLIC_READS_PER_CLIENT: limit::Rate = limit::Rate {
    burst: 30,
    per_minute: 120,
};
/// What everyone together may ask of the public lookup.
pub(super) const PUBLIC_READS_EVERYONE: limit::Rate = limit::Rate {
    burst: 300,
    per_minute: 3_000,
};
/// What one client may have printed: a print is up to a minute of the
/// renderer's time, and a version prints once (see [`Guards::pdfs`]).
pub(super) const EXPORTS_PER_CLIENT: limit::Rate = limit::Rate {
    burst: 3,
    per_minute: 6,
};
/// What everyone together may have printed.
pub(super) const EXPORTS_EVERYONE: limit::Rate = limit::Rate {
    burst: 20,
    per_minute: 60,
};
/// A print this size or smaller is kept for the next request of its version.
const MAX_CACHED_PDF_BYTES: usize = 8 * 1024 * 1024;
/// How many printed versions are kept.
const CACHED_PDFS: usize = 16;

/// Printed PDFs by (slug, version).
type PrintedPdfs = Mutex<Bounded<(String, u32), Arc<Vec<u8>>>>;

/// What bounds the routes here that answer without a token.
pub(super) struct Guards {
    /// The budgets for `GET /relic/{slug}/public`.
    pub(super) public_reads: limit::Limiter,
    /// The budgets for printing a relic as PDF.
    pub(super) exports: limit::Limiter,
    /// Printed PDFs by (slug, version), so a repeat request of the same version
    /// is answered without printing again.
    pub(super) pdfs: PrintedPdfs,
}

impl Guards {
    pub(super) fn new() -> Self {
        Self {
            public_reads: limit::Limiter::new(PUBLIC_READS_PER_CLIENT, PUBLIC_READS_EVERYONE),
            exports: limit::Limiter::new(EXPORTS_PER_CLIENT, EXPORTS_EVERYONE),
            pdfs: Mutex::new(Bounded::new(CACHED_PDFS)),
        }
    }
}

/// The client a request comes from, for its bucket: its peer address when the
/// server was started with connection info, else one shared key, so a request
/// through `oneshot` in a test, or behind a proxy that hides the peer, still
/// meets everyone's bound.
fn client_of(peer: Result<ConnectInfo<SocketAddr>, ExtensionRejection>) -> String {
    match peer {
        Ok(ConnectInfo(addr)) => addr.ip().to_string(),
        Err(_) => "unknown".to_string(),
    }
}

/// The answer to a request past its budget.
fn too_many() -> Response {
    (
        StatusCode::TOO_MANY_REQUESTS,
        [
            (header::RETRY_AFTER, "10"),
            (header::CACHE_CONTROL, "no-store"),
        ],
        Json(json!({ "error": "too many requests; try again in a few seconds" })),
    )
        .into_response()
}

/// The pages' policy: the dashboard's, plus framing copal's render origin.
/// Without a render origin nothing may be framed and the pages say so.
fn policy(render_origin: Option<&str>) -> String {
    format!(
        "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; \
         img-src 'self' data:; frame-src {}; base-uri 'none'; form-action 'none'; \
         frame-ancestors 'none'",
        render_origin.unwrap_or("'none'")
    )
}

/// The relics routes, which need the server's state for the render origin and
/// the public lookup.
pub(super) fn routes() -> Router<Arc<HttpState>> {
    Router::new()
        .route("/relics", get(gallery))
        // The gallery links its assets relative to itself.
        .route(
            "/relics/",
            get(|| async { Redirect::temporary("../relics") }),
        )
        .route("/relic/{slug}", get(viewer))
        .route("/relic/{slug}/public", get(public_relic))
        .route("/relic/{slug}/export", get(export))
}

fn render_origin(state: &HttpState) -> Option<&str> {
    state.copal.as_deref().and_then(|c| c.render_origin())
}

fn page(state: &HttpState, body: &'static str) -> Response {
    (
        [
            (header::CONTENT_TYPE, "text/html; charset=utf-8".to_string()),
            (
                header::CONTENT_SECURITY_POLICY,
                policy(render_origin(state)),
            ),
            (header::X_CONTENT_TYPE_OPTIONS, "nosniff".to_string()),
            (header::REFERRER_POLICY, "no-referrer".to_string()),
            (header::CACHE_CONTROL, "no-cache".to_string()),
        ],
        body,
    )
        .into_response()
}

async fn gallery(State(state): State<Arc<HttpState>>) -> Response {
    page(&state, GALLERY)
}

async fn viewer(State(state): State<Arc<HttpState>>, Path(slug): Path<String>) -> Response {
    if !is_valid_slug(&slug) {
        return StatusCode::NOT_FOUND.into_response();
    }
    page(&state, VIEWER)
}

/// What a viewer without a token needs to show a public relic: its title,
/// version, when it changed, and the address that displays it. Never who owns
/// it, which tenant it is in, or that a private relic by this name exists.
async fn public_relic(
    State(state): State<Arc<HttpState>>,
    Path(slug): Path<String>,
    peer: Result<ConnectInfo<SocketAddr>, ExtensionRejection>,
) -> Response {
    if !state.relics.public_reads.allow(&client_of(peer)) {
        return too_many();
    }
    let not_found = || {
        (
            StatusCode::NOT_FOUND,
            [(header::CACHE_CONTROL, "no-store")],
            Json(json!({ "error": "no public relic by that name" })),
        )
            .into_response()
    };
    let Some(archive) = state.copal.as_deref() else {
        return not_found();
    };
    let Some(relic) = public_row(&state, &slug).await else {
        return not_found();
    };
    let Some(current) = relic.current() else {
        return not_found();
    };
    let Some(iframe_src) = archive.public_url(&current.copal_file) else {
        return not_found();
    };
    (
        [
            (header::CACHE_CONTROL, "no-store"),
            (header::X_CONTENT_TYPE_OPTIONS, "nosniff"),
        ],
        Json(json!({
            "slug": relic.slug,
            "title": relic.title,
            "version": current.number,
            "updated_at": relic.updated_at.to_rfc3339(),
            "iframe_src": iframe_src,
        })),
    )
        .into_response()
}

/// The public relic `slug`, read as the owner, or nothing: a private relic and
/// a missing one alike.
pub(super) async fn public_row(state: &HttpState, slug: &str) -> Option<Relic> {
    if !is_valid_slug(slug) {
        return None;
    }
    let relic = match &state.serving {
        // The owner connection is every identity's connection too, signed in
        // as whichever identity's call ran last; this read is the owner's,
        // under the lock every call takes.
        Serving::Shared(_) => {
            let _guard = state.auth.lock().await;
            state.store.signin_root().await.ok()?;
            store_relic::get_public(&state.store, slug).await.ok()??
        }
        // The owner connection is no one else's: it stays in the owner view,
        // and a read on it waits on nothing.
        Serving::PerIdentity { .. } => store_relic::get_public(&state.store, slug).await.ok()??,
    };
    (relic.visibility == Visibility::Public).then_some(relic)
}

#[derive(Deserialize)]
struct ExportQuery {
    format: Option<String>,
}

/// What an export can be.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Format {
    Html,
    Markdown,
    Pdf,
}

impl Format {
    fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "html" => Some(Format::Html),
            "md" | "markdown" => Some(Format::Markdown),
            "pdf" => Some(Format::Pdf),
            _ => None,
        }
    }

    fn extension(self) -> &'static str {
        match self {
            Format::Html => "html",
            Format::Markdown => "md",
            Format::Pdf => "pdf",
        }
    }

    fn content_type(self) -> &'static str {
        match self {
            Format::Html => "text/html; charset=utf-8",
            Format::Markdown => "text/markdown; charset=utf-8",
            Format::Pdf => "application/pdf",
        }
    }
}

/// An error an export answers with, as the viewer reads it.
fn problem(status: StatusCode, message: &str) -> Response {
    (
        status,
        [(header::CACHE_CONTROL, "no-store")],
        Json(json!({ "error": message })),
    )
        .into_response()
}

fn no_relic() -> Response {
    problem(
        StatusCode::NOT_FOUND,
        "no relic by that name that you can see",
    )
}

/// The current version of a relic as a file: see the module docs.
async fn export(
    State(state): State<Arc<HttpState>>,
    Path(slug): Path<String>,
    Query(query): Query<ExportQuery>,
    headers: HeaderMap,
    peer: Result<ConnectInfo<SocketAddr>, ExtensionRejection>,
) -> Response {
    let Some(format) = Format::parse(query.format.as_deref().unwrap_or("html")) else {
        return problem(StatusCode::BAD_REQUEST, "format is one of html, md or pdf");
    };
    // A print is the costly export, so it is the budgeted one, before any
    // other work is done for it.
    if format == Format::Pdf && !state.relics.exports.allow(&client_of(peer)) {
        return too_many();
    }
    if format == Format::Pdf && state.pdf.is_none() {
        return problem(
            StatusCode::NOT_IMPLEMENTED,
            "this server has no PDF renderer (--pdf-renderer); open the relic alone and \
             print it from the browser to save a PDF",
        );
    }
    let Some(archive) = state.copal.clone() else {
        return no_relic();
    };
    let relic = match headers.get(header::AUTHORIZATION) {
        Some(value) => {
            let Ok(identity) = state.verifier.verify_header(value.to_str().ok()) else {
                return problem(StatusCode::UNAUTHORIZED, "the token was refused");
            };
            match signed_in_row(&state, &identity, &slug).await {
                // A relic this identity cannot see may still be public.
                Ok(None) => public_row(&state, &slug).await,
                Ok(found) => found,
                Err(response) => return response,
            }
        }
        None => public_row(&state, &slug).await,
    };
    let Some(relic) = relic else {
        return no_relic();
    };
    // A version already printed is answered from the copy kept, without a
    // read from copal or a second print.
    if format == Format::Pdf {
        if let Some(current) = relic.current() {
            let key = (relic.slug.clone(), current.number);
            if let Some(pdf) = state.relics.pdfs.lock().await.get(&key) {
                return file(format, &relic.slug, current.number, pdf.as_ref().clone());
            }
        }
    }
    // The bytes are read and converted with no connection held.
    let source = match crate::server::read_relic(&archive, &relic).await {
        Ok(source) => source,
        Err(e) => {
            eprintln!("antumbra-mcp: export of relic {slug}: {e}");
            return problem(
                StatusCode::BAD_GATEWAY,
                "could not read the relic from copal",
            );
        }
    };
    let body = match format {
        Format::Html => source.html,
        Format::Markdown => html_to_markdown(&String::from_utf8_lossy(&source.html)).into_bytes(),
        Format::Pdf => {
            let Some(renderer) = state.pdf.as_deref() else {
                return no_relic();
            };
            match renderer.render(source.html).await {
                Ok(pdf) => {
                    if pdf.len() <= MAX_CACHED_PDF_BYTES {
                        state
                            .relics
                            .pdfs
                            .lock()
                            .await
                            .insert((source.slug.clone(), source.version), Arc::new(pdf.clone()));
                    }
                    pdf
                }
                Err(e) => {
                    eprintln!("antumbra-mcp: PDF of relic {slug}: {e}");
                    return problem(StatusCode::BAD_GATEWAY, &format!("could not print it: {e}"));
                }
            }
        }
    };
    file(format, &source.slug, source.version, body)
}

/// `body` as the file an export hands over: named for the relic and version,
/// never a page this origin runs.
fn file(format: Format, slug: &str, version: u32, body: Vec<u8>) -> Response {
    (
        [
            (header::CONTENT_TYPE, format.content_type().to_string()),
            (
                header::CONTENT_DISPOSITION,
                // The slug is letters, digits and hyphens: safe as written.
                format!(
                    "attachment; filename=\"{slug}-v{version}.{}\"",
                    format.extension()
                ),
            ),
            // A file to save, never a page this origin runs.
            (
                header::CONTENT_SECURITY_POLICY,
                "sandbox; default-src 'none'".to_string(),
            ),
            (header::X_CONTENT_TYPE_OPTIONS, "nosniff".to_string()),
            (header::REFERRER_POLICY, "no-referrer".to_string()),
            (header::CACHE_CONTROL, "no-store".to_string()),
        ],
        body,
    )
        .into_response()
}

/// The relic `slug` as `identity` may see it: `None` when it cannot, an error
/// response when the identity's server could not be opened.
async fn signed_in_row(
    state: &HttpState,
    identity: &Identity,
    slug: &str,
) -> Result<Option<Relic>, Response> {
    let mcp = state.rest_server_for(identity).await.map_err(|e| {
        eprintln!(
            "antumbra-mcp: relic export init failed for {}/{}: {e}",
            identity.tenant, identity.user
        );
        problem(
            StatusCode::INTERNAL_SERVER_ERROR,
            "could not open your workspace",
        )
    })?;
    let _guard = state.bind(identity).await.map_err(|e| {
        eprintln!(
            "antumbra-mcp: relic export signin failed for {}: {e}",
            identity.tenant
        );
        problem(
            StatusCode::INTERNAL_SERVER_ERROR,
            "could not open your workspace",
        )
    })?;
    Ok(mcp.relic_to_export(slug).await.ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pages_may_frame_the_render_origin_and_nothing_else() {
        let framed = policy(Some("http://render.example:8083"));
        assert!(
            framed.contains("frame-src http://render.example:8083;"),
            "{framed}"
        );
        assert!(framed.contains("frame-ancestors 'none'"));
        assert!(framed.contains("script-src 'self';"));
        assert!(policy(None).contains("frame-src 'none';"));
    }

    /// The pages hold nothing the policy would refuse, and load only the
    /// relics script and the dashboard's style, by paths that resolve from
    /// where each page is served.
    #[test]
    fn the_pages_have_nothing_the_policy_would_refuse() {
        for (name, page, prefix) in [
            ("relics.html", GALLERY, "dashboard/"),
            ("relic.html", VIEWER, "../dashboard/"),
        ] {
            assert!(!page.contains("<script>"), "{name}: inline script");
            assert!(!page.contains("<style"), "{name}: inline style element");
            assert!(!page.contains("style=\""), "{name}: style attribute");
            assert!(
                !page.contains("onclick") && !page.contains("onload"),
                "{name}"
            );
            assert!(
                page.contains(&format!(r#"src="{prefix}relics.js""#)),
                "{name}"
            );
            assert!(
                page.contains(&format!(r#"href="{prefix}app.css""#)),
                "{name}"
            );
            assert!(
                page.contains(&format!(r#"href="{prefix}relics.css""#)),
                "{name}"
            );
        }
    }
}
