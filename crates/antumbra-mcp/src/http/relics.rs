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

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Redirect, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::json;

use antumbra_core::relic::{html_to_markdown, is_valid_slug};
use antumbra_core::{Relic, Visibility};
use antumbra_store::repo::relic as store_relic;

use super::HttpState;
use crate::auth::Identity;

const GALLERY: &str = include_str!("dashboard/relics.html");
const VIEWER: &str = include_str!("dashboard/relic.html");

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
async fn public_relic(State(state): State<Arc<HttpState>>, Path(slug): Path<String>) -> Response {
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
async fn public_row(state: &HttpState, slug: &str) -> Option<Relic> {
    if !is_valid_slug(slug) {
        return None;
    }
    // The shared connection may be signed in as whichever identity the last
    // request was; this read is the owner's, under the lock that every
    // identity switch takes.
    let _guard = state.auth.lock().await;
    state.store.signin_root().await.ok()?;
    let relic = store_relic::get_public(&state.store, slug).await.ok()??;
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
) -> Response {
    let Some(format) = Format::parse(query.format.as_deref().unwrap_or("html")) else {
        return problem(StatusCode::BAD_REQUEST, "format is one of html, md or pdf");
    };
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
                Ok(pdf) => pdf,
                Err(e) => {
                    eprintln!("antumbra-mcp: PDF of relic {slug}: {e}");
                    return problem(StatusCode::BAD_GATEWAY, &format!("could not print it: {e}"));
                }
            }
        }
    };
    (
        [
            (header::CONTENT_TYPE, format.content_type().to_string()),
            (
                header::CONTENT_DISPOSITION,
                // The slug is letters, digits and hyphens: safe as written.
                format!(
                    "attachment; filename=\"{}-v{}.{}\"",
                    source.slug,
                    source.version,
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
        let framed = policy(Some("http://10.0.0.130:8083"));
        assert!(
            framed.contains("frame-src http://10.0.0.130:8083;"),
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
