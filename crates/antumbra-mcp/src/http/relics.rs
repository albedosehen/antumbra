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

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Redirect, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde_json::json;

use antumbra_core::relic::is_valid_slug;
use antumbra_core::Visibility;
use antumbra_store::repo::relic as store_relic;

use super::HttpState;

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
    if !is_valid_slug(&slug) {
        return not_found();
    }
    // The shared connection may be signed in as whichever identity the last
    // request was; this read is the owner's, under the lock that every
    // identity switch takes.
    let found = {
        let _guard = state.auth.lock().await;
        if state.store.signin_root().await.is_err() {
            return not_found();
        }
        store_relic::get_public(&state.store, &slug).await
    };
    let Ok(Some(relic)) = found else {
        return not_found();
    };
    let (Some(current), true) = (relic.current(), relic.visibility == Visibility::Public) else {
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
