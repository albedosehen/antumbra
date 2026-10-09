//! The read-only web dashboard: a browser surface over the same tools an
//! agent calls.
//!
//! The server hands out the page, its script and its style, and nothing else.
//! None of them carries data, so they need no token. Everything the page shows
//! it fetches through `POST /mcp/call` with the bearer token its user pastes in,
//! so it sees what that token's `(tenant, user)` sees and no more: the engine
//! ACL stays the only authority, as it is for any other client.
//!
//! The assets are compiled into the binary, so a deployment has nothing extra
//! to mount and the page always matches the server that serves it.

use axum::http::header;
use axum::response::{IntoResponse, Redirect, Response};
use axum::routing::get;
use axum::Router;

const PAGE: &str = include_str!("dashboard/index.html");
const SCRIPT: &str = include_str!("dashboard/app.js");
const STYLE: &str = include_str!("dashboard/app.css");

/// The page's own script and style, and requests back to this server: nothing
/// inline, nothing from elsewhere. The page renders stored text as text, never
/// as markup; this policy is what keeps a memory holding a `<script>` inert if
/// a later change forgets that.
const POLICY: &str = "default-src 'none'; script-src 'self'; style-src 'self'; \
                      connect-src 'self'; img-src 'self' data:; base-uri 'none'; \
                      form-action 'none'; frame-ancestors 'none'";

/// The dashboard's routes. They take no state, so they merge into any router.
pub(super) fn routes<S: Clone + Send + Sync + 'static>() -> Router<S> {
    Router::new()
        .route("/dashboard", get(page))
        // The page links its assets relative to itself, which only resolves
        // from `/dashboard`. Relative, so it holds behind a path prefix too.
        .route(
            "/dashboard/",
            get(|| async { Redirect::temporary("../dashboard") }),
        )
        .route("/dashboard/app.js", get(script))
        .route("/dashboard/app.css", get(style))
}

async fn page() -> Response {
    asset("text/html; charset=utf-8", PAGE)
}

async fn script() -> Response {
    asset("text/javascript; charset=utf-8", SCRIPT)
}

async fn style() -> Response {
    asset("text/css; charset=utf-8", STYLE)
}

fn asset(content_type: &'static str, body: &'static str) -> Response {
    (
        [
            (header::CONTENT_TYPE, content_type),
            (header::CONTENT_SECURITY_POLICY, POLICY),
            (header::X_CONTENT_TYPE_OPTIONS, "nosniff"),
            (header::REFERRER_POLICY, "no-referrer"),
            // Revalidated each load, so a redeployed server's page is the one
            // a reload shows.
            (header::CACHE_CONTROL, "no-cache"),
        ],
        body,
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt; // oneshot

    async fn get_path(path: &str) -> Response {
        let req = Request::builder().uri(path).body(Body::empty()).unwrap();
        routes::<()>().oneshot(req).await.unwrap()
    }

    #[tokio::test]
    async fn every_asset_is_served_with_its_type_and_the_policy() {
        for (path, kind) in [
            ("/dashboard", "text/html"),
            ("/dashboard/app.js", "text/javascript"),
            ("/dashboard/app.css", "text/css"),
        ] {
            let res = get_path(path).await;
            assert_eq!(res.status(), StatusCode::OK, "{path}");
            let headers = res.headers();
            let content_type = headers[header::CONTENT_TYPE].to_str().unwrap();
            assert!(content_type.starts_with(kind), "{path}: {content_type}");
            assert_eq!(headers[header::CONTENT_SECURITY_POLICY], POLICY, "{path}");
            assert_eq!(headers[header::X_CONTENT_TYPE_OPTIONS], "nosniff");
        }
    }

    /// The page's assets are relative to `/dashboard`, so the trailing-slash
    /// form sends the browser there instead of serving a page whose links
    /// would miss.
    #[tokio::test]
    async fn the_trailing_slash_form_redirects_to_the_page() {
        let res = get_path("/dashboard/").await;
        assert!(res.status().is_redirection());
        assert_eq!(res.headers()[header::LOCATION], "../dashboard");
    }

    /// Nothing inline: the policy allows only the page's own files, so an
    /// inline script or style would be refused by the browser, and a `style`
    /// attribute too. Checked here so the page cannot drift from the policy.
    #[test]
    fn the_page_has_nothing_the_policy_would_refuse() {
        assert!(!PAGE.contains("<script>"), "inline script");
        assert!(!PAGE.contains("<style"), "inline style element");
        assert!(!PAGE.contains("style=\""), "style attribute");
        assert!(!PAGE.contains("onclick") && !PAGE.contains("onload"));
        assert!(PAGE.contains(r#"src="dashboard/app.js""#));
        assert!(PAGE.contains(r#"href="dashboard/app.css""#));
    }

    /// The script writes stored text with `textContent`, never as markup, so a
    /// memory that holds markup is shown, not run.
    #[test]
    fn the_script_never_writes_markup() {
        for sink in [
            "innerHTML",
            "outerHTML",
            "insertAdjacentHTML",
            "document.write",
        ] {
            assert!(!SCRIPT.contains(sink), "{sink}");
        }
        assert!(!SCRIPT.contains("eval("), "eval");
    }
}
