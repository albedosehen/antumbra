//! The operator console's routes, for `antumbra tui`: `GET /console/snapshot`
//! reads what the console shows, and `POST /console/action` carries out an
//! operator's action ([`antumbra_core::console`]). Both take the same bearer
//! token as `/mcp`. Neither is an MCP tool, so no agent is offered them.
//!
//! The snapshot's workspace part is read through the caller's scoped
//! connection, so the engine's permissions keep it to that workspace. Its
//! substrate part (shadows, boundaries, the loop, evaluations, the router) and
//! every action need the owner's connection and an operator's token
//! ([`crate::auth::Access::operator`]), checked on each request.

use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, Request, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::json;

use antumbra_core::console::Action;
use antumbra_store::repo::console;

use super::{bad_request, init_failed, internal_error, unauthorized, HttpState};
use crate::auth::Access;

/// The largest action body read; an action is a name and one id.
const MAX_ACTION_BODY: usize = 16 * 1024;

pub(super) fn routes() -> Router<Arc<HttpState>> {
    Router::new()
        .route("/console/snapshot", get(snapshot))
        .route("/console/action", post(action))
}

/// The request's verified token, or `None` (said on stderr) for a 401.
fn verified(state: &HttpState, req: &Request<Body>, route: &str) -> Option<Access> {
    let header_val = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok());
    state
        .verifier
        .verify_header_access(header_val)
        .map_err(|e| eprintln!("antumbra-mcp: rejected {route}: {e}"))
        .ok()
}

async fn snapshot(State(state): State<Arc<HttpState>>, req: Request<Body>) -> Response {
    let Some(access) = verified(&state, &req, "/console/snapshot") else {
        return unauthorized();
    };
    let identity = &access.identity;
    // The identity's server, provisioned on its first call like `/mcp/call`'s,
    // so its scoped connection exists and can sign in.
    let mcp = match state.rest_server_for(identity).await {
        Ok(m) => m,
        Err(e) => {
            eprintln!(
                "antumbra-mcp: /console/snapshot init failed for {}/{}: {e}",
                identity.tenant, identity.user
            );
            return init_failed(&e);
        }
    };
    let workspace = {
        let _guard = match state.bind(identity).await {
            Ok(g) => g,
            Err(e) => {
                eprintln!("antumbra-mcp: /console/snapshot signin failed: {e}");
                return internal_error();
            }
        };
        console::workspace(mcp.scoped_store()).await
    };
    let mut snapshot = match workspace {
        Ok(s) => s,
        Err(e) => {
            eprintln!("antumbra-mcp: /console/snapshot read failed: {e}");
            return internal_error();
        }
    };
    if access.operator {
        let read = as_owner(&state, |store| async move {
            console::substrate(&store, &mut snapshot).await?;
            Ok(snapshot)
        })
        .await;
        snapshot = match read {
            Ok(s) => s,
            Err(e) => {
                eprintln!("antumbra-mcp: /console/snapshot substrate read failed: {e}");
                return internal_error();
            }
        };
    }
    let body = match serde_json::to_vec(&snapshot) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("antumbra-mcp: /console/snapshot could not be written: {e}");
            return internal_error();
        }
    };
    // The console asks every few seconds; an unchanged snapshot, which is most
    // of them, goes back as a 304 with no body.
    let tag = etag(&body);
    let seen = req
        .headers()
        .get(header::IF_NONE_MATCH)
        .and_then(|h| h.to_str().ok());
    if seen == Some(tag.as_str()) {
        return (StatusCode::NOT_MODIFIED, [(header::ETAG, tag)]).into_response();
    }
    (
        [
            (header::CONTENT_TYPE, "application/json".to_string()),
            (header::ETAG, tag),
        ],
        body,
    )
        .into_response()
}

/// A snapshot's entity tag: a hash of its body. It names one body within one
/// server process; a restarted server's tags differ, which costs one full read.
fn etag(body: &[u8]) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    body.hash(&mut hasher);
    format!("\"{:016x}\"", hasher.finish())
}

async fn action(State(state): State<Arc<HttpState>>, req: Request<Body>) -> Response {
    let Some(access) = verified(&state, &req, "/console/action") else {
        return unauthorized();
    };
    if !access.operator {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({
                "error": "this token is not an operator's: mint one with --mint-token --operator"
            })),
        )
            .into_response();
    }
    let body = match axum::body::to_bytes(req.into_body(), MAX_ACTION_BODY).await {
        Ok(b) => b,
        Err(_) => return bad_request("request body unreadable or too large"),
    };
    let action: Action = match serde_json::from_slice(&body) {
        Ok(a) => a,
        Err(e) => return bad_request(&format!("invalid action: {e}")),
    };
    let identity = &access.identity;
    // An operator's change to the substrate is said where the server's logs go.
    eprintln!(
        "antumbra-mcp: console action by {}/{}: {action:?}",
        identity.tenant, identity.user
    );
    match as_owner(&state, |store| async move {
        Ok(console::apply(&store, &action).await?)
    })
    .await
    {
        Ok(true) => Json(json!({ "applied": true })).into_response(),
        Ok(false) => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "no such shadow, expert, boundary or loop" })),
        )
            .into_response(),
        Err(e) => {
            eprintln!("antumbra-mcp: console action failed: {e}");
            internal_error()
        }
    }
}

/// Run `work` on the owner's connection. Embedded, that connection is the one
/// every request signs in as its caller, so it is signed in as root again
/// under `auth`, which every such sign-in holds, as provisioning does.
async fn as_owner<T, F, Fut>(state: &HttpState, work: F) -> anyhow::Result<T>
where
    F: FnOnce(antumbra_store::Store) -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<T>>,
{
    let _guard = state.auth.lock().await;
    state.store.signin_root().await?;
    work(state.store.clone()).await
}

#[cfg(test)]
mod tests;
