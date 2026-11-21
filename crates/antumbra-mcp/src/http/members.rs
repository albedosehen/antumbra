//! A workspace's members: who is in it, as what, and the changes its owners
//! and admins make ([`antumbra_core::membership`]).
//!
//! `GET /members` is any member's to read. `POST /members/role`,
//! `/members/remove` and `/members/readmit` are decided by the caller's role
//! in the workspace, which the server reads from the store on each request: a
//! role is not a claim in a token, so it changes without one being minted.
//! None is an MCP tool, so no agent is offered them.
//!
//! The membership table is the owner's alone, so each of these runs on the
//! owner's connection after the caller's token is verified and their identity
//! provisioned, as a tool call's would be.

use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, Request, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::json;

use antumbra_core::membership::Role;
use antumbra_core::{TenantId, UserId};
use antumbra_store::repo::{principal, profile};

use super::{bad_request, init_failed, internal_error, unauthorized, HttpState};
use crate::auth::Access;

/// The largest body read here; a change is a user id and a word.
const MAX_BODY: usize = 4 * 1024;

pub(super) fn routes() -> Router<Arc<HttpState>> {
    Router::new()
        .route("/members", get(list))
        .route("/members/role", post(set_role))
        .route("/members/remove", post(remove))
        .route("/members/readmit", post(readmit))
}

/// The caller, verified and provisioned in their workspace, or the response
/// that says why not.
///
/// The token is read off the request before anything is awaited: a request's
/// body is not something a task may hold a reference to across a wait.
pub(super) async fn caller(
    state: &HttpState,
    token: Option<String>,
    route: &str,
) -> Result<Access, Response> {
    let access = state.access_of(token.as_deref()).map_err(|e| {
        eprintln!("antumbra-mcp: rejected {route}: {e}");
        unauthorized()
    })?;
    // Provisioned on a first call here as on a first tool call, so the caller
    // has a membership to read.
    state
        .rest_server_for(&access.identity)
        .await
        .map_err(|e| init_failed(&e))?;
    Ok(access)
}

/// The request's `Authorization` header, as it was sent.
pub(super) fn token_of(req: &Request<Body>) -> Option<String> {
    req.headers()
        .get(header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .map(str::to_owned)
}

/// Run `work` on the owner's connection ([`super::console`] does the same).
pub(super) async fn as_owner<T, F, Fut>(state: &HttpState, work: F) -> antumbra_core::Result<T>
where
    F: FnOnce(antumbra_store::Store) -> Fut,
    Fut: std::future::Future<Output = antumbra_core::Result<T>>,
{
    let _guard = state.auth.lock().await;
    state.store.signin_root().await?;
    work(state.store.clone()).await
}

/// A refusal by the rules is the caller's to read (403); anything else is the
/// server's fault and is said on stderr.
pub(super) fn failed(route: &str, e: &antumbra_core::AntumbraError) -> Response {
    if e.is_rejection() {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response();
    }
    eprintln!("antumbra-mcp: {route} failed: {e}");
    internal_error()
}

async fn list(State(state): State<Arc<HttpState>>, req: Request<Body>) -> Response {
    let access = match caller(&state, token_of(&req), "/members").await {
        Ok(access) => access,
        Err(response) => return response,
    };
    let (tenant, user) = (
        TenantId::new(&access.identity.tenant),
        UserId::new(&access.identity.user),
    );
    let runs_the_server = access.operator;
    let read = as_owner(&state, |store| async move {
        let yours = principal::role_of(&store, &tenant, &user, runs_the_server).await?;
        let members = principal::members(&store, &tenant).await?;
        let profiles = profile::list(&store, &tenant).await?;
        // Who was removed is for those who could let them back in.
        let removed: Vec<String> = match yours.is_some_and(Role::admits) {
            true => principal::removed(&store)
                .await?
                .into_iter()
                .filter(|(t, _)| t == &tenant)
                .map(|(_, user)| user.as_str().to_string())
                .collect(),
            false => Vec::new(),
        };
        Ok((yours, members, profiles, removed))
    })
    .await;
    let (yours, members, profiles, removed) = match read {
        Ok(read) => read,
        Err(e) => return failed("/members", &e),
    };
    let members: Vec<_> = members
        .iter()
        .map(|m| {
            let held = profiles.iter().find(|p| p.user == m.user);
            json!({
                "user": m.user.as_str(),
                "role": m.role.as_str(),
                "joined_at": m.joined_at.map(|at| at.to_rfc3339()),
                "display": held.and_then(|p| p.display.clone()),
                "has_image": held.is_some_and(|p| p.avatar.is_some()),
            })
        })
        .collect();
    Json(json!({
        "you": access.identity.user,
        "your_role": yours.map(Role::as_str),
        "members": members,
        "removed": removed,
    }))
    .into_response()
}

#[derive(Deserialize)]
struct Change {
    user: String,
    #[serde(default)]
    role: Option<String>,
}

/// The request's body as a change, with its caller.
async fn change(
    state: &HttpState,
    req: Request<Body>,
    route: &str,
) -> Result<(Access, Change), Response> {
    let access = caller(state, token_of(&req), route).await?;
    let body = axum::body::to_bytes(req.into_body(), MAX_BODY)
        .await
        .map_err(|_| bad_request("request body unreadable or too large"))?;
    let change: Change =
        serde_json::from_slice(&body).map_err(|e| bad_request(&format!("invalid request: {e}")))?;
    if change.user.trim().is_empty() {
        return Err(bad_request("`user` names the member"));
    }
    Ok((access, change))
}

/// A change to the workspace's members is said where the server's logs go.
fn said(route: &str, access: &Access, change: &Change) {
    eprintln!(
        "antumbra-mcp: {route} by {}/{}: {} {}",
        access.identity.tenant,
        access.identity.user,
        change.user,
        change.role.as_deref().unwrap_or("")
    );
}

async fn set_role(State(state): State<Arc<HttpState>>, req: Request<Body>) -> Response {
    let (access, change) = match change(&state, req, "/members/role").await {
        Ok(read) => read,
        Err(response) => return response,
    };
    let role: Role = match change.role.as_deref().unwrap_or_default().parse() {
        Ok(role) => role,
        Err(e) => return bad_request(&e.to_string()),
    };
    said("/members/role", &access, &change);
    let (tenant, actor, target) = (
        TenantId::new(&access.identity.tenant),
        UserId::new(&access.identity.user),
        UserId::new(change.user.trim()),
    );
    let runs_the_server = access.operator;
    match as_owner(&state, |store| async move {
        // The caller's own role is settled first, for a workspace from before
        // roles whose owner this call is the first to name.
        principal::role_of(&store, &tenant, &actor, runs_the_server).await?;
        principal::set_role(&store, &tenant, &actor, &target, role).await
    })
    .await
    {
        Ok(()) => {
            Json(json!({ "user": change.user.trim(), "role": role.as_str() })).into_response()
        }
        Err(e) => failed("/members/role", &e),
    }
}

async fn remove(State(state): State<Arc<HttpState>>, req: Request<Body>) -> Response {
    let (access, change) = match change(&state, req, "/members/remove").await {
        Ok(read) => read,
        Err(response) => return response,
    };
    said("/members/remove", &access, &change);
    let (tenant, actor, target) = (
        TenantId::new(&access.identity.tenant),
        UserId::new(&access.identity.user),
        UserId::new(change.user.trim()),
    );
    let runs_the_server = access.operator;
    let gone = (tenant.as_str().to_string(), target.as_str().to_string());
    match as_owner(&state, |store| async move {
        principal::role_of(&store, &tenant, &actor, runs_the_server).await?;
        principal::remove(&store, &tenant, &actor, &target).await
    })
    .await
    {
        Ok(()) => {
            // Refused from their next request on this server; the others
            // read it within a refresh.
            state.forget_member(gone, true);
            Json(json!({ "user": change.user.trim(), "removed": true })).into_response()
        }
        Err(e) => failed("/members/remove", &e),
    }
}

async fn readmit(State(state): State<Arc<HttpState>>, req: Request<Body>) -> Response {
    let (access, change) = match change(&state, req, "/members/readmit").await {
        Ok(read) => read,
        Err(response) => return response,
    };
    said("/members/readmit", &access, &change);
    let (tenant, actor, target) = (
        TenantId::new(&access.identity.tenant),
        UserId::new(&access.identity.user),
        UserId::new(change.user.trim()),
    );
    let runs_the_server = access.operator;
    let back = (tenant.as_str().to_string(), target.as_str().to_string());
    match as_owner(&state, |store| async move {
        principal::role_of(&store, &tenant, &actor, runs_the_server).await?;
        principal::readmit(&store, &tenant, &actor, &target).await
    })
    .await
    {
        Ok(()) => {
            state.forget_member(back, false);
            Json(json!({ "user": change.user.trim(), "role": Role::Member.as_str() }))
                .into_response()
        }
        Err(e) => failed("/members/readmit", &e),
    }
}
