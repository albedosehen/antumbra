//! Naming your own machines.
//!
//! `POST /devices/name {host, name}` gives one of the caller's machines a
//! name for people to read, or takes it away when `name` is blank. The host
//! stays what the machine calls itself: handoffs are addressed to it and
//! writes are stamped with it. A caller names only their own machines.
//!
//! Not an MCP tool, so no agent is offered it: a name is its user's to give.

use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::Request;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::json;

use antumbra_store::repo::device;

use super::members::{as_owner, caller, failed, token_of};
use super::{bad_request, HttpState};

/// The largest body read here: a host and a name.
const MAX_BODY: usize = 4 * 1024;

pub(super) fn routes() -> Router<Arc<HttpState>> {
    Router::new().route("/devices/name", post(name))
}

#[derive(Deserialize)]
struct Naming {
    host: String,
    #[serde(default)]
    name: Option<String>,
}

async fn name(State(state): State<Arc<HttpState>>, req: Request<Body>) -> Response {
    let access = match caller(&state, token_of(&req), "/devices/name").await {
        Ok(access) => access,
        Err(refused) => return refused,
    };
    let Ok(body) = axum::body::to_bytes(req.into_body(), MAX_BODY).await else {
        return bad_request("request body unreadable or too large");
    };
    let naming: Naming = match serde_json::from_slice(&body) {
        Ok(naming) => naming,
        Err(e) => return bad_request(&format!("invalid request: {e}")),
    };
    let Some(host) = crate::server::machine_name(&naming.host) else {
        return bad_request("`host` names the machine");
    };
    let named = match antumbra_core::device::name_of(naming.name.as_deref().unwrap_or("")) {
        Ok(named) => named,
        Err(e) => return bad_request(&e.to_string()),
    };
    let tenant = antumbra_core::TenantId::new(access.identity.tenant.clone());
    let user = antumbra_core::UserId::new(access.identity.user.clone());
    let wrote = as_owner(&state, |store| {
        let (host, named) = (host.clone(), named.clone());
        async move { device::rename(&store, &tenant, &user, &host, named.as_deref()).await }
    })
    .await;
    match wrote {
        Ok(true) => Json(json!({ "host": host, "name": named })).into_response(),
        Ok(false) => bad_request(&format!("you have no device registered as {host}")),
        Err(e) => failed("/devices/name", &e),
    }
}
