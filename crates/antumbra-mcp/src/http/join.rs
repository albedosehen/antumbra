//! Joining a workspace someone else hosts ([`antumbra_core::join`]).
//!
//! For the workspace's admins and owners, with their token:
//! `POST /join/codes` makes a join code and shows it once, `GET /join/codes`
//! lists the workspace's, `POST /join/codes/revoke` ends one, `GET
//! /join/requests` lists who has asked, and `POST /join/decide` approves or
//! refuses a request.
//!
//! For someone who is not a member yet, with **no token**: `POST
//! /join/request` asks with a code, and `POST /join/collect` asks how the
//! request stands and, once it is approved, is answered with a token, once.
//! These two are the only routes on this server that act for someone it
//! cannot name, so they answer the same for every code that will not do, read
//! a small body, and are counted together against a rate.
//!
//! A server that verifies RS256 holds no key to sign with: it takes no
//! requests, and says so.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use chrono::Utc;
use serde::Deserialize;
use serde_json::json;

use antumbra_core::join::{self, Answer, JoinCode, JoinRequest, CODE_DAYS, CODE_USES};
use antumbra_core::profile;
use antumbra_core::{TenantId, UserId};
use antumbra_store::repo::{join as store_join, principal};

use super::members::{as_owner, caller, failed, token_of};
use super::{bad_request, HttpState};

/// The largest body read here.
const MAX_BODY: usize = 8 * 1024;

/// How many requests the two public routes take in a minute, from everyone
/// together. Someone joining makes two or three; this leaves room for a team
/// arriving at once and none for guessing codes.
pub(super) const PUBLIC_PER_MINUTE: u32 = 60;

/// How long a new member's token lasts.
const TOKEN_TTL: Duration = Duration::from_secs(365 * 24 * 60 * 60);

/// The most a code may be made to last, and to admit.
const MAX_CODE_DAYS: i64 = 90;
const MAX_CODE_USES: u32 = 100;

pub(super) fn routes() -> Router<Arc<HttpState>> {
    Router::new()
        .route("/join/codes", get(codes).post(make_code))
        .route("/join/codes/revoke", post(revoke_code))
        .route("/join/requests", get(requests))
        .route("/join/decide", post(decide))
        .route("/join/request", post(request))
        .route("/join/collect", post(collect))
}

async fn body<T: serde::de::DeserializeOwned>(req: Request<Body>) -> Result<T, Response> {
    let bytes = axum::body::to_bytes(req.into_body(), MAX_BODY)
        .await
        .map_err(|_| bad_request("request body unreadable or too large"))?;
    serde_json::from_slice(&bytes).map_err(|e| bad_request(&format!("invalid request: {e}")))
}

/// The caller's workspace and user, when their role lets people in.
async fn admin(
    state: &HttpState,
    token: Option<String>,
    route: &str,
) -> Result<(TenantId, UserId), Response> {
    let access = caller(state, token, route).await?;
    let (tenant, user) = (
        TenantId::new(&access.identity.tenant),
        UserId::new(&access.identity.user),
    );
    let runs_the_server = access.operator;
    let (t, u) = (tenant.clone(), user.clone());
    let role = as_owner(state, |store| async move {
        principal::role_of(&store, &t, &u, runs_the_server).await
    })
    .await
    .map_err(|e| failed(route, &e))?;
    if !role.is_some_and(|role| role.admits()) {
        return Err((
            StatusCode::FORBIDDEN,
            Json(json!({ "error": "only an admin or an owner of this workspace lets people in" })),
        )
            .into_response());
    }
    Ok((tenant, user))
}

fn code_view(code: &JoinCode) -> serde_json::Value {
    json!({
        "key": code.key,
        "label": code.label,
        "created_by": code.created_by.as_str(),
        "created_at": code.created_at.to_rfc3339(),
        "expires_at": code.expires_at.to_rfc3339(),
        "uses_left": code.uses_left,
        "needs_approval": code.needs_approval,
        "revoked": code.revoked,
        "admits": code.admits(Utc::now()),
    })
}

/// A request as an admin sees it. The digest of its secret is not theirs.
fn request_view(request: &JoinRequest) -> serde_json::Value {
    let now = Utc::now();
    json!({
        "key": request.key,
        "handle": request.handle,
        "user": request.user().as_str(),
        "display": request.display,
        "note": request.note,
        "host": request.host,
        "standing": match request.standing {
            join::Standing::Pending if !request.waits(now) => "expired",
            join::Standing::Approved if request.lapsed(now) => "expired",
            other => other.as_str(),
        },
        "created_at": request.created_at.to_rfc3339(),
        "expires_at": request.expires_at.to_rfc3339(),
        "decided_at": request.decided_at.map(|at| at.to_rfc3339()),
        "decided_by": request.decided_by.as_ref().map(|by| by.as_str().to_string()),
        "reason": request.reason,
        // An address a control plane verified they hold, when one did.
        "email": request.email,
    })
}

#[derive(Deserialize)]
struct NewCode {
    #[serde(default)]
    label: Option<String>,
    #[serde(default)]
    days: Option<i64>,
    #[serde(default)]
    uses: Option<u32>,
    /// Whether a request made with it waits for an admin. True unless said.
    #[serde(default)]
    needs_approval: Option<bool>,
}

async fn make_code(State(state): State<Arc<HttpState>>, req: Request<Body>) -> Response {
    let (tenant, user) = match admin(&state, token_of(&req), "/join/codes").await {
        Ok(who) => who,
        Err(response) => return response,
    };
    if state.issuer.is_none() && state.control_plane.is_none() {
        return bad_request(
            "this server verifies tokens with a public key and cannot sign one for a new member",
        );
    }
    let new: NewCode = match body(req).await {
        Ok(new) => new,
        Err(response) => return response,
    };
    let label = match profile::display_of(new.label.as_deref().unwrap_or_default()) {
        Ok(label) => label,
        Err(e) => return bad_request(&e.to_string()),
    };
    let now = Utc::now();
    // The code is this once in the clear, in the answer to its maker.
    let shown = format!("aj1_{}", uuid::Uuid::new_v4().simple());
    let code = JoinCode {
        key: join::digest(&shown),
        tenant,
        label,
        created_by: user.clone(),
        created_at: now,
        expires_at: now
            + chrono::Duration::days(new.days.unwrap_or(CODE_DAYS).clamp(1, MAX_CODE_DAYS)),
        uses_left: new.uses.unwrap_or(CODE_USES).clamp(1, MAX_CODE_USES),
        needs_approval: new.needs_approval.unwrap_or(true),
        revoked: false,
    };
    eprintln!(
        "antumbra-mcp: join code made by {}/{}: {} use(s), until {}",
        code.tenant.as_str(),
        user.as_str(),
        code.uses_left,
        code.expires_at.to_rfc3339()
    );
    let kept = code.clone();
    match as_owner(&state, |store| async move {
        store_join::make_code(&store, &kept).await
    })
    .await
    {
        Ok(()) => {
            let mut view = code_view(&code);
            view["code"] = json!(shown);
            Json(view).into_response()
        }
        Err(e) => failed("/join/codes", &e),
    }
}

async fn codes(State(state): State<Arc<HttpState>>, req: Request<Body>) -> Response {
    let (tenant, _) = match admin(&state, token_of(&req), "/join/codes").await {
        Ok(who) => who,
        Err(response) => return response,
    };
    match as_owner(&state, |store| async move {
        store_join::codes(&store, &tenant).await
    })
    .await
    {
        Ok(all) => Json(json!({
            "codes": all.iter().map(code_view).collect::<Vec<_>>(),
            "can_sign": state.issuer.is_some() || state.control_plane.is_some(),
            // Where someone with a code asks, when it is not here.
            "control_plane": state.control_plane,
        }))
        .into_response(),
        Err(e) => failed("/join/codes", &e),
    }
}

#[derive(Deserialize)]
struct Named {
    key: String,
}

async fn revoke_code(State(state): State<Arc<HttpState>>, req: Request<Body>) -> Response {
    let (tenant, _) = match admin(&state, token_of(&req), "/join/codes/revoke").await {
        Ok(who) => who,
        Err(response) => return response,
    };
    let named: Named = match body(req).await {
        Ok(named) => named,
        Err(response) => return response,
    };
    match as_owner(&state, |store| async move {
        store_join::revoke_code(&store, &tenant, named.key.trim()).await
    })
    .await
    {
        Ok(true) => Json(json!({ "revoked": true })).into_response(),
        Ok(false) => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "no such code in this workspace" })),
        )
            .into_response(),
        Err(e) => failed("/join/codes/revoke", &e),
    }
}

async fn requests(State(state): State<Arc<HttpState>>, req: Request<Body>) -> Response {
    let (tenant, _) = match admin(&state, token_of(&req), "/join/requests").await {
        Ok(who) => who,
        Err(response) => return response,
    };
    match as_owner(&state, |store| async move {
        store_join::requests(&store, &tenant).await
    })
    .await
    {
        Ok(all) => {
            let now = Utc::now();
            Json(json!({
                "waiting": all.iter().filter(|r| r.waits(now)).count(),
                "requests": all.iter().map(request_view).collect::<Vec<_>>(),
            }))
            .into_response()
        }
        Err(e) => failed("/join/requests", &e),
    }
}

#[derive(Deserialize)]
struct Decision {
    key: String,
    approve: bool,
    #[serde(default)]
    reason: Option<String>,
}

async fn decide(State(state): State<Arc<HttpState>>, req: Request<Body>) -> Response {
    let (tenant, user) = match admin(&state, token_of(&req), "/join/decide").await {
        Ok(who) => who,
        Err(response) => return response,
    };
    let decision: Decision = match body(req).await {
        Ok(decision) => decision,
        Err(response) => return response,
    };
    let reason = match join::note_of(decision.reason.as_deref().unwrap_or_default()) {
        Ok(reason) => reason,
        Err(e) => return bad_request(&e.to_string()),
    };
    eprintln!(
        "antumbra-mcp: join request {} {} by {}/{}",
        decision.key.trim(),
        if decision.approve {
            "approved"
        } else {
            "refused"
        },
        tenant.as_str(),
        user.as_str()
    );
    match as_owner(&state, |store| async move {
        store_join::decide(
            &store,
            &tenant,
            &user,
            decision.key.trim(),
            decision.approve,
            reason,
            Utc::now(),
        )
        .await
    })
    .await
    {
        Ok(decided) => Json(request_view(&decided)).into_response(),
        Err(e) => failed("/join/decide", &e),
    }
}

/// One more request against the minute's count; when the minute is full,
/// the answer that says to come back. The count is everyone's together: these routes cannot tell
/// their callers apart.
fn over_the_rate(state: &HttpState) -> Option<Response> {
    // A poisoned lock holds a time and a count, neither written in two steps.
    let mut minute = state
        .join_minute
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if minute.0.elapsed() >= Duration::from_secs(60) {
        *minute = (Instant::now(), 0);
    }
    if minute.1 >= PUBLIC_PER_MINUTE {
        return Some(
            (
                StatusCode::TOO_MANY_REQUESTS,
                Json(
                    json!({ "error": "too many requests to join just now; try again in a minute" }),
                ),
            )
                .into_response(),
        );
    }
    minute.1 += 1;
    None
}

/// The answer of a server that signs nothing: where to ask instead, when a
/// control plane signs for it, in words and as a field a client can follow.
fn elsewhere(state: &HttpState) -> Response {
    match &state.control_plane {
        Some(url) => (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": format!("this workspace's tokens are signed at {url}: ask to join there, with your email address"),
                "control_plane": url,
            })),
        )
            .into_response(),
        None => bad_request(
            "this server verifies tokens with a public key and takes no requests to join",
        ),
    }
}

#[derive(Deserialize)]
struct Asked {
    code: String,
    handle: String,
    #[serde(default)]
    display: Option<String>,
    #[serde(default)]
    note: Option<String>,
    #[serde(default)]
    host: Option<String>,
    /// The SHA-256, in hex, of the secret the token will be collected with.
    /// The secret itself is not sent until then.
    secret_digest: String,
}

async fn request(State(state): State<Arc<HttpState>>, req: Request<Body>) -> Response {
    if let Some(response) = over_the_rate(&state) {
        return response;
    }
    if state.issuer.is_none() {
        return elsewhere(&state);
    }
    let asked: Asked = match body(req).await {
        Ok(asked) => asked,
        Err(response) => return response,
    };
    let digest = asked.secret_digest.trim().to_ascii_lowercase();
    if digest.len() != 64 || !digest.chars().all(|c| c.is_ascii_hexdigit()) {
        return bad_request("`secret_digest` is the SHA-256 of your secret, in hex");
    }
    let (display, note) = match (
        profile::display_of(asked.display.as_deref().unwrap_or_default()),
        join::note_of(asked.note.as_deref().unwrap_or_default()),
    ) {
        (Ok(display), Ok(note)) => (display, note),
        (Err(e), _) | (_, Err(e)) => return bad_request(&e.to_string()),
    };
    let host = asked.host.as_deref().and_then(crate::server::machine_name);
    let id = uuid::Uuid::new_v4().simple().to_string();
    let made = as_owner(&state, |store| async move {
        store_join::ask(
            &store,
            store_join::Asking {
                id: &id,
                code: &asked.code,
                handle: &asked.handle,
                display,
                note,
                host,
                secret_digest: &digest,
                email: None,
            },
            Utc::now(),
        )
        .await
    })
    .await;
    match made {
        Ok(request) => {
            eprintln!(
                "antumbra-mcp: join request {} for {} as {}",
                request.key,
                request.tenant.as_str(),
                request.handle
            );
            Json(json!({
                "request": request.key,
                "handle": request.handle,
                "standing": request.standing.as_str(),
                "expires_at": request.expires_at.to_rfc3339(),
            }))
            .into_response()
        }
        Err(e) => failed("/join/request", &e),
    }
}

#[derive(Deserialize)]
struct Collecting {
    request: String,
    secret: String,
}

async fn collect(State(state): State<Arc<HttpState>>, req: Request<Body>) -> Response {
    if let Some(response) = over_the_rate(&state) {
        return response;
    }
    let Some(secret) = state.issuer.clone() else {
        return elsewhere(&state);
    };
    let collecting: Collecting = match body(req).await {
        Ok(collecting) => collecting,
        Err(response) => return response,
    };
    let digest = join::digest(&collecting.secret);
    let answer = as_owner(&state, |store| async move {
        store_join::collect(&store, collecting.request.trim(), &digest, Utc::now()).await
    })
    .await;
    let standing = |standing: &str| json!({ "standing": standing });
    match answer {
        Ok(Answer::Pending) => Json(standing("pending")).into_response(),
        Ok(Answer::Expired) => Json(standing("expired")).into_response(),
        Ok(Answer::AlreadyCollected) => Json(standing("collected")).into_response(),
        Ok(Answer::Denied { reason }) => {
            Json(json!({ "standing": "denied", "reason": reason })).into_response()
        }
        Ok(Answer::Admitted { tenant, user }) => {
            match antumbra_auth::mint_hs256(&secret, tenant.as_str(), user.as_str(), TOKEN_TTL) {
                Ok(token) => {
                    eprintln!("antumbra-mcp: {} joined {}", user.as_str(), tenant.as_str());
                    Json(json!({
                        "standing": "approved",
                        "workspace": tenant.as_str(),
                        "user": user.as_str(),
                        "token": token,
                    }))
                    .into_response()
                }
                Err(e) => {
                    eprintln!("antumbra-mcp: /join/collect could not sign a token: {e}");
                    super::internal_error()
                }
            }
        }
        Err(e) => failed("/join/collect", &e),
    }
}
