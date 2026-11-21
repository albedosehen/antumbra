//! The HTTP surface: signup / login / magic-link verify over the
//! `antumbra-control` flow, plus a store-checked health endpoint.
//!
//! Error discipline: a deliberate refusal ([`AntumbraError::Rejected`]) goes to
//! the caller as a 400 with its message; everything else is an internal fault
//! -- logged server-side, generic 500 on the wire. Link requests sit behind a
//! per-email cooldown so the endpoint cannot be driven as an email cannon.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::extract::rejection::ExtensionRejection;
use axum::extract::{ConnectInfo, Query, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use chrono::Utc;
use serde_json::json;

use antumbra_control::{
    ask_to_join, ask_to_sign_in, collect_join, collect_signin, follow, Collected, Followed, Issuer,
    JoinAsk, MagicLink, Mailer, SignedIn, SigninAsk,
};
use antumbra_core::join::Answer;
use antumbra_core::limit::{Limiter, Rate};
use antumbra_core::AntumbraError;
use antumbra_store::repo::invite;
use antumbra_store::Store;

/// What one client may ask of `/login` and `/signup`, each an email sent on
/// request: five at once covers a retry or two, and ten a minute is more than a
/// person needs.
pub const LINKS_PER_CLIENT: Rate = Rate {
    burst: 5,
    per_minute: 10,
};
/// What everyone together may ask: the ceiling on the sending reputation one
/// server spends in a minute, whatever the number of addresses or clients.
pub const LINKS_EVERYONE: Rate = Rate {
    burst: 50,
    per_minute: 100,
};
/// What one client may ask of `/join/collect`: a machine waiting on an admin
/// asks every few seconds, and a guess at a request's secret is one of these.
pub const COLLECTS_PER_CLIENT: Rate = Rate {
    burst: 20,
    per_minute: 60,
};
pub const COLLECTS_EVERYONE: Rate = Rate {
    burst: 200,
    per_minute: 600,
};

#[derive(Clone)]
pub struct AppState {
    pub store: Store,
    pub issuer: Arc<Issuer>,
    pub magic_secret: Arc<Vec<u8>>,
    pub base_url: Arc<String>,
    pub magic_ttl: Duration,
    pub mailer: Arc<dyn Mailer>,
    pub cooldown: Arc<Cooldown>,
    /// The per-client and server-wide budgets for link requests, in front of
    /// the per-address cooldown.
    pub limits: Arc<Limiter>,
    /// The budget for asking how a request to join stands. It sends no
    /// email, and a machine waiting on an admin asks every few seconds.
    pub collects: Arc<Limiter>,
}

/// The address as the cooldown keys it: trimmed, lowercased, and without a
/// `+tag` in its local part, so `A@x`, `a@x` and `a+1@x` share one window.
fn normalized(email: &str) -> String {
    let email = email.trim().to_ascii_lowercase();
    match email.split_once('@') {
        Some((local, domain)) => {
            let local = local.split('+').next().unwrap_or(local);
            format!("{local}@{domain}")
        }
        None => email,
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

/// Per-email cooldown on link requests. `/login` deliberately does not reveal
/// whether an account exists (every well-formed request is "link sent"), which
/// also means it would email arbitrary addresses on demand; this is the brake.
/// Size-capped: at the cap it first drops entries whose window has passed, then
/// the stalest entry, so a flood of distinct addresses cannot grow it
/// unboundedly or lock legitimate users out.
pub struct Cooldown {
    window: Duration,
    max_entries: usize,
    last_sent: Mutex<HashMap<String, Instant>>,
}

const COOLDOWN_MAX_ENTRIES: usize = 4096;

impl Cooldown {
    pub fn new(window: Duration) -> Self {
        Self {
            window,
            max_entries: COOLDOWN_MAX_ENTRIES,
            last_sent: Mutex::new(HashMap::new()),
        }
    }

    /// Record a send for `email` if its window is clear. `false` = still
    /// cooling down, refuse the request.
    fn try_acquire(&self, email: &str) -> bool {
        let now = Instant::now();
        // A poisoned lock holds the same map of instants: taken as it is.
        let mut map = self
            .last_sent
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(&at) = map.get(email) {
            if now.duration_since(at) < self.window {
                return false;
            }
        }
        if map.len() >= self.max_entries && !map.contains_key(email) {
            map.retain(|_, &mut at| now.duration_since(at) < self.window);
            if map.len() >= self.max_entries {
                if let Some(stalest) = map.iter().min_by_key(|(_, &at)| at).map(|(k, _)| k.clone())
                {
                    map.remove(&stalest);
                }
            }
        }
        map.insert(email.to_string(), now);
        true
    }
}

#[derive(serde::Deserialize)]
struct SignupReq {
    email: String,
    invite: String,
}

#[derive(serde::Deserialize)]
struct LoginReq {
    email: String,
}

#[derive(serde::Deserialize)]
struct VerifyQuery {
    token: String,
}

fn bad(msg: &str) -> Response {
    (StatusCode::BAD_REQUEST, Json(json!({ "error": msg }))).into_response()
}

fn internal(e: impl std::fmt::Display) -> Response {
    eprintln!("antumbra-control-server: {e}");
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({ "error": "internal error" })),
    )
        .into_response()
}

/// Route a flow error to the wire: a deliberate refusal carries its message as
/// a 400; an internal fault is logged and masked as a 500.
fn fail(e: AntumbraError) -> Response {
    if e.is_rejection() {
        bad(&e.to_string())
    } else {
        internal(e)
    }
}

fn too_many(message: &str) -> Response {
    (
        StatusCode::TOO_MANY_REQUESTS,
        [(header::RETRY_AFTER, "60")],
        Json(json!({ "error": message })),
    )
        .into_response()
}

/// Mint a magic link and (try to) email it. The mailer send may block (SMTP), so
/// it runs off the async runtime. Budgeted before anything is minted or sent:
/// the client's and everyone's buckets first, then the address's cooldown.
async fn send_link(st: &AppState, client: &str, email: String, invite: Option<String>) -> Response {
    if !st.limits.allow(client) {
        return too_many("too many sign-in requests from here; try again in a minute");
    }
    if !st.cooldown.try_acquire(&normalized(&email)) {
        return too_many("a link was sent for this address moments ago; try again shortly");
    }
    let mailer = st.mailer.clone();
    let secret = st.magic_secret.clone();
    let base = st.base_url.clone();
    let ttl = st.magic_ttl;
    let res = tokio::task::spawn_blocking(move || {
        let ml = MagicLink::new(&secret, ttl, &base, mailer.as_ref());
        match invite {
            Some(code) => ml.request_signup(&email, &code),
            None => ml.request(&email),
        }
    })
    .await;
    match res {
        Ok(Ok(())) => Json(json!({ "status": "link sent" })).into_response(),
        Ok(Err(e)) => fail(e),
        Err(e) => internal(e),
    }
}

async fn signup_handler(
    State(st): State<AppState>,
    peer: Result<ConnectInfo<SocketAddr>, ExtensionRejection>,
    Json(req): Json<SignupReq>,
) -> Response {
    // Fail fast before emailing if the invite is plainly unusable (a race that
    // consumes it before verify is still caught at signup time).
    match invite::get(&st.store, &req.invite).await {
        Ok(Some(inv)) if inv.is_usable_at(Utc::now()) => {}
        Ok(_) => return bad("unknown, expired, or already-used invite code"),
        Err(e) => return internal(e),
    }
    send_link(&st, &client_of(peer), req.email, Some(req.invite)).await
}

async fn login_handler(
    State(st): State<AppState>,
    peer: Result<ConnectInfo<SocketAddr>, ExtensionRejection>,
    Json(req): Json<LoginReq>,
) -> Response {
    send_link(&st, &client_of(peer), req.email, None).await
}

/// Text as it may stand in a page: nothing in it is read as markup.
fn escaped(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            c => out.push(c),
        }
    }
    out
}

/// A page for the person who followed a link: a heading, what happened in a
/// sentence or two, and, when there is one, a value to copy. It runs no
/// script and loads nothing; it is not kept by the browser or sent on as a
/// referrer, since it may hold a token.
fn page(status: StatusCode, heading: &str, lines: &[&str], copy: Option<&str>) -> Response {
    let mut body = String::new();
    for line in lines {
        body.push_str(&format!("<p>{}</p>", escaped(line)));
    }
    if let Some(copy) = copy {
        body.push_str(&format!("<pre>{}</pre>", escaped(copy)));
    }
    let html = format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\
         <meta name=\"color-scheme\" content=\"light dark\"><title>{title} · Antumbra</title>\
         <style>body{{font:16px/1.55 system-ui,sans-serif;max-width:34rem;margin:12vh auto;padding:0 16px}}\
         h1{{font-size:1.4rem}}pre{{padding:12px;border:1px solid;border-radius:8px;white-space:pre-wrap;overflow-wrap:anywhere;font-size:.85rem}}</style>\
         </head><body><h1>{title}</h1>{body}</body></html>",
        title = escaped(heading),
    );
    (
        status,
        [
            (header::CONTENT_TYPE, "text/html; charset=utf-8"),
            (header::CACHE_CONTROL, "no-store"),
            (header::REFERRER_POLICY, "no-referrer"),
            (header::X_CONTENT_TYPE_OPTIONS, "nosniff"),
            (
                header::CONTENT_SECURITY_POLICY,
                "default-src 'none'; style-src 'unsafe-inline'; frame-ancestors 'none'",
            ),
        ],
        html,
    )
        .into_response()
}

/// A followed link, answered as JSON to a program and as a page to a
/// browser: a browser says it takes `text/html`, and nothing else here does.
async fn verify_handler(
    State(st): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<VerifyQuery>,
) -> Response {
    let to_a_browser = headers
        .get(header::ACCEPT)
        .and_then(|accept| accept.to_str().ok())
        .is_some_and(|accept| accept.contains("text/html"));
    let ml = MagicLink::new(
        &st.magic_secret,
        st.magic_ttl,
        &st.base_url,
        st.mailer.as_ref(),
    );
    const ASKED: &str = "Your address is verified and the workspace's admins have your request. Run `antumbra join --status` on the machine you asked from.";
    const SIGNING_IN: &str = "Your address is verified. The machine that asked can collect its token now: run `antumbra login --status` there.";
    match follow(&st.store, &ml, &q.token, &st.issuer).await {
        Ok(Followed::Token(token)) if to_a_browser => page(
            StatusCode::OK,
            "You are signed in",
            &[
                "This is your token. It is shown once: keep it somewhere only you can read, and give it to `antumbra setup hosted`.",
            ],
            Some(&token),
        ),
        Ok(Followed::Token(token)) => Json(json!({ "token": token })).into_response(),
        Ok(Followed::Asked {
            workspace, handle, ..
        }) if to_a_browser => page(
            StatusCode::OK,
            "Your address is verified",
            &[
                &format!("You asked to join {workspace} as {handle}. Its admins have your request now."),
                "On the machine you asked from, run `antumbra join --status` to see how it stands and to collect your token once you are let in.",
                "You can close this page.",
            ],
            None,
        ),
        Ok(Followed::Asked {
            request,
            workspace,
            handle,
            standing,
        }) => Json(json!({
            "asked": true,
            "request": request,
            "workspace": workspace,
            "handle": handle,
            "standing": standing,
            "next": ASKED,
        }))
        .into_response(),
        Ok(Followed::SigningIn { .. }) if to_a_browser => page(
            StatusCode::OK,
            "Your address is verified",
            &[
                "The machine that asked to sign in can collect its token now: run `antumbra login --status` there.",
                "Nothing was given to this browser. You can close this page.",
            ],
            None,
        ),
        Ok(Followed::SigningIn { request }) => Json(json!({
            "signing_in": true,
            "request": request,
            "next": SIGNING_IN,
        }))
        .into_response(),
        Err(e) if to_a_browser && e.is_rejection() => page(
            StatusCode::BAD_REQUEST,
            "This link cannot be used",
            &[&e.to_string(), "A link works once and for a short time. Ask for another from where you asked for this one."],
            None,
        ),
        Err(e) if to_a_browser => {
            eprintln!("antumbra-control-server: {e}");
            page(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Something went wrong here",
                &["A link works once, so this one may be spent. Ask for another from where you asked for this one."],
                None,
            )
        }
        Err(e) => fail(e),
    }
}

#[derive(serde::Deserialize)]
struct SigninReq {
    email: String,
    /// The SHA-256, in hex, of the secret the token will be collected with.
    secret_digest: String,
    #[serde(default)]
    workspace: Option<String>,
}

/// Ask to sign a machine in to an account. A link is sent whether or not the
/// address has one; the id is told now, so the machine can collect with it.
async fn signin_request_handler(
    State(st): State<AppState>,
    peer: Result<ConnectInfo<SocketAddr>, ExtensionRejection>,
    Json(req): Json<SigninReq>,
) -> Response {
    if !st.limits.allow(&client_of(peer)) {
        return too_many("too many sign-in requests from here; try again in a minute");
    }
    if !st.cooldown.try_acquire(&normalized(&req.email)) {
        return too_many("a link was sent for this address moments ago; try again shortly");
    }
    let request = antumbra_control::new_request_id();
    let ask = SigninAsk {
        request: request.clone(),
        secret_digest: req.secret_digest,
        workspace: req.workspace,
    };
    let (mailer, secret, base, ttl, email) = (
        st.mailer.clone(),
        st.magic_secret.clone(),
        st.base_url.clone(),
        st.magic_ttl,
        req.email,
    );
    let sent = tokio::task::spawn_blocking(move || {
        let ml = MagicLink::new(&secret, ttl, &base, mailer.as_ref());
        ask_to_sign_in(&ml, &email, ask)
    })
    .await;
    match sent {
        Ok(Ok(())) => Json(json!({ "request": request, "standing": "unverified" })).into_response(),
        Ok(Err(e)) => fail(e),
        Err(e) => internal(e),
    }
}

#[derive(serde::Deserialize)]
struct SigninCollectReq {
    request: String,
    secret: String,
    #[serde(default)]
    workspace: Option<String>,
}

/// How a sign-in stands; followed, its token, once; and for an account in
/// several workspaces with none named, which they are.
async fn signin_collect_handler(
    State(st): State<AppState>,
    peer: Result<ConnectInfo<SocketAddr>, ExtensionRejection>,
    Json(req): Json<SigninCollectReq>,
) -> Response {
    if !st.collects.allow(&client_of(peer)) {
        return too_many("too many requests from here; try again in a minute");
    }
    let standing = |standing: &str| Json(json!({ "standing": standing })).into_response();
    match collect_signin(
        &st.store,
        &req.request,
        &req.secret,
        req.workspace.as_deref(),
        &st.issuer,
    )
    .await
    {
        Ok(SignedIn::Unverified) => standing("unverified"),
        Ok(SignedIn::Collected) => standing("collected"),
        Ok(SignedIn::Expired) => standing("expired"),
        Ok(SignedIn::Choose(offered)) => Json(json!({
            "standing": "choose",
            "workspaces": offered
                .iter()
                .map(|m| json!({ "workspace": m.tenant, "user": m.user }))
                .collect::<Vec<_>>(),
        }))
        .into_response(),
        Ok(SignedIn::Token {
            workspace,
            user,
            token,
        }) => Json(json!({
            "standing": "approved",
            "workspace": workspace,
            "user": user,
            "token": token,
        }))
        .into_response(),
        Err(e) => fail(e),
    }
}

#[derive(serde::Deserialize)]
struct JoinReq {
    email: String,
    code: String,
    handle: String,
    #[serde(default)]
    display: Option<String>,
    #[serde(default)]
    note: Option<String>,
    #[serde(default)]
    host: Option<String>,
    /// The SHA-256, in hex, of the secret the token will be collected with.
    secret_digest: String,
}

/// Ask to join a workspace with an address this server verifies. The request
/// is made when the emailed link is followed; its id is told now, so the
/// machine that asked can collect with it.
async fn join_request_handler(
    State(st): State<AppState>,
    peer: Result<ConnectInfo<SocketAddr>, ExtensionRejection>,
    Json(req): Json<JoinReq>,
) -> Response {
    if !st.limits.allow(&client_of(peer)) {
        return too_many("too many requests from here; try again in a minute");
    }
    if !st.cooldown.try_acquire(&normalized(&req.email)) {
        return too_many("a link was sent for this address moments ago; try again shortly");
    }
    let request = antumbra_control::new_request_id();
    let ask = JoinAsk {
        request: request.clone(),
        code: req.code,
        handle: req.handle,
        display: req.display,
        note: req.note,
        host: req.host,
        secret_digest: req.secret_digest,
    };
    let (store, mailer, secret, base, ttl, email) = (
        st.store.clone(),
        st.mailer.clone(),
        st.magic_secret.clone(),
        st.base_url.clone(),
        st.magic_ttl,
        req.email,
    );
    // The mailer may block (SMTP), so the send runs off the async runtime.
    let handle = tokio::runtime::Handle::current();
    let sent = tokio::task::spawn_blocking(move || {
        let ml = MagicLink::new(&secret, ttl, &base, mailer.as_ref());
        handle.block_on(ask_to_join(&store, &ml, &email, ask))
    })
    .await;
    match sent {
        Ok(Ok(())) => Json(json!({
            "request": request,
            "standing": "unverified",
            "next": "Follow the link sent to your address; the workspace's admins see your request once you have.",
        }))
        .into_response(),
        Ok(Err(e)) => fail(e),
        Err(e) => internal(e),
    }
}

#[derive(serde::Deserialize)]
struct CollectReq {
    request: String,
    secret: String,
}

/// How a request made through a link stands, and, approved, its token: once.
async fn join_collect_handler(
    State(st): State<AppState>,
    peer: Result<ConnectInfo<SocketAddr>, ExtensionRejection>,
    Json(req): Json<CollectReq>,
) -> Response {
    if !st.collects.allow(&client_of(peer)) {
        return too_many("too many requests from here; try again in a minute");
    }
    let standing = |standing: &str| Json(json!({ "standing": standing })).into_response();
    match collect_join(&st.store, &req.request, &req.secret, &st.issuer).await {
        Ok(Collected::Unverified) => standing("unverified"),
        Ok(Collected::Standing(Answer::Pending)) => standing("pending"),
        Ok(Collected::Standing(Answer::Expired)) => standing("expired"),
        Ok(Collected::Standing(Answer::AlreadyCollected)) => standing("collected"),
        Ok(Collected::Standing(Answer::Denied { reason })) => {
            Json(json!({ "standing": "denied", "reason": reason })).into_response()
        }
        // The store admits; `collect_join` turns that into a token.
        Ok(Collected::Standing(Answer::Admitted { .. })) => internal("an admission with no token"),
        Ok(Collected::Admitted {
            workspace,
            user,
            token,
        }) => {
            eprintln!("antumbra-control-server: {user} joined {workspace}");
            Json(json!({
                "standing": "approved",
                "workspace": workspace,
                "user": user,
                "token": token,
            }))
            .into_response()
        }
        Err(e) => fail(e),
    }
}

/// Health means "can serve a signup": the store round-trip is the check, so a
/// lost database connection turns the container unhealthy instead of lingering.
async fn healthz_handler(State(st): State<AppState>) -> Response {
    match invite::get(&st.store, "healthz-probe").await {
        Ok(_) => (StatusCode::OK, "ok").into_response(),
        Err(e) => {
            eprintln!("antumbra-control-server: healthz store check failed: {e}");
            (StatusCode::SERVICE_UNAVAILABLE, "store unreachable").into_response()
        }
    }
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/signup", post(signup_handler))
        .route("/login", post(login_handler))
        .route("/magic/verify", get(verify_handler))
        .route("/login/request", post(signin_request_handler))
        .route("/login/collect", post(signin_collect_handler))
        .route("/join/request", post(join_request_handler))
        .route("/join/collect", post(join_collect_handler))
        .route("/healthz", get(healthz_handler))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{Context, Result};
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    use antumbra_store::testing::mem_store;

    // The control crate's throwaway RS256 private key (test-only).
    const RS_PRIV: &[u8] = include_bytes!("../../antumbra-control/tests/test_jwt_priv.pem");

    /// Captures the links it would have sent, so a test can follow them.
    struct Capture(Mutex<Vec<String>>);

    impl Capture {
        /// Every link sent, in order. The fake only ever pushes to the list, so
        /// a poisoned lock holds a whole list and is read as is.
        fn sent(&self) -> Vec<String> {
            self.0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone()
        }
    }

    impl Mailer for Capture {
        fn send_link(&self, _to: &str, link: &str) -> antumbra_core::Result<()> {
            // Read as `sent` is: a push cannot leave the list half-written.
            self.0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(link.to_string());
            Ok(())
        }
    }

    fn state_with(store: Store, capture: Arc<Capture>, cooldown: Duration) -> AppState {
        AppState {
            store,
            issuer: Arc::new(Issuer::new(
                RS_PRIV.to_vec(),
                "antumbra",
                Duration::from_secs(3600),
            )),
            magic_secret: Arc::new(b"test-magic".to_vec()),
            base_url: Arc::new("https://app".to_string()),
            magic_ttl: Duration::from_secs(600),
            mailer: capture,
            cooldown: Arc::new(Cooldown::new(cooldown)),
            limits: Arc::new(Limiter::new(LINKS_PER_CLIENT, LINKS_EVERYONE)),
            collects: Arc::new(Limiter::new(COLLECTS_PER_CLIENT, COLLECTS_EVERYONE)),
        }
    }

    fn post_json(uri: &str, body: &str) -> Result<Request<Body>> {
        Ok(Request::post(uri)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))?)
    }

    async fn body_json(resp: Response) -> Result<serde_json::Value> {
        let body = axum::body::to_bytes(resp.into_body(), 64 * 1024).await?;
        Ok(serde_json::from_slice(&body)?)
    }

    #[tokio::test]
    async fn signup_then_verify_issues_a_token_and_the_link_is_single_use() -> Result<()> {
        let store = mem_store().await?;
        antumbra_store::repo::invite::mint(&store, "inv-1", None).await?;
        let capture = Arc::new(Capture(Mutex::new(Vec::new())));
        let app = router(state_with(store, capture.clone(), Duration::ZERO));

        // POST /signup -> a magic link is "sent" (captured).
        let resp = app
            .clone()
            .oneshot(post_json(
                "/signup",
                r#"{"email":"a@b.com","invite":"inv-1"}"#,
            )?)
            .await?;
        assert_eq!(resp.status(), StatusCode::OK);

        let link = capture.sent()[0].clone();
        let token = link
            .split("token=")
            .nth(1)
            .context("the link carries no token")?
            .to_string();

        // GET /magic/verify -> an issued bearer token.
        let verify_uri = format!("/magic/verify?token={token}");
        let resp = app
            .clone()
            .oneshot(Request::get(&verify_uri).body(Body::empty())?)
            .await?;
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await?;
        assert!(
            v["token"].as_str().unwrap_or_default().starts_with("ey"),
            "an issued JWT"
        );

        // Replaying the same link is refused as a caller error, with the
        // reason on the wire (a rejection, not an internal fault).
        let resp = app
            .oneshot(Request::get(&verify_uri).body(Body::empty())?)
            .await?;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let v = body_json(resp).await?;
        assert!(
            v["error"]
                .as_str()
                .is_some_and(|e| e.contains("already used")),
            "{v}"
        );
        Ok(())
    }

    /// Joining over the routes: a bad code is refused and nothing is sent;
    /// a good one sends a link and tells the request's id; collecting says
    /// unverified until the link is followed, pending until an admin decides,
    /// then gives the token once.
    #[tokio::test]
    async fn a_request_to_join_is_verified_by_its_link_and_collected_once() -> Result<()> {
        use antumbra_core::join::{digest, JoinCode};
        use antumbra_core::{TenantId, UserId};
        use antumbra_store::repo::join as store_join;
        let store = mem_store().await?;
        let (crew, ada, now) = (
            TenantId::new("ws:crew"),
            UserId::new("user:ada"),
            Utc::now(),
        );
        store_join::make_code(
            &store,
            &JoinCode {
                key: digest("crew-code"),
                tenant: crew.clone(),
                label: None,
                created_by: ada.clone(),
                created_at: now,
                expires_at: now + chrono::Duration::days(14),
                uses_left: 1,
                needs_approval: true,
                revoked: false,
            },
        )
        .await?;
        let capture = Arc::new(Capture(Mutex::new(Vec::new())));
        let app = router(state_with(store.clone(), capture.clone(), Duration::ZERO));
        let asking = |code: &str| {
            json!({ "email": "bodil@example.com", "code": code, "handle": "Bodil", "secret_digest": digest("my-secret") }).to_string()
        };

        let resp = app
            .clone()
            .oneshot(post_json("/join/request", &asking("nope"))?)
            .await?;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert!(
            capture.sent().is_empty(),
            "a link was sent for a code that admits no one"
        );

        let resp = app
            .clone()
            .oneshot(post_json("/join/request", &asking("crew-code"))?)
            .await?;
        assert_eq!(resp.status(), StatusCode::OK);
        let asked = body_json(resp).await?;
        assert_eq!(asked["standing"], "unverified");
        let request = asked["request"]
            .as_str()
            .context("no request id")?
            .to_string();
        let collecting = json!({ "request": request, "secret": "my-secret" }).to_string();
        let collect = || async {
            let resp = app
                .clone()
                .oneshot(post_json("/join/collect", &collecting)?)
                .await?;
            assert_eq!(resp.status(), StatusCode::OK);
            body_json(resp).await
        };
        assert_eq!(collect().await?["standing"], "unverified");
        assert!(store_join::requests(&store, &crew).await?.is_empty());

        let token = capture.sent()[0]
            .split("token=")
            .nth(1)
            .context("the link carries no token")?
            .to_string();
        let resp = app
            .clone()
            .oneshot(Request::get(format!("/magic/verify?token={token}")).body(Body::empty())?)
            .await?;
        assert_eq!(resp.status(), StatusCode::OK);
        let followed = body_json(resp).await?;
        assert_eq!(
            (
                followed["asked"].clone(),
                followed["request"].clone(),
                followed["workspace"].clone()
            ),
            (json!(true), json!(request), json!("ws:crew"))
        );
        assert!(
            followed.get("token").is_none(),
            "following a link that asks to join signed someone in"
        );

        assert_eq!(collect().await?["standing"], "pending");
        store_join::decide(&store, &crew, &ada, &request, true, None, Utc::now()).await?;
        let got = collect().await?;
        assert_eq!(
            (
                got["standing"].clone(),
                got["workspace"].clone(),
                got["user"].clone()
            ),
            (json!("approved"), json!("ws:crew"), json!("user:bodil"))
        );
        assert!(got["token"].as_str().unwrap_or_default().starts_with("ey"));
        let again = collect().await?;
        assert_eq!(again["standing"], "collected");
        assert!(again.get("token").is_none());
        Ok(())
    }

    /// A machine signs in over the routes: the link's follower is given no
    /// token, and the machine collects one, once.
    #[tokio::test]
    async fn a_machine_signs_in_over_the_routes() -> Result<()> {
        use antumbra_core::join::digest;
        let store = mem_store().await?;
        antumbra_store::repo::invite::mint(&store, "inv", None).await?;
        let capture = Arc::new(Capture(Mutex::new(Vec::new())));
        let app = router(state_with(store, capture.clone(), Duration::ZERO));
        let follow = |n: usize| {
            let (app, capture) = (app.clone(), capture.clone());
            async move {
                let token = capture.sent()[n]
                    .split("token=")
                    .nth(1)
                    .context("the link carries no token")?
                    .to_string();
                let resp = app
                    .oneshot(
                        Request::get(format!("/magic/verify?token={token}")).body(Body::empty())?,
                    )
                    .await?;
                assert_eq!(resp.status(), StatusCode::OK);
                body_json(resp).await
            }
        };
        let resp = app
            .clone()
            .oneshot(post_json(
                "/signup",
                r#"{"email":"dana@example.com","invite":"inv"}"#,
            )?)
            .await?;
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(follow(0).await?["token"].is_string());

        let asking =
            json!({ "email": "dana@example.com", "secret_digest": digest("machine-secret") })
                .to_string();
        let resp = app
            .clone()
            .oneshot(post_json("/login/request", &asking)?)
            .await?;
        assert_eq!(resp.status(), StatusCode::OK);
        let request = body_json(resp).await?["request"]
            .as_str()
            .context("no request id")?
            .to_string();
        let collecting = json!({ "request": request, "secret": "machine-secret" }).to_string();
        let collect = || async {
            let resp = app
                .clone()
                .oneshot(post_json("/login/collect", &collecting)?)
                .await?;
            assert_eq!(resp.status(), StatusCode::OK);
            body_json(resp).await
        };
        assert_eq!(collect().await?["standing"], "unverified");
        let followed = follow(1).await?;
        assert_eq!(followed["signing_in"], true);
        assert!(
            followed.get("token").is_none(),
            "the link's follower was given a token"
        );
        let got = collect().await?;
        assert_eq!(got["standing"], "approved");
        assert!(got["token"].as_str().unwrap_or_default().starts_with("ey"));
        assert!(got["workspace"]
            .as_str()
            .unwrap_or_default()
            .starts_with("ws:"));
        let again = collect().await?;
        assert_eq!(again["standing"], "collected");
        assert!(again.get("token").is_none());
        Ok(())
    }

    /// A browser that follows a link is given a page, with nothing read as
    /// markup and nothing kept: what happened for a request to join, the
    /// token for a signup, and why for a link already used. A program is
    /// still given JSON.
    #[tokio::test]
    async fn a_browser_that_follows_a_link_is_given_a_page() -> Result<()> {
        use antumbra_core::join::{digest, JoinCode};
        use antumbra_core::{TenantId, UserId};
        let store = mem_store().await?;
        antumbra_store::repo::invite::mint(&store, "inv", None).await?;
        let now = Utc::now();
        antumbra_store::repo::join::make_code(
            &store,
            &JoinCode {
                key: digest("crew-code"),
                tenant: TenantId::new("ws:crew"),
                label: None,
                created_by: UserId::new("user:ada"),
                created_at: now,
                expires_at: now + chrono::Duration::days(14),
                uses_left: 1,
                needs_approval: true,
                revoked: false,
            },
        )
        .await?;
        let capture = Arc::new(Capture(Mutex::new(Vec::new())));
        let app = router(state_with(store, capture.clone(), Duration::ZERO));
        let browse = |n: usize| {
            let (app, capture) = (app.clone(), capture.clone());
            async move {
                let token = capture.sent()[n]
                    .split("token=")
                    .nth(1)
                    .context("the link carries no token")?
                    .to_string();
                let resp = app
                    .oneshot(
                        Request::get(format!("/magic/verify?token={token}"))
                            .header(header::ACCEPT, "text/html,application/xhtml+xml,*/*;q=0.8")
                            .body(Body::empty())?,
                    )
                    .await?;
                let status = resp.status();
                let kind = resp
                    .headers()
                    .get(header::CONTENT_TYPE)
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or_default()
                    .to_string();
                let kept = resp
                    .headers()
                    .get(header::CACHE_CONTROL)
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or_default()
                    .to_string();
                let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20).await?;
                anyhow::Ok((
                    status,
                    kind,
                    kept,
                    String::from_utf8_lossy(&bytes).to_string(),
                ))
            }
        };

        let asking = json!({ "email": "bodil@example.com", "code": "crew-code", "handle": "bodil", "secret_digest": digest("s") }).to_string();
        assert_eq!(
            app.clone()
                .oneshot(post_json("/join/request", &asking)?)
                .await?
                .status(),
            StatusCode::OK
        );
        let (status, kind, kept, html) = browse(0).await?;
        assert_eq!((status, kept.as_str()), (StatusCode::OK, "no-store"));
        assert!(kind.starts_with("text/html"), "{kind}");
        assert!(
            html.contains("You asked to join ws:crew as bodil")
                && html.contains("antumbra join --status"),
            "{html}"
        );
        assert!(!html.contains("<script"), "{html}");
        // Followed again: a page that says why, not JSON and not a token.
        let (status, _, _, html) = browse(0).await?;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(
            html.contains("This link cannot be used") && html.contains("already used"),
            "{html}"
        );

        // A signup's token is on the page, once.
        assert_eq!(
            app.clone()
                .oneshot(post_json(
                    "/signup",
                    r#"{"email":"dana@example.com","invite":"inv"}"#
                )?)
                .await?
                .status(),
            StatusCode::OK
        );
        let (status, _, _, html) = browse(1).await?;
        assert_eq!(status, StatusCode::OK);
        assert!(html.contains("<pre>ey"), "{html}");

        // What a page holds is text, whatever it is.
        assert_eq!(
            escaped(r#"<b a="1">&'x'</b>"#),
            "&lt;b a=&quot;1&quot;&gt;&amp;&#39;x&#39;&lt;/b&gt;"
        );
        Ok(())
    }

    #[tokio::test]
    async fn signup_with_unknown_or_expired_invite_is_rejected_before_emailing() -> Result<()> {
        let store = mem_store().await?;
        antumbra_store::repo::invite::mint(
            &store,
            "stale",
            Some(Utc::now() - chrono::Duration::minutes(1)),
        )
        .await?;
        let capture = Arc::new(Capture(Mutex::new(Vec::new())));
        let app = router(state_with(store, capture.clone(), Duration::ZERO));
        for body in [
            r#"{"email":"a@b.com","invite":"nope"}"#,
            r#"{"email":"a@b.com","invite":"stale"}"#,
        ] {
            let resp = app.clone().oneshot(post_json("/signup", body)?).await?;
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        }
        assert!(
            capture.sent().is_empty(),
            "no link emailed for a dead invite"
        );
        Ok(())
    }

    #[tokio::test]
    async fn link_requests_cool_down_per_email() -> Result<()> {
        let store = mem_store().await?;
        let capture = Arc::new(Capture(Mutex::new(Vec::new())));
        // A long window: the second request for the same address must wait.
        let app = router(state_with(store, capture.clone(), Duration::from_secs(600)));

        let login = |email: &str| post_json("/login", &format!(r#"{{"email":"{email}"}}"#));
        let first = app.clone().oneshot(login("hot@x.com")?).await?;
        assert_eq!(first.status(), StatusCode::OK);
        let second = app.clone().oneshot(login("hot@x.com")?).await?;
        assert_eq!(second.status(), StatusCode::TOO_MANY_REQUESTS);
        // A different address is unaffected.
        let other = app.oneshot(login("cool@x.com")?).await?;
        assert_eq!(other.status(), StatusCode::OK);
        assert_eq!(capture.sent().len(), 2);
        Ok(())
    }

    // The cooldown keys on the address, not its spelling: case, surrounding
    // space and a `+tag` are one window.
    #[tokio::test]
    async fn link_requests_cool_down_across_an_addresss_spellings() -> Result<()> {
        let store = mem_store().await?;
        let capture = Arc::new(Capture(Mutex::new(Vec::new())));
        let app = router(state_with(store, capture.clone(), Duration::from_secs(600)));
        let login = |email: &str| post_json("/login", &format!(r#"{{"email":"{email}"}}"#));
        let first = app.clone().oneshot(login("Hot@X.com")?).await?;
        anyhow::ensure!(first.status() == StatusCode::OK);
        for spelling in ["hot@x.com", "HOT@x.com", "hot+promo@x.com", " hot@x.com "] {
            let again = app.clone().oneshot(login(spelling)?).await?;
            anyhow::ensure!(
                again.status() == StatusCode::TOO_MANY_REQUESTS,
                "{spelling:?} was not the same address"
            );
            anyhow::ensure!(again.headers().get(header::RETRY_AFTER).is_some());
        }
        let sent = capture.sent().len();
        anyhow::ensure!(sent == 1, "{sent} links sent for one address");
        Ok(())
    }

    // Distinct addresses pass the cooldown, so the client's bucket is what
    // stops a client cycling through them: its burst, then a refusal.
    #[tokio::test]
    async fn link_requests_past_a_clients_burst_are_refused() -> Result<()> {
        let store = mem_store().await?;
        let capture = Arc::new(Capture(Mutex::new(Vec::new())));
        let app = router(state_with(store, capture.clone(), Duration::ZERO));
        let login = |n: u32| post_json("/login", &format!(r#"{{"email":"person{n}@x.com"}}"#));
        for n in 0..LINKS_PER_CLIENT.burst {
            let resp = app.clone().oneshot(login(n)?).await?;
            anyhow::ensure!(resp.status() == StatusCode::OK, "request {n}");
        }
        let refused = app.clone().oneshot(login(LINKS_PER_CLIENT.burst)?).await?;
        anyhow::ensure!(refused.status() == StatusCode::TOO_MANY_REQUESTS);
        let said = body_json(refused).await?;
        anyhow::ensure!(
            said["error"]
                .as_str()
                .is_some_and(|s| s.contains("from here")),
            "{said}"
        );
        let sent = capture.sent().len();
        anyhow::ensure!(sent as u32 == LINKS_PER_CLIENT.burst, "{sent} links sent");
        Ok(())
    }

    #[test]
    fn an_address_normalizes_to_one_spelling() -> Result<()> {
        for spelling in [
            " Hot@X.com ",
            "hot@x.com",
            "HOT+promo+more@X.COM",
            "hot+1@x.com",
        ] {
            assert_eq!(normalized(spelling), "hot@x.com", "{spelling:?}");
        }
        assert_eq!(normalized("no-at-sign"), "no-at-sign");
        Ok(())
    }

    #[tokio::test]
    async fn a_malformed_email_is_a_rejection_not_an_internal_error() -> Result<()> {
        let store = mem_store().await?;
        let capture = Arc::new(Capture(Mutex::new(Vec::new())));
        let app = router(state_with(store, capture, Duration::ZERO));
        let resp = app
            .oneshot(post_json("/login", r#"{"email":"not-an-email"}"#)?)
            .await?;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let v = body_json(resp).await?;
        assert!(
            v["error"]
                .as_str()
                .is_some_and(|e| e.contains("valid email")),
            "{v}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn healthz_reports_the_store_round_trip() -> Result<()> {
        let store = mem_store().await?;
        let capture = Arc::new(Capture(Mutex::new(Vec::new())));
        let app = router(state_with(store, capture, Duration::ZERO));
        let resp = app
            .oneshot(Request::get("/healthz").body(Body::empty())?)
            .await?;
        assert_eq!(resp.status(), StatusCode::OK);
        Ok(())
    }

    #[test]
    fn the_cooldown_cap_evicts_stale_entries_first() -> Result<()> {
        let cd = Cooldown {
            window: Duration::from_secs(600),
            max_entries: 2,
            last_sent: Mutex::new(HashMap::new()),
        };
        assert!(cd.try_acquire("a@x.com"));
        assert!(cd.try_acquire("b@x.com"));
        // At the cap, a third address still acquires (the stalest is evicted)
        // rather than locking new users out...
        assert!(cd.try_acquire("c@x.com"));
        // ...and an address inside its window stays refused.
        assert!(!cd.try_acquire("c@x.com"));
        Ok(())
    }
}
