//! The HTTP surface: signup / login / magic-link verify over the
//! `antumbra-control` flow, plus a store-checked health endpoint.
//!
//! Error discipline: a deliberate refusal ([`AntumbraError::Rejected`]) goes to
//! the caller as a 400 with its message; everything else is an internal fault
//! -- logged server-side, generic 500 on the wire. Link requests sit behind a
//! per-email cooldown so the endpoint cannot be driven as an email cannon.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use chrono::Utc;
use serde_json::json;

use antumbra_control::{authenticate, Issuer, MagicLink, Mailer};
use antumbra_core::AntumbraError;
use antumbra_store::repo::invite;
use antumbra_store::Store;

#[derive(Clone)]
pub struct AppState {
    pub store: Store,
    pub issuer: Arc<Issuer>,
    pub magic_secret: Arc<Vec<u8>>,
    pub base_url: Arc<String>,
    pub magic_ttl: Duration,
    pub mailer: Arc<dyn Mailer>,
    pub cooldown: Arc<Cooldown>,
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
        let mut map = self.last_sent.lock().expect("cooldown lock");
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

fn too_many() -> Response {
    (
        StatusCode::TOO_MANY_REQUESTS,
        Json(json!({ "error": "a link was sent for this address moments ago; try again shortly" })),
    )
        .into_response()
}

/// Mint a magic link and (try to) email it. The mailer send may block (SMTP), so
/// it runs off the async runtime.
async fn send_link(st: &AppState, email: String, invite: Option<String>) -> Response {
    if !st.cooldown.try_acquire(&email) {
        return too_many();
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

async fn signup_handler(State(st): State<AppState>, Json(req): Json<SignupReq>) -> Response {
    // Fail fast before emailing if the invite is plainly unusable (a race that
    // consumes it before verify is still caught at signup time).
    match invite::get(&st.store, &req.invite).await {
        Ok(Some(inv)) if inv.is_usable_at(Utc::now()) => {}
        Ok(_) => return bad("unknown, expired, or already-used invite code"),
        Err(e) => return internal(e),
    }
    send_link(&st, req.email, Some(req.invite)).await
}

async fn login_handler(State(st): State<AppState>, Json(req): Json<LoginReq>) -> Response {
    send_link(&st, req.email, None).await
}

async fn verify_handler(State(st): State<AppState>, Query(q): Query<VerifyQuery>) -> Response {
    let ml = MagicLink::new(
        &st.magic_secret,
        st.magic_ttl,
        &st.base_url,
        st.mailer.as_ref(),
    );
    match authenticate(&st.store, &ml, &q.token, &st.issuer).await {
        Ok(token) => Json(json!({ "token": token })).into_response(),
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
        .route("/healthz", get(healthz_handler))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    use antumbra_store::EMBED_DIM;

    // The control crate's throwaway RS256 private key (test-only).
    const RS_PRIV: &[u8] = include_bytes!("../../antumbra-control/tests/test_jwt_priv.pem");

    struct Capture(Mutex<Vec<String>>);
    impl Mailer for Capture {
        fn send_link(&self, _to: &str, link: &str) -> antumbra_core::Result<()> {
            self.0.lock().unwrap().push(link.to_string());
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
        }
    }

    fn post_json(uri: &str, body: &str) -> Request<Body> {
        Request::post(uri)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    async fn body_json(resp: Response) -> serde_json::Value {
        let body = axum::body::to_bytes(resp.into_body(), 64 * 1024)
            .await
            .unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    #[tokio::test]
    async fn signup_then_verify_issues_a_token_and_the_link_is_single_use() {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        antumbra_store::repo::invite::mint(&store, "inv-1", None)
            .await
            .unwrap();
        let capture = Arc::new(Capture(Mutex::new(Vec::new())));
        let app = router(state_with(store, capture.clone(), Duration::ZERO));

        // POST /signup -> a magic link is "sent" (captured).
        let resp = app
            .clone()
            .oneshot(post_json(
                "/signup",
                r#"{"email":"a@b.com","invite":"inv-1"}"#,
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let link = capture.0.lock().unwrap()[0].clone();
        let token = link.split("token=").nth(1).unwrap().to_string();

        // GET /magic/verify -> an issued bearer token.
        let verify_uri = format!("/magic/verify?token={token}");
        let resp = app
            .clone()
            .oneshot(Request::get(&verify_uri).body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert!(
            v["token"].as_str().unwrap_or_default().starts_with("ey"),
            "an issued JWT"
        );

        // Replaying the same link is refused as a caller error, with the
        // reason on the wire (a rejection, not an internal fault).
        let resp = app
            .oneshot(Request::get(&verify_uri).body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let v = body_json(resp).await;
        assert!(v["error"].as_str().unwrap().contains("already used"));
    }

    #[tokio::test]
    async fn signup_with_unknown_or_expired_invite_is_rejected_before_emailing() {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        antumbra_store::repo::invite::mint(
            &store,
            "stale",
            Some(Utc::now() - chrono::Duration::minutes(1)),
        )
        .await
        .unwrap();
        let capture = Arc::new(Capture(Mutex::new(Vec::new())));
        let app = router(state_with(store, capture.clone(), Duration::ZERO));
        for body in [
            r#"{"email":"a@b.com","invite":"nope"}"#,
            r#"{"email":"a@b.com","invite":"stale"}"#,
        ] {
            let resp = app
                .clone()
                .oneshot(post_json("/signup", body))
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        }
        assert!(
            capture.0.lock().unwrap().is_empty(),
            "no link emailed for a dead invite"
        );
    }

    #[tokio::test]
    async fn link_requests_cool_down_per_email() {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        let capture = Arc::new(Capture(Mutex::new(Vec::new())));
        // A long window: the second request for the same address must wait.
        let app = router(state_with(store, capture.clone(), Duration::from_secs(600)));

        let login = |email: &str| post_json("/login", &format!(r#"{{"email":"{email}"}}"#));
        let first = app.clone().oneshot(login("hot@x.com")).await.unwrap();
        assert_eq!(first.status(), StatusCode::OK);
        let second = app.clone().oneshot(login("hot@x.com")).await.unwrap();
        assert_eq!(second.status(), StatusCode::TOO_MANY_REQUESTS);
        // A different address is unaffected.
        let other = app.oneshot(login("cool@x.com")).await.unwrap();
        assert_eq!(other.status(), StatusCode::OK);
        assert_eq!(capture.0.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn a_malformed_email_is_a_rejection_not_an_internal_error() {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        let capture = Arc::new(Capture(Mutex::new(Vec::new())));
        let app = router(state_with(store, capture, Duration::ZERO));
        let resp = app
            .oneshot(post_json("/login", r#"{"email":"not-an-email"}"#))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let v = body_json(resp).await;
        assert!(v["error"].as_str().unwrap().contains("valid email"));
    }

    #[tokio::test]
    async fn healthz_reports_the_store_round_trip() {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        let capture = Arc::new(Capture(Mutex::new(Vec::new())));
        let app = router(state_with(store, capture, Duration::ZERO));
        let resp = app
            .oneshot(Request::get("/healthz").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[test]
    fn the_cooldown_cap_evicts_stale_entries_first() {
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
    }
}
