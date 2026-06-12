//! The hosted **control plane** HTTP server: invite-gated signup, magic-link
//! login, and RS256 token issuance, run as its own service (separate from the
//! data-plane MCP server, which only *verifies* the tokens this server signs).
//!
//! The whole signup -> token -> login flow lives in `antumbra-control`; this
//! binary is the thin HTTP + config + mailer shell over it. Magic links are
//! delivered by a [`Mailer`]: a dev mailer that logs the link to stderr by
//! default, or a real SMTP mailer under `--features smtp` when `SMTP_*` is set.

use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::{Args, Parser, Subcommand};
use serde_json::json;

use antumbra_control::{authenticate, Issuer, MagicLink, Mailer};
use antumbra_store::{ConnectionConfig, Store, EMBED_DIM};

#[derive(Parser)]
#[command(
    name = "antumbra-control-server",
    about = "Antumbra hosted control plane"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the HTTP control plane (signup / login / magic-link verify).
    Serve(ServeArgs),
    /// Mint a single-use invite code and print it (hand it to a new user).
    MintInvite(StoreArgs),
}

#[derive(Args)]
struct StoreArgs {
    #[arg(
        long,
        env = "ANTUMBRA_DB_URL",
        default_value = "ws://127.0.0.1:8000/rpc"
    )]
    url: String,
    #[arg(long, env = "ANTUMBRA_DB_USER")]
    db_user: Option<String>,
    #[arg(long, env = "ANTUMBRA_DB_PASS")]
    db_pass: Option<String>,
}

#[derive(Args)]
struct ServeArgs {
    #[command(flatten)]
    store: StoreArgs,
    #[arg(long, default_value = "0.0.0.0:8090")]
    addr: String,
    /// External base URL the magic links point back to (this server).
    #[arg(long, env = "ANTUMBRA_CONTROL_BASE_URL")]
    base_url: String,
    /// Path to the RS256 PRIVATE key (PEM) issued tokens are signed with. The MCP
    /// server verifies with the matching public key (its `--jwt-public-key`).
    #[arg(long, env = "ANTUMBRA_SIGNING_KEY")]
    signing_key: String,
    /// Secret signing the short-lived magic-link tokens (distinct from the RS256 key).
    #[arg(long, env = "ANTUMBRA_MAGIC_SECRET")]
    magic_secret: String,
    /// Audience claim on issued tokens (must match the MCP server's verifier).
    #[arg(long, env = "ANTUMBRA_AUDIENCE", default_value = "antumbra")]
    audience: String,
    /// Issued bearer-token lifetime, in days.
    #[arg(long, default_value_t = 30)]
    token_ttl_days: u64,
    /// Magic-link lifetime, in minutes.
    #[arg(long, default_value_t = 15)]
    magic_ttl_mins: u64,
}

#[derive(Clone)]
struct AppState {
    store: Store,
    issuer: Arc<Issuer>,
    magic_secret: Arc<Vec<u8>>,
    base_url: Arc<String>,
    magic_ttl: Duration,
    mailer: Arc<dyn Mailer>,
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

/// The dev mailer: log the link to stderr so the passwordless flow is usable
/// without an email provider. Swap in [`SmtpMailer`] (under `--features smtp`).
struct StderrMailer;

impl Mailer for StderrMailer {
    fn send_link(&self, to: &str, link: &str) -> antumbra_core::Result<()> {
        eprintln!("[control-server dev mailer] magic link for {to}:\n  {link}");
        Ok(())
    }
}

#[cfg(feature = "smtp")]
struct SmtpMailer {
    transport: lettre::SmtpTransport,
    from: String,
}

#[cfg(feature = "smtp")]
impl SmtpMailer {
    fn from_env() -> anyhow::Result<Self> {
        use lettre::transport::smtp::authentication::Credentials;
        let host = std::env::var("SMTP_HOST")?;
        let from = std::env::var("SMTP_FROM")?;
        let mut relay = lettre::SmtpTransport::relay(&host)?;
        if let Ok(port) = std::env::var("SMTP_PORT") {
            relay = relay.port(port.parse()?);
        }
        if let (Ok(u), Ok(p)) = (std::env::var("SMTP_USER"), std::env::var("SMTP_PASS")) {
            relay = relay.credentials(Credentials::new(u, p));
        }
        Ok(Self {
            transport: relay.build(),
            from,
        })
    }
}

#[cfg(feature = "smtp")]
impl Mailer for SmtpMailer {
    fn send_link(&self, to: &str, link: &str) -> antumbra_core::Result<()> {
        use lettre::Transport;
        let oops = |e: String| antumbra_core::AntumbraError::other(format!("smtp: {e}"));
        let email = lettre::Message::builder()
            .from(self.from.parse().map_err(|e| oops(format!("from: {e}")))?)
            .to(to.parse().map_err(|e| oops(format!("to: {e}")))?)
            .subject("Your Antumbra sign-in link")
            .body(format!(
                "Sign in to Antumbra:\n\n{link}\n\nThe link expires shortly."
            ))
            .map_err(|e| oops(format!("build: {e}")))?;
        self.transport
            .send(&email)
            .map_err(|e| oops(format!("send: {e}")))?;
        Ok(())
    }
}

fn build_mailer() -> Arc<dyn Mailer> {
    #[cfg(feature = "smtp")]
    {
        // Treat an empty SMTP_HOST as unset, so a compose file can pass the
        // SMTP_* seam through with empty defaults without forcing SMTP on.
        if std::env::var("SMTP_HOST")
            .map(|h| !h.is_empty())
            .unwrap_or(false)
        {
            match SmtpMailer::from_env() {
                Ok(m) => {
                    eprintln!("antumbra-control-server: delivering magic links via SMTP");
                    return Arc::new(m);
                }
                Err(e) => {
                    eprintln!("antumbra-control-server: SMTP config failed ({e}); using dev mailer")
                }
            }
        }
    }
    eprintln!(
        "antumbra-control-server: DEV mailer -- links are logged to stderr \
         (build --features smtp and set SMTP_* for real email)"
    );
    Arc::new(StderrMailer)
}

async fn connect(url: &str, user: Option<&str>, pass: Option<&str>) -> anyhow::Result<Store> {
    let mut builder = ConnectionConfig::builder()
        .url(url)
        .namespace("antumbra")
        .database("main");
    if let (Some(u), Some(p)) = (user, pass) {
        builder = builder.username(u).password(p);
    }
    Ok(Store::connect(builder.build()?, EMBED_DIM).await?)
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

/// Mint a magic link and (try to) email it. The mailer send may block (SMTP), so
/// it runs off the async runtime.
async fn send_link(st: &AppState, email: String, invite: Option<String>) -> Response {
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
        Ok(Err(e)) => bad(&e.to_string()),
        Err(e) => internal(e),
    }
}

async fn signup_handler(State(st): State<AppState>, Json(req): Json<SignupReq>) -> Response {
    // Fail fast before emailing if the invite is plainly unusable (a race that
    // consumes it before verify is still caught at signup time).
    match antumbra_store::repo::invite::get(&st.store, &req.invite).await {
        Ok(Some(_)) => {}
        Ok(None) => return bad("unknown or already-used invite code"),
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
        Err(e) => bad(&e.to_string()),
    }
}

fn router(state: AppState) -> Router {
    Router::new()
        .route("/signup", post(signup_handler))
        .route("/login", post(login_handler))
        .route("/magic/verify", get(verify_handler))
        .route("/healthz", get(|| async { "ok" }))
        .with_state(state)
}

async fn serve(a: ServeArgs) -> anyhow::Result<()> {
    let store = connect(
        &a.store.url,
        a.store.db_user.as_deref(),
        a.store.db_pass.as_deref(),
    )
    .await?;
    let private_pem = std::fs::read(&a.signing_key)
        .map_err(|e| anyhow::anyhow!("read signing key {}: {e}", a.signing_key))?;
    let issuer = Issuer::new(
        private_pem,
        a.audience,
        Duration::from_secs(a.token_ttl_days * 86_400),
    );
    let state = AppState {
        store,
        issuer: Arc::new(issuer),
        magic_secret: Arc::new(a.magic_secret.into_bytes()),
        base_url: Arc::new(a.base_url),
        magic_ttl: Duration::from_secs(a.magic_ttl_mins * 60),
        mailer: build_mailer(),
    };
    let app = router(state);
    let listener = tokio::net::TcpListener::bind(&a.addr).await?;
    eprintln!(
        "antumbra-control-server: listening on http://{}/ (signup / login / magic)",
        a.addr
    );
    axum::serve(listener, app).await?;
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    match Cli::parse().cmd {
        Command::Serve(a) => serve(a).await,
        Command::MintInvite(s) => {
            let store = connect(&s.url, s.db_user.as_deref(), s.db_pass.as_deref()).await?;
            let code = antumbra_control::new_invite_code();
            antumbra_store::repo::invite::mint(&store, &code, None).await?;
            println!("{code}");
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use std::sync::Mutex;
    use tower::ServiceExt;

    // The control crate's throwaway RS256 private key (test-only).
    const RS_PRIV: &[u8] = include_bytes!("../../antumbra-control/tests/test_jwt_priv.pem");

    struct Capture(Mutex<Vec<String>>);
    impl Mailer for Capture {
        fn send_link(&self, _to: &str, link: &str) -> antumbra_core::Result<()> {
            self.0.lock().unwrap().push(link.to_string());
            Ok(())
        }
    }

    fn state_with(store: Store, capture: Arc<Capture>) -> AppState {
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
        }
    }

    #[tokio::test]
    async fn signup_then_verify_issues_a_token() {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        antumbra_store::repo::invite::mint(&store, "inv-1", None)
            .await
            .unwrap();
        let capture = Arc::new(Capture(Mutex::new(Vec::new())));
        let app = router(state_with(store, capture.clone()));

        // POST /signup -> a magic link is "sent" (captured).
        let resp = app
            .clone()
            .oneshot(
                Request::post("/signup")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"email":"a@b.com","invite":"inv-1"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let link = capture.0.lock().unwrap()[0].clone();
        let token = link.split("token=").nth(1).unwrap().to_string();

        // GET /magic/verify -> an issued bearer token.
        let resp = app
            .oneshot(
                Request::get(format!("/magic/verify?token={token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 64 * 1024)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(
            v["token"].as_str().unwrap_or_default().starts_with("ey"),
            "an issued JWT"
        );
    }

    #[tokio::test]
    async fn signup_with_unknown_invite_is_rejected_before_emailing() {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        let capture = Arc::new(Capture(Mutex::new(Vec::new())));
        let resp = router(state_with(store, capture.clone()))
            .oneshot(
                Request::post("/signup")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"email":"a@b.com","invite":"nope"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert!(
            capture.0.lock().unwrap().is_empty(),
            "no link emailed for a bad invite"
        );
    }
}
