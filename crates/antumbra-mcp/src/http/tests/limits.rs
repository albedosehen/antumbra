//! The bounds on what a client may hold open: sessions per identity, a
//! session's idle time, and the time a request has to begin its response.
//! The connection's own bound, on reading its headers, is in `shutdown`,
//! beside the other tests that need a real socket.

use super::*;
use std::time::Duration;

/// The handshake's first request, as a client sends it.
fn initialize(id: u32) -> String {
    serde_json::json!({
        "jsonrpc": "2.0", "id": id, "method": "initialize",
        "params": { "protocolVersion": "2025-06-18", "capabilities": {},
                    "clientInfo": { "name": "bounded", "version": "0" } }
    })
    .to_string()
}

/// Opens a session for `jwt`: its id.
async fn open(app: &Router, jwt: &str) -> anyhow::Result<String> {
    let resp = app
        .clone()
        .oneshot(mcp_request(jwt, None, initialize(1))?)
        .await?;
    anyhow::ensure!(
        resp.status() == StatusCode::OK,
        "initialize: {}",
        resp.status()
    );
    let Some(id) = resp
        .headers()
        .get(SESSION_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned)
    else {
        anyhow::bail!("a session id comes back with the initialize response");
    };
    Ok(id)
}

/// A `tools/list` on `session`.
fn list_tools(jwt: &str, session: &str) -> anyhow::Result<Request<Body>> {
    mcp_request(
        jwt,
        Some(session),
        serde_json::json!({"jsonrpc": "2.0", "id": 7, "method": "tools/list"}).to_string(),
    )
}

/// An identity's third session, when it may hold two, is refused with the
/// bound and the way past it; another identity is not held to its count; and
/// closing one of its sessions makes room for one.
#[tokio::test]
async fn a_session_past_an_identitys_bound_is_refused_until_one_closes() -> anyhow::Result<()> {
    let st = state_limited(Limits {
        sessions_per_identity: 2,
        ..Limits::default()
    })
    .await?;
    let app = router(st.clone());
    let jwt = format!("Bearer {}", token("ws:bounded", "user:bounded")?);

    let first = open(&app, &jwt).await?;
    let _second = open(&app, &jwt).await?;

    let refused = app
        .clone()
        .oneshot(mcp_request(&jwt, None, initialize(3))?)
        .await?;
    anyhow::ensure!(
        refused.status() == StatusCode::TOO_MANY_REQUESTS,
        "{}",
        refused.status()
    );
    anyhow::ensure!(
        refused.headers().get(header::RETRY_AFTER).is_some(),
        "says when to try again"
    );
    let body: serde_json::Value = serde_json::from_str(&body_text(refused).await?)?;
    anyhow::ensure!(body["id"] == 3, "the request's own id: {body}");
    let Some(message) = body["error"]["message"].as_str() else {
        anyhow::bail!("a JSON-RPC error with a message: {body}");
    };
    anyhow::ensure!(
        message.contains("the most it may hold (2)"),
        "names the bound: {message}"
    );
    anyhow::ensure!(
        message.contains("DELETE /mcp"),
        "names the way out: {message}"
    );

    let other = format!("Bearer {}", token("ws:bounded", "user:other")?);
    open(&app, &other).await?;

    let closed = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/mcp")
                .header(header::HOST, "localhost")
                .header(header::AUTHORIZATION, &jwt)
                .header(SESSION_HEADER, &first)
                .body(Body::empty())?,
        )
        .await?;
    anyhow::ensure!(closed.status().is_success(), "DELETE: {}", closed.status());
    open(&app, &jwt).await?;
    Ok(())
}

/// A session that carries no message for its idle time is closed: a request
/// on it is answered `404`, and it is no longer counted against its identity.
#[tokio::test]
async fn an_idle_session_is_closed() -> anyhow::Result<()> {
    let idle = Duration::from_millis(200);
    let st = state_limited(Limits {
        session_idle: idle,
        ..Limits::default()
    })
    .await?;
    let app = router(st.clone());
    let jwt = format!("Bearer {}", token("ws:idle", "user:idle")?);
    let session = open(&app, &jwt).await?;
    let initialized =
        serde_json::json!({"jsonrpc": "2.0", "method": "notifications/initialized"}).to_string();
    let resp = app
        .clone()
        .oneshot(mcp_request(&jwt, Some(&session), initialized)?)
        .await?;
    anyhow::ensure!(
        resp.status() == StatusCode::ACCEPTED,
        "initialized: {}",
        resp.status()
    );
    let identity = Identity {
        tenant: "ws:idle".into(),
        user: "user:idle".into(),
    };
    let open_now = st
        .service_for(&identity)
        .await?
        .manager
        .sessions
        .read()
        .await
        .len();
    anyhow::ensure!(open_now == 1, "one session is open: {open_now}");

    tokio::time::sleep(idle * 4).await;
    let gone = app.clone().oneshot(list_tools(&jwt, &session)?).await?;
    anyhow::ensure!(
        gone.status() == StatusCode::NOT_FOUND,
        "the session is unknown now: {}",
        gone.status()
    );
    let open_after = st
        .service_for(&identity)
        .await?
        .manager
        .sessions
        .read()
        .await
        .len();
    anyhow::ensure!(open_after == 0, "and counted no more: {open_after}");
    Ok(())
}

/// A request whose handler has not begun its response within the bound is
/// answered `408`; one that answers in time is answered as it was.
#[tokio::test]
async fn a_request_that_does_not_begin_its_response_in_time_is_answered_408() -> anyhow::Result<()>
{
    let app = bounded(
        Router::new()
            .route(
                "/late",
                axum::routing::get(|| async {
                    tokio::time::sleep(Duration::from_millis(400)).await;
                    "late"
                }),
            )
            .route("/prompt", axum::routing::get(|| async { "prompt" })),
        Duration::from_millis(100),
    );
    let late = app
        .clone()
        .oneshot(Request::builder().uri("/late").body(Body::empty())?)
        .await?;
    anyhow::ensure!(
        late.status() == StatusCode::REQUEST_TIMEOUT,
        "{}",
        late.status()
    );
    let prompt = app
        .oneshot(Request::builder().uri("/prompt").body(Body::empty())?)
        .await?;
    anyhow::ensure!(prompt.status() == StatusCode::OK, "{}", prompt.status());
    Ok(())
}
