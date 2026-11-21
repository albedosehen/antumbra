//! The transport tests: the auth boundary, the REST shim, the authenticated
//! initialize, and the live-propagation round trip over the real router.

use super::*;
use antumbra_core::testing::FixedEmbedder;
use antumbra_store::testing::mem_store;
use antumbra_store::EMBED_DIM;
use tower::ServiceExt; // oneshot

async fn state() -> anyhow::Result<Arc<HttpState>> {
    Ok(state_with(mem_store().await?).await)
}

async fn state_with(store: Store) -> Arc<HttpState> {
    state_embedding(store, Arc::new(FixedEmbedder::new(EMBED_DIM))).await
}

/// A server state over `store` whose default embedder is `embedder`.
pub(super) async fn state_embedding(store: Store, embedder: Arc<dyn Embedder>) -> Arc<HttpState> {
    state_archived(store, embedder, None).await
}

/// As [`state_embedding`], with a copal archive.
pub(super) async fn state_archived(
    store: Store,
    embedder: Arc<dyn Embedder>,
    copal: Option<Arc<antumbra_copal::CopalArchive>>,
) -> Arc<HttpState> {
    state_serving(store, embedder, copal, None, false).await
}

/// As [`state_archived`], with a PDF renderer.
pub(super) async fn state_printing(
    store: Store,
    embedder: Arc<dyn Embedder>,
    copal: Option<Arc<antumbra_copal::CopalArchive>>,
    pdf: Arc<crate::pdf::PdfRenderer>,
) -> Arc<HttpState> {
    state_serving(store, embedder, copal, Some(pdf), false).await
}

/// As [`state_archived`], serving each identity on a connection of its own, as
/// a remote deployment does, with `store` the owner connection alone. No
/// identity connects in these tests, so the address is never dialed.
pub(super) async fn state_per_identity(
    store: Store,
    embedder: Arc<dyn Embedder>,
    copal: Option<Arc<antumbra_copal::CopalArchive>>,
) -> Arc<HttpState> {
    state_serving(store, embedder, copal, None, true).await
}

/// A server state over a fresh store, under `limits` instead of the defaults.
async fn state_limited(limits: Limits) -> anyhow::Result<Arc<HttpState>> {
    Ok(state_bounded(
        mem_store().await?,
        Arc::new(FixedEmbedder::new(EMBED_DIM)),
        None,
        None,
        false,
        limits,
    )
    .await)
}

async fn state_serving(
    store: Store,
    embedder: Arc<dyn Embedder>,
    copal: Option<Arc<antumbra_copal::CopalArchive>>,
    pdf: Option<Arc<crate::pdf::PdfRenderer>>,
    per_identity: bool,
) -> Arc<HttpState> {
    state_bounded(store, embedder, copal, pdf, per_identity, Limits::default()).await
}

async fn state_bounded(
    store: Store,
    embedder: Arc<dyn Embedder>,
    copal: Option<Arc<antumbra_copal::CopalArchive>>,
    pdf: Option<Arc<crate::pdf::PdfRenderer>>,
    per_identity: bool,
    limits: Limits,
) -> Arc<HttpState> {
    let serving = if per_identity {
        Serving::PerIdentity {
            url: "ws://never-dialed.example:8000".into(),
            conns: Mutex::new(Bounded::new(MAX_SESSIONS)),
        }
    } else {
        Serving::Shared(store.clone())
    };
    Arc::new(HttpState {
        serving,
        store,
        host: "test".into(),
        verifier: JwtVerifier::hs256(b"test-secret"),
        embedder,
        embedders: Mutex::new(Bounded::new(MAX_EMBEDDERS)),
        auth: Arc::new(Mutex::new(())),
        auto_propose: None,
        recipe: antumbra_core::behavior::Recipe::default(),
        tune_confidence: false,
        behave_drift: "off".into(),
        serve: None,
        reranker: None,
        decider: None,
        copal,
        pdf,
        relics: crate::http::relics::Guards::new(),
        experts: crate::http::experts::Guards::new(),
        source: None,
        profile: None,
        github: None,
        revoked: std::sync::RwLock::new(std::collections::HashSet::new()),
        removed: std::sync::RwLock::new(std::collections::HashSet::new()),
        issuer: Some(b"test-secret".to_vec()),
        members_only: false,
        join_minute: std::sync::Mutex::new((std::time::Instant::now(), 0)),
        limits,
        sessions: Mutex::new(Bounded::new(MAX_SESSIONS)),
        servers: Mutex::new(Bounded::new(MAX_SESSIONS)),
        registry: crate::notify::PeerRegistry::new(),
        steward: Arc::default(),
    })
}

async fn status_for(auth: Option<&str>) -> anyhow::Result<StatusCode> {
    let mut builder = Request::builder().method("POST").uri("/mcp");
    if let Some(a) = auth {
        builder = builder.header(header::AUTHORIZATION, a);
    }
    let req = builder.body(Body::empty())?;
    Ok(router(state().await?).oneshot(req).await?.status())
}

// The auth boundary rejects before any store/session work, so these need no
// running socket and no provisioned tenant.
#[tokio::test]
async fn missing_token_is_unauthorized() -> anyhow::Result<()> {
    assert_eq!(status_for(None).await?, StatusCode::UNAUTHORIZED);
    Ok(())
}

#[tokio::test]
async fn garbage_token_is_unauthorized() -> anyhow::Result<()> {
    assert_eq!(
        status_for(Some("Bearer not-a-real-jwt")).await?,
        StatusCode::UNAUTHORIZED
    );
    Ok(())
}

#[tokio::test]
async fn non_bearer_scheme_is_unauthorized() -> anyhow::Result<()> {
    assert_eq!(
        status_for(Some("Basic dXNlcjpwYXNz")).await?,
        StatusCode::UNAUTHORIZED
    );
    Ok(())
}

/// The dashboard's page needs no token, since it carries no data, and serving
/// it opens nothing else: the tools behind it still refuse a call without one.
#[tokio::test]
async fn the_dashboard_is_public_and_the_tools_behind_it_are_not() -> anyhow::Result<()> {
    let app = router(state().await?);
    let page = Request::builder().uri("/dashboard").body(Body::empty())?;
    assert_eq!(app.clone().oneshot(page).await?.status(), StatusCode::OK);
    let call = Request::builder()
        .method("POST")
        .uri("/mcp/call")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(r#"{"tool":"workspace_stats"}"#))?;
    assert_eq!(app.oneshot(call).await?.status(), StatusCode::UNAUTHORIZED);
    Ok(())
}

fn token(tenant: &str, user: &str) -> anyhow::Result<String> {
    use jsonwebtoken::{encode, get_current_timestamp, Algorithm, EncodingKey, Header};
    #[derive(serde::Serialize)]
    struct C {
        tenant: String,
        user: String,
        exp: u64,
    }
    Ok(encode(
        &Header::new(Algorithm::HS256),
        &C {
            tenant: tenant.into(),
            user: user.into(),
            exp: get_current_timestamp() + 3600,
        },
        &EncodingKey::from_secret(b"test-secret"),
    )?)
}

fn valid_token() -> anyhow::Result<String> {
    token("ws:t", "user:t")
}

fn call_request(tok: Option<&str>, body: &str) -> anyhow::Result<Request<Body>> {
    let mut b = Request::builder().method("POST").uri("/mcp/call");
    if let Some(t) = tok {
        b = b.header(header::AUTHORIZATION, format!("Bearer {t}"));
    }
    Ok(b.body(Body::from(body.to_owned()))?)
}

// The REST shim: a single POST dispatches a tool under the caller's JWT
// identity -- no initialize handshake -- with the same auth + engine ACL.
#[tokio::test]
async fn rest_call_dispatches_a_tool_then_reads_it_back() -> anyhow::Result<()> {
    let st = state().await?;
    let tok = token("ws:rest", "user:rest")?;

    // Store a memory through POST /mcp/call.
    let store = call_request(
        Some(&tok),
        r#"{"tool":"store_memory","arguments":{"content":"the deno runtime"}}"#,
    )?;
    let resp = router(st.clone()).oneshot(store).await?;
    assert_eq!(resp.status(), StatusCode::OK);

    // Read it back the same way; the result carries the stored content.
    let list = call_request(Some(&tok), r#"{"tool":"list_memories","arguments":{}}"#)?;
    let resp = router(st.clone()).oneshot(list).await?;
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20).await?;
    assert!(String::from_utf8_lossy(&bytes).contains("deno"));

    // No token -> 401 (auth before any work).
    let resp = router(st.clone())
        .oneshot(call_request(
            None,
            r#"{"tool":"list_memories","arguments":{}}"#,
        )?)
        .await?;
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

    // Unknown tool / malformed body -> 400 (the caller's fault, not a panic).
    let resp = router(st.clone())
        .oneshot(call_request(
            Some(&tok),
            r#"{"tool":"nope","arguments":{}}"#,
        )?)
        .await?;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let resp = router(st)
        .oneshot(call_request(Some(&tok), "not json")?)
        .await?;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    Ok(())
}

/// The owner-side pattern the public relic read, the keepers and the
/// propagation watcher share: return the shared connection to the owner view
/// under the lock.
async fn owner_signin(st: &HttpState) -> anyhow::Result<()> {
    let _guard = st.auth.lock().await;
    st.store.signin_root().await?;
    Ok(())
}

/// What a `list_memories` answer must and must not carry for one caller.
fn only_own_rows(text: &str, mine: &str, theirs: &str) -> anyhow::Result<()> {
    anyhow::ensure!(text.contains(mine), "a caller's own row is missing: {text}");
    anyhow::ensure!(
        !text.contains(theirs),
        "a row of another caller's came back: {text}"
    );
    Ok(())
}

// On the shared (embedded) connection a tool call runs as its own caller for
// the whole call, while another identity's calls and owner-side signins take
// the same connection between calls: every answer carries only the caller's
// own rows. Through the servers directly, as the REST shim drives them.
//
// Two users of ONE workspace, each reading their private compartment: the
// repos filter by tenant themselves, so two tenants would stay apart even on
// an unbound connection, while a user's private compartment is hidden by the
// engine's row rule alone. A call that ran under the owner view, or under the
// other user's session, would show the other user's row or lose its own.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_call_on_the_shared_connection_runs_as_its_own_caller() -> anyhow::Result<()> {
    let st = state().await?;
    let a = Identity {
        tenant: "ws:shared".into(),
        user: "user:a".into(),
    };
    let b = Identity {
        tenant: "ws:shared".into(),
        user: "user:b".into(),
    };
    let server_a = st.rest_server_for(&a).await?;
    let server_b = st.rest_server_for(&b).await?;
    server_a
        .call_tool("store_memory", serde_json::json!({"content": "alpha only"}))
        .await?;
    server_b
        .call_tool("store_memory", serde_json::json!({"content": "beta only"}))
        .await?;

    const ROUNDS: usize = 40;
    let reads = |server: McpServer, mine: &'static str, theirs: &'static str| {
        tokio::spawn(async move {
            for _ in 0..ROUNDS {
                let out = server
                    .call_tool("list_memories", serde_json::json!({}))
                    .await?;
                only_own_rows(&out.to_string(), mine, theirs)?;
                tokio::task::yield_now().await;
            }
            anyhow::Ok(())
        })
    };
    let owner = {
        let st = st.clone();
        tokio::spawn(async move {
            for _ in 0..ROUNDS {
                owner_signin(&st).await?;
                tokio::task::yield_now().await;
            }
            anyhow::Ok(())
        })
    };
    let (ra, rb, ro) = tokio::join!(
        reads(server_a, "alpha only", "beta only"),
        reads(server_b, "beta only", "alpha only"),
        owner
    );
    ra??;
    rb??;
    ro??;
    Ok(())
}

/// A `/mcp` request for `session_id` (none for the first), as a client sends it.
fn mcp_request(jwt: &str, session_id: Option<&str>, body: String) -> anyhow::Result<Request<Body>> {
    let mut b = Request::builder()
        .method("POST")
        .uri("/mcp")
        .header(header::HOST, "localhost")
        .header(header::AUTHORIZATION, jwt)
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::ACCEPT, "application/json, text/event-stream");
    if let Some(id) = session_id {
        b = b.header("mcp-session-id", id);
    }
    Ok(b.body(Body::from(body))?)
}

/// The whole body of a response as text.
async fn body_text(resp: Response) -> anyhow::Result<String> {
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20).await?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

// The streamed path end to end: a `tools/call` carried by `POST /mcp` runs on
// the session's task after that request has been answered, so only a bind
// taken by the call itself can cover it. With another user's REST calls and
// owner-side signins racing it on the shared connection, each user still
// reads back only their own private row (two users of one workspace, for
// the reason the test above gives).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_streamed_call_on_the_shared_connection_runs_as_its_own_caller() -> anyhow::Result<()> {
    let st = state().await?;
    let app = router(st.clone());
    let a_tok = token("ws:stream", "user:a")?;
    let a_jwt = format!("Bearer {a_tok}");
    let b_tok = token("ws:stream", "user:b")?;

    // One row each, over the REST shim.
    for (tok, content) in [(&a_tok, "alpha only"), (&b_tok, "beta only")] {
        let body = serde_json::json!({"tool": "store_memory", "arguments": {"content": content}});
        let resp = app
            .clone()
            .oneshot(call_request(Some(tok), &body.to_string())?)
            .await?;
        anyhow::ensure!(resp.status() == StatusCode::OK, "seeding {content}");
    }

    // A's session: initialize, then initialized.
    let init = serde_json::json!({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": { "protocolVersion": "2025-06-18", "capabilities": {},
                    "clientInfo": { "name": "a", "version": "0" } }
    });
    let resp = app
        .clone()
        .oneshot(mcp_request(&a_jwt, None, init.to_string())?)
        .await?;
    anyhow::ensure!(resp.status() == StatusCode::OK, "initialize");
    let Some(session_id) = resp
        .headers()
        .get("mcp-session-id")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned)
    else {
        anyhow::bail!("stateful mode returns a session id");
    };
    let initialized = serde_json::json!({"jsonrpc": "2.0", "method": "notifications/initialized"});
    let resp = app
        .clone()
        .oneshot(mcp_request(
            &a_jwt,
            Some(&session_id),
            initialized.to_string(),
        )?)
        .await?;
    anyhow::ensure!(resp.status() == StatusCode::ACCEPTED, "initialized");

    for round in 0..12u32 {
        let call = serde_json::json!({
            "jsonrpc": "2.0", "id": 10 + round, "method": "tools/call",
            "params": { "name": "list_memories", "arguments": {} }
        });
        let a_read = app
            .clone()
            .oneshot(mcp_request(&a_jwt, Some(&session_id), call.to_string())?);
        let b_read = app.clone().oneshot(call_request(
            Some(&b_tok),
            r#"{"tool":"list_memories","arguments":{}}"#,
        )?);
        let (a_resp, b_resp, owner) = tokio::join!(a_read, b_read, owner_signin(&st));
        owner?;
        let a_resp = a_resp?;
        anyhow::ensure!(a_resp.status() == StatusCode::OK, "round {round}: A's call");
        only_own_rows(&body_text(a_resp).await?, "alpha only", "beta only")?;
        let b_resp = b_resp?;
        anyhow::ensure!(b_resp.status() == StatusCode::OK, "round {round}: B's call");
        only_own_rows(&body_text(b_resp).await?, "beta only", "alpha only")?;
    }
    Ok(())
}

/// The REST shim takes a body the size of a real page: a knowledge document or
/// a relic's HTML past the old 64 KiB ceiling lands, and a body past the new
/// ceiling is refused as the caller's fault rather than read.
#[tokio::test]
async fn rest_calls_take_a_page_sized_body_and_refuse_a_larger_one() -> anyhow::Result<()> {
    let st = state().await?;
    let tok = token("ws:big", "user:big")?;
    let content = "The ceiling pays for the floor. ".repeat(4_000);
    assert!(content.len() > 100 * 1024);
    let ingest = serde_json::json!({
        "tool": "ingest_document",
        "arguments": { "title": "A long essay", "content": content },
    })
    .to_string();
    let resp = router(st.clone())
        .oneshot(call_request(Some(&tok), &ingest)?)
        .await?;
    assert_eq!(resp.status(), StatusCode::OK);

    let huge = format!(
        r#"{{"tool":"ingest_document","arguments":{{"title":"x","content":"{}"}}}}"#,
        "x".repeat(MAX_CALL_BODY)
    );
    let resp = router(st).oneshot(call_request(Some(&tok), &huge)?).await?;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    Ok(())
}

/// A user id names one user in one workspace: a token whose user already
/// belongs to another workspace is refused with the reason, not a server
/// error, and without naming the other workspace.
#[tokio::test]
async fn a_user_who_belongs_to_another_workspace_is_refused_with_the_reason() -> anyhow::Result<()>
{
    let st = state().await?;
    let first = token("ws:first", "user:same")?;
    let list = r#"{"tool":"list_memories","arguments":{}}"#;
    let resp = router(st.clone())
        .oneshot(call_request(Some(&first), list)?)
        .await?;
    assert_eq!(resp.status(), StatusCode::OK);

    let second = token("ws:second", "user:same")?;
    let resp = router(st)
        .oneshot(call_request(Some(&second), list)?)
        .await?;
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    let body = axum::body::to_bytes(resp.into_body(), 1 << 20).await?;
    let body = String::from_utf8_lossy(&body);
    assert!(
        body.contains("already belongs to another workspace"),
        "{body}"
    );
    assert!(!body.contains("ws:first"), "{body}");
    Ok(())
}

/// An identity's REST calls share one server: provisioned, and this node
/// registered, on the first call only. Built per call, each one signed in as
/// root and re-registered the node under the owner lock, and the node's
/// registration time moved with every call.
#[tokio::test]
async fn rest_calls_provision_the_identity_once() -> anyhow::Result<()> {
    let st = state().await?;
    let tok = token("ws:once", "user:once")?;
    let registered = || async {
        st.store.signin_root().await?;
        let nodes = antumbra_store::repo::device::list_for_user(
            &st.store,
            &TenantId::new("ws:once"),
            &UserId::new("user:once"),
        )
        .await?;
        anyhow::Ok(nodes.into_iter().map(|d| d.updated_at).collect::<Vec<_>>())
    };
    let call = || call_request(Some(&tok), r#"{"tool":"list_memories","arguments":{}}"#);

    let resp = router(st.clone()).oneshot(call()?).await?;
    assert_eq!(resp.status(), StatusCode::OK);
    let first = registered().await?;
    assert_eq!(first.len(), 1, "the first call registers this node");

    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    for _ in 0..3 {
        let resp = router(st.clone()).oneshot(call()?).await?;
        assert_eq!(resp.status(), StatusCode::OK);
    }
    assert_eq!(
        registered().await?,
        first,
        "later calls register nothing again"
    );
    assert_eq!(st.servers.lock().await.map.len(), 1);
    Ok(())
}

// The happy path: a valid token verifies, the shared connection signs in as
// the identity, and rmcp dispatches `initialize` to the McpServer, proving
// the full auth -> signin -> handle wiring end to end (provisioning the
// identity on first contact). No socket: the router is driven via oneshot.
#[tokio::test]
async fn valid_token_reaches_the_service() -> anyhow::Result<()> {
    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": { "name": "test", "version": "0" }
        }
    })
    .to_string();
    let req = Request::builder()
        .method("POST")
        .uri("/mcp")
        // A real HTTP client always sends Host; oneshot does not synthesize it.
        .header(header::HOST, "localhost")
        .header(header::AUTHORIZATION, format!("Bearer {}", valid_token()?))
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::ACCEPT, "application/json, text/event-stream")
        .body(Body::from(body))?;
    let resp = router(state().await?).oneshot(req).await?;
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20).await?;
    let text = String::from_utf8_lossy(&bytes);
    assert_eq!(status, StatusCode::OK, "reached service? body: {text}");
    // Stateless mode may answer as JSON or a single SSE `data:` line; either
    // way the initialize result (serverInfo) must be present.
    assert!(
        text.contains("serverInfo") || text.contains("protocolVersion"),
        "expected an initialize result, got: {text}"
    );
    Ok(())
}

// Live propagation's last mile, over the wire: a grantee (B) holds an open MCP SSE
// stream; when A writes into the shared compartment, B's stream receives the
// `antumbra/memory_changed` notification. Drives the real `/mcp` router
// through the full stateful handshake (initialize -> initialized -> GET SSE),
// so it exercises the actual transport path, peer capture, the store's
// announced write, audience resolution, and push. The store is embedded, so
// propagation follows its announced writes, as the server does there.
#[tokio::test]
async fn live_notification_reaches_a_grantees_stream() -> anyhow::Result<()> {
    use antumbra_core::{
        Capability, Compartment, CompartmentId, Grant, Memory, MemoryNetwork, Origin, UserId,
    };
    use antumbra_store::repo::{compartment, memory};
    use antumbra_store::testing::members;
    use futures::StreamExt;

    let mut store = mem_store().await?;
    let announced = store.announce_changes();
    let state = state_with(store).await;
    let tenant = TenantId::new("ws:t");
    let alice = UserId::new("user:a");
    let bob = UserId::new("user:b");
    let comp = CompartmentId::new("comp:shared");
    let now = chrono::Utc::now();

    // Owner-mode setup: both principals, a compartment alice owns, bob granted.
    members(&state.store, &tenant, &[&alice, &bob]).await?;
    compartment::create(
        &state.store,
        &Compartment {
            id: comp.clone(),
            tenant: tenant.clone(),
            owner: alice.clone(),
            name: "shared".into(),
            origin: Origin::User,
            created_at: now,
            updated_at: now,
            deleted_at: None,
        },
    )
    .await?;
    compartment::grant(
        &state.store,
        &Grant {
            tenant: tenant.clone(),
            compartment: comp.clone(),
            grantee: bob.clone(),
            capability: Capability::Reference,
            granted_by: alice.clone(),
            created_at: now,
            updated_at: now,
            deleted_at: None,
        },
    )
    .await?;

    // Start live propagation, following the store's announced writes.
    spawn_live_propagation(state.clone(), Some(announced));

    let app = router(state.clone());
    let bob_jwt = format!("Bearer {}", token("ws:t", "user:b")?);

    // 1. initialize -> capture the session id rmcp assigns.
    let init = serde_json::json!({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": { "protocolVersion": "2025-06-18", "capabilities": {},
                    "clientInfo": { "name": "bob", "version": "0" } }
    })
    .to_string();
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header(header::HOST, "localhost")
                .header(header::AUTHORIZATION, &bob_jwt)
                .header(header::CONTENT_TYPE, "application/json")
                .header(header::ACCEPT, "application/json, text/event-stream")
                .body(Body::from(init))?,
        )
        .await?;
    assert_eq!(resp.status(), StatusCode::OK);
    let Some(session_id) = resp
        .headers()
        .get("mcp-session-id")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned)
    else {
        anyhow::bail!("stateful mode returns a session id");
    };

    // 2. notifications/initialized -> triggers on_initialized, capturing bob's peer.
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header(header::HOST, "localhost")
                .header(header::AUTHORIZATION, &bob_jwt)
                .header(header::CONTENT_TYPE, "application/json")
                .header(header::ACCEPT, "application/json, text/event-stream")
                .header("mcp-session-id", &session_id)
                .body(Body::from(
                    serde_json::json!({"jsonrpc":"2.0","method":"notifications/initialized"})
                        .to_string(),
                ))?,
        )
        .await?;
    assert_eq!(resp.status(), StatusCode::ACCEPTED);

    // 3. GET -> open bob's server->client SSE stream; hold its body.
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/mcp")
                .header(header::HOST, "localhost")
                .header(header::AUTHORIZATION, &bob_jwt)
                .header(header::ACCEPT, "text/event-stream")
                .header("mcp-session-id", &session_id)
                .body(Body::empty())?,
        )
        .await?;
    assert_eq!(resp.status(), StatusCode::OK, "GET opens the SSE stream");
    let mut stream = resp.into_body().into_data_stream();

    // A write into the shared compartment (owner mode), the change alice makes.
    {
        let _guard = state.auth.lock().await;
        state.store.invalidate().await?;
        let m = Memory::new(
            "cccccccc-0000-0000-0000-00000000000c",
            tenant.clone(),
            MemoryNetwork::World,
            "shared via SSE",
            0.9,
            now,
        )
        .in_compartment(comp.clone());
        memory::upsert(&state.store, &m).await?;
    }

    // 4. The notification arrives on bob's SSE stream.
    let mut seen = String::new();
    let found = match tokio::time::timeout(std::time::Duration::from_secs(8), async {
        while let Some(chunk) = stream.next().await {
            let bytes = chunk?;
            seen.push_str(&String::from_utf8_lossy(&bytes));
            if seen.contains("antumbra/memory_changed") {
                return anyhow::Ok(true);
            }
        }
        anyhow::Ok(false)
    })
    .await
    {
        Ok(found) => found?,
        Err(_timed_out) => false,
    };
    assert!(found, "grantee's SSE stream got the change; saw: {seen}");
    Ok(())
}

mod experts;
mod limits;
mod relics;
mod shutdown;
mod sync;
mod telemetry;
mod written_from;

/// A token the operator revoked is refused like an expired one, on every
/// route that takes a token, once the denylist has been read; the same token
/// passed before.
#[tokio::test]
async fn a_revoked_token_is_refused_like_an_expired_one() -> anyhow::Result<()> {
    use antumbra_store::repo::revocation;
    let st = state().await?;
    // The real mint, not the tests' bare helper: a revocation names the id a
    // minted token carries.
    let tok = antumbra_auth::mint_hs256(
        b"test-secret",
        "ws:rev",
        "user:rev",
        std::time::Duration::from_secs(3600),
    )?;
    let Some(jti) = antumbra_auth::unverified::jti(&tok) else {
        anyhow::bail!("a minted token carries an id");
    };
    let list = r#"{"tool":"list_memories","arguments":{}}"#;
    let before = router(st.clone())
        .oneshot(call_request(Some(&tok), list)?)
        .await?
        .status();
    anyhow::ensure!(
        before != StatusCode::UNAUTHORIZED,
        "the token passed before: {before}"
    );

    {
        let _guard = st.auth.lock().await;
        st.store.signin_root().await?;
        revocation::revoke(
            &st.store,
            &jti,
            chrono::Utc::now() + chrono::Duration::days(1),
            Some("laptop lost".into()),
        )
        .await?;
    }
    anyhow::ensure!(super::revocation::refresh(&st).await? == 1);

    let rest = router(st.clone())
        .oneshot(call_request(Some(&tok), list)?)
        .await?
        .status();
    anyhow::ensure!(rest == StatusCode::UNAUTHORIZED, "/mcp/call: {rest}");
    let stream = router(st.clone())
        .oneshot(mcp_request(&tok, None, "{}".into())?)
        .await?
        .status();
    anyhow::ensure!(stream == StatusCode::UNAUTHORIZED, "/mcp: {stream}");
    let console = router(st.clone())
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/console/snapshot")
                .header(header::AUTHORIZATION, format!("Bearer {tok}"))
                .body(Body::empty())?,
        )
        .await?
        .status();
    anyhow::ensure!(
        console == StatusCode::UNAUTHORIZED,
        "/console/snapshot: {console}"
    );
    Ok(())
}

/// The policy's knobs are read with a token and not without one, and every
/// knob the policy has is among them with a range the default sits in.
#[tokio::test]
async fn the_policys_knobs_are_named_with_their_ranges() -> anyhow::Result<()> {
    let app = router(state().await?);
    let ask = |tok: Option<String>| -> anyhow::Result<Request<Body>> {
        let mut req = Request::builder().method("GET").uri("/console/knobs");
        if let Some(tok) = tok {
            req = req.header(header::AUTHORIZATION, format!("Bearer {tok}"));
        }
        Ok(req.body(Body::empty())?)
    };
    let refused = app.clone().oneshot(ask(None)?).await?.status();
    anyhow::ensure!(refused == StatusCode::UNAUTHORIZED, "no token: {refused}");

    let res = app.oneshot(ask(Some(valid_token()?))?).await?;
    anyhow::ensure!(res.status() == StatusCode::OK, "{}", res.status());
    let body: serde_json::Value = serde_json::from_str(&body_text(res).await?)?;
    let knobs = body["knobs"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("no knobs: {body}"))?;
    let policy = serde_json::to_value(antumbra_core::MemoryPolicy::default())?;
    let fields = policy
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("the policy is not an object"))?;
    anyhow::ensure!(knobs.len() == fields.len(), "{} knobs", knobs.len());
    for knob in knobs {
        let key = knob["key"].as_str().unwrap_or_default();
        anyhow::ensure!(fields.contains_key(key), "unknown knob {key}");
        let (min, max) = (knob["min"].as_f64(), knob["max"].as_f64());
        anyhow::ensure!(min < max, "{key}: {min:?}..{max:?}");
        anyhow::ensure!(knob["step"].as_f64() > Some(0.0), "{key}: no step");
        anyhow::ensure!(!knob["help"].as_str().unwrap_or_default().is_empty());
    }
    Ok(())
}

/// A member says who they are; the workspace's other members see it, another
/// workspace does not, and nothing a member sends changes anyone's profile
/// but their own. What may be an image is decided by its bytes.
#[tokio::test]
async fn a_profile_is_its_own_users_to_write_and_the_workspaces_to_read() -> anyhow::Result<()> {
    use base64::Engine as _;
    let st = state().await?;
    let (dana, eli, other) = (
        token("ws:team", "user:dana")?,
        token("ws:team", "user:eli")?,
        token("ws:elsewhere", "user:fay")?,
    );
    let ask = |tok: &str, tool: &str, arguments: serde_json::Value| {
        let (st, tok) = (st.clone(), tok.to_string());
        let body = serde_json::json!({ "tool": tool, "arguments": arguments }).to_string();
        async move {
            let res = router(st).oneshot(call_request(Some(&tok), &body)?).await?;
            let status = res.status();
            let text = body_text(res).await?;
            anyhow::Ok((status, serde_json::from_str::<serde_json::Value>(&text)?))
        }
    };
    let b64 = |bytes: &[u8]| base64::engine::general_purpose::STANDARD.encode(bytes);
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    png.extend_from_slice(&[7; 64]);

    let (status, set) = ask(
        &dana,
        "set_profile",
        serde_json::json!({ "display": "  Dana   Whitfield ", "image": b64(&png) }),
    )
    .await?;
    anyhow::ensure!(status == StatusCode::OK, "{status}: {set}");
    anyhow::ensure!(
        set["display"] == "Dana Whitfield" && set["has_image"] == true,
        "{set}"
    );

    // Eli, in the same workspace, sees Dana with her image as it was sent.
    let (_, seen) = ask(&eli, "profiles", serde_json::json!({ "images": true })).await?;
    anyhow::ensure!(seen["you"] == "user:eli", "{seen}");
    let listed = seen["profiles"].as_array().cloned().unwrap_or_default();
    anyhow::ensure!(
        listed.len() == 1 && listed[0]["user"] == "user:dana",
        "{seen}"
    );
    let image = format!("data:image/png;base64,{}", b64(&png));
    anyhow::ensure!(listed[0]["image"] == image.as_str(), "{seen}");
    // Without asking for images, only that there is one.
    let (_, brief) = ask(&eli, "profiles", serde_json::json!({})).await?;
    anyhow::ensure!(brief["profiles"][0]["image"].is_null(), "{brief}");
    anyhow::ensure!(brief["profiles"][0]["has_image"] == true, "{brief}");

    // Eli's own write is Eli's: Dana's stands as it was.
    let (status, _) = ask(&eli, "set_profile", serde_json::json!({ "display": "Eli" })).await?;
    anyhow::ensure!(status == StatusCode::OK);
    let (_, both) = ask(&dana, "profiles", serde_json::json!({})).await?;
    let names: Vec<&str> = both["profiles"]
        .as_array()
        .map(|all| all.iter().filter_map(|p| p["display"].as_str()).collect())
        .unwrap_or_default();
    anyhow::ensure!(names == ["Dana Whitfield", "Eli"], "{both}");

    // Another workspace sees neither.
    let (_, none) = ask(&other, "profiles", serde_json::json!({ "images": true })).await?;
    anyhow::ensure!(
        none["profiles"].as_array().is_some_and(Vec::is_empty),
        "{none}"
    );

    // A name alone leaves the image; clearing the image leaves the name.
    let (_, renamed) = ask(
        &dana,
        "set_profile",
        serde_json::json!({ "display": "Dana" }),
    )
    .await?;
    anyhow::ensure!(
        renamed["has_image"] == true && renamed["display"] == "Dana",
        "{renamed}"
    );
    let (_, cleared) = ask(
        &dana,
        "set_profile",
        serde_json::json!({ "clear_image": true }),
    )
    .await?;
    anyhow::ensure!(
        cleared["has_image"] == false && cleared["display"] == "Dana",
        "{cleared}"
    );

    // What is not an image of a kind a profile holds is refused, whatever it
    // is called, and so is one too large.
    for (what, sent) in [
        (
            "an svg",
            format!("data:image/png;base64,{}", b64(b"<svg onload='x()'></svg>")),
        ),
        ("not base64", "!!!".to_string()),
        (
            "too large",
            b64(&[png.clone(), vec![0; 300 * 1024]].concat()),
        ),
    ] {
        let (status, why) = ask(&dana, "set_profile", serde_json::json!({ "image": sent })).await?;
        anyhow::ensure!(status == StatusCode::BAD_REQUEST, "{what}: {status} {why}");
    }
    let (_, after) = ask(&dana, "profiles", serde_json::json!({})).await?;
    anyhow::ensure!(after["profiles"][0]["has_image"] == false, "{after}");
    Ok(())
}

/// A workspace's first member owns it. Roles are the owner's to change, an
/// admin removes a member and not the owner, a removed member's token is
/// refused on every route at once, and the workspace keeps an owner.
#[tokio::test]
async fn members_are_listed_and_changed_by_role() -> anyhow::Result<()> {
    let st = state().await?;
    let (ada, bo, cy) = (
        token("ws:crew", "user:ada")?,
        token("ws:crew", "user:bo")?,
        token("ws:crew", "user:cy")?,
    );
    let ask = |tok: &str, method: &str, path: &str, body: serde_json::Value| {
        let (st, tok, method, path) = (
            st.clone(),
            tok.to_string(),
            method.to_string(),
            path.to_string(),
        );
        async move {
            let req = Request::builder()
                .method(method.as_str())
                .uri(path)
                .header(header::AUTHORIZATION, format!("Bearer {tok}"))
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body.to_string()))?;
            let res = router(st).oneshot(req).await?;
            let status = res.status();
            let text = body_text(res).await?;
            anyhow::Ok((
                status,
                serde_json::from_str::<serde_json::Value>(&text).unwrap_or_default(),
            ))
        }
    };
    let none = serde_json::Value::Null;
    let roles = |listed: &serde_json::Value| -> Vec<(String, String)> {
        listed["members"]
            .as_array()
            .map(|all| {
                all.iter()
                    .map(|m| {
                        (
                            m["user"].as_str().unwrap_or_default().to_string(),
                            m["role"].as_str().unwrap_or_default().to_string(),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default()
    };
    let pair = |user: &str, role: &str| (user.to_string(), role.to_string());

    // In the order they first call: the first owns the workspace.
    let (status, first) = ask(&ada, "GET", "/members", none.clone()).await?;
    anyhow::ensure!(status == StatusCode::OK, "{status} {first}");
    anyhow::ensure!(
        first["your_role"] == "owner" && first["you"] == "user:ada",
        "{first}"
    );
    let (_, second) = ask(&bo, "GET", "/members", none.clone()).await?;
    anyhow::ensure!(second["your_role"] == "member", "{second}");
    ask(&cy, "GET", "/members", none.clone()).await?;

    // A member changes no one's role; the owner makes an admin.
    let change = |user: &str, role: &str| serde_json::json!({ "user": user, "role": role });
    let (status, why) = ask(&bo, "POST", "/members/role", change("user:cy", "admin")).await?;
    anyhow::ensure!(status == StatusCode::FORBIDDEN, "{status} {why}");
    let (status, _) = ask(&ada, "POST", "/members/role", change("user:bo", "admin")).await?;
    anyhow::ensure!(status == StatusCode::OK);
    let (status, _) = ask(&ada, "POST", "/members/role", change("user:bo", "captain")).await?;
    anyhow::ensure!(
        status == StatusCode::BAD_REQUEST,
        "an unknown role: {status}"
    );

    // An admin removes a member, and not the owner.
    let who = |user: &str| serde_json::json!({ "user": user });
    let (status, _) = ask(&bo, "POST", "/members/remove", who("user:ada")).await?;
    anyhow::ensure!(
        status == StatusCode::FORBIDDEN,
        "an admin removed the owner: {status}"
    );
    let (status, _) = ask(&bo, "POST", "/members/remove", who("user:cy")).await?;
    anyhow::ensure!(status == StatusCode::OK);

    // The removed member's token is refused at once, on every route.
    let (status, _) = ask(&cy, "GET", "/members", none.clone()).await?;
    anyhow::ensure!(status == StatusCode::UNAUTHORIZED, "/members: {status}");
    let call = router(st.clone())
        .oneshot(call_request(
            Some(&cy),
            r#"{"tool":"workspace_stats","arguments":{}}"#,
        )?)
        .await?
        .status();
    anyhow::ensure!(call == StatusCode::UNAUTHORIZED, "/mcp/call: {call}");
    let (_, listed) = ask(&ada, "GET", "/members", none.clone()).await?;
    anyhow::ensure!(
        roles(&listed) == [pair("user:ada", "owner"), pair("user:bo", "admin")],
        "{listed}"
    );

    // The workspace keeps an owner.
    let (status, why) = ask(&ada, "POST", "/members/role", change("user:ada", "member")).await?;
    anyhow::ensure!(status == StatusCode::FORBIDDEN, "{status} {why}");
    anyhow::ensure!(
        why["error"]
            .as_str()
            .is_some_and(|w| w.contains("keeps an owner")),
        "{why}"
    );

    // Let back in by an admin, as a member, and served again.
    let (status, _) = ask(&bo, "POST", "/members/readmit", who("user:cy")).await?;
    anyhow::ensure!(status == StatusCode::OK);
    let (status, back) = ask(&cy, "GET", "/members", none.clone()).await?;
    anyhow::ensure!(
        status == StatusCode::OK && back["your_role"] == "member",
        "{status} {back}"
    );

    // Another workspace's member sees none of this and changes none of it.
    let other = token("ws:elsewhere", "user:dee")?;
    let (_, theirs) = ask(&other, "GET", "/members", none.clone()).await?;
    anyhow::ensure!(roles(&theirs) == [pair("user:dee", "owner")], "{theirs}");
    let (status, _) = ask(&other, "POST", "/members/remove", who("user:bo")).await?;
    anyhow::ensure!(
        status == StatusCode::FORBIDDEN,
        "across workspaces: {status}"
    );
    let (status, _) = ask(&ada, "GET", "/members", none).await?;
    anyhow::ensure!(status == StatusCode::OK);

    // No token, no list.
    let bare = Request::builder().uri("/members").body(Body::empty())?;
    anyhow::ensure!(router(st).oneshot(bare).await?.status() == StatusCode::UNAUTHORIZED);
    Ok(())
}

/// Joining: an admin makes a code; someone with no token asks with it,
/// choosing a handle; the admins see the request and approve it; the token is
/// collected once with the secret, and is the new member's. A code that will
/// not do, a taken or reserved handle, a wrong secret and a member who is not
/// an admin are each refused.
#[tokio::test]
async fn someone_asks_to_join_and_is_let_in_by_an_admin() -> anyhow::Result<()> {
    let st = state().await?;
    let ada = token("ws:crew", "user:ada")?;
    let send = |tok: Option<&str>, method: &str, path: &str, body: serde_json::Value| {
        let (st, tok, method, path) = (
            st.clone(),
            tok.map(str::to_string),
            method.to_string(),
            path.to_string(),
        );
        async move {
            let mut req = Request::builder()
                .method(method.as_str())
                .uri(path)
                .header(header::CONTENT_TYPE, "application/json");
            if let Some(tok) = tok {
                req = req.header(header::AUTHORIZATION, format!("Bearer {tok}"));
            }
            let res = router(st)
                .oneshot(req.body(Body::from(body.to_string()))?)
                .await?;
            let status = res.status();
            let text = body_text(res).await?;
            anyhow::Ok((
                status,
                serde_json::from_str::<serde_json::Value>(&text).unwrap_or_default(),
            ))
        }
    };
    let none = serde_json::Value::Null;
    let digest = antumbra_core::join::digest;

    // The workspace's first caller owns it, and makes a code for two people.
    let (status, made) = send(
        Some(&ada),
        "POST",
        "/join/codes",
        serde_json::json!({ "uses": 2, "label": "the team" }),
    )
    .await?;
    anyhow::ensure!(status == StatusCode::OK, "{status} {made}");
    let code = made["code"].as_str().unwrap_or_default().to_string();
    anyhow::ensure!(
        code.starts_with("aj1_") && made["uses_left"] == 2 && made["needs_approval"] == true,
        "{made}"
    );
    // The list holds its digest and never the code.
    let (_, listed) = send(Some(&ada), "GET", "/join/codes", none.clone()).await?;
    anyhow::ensure!(
        !listed.to_string().contains(&code),
        "the code is kept in the clear"
    );
    anyhow::ensure!(
        listed["codes"][0]["key"] == digest(&code).as_str(),
        "{listed}"
    );

    // Asking needs no token, and a code that will not do says nothing of why.
    let ask = |code: &str, handle: &str, secret: &str| {
        serde_json::json!({
            "code": code, "handle": handle, "display": "Bodil Lindqvist",
            "note": "I work on the checkout service.", "host": "Bodils-Laptop",
            "secret_digest": antumbra_core::join::digest(secret),
        })
    };
    let (status, why) = send(None, "POST", "/join/request", ask("aj1_nope", "bodil", "s")).await?;
    anyhow::ensure!(status == StatusCode::FORBIDDEN, "{status} {why}");
    anyhow::ensure!(
        why["error"] == antumbra_store::repo::join::NO_SUCH_CODE,
        "{why}"
    );
    for handle in ["default", "ada", "x"] {
        let (status, _) = send(None, "POST", "/join/request", ask(&code, handle, "s")).await?;
        anyhow::ensure!(status == StatusCode::FORBIDDEN, "{handle}: {status}");
    }
    let (status, asked) = send(
        None,
        "POST",
        "/join/request",
        ask(&code, "Bodil", "bodils-secret"),
    )
    .await?;
    anyhow::ensure!(status == StatusCode::OK, "{status} {asked}");
    anyhow::ensure!(
        asked["standing"] == "pending" && asked["handle"] == "bodil",
        "{asked}"
    );
    let request = asked["request"].as_str().unwrap_or_default().to_string();

    // Waiting: no token yet, and none for a wrong secret or with none at all.
    let collect = |secret: &str| serde_json::json!({ "request": request, "secret": secret });
    let (_, waiting) = send(None, "POST", "/join/collect", collect("bodils-secret")).await?;
    anyhow::ensure!(
        waiting["standing"] == "pending" && waiting["token"].is_null(),
        "{waiting}"
    );
    let (status, _) = send(None, "POST", "/join/collect", collect("a-guess")).await?;
    anyhow::ensure!(status == StatusCode::FORBIDDEN, "a wrong secret: {status}");

    // The admins see who asks: the handle, the name, the note, the machine.
    let (_, queue) = send(Some(&ada), "GET", "/join/requests", none.clone()).await?;
    anyhow::ensure!(queue["waiting"] == 1, "{queue}");
    let seen = &queue["requests"][0];
    anyhow::ensure!(
        seen["handle"] == "bodil"
            && seen["display"] == "Bodil Lindqvist"
            && seen["host"] == "bodils-laptop",
        "{seen}"
    );
    anyhow::ensure!(
        !queue.to_string().contains(&digest("bodils-secret")),
        "the secret's digest is shown"
    );

    // Approved by the owner; the token is given once and is Bodil's.
    let (status, _) = send(
        Some(&ada),
        "POST",
        "/join/decide",
        serde_json::json!({ "key": request, "approve": true }),
    )
    .await?;
    anyhow::ensure!(status == StatusCode::OK);
    let (_, got) = send(None, "POST", "/join/collect", collect("bodils-secret")).await?;
    anyhow::ensure!(
        got["standing"] == "approved"
            && got["user"] == "user:bodil"
            && got["workspace"] == "ws:crew",
        "{got}"
    );
    let bodil = got["token"].as_str().unwrap_or_default().to_string();
    let (_, again) = send(None, "POST", "/join/collect", collect("bodils-secret")).await?;
    anyhow::ensure!(
        again["standing"] == "collected" && again["token"].is_null(),
        "{again}"
    );

    // The token works, as a member, with the name asked under.
    let (status, members) = send(Some(&bodil), "GET", "/members", none.clone()).await?;
    anyhow::ensure!(
        status == StatusCode::OK && members["your_role"] == "member",
        "{status} {members}"
    );
    let hers = members["members"]
        .as_array()
        .and_then(|all| all.iter().find(|m| m["user"] == "user:bodil").cloned());
    anyhow::ensure!(
        hers.is_some_and(|m| m["display"] == "Bodil Lindqvist"),
        "{members}"
    );
    // A member lets no one in, and reads no one's requests.
    for (method, path) in [
        ("POST", "/join/codes"),
        ("GET", "/join/codes"),
        ("GET", "/join/requests"),
    ] {
        let (status, _) = send(Some(&bodil), method, path, serde_json::json!({})).await?;
        anyhow::ensure!(status == StatusCode::FORBIDDEN, "{method} {path}: {status}");
    }

    // A refusal is told with its reason; a revoked code admits no one.
    let (_, second) = send(None, "POST", "/join/request", ask(&code, "cyrus", "s2")).await?;
    let key = second["request"].clone();
    send(
        Some(&ada),
        "POST",
        "/join/decide",
        serde_json::json!({ "key": key, "approve": false, "reason": "not this quarter" }),
    )
    .await?;
    let (_, told) = send(
        None,
        "POST",
        "/join/collect",
        serde_json::json!({ "request": key, "secret": "s2" }),
    )
    .await?;
    anyhow::ensure!(
        told["standing"] == "denied" && told["reason"] == "not this quarter",
        "{told}"
    );
    let (_, spare) = send(Some(&ada), "POST", "/join/codes", serde_json::json!({})).await?;
    let (status, _) = send(
        Some(&ada),
        "POST",
        "/join/codes/revoke",
        serde_json::json!({ "key": spare["key"] }),
    )
    .await?;
    anyhow::ensure!(status == StatusCode::OK);
    let (status, _) = send(
        None,
        "POST",
        "/join/request",
        ask(spare["code"].as_str().unwrap_or_default(), "dee", "s3"),
    )
    .await?;
    anyhow::ensure!(status == StatusCode::FORBIDDEN, "a revoked code: {status}");
    Ok(())
}

/// A server that serves only members provisions no one: a signed token for a
/// pair the workspace does not hold is refused, and one let in by a code is
/// served.
#[tokio::test]
async fn a_members_only_server_serves_no_one_it_did_not_let_in() -> anyhow::Result<()> {
    let open = state().await?;
    // Ada opens her workspace and makes an invitation while the server is open.
    let ada = token("ws:shut", "user:ada")?;
    let with = |st: Arc<HttpState>,
                tok: Option<String>,
                method: &'static str,
                path: &'static str,
                body: serde_json::Value| async move {
        let mut req = Request::builder()
            .method(method)
            .uri(path)
            .header(header::CONTENT_TYPE, "application/json");
        if let Some(tok) = tok {
            req = req.header(header::AUTHORIZATION, format!("Bearer {tok}"));
        }
        let res = router(st)
            .oneshot(req.body(Body::from(body.to_string()))?)
            .await?;
        let status = res.status();
        let text = body_text(res).await?;
        anyhow::Ok((
            status,
            serde_json::from_str::<serde_json::Value>(&text).unwrap_or_default(),
        ))
    };
    let (_, made) = with(
        open.clone(),
        Some(ada.clone()),
        "POST",
        "/join/codes",
        serde_json::json!({ "needs_approval": false }),
    )
    .await?;
    let code = made["code"].as_str().unwrap_or_default().to_string();

    // The same store, served to members only.
    let mut shut = state_with(open.store.clone()).await;
    let Some(inner) = Arc::get_mut(&mut shut) else {
        anyhow::bail!("the state is shared before it is served");
    };
    inner.members_only = true;

    // A signed token for someone the workspace does not hold is refused...
    let stranger = token("ws:shut", "user:mallory")?;
    let (status, why) = with(
        shut.clone(),
        Some(stranger),
        "GET",
        "/members",
        serde_json::Value::Null,
    )
    .await?;
    anyhow::ensure!(status == StatusCode::FORBIDDEN, "{status} {why}");
    anyhow::ensure!(
        why["error"]
            .as_str()
            .is_some_and(|w| w.contains("not a member")),
        "{why}"
    );
    // ...its owner is served...
    let (status, _) = with(
        shut.clone(),
        Some(ada),
        "GET",
        "/members",
        serde_json::Value::Null,
    )
    .await?;
    anyhow::ensure!(status == StatusCode::OK, "the owner: {status}");
    // ...and so is someone an invitation let in.
    let asked = serde_json::json!({ "code": code, "handle": "bodil", "secret_digest": antumbra_core::join::digest("s") });
    let (_, request) = with(shut.clone(), None, "POST", "/join/request", asked).await?;
    anyhow::ensure!(request["standing"] == "approved", "{request}");
    let (_, got) = with(
        shut.clone(),
        None,
        "POST",
        "/join/collect",
        serde_json::json!({ "request": request["request"], "secret": "s" }),
    )
    .await?;
    let bodil = got["token"].as_str().map(str::to_string);
    anyhow::ensure!(bodil.is_some(), "{got}");
    let (status, _) = with(shut, bodil, "GET", "/members", serde_json::Value::Null).await?;
    anyhow::ensure!(status == StatusCode::OK, "the invited member: {status}");
    Ok(())
}

/// What to share is suggested from what was shared before: nothing until a
/// few were, then the undecided memories that lie with them and not the ones
/// that lie with what was left out, nearest first; choosing everything
/// suggested is one call; and what is held back is never suggested.
#[tokio::test]
async fn what_to_share_is_suggested_from_what_was_shared_before() -> anyhow::Result<()> {
    let st = state().await?;
    let dana = token("ws:own", "user:dana")?;
    let ask = |tool: &str, arguments: serde_json::Value| {
        let (st, tok) = (st.clone(), dana.clone());
        let body = serde_json::json!({ "tool": tool, "arguments": arguments }).to_string();
        async move {
            let res = router(st).oneshot(call_request(Some(&tok), &body)?).await?;
            let status = res.status();
            let text = body_text(res).await?;
            anyhow::ensure!(status == StatusCode::OK, "{status}: {text}");
            anyhow::Ok(serde_json::from_str::<serde_json::Value>(&text)?)
        }
    };
    // The test embedder lays a text's bytes out in order, so two kinds of
    // text far apart by it: one heavy where it starts, one heavy after that.
    let work = |i: usize| format!("~~~~~~~~{}", "!".repeat(i + 1));
    let home = |i: usize| format!("!!!!!!!!~~~~~~~~{}", "!".repeat(i + 1));
    let mut stored = std::collections::HashMap::new();
    for text in (0..7).map(work).chain((0..7).map(home)) {
        let out = ask(
            "store_memory",
            serde_json::json!({ "content": text, "network": "world" }),
        )
        .await?;
        stored.insert(text, out["id"].as_str().unwrap_or_default().to_string());
    }
    let suggested = serde_json::json!({ "filter": { "suggested": true } });

    // Four shared: not yet, and it says how many more.
    let ids =
        |texts: Vec<String>| -> Vec<String> { texts.iter().map(|t| stored[t].clone()).collect() };
    ask(
        "bring_choose",
        serde_json::json!({ "choice": "chosen", "ids": ids((0..4).map(work).collect()) }),
    )
    .await?;
    let early = ask("bring_candidates", suggested.clone()).await?;
    anyhow::ensure!(
        early["total"] == 0 && early["suggests_after"] == 1,
        "{early}"
    );

    // Five shared and five left out: the two undecided of the shared kind
    // are suggested, and neither of the other kind.
    ask(
        "bring_choose",
        serde_json::json!({ "choice": "brought", "ids": ids(vec![work(4)]) }),
    )
    .await?;
    ask(
        "bring_choose",
        serde_json::json!({ "choice": "declined", "ids": ids((0..5).map(home).collect()) }),
    )
    .await?;
    let now = ask("bring_candidates", suggested.clone()).await?;
    let said: Vec<&str> = now["memories"]
        .as_array()
        .map(|m| m.iter().filter_map(|m| m["content"].as_str()).collect())
        .unwrap_or_default();
    anyhow::ensure!(
        now["total"] == 2 && now.get("suggests_after").is_none(),
        "{now}"
    );
    anyhow::ensure!(said.iter().all(|t| t.starts_with("~~~~")), "{said:?}");
    let leanings: Vec<f64> = now["memories"]
        .as_array()
        .map(|m| m.iter().filter_map(|m| m["leaning"].as_f64()).collect())
        .unwrap_or_default();
    anyhow::ensure!(
        leanings.len() == 2 && leanings[0] >= leanings[1] && leanings[1] > 0.05,
        "{leanings:?}"
    );
    // Without the filter nothing changed: all four undecided, newest first.
    let plain = ask("bring_candidates", serde_json::json!({})).await?;
    anyhow::ensure!(
        plain["total"] == 4 && plain["memories"][0].get("leaning").is_none(),
        "{plain}"
    );

    // A secret of the shared kind is held back, and so is not suggested.
    ask(
        "store_memory",
        serde_json::json!({ "content": "~~~~~~~~ password: aAaa4aaaAaaa9aaa", "network": "world" }),
    )
    .await?;
    let still = ask("bring_candidates", suggested.clone()).await?;
    anyhow::ensure!(still["total"] == 2, "{still}");

    // Everything suggested, chosen in one call; the other kind is untouched.
    let chosen = ask(
        "bring_choose",
        serde_json::json!({ "choice": "chosen", "filter": { "suggested": true } }),
    )
    .await?;
    anyhow::ensure!(chosen["changed"] == 2, "{chosen}");
    let left = ask("bring_candidates", serde_json::json!({})).await?;
    let said: Vec<&str> = left["memories"]
        .as_array()
        .map(|m| m.iter().filter_map(|m| m["content"].as_str()).collect())
        .unwrap_or_default();
    anyhow::ensure!(
        left["total"] == 2 && said.iter().all(|t| t.starts_with("!!!!")),
        "{left}"
    );
    Ok(())
}

/// Choosing what to bring to a workspace: a person's own memories are
/// narrowed and counted on the server, the hooks' own and secret-looking ones
/// are held back until asked for, a choice is recorded for some or for
/// everything a filter matches, and nobody decides about another's memory.
#[tokio::test]
async fn memories_are_chosen_to_bring_by_filter_and_by_hand() -> anyhow::Result<()> {
    let st = state().await?;
    let (dana, eli) = (token("ws:own", "user:dana")?, token("ws:own", "user:eli")?);
    let ask = |tok: &str, tool: &str, arguments: serde_json::Value| {
        let (st, tok) = (st.clone(), tok.to_string());
        let body = serde_json::json!({ "tool": tool, "arguments": arguments }).to_string();
        async move {
            let res = router(st).oneshot(call_request(Some(&tok), &body)?).await?;
            let status = res.status();
            let text = body_text(res).await?;
            anyhow::ensure!(status == StatusCode::OK, "{status}: {text}");
            anyhow::Ok(serde_json::from_str::<serde_json::Value>(&text)?)
        }
    };
    let billing = serde_json::json!({ "repo": "github.com/acme/billing", "commit": "3f9c2ab", "branch": "main" });
    let search = serde_json::json!({ "repo": "github.com/acme/search", "commit": "91aa0de", "branch": "main" });
    let mut ids = Vec::new();
    for (content, network, provenance) in [
        (
            "Billing retries a failed charge three times.",
            "world",
            Some(&billing),
        ),
        (
            "Invoices are numbered per legal entity.",
            "world",
            Some(&billing),
        ),
        (
            "A Friday deploy took checkout down for eleven minutes.",
            "bank",
            Some(&billing),
        ),
        (
            "The search tier's limit was the connection pool.",
            "bank",
            Some(&search),
        ),
        ("Prefer small pull requests.", "opinion", None),
        ("the staging password: Tr0ub4dor&3xyz", "world", None),
        ("[skill-use:dataviz] dataviz", "world", None),
    ] {
        let mut arguments = serde_json::json!({ "content": content, "network": network });
        if let Some(provenance) = provenance {
            arguments["provenance"] = provenance.clone();
        }
        let stored = ask(&dana, "store_memory", arguments).await?;
        ids.push(stored["id"].as_str().unwrap_or_default().to_string());
    }
    let theirs = ask(
        &eli,
        "store_memory",
        serde_json::json!({ "content": "Eli's own note.", "network": "world" }),
    )
    .await?;

    // What is offered: her own, with the two held back counted and left out.
    let all = ask(&dana, "bring_candidates", serde_json::json!({})).await?;
    anyhow::ensure!(all["total"] == 5 && all["held_back"] == 2, "{all}");
    anyhow::ensure!(
        !all.to_string().contains("Eli's own note"),
        "another member's memory was offered"
    );
    anyhow::ensure!(
        !all.to_string().contains("Tr0ub4dor"),
        "a secret was offered"
    );
    let repos = serde_json::json!([
        ["github.com/acme/billing", 3],
        ["github.com/acme/search", 1],
        ["none", 1]
    ]);
    anyhow::ensure!(all["facets"]["repo"] == repos, "{}", all["facets"]);
    let held = ask(
        &dana,
        "bring_candidates",
        serde_json::json!({ "filter": { "include_held": true } }),
    )
    .await?;
    anyhow::ensure!(held["total"] == 7, "{held}");
    let why: Vec<&str> = held["memories"]
        .as_array()
        .map(|m| m.iter().filter_map(|m| m["held"].as_str()).collect())
        .unwrap_or_default();
    anyhow::ensure!(
        why.contains(&"looks_secret") && why.contains(&"housekeeping"),
        "{why:?}"
    );

    // Narrowed on the server, and a page is a page of the whole.
    let narrow = serde_json::json!({ "repo": "github.com/acme/billing", "network": "world" });
    let page = ask(
        &dana,
        "bring_candidates",
        serde_json::json!({ "filter": narrow, "limit": 1 }),
    )
    .await?;
    anyhow::ensure!(
        page["total"] == 2 && page["memories"].as_array().is_some_and(|m| m.len() == 1),
        "{page}"
    );
    let found = ask(
        &dana,
        "bring_candidates",
        serde_json::json!({ "filter": { "contains": "CHECKOUT" } }),
    )
    .await?;
    anyhow::ensure!(found["total"] == 1, "{found}");

    // Everything a filter matches is chosen in one call, with nothing listed.
    let chose = ask(
        &dana,
        "bring_choose",
        serde_json::json!({ "choice": "chosen", "filter": { "repo": "github.com/acme/billing" } }),
    )
    .await?;
    anyhow::ensure!(chose["changed"] == 3, "{chose}");
    let after = ask(&dana, "bring_candidates", serde_json::json!({})).await?;
    anyhow::ensure!(
        after["total"] == 2 && after["states"]["chosen"] == 3,
        "{after}"
    );
    let chosen = ask(
        &dana,
        "bring_candidates",
        serde_json::json!({ "filter": { "state": "chosen" } }),
    )
    .await?;
    anyhow::ensure!(chosen["total"] == 3, "{chosen}");

    // By hand: one declined, one undone, and another member's left alone.
    let by_hand =
        serde_json::json!({ "choice": "never", "ids": [ids[4], theirs["id"], "memory:nope"] });
    anyhow::ensure!(ask(&dana, "bring_choose", by_hand).await?["changed"] == 1);
    anyhow::ensure!(
        ask(
            &dana,
            "bring_choose",
            serde_json::json!({ "choice": "clear", "ids": [ids[0]] })
        )
        .await?["changed"]
            == 1
    );
    let states = ask(&dana, "bring_candidates", serde_json::json!({})).await?["states"].clone();
    anyhow::ensure!(states["chosen"] == 2 && states["never"] == 1, "{states}");
    // "Choose all" leaves what is held back alone unless told to take it.
    let rest = ask(
        &dana,
        "bring_choose",
        serde_json::json!({ "choice": "chosen" }),
    )
    .await?;
    anyhow::ensure!(rest["changed"] == 2, "{rest}");
    let states = ask(&dana, "bring_candidates", serde_json::json!({})).await?["states"].clone();
    anyhow::ensure!(
        states["chosen"] == 4 && states["undecided"] == 2,
        "{states}"
    );

    // Eli's choices are Eli's: none yet, and Dana's are not his to see.
    let his = ask(
        &eli,
        "bring_candidates",
        serde_json::json!({ "filter": { "state": "any" } }),
    )
    .await?;
    anyhow::ensure!(
        his["total"] == 1 && his["states"]["undecided"] == 1,
        "{his}"
    );
    Ok(())
}

/// A compartment is its owner's until it is offered to the workspace and an
/// admin accepts it; then every member reads it, by the engine's own rule.
/// The pool is its owner's to open, an offer its offerer's to take back, and
/// leaving the pool takes what one offered out of everyone's reach.
#[tokio::test]
async fn an_offered_compartment_is_the_workspaces_once_an_admin_accepts_it() -> anyhow::Result<()> {
    let st = state().await?;
    let (ada, bodil, cyrus) = (
        token("ws:pool", "user:ada")?,
        token("ws:pool", "user:bodil")?,
        token("ws:pool", "user:cyrus")?,
    );
    let send = |tok: &str, method: &str, path: &str, body: serde_json::Value| {
        let (st, tok, method, path) = (
            st.clone(),
            tok.to_string(),
            method.to_string(),
            path.to_string(),
        );
        async move {
            let req = Request::builder()
                .method(method.as_str())
                .uri(path)
                .header(header::AUTHORIZATION, format!("Bearer {tok}"))
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body.to_string()))?;
            let res = router(st).oneshot(req).await?;
            let status = res.status();
            let text = body_text(res).await?;
            anyhow::Ok((
                status,
                serde_json::from_str::<serde_json::Value>(&text).unwrap_or_default(),
            ))
        }
    };
    let tool = |tok: &str, tool: &str, arguments: serde_json::Value| {
        let send = send(
            tok,
            "POST",
            "/mcp/call",
            serde_json::json!({ "tool": tool, "arguments": arguments }),
        );
        async move {
            let (status, out) = send.await?;
            anyhow::ensure!(status == StatusCode::OK, "{status}: {out}");
            anyhow::Ok(out)
        }
    };
    let none = serde_json::Value::Null;
    let reads = |tok: &str| {
        let listed = tool(tok, "list_memories", serde_json::json!({}));
        async move { anyhow::Ok(listed.await?.to_string().contains("nullable first")) }
    };
    // Ada is first in, so the owner; the others are members.
    send(&ada, "GET", "/members", none.clone()).await?;
    let brought = tool(
        &bodil,
        "create_compartment",
        serde_json::json!({ "name": "brought from laptop" }),
    )
    .await?;
    let comp = brought["id"].as_str().unwrap_or_default().to_string();
    tool(&bodil, "store_memory", serde_json::json!({ "content": "Migrations run in two steps: nullable first.", "network": "bank", "compartment": comp })).await?;
    anyhow::ensure!(
        reads(&bodil).await? && !reads(&ada).await? && !reads(&cyrus).await?,
        "it is hers alone to begin with"
    );

    // The pool is the owner's to open.
    let (status, _) = send(
        &bodil,
        "POST",
        "/pool/open",
        serde_json::json!({ "on": true }),
    )
    .await?;
    anyhow::ensure!(
        status == StatusCode::FORBIDDEN,
        "a member opened the pool: {status}"
    );
    let (status, _) = send(
        &ada,
        "POST",
        "/pool/open",
        serde_json::json!({ "on": true }),
    )
    .await?;
    anyhow::ensure!(status == StatusCode::OK);

    // Only its owner offers a compartment; offering joins the pool.
    let (status, _) = send(
        &cyrus,
        "POST",
        "/pool/offer",
        serde_json::json!({ "compartment": comp }),
    )
    .await?;
    anyhow::ensure!(
        status == StatusCode::FORBIDDEN,
        "another member offered it: {status}"
    );
    let (status, offered) = send(
        &bodil,
        "POST",
        "/pool/offer",
        serde_json::json!({ "compartment": comp }),
    )
    .await?;
    anyhow::ensure!(
        status == StatusCode::OK && offered["status"] == "offered",
        "{status} {offered}"
    );
    let key = offered["key"].clone();
    let (_, again) = send(
        &bodil,
        "POST",
        "/pool/offer",
        serde_json::json!({ "compartment": comp }),
    )
    .await?;
    anyhow::ensure!(again["key"] == key, "one compartment is one offer");
    anyhow::ensure!(
        !reads(&ada).await?,
        "an offer nobody accepted is read by nobody new"
    );

    // Who decides sees what they would be accepting; another member does not.
    let (_, pool) = send(&ada, "GET", "/pool", none.clone()).await?;
    let seen = &pool["offers"][0];
    anyhow::ensure!(
        pool["open"] == true && seen["name"] == "brought from laptop" && seen["memories"] == 1,
        "{pool}"
    );
    anyhow::ensure!(
        seen["preview"][0]
            .as_str()
            .is_some_and(|p| p.contains("nullable first")),
        "{seen}"
    );
    let (_, others) = send(&cyrus, "GET", "/pool", none.clone()).await?;
    anyhow::ensure!(
        others["offers"][0]["preview"]
            .as_array()
            .is_some_and(Vec::is_empty),
        "{others}"
    );
    anyhow::ensure!(others["joined"] == false, "{others}");

    // A member accepts nothing; an owner does, and then everyone reads it.
    let accept = serde_json::json!({ "key": key, "accept": true });
    let (status, _) = send(&cyrus, "POST", "/pool/rule", accept.clone()).await?;
    anyhow::ensure!(
        status == StatusCode::FORBIDDEN,
        "a member accepted an offer: {status}"
    );
    let (status, _) = send(&ada, "POST", "/pool/rule", accept).await?;
    anyhow::ensure!(status == StatusCode::OK);
    anyhow::ensure!(
        reads(&ada).await? && reads(&cyrus).await?,
        "accepted, and not read by the workspace"
    );
    let recalled = tool(
        &cyrus,
        "recall_memories",
        serde_json::json!({ "query": "how do migrations run", "top_k": 5 }),
    )
    .await?;
    anyhow::ensure!(
        recalled.to_string().contains("nullable first"),
        "accepted, and not recalled: {recalled}"
    );
    // Another workspace reads none of it.
    anyhow::ensure!(!reads(&token("ws:elsewhere", "user:dee")?).await?);

    // Leaving the pool takes it back out of the others' reach; rejoining returns it.
    send(
        &bodil,
        "POST",
        "/pool/join",
        serde_json::json!({ "on": false }),
    )
    .await?;
    anyhow::ensure!(
        reads(&bodil).await? && !reads(&ada).await?,
        "she left, and it is still read"
    );
    send(
        &bodil,
        "POST",
        "/pool/join",
        serde_json::json!({ "on": true }),
    )
    .await?;
    anyhow::ensure!(reads(&ada).await?);
    // Closing the pool does the same for everything in it.
    send(
        &ada,
        "POST",
        "/pool/open",
        serde_json::json!({ "on": false }),
    )
    .await?;
    anyhow::ensure!(
        !reads(&cyrus).await?,
        "the pool is closed, and it is still read"
    );
    send(
        &ada,
        "POST",
        "/pool/open",
        serde_json::json!({ "on": true }),
    )
    .await?;

    // An offer is its offerer's to take back, and nobody else's.
    let named = serde_json::json!({ "key": key });
    let (status, _) = send(&ada, "POST", "/pool/withdraw", named.clone()).await?;
    anyhow::ensure!(
        status == StatusCode::FORBIDDEN,
        "someone else withdrew her offer: {status}"
    );
    let (status, _) = send(&bodil, "POST", "/pool/withdraw", named).await?;
    anyhow::ensure!(status == StatusCode::OK);
    anyhow::ensure!(
        reads(&bodil).await? && !reads(&ada).await?,
        "withdrawn, and still read"
    );
    Ok(())
}
