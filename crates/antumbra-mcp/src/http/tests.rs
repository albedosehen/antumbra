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
