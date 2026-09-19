//! The transport tests: the auth boundary, the REST shim, the authenticated
//! initialize, and the live-propagation round trip over the real router.

use super::*;
use antumbra_core::testing::FixedEmbedder;
use antumbra_store::EMBED_DIM;
use tower::ServiceExt; // oneshot

async fn state() -> Arc<HttpState> {
    let store = Store::connect_memory(EMBED_DIM).await.unwrap();
    Arc::new(HttpState {
        serving: Serving::Shared(store.clone()),
        store,
        host: "test".into(),
        verifier: JwtVerifier::hs256(b"test-secret"),
        embedder: Arc::new(FixedEmbedder::new(EMBED_DIM)),
        embedders: Mutex::new(Bounded::new(MAX_EMBEDDERS)),
        auth: Mutex::new(()),
        auto_propose: None,
        auto_consolidate: false,
        serve: None,
        reranker: None,
        copal: None,
        profile: None,
        github: None,
        sessions: Mutex::new(Bounded::new(MAX_SESSIONS)),
        consolidating: Arc::new(Mutex::new(std::collections::HashSet::new())),
        registry: crate::notify::PeerRegistry::new(),
    })
}

async fn status_for(auth: Option<&str>) -> StatusCode {
    let mut builder = Request::builder().method("POST").uri("/mcp");
    if let Some(a) = auth {
        builder = builder.header(header::AUTHORIZATION, a);
    }
    let req = builder.body(Body::empty()).unwrap();
    router(state().await).oneshot(req).await.unwrap().status()
}

// The auth boundary rejects before any store/session work, so these need no
// running socket and no provisioned tenant.
#[tokio::test]
async fn missing_token_is_unauthorized() {
    assert_eq!(status_for(None).await, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn garbage_token_is_unauthorized() {
    assert_eq!(
        status_for(Some("Bearer not-a-real-jwt")).await,
        StatusCode::UNAUTHORIZED
    );
}

#[tokio::test]
async fn non_bearer_scheme_is_unauthorized() {
    assert_eq!(
        status_for(Some("Basic dXNlcjpwYXNz")).await,
        StatusCode::UNAUTHORIZED
    );
}

fn token(tenant: &str, user: &str) -> String {
    use jsonwebtoken::{encode, get_current_timestamp, Algorithm, EncodingKey, Header};
    #[derive(serde::Serialize)]
    struct C {
        tenant: String,
        user: String,
        exp: u64,
    }
    encode(
        &Header::new(Algorithm::HS256),
        &C {
            tenant: tenant.into(),
            user: user.into(),
            exp: get_current_timestamp() + 3600,
        },
        &EncodingKey::from_secret(b"test-secret"),
    )
    .unwrap()
}

fn valid_token() -> String {
    token("ws:t", "user:t")
}

fn call_request(tok: Option<&str>, body: &str) -> Request<Body> {
    let mut b = Request::builder().method("POST").uri("/mcp/call");
    if let Some(t) = tok {
        b = b.header(header::AUTHORIZATION, format!("Bearer {t}"));
    }
    b.body(Body::from(body.to_owned())).unwrap()
}

// The REST shim (P-1b): a single POST dispatches a tool under the caller's JWT
// identity -- no initialize handshake -- with the same auth + engine ACL.
#[tokio::test]
async fn rest_call_dispatches_a_tool_then_reads_it_back() {
    let st = state().await;
    let tok = token("ws:rest", "user:rest");

    // Store a memory through POST /mcp/call.
    let store = call_request(
        Some(&tok),
        r#"{"tool":"store_memory","arguments":{"content":"the deno runtime"}}"#,
    );
    let resp = router(st.clone()).oneshot(store).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Read it back the same way; the result carries the stored content.
    let list = call_request(Some(&tok), r#"{"tool":"list_memories","arguments":{}}"#);
    let resp = router(st.clone()).oneshot(list).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    assert!(String::from_utf8_lossy(&bytes).contains("deno"));

    // No token -> 401 (auth before any work).
    let resp = router(st.clone())
        .oneshot(call_request(
            None,
            r#"{"tool":"list_memories","arguments":{}}"#,
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

    // Unknown tool / malformed body -> 400 (the caller's fault, not a panic).
    let resp = router(st.clone())
        .oneshot(call_request(
            Some(&tok),
            r#"{"tool":"nope","arguments":{}}"#,
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let resp = router(st)
        .oneshot(call_request(Some(&tok), "not json"))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

// The happy path: a valid token verifies, the shared connection signs in as
// the identity, and rmcp dispatches `initialize` to the McpServer, proving
// the full auth -> signin -> handle wiring end to end (provisioning the
// identity on first contact). No socket: the router is driven via oneshot.
#[tokio::test]
async fn valid_token_reaches_the_service() {
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
        .header(header::AUTHORIZATION, format!("Bearer {}", valid_token()))
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::ACCEPT, "application/json, text/event-stream")
        .body(Body::from(body))
        .unwrap();
    let resp = router(state().await).oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    let text = String::from_utf8_lossy(&bytes);
    assert_eq!(status, StatusCode::OK, "reached service? body: {text}");
    // Stateless mode may answer as JSON or a single SSE `data:` line; either
    // way the initialize result (serverInfo) must be present.
    assert!(
        text.contains("serverInfo") || text.contains("protocolVersion"),
        "expected an initialize result, got: {text}"
    );
}

// The R-2 last mile, over the wire: a grantee (B) holds an open MCP SSE
// stream; when A writes into the shared compartment, B's stream receives the
// `antumbra/memory_changed` notification. Drives the real `/mcp` router
// through the full stateful handshake (initialize -> initialized -> GET SSE),
// so it exercises the actual transport path, peer capture, watcher, and push.
#[tokio::test]
async fn live_notification_reaches_a_grantees_stream() {
    use antumbra_core::{
        Capability, Compartment, CompartmentId, Grant, Memory, MemoryNetwork, Origin, UserId,
    };
    use antumbra_store::repo::{compartment, memory, principal};
    use futures::StreamExt;

    let state = state().await;
    let tenant = TenantId::new("ws:t");
    let alice = UserId::new("user:a");
    let bob = UserId::new("user:b");
    let comp = CompartmentId::new("comp:shared");
    let now = chrono::Utc::now();

    // Owner-mode setup: both principals, a compartment alice owns, bob granted.
    principal::provision(&state.store, &tenant, &alice)
        .await
        .unwrap();
    principal::provision(&state.store, &tenant, &bob)
        .await
        .unwrap();
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
    .await
    .unwrap();
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
    .await
    .unwrap();

    // Start the live watcher (registers its LIVE subscription in owner mode).
    spawn_live_propagation(state.clone());

    let app = router(state.clone());
    let bob_jwt = format!("Bearer {}", token("ws:t", "user:b"));

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
                .body(Body::from(init))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let session_id = resp
        .headers()
        .get("mcp-session-id")
        .expect("stateful mode returns a session id")
        .to_str()
        .unwrap()
        .to_string();

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
                ))
                .unwrap(),
        )
        .await
        .unwrap();
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
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "GET opens the SSE stream");
    let mut stream = resp.into_body().into_data_stream();

    // A write into the shared compartment (owner mode), the change alice makes.
    {
        let _guard = state.auth.lock().await;
        state.store.invalidate().await.unwrap();
        let m = Memory::new(
            "cccccccc-0000-0000-0000-00000000000c",
            tenant.clone(),
            MemoryNetwork::World,
            "shared via SSE",
            0.9,
            now,
        )
        .in_compartment(comp.clone());
        memory::upsert(&state.store, &m).await.unwrap();
    }

    // 4. The notification arrives on bob's SSE stream.
    let mut seen = String::new();
    let found = tokio::time::timeout(std::time::Duration::from_secs(8), async {
        while let Some(chunk) = stream.next().await {
            let bytes = chunk.expect("stream chunk");
            seen.push_str(&String::from_utf8_lossy(&bytes));
            if seen.contains("antumbra/memory_changed") {
                return true;
            }
        }
        false
    })
    .await
    .unwrap_or(false);
    assert!(found, "grantee's SSE stream got the change; saw: {seen}");
}
