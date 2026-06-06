//! The networked multi-tenant HTTP transport (ADR-0015).
//!
//! Each request carries a signed JWT; its verified `tenant`/`user` claims become
//! `$auth`, so the engine enforces isolation **per request** — one server, many
//! tenants, no app-side filtering. A leaked token grants exactly its claimed
//! scope and nothing wider.
//!
//! ## One connection, serialized — works on embedded too
//!
//! An embedded SurrealDB (`surrealkv://`, the edge/IoT case) is **single-writer**:
//! only one connection may open the datastore. So the server holds **one** shared
//! connection and multiplexes tenants over it — it does *not* open a connection
//! per identity. Because `signin` binds the whole connection, each request takes
//! a lock, signs the shared connection in as its identity, runs, and releases;
//! the next request re-signs-in. The engine then hides other tenants' rows even
//! on an unfiltered query (proven in `antumbra-store`'s embedded tests). The cost
//! is serialization of the authenticated section — fine for an edge device; a
//! high-concurrency deployment points `--url` at a real `ws://` server.
//!
//! ## Stateful (SSE) mode for live propagation (R-2)
//!
//! The transport runs in rmcp's stateful mode so a client can hold an open
//! GET/SSE stream that carries **server-initiated** notifications — the only
//! channel the MCP spec defines for push. That does not reintroduce the lock
//! concern: the auth lock is held only while `handle` *builds* a response, and an
//! SSE stream is MCP transport state (a channel + cache) that does no DB work and
//! streams *after* the handler returns — it never holds the DB connection. Live
//! delivery is wired in [`spawn_live_propagation`]: one owner-mode `LIVE`
//! subscription (registered at startup) feeds the change watcher, audience is
//! resolved under the auth lock in owner mode, and the change is pushed to each
//! recipient's captured peer ([`crate::notify`]).

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use axum::body::Body;
use axum::extract::State;
use axum::http::{header, Request, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::any;
use axum::Router;
use tokio::sync::Mutex;

use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::{StreamableHttpServerConfig, StreamableHttpService};

use antumbra_core::ports::Embedder;
use antumbra_core::{TenantId, UserId};
use antumbra_store::Store;

use crate::auth::{Identity, JwtVerifier};
use crate::server::McpServer;

type IdentityService = StreamableHttpService<McpServer, LocalSessionManager>;

struct HttpState {
    /// The one shared connection. Embedded is single-writer, so every tenant is
    /// served over this; `auth` serializes the signed-in section.
    store: Store,
    host: String,
    verifier: JwtVerifier,
    embedder: Arc<dyn Embedder>,
    /// Serializes `signin(identity) -> handle` so two identities never share the
    /// connection's auth state concurrently.
    auth: Mutex<()>,
    /// Autonomous propose threshold, applied to every per-identity server.
    auto_propose: Option<usize>,
    /// The serving engine the `answer` tool drives, built once from the owner
    /// view of the population and shared by every per-identity server (routing
    /// scopes which expert a session may pick).
    serve: Option<Arc<dyn antumbra_core::ports::Serve>>,
    /// One MCP service per identity (provisioned once), all sharing `store`.
    sessions: Mutex<HashMap<Identity, IdentityService>>,
    /// Live-propagation (R-2) delivery: each session registers its peer here on
    /// initialize; the change watcher pushes shared-memory changes to recipients.
    registry: crate::notify::PeerRegistry,
}

/// Serve the networked surface on `addr` over one shared connection to `url`.
/// Every `/mcp` request must present a JWT this `verifier` accepts.
#[allow(clippy::too_many_arguments)]
pub async fn serve(
    addr: String,
    url: String,
    db_user: Option<String>,
    db_pass: Option<String>,
    host: String,
    embedder: Arc<dyn Embedder>,
    verifier: JwtVerifier,
    auto_propose: Option<usize>,
) -> Result<()> {
    let store = crate::connect(&url, db_user.as_deref(), db_pass.as_deref()).await?;
    // Built once here in owner mode (before any per-request signin), so it sees
    // the whole population; the answer tool's routing enforces per-session scope.
    let serve = crate::build_serve(&store).await?;
    let state = Arc::new(HttpState {
        store,
        host,
        verifier,
        embedder,
        auth: Mutex::new(()),
        auto_propose,
        serve,
        sessions: Mutex::new(HashMap::new()),
        registry: crate::notify::PeerRegistry::new(),
    });
    spawn_live_propagation(state.clone());
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    eprintln!("antumbra-mcp: networked surface on http://{addr}/mcp (JWT-authenticated)");
    axum::serve(listener, router(state)).await?;
    Ok(())
}

/// R-2 live propagation: watch the memory change feed and push each shared-memory
/// change to its recipients' open sessions. The feed and audience resolution run
/// over the one shared connection, so the resolution is serialized under the auth
/// lock in **owner** mode (it must see across tenants); the notification fan-out
/// touches no DB. The `LIVE` subscription is registered now, while the connection
/// is still the startup owner session (before any per-request signin).
fn spawn_live_propagation(state: Arc<HttpState>) {
    tokio::spawn(async move {
        let mut feed = match antumbra_store::repo::sync::watch_table(&state.store, "memory").await {
            Ok(feed) => feed,
            Err(e) => {
                eprintln!("antumbra-mcp: live propagation disabled (watch failed): {e}");
                return;
            }
        };
        eprintln!("antumbra-mcp: live propagation watching shared-memory changes");
        while let Some(event) = feed.recv().await {
            let change = {
                let _guard = state.auth.lock().await;
                if state.store.signin_root().await.is_err() {
                    continue; // could not return to owner view; skip this event
                }
                antumbra_sync::resolve_change(&state.store, &event).await
            };
            if let Some(change) = change {
                state.registry.notify(&change).await;
            }
        }
    });
}

/// The `/mcp` router. Extracted so the auth boundary can be exercised with
/// `oneshot` (no socket) in tests.
fn router(state: Arc<HttpState>) -> Router {
    Router::new().route("/mcp", any(handle)).with_state(state)
}

async fn handle(State(state): State<Arc<HttpState>>, req: Request<Body>) -> Response {
    let header_val = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok());
    let identity = match state.verifier.verify_header(header_val) {
        Ok(id) => id,
        Err(e) => {
            // Log the reason server-side; the client only ever sees a generic 401.
            eprintln!("antumbra-mcp: rejected request: {e}");
            return unauthorized();
        }
    };

    let service = match state.service_for(&identity).await {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "antumbra-mcp: session init failed for {}/{}: {e}",
                identity.tenant, identity.user
            );
            return internal_error();
        }
    };

    // Serialize the authenticated section over the single shared connection: bind
    // this identity, run the request, then release so the next request re-binds.
    let _guard = state.auth.lock().await;
    if let Err(e) = state
        .store
        .signin(&TenantId::new(&identity.tenant), &UserId::new(&identity.user))
        .await
    {
        eprintln!("antumbra-mcp: signin failed for {}: {e}", identity.tenant);
        return internal_error();
    }
    // rmcp returns its own boxed body; rewrap it as an axum body.
    let (parts, body) = service.handle(req).await.into_parts();
    Response::from_parts(parts, Body::new(body))
}

impl HttpState {
    /// Get (or first-time provision) the MCP service for an identity. All
    /// services share the one `store`; only their provenance fields differ.
    async fn service_for(&self, identity: &Identity) -> Result<IdentityService> {
        if let Some(s) = self.sessions.lock().await.get(identity) {
            return Ok(s.clone());
        }
        let tenant = TenantId::new(&identity.tenant);
        let user = UserId::new(&identity.user);

        // Provision owner-side (principal + default compartment). Done under the
        // auth lock in owner mode so it never races a signed-in request.
        let default_compartment = {
            let _guard = self.auth.lock().await;
            self.store.signin_root().await?; // owner mode for the writes
            crate::provision_identity(&self.store, &tenant, &user).await?
        };

        let mut mcp = McpServer::new(
            self.store.clone(),
            self.embedder.clone(),
            tenant,
            user,
            self.host.clone(),
            default_compartment,
            self.serve.clone(),
        );
        if let Some(threshold) = self.auto_propose {
            mcp = mcp.with_auto_propose(threshold);
        }
        mcp = mcp.with_registry(self.registry.clone());
        // The factory clones the (store-sharing) server for each MCP exchange.
        let service = StreamableHttpService::new(
            move || Ok(mcp.clone()),
            Arc::new(LocalSessionManager::default()),
            server_config(),
        );
        self.sessions
            .lock()
            .await
            .insert(identity.clone(), service.clone());
        Ok(service)
    }
}

fn server_config() -> StreamableHttpServerConfig {
    StreamableHttpServerConfig::default()
        // Stateful (SSE) mode: required for the client's GET stream that carries
        // server-initiated notifications (live propagation, R-2). The auth lock is
        // still only held while `handle` builds each response -- the GET stream
        // does no DB work and streams *after* the handler returns -- so the lock
        // never spans the stream (the ADR-0015 concern does not apply: an SSE
        // stream is MCP transport state, it does not hold the DB connection).
        .with_stateful_mode(true)
        .with_json_response(false)
        // The JWT is the access guard, so we do not restrict by `Host` (the
        // default loopback-only allowlist would refuse LAN clients). DNS-rebinding
        // is moot: every request re-proves identity with a bearer token, there is
        // no ambient-authority cookie or session.
        .disable_allowed_hosts()
}

fn unauthorized() -> Response {
    (StatusCode::UNAUTHORIZED, "invalid or missing bearer token").into_response()
}

fn internal_error() -> Response {
    (StatusCode::INTERNAL_SERVER_ERROR, "session initialization failed").into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use antumbra_core::testing::FixedEmbedder;
    use antumbra_store::EMBED_DIM;
    use tower::ServiceExt; // oneshot

    async fn state() -> Arc<HttpState> {
        Arc::new(HttpState {
            store: Store::connect_memory(EMBED_DIM).await.unwrap(),
            host: "test".into(),
            verifier: JwtVerifier::hs256(b"test-secret"),
            embedder: Arc::new(FixedEmbedder::new(EMBED_DIM)),
            auth: Mutex::new(()),
            auto_propose: None,
            serve: None,
            sessions: Mutex::new(HashMap::new()),
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

    // The happy path: a valid token verifies, the shared connection signs in as
    // the identity, and rmcp dispatches `initialize` to the McpServer — proving
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
        let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20).await.unwrap();
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
        principal::provision(&state.store, &tenant, &alice).await.unwrap();
        principal::provision(&state.store, &tenant, &bob).await.unwrap();
        compartment::create(
            &state.store,
            &Compartment {
                id: comp.clone(),
                tenant: tenant.clone(),
                owner: alice.clone(),
                name: "shared".into(),
                origin: Origin::User,
                created_at: now,
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
}
