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
//! high-concurrency deployment points `--url` at a real `ws://` server. The
//! stateless JSON response mode keeps each `handle` bounded so the lock is never
//! held across a long-lived stream.

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
    /// One MCP service per identity (provisioned once), all sharing `store`.
    sessions: Mutex<HashMap<Identity, IdentityService>>,
}

/// Serve the networked surface on `addr` over one shared connection to `url`.
/// Every `/mcp` request must present a JWT this `verifier` accepts.
pub async fn serve(
    addr: String,
    url: String,
    host: String,
    embedder: Arc<dyn Embedder>,
    verifier: JwtVerifier,
) -> Result<()> {
    let store = crate::connect(&url).await?;
    let state = Arc::new(HttpState {
        store,
        host,
        verifier,
        embedder,
        auth: Mutex::new(()),
        sessions: Mutex::new(HashMap::new()),
    });
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    eprintln!("antumbra-mcp: networked surface on http://{addr}/mcp (JWT-authenticated)");
    axum::serve(listener, router(state)).await?;
    Ok(())
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
            self.store.invalidate().await?; // owner mode for the writes
            crate::provision_identity(&self.store, &tenant, &user).await?
        };

        let mcp = McpServer::new(
            self.store.clone(),
            self.embedder.clone(),
            tenant,
            user,
            self.host.clone(),
            default_compartment,
        );
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
        // Stateless JSON: each POST is a complete request/response, so the auth
        // lock is never held across a long-lived SSE stream.
        .with_stateful_mode(false)
        .with_json_response(true)
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
            sessions: Mutex::new(HashMap::new()),
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

    fn valid_token() -> String {
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
                tenant: "ws:t".into(),
                user: "user:t".into(),
                exp: get_current_timestamp() + 3600,
            },
            &EncodingKey::from_secret(b"test-secret"),
        )
        .unwrap()
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
}
