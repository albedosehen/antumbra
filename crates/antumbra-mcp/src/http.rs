//! The networked multi-tenant HTTP transport (ADR-0015).
//!
//! Each request carries a signed JWT; its verified `tenant`/`user` claims select
//! (or lazily create) a session signed in as that identity, so the engine
//! enforces isolation **per request** — one server, many tenants, no app-side
//! filtering. rmcp's `StreamableHttpService` factory takes no request context, so
//! identity is resolved *before* dispatch: the axum handler verifies the token,
//! looks up the per-identity service (whose `McpServer` is already signed in),
//! and delegates. The bearer token is the boundary — a leaked token grants
//! exactly its claimed `(tenant, user)` scope and nothing wider.

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

use crate::auth::{Identity, JwtVerifier};
use crate::server::McpServer;

type IdentityService = StreamableHttpService<McpServer, LocalSessionManager>;

struct HttpState {
    url: String,
    host: String,
    verifier: JwtVerifier,
    embedder: Arc<dyn Embedder>,
    /// One signed-in MCP service per identity, created on first authenticated
    /// request. Each owns its own store connection bound to that `(tenant, user)`.
    sessions: Mutex<HashMap<Identity, IdentityService>>,
}

/// Serve the networked surface on `addr`. Every `/mcp` request must present a
/// JWT whose claims this `verifier` accepts.
pub async fn serve(
    addr: String,
    url: String,
    host: String,
    embedder: Arc<dyn Embedder>,
    verifier: JwtVerifier,
) -> Result<()> {
    let state = Arc::new(HttpState {
        url,
        host,
        verifier,
        embedder,
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

    let service = match state.session_for(&identity).await {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "antumbra-mcp: session init failed for {}/{}: {e}",
                identity.tenant, identity.user
            );
            return internal_error();
        }
    };

    // rmcp returns its own boxed body; rewrap it as an axum body.
    let (parts, body) = service.handle(req).await.into_parts();
    Response::from_parts(parts, Body::new(body))
}

impl HttpState {
    async fn session_for(&self, identity: &Identity) -> Result<IdentityService> {
        let mut map = self.sessions.lock().await;
        if let Some(s) = map.get(identity) {
            return Ok(s.clone());
        }
        let mcp = crate::build_session(
            &self.url,
            TenantId::new(identity.tenant.as_str()),
            UserId::new(identity.user.as_str()),
            self.host.clone(),
            self.embedder.clone(),
        )
        .await?;
        // The factory clones the already-signed-in server for each new MCP
        // session opened under this identity.
        let service = StreamableHttpService::new(
            move || Ok(mcp.clone()),
            Arc::new(LocalSessionManager::default()),
            server_config(),
        );
        map.insert(identity.clone(), service.clone());
        Ok(service)
    }
}

fn server_config() -> StreamableHttpServerConfig {
    // The JWT is the access guard, so we do not restrict by `Host` (the default
    // loopback-only allowlist would refuse LAN clients). DNS-rebinding is moot:
    // there is no ambient-authority cookie or session — every request re-proves
    // identity with a bearer token.
    StreamableHttpServerConfig::default().disable_allowed_hosts()
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

    fn state() -> Arc<HttpState> {
        Arc::new(HttpState {
            url: "mem://".into(),
            host: "test".into(),
            verifier: JwtVerifier::hs256(b"test-secret"),
            embedder: Arc::new(FixedEmbedder::new(EMBED_DIM)),
            sessions: Mutex::new(HashMap::new()),
        })
    }

    async fn status_for(auth: Option<&str>) -> StatusCode {
        let mut builder = Request::builder().method("POST").uri("/mcp");
        if let Some(a) = auth {
            builder = builder.header(header::AUTHORIZATION, a);
        }
        let req = builder.body(Body::empty()).unwrap();
        router(state()).oneshot(req).await.unwrap().status()
    }

    // The auth boundary rejects before any store/session work, so these need no
    // running socket and no database.
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
}
