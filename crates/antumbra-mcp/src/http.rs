//! The networked multi-tenant HTTP transport: the MCP runtime surface.
//!
//! Each request carries a signed JWT; its verified `tenant`/`user` claims become
//! `$auth`, so the engine enforces isolation **per request**: one server, many
//! tenants, no app-side filtering. A leaked token grants exactly its claimed
//! scope and nothing wider.
//!
//! ## One connection, serialized: works on embedded too
//!
//! An embedded SurrealDB (`surrealkv://`, the edge/IoT case) is **single-writer**:
//! only one connection may open the datastore. So the server holds **one** shared
//! connection and multiplexes tenants over it; it does *not* open a connection
//! per identity. Because `signin` binds the whole connection, each request takes
//! a lock, signs the shared connection in as its identity, runs, and releases;
//! the next request re-signs-in. The engine then hides other tenants' rows even
//! on an unfiltered query (proven in `antumbra-store`'s embedded tests). The cost
//! is serialization of the authenticated section: fine for an edge device; a
//! high-concurrency deployment points `--url` at a real `ws://` server.
//!
//! ## Stateful (SSE) mode for live propagation (R-2)
//!
//! The transport runs in rmcp's stateful mode so a client can hold an open
//! GET/SSE stream that carries **server-initiated** notifications: the only
//! channel the MCP spec defines for push. That does not reintroduce the lock
//! concern: the auth lock is held only while `handle` *builds* a response, and an
//! SSE stream is MCP transport state (a channel + cache) that does no DB work and
//! streams *after* the handler returns; it never holds the DB connection. Live
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
use axum::routing::{any, post};
use axum::{Json, Router};
use tokio::sync::Mutex;

use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::{StreamableHttpServerConfig, StreamableHttpService};

use antumbra_core::ports::Embedder;
use antumbra_core::{TenantId, UserId};
use antumbra_store::Store;

use crate::auth::{Identity, JwtVerifier};
use crate::server::McpServer;

type IdentityService = StreamableHttpService<McpServer, LocalSessionManager>;

/// Cap on cached per-identity MCP services / serving connections. Bounds memory
/// on a long-running multi-tenant host; an evicted identity just re-provisions on
/// its next request.
const MAX_SESSIONS: usize = 4096;
/// Cap on cached per-tenant resolved embedders.
const MAX_EMBEDDERS: usize = 1024;

/// A bounded, insertion-ordered map: at capacity it evicts the oldest entry
/// before adding a new key. Bounds the per-identity caches so a long-running
/// server cannot grow them without limit. Eviction only drops the cache's handle;
/// an in-flight request holding a clone keeps its value alive (cached values are
/// reference-counted).
struct Bounded<K, V> {
    map: HashMap<K, V>,
    order: std::collections::VecDeque<K>,
    cap: usize,
}

impl<K: Clone + Eq + std::hash::Hash, V> Bounded<K, V> {
    fn new(cap: usize) -> Self {
        Self {
            map: HashMap::new(),
            order: std::collections::VecDeque::new(),
            cap: cap.max(1),
        }
    }

    fn get(&self, key: &K) -> Option<&V> {
        self.map.get(key)
    }

    fn insert(&mut self, key: K, value: V) {
        // A brand-new key extends the order ring and may evict the oldest; an
        // update to an existing key just replaces the value (no reorder).
        if self.map.insert(key.clone(), value).is_none() {
            self.order.push_back(key);
            while self.order.len() > self.cap {
                if let Some(old) = self.order.pop_front() {
                    self.map.remove(&old);
                }
            }
        }
    }
}

/// How a request reaches a scoped (record-signed) connection so the engine ACL is
/// enforced -- never the root `store`, which bypasses it (R-6). Chosen at startup
/// by deployment.
enum Serving {
    /// Embedded (no DB credentials): the single connection, the same as `store`.
    /// A request signs it in as its record under `auth` for the call; with one
    /// writer, serializing is fine, and embedded mem:// / surrealkv expose only
    /// one connection anyway.
    Shared(Store),
    /// Remote (authenticated): a *dedicated, credential-less* connection per
    /// identity, signed in once as its record and reused across that identity's
    /// concurrent requests (the surreal client multiplexes them), so requests run
    /// WITHOUT the global lock. Bounded; eviction drops only the cache handle --
    /// an in-flight request keeps its clone alive.
    PerIdentity {
        url: String,
        conns: Mutex<Bounded<Identity, Store>>,
    },
}

struct HttpState {
    /// The **root/owner** connection: schema, per-identity provisioning, and the
    /// owner-view R-2 watcher run here. On a remote it is root-authenticated.
    store: Store,
    /// The scoped serving strategy (per-identity connections on a remote, the
    /// shared connection on embedded).
    serving: Serving,
    host: String,
    verifier: JwtVerifier,
    /// The server default embedder (the `--embedder-url` / built-in), used for
    /// any workspace without its own configured endpoint.
    embedder: Arc<dyn Embedder>,
    /// Per-tenant resolved embedders (hosted P-1c): a workspace's own configured
    /// endpoint when set, else `embedder`. Resolved once per tenant and cached.
    embedders: Mutex<Bounded<String, Arc<dyn Embedder>>>,
    /// Serializes the brief owner-side work on the shared connections: the
    /// `signin_root` + provision on `store`, and (embedded only) a request's
    /// record signin on the shared serving connection. On a remote the per-request
    /// hot path holds NO lock -- each identity has its own serving connection.
    auth: Mutex<()>,
    /// Autonomous propose threshold, applied to every per-identity server.
    auto_propose: Option<usize>,
    /// Autonomous consolidation trigger, applied to every per-identity server.
    auto_consolidate: bool,
    /// The serving engine the `answer` tool drives, built once from the owner
    /// view of the population and shared by every per-identity server (routing
    /// scopes which expert a session may pick).
    serve: Option<Arc<dyn antumbra_core::ports::Serve>>,
    /// One MCP service per identity (provisioned once), all sharing `store`.
    /// Bounded so a host that sees many distinct identities cannot grow it without
    /// limit; an evicted identity rebuilds its service on the next request.
    sessions: Mutex<Bounded<Identity, IdentityService>>,
    /// Compartments with a consolidation in flight, SHARED across every
    /// per-identity server so concurrent reinforces of the same compartment
    /// collapse into one train (each request builds a fresh `McpServer`, so a
    /// per-instance guard never coalesces and they race on the weight download).
    consolidating: Arc<Mutex<std::collections::HashSet<String>>>,
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
    auto_consolidate: bool,
) -> Result<()> {
    let store = crate::connect(&url, db_user.as_deref(), db_pass.as_deref()).await?;
    // The scoped serving strategy. On an authenticated remote, requests must run
    // on NON-root connections or the engine ACL is bypassed (R-6): give each
    // identity its own credential-less connection (schema already applied by
    // `store`), reused across its requests so the hot path needs no global lock.
    // On embedded there are no credentials and only one connection is possible, so
    // serving reuses `store`; a per-request record signin scopes it.
    let serving = match (db_user.as_deref(), db_pass.as_deref()) {
        (Some(_), Some(_)) => Serving::PerIdentity {
            url: url.clone(),
            conns: Mutex::new(Bounded::new(MAX_SESSIONS)),
        },
        _ => Serving::Shared(store.clone()),
    };
    // Built once here in owner mode (before any per-request signin), so it sees
    // the whole population; the answer tool's routing enforces per-session scope.
    let serve = crate::build_serve(&store).await?;
    let state = Arc::new(HttpState {
        store,
        serving,
        host,
        verifier,
        embedder,
        embedders: Mutex::new(Bounded::new(MAX_EMBEDDERS)),
        auth: Mutex::new(()),
        auto_propose,
        auto_consolidate,
        serve,
        sessions: Mutex::new(Bounded::new(MAX_SESSIONS)),
        consolidating: Arc::new(Mutex::new(std::collections::HashSet::new())),
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
            // Return the shared connection to the owner view under the lock, then
            // resolve the audience OUTSIDE it: resolution only reads, on the
            // always-root `store`, so holding the lock across it would needlessly
            // serialize the watcher with every request.
            {
                let _guard = state.auth.lock().await;
                if state.store.signin_root().await.is_err() {
                    continue; // could not return to owner view; skip this event
                }
            }
            let change = antumbra_sync::resolve_change(&state.store, &event).await;
            if let Some(change) = change {
                state.registry.notify(&change).await;
            }
        }
        // The feed only closes when the watch task ends (stream error/kill or the
        // connection dropping). Surface it: otherwise live propagation would stop
        // for every connected client with no trace.
        eprintln!("antumbra-mcp: live propagation stopped (memory change feed closed)");
    });
}

/// The `/mcp` router. Extracted so the auth boundary can be exercised with
/// `oneshot` (no socket) in tests.
fn router(state: Arc<HttpState>) -> Router {
    Router::new()
        .route("/mcp", any(handle))
        .route("/mcp/call", post(handle_call))
        .with_state(state)
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

    // Bind the scoped connection to this identity. On a remote each identity has
    // its own already-signed-in serving connection, so this is a no-op and the
    // request runs with NO lock; on embedded it signs the shared connection in
    // under `auth`, held across the exchange.
    let _guard = match state.bind(&identity).await {
        Ok(g) => g,
        Err(e) => {
            eprintln!("antumbra-mcp: signin failed for {}: {e}", identity.tenant);
            return internal_error();
        }
    };
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
        // The SSE service registers each session's peer for live propagation (R-2).
        let mcp = self
            .mcp_for(identity)
            .await?
            .with_registry(self.registry.clone());
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

    /// Provision (owner-side) and build the per-identity [`McpServer`]: the base
    /// the JSON-RPC SSE service wraps, and the one the REST `/mcp/call` shim drives
    /// directly. Its tools run on the SCOPED serving connection so the engine ACL
    /// is enforced per request (not the root `store`, which would bypass it on a
    /// remote; R-6).
    async fn mcp_for(&self, identity: &Identity) -> Result<McpServer> {
        let tenant = TenantId::new(&identity.tenant);
        let user = UserId::new(&identity.user);
        // Provision owner-side (principal + default compartment), under the auth
        // lock in owner mode so it never races a signed-in request.
        let (default_compartment, embedder) = {
            let _guard = self.auth.lock().await;
            self.store.signin_root().await?;
            let dc = crate::provision_identity(&self.store, &tenant, &user).await?;
            // Resolve the workspace's embedder under the owner connection (P-1c).
            let emb = self.embedder_for(&tenant).await;
            (dc, emb)
        };
        // The connection the tools run on: the identity's dedicated serving
        // connection on a remote (signed in once, reused, no per-request lock), or
        // the shared `store` on embedded (a request signs it in per call; `bind`).
        let conn = self.serving_conn(identity, &tenant, &user).await?;
        let mut mcp = McpServer::new(
            conn,
            embedder,
            tenant,
            user,
            self.host.clone(),
            default_compartment,
            self.serve.clone(),
        );
        if let Some(threshold) = self.auto_propose {
            mcp = mcp.with_auto_propose(threshold);
        }
        if self.auto_consolidate {
            // The autonomous trigger must consolidate on a stable OWNER
            // connection, not the scoped serving connection this server is built
            // with. `store` is the root/owner connection (only ever
            // signed-in-as-root), so the detached background task gathers,
            // provisions, and mints as owner regardless of request churn.
            mcp = mcp
                .with_auto_consolidate()
                .with_consolidation_store(self.store.clone())
                .with_consolidating(self.consolidating.clone());
        }
        Ok(mcp)
    }

    /// The scoped connection an identity's tools run on. Embedded: the shared
    /// `store` (a request binds it via [`HttpState::bind`]). Remote: the
    /// identity's own credential-less connection, created + record-signed once and
    /// cached, then reused across that identity's concurrent requests.
    async fn serving_conn(
        &self,
        identity: &Identity,
        tenant: &TenantId,
        user: &UserId,
    ) -> Result<Store> {
        let (url, conns) = match &self.serving {
            Serving::Shared(conn) => return Ok(conn.clone()),
            Serving::PerIdentity { url, conns } => (url, conns),
        };
        if let Some(c) = conns.lock().await.get(identity) {
            return Ok(c.clone());
        }
        // The principal was just provisioned on `store`; a fresh connection's
        // signin can briefly not see it (the cold-start race), so verify with a
        // cheap authed read and retry with backoff before caching.
        let conn = crate::connect_serving(url).await?;
        let mut attempt = 0u64;
        loop {
            conn.signin(tenant, user).await?;
            match warmup_probe(&conn, tenant).await {
                Ok(()) => break,
                Err(e) if attempt < 3 && is_cold_auth_race_msg(&e.to_string()) => {
                    attempt += 1;
                    tokio::time::sleep(std::time::Duration::from_millis(150 * attempt)).await;
                }
                Err(e) => return Err(e),
            }
        }
        conns.lock().await.insert(identity.clone(), conn.clone());
        Ok(conn)
    }

    /// Bind the scoped serving connection to `identity` for a request. Embedded:
    /// sign the shared connection in under `auth` and return the guard to hold
    /// across the exchange. Remote: the identity's own connection is already
    /// signed in, so return `None` -- the request runs with no lock.
    async fn bind(&self, identity: &Identity) -> Result<Option<tokio::sync::MutexGuard<'_, ()>>> {
        match &self.serving {
            Serving::Shared(conn) => {
                let guard = self.auth.lock().await;
                conn.signin(
                    &TenantId::new(&identity.tenant),
                    &UserId::new(&identity.user),
                )
                .await?;
                Ok(Some(guard))
            }
            Serving::PerIdentity { .. } => Ok(None),
        }
    }

    /// Resolve a workspace's embedder (hosted P-1c): its own configured endpoint
    /// when set, else the server default. Cached per tenant after the first
    /// resolution; a config read error falls back to the default rather than
    /// failing the session. The caller holds the auth lock in owner mode, so the
    /// (tenant-scoped) config row is read on the owner connection.
    async fn embedder_for(&self, tenant: &TenantId) -> Arc<dyn Embedder> {
        let key = tenant.as_str().to_string();
        if let Some(e) = self.embedders.lock().await.get(&key) {
            return e.clone();
        }
        let resolved = match antumbra_store::repo::embedder_config::get(&self.store, tenant).await {
            Ok(Some(cfg)) => Arc::new(antumbra_embed::HttpEmbedder::new(
                cfg.url,
                cfg.model,
                cfg.api_key,
            )) as Arc<dyn Embedder>,
            Ok(None) => self.embedder.clone(),
            Err(e) => {
                eprintln!(
                    "antumbra-mcp: embedder config read failed for {key}: {e}; using default"
                );
                self.embedder.clone()
            }
        };
        self.embedders.lock().await.insert(key, resolved.clone());
        resolved
    }
}

fn server_config() -> StreamableHttpServerConfig {
    StreamableHttpServerConfig::default()
        // Stateful (SSE) mode: required for the client's GET stream that carries
        // server-initiated notifications (live propagation, R-2). The auth lock is
        // still only held while `handle` builds each response -- the GET stream
        // does no DB work and streams *after* the handler returns -- so the lock
        // never spans the stream (the runtime-surface concern does not apply: an SSE
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
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        "session initialization failed",
    )
        .into_response()
}

fn bad_request(msg: &str) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({ "error": msg })),
    )
        .into_response()
}

/// The body of a `POST /mcp/call`: a tool name and its arguments.
#[derive(serde::Deserialize)]
struct RestCall {
    tool: String,
    #[serde(default)]
    arguments: serde_json::Value,
}

/// A cheap authed SELECT confirming a serving connection's record session is
/// live: a scoped session returns `Ok(None)`; a cold/unscoped one errors with the
/// "anonymous / not enough permissions" message.
async fn warmup_probe(conn: &Store, tenant: &TenantId) -> Result<()> {
    antumbra_store::repo::memory::get(
        conn,
        tenant,
        &antumbra_core::MemoryId::new("memory:__warmup__"),
    )
    .await?;
    Ok(())
}

/// True for the transient "anonymous / not enough permissions" a freshly
/// established serving connection can return before a just-provisioned principal
/// is visible to it; a retry after a short backoff clears it.
fn is_cold_auth_race_msg(msg: &str) -> bool {
    msg.contains("Anonymous access") || msg.contains("Not enough permissions")
}

/// REST convenience surface (P-1b): `POST /mcp/call {tool, arguments}` returns the
/// tool's JSON result, so a one-shot client (a lifecycle hook fetching bootstrap
/// context) can call a tool without the JSON-RPC initialize -> tools/call
/// handshake. Same JWT auth and scoped-connection engine ACL as `/mcp` -- a thin
/// transport over the identical tools, not a second authority.
async fn handle_call(State(state): State<Arc<HttpState>>, req: Request<Body>) -> Response {
    let header_val = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok());
    let identity = match state.verifier.verify_header(header_val) {
        Ok(id) => id,
        Err(e) => {
            eprintln!("antumbra-mcp: rejected /mcp/call: {e}");
            return unauthorized();
        }
    };

    let body = match axum::body::to_bytes(req.into_body(), 64 * 1024).await {
        Ok(b) => b,
        Err(_) => return bad_request("request body unreadable or too large"),
    };
    let call: RestCall = match serde_json::from_slice(&body) {
        Ok(c) => c,
        Err(e) => return bad_request(&format!("invalid JSON body: {e}")),
    };

    // Provision + build the per-identity server (its dedicated serving connection
    // is created and warmed here on a remote, retrying the cold-start race once
    // per identity rather than per request).
    let mcp = match state.mcp_for(&identity).await {
        Ok(m) => m,
        Err(e) => {
            eprintln!(
                "antumbra-mcp: /mcp/call init failed for {}/{}: {e}",
                identity.tenant, identity.user
            );
            return internal_error();
        }
    };
    // Bind the scoped connection, then dispatch. Embedded: lock + signin the
    // shared connection, held across the call. Remote: a no-op (the identity's
    // own connection is already signed in), so the call runs with no lock.
    let result = {
        let _guard = match state.bind(&identity).await {
            Ok(g) => g,
            Err(e) => {
                eprintln!(
                    "antumbra-mcp: /mcp/call signin failed for {}: {e}",
                    identity.tenant
                );
                return internal_error();
            }
        };
        mcp.call_tool(&call.tool, call.arguments).await
    };

    match result {
        Ok(value) => Json(value).into_response(),
        // A tool error (unknown tool, bad arguments, not found) is the caller's
        // fault -> 400 with the tool's message; auth/ACL failures returned above.
        Err(e) => bad_request(e.message.as_ref()),
    }
}

#[cfg(test)]
mod tests {
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
}
