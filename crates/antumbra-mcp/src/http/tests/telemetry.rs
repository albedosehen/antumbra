//! The spans the router exports: which requests open one, and when a request
//! on a long-lived stream ends its span. Read back from an in-memory exporter
//! behind the same filter the server exports through.

use super::*;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider, SpanData};
use tracing_subscriber::layer::SubscriberExt as _;
use tracing_subscriber::{EnvFilter, Layer as _};

/// Spans exported while it is held, from this thread: a `#[tokio::test]` runs
/// its tasks on the test's own thread, rmcp's session tasks included.
struct Traces {
    exporter: InMemorySpanExporter,
    _provider: SdkTracerProvider,
    _guard: tracing::subscriber::DefaultGuard,
}

impl Traces {
    fn start() -> Self {
        let exporter = InMemorySpanExporter::default();
        let provider = SdkTracerProvider::builder()
            .with_simple_exporter(exporter.clone())
            .build();
        let subscriber = tracing_subscriber::registry().with(
            tracing_opentelemetry::layer()
                .with_tracer(provider.tracer("test"))
                .with_filter(EnvFilter::new(crate::telemetry::TRACE_FILTER)),
        );
        Self {
            exporter,
            _provider: provider,
            _guard: tracing::subscriber::set_default(subscriber),
        }
    }

    /// The spans that have ended so far.
    fn ended(&self) -> Vec<SpanData> {
        self.exporter
            .get_finished_spans()
            .expect("the exporter is up")
    }

    /// The one ended span named `name`.
    fn one(&self, name: &str) -> SpanData {
        let mut found: Vec<_> = self
            .ended()
            .into_iter()
            .filter(|s| s.name == name)
            .collect();
        assert_eq!(found.len(), 1, "one `{name}` span, got {}", found.len());
        found.remove(0)
    }
}

/// A request to `/mcp` as `jwt`, in `session` when there is one.
fn mcp(
    method: &str,
    jwt: &str,
    session: Option<&str>,
    body: Option<serde_json::Value>,
) -> Request<Body> {
    let mut b = Request::builder()
        .method(method)
        .uri("/mcp")
        .header(header::HOST, "localhost")
        .header(header::AUTHORIZATION, jwt)
        .header(header::ACCEPT, "application/json, text/event-stream");
    if let Some(id) = session {
        b = b.header("mcp-session-id", id);
    }
    match body {
        Some(body) => b
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .unwrap(),
        None => b.body(Body::empty()).unwrap(),
    }
}

/// The stateful handshake, initialize then initialized: the session id.
async fn open_session(app: &Router, jwt: &str) -> String {
    let init = serde_json::json!({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": { "protocolVersion": "2025-06-18", "capabilities": {},
                    "clientInfo": { "name": "traced", "version": "0" } }
    });
    let resp = app
        .clone()
        .oneshot(mcp("POST", jwt, None, Some(init)))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let session = resp.headers()["mcp-session-id"]
        .to_str()
        .unwrap()
        .to_string();
    let initialized = serde_json::json!({"jsonrpc": "2.0", "method": "notifications/initialized"});
    let resp = app
        .clone()
        .oneshot(mcp("POST", jwt, Some(&session), Some(initialized)))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::ACCEPTED);
    session
}

#[test]
fn requests_are_traced_but_what_the_dashboard_page_loads_is_not() {
    for path in [
        "/mcp",
        "/mcp/call",
        "/github/webhook",
        "/dashboard",
        "/no-such-path",
        "/dashboards",
    ] {
        assert!(traced(path), "{path} should be traced");
    }
    for path in ["/dashboard/", "/dashboard/app.js", "/dashboard/app.css"] {
        assert!(!traced(path), "{path} should not be traced");
    }
}

/// A client holds its GET /mcp stream open for as long as its session lasts.
/// The request's span ends when the response starts, so it is exported while
/// the stream is still open rather than hours later when it closes.
#[tokio::test]
async fn an_open_event_stream_has_already_ended_its_span() {
    let traces = Traces::start();
    let app = router(state().await);
    let jwt = format!("Bearer {}", token("ws:traced", "user:traced"));
    let session = open_session(&app, &jwt).await;

    let resp = app
        .clone()
        .oneshot(mcp("GET", &jwt, Some(&session), None))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "GET opens the SSE stream");
    let stream = resp.into_body();

    let get = traces.one("GET /mcp");
    assert_eq!(get.span_kind, opentelemetry::trace::SpanKind::Server);
    assert!(get.end_time >= get.start_time);
    drop(stream);
}
