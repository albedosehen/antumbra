//! The spans the server exports: which requests open one, when a request on a
//! long-lived stream ends its span, and the span each tool call runs in, over
//! both transports. Read back from an in-memory exporter behind the same filter
//! the server exports through.

use super::*;
use futures::StreamExt;
use opentelemetry::trace::{Status, TracerProvider as _};
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

/// The value of the attribute `key` on `span`, if it has one.
fn attribute(span: &SpanData, key: &str) -> Option<String> {
    span.attributes
        .iter()
        .find(|kv| kv.key.as_str() == key)
        .map(|kv| kv.value.as_str().into_owned())
}

/// Whether anything exported says `text`: a span's name, an attribute, an
/// event, a status.
fn mentions(spans: &[SpanData], text: &str) -> bool {
    spans.iter().any(|s| {
        s.name.contains(text)
            || s.attributes
                .iter()
                .any(|kv| kv.value.as_str().contains(text))
            || s.events.iter().any(|e| {
                e.name.contains(text)
                    || e.attributes
                        .iter()
                        .any(|kv| kv.value.as_str().contains(text))
            })
            || matches!(&s.status, Status::Error { description } if description.contains(text))
    })
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

/// A tool call over the REST shim is a span inside its request's: named for
/// the tool, saying the workspace it ran in and that it went well.
#[tokio::test]
async fn a_rest_tool_call_is_a_span_inside_its_request() {
    let traces = Traces::start();
    let resp = router(state().await)
        .oneshot(call_request(
            Some(&token("ws:traced", "user:traced")),
            r#"{"tool":"workspace_stats"}"#,
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let request = traces.one("POST /mcp/call");
    let tool = traces.one("tool workspace_stats");
    assert_eq!(tool.parent_span_id, request.span_context.span_id());
    assert_eq!(
        tool.span_context.trace_id(),
        request.span_context.trace_id()
    );
    assert_eq!(
        attribute(&tool, "gen_ai.tool.name").as_deref(),
        Some("workspace_stats")
    );
    assert_eq!(
        attribute(&tool, "antumbra.workspace").as_deref(),
        Some("ws:traced")
    );
    assert_eq!(attribute(&tool, "antumbra.outcome").as_deref(), Some("ok"));
    assert_eq!(attribute(&tool, "error.type"), None);
    assert_eq!(tool.status, Status::Unset);
}

/// A failed call marks its span an error, by kind. The error's message quotes
/// the arguments: the caller is told, the trace is not.
#[tokio::test]
async fn a_failed_tool_call_marks_its_span_an_error_and_leaves_the_message_out() {
    const SECRET: &str = "the launch codes";
    let traces = Traces::start();
    let body = serde_json::json!({
        "tool": "store_memory",
        "arguments": { "content": "a memory", "confidence": SECRET }
    })
    .to_string();
    let resp = router(state().await)
        .oneshot(call_request(
            Some(&token("ws:traced", "user:traced")),
            &body,
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let told = axum::body::to_bytes(resp.into_body(), 1 << 16)
        .await
        .unwrap();
    let told = String::from_utf8_lossy(&told);
    assert!(
        told.contains(SECRET),
        "the caller hears what was wrong: {told}"
    );

    let tool = traces.one("tool store_memory");
    assert!(
        matches!(tool.status, Status::Error { .. }),
        "{:?}",
        tool.status
    );
    assert_eq!(
        attribute(&tool, "error.type").as_deref(),
        Some("invalid_params")
    );
    assert_eq!(
        attribute(&tool, "antumbra.outcome").as_deref(),
        Some("error")
    );
    assert!(
        !mentions(&traces.ended(), SECRET),
        "nothing exported quotes the arguments"
    );
    assert!(!mentions(&traces.ended(), "a memory"), "nor the memory");
}

/// A name no tool has is the caller's text, so its span says `unknown`.
#[tokio::test]
async fn an_unknown_tool_is_traced_as_unknown() {
    const NAME: &str = "recite_the_launch_codes";
    let traces = Traces::start();
    let resp = router(state().await)
        .oneshot(call_request(
            Some(&token("ws:traced", "user:traced")),
            &serde_json::json!({ "tool": NAME }).to_string(),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

    let tool = traces.one("tool unknown");
    assert_eq!(
        attribute(&tool, "gen_ai.tool.name").as_deref(),
        Some("unknown")
    );
    assert_eq!(
        attribute(&tool, "error.type").as_deref(),
        Some("invalid_params")
    );
    assert!(!mentions(&traces.ended(), NAME));
}

/// Over JSON-RPC, rmcp runs the call on the session's task, away from the
/// request that carried it, and the request's span has ended by the time the
/// call does (the result streams after the response starts). The call's span
/// still joins that request's trace.
#[tokio::test]
async fn a_jsonrpc_tool_call_joins_the_request_that_carried_it() {
    let traces = Traces::start();
    let app = router(state().await);
    let jwt = format!("Bearer {}", token("ws:traced", "user:traced"));
    let session = open_session(&app, &jwt).await;

    let call = serde_json::json!({
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": { "name": "workspace_stats", "arguments": {} }
    });
    let resp = app
        .clone()
        .oneshot(mcp("POST", &jwt, Some(&session), Some(call)))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let mut stream = resp.into_body().into_data_stream();
    let mut seen = String::new();
    let answered = tokio::time::timeout(std::time::Duration::from_secs(8), async {
        while let Some(chunk) = stream.next().await {
            seen.push_str(&String::from_utf8_lossy(&chunk.expect("stream chunk")));
            if seen.contains("\"result\"") {
                return true;
            }
        }
        false
    })
    .await
    .unwrap_or(false);
    assert!(answered, "the call answered on its stream; saw: {seen}");

    let tool = traces.one("tool workspace_stats");
    let carried = traces
        .ended()
        .into_iter()
        .find(|s| s.span_context.span_id() == tool.parent_span_id)
        .expect("the call's parent is an exported span");
    assert_eq!(carried.name, "POST /mcp");
    assert_eq!(
        tool.span_context.trace_id(),
        carried.span_context.trace_id()
    );
    assert_eq!(
        attribute(&tool, "antumbra.workspace").as_deref(),
        Some("ws:traced")
    );
    assert_eq!(attribute(&tool, "antumbra.outcome").as_deref(), Some("ok"));
}
