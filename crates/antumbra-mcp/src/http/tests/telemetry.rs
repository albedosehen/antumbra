//! The spans the server exports: which requests open one, when a request on a
//! long-lived stream ends its span, and the span each tool call runs in, over
//! both transports. Read back from an in-memory exporter behind the same filter
//! the server exports through.

use super::*;
use anyhow::Context as _;
use futures::StreamExt;
use opentelemetry::trace::{Status, TracerProvider as _};
use opentelemetry_sdk::trace::{
    InMemorySpanExporter, SdkTracerProvider, SimpleSpanProcessor, SpanData,
};
use std::time::{Duration, Instant};
use tracing_subscriber::layer::SubscriberExt as _;
use tracing_subscriber::{EnvFilter, Layer as _};

/// Spans exported while it is held, from this thread: a `#[tokio::test]` runs
/// its tasks on the test's own thread, rmcp's session tasks included. Through
/// the same filter and path rewrite as the server's export.
struct Traces {
    exporter: InMemorySpanExporter,
    _provider: SdkTracerProvider,
    _guard: tracing::subscriber::DefaultGuard,
}

impl Traces {
    fn start() -> Self {
        let exporter = InMemorySpanExporter::default();
        let provider = SdkTracerProvider::builder()
            .with_span_processor(crate::telemetry::RoutesOnly(SimpleSpanProcessor::new(
                exporter.clone(),
            )))
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
    fn ended(&self) -> anyhow::Result<Vec<SpanData>> {
        self.exporter
            .get_finished_spans()
            .context("the exporter is up")
    }

    /// How long a span the test caused is given to end. A call carried by a
    /// stream runs on the session's task after the response the test awaited,
    /// so its span ends a moment after the test's next line, and later still
    /// on a loaded runner.
    const WAIT: Duration = Duration::from_secs(2);

    /// The one ended span named `name`, waited for up to [`Self::WAIT`]: a
    /// slow runner changes how long this takes, not what it finds. More than
    /// one is wrong at once; none by the end of the wait is wrong by name.
    async fn one(&self, name: &str) -> anyhow::Result<SpanData> {
        let deadline = Instant::now() + Self::WAIT;
        loop {
            let mut found: Vec<_> = self
                .ended()?
                .into_iter()
                .filter(|s| s.name == name)
                .collect();
            match found.len() {
                1 => return found.pop().context("the one span"),
                0 if Instant::now() < deadline => {
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
                n => anyhow::bail!("one `{name}` span, got {n}"),
            }
        }
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
) -> anyhow::Result<Request<Body>> {
    let mut b = Request::builder()
        .method(method)
        .uri("/mcp")
        .header(header::HOST, "localhost")
        .header(header::AUTHORIZATION, jwt)
        .header(header::ACCEPT, "application/json, text/event-stream");
    if let Some(id) = session {
        b = b.header("mcp-session-id", id);
    }
    Ok(match body {
        Some(body) => b
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))?,
        None => b.body(Body::empty())?,
    })
}

/// The stateful handshake, initialize then initialized: the session id.
async fn open_session(app: &Router, jwt: &str) -> anyhow::Result<String> {
    let init = serde_json::json!({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": { "protocolVersion": "2025-06-18", "capabilities": {},
                    "clientInfo": { "name": "traced", "version": "0" } }
    });
    let resp = app
        .clone()
        .oneshot(mcp("POST", jwt, None, Some(init))?)
        .await?;
    assert_eq!(resp.status(), StatusCode::OK);
    let Some(session) = resp
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
        .oneshot(mcp("POST", jwt, Some(&session), Some(initialized))?)
        .await?;
    assert_eq!(resp.status(), StatusCode::ACCEPTED);
    Ok(session)
}

#[test]
fn requests_are_traced_but_what_the_dashboard_page_loads_is_not() -> anyhow::Result<()> {
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
    Ok(())
}

/// A request span keeps the route as its path and no query; a request that
/// matched no route keeps no path at all, since the path is then whatever the
/// caller sent.
#[tokio::test]
async fn a_request_span_keeps_its_route_and_not_the_path_it_was_sent() -> anyhow::Result<()> {
    let traces = Traces::start();
    let app = router(state().await?);
    let lost = Request::builder()
        .uri("/mcp/recite-the-launch-codes?token=sesame")
        .body(Body::empty())?;
    assert_eq!(
        app.clone().oneshot(lost).await?.status(),
        StatusCode::NOT_FOUND
    );
    let page = Request::builder()
        .uri("/dashboard?token=sesame")
        .body(Body::empty())?;
    assert_eq!(app.oneshot(page).await?.status(), StatusCode::OK);

    let ended = traces.ended()?;
    let page = traces.one("GET /dashboard").await?;
    assert_eq!(attribute(&page, "url.path").as_deref(), Some("/dashboard"));
    assert_eq!(attribute(&page, "url.query"), None);
    let Some(lost) = ended
        .iter()
        .find(|s| attribute(s, "http.response.status_code").as_deref() == Some("404"))
    else {
        anyhow::bail!("the unmatched request has a span");
    };
    assert_eq!(attribute(lost, "url.path"), None);
    assert!(!mentions(&ended, "launch-codes"), "the path it was sent");
    assert!(!mentions(&ended, "sesame"), "the query");
    Ok(())
}

/// The store turns on tracing's `release_max_level_debug`, which compiles
/// TRACE out of a release build. The request spans are opened above it, or a
/// release build (what runs in the cluster) would export no request span at
/// all while every debug build, these tests included, still did.
#[test]
fn request_spans_are_opened_at_a_level_a_release_build_keeps() -> anyhow::Result<()> {
    assert!(
        tracing_opentelemetry_instrumentation_sdk::TRACING_LEVEL <= tracing::Level::DEBUG,
        "request spans at {}",
        tracing_opentelemetry_instrumentation_sdk::TRACING_LEVEL
    );
    Ok(())
}

/// A client holds its GET /mcp stream open for as long as its session lasts.
/// The request's span ends when the response starts, so it is exported while
/// the stream is still open rather than hours later when it closes.
#[tokio::test]
async fn an_open_event_stream_has_already_ended_its_span() -> anyhow::Result<()> {
    let traces = Traces::start();
    let app = router(state().await?);
    let jwt = format!("Bearer {}", token("ws:traced", "user:traced")?);
    let session = open_session(&app, &jwt).await?;

    let resp = app
        .clone()
        .oneshot(mcp("GET", &jwt, Some(&session), None)?)
        .await?;
    assert_eq!(resp.status(), StatusCode::OK, "GET opens the SSE stream");
    let stream = resp.into_body();

    let get = traces.one("GET /mcp").await?;
    assert_eq!(get.span_kind, opentelemetry::trace::SpanKind::Server);
    assert!(get.end_time >= get.start_time);
    drop(stream);
    Ok(())
}

/// A tool call over the REST shim is a span inside its request's: named for
/// the tool, saying the workspace it ran in and that it went well.
#[tokio::test]
async fn a_rest_tool_call_is_a_span_inside_its_request() -> anyhow::Result<()> {
    let traces = Traces::start();
    let resp = router(state().await?)
        .oneshot(call_request(
            Some(&token("ws:traced", "user:traced")?),
            r#"{"tool":"workspace_stats"}"#,
        )?)
        .await?;
    assert_eq!(resp.status(), StatusCode::OK);

    let request = traces.one("POST /mcp/call").await?;
    let tool = traces.one("tool workspace_stats").await?;
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
    Ok(())
}

/// An identity's first call builds its server inside the request's span, one
/// span for each part, so a slow first call says whether it waited for the
/// auth lock, provisioned, or connected. The second call reuses the server and
/// builds nothing. No part names the identity.
#[tokio::test]
async fn a_first_call_traces_building_its_server_and_a_second_builds_nothing() -> anyhow::Result<()>
{
    let traces = Traces::start();
    let app = router(state().await?);
    for _ in 0..2 {
        let resp = app
            .clone()
            .oneshot(call_request(
                Some(&token("ws:traced", "user:traced")?),
                r#"{"tool":"workspace_stats"}"#,
            )?)
            .await?;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    let build = traces.one("session build").await?;
    let requests: Vec<SpanData> = traces
        .ended()?
        .into_iter()
        .filter(|s| s.name == "POST /mcp/call")
        .collect();
    assert_eq!(requests.len(), 2);
    assert!(
        requests
            .iter()
            .any(|r| r.span_context.span_id() == build.parent_span_id),
        "the build is inside a request"
    );
    for stage in ["session auth wait", "session provision", "session connect"] {
        assert_eq!(
            traces.one(stage).await?.parent_span_id,
            build.span_context.span_id(),
            "{stage} is part of the build"
        );
    }
    let session: Vec<SpanData> = traces
        .ended()?
        .into_iter()
        .filter(|s| s.name.starts_with("session "))
        .collect();
    assert!(!mentions(&session, "user:traced"), "the user");
    assert!(!mentions(&session, "ws:traced"), "the workspace");
    Ok(())
}

/// A failed call marks its span an error, by kind. The error's message quotes
/// the arguments: the caller is told, the trace is not.
#[tokio::test]
async fn a_failed_tool_call_marks_its_span_an_error_and_leaves_the_message_out(
) -> anyhow::Result<()> {
    const SECRET: &str = "the launch codes";
    let traces = Traces::start();
    let body = serde_json::json!({
        "tool": "store_memory",
        "arguments": { "content": "a memory", "confidence": SECRET }
    })
    .to_string();
    let resp = router(state().await?)
        .oneshot(call_request(
            Some(&token("ws:traced", "user:traced")?),
            &body,
        )?)
        .await?;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let told = axum::body::to_bytes(resp.into_body(), 1 << 16).await?;
    let told = String::from_utf8_lossy(&told);
    assert!(
        told.contains(SECRET),
        "the caller hears what was wrong: {told}"
    );

    let tool = traces.one("tool store_memory").await?;
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
        !mentions(&traces.ended()?, SECRET),
        "nothing exported quotes the arguments"
    );
    assert!(!mentions(&traces.ended()?, "a memory"), "nor the memory");
    Ok(())
}

/// A span that never ends is reported by name once the wait is out, not
/// waited for without end.
#[tokio::test]
async fn a_span_that_never_comes_is_reported_by_name_after_the_wait() -> anyhow::Result<()> {
    let traces = Traces::start();
    let asked = Instant::now();
    let Err(e) = traces.one("never").await else {
        anyhow::bail!("no span was opened, so none can be found");
    };
    anyhow::ensure!(e.to_string().contains("one `never` span, got 0"), "{e}");
    anyhow::ensure!(asked.elapsed() >= Traces::WAIT, "waited out the bound");
    Ok(())
}

/// A name no tool has is the caller's text, so its span says `unknown`.
#[tokio::test]
async fn an_unknown_tool_is_traced_as_unknown() -> anyhow::Result<()> {
    const NAME: &str = "recite_the_launch_codes";
    let traces = Traces::start();
    let resp = router(state().await?)
        .oneshot(call_request(
            Some(&token("ws:traced", "user:traced")?),
            &serde_json::json!({ "tool": NAME }).to_string(),
        )?)
        .await?;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

    let tool = traces.one("tool unknown").await?;
    assert_eq!(
        attribute(&tool, "gen_ai.tool.name").as_deref(),
        Some("unknown")
    );
    assert_eq!(
        attribute(&tool, "error.type").as_deref(),
        Some("invalid_params")
    );
    assert!(!mentions(&traces.ended()?, NAME));
    Ok(())
}

/// Over JSON-RPC, rmcp runs the call on the session's task, away from the
/// request that carried it, and the request's span has ended by the time the
/// call does (the result streams after the response starts). The call's span
/// still joins that request's trace.
#[tokio::test]
async fn a_jsonrpc_tool_call_joins_the_request_that_carried_it() -> anyhow::Result<()> {
    let traces = Traces::start();
    let app = router(state().await?);
    let jwt = format!("Bearer {}", token("ws:traced", "user:traced")?);
    let session = open_session(&app, &jwt).await?;

    let call = serde_json::json!({
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": { "name": "workspace_stats", "arguments": {} }
    });
    let resp = app
        .clone()
        .oneshot(mcp("POST", &jwt, Some(&session), Some(call))?)
        .await?;
    assert_eq!(resp.status(), StatusCode::OK);
    let mut stream = resp.into_body().into_data_stream();
    let mut seen = String::new();
    let answered = match tokio::time::timeout(std::time::Duration::from_secs(8), async {
        while let Some(chunk) = stream.next().await {
            seen.push_str(&String::from_utf8_lossy(&chunk?));
            if seen.contains("\"result\"") {
                return anyhow::Ok(true);
            }
        }
        anyhow::Ok(false)
    })
    .await
    {
        Ok(answered) => answered?,
        Err(_timed_out) => false,
    };
    assert!(answered, "the call answered on its stream; saw: {seen}");

    let tool = traces.one("tool workspace_stats").await?;
    let Some(carried) = traces
        .ended()?
        .into_iter()
        .find(|s| s.span_context.span_id() == tool.parent_span_id)
    else {
        anyhow::bail!("the call's parent is an exported span");
    };
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
    Ok(())
}
