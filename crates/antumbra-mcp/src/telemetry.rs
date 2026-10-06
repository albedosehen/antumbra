//! Logs and traces.
//!
//! The server's own lines are still `eprintln!`. What this adds is a log
//! subscriber that puts the dependencies' warnings and errors on the same
//! stderr (`RUST_LOG` widens it), and, when `OTEL_EXPORTER_OTLP_ENDPOINT` is
//! set, a trace export over OTLP/HTTP to that collector, which the exporter
//! addresses at `/v1/traces`. `OTEL_SERVICE_NAME` and
//! `OTEL_RESOURCE_ATTRIBUTES` describe the service. Without the endpoint, or
//! when the exporter cannot be built, the server runs as before and exports
//! nothing.
//!
//! Two kinds of span are exported: one per HTTP request (see the router in
//! [`crate::http`]) and one per tool call ([`tool_span`]). Neither carries
//! what the request was about: no query, memory or document text, no token,
//! no request body. A tool call's span says which tool ran, in which
//! workspace, and whether it failed and how, by kind and never by message,
//! since a message can quote what the caller sent.

use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::SdkTracerProvider;
use opentelemetry_sdk::Resource;
use rmcp::model::ErrorCode;
use rmcp::ErrorData;
use tracing::field::Empty;
use tracing_opentelemetry::OpenTelemetrySpanExt as _;
use tracing_subscriber::layer::SubscriberExt as _;
use tracing_subscriber::util::SubscriberInitExt as _;
use tracing_subscriber::{EnvFilter, Layer as _};

/// The log filter when `RUST_LOG` is not set: the dependencies' warnings and
/// errors. Their INFO lines (rmcp's per-session chatter among them) stay out of
/// the log, as they were before a subscriber existed.
const LOG_FILTER: &str = "warn";

/// What reaches the exporter: the HTTP server spans, which
/// axum-tracing-opentelemetry opens at TRACE under the `otel::tracing`
/// target, and this crate's tool spans, and nothing else. Not even the
/// dependencies' warnings: an event inside a span is exported with it, and
/// what rmcp and the store say about a failed request can quote the request.
pub(crate) const TRACE_FILTER: &str = "off,otel::tracing=trace,antumbra_mcp=info";

/// The tracer provider, when spans are exported. Hold it for the life of the
/// process: dropping it flushes the spans still queued and stops the exporter.
/// Dropped outside the runtime, since the exporter's HTTP client is a blocking
/// one.
#[must_use = "dropping it stops the trace export"]
pub struct Telemetry(Option<SdkTracerProvider>);

impl Drop for Telemetry {
    fn drop(&mut self) {
        if let Some(provider) = self.0.take() {
            if let Err(e) = provider.shutdown() {
                eprintln!("antumbra-mcp: flushing the trace export failed: {e}");
            }
        }
    }
}

/// Installs the global subscriber: the log lines on stderr, filtered by
/// `RUST_LOG` (default `warn`), and the trace export when it is configured. A
/// caller's `traceparent` is honored either way, so a request's span continues
/// the trace it arrived with.
pub fn init() -> Telemetry {
    opentelemetry::global::set_text_map_propagator(TraceContextPropagator::new());

    let mut failure = None;
    let provider = std::env::var_os("OTEL_EXPORTER_OTLP_ENDPOINT").and_then(|_| {
        match opentelemetry_otlp::SpanExporter::builder()
            .with_http()
            .build()
        {
            Ok(exporter) => {
                let mut resource = Resource::builder();
                // A server that says what it is when the deployment does not.
                if std::env::var_os("OTEL_SERVICE_NAME").is_none() {
                    resource = resource.with_service_name(env!("CARGO_PKG_NAME"));
                }
                Some(
                    SdkTracerProvider::builder()
                        .with_resource(resource.build())
                        .with_batch_exporter(exporter)
                        .build(),
                )
            }
            Err(e) => {
                failure = Some(e);
                None
            }
        }
    });

    // stdout is the stdio transport's JSON-RPC channel, so the log goes where
    // the server's own lines go.
    let logs = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stderr)
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stderr()))
        .with_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| LOG_FILTER.into()));
    let traces = provider.as_ref().map(|provider| {
        tracing_opentelemetry::layer()
            .with_tracer(provider.tracer(env!("CARGO_PKG_NAME")))
            .with_filter(EnvFilter::new(TRACE_FILTER))
    });
    tracing_subscriber::registry()
        .with(logs)
        .with(traces)
        .init();

    if let Some(e) = failure {
        eprintln!(
            "antumbra-mcp: the OTLP exporter failed to build; serving without trace export: {e}"
        );
    }
    Telemetry(provider)
}

/// The trace an HTTP request runs in, carried into the request so the tool
/// call it makes over JSON-RPC can join it. rmcp runs that call on the
/// session's own task, where the request's span is not the current one; it
/// does hand the tool the request's parts, and this travels in their
/// extensions. A context rather than the span itself, which would hold the
/// request's span open for as long as the call runs.
#[derive(Clone)]
pub(crate) struct RequestTrace(opentelemetry::Context);

impl RequestTrace {
    /// The trace of the span the caller is in.
    pub(crate) fn current() -> Self {
        Self(tracing::Span::current().context())
    }
}

/// The span one tool call runs in: `tool <name>`, with the tool and the
/// workspace (the verified tenant) as attributes, and the outcome recorded by
/// [`record_outcome`] once it returns. A compartment is left out: it arrives
/// as an argument, and an argument is the caller's text until a tool has
/// checked it.
///
/// `known` says whether `name` is one of this server's tools. A name that is
/// not is the caller's text too, so the span says `unknown` in its place.
///
/// Joined to `parent` when the call came over an HTTP request whose trace
/// rides along ([`RequestTrace`]); otherwise it is a child of the current
/// span (the REST shim's request), or starts a trace of its own (stdio).
pub(crate) fn tool_span(
    name: &str,
    known: bool,
    workspace: &str,
    parent: Option<&RequestTrace>,
) -> tracing::Span {
    let tool = if known { name } else { "unknown" };
    let span = tracing::info_span!(
        "tool",
        otel.name = format!("tool {tool}"),
        gen_ai.tool.name = tool,
        antumbra.workspace = workspace,
        antumbra.outcome = Empty,
        "error.type" = Empty,
        otel.status_code = Empty,
    );
    if let Some(RequestTrace(cx)) = parent {
        // Fails only where nothing exports (no layer, or the span filtered
        // out), and then there is no trace to join.
        let _ = span.set_parent(cx.clone());
    }
    span
}

/// Records how a tool call ended on its span: `ok`, or `error` with the
/// span's status set to ERROR and `error.type` naming the kind of failure.
pub(crate) fn record_outcome(span: &tracing::Span, failure: Option<&'static str>) {
    match failure {
        None => {
            span.record("antumbra.outcome", "ok");
        }
        Some(kind) => {
            span.record("antumbra.outcome", "error");
            span.record("error.type", kind);
            span.record("otel.status_code", "ERROR");
        }
    }
}

/// The kind of a tool error, from its JSON-RPC code. The message stays out of
/// telemetry: "bad arguments" quotes the arguments, and a store error can quote
/// what was stored.
pub(crate) fn error_kind(error: &ErrorData) -> &'static str {
    match error.code {
        ErrorCode::INVALID_PARAMS => "invalid_params",
        ErrorCode::INTERNAL_ERROR => "internal_error",
        ErrorCode::RESOURCE_NOT_FOUND => "resource_not_found",
        ErrorCode::METHOD_NOT_FOUND => "method_not_found",
        ErrorCode::INVALID_REQUEST => "invalid_request",
        _ => "other",
    }
}
