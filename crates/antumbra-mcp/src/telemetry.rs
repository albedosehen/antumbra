//! Logs and traces.
//!
//! The server's own lines are still `eprintln!`. What this adds is a log
//! subscriber that puts the dependencies' logs on the same stderr when
//! `RUST_LOG` asks for them (none by default, as before), and, when
//! `OTEL_EXPORTER_OTLP_ENDPOINT` is set, a trace export over OTLP/HTTP to
//! that collector, which the exporter addresses at `/v1/traces`.
//! `OTEL_SERVICE_NAME` and `OTEL_RESOURCE_ATTRIBUTES` describe the service.
//! Without the endpoint, or when the exporter cannot be built, the server runs
//! as before and exports nothing. An empty variable counts as an unset one, as
//! the OpenTelemetry specification reads it.
//!
//! Two kinds of span are exported: one per HTTP request (see the router in
//! [`crate::http`]) and one per tool call ([`tool_span`]), and under a recall,
//! one per stage ([`recall_stage`]) and per retrieval leg. None carries what
//! the request was about: no query, memory or document text, no token, no
//! request body. A tool call's span says which tool ran, in which
//! workspace, and whether it failed and how, by kind and never by message,
//! since a message can quote what the caller sent.

use std::ffi::OsStr;

use opentelemetry::trace::TracerProvider as _;
use opentelemetry::KeyValue;
use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::{BatchSpanProcessor, SdkTracerProvider, SpanData, SpanProcessor};
use opentelemetry_sdk::Resource;
use rmcp::model::ErrorCode;
use rmcp::ErrorData;
use tracing::field::Empty;
use tracing_opentelemetry::OpenTelemetrySpanExt as _;
use tracing_subscriber::layer::SubscriberExt as _;
use tracing_subscriber::util::SubscriberInitExt as _;
use tracing_subscriber::{EnvFilter, Layer as _};

/// The log filter when `RUST_LOG` is not set: nothing, which is what the
/// dependencies logged before this subscriber existed. Not even their errors
/// by default, since what they say about a request can quote it: rmcp's
/// request errors carry a tool call's arguments, and the store's client can
/// log the query results it failed to deliver. `RUST_LOG` turns them on to
/// debug.
const LOG_FILTER: &str = "off";

/// What reaches the exporter: the HTTP server spans, which
/// axum-tracing-opentelemetry opens under the `otel::tracing` target (at
/// INFO, see the manifest), this crate's tool and recall-stage spans, and the
/// store's spans around recall's retrieval legs (`antumbra_store::recall`,
/// a leg name and a pool size each), and nothing else. Not even the
/// dependencies' warnings: an event inside a span is exported with it, and
/// what rmcp and the store say about a failed request can quote the request.
pub(crate) const TRACE_FILTER: &str =
    "off,otel::tracing=info,antumbra_mcp=info,antumbra_store::recall=info";

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
/// `RUST_LOG` (nothing by default), and the trace export when it is
/// configured. A caller's `traceparent` is honored either way, so a
/// request's span continues the trace it arrived with.
pub fn init() -> Telemetry {
    opentelemetry::global::set_text_map_propagator(TraceContextPropagator::new());

    let mut failure = None;
    let provider = if set(std::env::var_os("OTEL_EXPORTER_OTLP_ENDPOINT").as_deref()) {
        match opentelemetry_otlp::SpanExporter::builder()
            .with_http()
            .build()
        {
            Ok(exporter) => {
                let mut resource = Resource::builder();
                // A server that says what it is when the deployment does not.
                // The SDK's own detector reads an empty name as unset too, but
                // then names the service `unknown_service:<executable>`, and it
                // keeps a blank one as the name. Added after the detectors have
                // run, this wins over both.
                if !set(std::env::var_os("OTEL_SERVICE_NAME").as_deref()) {
                    resource = resource.with_service_name(env!("CARGO_PKG_NAME"));
                }
                Some(
                    SdkTracerProvider::builder()
                        .with_resource(resource.build())
                        .with_span_processor(RoutesOnly(
                            BatchSpanProcessor::builder(exporter).build(),
                        ))
                        .build(),
                )
            }
            Err(e) => {
                failure = Some(e);
                None
            }
        }
    } else {
        None
    };

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

/// Whether an `OTEL_*` variable holds a value. The OpenTelemetry
/// specification reads an empty one as unset, and so does this, a blank one
/// too. Compose passes `${OTEL_EXPORTER_OTLP_ENDPOINT:-}` as an empty string
/// when docker/.env sets nothing, and that must leave the export off rather
/// than send it to the exporter's default, `localhost:4318`.
fn set(value: Option<&OsStr>) -> bool {
    value.is_some_and(|value| !value.to_string_lossy().trim().is_empty())
}

/// Keeps a request span's path to the route it matched. The request layer
/// records the raw path and query as `url.path` and `url.query`. Every route
/// here is a fixed path (`/mcp`, `/mcp/call`, `/github/webhook`, the
/// dashboard's), so for a request that matched one the raw path says nothing
/// the route does not; for one that matched none it is whatever the caller
/// sent, and so is a query, which no route reads. Recording the attribute again
/// on the live span would only add a second value beside the first, so each
/// finished span is rewritten here on its way to the exporter.
#[derive(Debug)]
pub(crate) struct RoutesOnly<P>(pub(crate) P);

impl<P: SpanProcessor> SpanProcessor for RoutesOnly<P> {
    fn on_start(&self, span: &mut opentelemetry_sdk::trace::Span, cx: &opentelemetry::Context) {
        self.0.on_start(span, cx);
    }

    fn on_end(&self, mut span: SpanData) {
        routes_only(&mut span.attributes);
        self.0.on_end(span);
    }

    fn force_flush(&self) -> OTelSdkResult {
        self.0.force_flush()
    }

    fn shutdown_with_timeout(&self, timeout: std::time::Duration) -> OTelSdkResult {
        self.0.shutdown_with_timeout(timeout)
    }

    fn set_resource(&mut self, resource: &Resource) {
        self.0.set_resource(resource);
    }
}

/// `url.path` becomes the route the request matched (`http.route`), a request
/// that matched none keeps no path, and `url.query` goes.
fn routes_only(attributes: &mut Vec<KeyValue>) {
    let route = attributes
        .iter()
        .find(|kv| kv.key.as_str() == "http.route")
        .map(|kv| kv.value.clone())
        .filter(|route| !route.as_str().is_empty());
    attributes.retain_mut(|kv| match kv.key.as_str() {
        "url.query" => false,
        "url.path" => match &route {
            Some(route) => {
                kv.value = route.clone();
                true
            }
            None => false,
        },
        _ => true,
    });
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

/// The span one stage of a recall runs in, under its tool call's: `recall
/// embed` (the prompt and its parts), `recall retrieve` (the pool drawn from
/// the store, whose legs have spans of their own), `recall rerank` (the
/// candidates the cross-encoder reads) or `recall floor` (the candidates the
/// relevance floor judges). `items` is how many the stage handled: a count,
/// never what they say.
pub(crate) fn recall_stage(stage: &'static str, items: usize) -> tracing::Span {
    tracing::info_span!(
        "recall stage",
        otel.name = stage,
        antumbra.recall.items = items,
    )
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_endpoint_that_is_unset_empty_or_blank_exports_nothing() {
        assert!(!set(None));
        // What compose passes for `${OTEL_EXPORTER_OTLP_ENDPOINT:-}` when
        // docker/.env does not set it.
        assert!(!set(Some(OsStr::new(""))));
        assert!(!set(Some(OsStr::new(" \t"))));
        assert!(set(Some(OsStr::new("http://openobserve:5080/api/default"))));
    }

    #[test]
    fn an_empty_service_name_leaves_the_server_to_name_itself() {
        assert!(!set(None));
        assert!(!set(Some(OsStr::new(""))));
        assert!(!set(Some(OsStr::new(" "))));
        assert!(set(Some(OsStr::new("antumbra-kuskokwim"))));
    }
}
