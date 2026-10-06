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
//! The spans exported are one per HTTP request (see the router in
//! [`crate::http`]), and they do not carry what the request was about: no
//! query, memory or document text, no token, no request body.

use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::SdkTracerProvider;
use opentelemetry_sdk::Resource;
use tracing_subscriber::layer::SubscriberExt as _;
use tracing_subscriber::util::SubscriberInitExt as _;
use tracing_subscriber::{EnvFilter, Layer as _};

/// The log filter when `RUST_LOG` is not set: the dependencies' warnings and
/// errors. Their INFO lines (rmcp's per-session chatter among them) stay out of
/// the log, as they were before a subscriber existed.
const LOG_FILTER: &str = "warn";

/// What reaches the exporter: the HTTP server spans, which
/// axum-tracing-opentelemetry opens at TRACE under the `otel::tracing`
/// target, and this crate's own spans, and nothing else. Not even the
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
