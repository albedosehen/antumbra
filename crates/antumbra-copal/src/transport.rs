//! The production transport: blocking `ureq` calls over an agent with a
//! bounded global timeout, so a dead copal fails fast instead of hanging the
//! worker.

use serde_json::Value;

use antumbra_core::{AntumbraError, Result};

use crate::{CopalCredential, CopalTransport};

/// Default per-request budget for the copal endpoint, overridable with
/// `ANTUMBRA_COPAL_TIMEOUT_SECS` (the embedder's `EMBED_TIMEOUT_SECS`
/// pattern). Without a bound a hung archive pins the ingest forever -- and the
/// ingest deliberately FAILS when a configured archive is unreachable, so the
/// bound is what turns "hangs" into "fails fast with a clear error".
const COPAL_TIMEOUT_SECS: u64 = 30;

/// The production transport: blocking `ureq` calls over an agent with a bounded
/// global timeout, so a dead copal fails fast instead of hanging the worker.
pub(crate) struct UreqTransport {
    agent: ureq::Agent,
}

impl UreqTransport {
    pub(crate) fn new() -> Self {
        let secs = std::env::var("ANTUMBRA_COPAL_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .filter(|&s| s > 0)
            .unwrap_or(COPAL_TIMEOUT_SECS);
        // A refusal comes back as a response rather than a bare status error,
        // so the ingest can say what copal said.
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(std::time::Duration::from_secs(secs)))
            .http_status_as_error(false)
            .build()
            .into();
        Self { agent }
    }
}

impl CopalTransport for UreqTransport {
    fn post_json(&self, url: &str, credential: &CopalCredential, body: &Value) -> Result<Value> {
        let req = self.agent.post(url);
        let resp = authed(req, credential)
            .send_json(body)
            .map_err(|e| AntumbraError::other(format!("copal POST {url} failed: {e}")))?;
        answer("POST", url, resp)
    }

    fn put_bytes(
        &self,
        url: &str,
        credential: &CopalCredential,
        content_type: &str,
        digest: &str,
        body: &[u8],
    ) -> Result<Value> {
        let req = self
            .agent
            .put(url)
            .header("content-type", content_type)
            .header("x-copal-digest", digest);
        let resp = authed(req, credential)
            .send(body)
            .map_err(|e| AntumbraError::other(format!("copal PUT {url} failed: {e}")))?;
        answer("PUT", url, resp)
    }
}

/// A response's JSON, or an error carrying copal's own account of why it
/// refused the call.
fn answer(method: &str, url: &str, mut resp: ureq::http::Response<ureq::Body>) -> Result<Value> {
    let status = resp.status();
    if status.is_success() {
        return resp
            .body_mut()
            .read_json::<Value>()
            .map_err(|e| AntumbraError::other(format!("copal response was not JSON: {e}")));
    }
    let text = resp.body_mut().read_to_string().unwrap_or_default();
    Err(AntumbraError::other(format!(
        "copal {method} {url} answered {}: {}",
        status.as_u16(),
        refusal(&text)
    )))
}

/// What copal said in a refusal. Its errors are `{"error": {"kind",
/// "message"}}`; anything else is shown as it came, cut short.
pub(crate) fn refusal(body: &str) -> String {
    let parsed: Option<Value> = serde_json::from_str(body).ok();
    match parsed.as_ref().and_then(|v| v.get("error")) {
        Some(error) => format!(
            "{} ({})",
            error["message"].as_str().unwrap_or("no message"),
            error["kind"].as_str().unwrap_or("no kind")
        ),
        None if body.trim().is_empty() => "no body".to_string(),
        None => body.chars().take(200).collect(),
    }
}

/// Apply `credential` to a request: the tenant header (copal header auth mode)
/// or the bearer key (keys mode). The key rides the `Authorization` header,
/// never the URL.
fn authed<B>(
    req: ureq::RequestBuilder<B>,
    credential: &CopalCredential,
) -> ureq::RequestBuilder<B> {
    match credential {
        CopalCredential::Tenant(t) => req.header("x-copal-tenant", t),
        CopalCredential::Bearer(k) => req.header("authorization", &format!("Bearer {k}")),
    }
}

#[cfg(test)]
mod tests {
    use super::refusal;

    #[test]
    fn a_refusal_reads_copal_s_error_body() {
        let body = r#"{"error":{"kind":"conflict","message":"idempotency key consumed by a deleted file"}}"#;
        assert_eq!(
            refusal(body),
            "idempotency key consumed by a deleted file (conflict)"
        );
        assert_eq!(refusal(""), "no body");
        assert_eq!(refusal("upstream timed out"), "upstream timed out");
        assert_eq!(refusal(&"x".repeat(500)).len(), 200);
    }
}
