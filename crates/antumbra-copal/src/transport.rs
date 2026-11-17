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
        // The shared agent: a refusal comes back as a response rather than a
        // bare status error, so the ingest can say what copal said.
        Self {
            agent: antumbra_http::agent(antumbra_http::patience(
                "ANTUMBRA_COPAL_TIMEOUT_SECS",
                COPAL_TIMEOUT_SECS,
            )),
        }
    }
}

impl CopalTransport for UreqTransport {
    fn post_json(&self, url: &str, credential: &CopalCredential, body: &Value) -> Result<Value> {
        let req = self.agent.post(url);
        let resp = authed(req, credential)
            .send_json(body)
            .map_err(|e| AntumbraError::other(format!("copal POST {url} failed: {e}")))?;
        antumbra_http::read_json_or_refusal(&format!("copal POST {url}"), resp)
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
        antumbra_http::read_json_or_refusal(&format!("copal PUT {url}"), resp)
    }

    fn get_bytes(&self, url: &str, credential: &CopalCredential, limit: u64) -> Result<Vec<u8>> {
        let mut resp = authed(self.agent.get(url), credential)
            .call()
            .map_err(|e| AntumbraError::other(format!("copal GET {url} failed: {e}")))?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.body_mut().read_to_string().unwrap_or_default();
            return Err(AntumbraError::other(format!(
                "copal GET {url} answered {}: {}",
                status.as_u16(),
                antumbra_http::refusal(&text)
            )));
        }
        resp.body_mut()
            .with_config()
            .limit(limit)
            .read_to_vec()
            .map_err(|e| AntumbraError::other(format!("copal GET {url} body: {e}")))
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
    use anyhow::Context;
    use serde_json::json;

    use antumbra_http::testing::serve;

    use super::*;

    /// Copal's refusal reaches the ingest in copal's words, through the shared
    /// reader.
    #[test]
    fn a_refusals_body_is_surfaced() -> anyhow::Result<()> {
        let (url, served) = serve(
            1,
            409,
            r#"{"error":{"kind":"conflict","message":"idempotency key consumed by a deleted file"}}"#,
        )?;
        let answer = UreqTransport::new().post_json(
            &url,
            &CopalCredential::Tenant("ws:test".into()),
            &json!({ "title": "t" }),
        );
        let Err(e) = answer else {
            anyhow::bail!("a 409 was accepted");
        };
        let shown = e.to_string();
        if !(shown.starts_with("copal POST http://")
            && shown
                .contains("answered 409: idempotency key consumed by a deleted file (conflict)"))
        {
            anyhow::bail!("{shown}");
        }
        let head = served
            .requests()?
            .into_iter()
            .next()
            .context("one request")?
            .to_lowercase();
        if !head.contains("x-copal-tenant: ws:test") {
            anyhow::bail!("{head}");
        }
        Ok(())
    }
}
