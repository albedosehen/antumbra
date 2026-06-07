//! A runtime-configured **HTTP embedder** (P-1c): instead of the build-time
//! choice between the candle BERT model and the byte-histogram fake, the server
//! can POST to an OpenAI-compatible `/embeddings` endpoint the operator runs
//! (Ollama, llama.cpp, text-embeddings-inference, …). This keeps the embedding
//! step on the tenant's side without baking a model into the binary.
//!
//! The endpoint **must** return `EMBED_DIM`-wide vectors (the HNSW index has a
//! fixed dimension); a mismatch is rejected rather than silently corrupting
//! recall. The irreducible network call is isolated behind [`EmbedTransport`] so
//! the request shaping, response parsing, and dimension check are mock-tested
//! offline, with the real `ureq` POST covered by a gated `#[ignore]` test.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use antumbra_core::ports::Embedder;
use antumbra_core::{AntumbraError, Result};
use antumbra_store::EMBED_DIM;

/// The one network operation an [`HttpEmbedder`] performs: POST a JSON body to an
/// embeddings endpoint and return the parsed JSON response. Behind a trait so the
/// embedder's logic is testable without a socket.
pub trait EmbedTransport: Send + Sync {
    fn post(&self, url: &str, api_key: Option<&str>, body: &Value) -> Result<Value>;
}

/// The production transport: a blocking `ureq` POST.
struct UreqTransport;

impl EmbedTransport for UreqTransport {
    fn post(&self, url: &str, api_key: Option<&str>, body: &Value) -> Result<Value> {
        let mut req = ureq::post(url).header("content-type", "application/json");
        if let Some(key) = api_key {
            req = req.header("authorization", &format!("Bearer {key}"));
        }
        let mut resp = req
            .send_json(body)
            .map_err(|e| AntumbraError::other(format!("embed POST {url} failed: {e}")))?;
        resp.body_mut()
            .read_json::<Value>()
            .map_err(|e| AntumbraError::other(format!("embed response was not JSON: {e}")))
    }
}

/// Embeds text by calling an OpenAI-compatible `/embeddings` endpoint.
pub struct HttpEmbedder {
    url: String,
    model: String,
    api_key: Option<String>,
    transport: Arc<dyn EmbedTransport>,
}

impl HttpEmbedder {
    /// Point at `url` (the full endpoint), naming `model`, optionally bearer-authed.
    pub fn new(url: String, model: String, api_key: Option<String>) -> Self {
        Self {
            url,
            model,
            api_key,
            transport: Arc::new(UreqTransport),
        }
    }

    fn request_body(&self, text: &str) -> Value {
        json!({ "model": self.model, "input": text })
    }

    /// Pull the embedding out of an OpenAI-shaped response (`data[0].embedding`)
    /// and enforce the index dimension.
    fn parse(resp: &Value) -> Result<Vec<f32>> {
        let embedding = resp
            .get("data")
            .and_then(|d| d.get(0))
            .and_then(|e| e.get("embedding"))
            .ok_or_else(|| {
                AntumbraError::other("embed response missing `data[0].embedding`".to_string())
            })?;
        let vector: Vec<f32> = serde_json::from_value(embedding.clone())
            .map_err(|e| AntumbraError::other(format!("embedding was not a float array: {e}")))?;
        if vector.len() != EMBED_DIM {
            return Err(AntumbraError::other(format!(
                "embedder returned dimension {} but the HNSW index needs {EMBED_DIM}",
                vector.len()
            )));
        }
        Ok(vector)
    }
}

#[async_trait]
impl Embedder for HttpEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // `ureq` is blocking; run it off the async runtime so it never stalls a
        // worker (the same offload the model loaders use).
        let url = self.url.clone();
        let api_key = self.api_key.clone();
        let body = self.request_body(text);
        let transport = self.transport.clone();
        let resp =
            tokio::task::spawn_blocking(move || transport.post(&url, api_key.as_deref(), &body))
                .await
                .map_err(|e| AntumbraError::other(format!("embed task panicked: {e}")))??;
        Self::parse(&resp)
    }

    fn dim(&self) -> usize {
        EMBED_DIM
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Returns a canned response, so the request shaping + parsing are tested
    /// without a socket.
    struct FakeTransport(Result<Value>);

    impl EmbedTransport for FakeTransport {
        fn post(&self, _url: &str, _api_key: Option<&str>, _body: &Value) -> Result<Value> {
            self.0
                .as_ref()
                .map(Clone::clone)
                .map_err(|e| AntumbraError::other(e.to_string()))
        }
    }

    fn embedder(resp: Result<Value>) -> HttpEmbedder {
        HttpEmbedder {
            url: "http://localhost/embeddings".into(),
            model: "all-MiniLM-L6-v2".into(),
            api_key: None,
            transport: Arc::new(FakeTransport(resp)),
        }
    }

    #[tokio::test]
    async fn parses_an_openai_shaped_embedding() {
        let resp = json!({ "data": [ { "embedding": vec![0.25f32; EMBED_DIM] } ] });
        let v = embedder(Ok(resp)).embed("hello").await.unwrap();
        assert_eq!(v.len(), EMBED_DIM);
        assert!((v[0] - 0.25).abs() < 1e-6);
    }

    #[tokio::test]
    async fn rejects_a_wrong_dimension() {
        let resp = json!({ "data": [ { "embedding": vec![0.1f32; 8] } ] });
        assert!(embedder(Ok(resp)).embed("hello").await.is_err());
    }

    #[tokio::test]
    async fn errors_on_a_malformed_response() {
        assert!(embedder(Ok(json!({ "unexpected": true })))
            .embed("hello")
            .await
            .is_err());
    }

    #[tokio::test]
    async fn propagates_a_transport_error() {
        assert!(embedder(Err(AntumbraError::other("connection refused")))
            .embed("hello")
            .await
            .is_err());
    }

    #[test]
    fn request_body_is_openai_shaped() {
        let e = embedder(Ok(Value::Null));
        let body = e.request_body("hi");
        assert_eq!(body["model"], "all-MiniLM-L6-v2");
        assert_eq!(body["input"], "hi");
    }

    // The irreducible real POST: point `ANTUMBRA_EMBED_URL` at a running
    // OpenAI-compatible embeddings endpoint (e.g. text-embeddings-inference with a
    // 384-dim model). Opt-in; it needs a server, so it cannot run offline.
    #[tokio::test]
    #[ignore = "needs a live embeddings endpoint (set ANTUMBRA_EMBED_URL)"]
    async fn real_endpoint_returns_the_right_dimension() {
        let Ok(url) = std::env::var("ANTUMBRA_EMBED_URL") else {
            return;
        };
        let model =
            std::env::var("ANTUMBRA_EMBED_MODEL").unwrap_or_else(|_| "all-MiniLM-L6-v2".into());
        let key = std::env::var("ANTUMBRA_EMBED_KEY").ok();
        let v = HttpEmbedder::new(url, model, key)
            .embed("a probe sentence")
            .await
            .unwrap();
        assert_eq!(v.len(), EMBED_DIM);
    }
}
