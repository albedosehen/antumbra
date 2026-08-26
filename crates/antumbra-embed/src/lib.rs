//! A runtime-configured **HTTP embedder** (P-1c): rather than baking an embedding
//! model into each binary, point at an OpenAI-compatible `/embeddings` endpoint
//! the operator runs (Ollama, llama.cpp, text-embeddings-inference, …). Shared by
//! the MCP server and the operator console so a route/ask uses the *same*
//! embedding the population's capability vectors were built with.
//!
//! By default the endpoint **must** return `EMBED_DIM`-wide vectors (the HNSW
//! index has a fixed dimension); a mismatch is rejected rather than silently
//! corrupting recall. Optionally, an operator selects the **Matryoshka** path by
//! configuring a `source_dim` (e.g. `1024` for BGE-M3 / multilingual-e5): the
//! endpoint then returns `source_dim`-wide vectors and this embedder stores the
//! re-normalized leading `EMBED_DIM` prefix (see
//! [`antumbra_core::truncate_renormalize`]). Either way the vector handed to the
//! index is exactly `EMBED_DIM`-wide, so the HNSW invariant holds and a richer
//! generalist model is a *configuration*, not an index migration. The
//! irreducible network call is isolated behind [`EmbedTransport`] so the request
//! shaping, response parsing, and dimension handling are mock-tested offline,
//! with the real `ureq` POST covered by a gated `#[ignore]` test.
//!
//! Security: the endpoint URL and bearer key are **operator-configured**
//! (`--embed-url` / `ANTUMBRA_EMBED_URL` / `ANTUMBRA_EMBED_KEY`) and are never
//! derived from tenant or stored data, so this is not an SSRF sink. If a future
//! feature ever lets a request choose the URL, validate it against
//! localhost/private ranges first. The key rides the `Authorization` header
//! (never the URL, never logged; transport errors carry only the URL), and
//! [`HttpEmbedder`] deliberately has no `Debug` impl, so the key cannot leak
//! through `{:?}`.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use antumbra_core::ports::Embedder;
use antumbra_core::{truncate_renormalize, AntumbraError, Result};
use antumbra_store::EMBED_DIM;

/// The one network operation an [`HttpEmbedder`] performs: POST a JSON body to an
/// embeddings endpoint and return the parsed JSON response. Behind a trait so the
/// embedder's logic is testable without a socket.
pub trait EmbedTransport: Send + Sync {
    fn post(&self, url: &str, api_key: Option<&str>, body: &Value) -> Result<Value>;
}

/// Default per-request budget for the embeddings endpoint, overridable with
/// `ANTUMBRA_EMBED_TIMEOUT_SECS`. Without a bound a hung or unreachable endpoint
/// pins the blocking worker forever; because the HTTP server runs a tool (and so
/// this embed) under its per-request lock, an unbounded call stalls every tenant.
const EMBED_TIMEOUT_SECS: u64 = 30;

/// Default cap on the text shipped to the embeddings endpoint, in characters,
/// overridable with `ANTUMBRA_EMBED_MAX_CHARS`. The 384-dim sentence models
/// this store targets (all-MiniLM and friends) have a 512-token window, and
/// llama.cpp answers over-window input with a hard 500 rather than truncating
/// server-side -- so before this cap, every long memory FAILED to store (hit
/// live on the first Kushtakas import: 1737 of 4720 records exceeded the
/// window). The caller still stores the full content; only the text the vector
/// is computed from is capped, which is the inherent trade of a small-window
/// embedder, not data loss. 1200 chars stays under 512 tokens for English
/// prose and code (~2.5-3.5 chars/token); operators running a longer-window
/// endpoint can raise it.
const EMBED_MAX_CHARS: usize = 1200;

fn embed_max_chars() -> usize {
    static CACHE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("ANTUMBRA_EMBED_MAX_CHARS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(EMBED_MAX_CHARS)
    })
}

/// The leading `max` characters of `text` as a slice -- no allocation, and the
/// cut lands on a `char` boundary by construction (`char_indices` yields only
/// boundaries), so multi-byte text can never be split mid-codepoint.
fn cap_chars(text: &str, max: usize) -> &str {
    match text.char_indices().nth(max) {
        Some((idx, _)) => &text[..idx],
        None => text,
    }
}

/// The production transport: a blocking `ureq` POST over an agent with a bounded
/// global timeout, so a dead endpoint fails fast instead of hanging the worker.
struct UreqTransport {
    agent: ureq::Agent,
}

impl UreqTransport {
    fn new() -> Self {
        let secs = std::env::var("ANTUMBRA_EMBED_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .filter(|&s| s > 0)
            .unwrap_or(EMBED_TIMEOUT_SECS);
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(std::time::Duration::from_secs(secs)))
            .build()
            .into();
        Self { agent }
    }
}

impl EmbedTransport for UreqTransport {
    fn post(&self, url: &str, api_key: Option<&str>, body: &Value) -> Result<Value> {
        let mut req = self
            .agent
            .post(url)
            .header("content-type", "application/json");
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
    /// The Matryoshka source dimension. `None` = strict: require exactly
    /// `EMBED_DIM`. `Some(n)` = expect `n`-wide vectors and store the
    /// re-normalized leading `EMBED_DIM` prefix. See [`Self::new_with_dim`].
    source_dim: Option<u32>,
    transport: Arc<dyn EmbedTransport>,
}

impl HttpEmbedder {
    /// Point at `url` (the full endpoint), naming `model`, optionally
    /// bearer-authed. Strict-dimension path: the endpoint must return exactly
    /// `EMBED_DIM`-wide vectors.
    pub fn new(url: String, model: String, api_key: Option<String>) -> Self {
        Self::new_with_dim(url, model, api_key, None)
    }

    /// As [`Self::new`], but selecting the embedder's dimension behavior.
    ///
    /// `source_dim`:
    /// - `None` — strict: the endpoint must return exactly `EMBED_DIM`-wide
    ///   vectors (the legacy/default behavior).
    /// - `Some(n)` — Matryoshka: the endpoint returns `n`-wide vectors and this
    ///   embedder stores the re-normalized leading `EMBED_DIM` prefix. `n` is
    ///   expected to exceed `EMBED_DIM` (a longer generalist embedding); a
    ///   returned vector whose length is not exactly `n` is rejected.
    pub fn new_with_dim(
        url: String,
        model: String,
        api_key: Option<String>,
        source_dim: Option<u32>,
    ) -> Self {
        Self {
            url,
            model,
            api_key,
            source_dim,
            transport: Arc::new(UreqTransport::new()),
        }
    }

    fn request_body_capped(&self, text: &str, cap: usize) -> Value {
        json!({ "model": self.model, "input": cap_chars(text, cap) })
    }

    /// Pull the embedding out of an OpenAI-shaped response (`data[0].embedding`)
    /// and reconcile it with the index dimension. Strict mode rejects anything
    /// but `EMBED_DIM`; Matryoshka mode requires the configured `source_dim`
    /// then truncates + re-normalizes to `EMBED_DIM` (always index-wide on exit).
    fn parse(&self, resp: &Value) -> Result<Vec<f32>> {
        let embedding = resp
            .get("data")
            .and_then(|d| d.get(0))
            .and_then(|e| e.get("embedding"))
            .ok_or_else(|| {
                AntumbraError::other("embed response missing `data[0].embedding`".to_string())
            })?;
        let vector: Vec<f32> = serde_json::from_value(embedding.clone())
            .map_err(|e| AntumbraError::other(format!("embedding was not a float array: {e}")))?;
        match self.source_dim {
            // Matryoshka: require the model's full width, then store the
            // re-normalized EMBED_DIM prefix.
            Some(n) => {
                let expected = n as usize;
                if vector.len() != expected {
                    return Err(AntumbraError::other(format!(
                        "embedder returned dimension {} but the configured Matryoshka \
                         source dimension is {expected}",
                        vector.len()
                    )));
                }
                Ok(truncate_renormalize(vector, EMBED_DIM))
            }
            // Strict: the endpoint must already produce the index dimension.
            None => {
                if vector.len() != EMBED_DIM {
                    return Err(AntumbraError::other(format!(
                        "embedder returned dimension {} but the HNSW index needs {EMBED_DIM}",
                        vector.len()
                    )));
                }
                Ok(vector)
            }
        }
    }
}

#[async_trait]
impl Embedder for HttpEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // The char cap only approximates the model's TOKEN window, and the
        // chars-per-token ratio swings with content (dense technical text
        // runs ~2.3, prose ~4) -- so a fixed cap that fits most memories
        // still lands a hard 500 on the densest ones. Rather than chase a
        // magic number, halve the cap and retry on failure: self-adapting to
        // whatever tokenizer sits behind the endpoint. A genuinely dead
        // endpoint just fails all three attempts and reports the last error.
        let mut cap = embed_max_chars();
        let mut last_err = AntumbraError::other("embed: no attempt made");
        for _ in 0..3 {
            // `ureq` is blocking; run it off the async runtime so it never
            // stalls a worker (the same offload the model loaders use).
            let url = self.url.clone();
            let api_key = self.api_key.clone();
            let body = self.request_body_capped(text, cap);
            let transport = self.transport.clone();
            let resp = tokio::task::spawn_blocking(move || {
                transport.post(&url, api_key.as_deref(), &body)
            })
            .await
            .map_err(|e| AntumbraError::other(format!("embed task panicked: {e}")))?;
            match resp {
                Ok(resp) => return self.parse(&resp),
                Err(e) => {
                    last_err = e;
                    cap /= 2;
                    if cap == 0 {
                        break;
                    }
                }
            }
        }
        Err(last_err)
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
        embedder_with_dim(resp, None)
    }

    fn embedder_with_dim(resp: Result<Value>, source_dim: Option<u32>) -> HttpEmbedder {
        HttpEmbedder {
            url: "http://localhost/embeddings".into(),
            model: "all-MiniLM-L6-v2".into(),
            api_key: None,
            source_dim,
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
    async fn matryoshka_truncates_and_renormalizes_to_unit_norm() {
        // A 1024-dim Matryoshka response with source_dim=Some(1024) yields a
        // unit-norm EMBED_DIM (384) vector: the stored vector is always the
        // index width regardless of the model's native dimension.
        let resp = json!({ "data": [ { "embedding": vec![0.5f32; 1024] } ] });
        let v = embedder_with_dim(Ok(resp), Some(1024))
            .embed("hello")
            .await
            .unwrap();
        assert_eq!(v.len(), EMBED_DIM);
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "unit norm, got {norm}");
    }

    #[tokio::test]
    async fn matryoshka_rejects_when_source_dim_mismatches() {
        // source_dim=Some(1024) but the endpoint returned 512: rejected rather
        // than silently storing a wrong-width prefix.
        let resp = json!({ "data": [ { "embedding": vec![0.1f32; 512] } ] });
        assert!(embedder_with_dim(Ok(resp), Some(1024))
            .embed("hello")
            .await
            .is_err());
    }

    #[tokio::test]
    async fn matryoshka_rejects_an_exact_embed_dim_response_when_source_is_longer() {
        // With source_dim=Some(1024) a bare EMBED_DIM response is the wrong
        // width for the configured Matryoshka source and is rejected.
        let resp = json!({ "data": [ { "embedding": vec![0.25f32; EMBED_DIM] } ] });
        assert!(embedder_with_dim(Ok(resp), Some(1024))
            .embed("hello")
            .await
            .is_err());
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

    /// Refuses input longer than a token-ish budget, like llama.cpp answering
    /// an over-window request with a 500 -- so the halve-and-retry ladder in
    /// `embed` is what gets exercised, not the happy path.
    struct WindowedTransport {
        max_input_chars: usize,
    }

    impl EmbedTransport for WindowedTransport {
        fn post(&self, _url: &str, _api_key: Option<&str>, body: &Value) -> Result<Value> {
            let input = body["input"].as_str().unwrap_or_default();
            if input.chars().count() > self.max_input_chars {
                return Err(AntumbraError::other("http status: 500"));
            }
            Ok(json!({ "data": [ { "embedding": vec![0.25f32; EMBED_DIM] } ] }))
        }
    }

    #[tokio::test]
    async fn dense_input_that_still_overflows_the_window_retries_at_a_halved_cap() {
        // The endpoint's real window sits BELOW the char cap (the dense-text
        // case: ~2.3 chars/token puts 1200 chars past 512 tokens). First
        // attempt 500s; the halved retry fits and succeeds.
        let e = HttpEmbedder {
            url: "http://localhost/embeddings".into(),
            model: "all-MiniLM-L6-v2".into(),
            api_key: None,
            source_dim: None,
            transport: Arc::new(WindowedTransport {
                max_input_chars: EMBED_MAX_CHARS / 2,
            }),
        };
        let long = "x".repeat(EMBED_MAX_CHARS * 2);
        let v = e.embed(&long).await.unwrap();
        assert_eq!(v.len(), EMBED_DIM);
    }

    #[test]
    fn request_body_is_openai_shaped() {
        let e = embedder(Ok(Value::Null));
        let body = e.request_body_capped("hi", embed_max_chars());
        assert_eq!(body["model"], "all-MiniLM-L6-v2");
        assert_eq!(body["input"], "hi");
    }

    #[test]
    fn request_body_caps_over_window_input() {
        // A memory far past the embedding window ships only its head -- the
        // endpoint must never see input it answers with a hard 500.
        let e = embedder(Ok(Value::Null));
        let long = "x".repeat(EMBED_MAX_CHARS * 4);
        let body = e.request_body_capped(&long, embed_max_chars());
        assert_eq!(
            body["input"].as_str().unwrap().chars().count(),
            EMBED_MAX_CHARS
        );
    }

    #[test]
    fn cap_chars_counts_characters_not_bytes_and_keeps_short_input_whole() {
        // Multi-byte text: the cap is a character count and the cut cannot
        // split a codepoint (a byte-indexed slice here would panic).
        let s = "é".repeat(10);
        assert_eq!(cap_chars(&s, 4), "éééé");
        // Under the cap: same slice back, untouched.
        assert_eq!(cap_chars("short", 1200), "short");
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
