//! A runtime-configured **HTTP reranker** (P-2): the cross-encoder precision
//! stage that re-scores the wide hybrid-recall pool. Rather than baking a
//! reranker model into each binary, point at a TEI/Cohere-style `/rerank`
//! endpoint the operator runs (text-embeddings-inference, Jina, Cohere, …). A
//! single-vector dense retriever has a dimension-bounded recall ceiling; a
//! cross-encoder reads the query and each candidate *jointly*, so it reorders
//! the recalled candidates far more precisely than the bi-encoder scores can.
//!
//! The reranker is applied **downstream of** [`recall_hybrid`] by the caller
//! (the MCP server), so the persistence layer stays free of any model/HTTP
//! concern — exactly as the [`Embedder`](antumbra_core::ports::Embedder) lives
//! outside `antumbra-store`. The irreducible network call is isolated behind
//! [`RerankTransport`] so request shaping and response parsing are mock-tested
//! offline, with the real `ureq` POST covered by a gated `#[ignore]` test.
//!
//! Security: the endpoint URL and bearer key are **operator-configured**
//! (`--rerank-url` / `ANTUMBRA_RERANK_URL` / `ANTUMBRA_RERANK_KEY`) and are never
//! derived from tenant or stored data, so this is not an SSRF sink. If a future
//! feature ever lets a request choose the URL, validate it against
//! localhost/private ranges first. The key rides the `Authorization` header
//! (never the URL, never logged; transport errors carry only the URL), and
//! [`HttpReranker`] deliberately has no `Debug` impl, so the key cannot leak
//! through `{:?}`.

pub mod floor;

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use antumbra_core::ports::Reranker;
use antumbra_core::{AntumbraError, Result};

/// The one network operation an [`HttpReranker`] performs: POST a JSON body to a
/// rerank endpoint and return the parsed JSON response. Behind a trait so the
/// reranker's logic is testable without a socket.
pub trait RerankTransport: Send + Sync {
    fn post(&self, url: &str, api_key: Option<&str>, body: &Value) -> Result<Value>;
}

/// Default per-request budget for the rerank endpoint, overridable with
/// `ANTUMBRA_RERANK_TIMEOUT_SECS`. Without a bound a hung or unreachable endpoint
/// pins the blocking worker forever; because the HTTP server runs a tool (and so
/// this rerank) under its per-request lock, an unbounded call stalls every
/// tenant.
const RERANK_TIMEOUT_SECS: u64 = 30;

/// The production transport: a blocking `ureq` POST over an agent with a bounded
/// global timeout, so a dead endpoint fails fast instead of hanging the worker.
struct UreqTransport {
    agent: ureq::Agent,
}

impl UreqTransport {
    fn new() -> Self {
        let secs = std::env::var("ANTUMBRA_RERANK_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .filter(|&s| s > 0)
            .unwrap_or(RERANK_TIMEOUT_SECS);
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(std::time::Duration::from_secs(secs)))
            .build()
            .into();
        Self { agent }
    }
}

impl RerankTransport for UreqTransport {
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
            .map_err(|e| AntumbraError::other(format!("rerank POST {url} failed: {e}")))?;
        resp.body_mut()
            .read_json::<Value>()
            .map_err(|e| AntumbraError::other(format!("rerank response was not JSON: {e}")))
    }
}

/// Re-scores candidates by calling a TEI/Cohere-style `/rerank` endpoint.
pub struct HttpReranker {
    url: String,
    /// Model name sent in the body. TEI ignores it; Cohere/Jina require it, so it
    /// is omitted from the JSON when `None`.
    model: Option<String>,
    api_key: Option<String>,
    transport: Arc<dyn RerankTransport>,
}

impl HttpReranker {
    /// Point at `url` (the full `/rerank` endpoint), optionally naming `model`
    /// (omit for TEI) and bearer-authed.
    pub fn new(url: String, model: Option<String>, api_key: Option<String>) -> Self {
        Self {
            url,
            model,
            api_key,
            transport: Arc::new(UreqTransport::new()),
        }
    }

    /// The rerank request body. PERISHABLE wire format, isolated here: TEI takes
    /// `{query, texts, raw_scores}`; Cohere/Jina additionally want a `model` (and
    /// call the field `documents`, but accept `texts` via TEI-compatible servers).
    /// `raw_scores: false` keeps the endpoint's normalized relevance scores.
    fn request_body(&self, query: &str, texts: &[&str]) -> Value {
        let mut body = json!({ "query": query, "texts": texts, "raw_scores": false });
        if let Some(model) = &self.model {
            body["model"] = json!(model);
        }
        body
    }

    /// Pull the scored-candidate order out of a rerank response, tolerant of the
    /// two common shapes: a bare top-level array of `{index, score}` (TEI) or
    /// `{ "results": [{index, relevance_score}] }` (Cohere/Jina). Returns the
    /// candidate indices best-first. Out-of-range indices are dropped; the result
    /// is re-sorted by score defensively rather than trusting the wire order.
    fn parse(resp: &Value, n: usize) -> Result<Vec<usize>> {
        let arr = resp
            .as_array()
            .or_else(|| resp.get("results").and_then(Value::as_array))
            .ok_or_else(|| {
                AntumbraError::other("rerank response was not a scored array".to_string())
            })?;
        let mut scored: Vec<(usize, f64)> = arr
            .iter()
            .filter_map(|o| {
                let idx = o.get("index").and_then(Value::as_u64)? as usize;
                let score = o
                    .get("score")
                    .or_else(|| o.get("relevance_score"))
                    .and_then(Value::as_f64)?;
                (idx < n).then_some((idx, score))
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored.into_iter().map(|(i, _)| i).collect())
    }

    /// The same response, keeping the SCORES and restoring the caller's order.
    ///
    /// [`parse`](Self::parse) throws the scores away because ordering is all a
    /// precision stage needs. A relevance floor needs the magnitudes, so this
    /// returns one score per input position rather than a permutation. A
    /// candidate the endpoint omitted scores [`f32::NEG_INFINITY`], which reads
    /// as "not relevant" through any monotone calibration and cannot be mistaken
    /// for a real low score.
    fn parse_scores(resp: &Value, n: usize) -> Result<Vec<f32>> {
        let arr = resp
            .as_array()
            .or_else(|| resp.get("results").and_then(Value::as_array))
            .ok_or_else(|| {
                AntumbraError::other("rerank response was not a scored array".to_string())
            })?;
        let mut out = vec![f32::NEG_INFINITY; n];
        for o in arr {
            let Some(idx) = o.get("index").and_then(Value::as_u64).map(|i| i as usize) else {
                continue;
            };
            let Some(score) = o
                .get("score")
                .or_else(|| o.get("relevance_score"))
                .and_then(Value::as_f64)
            else {
                continue;
            };
            if idx < n {
                out[idx] = score as f32;
            }
        }
        Ok(out)
    }
}

#[async_trait]
impl antumbra_core::ports::RelevanceScorer for HttpReranker {
    async fn relevance(&self, query: &str, texts: &[String]) -> Result<Vec<f32>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let url = self.url.clone();
        let api_key = self.api_key.clone();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let body = self.request_body(query, &refs);
        let transport = self.transport.clone();
        let resp =
            tokio::task::spawn_blocking(move || transport.post(&url, api_key.as_deref(), &body))
                .await
                .map_err(|e| AntumbraError::other(format!("rerank task panicked: {e}")))??;
        Self::parse_scores(&resp, texts.len())
    }
}

#[async_trait]
impl Reranker for HttpReranker {
    async fn rerank(&self, query: &str, candidates: &[(String, String)]) -> Result<Vec<String>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        // `ureq` is blocking; run it off the async runtime so it never stalls a
        // worker (the same offload the embedder uses).
        let url = self.url.clone();
        let api_key = self.api_key.clone();
        let texts: Vec<&str> = candidates.iter().map(|(_, t)| t.as_str()).collect();
        let body = self.request_body(query, &texts);
        let transport = self.transport.clone();
        let resp =
            tokio::task::spawn_blocking(move || transport.post(&url, api_key.as_deref(), &body))
                .await
                .map_err(|e| AntumbraError::other(format!("rerank task panicked: {e}")))??;

        let order = Self::parse(&resp, candidates.len())?;
        // Map indices back to ids, then append any ids the endpoint dropped in
        // their original order, so the return is always a full permutation (the
        // caller's id->row reorder can never silently shrink below the requested
        // k because of a misbehaving endpoint).
        let mut ids: Vec<String> = order.iter().map(|&i| candidates[i].0.clone()).collect();
        if ids.len() < candidates.len() {
            let seen: std::collections::HashSet<String> = ids.iter().cloned().collect();
            for (id, _) in candidates {
                if !seen.contains(id) {
                    ids.push(id.clone());
                }
            }
        }
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Returns a canned response, so the request shaping + parsing are tested
    /// without a socket.
    struct FakeTransport(Result<Value>);

    impl RerankTransport for FakeTransport {
        fn post(&self, _url: &str, _api_key: Option<&str>, _body: &Value) -> Result<Value> {
            self.0
                .as_ref()
                .map(Clone::clone)
                .map_err(|e| AntumbraError::other(e.to_string()))
        }
    }

    fn reranker(resp: Result<Value>) -> HttpReranker {
        HttpReranker {
            url: "http://localhost/rerank".into(),
            model: None,
            api_key: None,
            transport: Arc::new(FakeTransport(resp)),
        }
    }

    fn candidates() -> Vec<(String, String)> {
        vec![
            ("a".into(), "alpha text".into()),
            ("b".into(), "bravo text".into()),
            ("c".into(), "charlie text".into()),
        ]
    }

    #[tokio::test]
    async fn parses_tei_array_and_reorders_best_first() {
        // index 2 (id "c") is most relevant, then 0 ("a"), then 1 ("b").
        let resp = json!([
            { "index": 1, "score": 0.10 },
            { "index": 0, "score": 0.80 },
            { "index": 2, "score": 0.95 },
        ]);
        let out = reranker(Ok(resp)).rerank("q", &candidates()).await.unwrap();
        assert_eq!(out, vec!["c", "a", "b"]);
    }

    #[tokio::test]
    async fn parses_cohere_results_shape() {
        let resp = json!({ "results": [
            { "index": 0, "relevance_score": 0.9 },
            { "index": 2, "relevance_score": 0.5 },
            { "index": 1, "relevance_score": 0.1 },
        ]});
        let out = reranker(Ok(resp)).rerank("q", &candidates()).await.unwrap();
        assert_eq!(out, vec!["a", "c", "b"]);
    }

    #[tokio::test]
    async fn appends_dropped_ids_so_result_is_a_permutation() {
        // Endpoint scored only index 1; the other ids must still come back.
        let resp = json!([{ "index": 1, "score": 0.9 }]);
        let out = reranker(Ok(resp)).rerank("q", &candidates()).await.unwrap();
        assert_eq!(out.len(), 3, "every id returned: {out:?}");
        assert_eq!(out[0], "b", "the scored id leads");
        assert!(out.contains(&"a".to_string()) && out.contains(&"c".to_string()));
    }

    #[tokio::test]
    async fn ignores_out_of_range_index() {
        let resp = json!([
            { "index": 9, "score": 0.99 },
            { "index": 0, "score": 0.5 },
        ]);
        let out = reranker(Ok(resp)).rerank("q", &candidates()).await.unwrap();
        // index 9 dropped; "a" (0) leads, the rest appended.
        assert_eq!(out[0], "a");
        assert_eq!(out.len(), 3);
    }

    #[tokio::test]
    async fn errors_on_a_malformed_response() {
        assert!(reranker(Ok(json!({ "unexpected": true })))
            .rerank("q", &candidates())
            .await
            .is_err());
    }

    #[tokio::test]
    async fn propagates_a_transport_error() {
        assert!(reranker(Err(AntumbraError::other("connection refused")))
            .rerank("q", &candidates())
            .await
            .is_err());
    }

    #[tokio::test]
    async fn empty_candidates_is_empty_without_a_call() {
        // No transport call needed; returns empty.
        let out = reranker(Err(AntumbraError::other("must not be called")))
            .rerank("q", &[])
            .await
            .unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn request_body_is_tei_shaped() {
        let r = reranker(Ok(Value::Null));
        let body = r.request_body("the query", &["t0", "t1"]);
        assert_eq!(body["query"], "the query");
        assert_eq!(body["texts"], json!(["t0", "t1"]));
        assert_eq!(body["raw_scores"], false);
        assert!(body.get("model").is_none(), "model omitted when None");
    }

    #[test]
    fn request_body_includes_model_when_set() {
        let r = HttpReranker {
            url: "http://localhost/rerank".into(),
            model: Some("rerank-v2".into()),
            api_key: None,
            transport: Arc::new(FakeTransport(Ok(Value::Null))),
        };
        let body = r.request_body("q", &["t"]);
        assert_eq!(body["model"], "rerank-v2");
    }

    // The irreducible real POST: point `ANTUMBRA_RERANK_URL` at a running
    // TEI/Cohere-style `/rerank` endpoint. Opt-in; it needs a server, so it
    // cannot run offline.
    #[tokio::test]
    #[ignore = "needs a live rerank endpoint (set ANTUMBRA_RERANK_URL)"]
    async fn real_endpoint_reorders_candidates() {
        let Ok(url) = std::env::var("ANTUMBRA_RERANK_URL") else {
            return;
        };
        let model = std::env::var("ANTUMBRA_RERANK_MODEL").ok();
        let key = std::env::var("ANTUMBRA_RERANK_KEY").ok();
        let cands = vec![
            ("doc:far".into(), "an unrelated note about gardening".into()),
            (
                "doc:near".into(),
                "the quarterly revenue beat analyst estimates".into(),
            ),
        ];
        let out = HttpReranker::new(url, model, key)
            .rerank("company earnings results", &cands)
            .await
            .unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], "doc:near", "the on-topic doc should rank first");
    }
}
