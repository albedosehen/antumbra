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

/// The most texts sent in one request, overridable with `ANTUMBRA_RERANK_BATCH`.
///
/// text-embeddings-inference refuses a request carrying more than its
/// `--max-client-batch-size` texts, 32 by default, with a 422. Recall sends a
/// pool of up to 100 candidates -- ten per requested row, more when a repo scope
/// widens it -- so an unbatched call failed whenever a caller asked for four or
/// more rows, and recall fell back to the fused order without its precision
/// stage. The per-prompt hook asks for more than that on every prompt. Batches
/// are sent one after another; a cross-encoder scores each (query, text) pair on
/// its own, so scores from different batches compare directly.
const RERANK_MAX_BATCH: usize = 32;

/// The most one scoring call may take across all of its batches, overridable
/// with `ANTUMBRA_RERANK_BUDGET_MS`.
///
/// Recall waits on the cross-encoder before it answers, and the callers that
/// recall most are hooks with a few seconds to spend: the session bootstrap gives
/// up after five, the per-prompt hook after ten. On a CPU, `bge-reranker-base`
/// scores a batch of 32 memories in 3.5 to 5 seconds, so the 100-candidate pool
/// of a twelve-row recall took 13 seconds and every session started cold. Past
/// the budget the call fails, and recall falls back to the fused order as it
/// does for any reranker fault: a slow precision stage costs its budget, never
/// the answer.
const RERANK_BUDGET_MS: u64 = 2000;

fn max_batch() -> usize {
    std::env::var("ANTUMBRA_RERANK_BATCH")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(RERANK_MAX_BATCH)
}

fn budget() -> std::time::Duration {
    let ms = std::env::var("ANTUMBRA_RERANK_BUDGET_MS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(RERANK_BUDGET_MS);
    std::time::Duration::from_millis(ms)
}

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

/// The model a text-embeddings-inference endpoint serves, from its `/info`, if
/// it answers within two seconds: what the relevance floor's calibration is
/// chosen by. `None` for an endpoint with no `/info` (Cohere, Jina) or one not
/// up yet.
pub fn served_model(rerank_url: &str, api_key: Option<&str>) -> Option<String> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(std::time::Duration::from_secs(2)))
        .build()
        .into();
    let mut req = agent.get(&info_url(rerank_url));
    if let Some(key) = api_key {
        req = req.header("authorization", &format!("Bearer {key}"));
    }
    let info = req.call().ok()?.body_mut().read_json::<Value>().ok()?;
    model_from_info(&info)
}

/// `/info` beside the `/rerank` endpoint.
fn info_url(rerank_url: &str) -> String {
    let base = rerank_url.trim_end_matches('/');
    let base = base.strip_suffix("/rerank").unwrap_or(base);
    format!("{base}/info")
}

fn model_from_info(info: &Value) -> Option<String> {
    info.get("model_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|m| !m.is_empty())
        .map(str::to_string)
}

/// Re-scores candidates by calling a TEI/Cohere-style `/rerank` endpoint.
pub struct HttpReranker {
    url: String,
    /// Model name sent in the body. TEI ignores it; Cohere/Jina require it, so it
    /// is omitted from the JSON when `None`.
    model: Option<String>,
    api_key: Option<String>,
    transport: Arc<dyn RerankTransport>,
    /// Texts per request; see [`RERANK_MAX_BATCH`].
    max_batch: usize,
    /// Time allowed per scoring call; see [`RERANK_BUDGET_MS`].
    budget: std::time::Duration,
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
            max_batch: max_batch(),
            budget: budget(),
        }
    }

    /// The rerank request body. PERISHABLE wire format, isolated here: TEI takes
    /// `{query, texts, raw_scores}`; Cohere/Jina additionally want a `model` (and
    /// call the field `documents`, but accept `texts` via TEI-compatible servers).
    /// `raw_scores: false` keeps the endpoint's normalized relevance scores.
    /// `truncate: true` has the endpoint cut a text to the model's input window
    /// rather than refuse it: memories run to thousands of characters, past the
    /// 512 tokens a BERT-sized cross-encoder reads, and the head of a memory is
    /// what it would read anyway.
    fn request_body(&self, query: &str, texts: &[&str]) -> Value {
        let mut body = json!({
            "query": query,
            "texts": texts,
            "raw_scores": false,
            "truncate": true,
        });
        if let Some(model) = &self.model {
            body["model"] = json!(model);
        }
        body
    }

    /// A rerank response as one score per input position, tolerant of the two
    /// common shapes: a bare top-level array of `{index, score}` (TEI) or
    /// `{ "results": [{index, relevance_score}] }` (Cohere/Jina). A
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

impl HttpReranker {
    /// One score per text, in the caller's order, however many texts there are:
    /// they go to the endpoint in batches of at most `max_batch` and each
    /// batch's indices are offset back into place. Any batch failing fails the
    /// whole call, so a caller never ranks a pool that was only partly scored,
    /// and so does running past the budget: the batches not yet sent are never
    /// sent.
    async fn scores(&self, query: &str, texts: &[&str]) -> Result<Vec<f32>> {
        let scoring = async {
            let mut out = Vec::with_capacity(texts.len());
            for batch in texts.chunks(self.max_batch.max(1)) {
                let url = self.url.clone();
                let api_key = self.api_key.clone();
                let body = self.request_body(query, batch);
                let transport = self.transport.clone();
                // `ureq` is blocking; run it off the async runtime so it never
                // stalls a worker (the same offload the embedder uses).
                let resp = tokio::task::spawn_blocking(move || {
                    transport.post(&url, api_key.as_deref(), &body)
                })
                .await
                .map_err(|e| AntumbraError::other(format!("rerank task panicked: {e}")))??;
                out.extend(Self::parse_scores(&resp, batch.len())?);
            }
            Ok(out)
        };
        tokio::time::timeout(self.budget, scoring)
            .await
            .unwrap_or_else(|_| {
                Err(AntumbraError::other(format!(
                    "rerank of {} texts exceeded its {} ms budget",
                    texts.len(),
                    self.budget.as_millis()
                )))
            })
    }
}

#[async_trait]
impl antumbra_core::ports::RelevanceScorer for HttpReranker {
    async fn relevance(&self, query: &str, texts: &[String]) -> Result<Vec<f32>> {
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        self.scores(query, &refs).await
    }
}

#[async_trait]
impl Reranker for HttpReranker {
    async fn rerank(&self, query: &str, candidates: &[(String, String)]) -> Result<Vec<String>> {
        let texts: Vec<&str> = candidates.iter().map(|(_, t)| t.as_str()).collect();
        let scores = self.scores(query, &texts).await?;
        // Best first. The sort is stable, and a candidate the endpoint omitted
        // scores negative infinity, so it keeps its place at the end in the
        // order it arrived: the return is always a full permutation, and the
        // caller's id-to-row reorder can never shrink below the requested k
        // because of a misbehaving endpoint.
        let mut order: Vec<usize> = (0..candidates.len()).collect();
        order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
        Ok(order.into_iter().map(|i| candidates[i].0.clone()).collect())
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
            max_batch: RERANK_MAX_BATCH,
            budget: std::time::Duration::from_millis(RERANK_BUDGET_MS),
        }
    }

    fn candidates() -> Vec<(String, String)> {
        vec![
            ("a".into(), "alpha text".into()),
            ("b".into(), "bravo text".into()),
            ("c".into(), "charlie text".into()),
        ]
    }

    /// Behaves like text-embeddings-inference: refuses a request carrying more
    /// than `limit` texts, and otherwise scores each text as the number it
    /// spells (`"t42"` scores 0.42). Records every batch it was sent, and takes
    /// `delay` to answer each one, the way a cross-encoder on a CPU does.
    struct TeiLike {
        limit: usize,
        batches: std::sync::Mutex<Vec<usize>>,
        fail_call: Option<usize>,
        delay: std::time::Duration,
    }

    impl TeiLike {
        fn new(limit: usize) -> Self {
            Self {
                limit,
                batches: std::sync::Mutex::new(Vec::new()),
                fail_call: None,
                delay: std::time::Duration::ZERO,
            }
        }

        fn sizes(&self) -> Vec<usize> {
            self.batches.lock().map(|b| b.clone()).unwrap_or_default()
        }
    }

    impl RerankTransport for TeiLike {
        fn post(&self, _url: &str, _api_key: Option<&str>, body: &Value) -> Result<Value> {
            let texts = body["texts"].as_array().cloned().unwrap_or_default();
            let call = {
                let mut batches = self
                    .batches
                    .lock()
                    .map_err(|_| AntumbraError::other("poisoned"))?;
                batches.push(texts.len());
                batches.len() - 1
            };
            if texts.len() > self.limit {
                return Err(AntumbraError::other(format!(
                    "http status: 422 (batch size {} > maximum allowed batch size {})",
                    texts.len(),
                    self.limit
                )));
            }
            if self.fail_call == Some(call) {
                return Err(AntumbraError::other("connection reset"));
            }
            std::thread::sleep(self.delay);
            let scored: Vec<Value> = texts
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    let n: f64 = t.as_str().unwrap_or("t0")[1..].parse().unwrap_or(0.0);
                    json!({ "index": i, "score": n / 100.0 })
                })
                .collect();
            Ok(Value::Array(scored))
        }
    }

    fn batched(transport: Arc<TeiLike>) -> HttpReranker {
        HttpReranker {
            url: "http://localhost/rerank".into(),
            model: None,
            api_key: None,
            transport,
            max_batch: RERANK_MAX_BATCH,
            budget: std::time::Duration::from_millis(RERANK_BUDGET_MS),
        }
    }

    /// A pool of 100, as recall sends when a caller asks for ten rows, used to
    /// fail outright against the endpoint's 32-text limit and leave recall in
    /// its fused order. It is scored in batches, and the order is by score
    /// across all of them rather than within each.
    #[tokio::test]
    async fn a_pool_past_the_endpoints_batch_limit_is_scored_in_batches() -> Result<()> {
        let tei = Arc::new(TeiLike::new(32));
        // Scores that interleave across batch boundaries: id i scores (i * 37) % 100.
        let pool: Vec<(String, String)> = (0..100)
            .map(|i| (format!("c{i}"), format!("t{}", (i * 37) % 100)))
            .collect();
        let out = batched(tei.clone()).rerank("q", &pool).await?;
        assert_eq!(tei.sizes(), vec![32, 32, 32, 4]);
        assert_eq!(out.len(), 100);
        // The best score, 99, belongs to i = 27 (27 * 37 = 999); the worst, 0, to i = 0.
        assert_eq!(out.first().map(String::as_str), Some("c27"));
        assert_eq!(out.last().map(String::as_str), Some("c0"));
        Ok(())
    }

    #[tokio::test]
    async fn relevance_keeps_the_callers_order_across_batches() -> Result<()> {
        let tei = Arc::new(TeiLike::new(32));
        let texts: Vec<String> = (0..70).map(|i| format!("t{}", 70 - i)).collect();
        let scores =
            antumbra_core::ports::RelevanceScorer::relevance(&batched(tei.clone()), "q", &texts)
                .await?;
        assert_eq!(tei.sizes(), vec![32, 32, 6]);
        let expected: Vec<f32> = (0..70).map(|i| (70 - i) as f32 / 100.0).collect();
        assert_eq!(scores, expected);
        Ok(())
    }

    /// A batch that fails fails the call: ranking a pool that was only partly
    /// scored would put every unscored candidate last for no reason.
    #[tokio::test]
    async fn one_failed_batch_fails_the_whole_ranking() {
        let tei = Arc::new(TeiLike {
            fail_call: Some(1),
            ..TeiLike::new(32)
        });
        let pool: Vec<(String, String)> = (0..50)
            .map(|i| (format!("c{i}"), format!("t{i}")))
            .collect();
        assert!(batched(tei).rerank("q", &pool).await.is_err());
    }

    fn slow(per_batch_ms: u64) -> Arc<TeiLike> {
        Arc::new(TeiLike {
            delay: std::time::Duration::from_millis(per_batch_ms),
            ..TeiLike::new(32)
        })
    }

    fn within(tei: Arc<TeiLike>, budget_ms: u64) -> HttpReranker {
        HttpReranker {
            budget: std::time::Duration::from_millis(budget_ms),
            ..batched(tei)
        }
    }

    /// The budget covers the whole call, not each batch: four batches of 150 ms
    /// overrun 400 ms although each fits, and the call fails rather than make
    /// recall wait. Past the budget no further batch is sent, so a slow endpoint
    /// is not handed work nobody will read.
    #[tokio::test]
    async fn a_ranking_that_overruns_its_budget_fails_and_stops_sending() {
        let tei = slow(150);
        let pool: Vec<(String, String)> = (0..100)
            .map(|i| (format!("c{i}"), format!("t{i}")))
            .collect();
        let err = within(tei.clone(), 400).rerank("q", &pool).await;
        assert!(
            err.as_ref()
                .is_err_and(|e| e.to_string().contains("exceeded its 400 ms budget")),
            "{err:?}"
        );
        // Longer than a batch takes, so a batch sent late would have landed.
        tokio::time::sleep(std::time::Duration::from_millis(400)).await;
        assert!(tei.sizes().len() < 4, "sent {:?}", tei.sizes());
    }

    #[tokio::test]
    async fn a_ranking_inside_its_budget_is_untouched() -> Result<()> {
        let tei = slow(150);
        let pool: Vec<(String, String)> = (0..10)
            .map(|i| (format!("c{i}"), format!("t{i}")))
            .collect();
        let out = within(tei.clone(), 400).rerank("q", &pool).await?;
        assert_eq!(out.first().map(String::as_str), Some("c9"));
        assert_eq!(tei.sizes(), vec![10]);
        Ok(())
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
        assert_eq!(body["truncate"], true, "a long memory is cut, not refused");
        assert!(body.get("model").is_none(), "model omitted when None");
    }

    #[test]
    fn the_info_endpoint_sits_beside_rerank_and_names_the_model() {
        assert_eq!(info_url("http://rerank/rerank"), "http://rerank/info");
        assert_eq!(
            info_url("http://10.0.0.132:8091/rerank/"),
            "http://10.0.0.132:8091/info"
        );
        assert_eq!(info_url("http://host:80"), "http://host:80/info");
        let info = json!({"model_id": "Alibaba-NLP/gte-reranker-modernbert-base", "max_input_length": 8192});
        assert_eq!(
            model_from_info(&info).as_deref(),
            Some("Alibaba-NLP/gte-reranker-modernbert-base")
        );
        assert_eq!(model_from_info(&json!({"model_id": " "})), None);
        assert_eq!(model_from_info(&json!({})), None);
    }

    #[test]
    fn request_body_includes_model_when_set() {
        let r = HttpReranker {
            url: "http://localhost/rerank".into(),
            model: Some("rerank-v2".into()),
            api_key: None,
            transport: Arc::new(FakeTransport(Ok(Value::Null))),
            max_batch: RERANK_MAX_BATCH,
            budget: std::time::Duration::from_millis(RERANK_BUDGET_MS),
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
