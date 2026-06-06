//! Live harness ingestion (R-5): pull successful orchestration traces from a
//! running Kushtaka harness over its MCP HTTP surface, instead of a hand-exported
//! file, and hand them to the pure normalizer in `antumbra-train`
//! ([`traces_from_kushtaka`]). The transport is *injected*, so the fetch
//! orchestration (call body, response normalization, error / empty / malformed
//! paths) is exercised against a mock without a network; the only piece a unit
//! test cannot reach deterministically -- the real `ureq` POST to a live socket
//! -- is isolated in [`live_call`] and covered by a gated smoke test.

use serde_json::{json, Value};

use antumbra_train::{traces_from_kushtaka, HarnessTrace};

/// Where to reach a running Kushtaka harness and how to authenticate.
#[derive(Debug, Clone)]
pub struct KushtakaConfig {
    /// Base URL of the MCP engine (e.g. `http://10.0.0.110:8081`).
    pub base_url: String,
    /// API key sent as `X-API-Key` (Kushtaka rejects calls without it).
    pub api_key: String,
    /// Workspace/scope passed through on every call (Kushtaka scopes by it).
    pub scope: Option<String>,
    /// The trace-returning tool to call (`list_tasks`, `get_task_trace`,
    /// `list_behavior_graph_evaluations`, …). Its response is normalized
    /// tolerantly, so any trace-shaped payload works without a per-tool schema.
    pub tool: String,
    /// Extra JSON args merged into the call body (e.g. a `graph_id` or `limit`).
    pub args: Value,
}

/// The MCP call body Kushtaka expects: the tool name, the scope (under both the
/// `scope` and `workspace_id` keys the memory/trace tools accept), and any extra
/// args merged in at the top level.
fn call_body(cfg: &KushtakaConfig) -> Value {
    let mut body = serde_json::Map::new();
    body.insert("tool".into(), json!(cfg.tool));
    if let Some(scope) = &cfg.scope {
        body.insert("scope".into(), json!(scope));
        body.insert("workspace_id".into(), json!(scope));
    }
    if let Value::Object(extra) = &cfg.args {
        for (k, v) in extra {
            body.insert(k.clone(), v.clone());
        }
    }
    Value::Object(body)
}

/// Fetch and normalize traces using an injected transport `call(tool, body) ->
/// Value`. The testable core: orchestration + normalization are exercised with a
/// mock transport (happy path, empty, malformed, error) and no network.
pub fn fetch_traces_with<F>(cfg: &KushtakaConfig, call: F) -> anyhow::Result<Vec<HarnessTrace>>
where
    F: Fn(&str, Value) -> anyhow::Result<Value>,
{
    let response = call(&cfg.tool, call_body(cfg))?;
    Ok(traces_from_kushtaka(&response))
}

/// The real transport: POST the MCP call to `{base}/mcp/call` with the API key.
/// Blocking (`ureq`); the CLI runs it under `block_in_place`. Covered only by the
/// gated live smoke test -- a real socket cannot be exercised deterministically
/// in a unit test, so everything around it ([`fetch_traces_with`], [`call_body`],
/// the normalizer) is mocked instead.
pub fn live_call(base_url: &str, api_key: &str, body: Value) -> anyhow::Result<Value> {
    let url = format!("{}/mcp/call", base_url.trim_end_matches('/'));
    let mut resp = ureq::post(&url)
        .header("X-API-Key", api_key)
        .send_json(&body)
        .map_err(|e| anyhow::anyhow!("kushtaka call to {url} failed: {e}"))?;
    resp.body_mut()
        .read_json::<Value>()
        .map_err(|e| anyhow::anyhow!("kushtaka response was not JSON: {e}"))
}

/// Fetch traces from a live Kushtaka harness over the real transport.
pub fn fetch_live(cfg: &KushtakaConfig) -> anyhow::Result<Vec<HarnessTrace>> {
    fetch_traces_with(cfg, |_tool, body| live_call(&cfg.base_url, &cfg.api_key, body))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> KushtakaConfig {
        KushtakaConfig {
            base_url: "http://h:8081".into(),
            api_key: "k".into(),
            scope: Some("ws".into()),
            tool: "list_tasks".into(),
            args: json!({ "limit": 5 }),
        }
    }

    #[test]
    fn call_body_carries_tool_scope_and_merged_args() {
        let b = call_body(&cfg());
        assert_eq!(b["tool"], json!("list_tasks"));
        assert_eq!(b["scope"], json!("ws"));
        assert_eq!(b["workspace_id"], json!("ws"));
        assert_eq!(b["limit"], json!(5), "extra args merge at the top level");
    }

    #[test]
    fn fetch_normalizes_a_task_list_envelope_through_the_transport() {
        let traces = fetch_traces_with(&cfg(), |tool, body| {
            assert_eq!(tool, "list_tasks");
            assert_eq!(body["workspace_id"], json!("ws"), "scope reaches the transport");
            Ok(json!({
                "count": 1,
                "tasks": [
                    {"task_id": "k1", "prompt": "do x", "outcome": "did x", "status": "completed"}
                ]
            }))
        })
        .unwrap();
        assert_eq!(traces.len(), 1);
        assert_eq!(traces[0].goal, "do x");
        assert_eq!(traces[0].success, Some(true));
    }

    #[test]
    fn fetch_propagates_a_transport_error() {
        // A 401 / network failure must surface, not be swallowed as "no traces".
        let err = fetch_traces_with(&cfg(), |_, _| anyhow::bail!("401 unauthorized"));
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("401"));
    }

    #[test]
    fn empty_and_malformed_responses_yield_no_metabolizable_traces() {
        // An empty task list.
        assert!(fetch_traces_with(&cfg(), |_, _| Ok(json!({ "tasks": [] })))
            .unwrap()
            .is_empty());
        // A non-trace scalar.
        assert!(fetch_traces_with(&cfg(), |_, _| Ok(json!("nope")))
            .unwrap()
            .is_empty());
        // A Kushtaka error envelope (e.g. the 401 JSON body) carries no goal, so
        // it normalizes to a trace that will not metabolize -- never a panic.
        let t = fetch_traces_with(&cfg(), |_, _| {
            Ok(json!({ "status_code": 401, "detail": "API key required" }))
        })
        .unwrap();
        assert!(
            t.iter().all(|x| x.goal.is_empty()),
            "an error body has no goal, so nothing is internalized"
        );
    }

    // Opt-in live smoke test: a real socket cannot be exercised deterministically
    // in a unit test, so the real transport is covered here, gated. Run with a
    // reachable harness (skips cleanly when the env is unset):
    //   ANTUMBRA_KUSHTAKA_URL=http://host:8081 ANTUMBRA_KUSHTAKA_KEY=... \
    //   cargo test -p antumbra-cli --features models harness -- --ignored --nocapture
    // The instance may legitimately hold zero traces; the assertion is only that
    // the auth + POST + JSON round-trip succeeds.
    #[test]
    #[ignore = "needs a live Kushtaka harness: ANTUMBRA_KUSHTAKA_URL + ANTUMBRA_KUSHTAKA_KEY"]
    fn live_fetch_round_trips() {
        let (Ok(url), Ok(api_key)) = (
            std::env::var("ANTUMBRA_KUSHTAKA_URL"),
            std::env::var("ANTUMBRA_KUSHTAKA_KEY"),
        ) else {
            eprintln!("skipped: set ANTUMBRA_KUSHTAKA_URL + ANTUMBRA_KUSHTAKA_KEY");
            return;
        };
        let cfg = KushtakaConfig {
            base_url: url,
            api_key,
            scope: std::env::var("ANTUMBRA_KUSHTAKA_SCOPE").ok(),
            tool: std::env::var("ANTUMBRA_KUSHTAKA_TOOL").unwrap_or_else(|_| "list_tasks".into()),
            args: json!({ "limit": 10 }),
        };
        let traces = fetch_live(&cfg).expect("a live Kushtaka fetch round-trips");
        eprintln!("RESULT: live fetch returned {} trace(s)", traces.len());
    }
}
