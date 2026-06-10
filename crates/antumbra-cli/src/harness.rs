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

/// The MCP call body Kushtaka's `/mcp/call` expects: the tool `name` and an
/// `arguments` object (it rejects a flat body with HTTP 400). The scope rides
/// inside `arguments` under `workspace_id` -- the key the planning tools accept
/// (`list_tasks` / `list_behavior_graph_evaluations`); they reject a `scope`
/// kwarg outright. Any extra args (e.g. a `graph_id` or `limit`) merge alongside.
fn call_body(cfg: &KushtakaConfig) -> Value {
    let mut arguments = serde_json::Map::new();
    if let Some(scope) = &cfg.scope {
        arguments.insert("workspace_id".into(), json!(scope));
    }
    if let Value::Object(extra) = &cfg.args {
        for (k, v) in extra {
            arguments.insert(k.clone(), v.clone());
        }
    }
    json!({ "name": cfg.tool, "arguments": Value::Object(arguments) })
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

/// Kushtaka wraps every `/mcp/call` result as `{success, result}`. Pull the inner
/// `result` out so the normalizer sees the tool's own payload, and surface an
/// explicit `success: false` as an error rather than silently normalizing an
/// error body down to zero traces. A bare (unwrapped) payload passes through.
fn unwrap_envelope(v: Value) -> anyhow::Result<Value> {
    if v.get("success").and_then(Value::as_bool) == Some(false) {
        let msg = v
            .get("error")
            .or_else(|| v.get("detail"))
            .and_then(Value::as_str)
            .unwrap_or("unknown error");
        anyhow::bail!("kushtaka call returned success=false: {msg}");
    }
    Ok(v.get("result").cloned().unwrap_or(v))
}

/// The real transport: POST the MCP call to `{base}/mcp/call` with the API key,
/// then unwrap the `{success, result}` envelope. Blocking (`ureq`); the CLI runs
/// it under `block_in_place`. Covered only by the gated live smoke test -- a real
/// socket cannot be exercised deterministically in a unit test, so everything
/// around it ([`fetch_traces_with`], [`call_body`], [`unwrap_envelope`], the
/// normalizer) is unit-tested instead.
pub fn live_call(base_url: &str, api_key: &str, body: Value) -> anyhow::Result<Value> {
    let url = format!("{}/mcp/call", base_url.trim_end_matches('/'));
    let mut resp = ureq::post(&url)
        .header("X-API-Key", api_key)
        .send_json(&body)
        .map_err(|e| anyhow::anyhow!("kushtaka call to {url} failed: {e}"))?;
    let value = resp
        .body_mut()
        .read_json::<Value>()
        .map_err(|e| anyhow::anyhow!("kushtaka response was not JSON: {e}"))?;
    unwrap_envelope(value)
}

/// Fetch traces from a live Kushtaka harness over the real transport.
pub fn fetch_live(cfg: &KushtakaConfig) -> anyhow::Result<Vec<HarnessTrace>> {
    fetch_traces_with(cfg, |_tool, body| {
        live_call(&cfg.base_url, &cfg.api_key, body)
    })
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
    fn call_body_uses_name_and_nests_workspace_and_args_under_arguments() {
        let b = call_body(&cfg());
        assert_eq!(b["name"], json!("list_tasks"));
        // The scope rides as `workspace_id` only; the planning tools reject `scope`.
        assert_eq!(b["arguments"]["workspace_id"], json!("ws"));
        assert!(b["arguments"].get("scope").is_none());
        assert_eq!(
            b["arguments"]["limit"],
            json!(5),
            "extra args nest under arguments"
        );
    }

    #[test]
    fn fetch_normalizes_a_task_list_envelope_through_the_transport() {
        let traces = fetch_traces_with(&cfg(), |tool, body| {
            assert_eq!(tool, "list_tasks");
            assert_eq!(
                body["arguments"]["workspace_id"],
                json!("ws"),
                "scope reaches the transport under arguments"
            );
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

    #[test]
    fn unwrap_envelope_pulls_the_result_payload_out() {
        // Kushtaka's real shape: {success, result:{tasks:[...]}}. The inner payload
        // must reach the normalizer, or real traces are invisible.
        let v = json!({
            "success": true,
            "result": { "count": 1, "tasks": [
                {"task_id": "k1", "prompt": "do x", "outcome": "did x", "status": "completed"}
            ]}
        });
        let traces = traces_from_kushtaka(&unwrap_envelope(v).unwrap());
        assert_eq!(traces.len(), 1);
        assert_eq!(traces[0].goal, "do x");
    }

    #[test]
    fn unwrap_envelope_surfaces_an_explicit_failure() {
        let v = json!({ "success": false, "detail": "API key required" });
        let err = unwrap_envelope(v);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("API key required"));
    }

    #[test]
    fn unwrap_envelope_passes_a_bare_payload_through() {
        let v = json!({ "tasks": [] });
        assert!(unwrap_envelope(v).unwrap().get("tasks").is_some());
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
