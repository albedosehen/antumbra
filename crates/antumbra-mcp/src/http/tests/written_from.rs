//! The machine a request names (`X-Antumbra-Host`, #193) stamps what that
//! request writes, over both doors: the REST shim and a JSON-RPC session. The
//! identity's server is cached and shared, so the name has to be read per
//! request, and two machines holding one token must still stamp their own.

use super::*;

/// A `POST /mcp/call`, naming a machine when `device` is given.
fn call_from(tok: &str, device: Option<&str>, body: serde_json::Value) -> Request<Body> {
    let mut b = Request::builder()
        .method("POST")
        .uri("/mcp/call")
        .header(header::AUTHORIZATION, format!("Bearer {tok}"));
    if let Some(d) = device {
        b = b.header(DEVICE_HEADER, d);
    }
    b.body(Body::from(body.to_string())).unwrap()
}

async fn json_of(resp: Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null)
}

/// What the identity's memories say about where each came from, by content.
async fn stamps(st: &Arc<HttpState>, tok: &str) -> Vec<(String, Option<String>)> {
    let listed = router(st.clone())
        .oneshot(call_from(
            tok,
            None,
            serde_json::json!({ "tool": "list_memories", "arguments": {} }),
        ))
        .await
        .unwrap();
    let mut out: Vec<(String, Option<String>)> = json_of(listed).await["memories"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| {
            (
                m["content"].as_str().unwrap().to_string(),
                m["author_host"].as_str().map(str::to_string),
            )
        })
        .collect();
    out.sort();
    out
}

/// One token, three callers: two that name their machines and one that names
/// none, through the REST shim. Each write carries its own machine, the
/// unnamed one the server's, and `list_memories {host}` returns one machine's.
#[tokio::test]
async fn rest_calls_are_stamped_with_the_machine_each_names() {
    let st = state().await;
    let tok = token("ws:dev", "user:dev");
    for (device, content) in [
        (Some("mac"), "from the laptop"),
        (Some("Windows"), "from the desk"),
        (None, "from somewhere"),
        // Not one machine's name: stamped as if none were sent, and the call
        // still succeeds.
        (Some("local"), "from an unnamed machine"),
    ] {
        let resp = router(st.clone())
            .oneshot(call_from(
                &tok,
                device,
                serde_json::json!({ "tool": "store_memory", "arguments": { "content": content } }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "{device:?}");
    }

    let at = |host: &str| Some(host.to_string());
    assert_eq!(
        stamps(&st, &tok).await,
        [
            ("from an unnamed machine".to_string(), at("test")),
            ("from somewhere".to_string(), at("test")),
            ("from the desk".to_string(), at("windows")),
            ("from the laptop".to_string(), at("mac")),
        ]
    );
    let theirs = router(st.clone())
        .oneshot(call_from(
            &tok,
            None,
            serde_json::json!({ "tool": "list_memories", "arguments": { "host": "windows", "limit": 5 } }),
        ))
        .await
        .unwrap();
    let theirs = json_of(theirs).await;
    let theirs = theirs["memories"].as_array().unwrap();
    assert_eq!(theirs.len(), 1);
    assert_eq!(theirs[0]["content"], "from the desk");

    // The identity's one cached server named no machine of its own.
    let said = router(st.clone())
        .oneshot(call_from(
            &tok,
            Some("mac"),
            serde_json::json!({ "tool": "devices", "arguments": {} }),
        ))
        .await
        .unwrap();
    let said = json_of(said).await;
    assert_eq!(said["this_device"], "mac");
    assert_eq!(said["named_by_client"], true);
    assert_eq!(st.servers.lock().await.map.len(), 1);
}

/// A POST to `/mcp` within an initialized session.
fn rpc(
    jwt: &str,
    session: Option<&str>,
    device: Option<&str>,
    body: serde_json::Value,
) -> Request<Body> {
    let mut b = Request::builder()
        .method("POST")
        .uri("/mcp")
        .header(header::HOST, "localhost")
        .header(header::AUTHORIZATION, jwt)
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::ACCEPT, "application/json, text/event-stream");
    if let Some(s) = session {
        b = b.header("mcp-session-id", s);
    }
    if let Some(d) = device {
        b = b.header(DEVICE_HEADER, d);
    }
    b.body(Body::from(body.to_string())).unwrap()
}

/// Over JSON-RPC the name is read from each `tools/call` request, not fixed
/// when the session starts: one session whose calls name two machines stamps
/// two machines.
#[tokio::test]
async fn each_json_rpc_call_is_stamped_with_the_machine_it_names() {
    let st = state().await;
    let tok = token("ws:rpc", "user:rpc");
    let jwt = format!("Bearer {tok}");
    let app = router(st.clone());

    let resp = app
        .clone()
        .oneshot(rpc(
            &jwt,
            None,
            Some("mac"),
            serde_json::json!({
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": { "protocolVersion": "2025-06-18", "capabilities": {},
                            "clientInfo": { "name": "test", "version": "0" } }
            }),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let session = resp
        .headers()
        .get("mcp-session-id")
        .expect("stateful mode returns a session id")
        .to_str()
        .unwrap()
        .to_string();
    let resp = app
        .clone()
        .oneshot(rpc(
            &jwt,
            Some(&session),
            Some("mac"),
            serde_json::json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::ACCEPTED);

    for (id, device, content) in [
        (2, "mac", "said on the laptop"),
        (3, "windows", "said on the desk"),
    ] {
        let resp = app
            .clone()
            .oneshot(rpc(
                &jwt,
                Some(&session),
                Some(device),
                serde_json::json!({
                    "jsonrpc": "2.0", "id": id, "method": "tools/call",
                    "params": { "name": "store_memory", "arguments": { "content": content } }
                }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        // The answer streams; read it through so the call has run.
        let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains("memory:"), "{device}: {text}");
    }

    assert_eq!(
        stamps(&st, &tok).await,
        [
            ("said on the desk".to_string(), Some("windows".to_string())),
            ("said on the laptop".to_string(), Some("mac".to_string())),
        ]
    );
}
