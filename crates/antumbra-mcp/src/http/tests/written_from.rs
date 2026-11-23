//! The machine a request names (`X-Antumbra-Host`, #193) stamps what that
//! request writes, over both doors: the REST shim and a JSON-RPC session. The
//! identity's server is cached and shared, so the name has to be read per
//! request, and two machines holding one token must still stamp their own.

use super::*;

/// A `POST /mcp/call`, naming a machine when `device` is given.
fn call_from(
    tok: &str,
    device: Option<&str>,
    body: serde_json::Value,
) -> anyhow::Result<Request<Body>> {
    let mut b = Request::builder()
        .method("POST")
        .uri("/mcp/call")
        .header(header::AUTHORIZATION, format!("Bearer {tok}"));
    if let Some(d) = device {
        b = b.header(DEVICE_HEADER, d);
    }
    Ok(b.body(Body::from(body.to_string()))?)
}

async fn json_of(resp: Response) -> anyhow::Result<serde_json::Value> {
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20).await?;
    Ok(serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null))
}

/// What the identity's memories say about where each came from, by content.
async fn stamps(st: &Arc<HttpState>, tok: &str) -> anyhow::Result<Vec<(String, Option<String>)>> {
    let listed = router(st.clone())
        .oneshot(call_from(
            tok,
            None,
            serde_json::json!({ "tool": "list_memories", "arguments": {} }),
        )?)
        .await?;
    let listed = json_of(listed).await?;
    let Some(memories) = listed["memories"].as_array() else {
        anyhow::bail!("the listing carries no memories: {listed}");
    };
    let mut out = Vec::with_capacity(memories.len());
    for m in memories {
        let Some(content) = m["content"].as_str() else {
            anyhow::bail!("a listed memory has no content: {m}");
        };
        out.push((
            content.to_string(),
            m["author_host"].as_str().map(str::to_string),
        ));
    }
    out.sort();
    Ok(out)
}

/// One token, three callers: two that name their machines and one that names
/// none, through the REST shim. Each write carries its own machine, the
/// unnamed one the server's, and `list_memories {host}` returns one machine's.
#[tokio::test]
async fn rest_calls_are_stamped_with_the_machine_each_names() -> anyhow::Result<()> {
    let st = state().await?;
    let tok = token("ws:dev", "user:dev")?;
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
            )?)
            .await?;
        assert_eq!(resp.status(), StatusCode::OK, "{device:?}");
    }

    let at = |host: &str| Some(host.to_string());
    assert_eq!(
        stamps(&st, &tok).await?,
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
        )?)
        .await?;
    let theirs = json_of(theirs).await?;
    let Some(theirs) = theirs["memories"].as_array() else {
        anyhow::bail!("the listing carries no memories: {theirs}");
    };
    assert_eq!(theirs.len(), 1);
    assert_eq!(theirs[0]["content"], "from the desk");

    // The identity's one cached server named no machine of its own.
    let said = router(st.clone())
        .oneshot(call_from(
            &tok,
            Some("mac"),
            serde_json::json!({ "tool": "devices", "arguments": {} }),
        )?)
        .await?;
    let said = json_of(said).await?;
    assert_eq!(said["this_device"], "mac");
    assert_eq!(said["named_by_client"], true);
    assert_eq!(st.servers.lock().await.map.len(), 1);
    Ok(())
}

/// A POST to `/mcp` within an initialized session.
fn rpc(
    jwt: &str,
    session: Option<&str>,
    device: Option<&str>,
    body: serde_json::Value,
) -> anyhow::Result<Request<Body>> {
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
    Ok(b.body(Body::from(body.to_string()))?)
}

/// Over JSON-RPC the name is read from each `tools/call` request, not fixed
/// when the session starts: one session whose calls name two machines stamps
/// two machines.
#[tokio::test]
async fn each_json_rpc_call_is_stamped_with_the_machine_it_names() -> anyhow::Result<()> {
    let st = state().await?;
    let tok = token("ws:rpc", "user:rpc")?;
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
        )?)
        .await?;
    assert_eq!(resp.status(), StatusCode::OK);
    let Some(session) = resp
        .headers()
        .get("mcp-session-id")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned)
    else {
        anyhow::bail!("stateful mode returns a session id");
    };
    let resp = app
        .clone()
        .oneshot(rpc(
            &jwt,
            Some(&session),
            Some("mac"),
            serde_json::json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }),
        )?)
        .await?;
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
            )?)
            .await?;
        assert_eq!(resp.status(), StatusCode::OK);
        // The answer streams; read it through so the call has run.
        let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20).await?;
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains("memory:"), "{device}: {text}");
    }

    assert_eq!(
        stamps(&st, &tok).await?,
        [
            ("said on the desk".to_string(), Some("windows".to_string())),
            ("said on the laptop".to_string(), Some("mac".to_string())),
        ]
    );
    Ok(())
}

/// `POST /devices/name` as `tok`, with `body`.
fn naming(tok: &str, body: serde_json::Value) -> anyhow::Result<Request<Body>> {
    Ok(Request::builder()
        .method("POST")
        .uri("/devices/name")
        .header(header::AUTHORIZATION, format!("Bearer {tok}"))
        .body(Body::from(body.to_string()))?)
}

/// The caller's devices, as the `devices` tool lists them.
async fn devices_of(st: &Arc<HttpState>, tok: &str) -> anyhow::Result<Vec<serde_json::Value>> {
    let listed = router(st.clone())
        .oneshot(call_from(
            tok,
            None,
            serde_json::json!({ "tool": "devices", "arguments": {} }),
        )?)
        .await?;
    Ok(json_of(listed).await?["devices"]
        .as_array()
        .cloned()
        .unwrap_or_default())
}

/// A user names one of their machines and takes the name away; the host it
/// goes by stays. Another member cannot name it, a machine nobody registered
/// cannot be named, and a name that is not one is refused.
#[tokio::test]
async fn a_user_names_their_own_machine() -> anyhow::Result<()> {
    let st = state().await?;
    let (mine, theirs) = (token("ws:dev", "user:dev")?, token("ws:dev", "user:other")?);
    for tok in [&mine, &theirs] {
        router(st.clone())
            .oneshot(call_from(
                tok,
                None,
                serde_json::json!({ "tool": "register_device", "arguments": { "host": "Laptop", "os": "macOS 15.6" } }),
            )?)
            .await?;
    }
    let named = router(st.clone())
        .oneshot(naming(
            &mine,
            serde_json::json!({ "host": " LAPTOP ", "name": "  Work laptop " }),
        )?)
        .await?;
    assert_eq!(named.status(), StatusCode::OK);
    assert_eq!(
        json_of(named).await?,
        serde_json::json!({ "host": "laptop", "name": "Work laptop" })
    );
    let of = |devices: Vec<serde_json::Value>| {
        devices
            .into_iter()
            .find(|d| d["host"] == "laptop")
            .unwrap_or_default()
    };
    let laptop = of(devices_of(&st, &mine).await?);
    assert_eq!(
        (laptop["name"].clone(), laptop["os"].clone()),
        (
            serde_json::json!("Work laptop"),
            serde_json::json!("macOS 15.6")
        )
    );
    // The other member's machine of the same host is theirs, and unnamed.
    assert!(of(devices_of(&st, &theirs).await?).get("name").is_none());

    for (body, why) in [
        (
            serde_json::json!({ "host": "desktop", "name": "x" }),
            "no such device",
        ),
        (
            serde_json::json!({ "host": "laptop", "name": "x".repeat(65) }),
            "too long",
        ),
        (
            serde_json::json!({ "host": "any", "name": "x" }),
            "no machine",
        ),
    ] {
        let refused = router(st.clone()).oneshot(naming(&mine, body)?).await?;
        assert_eq!(refused.status(), StatusCode::BAD_REQUEST, "{why}");
    }
    let unsigned = router(st.clone())
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/devices/name")
                .body(Body::from(r#"{"host":"laptop","name":"x"}"#))?,
        )
        .await?;
    assert_eq!(unsigned.status(), StatusCode::UNAUTHORIZED);

    // A blank name takes it away.
    let cleared = router(st.clone())
        .oneshot(naming(
            &mine,
            serde_json::json!({ "host": "laptop", "name": " " }),
        )?)
        .await?;
    assert_eq!(cleared.status(), StatusCode::OK);
    assert!(of(devices_of(&st, &mine).await?).get("name").is_none());
    Ok(())
}

/// #597: the server's own row moves when the server notes itself up, and a
/// machine a session named does not.
#[tokio::test]
async fn the_servers_own_row_moves_on_its_beat() -> anyhow::Result<()> {
    let st = state().await?;
    let tok = token("ws:dev", "user:dev")?;
    router(st.clone())
        .oneshot(call_from(
            &tok,
            None,
            serde_json::json!({ "tool": "register_device", "arguments": { "host": "laptop" } }),
        )?)
        .await?;
    let long_ago = chrono::Utc::now() - chrono::Duration::days(2);
    let (tenant, user) = (
        antumbra_core::TenantId::new("ws:dev"),
        antumbra_core::UserId::new("user:dev"),
    );
    {
        let _guard = st.auth.lock().await;
        st.store.signin_root().await?;
        for mut d in antumbra_store::repo::device::list_for_user(&st.store, &tenant, &user).await? {
            d.updated_at = long_ago;
            d.seen_at = Some(long_ago);
            antumbra_store::repo::device::upsert(&st.store, &d).await?;
        }
    }
    note_own_machine_up(&st).await;
    let devices = devices_of(&st, &tok).await?;
    assert_eq!(devices.len(), 2, "{devices:?}");
    for d in devices {
        let moved = d["last_seen"] != long_ago.to_rfc3339();
        assert_eq!(moved, d["backend"] != "client", "{d}");
        assert_eq!(d["registered"], long_ago.to_rfc3339());
    }
    Ok(())
}
