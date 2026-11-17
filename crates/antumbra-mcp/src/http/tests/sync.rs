//! The collector inside the server (#393): a pass on the server's one
//! connection, taken for a step at a time as its other owner work takes it,
//! and let go before the pass waits on the remote.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use super::*;
use crate::http::sync::{OwnerTurn, Settings};
use antumbra_core::{Memory, MemoryNetwork};
use antumbra_store::repo::memory;
use antumbra_sync::{reconcile_all_since_on, Cursors, Hold, ReconcileStats, Side, PENUMBRA_TABLES};

fn call(tok: &str, body: serde_json::Value) -> anyhow::Result<Request<Body>> {
    call_request(Some(tok), &body.to_string())
}

async fn store(st: &Arc<HttpState>, tok: &str, content: &str) -> anyhow::Result<()> {
    let resp = router(st.clone())
        .oneshot(call(
            tok,
            serde_json::json!({ "tool": "store_memory", "arguments": { "content": content } }),
        )?)
        .await?;
    anyhow::ensure!(
        resp.status() == StatusCode::OK,
        "store_memory: {}",
        resp.status()
    );
    Ok(())
}

/// What `tok`'s identity is shown of its workspace's memories, in order.
async fn shown(st: &Arc<HttpState>, tok: &str) -> anyhow::Result<Vec<String>> {
    let resp = router(st.clone())
        .oneshot(call(
            tok,
            serde_json::json!({ "tool": "list_memories", "arguments": {} }),
        )?)
        .await?;
    anyhow::ensure!(
        resp.status() == StatusCode::OK,
        "list_memories: {}",
        resp.status()
    );
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20).await?;
    let listed: serde_json::Value = serde_json::from_slice(&bytes)?;
    let Some(memories) = listed["memories"].as_array() else {
        anyhow::bail!("the listing carries no memories: {listed}");
    };
    let mut contents: Vec<String> = memories
        .iter()
        .filter_map(|m| m["content"].as_str().map(str::to_string))
        .collect();
    contents.sort();
    Ok(contents)
}

fn written_there(id: &str, tenant: &str, content: &str) -> Memory {
    Memory::new(
        id,
        TenantId::new(tenant),
        MemoryNetwork::World,
        content,
        0.9,
        chrono::Utc::now(),
    )
}

/// One pass between the server's own store and `remote`, as the server's
/// collector runs it.
async fn pass<R: Hold>(
    st: &Arc<HttpState>,
    remote: Side<'_, R>,
) -> antumbra_core::Result<ReconcileStats> {
    let turn = OwnerTurn { state: st.clone() };
    let local = Side {
        store: &st.store,
        hold: &turn,
    };
    reconcile_all_since_on(
        local,
        remote,
        PENUMBRA_TABLES,
        &mut Cursors::new(),
        Duration::ZERO,
        None,
    )
    .await
}

/// The point of it: a server that is never stopped gets what was written
/// elsewhere and gives what was written to it. And the pass leaves the
/// connection in the owner's view, so the request after it has to be served
/// as its own user all the same: it is shown its own workspace and no other.
#[tokio::test]
async fn a_pass_on_the_servers_connection_carries_rows_both_ways() -> anyhow::Result<()> {
    let st = state().await?;
    let remote = mem_store().await?;
    let tok = valid_token()?;
    store(&st, &tok, "written here").await?;
    memory::upsert(
        &remote,
        &written_there(
            "aaaaaaaa-0000-0000-0000-000000000001",
            "ws:t",
            "written there",
        ),
    )
    .await?;
    memory::upsert(
        &remote,
        &written_there(
            "bbbbbbbb-0000-0000-0000-000000000002",
            "ws:other",
            "someone else's",
        ),
    )
    .await?;

    let stats = pass(&st, Side::own(&remote)).await?;
    assert_eq!(stats.pulled, 2, "both of the remote's memories");
    assert!(stats.pushed >= 1, "and the one written here: {stats:?}");
    assert_eq!(stats.refused, 0);
    // Settled: the pass after it moves nothing.
    let again = pass(&st, Side::own(&remote)).await?;
    assert_eq!((again.pushed, again.pulled), (0, 0));

    let there: Vec<String> = memory::list(&remote, &TenantId::new("ws:t"))
        .await?
        .into_iter()
        .map(|m| m.content)
        .collect();
    assert!(there.iter().any(|c| c == "written here"), "{there:?}");

    assert_eq!(
        shown(&st, &tok).await?,
        vec!["written here", "written there"]
    );
    assert_eq!(
        shown(&st, &token("ws:other", "user:o")?).await?,
        vec!["someone else's"]
    );
    Ok(())
}

/// A remote that has been reached and does not answer yet: its first step
/// says it was reached, and waits to be let on.
struct Stalled {
    reached: tokio::sync::Notify,
    waiting: AtomicBool,
    go: tokio::sync::Notify,
}

impl Hold for Stalled {
    const SHARED: bool = false;
    type Held = ();

    async fn hold(&self) -> antumbra_core::Result<()> {
        if !self.waiting.swap(true, Ordering::SeqCst) {
            self.reached.notify_one();
            self.go.notified().await;
        }
        Ok(())
    }
}

/// What taking the pass apart is for. While the pass waits on the remote the
/// server's connection is free, and a request is answered; held across the
/// remote, it would wait for as long as the remote took.
#[tokio::test]
async fn a_request_is_answered_while_a_pass_waits_on_the_remote() -> anyhow::Result<()> {
    let st = state().await?;
    let tok = valid_token()?;
    store(&st, &tok, "written here").await?;
    let remote = Arc::new(mem_store().await?);
    let stalled = Arc::new(Stalled {
        reached: tokio::sync::Notify::new(),
        waiting: AtomicBool::new(false),
        go: tokio::sync::Notify::new(),
    });

    let running = {
        let (st, remote, stalled) = (st.clone(), remote.clone(), stalled.clone());
        tokio::spawn(async move {
            let there = Side {
                store: remote.as_ref(),
                hold: stalled.as_ref(),
            };
            pass(&st, there).await
        })
    };
    tokio::time::timeout(Duration::from_secs(10), stalled.reached.notified())
        .await
        .map_err(|_| anyhow::anyhow!("the pass never reached the remote"))?;

    // The pass has read the server's store and is now waiting on the remote.
    let answered = tokio::time::timeout(Duration::from_secs(10), shown(&st, &tok))
        .await
        .map_err(|_| anyhow::anyhow!("the request waited on the pass's remote"))??;
    assert_eq!(answered, vec!["written here"]);

    stalled.go.notify_one();
    let stats = running.await??;
    assert!(stats.pushed >= 1, "the pass then finished: {stats:?}");
    Ok(())
}

/// What the flags become. A remote with a user and a password is signed in
/// to as that user; one with neither is not signed in to at all. The local
/// endpoint is a placeholder, never connected.
#[test]
fn settings_name_the_remote_and_how_to_sign_in_to_it() {
    let open = Settings {
        remote: "ws://db:8000/rpc".into(),
        credentials: None,
        interval: Duration::from_secs(7),
    };
    let cfg = open.config();
    assert_eq!(cfg.remote.url, "ws://db:8000/rpc");
    assert_eq!((cfg.remote.username, cfg.remote.password), (None, None));
    assert_eq!(cfg.interval, Duration::from_secs(7));
    assert!(cfg.fabric.is_none(), "as owner");

    let signed = Settings {
        credentials: Some(("root".into(), "a password".into())),
        ..open
    };
    let cfg = signed.config();
    assert_eq!(cfg.remote.username.as_deref(), Some("root"));
    assert_eq!(cfg.remote.password.as_deref(), Some("a password"));
}

/// An address may carry credentials before its host. The line the server
/// prints at start does not.
#[test]
fn the_remote_is_printed_without_what_signs_in_to_it() {
    let at = |remote: &str| Settings {
        remote: remote.into(),
        credentials: None,
        interval: Duration::from_secs(15),
    };
    assert_eq!(at("ws://db:8000/rpc").shown(), "db:8000/rpc");
    assert_eq!(
        at("wss://root:a-password@db:8000/rpc").shown(),
        "db:8000/rpc"
    );
    assert_eq!(at("not an address").shown(), "the remote");
}
