//! Live, over-`ws://` validation that incremental cursors preserve convergence
//! against a real SurrealDB v3 engine (R-1). The unknown the embedded tests can't
//! cover is the engine's own string `>` comparison on the RFC3339 version field
//! (the watermark filter) plus the version-field index -- so this drives the real
//! collector path against a networked authoritative store.
//!
//! Gated on `ANTUMBRA_SYNC_WS` (skips when unset), so the normal suite stays
//! GPU-/network-free. Reproduce with a fresh SurrealDB v3 container:
//!
//!   docker run --rm -d -p 8000:8000 --name antumbra-sync-ws surrealdb/surrealdb:v3.0.5 \
//!     start --user root --pass root --bind 0.0.0.0:8000
//!   $env:ANTUMBRA_SYNC_WS = "ws://127.0.0.1:8000/rpc"   # PowerShell
//!   cargo test -p antumbra-sync --test ws_incremental -- --nocapture
//!   docker rm -f antumbra-sync-ws
//!
//! Optional: `ANTUMBRA_SYNC_WS_USER` / `ANTUMBRA_SYNC_WS_PASS` (default root/root).
//! Expects a fresh remote (fixed record ids; a dirty DB would carry stale rows).

use std::time::Duration;

use antumbra_core::{Memory, MemoryId, MemoryNetwork, TenantId};
use antumbra_store::repo::memory;
use antumbra_sync::{reconcile_all_since, Cursors, Endpoint, ReconcileStats, PENUMBRA_TABLES};

fn mem(id: &str, tenant: &TenantId, content: &str, at: chrono::DateTime<chrono::Utc>) -> Memory {
    Memory::new(id, tenant.clone(), MemoryNetwork::World, content, 0.9, at)
}

#[tokio::test]
async fn incremental_cursors_converge_over_ws() {
    let Ok(url) = std::env::var("ANTUMBRA_SYNC_WS") else {
        eprintln!("skipped: set ANTUMBRA_SYNC_WS=ws://127.0.0.1:8000/rpc to run");
        return;
    };
    let user = std::env::var("ANTUMBRA_SYNC_WS_USER").unwrap_or_else(|_| "root".into());
    let pass = std::env::var("ANTUMBRA_SYNC_WS_PASS").unwrap_or_else(|_| "root".into());

    let local = Endpoint::embedded("mem://").connect().await.unwrap();
    let remote = Endpoint::authoritative(&url, user, pass).connect().await.unwrap();
    let tenant = TenantId::new("ws:sync");
    let mut cursors = Cursors::new();
    let t0 = chrono::Utc::now();
    let t1 = t0 + chrono::Duration::seconds(30);
    let t2 = t0 + chrono::Duration::seconds(60);
    let lookback = Duration::from_secs(5);

    // First pass (empty cursors): a full scan seeds A onto the networked remote.
    memory::upsert(&local, &mem("ffffffff-0000-0000-0000-0000000000f1", &tenant, "A", t0))
        .await
        .unwrap();
    let s = reconcile_all_since(&local, &remote, PENUMBRA_TABLES, &mut cursors, lookback)
        .await
        .unwrap();
    assert_eq!(s, ReconcileStats { pushed: 1, pulled: 0 }, "first pass seeds A to ws://");

    // A local write past the watermark is the only thing the incremental query
    // (engine-side string `>` over the network) fetches and pushes.
    memory::upsert(&local, &mem("ffffffff-0000-0000-0000-0000000000f2", &tenant, "B", t1))
        .await
        .unwrap();
    let s = reconcile_all_since(&local, &remote, PENUMBRA_TABLES, &mut cursors, lookback)
        .await
        .unwrap();
    assert_eq!(s, ReconcileStats { pushed: 1, pulled: 0 }, "only B moves on the incremental pass");

    // An update on the remote alone past the watermark is pulled back (the
    // asymmetry: it sits only in the remote's window, provably newer).
    let aid = MemoryId::new("ffffffff-0000-0000-0000-0000000000f1");
    memory::upsert(&remote, &mem(aid.as_str(), &tenant, "A2", t2)).await.unwrap();
    let s = reconcile_all_since(&local, &remote, PENUMBRA_TABLES, &mut cursors, lookback)
        .await
        .unwrap();
    assert_eq!(s, ReconcileStats { pushed: 0, pulled: 1 }, "remote-only update pulls back");
    assert_eq!(
        memory::get(&local, &tenant, &aid).await.unwrap().unwrap().content,
        "A2",
        "the local took the remote's newer version"
    );

    // Converged: another pass over the network moves nothing.
    let s = reconcile_all_since(&local, &remote, PENUMBRA_TABLES, &mut cursors, lookback)
        .await
        .unwrap();
    assert_eq!(s, ReconcileStats::default(), "incremental reconcile settled over ws://");
    eprintln!("RESULT: PASS - incremental cursors converge over ws://");
}
