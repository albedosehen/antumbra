//! The supervised collector loop: connect both stores, reconcile on a cadence,
//! reconnect with exponential backoff when a cycle fails, and shut down promptly
//! on signal. Mirrors the tinytropolis sync supervisor.

use std::time::Duration;

use tokio::sync::watch;

use antumbra_core::Result;
use antumbra_store::Store;

use crate::config::SyncConfig;
use crate::reconcile::{reconcile_all, reconcile_all_since, Cursors, ReconcileStats};
use crate::table::PENUMBRA_TABLES;

/// Connect both endpoints and run one reconcile pass over every penumbra table.
/// Useful for a one-shot `sync --once` and for tests; the long-running collector
/// is [`run`].
pub async fn run_once(cfg: &SyncConfig) -> Result<ReconcileStats> {
    let local = cfg.local.connect().await?;
    let remote = cfg.remote.connect().await?;
    reconcile_all(&local, &remote, PENUMBRA_TABLES).await
}

/// Run the collector until `shutdown` is set. Reconciles every `cfg.interval`
/// while connected; on any connect/cycle error it backs off (bounded, doubling)
/// and reconnects. Returns `Ok(())` on a clean shutdown.
pub async fn run(cfg: SyncConfig, mut shutdown: watch::Receiver<bool>) -> Result<()> {
    let mut backoff = cfg.min_backoff;
    loop {
        if *shutdown.borrow() {
            return Ok(());
        }

        // (Re)connect both sides; connecting also applies the schema.
        let session = tokio::select! {
            pair = connect_both(&cfg) => pair,
            _ = shutdown.changed() => return Ok(()),
        };
        let (local, remote) = match session {
            Ok(pair) => pair,
            Err(e) => {
                eprintln!("sync: connect failed: {e}; retry in {}ms", backoff.as_millis());
                if sleep_or_shutdown(backoff, &mut shutdown).await {
                    return Ok(());
                }
                backoff = (backoff * 2).min(cfg.max_backoff);
                continue;
            }
        };
        backoff = cfg.min_backoff; // connected: reset the backoff

        // Fresh watermarks per connection: the first pass after (re)connecting is
        // a full scan -- the backstop that re-syncs anything changed while down --
        // then later passes move only what changed.
        let mut cursors = Cursors::new();

        // Reconcile on the cadence until a cycle fails or shutdown is requested.
        loop {
            match reconcile_all_since(&local, &remote, PENUMBRA_TABLES, &mut cursors, cfg.lookback).await {
                Ok(stats) if stats.total() > 0 => {
                    eprintln!("sync: {} pushed, {} pulled", stats.pushed, stats.pulled);
                }
                Ok(_) => {}
                Err(e) => {
                    eprintln!("sync: cycle failed: {e}; reconnecting");
                    break;
                }
            }
            if sleep_or_shutdown(cfg.interval, &mut shutdown).await {
                return Ok(());
            }
        }
    }
}

async fn connect_both(cfg: &SyncConfig) -> Result<(Store, Store)> {
    let local = cfg.local.connect().await?;
    let remote = cfg.remote.connect().await?;
    Ok((local, remote))
}

/// Sleep for `dur`, returning `true` if shutdown is (or becomes) requested first.
async fn sleep_or_shutdown(dur: Duration, shutdown: &mut watch::Receiver<bool>) -> bool {
    if *shutdown.borrow() {
        return true;
    }
    tokio::select! {
        _ = tokio::time::sleep(dur) => false,
        _ = shutdown.changed() => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Endpoint;

    fn mem_cfg() -> SyncConfig {
        SyncConfig::new(Endpoint::embedded("mem://"), Endpoint::embedded("mem://"))
    }

    // A one-shot pass over two fresh (empty) stores connects, reconciles, and
    // moves nothing -- the plumbing end to end without disk.
    #[tokio::test]
    async fn run_once_over_empty_stores_is_a_noop() {
        let stats = run_once(&mem_cfg()).await.unwrap();
        assert_eq!(stats.total(), 0);
    }

    // A pre-signalled shutdown returns immediately without attempting to connect.
    #[tokio::test]
    async fn run_returns_on_shutdown() {
        let (tx, rx) = watch::channel(true);
        let _ = tx; // already signalled
        run(mem_cfg(), rx).await.unwrap();
    }

    // The full supervisor path: connect both, reconcile on the cadence, then exit
    // cleanly when shutdown fires mid-cadence.
    #[tokio::test]
    async fn run_reconciles_on_a_cadence_then_stops() {
        let cfg = mem_cfg().with_interval(Duration::from_millis(20));
        let (tx, rx) = watch::channel(false);
        let handle = tokio::spawn(run(cfg, rx));
        tokio::time::sleep(Duration::from_millis(70)).await; // a few reconcile cycles
        tx.send(true).unwrap();
        tokio::time::timeout(Duration::from_secs(2), handle)
            .await
            .expect("worker stops promptly")
            .unwrap()
            .unwrap();
    }

    // A failing remote sends the supervisor into the backoff path; shutdown during
    // backoff still exits cleanly.
    #[tokio::test]
    async fn run_backs_off_on_connect_failure_then_stops() {
        let cfg = SyncConfig {
            min_backoff: Duration::from_millis(20),
            max_backoff: Duration::from_millis(40),
            ..SyncConfig::new(
                Endpoint::embedded("mem://"),
                // A refused port: connecting fails fast.
                Endpoint::authoritative("ws://127.0.0.1:1/rpc", "root", "x"),
            )
        };
        let (tx, rx) = watch::channel(false);
        let handle = tokio::spawn(run(cfg, rx));
        tokio::time::sleep(Duration::from_millis(120)).await;
        tx.send(true).unwrap();
        tokio::time::timeout(Duration::from_secs(5), handle)
            .await
            .expect("worker stops promptly after a connect failure")
            .unwrap()
            .unwrap();
    }
}
