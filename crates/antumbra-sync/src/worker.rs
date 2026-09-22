//! The supervised collector loop: connect both stores, reconcile on a cadence,
//! reconnect with exponential backoff when a cycle fails, and shut down promptly
//! on signal. Mirrors the tinytropolis sync supervisor.

use std::time::Duration;

use tokio::sync::watch;

use antumbra_core::Result;
use antumbra_store::Store;

use crate::config::SyncConfig;
use crate::reconcile::{reconcile_all_scoped, reconcile_all_since_scoped, Cursors, ReconcileStats};
use crate::scope::Scope;
use crate::table::PENUMBRA_TABLES;

/// Connect both endpoints and run one reconcile pass over every penumbra table.
/// Useful for a one-shot `sync --once` and for tests; the long-running collector
/// is [`run`].
pub async fn run_once(cfg: &SyncConfig) -> Result<ReconcileStats> {
    let (local, remote) = connect_both(cfg).await?;
    let scope = resolve_scope(cfg, &local).await?;
    reconcile_all_scoped(&local, &remote, PENUMBRA_TABLES, scope.as_ref()).await
}

/// The replication scope for this collector, or `None` when it runs as owner.
/// Resolved from the LOCAL store, which is already signed in: the owned-
/// compartment set is read through the engine, so it is the user's own answer
/// rather than one assembled around them.
async fn resolve_scope(cfg: &SyncConfig, local: &Store) -> Result<Option<Scope>> {
    match &cfg.fabric {
        Some(fabric) => Ok(Some(Scope::resolve(local, fabric).await?)),
        None => Ok(None),
    }
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
                eprintln!(
                    "sync: connect failed: {e}; retry in {}ms",
                    backoff.as_millis()
                );
                if sleep_or_shutdown(backoff, &mut shutdown).await {
                    return Ok(());
                }
                backoff = (backoff * 2).min(cfg.max_backoff);
                continue;
            }
        };
        backoff = cfg.min_backoff; // connected: reset the backoff

        // Resolved once per connection, like the cursors: a compartment created
        // mid-run is picked up on the next reconnect, which is the same
        // freshness every other part of a cadence-based collector has.
        let scope = match resolve_scope(&cfg, &local).await {
            Ok(scope) => scope,
            Err(e) => {
                eprintln!("sync: could not resolve the replication scope: {e}; reconnecting");
                continue;
            }
        };

        // Fresh watermarks per connection: the first pass after (re)connecting is
        // a full scan -- the backstop that re-syncs anything changed while down --
        // then later passes move only what changed.
        let mut cursors = Cursors::new();
        let mut cycle: u64 = 0;

        // Reconcile on the cadence until a cycle fails or shutdown is requested.
        loop {
            match reconcile_all_since_scoped(
                &local,
                &remote,
                PENUMBRA_TABLES,
                &mut cursors,
                cfg.lookback,
                scope.as_ref(),
            )
            .await
            {
                Ok(stats) if stats.total() > 0 || stats.refused > 0 || stats.declined > 0 => {
                    eprintln!("sync: {} pushed, {} pulled", stats.pushed, stats.pulled);
                    // Loud, and every cycle rather than once: a refusal means
                    // rows the collector can see and cannot write, so it is
                    // silently replicating less than it appears to. The engine
                    // says nothing about this, so this line is the only notice
                    // anyone gets.
                    if stats.refused > 0 {
                        eprintln!(
                            "sync: WARNING {} row(s) refused: the session may read them and not write them, so this cycle replicated less than it appears to",
                            stats.refused
                        );
                    }
                }
                Ok(_) => {}
                Err(e) => {
                    eprintln!("sync: cycle failed: {e}; reconnecting");
                    break;
                }
            }
            cycle += 1;
            collect_garbage(&cfg, &local, &remote, cycle).await;
            if sleep_or_shutdown(cfg.interval, &mut shutdown).await {
                return Ok(());
            }
        }
    }
}

/// Hard-purge tombstones past the grace window on both stores, every `gc_every`
/// reconcile cycles (so the soft-delete tables don't grow without bound). Best
/// effort: a GC error is logged, not fatal -- it must never break the sync loop.
async fn collect_garbage(cfg: &SyncConfig, local: &Store, remote: &Store, cycle: u64) {
    if cfg.gc_every == 0 || !cycle.is_multiple_of(cfg.gc_every as u64) {
        return;
    }
    let Ok(grace) = chrono::Duration::from_std(cfg.gc_grace) else {
        return;
    };
    let older_than = chrono::Utc::now() - grace;
    for store in [local, remote] {
        match crate::gc::purge_store(store, older_than).await {
            Ok(n) if n > 0 => eprintln!("sync: gc purged {n} tombstone(s)"),
            Ok(_) => {}
            Err(e) => eprintln!("sync: gc error: {e}"),
        }
    }
}

async fn connect_both(cfg: &SyncConfig) -> Result<(Store, Store)> {
    let local = cfg.local.connect().await?;
    let remote = cfg.remote.connect().await?;
    // Both sides, or neither: a collector signed in on one connection and
    // running as owner on the other would replicate a scoped read into an
    // unscoped write, which is the asymmetry this exists to remove.
    if let Some(fabric) = &cfg.fabric {
        fabric.bind(&local).await?;
        fabric.bind(&remote).await?;
    }
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

    // The GC cadence: every `gc_every` cycles it purges tombstones on both stores
    // (here there are none, so it's a no-op); a non-multiple cycle is skipped, and
    // `gc_every == 0` disables it entirely. The default `gc_every` (240) is why the
    // cadence tests above never reach this path.
    #[tokio::test]
    async fn collect_garbage_honours_the_gc_cadence() {
        let local = Endpoint::embedded("mem://").connect().await.unwrap();
        let remote = Endpoint::embedded("mem://").connect().await.unwrap();

        let on = SyncConfig {
            gc_every: 1,
            ..mem_cfg()
        };
        collect_garbage(&on, &local, &remote, 1).await; // 1 % 1 == 0 → runs

        let every_two = SyncConfig {
            gc_every: 2,
            ..mem_cfg()
        };
        collect_garbage(&every_two, &local, &remote, 1).await; // 1 % 2 != 0 → skipped

        let off = SyncConfig {
            gc_every: 0,
            ..mem_cfg()
        };
        collect_garbage(&off, &local, &remote, 1).await; // disabled
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
