//! The supervised collector loop: connect both stores, reconcile on a cadence,
//! reconnect with exponential backoff when a cycle fails, and shut down promptly
//! on signal. Mirrors the tinytropolis sync supervisor.
//!
//! [`run`] opens both stores itself, which is `antumbra sync`. [`run_on`] is
//! given its local store by a caller that holds it open and shares it, which
//! is a server on an embedded store: that file has one writer, so a second
//! process cannot sync it while the server runs.

use std::time::Duration;

use tokio::sync::watch;

use antumbra_core::{AntumbraError, Result};
use antumbra_store::Store;

use crate::config::SyncConfig;
use crate::hold::Hold;
use crate::reconcile::{
    reconcile_all_scoped, reconcile_all_since_on, Cursors, ReconcileStats, Side,
};
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

        let (local, remote) = (Side::own(&local), Side::own(&remote));
        match cycles(local, remote, &cfg, scope.as_ref(), &mut shutdown).await {
            Ended::Shutdown => return Ok(()),
            Ended::Failed { .. } => {}
        }
    }
}

/// Run the collector on a local store the caller holds open and shares with
/// other work, until `shutdown` is set: the remote is connected here, and
/// reconnected with backoff when it is lost; the local store is used as it is
/// given and taken hold of, through `hold`, only for the steps of a pass that
/// read or write it. Returns `Ok(())` on a clean shutdown.
///
/// As owner: a fabric in `cfg` is refused, since the caller's store is not
/// this collector's to sign in on as someone else.
///
/// A pass that fails is followed by a wait, doubling while the passes keep
/// failing at once: this loop runs inside something that has other work to
/// do, and a remote that refuses every pass is not asked again at full speed.
pub async fn run_on<H: Hold>(
    local: &Store,
    hold: &H,
    cfg: SyncConfig,
    mut shutdown: watch::Receiver<bool>,
) -> Result<()> {
    if cfg.fabric.is_some() {
        return Err(AntumbraError::other(
            "a collector on a store it shares runs as owner and takes no fabric",
        ));
    }
    let mut backoff = cfg.min_backoff;
    loop {
        if *shutdown.borrow() {
            return Ok(());
        }
        let remote = tokio::select! {
            remote = cfg.remote.connect() => remote,
            _ = shutdown.changed() => return Ok(()),
        };
        let ended = match remote {
            Ok(remote) => {
                let local = Side { store: local, hold };
                cycles(local, Side::own(&remote), &cfg, None, &mut shutdown).await
            }
            Err(e) => {
                eprintln!(
                    "sync: connect failed: {e}; retry in {}ms",
                    backoff.as_millis()
                );
                Ended::Failed { ran: 0 }
            }
        };
        match ended {
            Ended::Shutdown => return Ok(()),
            // It ran before it failed: start again from the shortest wait.
            Ended::Failed { ran } if ran > 0 => backoff = cfg.min_backoff,
            Ended::Failed { .. } => {}
        }
        if sleep_or_shutdown(backoff, &mut shutdown).await {
            return Ok(());
        }
        backoff = (backoff * 2).min(cfg.max_backoff);
    }
}

/// How a run of cycles on one connection ended.
enum Ended {
    /// Shutdown was requested.
    Shutdown,
    /// A cycle failed, after `ran` that did not.
    Failed { ran: u64 },
}

/// Reconcile on the cadence until a cycle fails or shutdown is requested.
///
/// Fresh watermarks per connection: the first pass after (re)connecting is a
/// full scan -- the backstop that re-syncs anything changed while down -- then
/// later passes move only what changed.
async fn cycles<L: Hold, R: Hold>(
    local: Side<'_, L>,
    remote: Side<'_, R>,
    cfg: &SyncConfig,
    scope: Option<&Scope>,
    shutdown: &mut watch::Receiver<bool>,
) -> Ended {
    let mut cursors = Cursors::new();
    let mut cycle: u64 = 0;
    loop {
        match reconcile_all_since_on(
            local,
            remote,
            PENUMBRA_TABLES,
            &mut cursors,
            cfg.lookback,
            scope,
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
                return Ended::Failed { ran: cycle };
            }
        }
        cycle += 1;
        collect_garbage(cfg, local, remote, cycle).await;
        if sleep_or_shutdown(cfg.interval, shutdown).await {
            return Ended::Shutdown;
        }
    }
}

/// Hard-purge tombstones past the grace window on both stores, every `gc_every`
/// reconcile cycles (so the soft-delete tables don't grow without bound). Best
/// effort: a GC error is logged, not fatal -- it must never break the sync loop.
async fn collect_garbage<L: Hold, R: Hold>(
    cfg: &SyncConfig,
    local: Side<'_, L>,
    remote: Side<'_, R>,
    cycle: u64,
) {
    if cfg.gc_every == 0 || !cycle.is_multiple_of(cfg.gc_every as u64) {
        return;
    }
    let Ok(grace) = chrono::Duration::from_std(cfg.gc_grace) else {
        return;
    };
    let older_than = chrono::Utc::now() - grace;
    purge(local, older_than).await;
    purge(remote, older_than).await;
}

/// Purge one side's tombstones, held for as long as it takes.
async fn purge<H: Hold>(side: Side<'_, H>, older_than: chrono::DateTime<chrono::Utc>) {
    let purged = async {
        let _held = side.hold.hold().await?;
        crate::gc::purge_store(side.store, older_than).await
    };
    match purged.await {
        Ok(n) if n > 0 => eprintln!("sync: gc purged {n} tombstone(s)"),
        Ok(_) => {}
        Err(e) => eprintln!("sync: gc error: {e}"),
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
    use antumbra_core::AntumbraError;
    use tokio::task::JoinHandle;

    fn mem_cfg() -> SyncConfig {
        SyncConfig::new(Endpoint::embedded("mem://"), Endpoint::embedded("mem://"))
    }

    /// Signal shutdown and wait up to `within` for the spawned worker to return
    /// cleanly: not stopping in time, panicking, and failing are each an error.
    async fn stop_within(
        tx: &watch::Sender<bool>,
        handle: JoinHandle<Result<()>>,
        within: Duration,
    ) -> Result<()> {
        tx.send(true).map_err(|e| {
            AntumbraError::other(format!("the worker dropped its shutdown receiver: {e}"))
        })?;
        tokio::time::timeout(within, handle)
            .await
            .map_err(|_| AntumbraError::other("the worker did not stop promptly"))?
            .map_err(|e| AntumbraError::other(format!("the worker task did not finish: {e}")))?
    }

    // A one-shot pass over two fresh (empty) stores connects, reconciles, and
    // moves nothing -- the plumbing end to end without disk.
    #[tokio::test]
    async fn run_once_over_empty_stores_is_a_noop() -> Result<()> {
        let stats = run_once(&mem_cfg()).await?;
        assert_eq!(stats.total(), 0);
        Ok(())
    }

    // A pre-signaled shutdown returns immediately without attempting to connect.
    #[tokio::test]
    async fn run_returns_on_shutdown() -> Result<()> {
        let (tx, rx) = watch::channel(true);
        let _ = tx; // already signaled
        run(mem_cfg(), rx).await?;
        Ok(())
    }

    // The full supervisor path: connect both, reconcile on the cadence, then exit
    // cleanly when shutdown fires mid-cadence.
    #[tokio::test]
    async fn run_reconciles_on_a_cadence_then_stops() -> Result<()> {
        let cfg = mem_cfg().with_interval(Duration::from_millis(20));
        let (tx, rx) = watch::channel(false);
        let handle = tokio::spawn(run(cfg, rx));
        tokio::time::sleep(Duration::from_millis(70)).await; // a few reconcile cycles
        stop_within(&tx, handle, Duration::from_secs(2)).await
    }

    // The GC cadence: every `gc_every` cycles it purges tombstones on both stores
    // (here there are none, so it's a no-op); a non-multiple cycle is skipped, and
    // `gc_every == 0` disables it entirely. The default `gc_every` (240) is why the
    // cadence tests above never reach this path.
    #[tokio::test]
    async fn collect_garbage_honors_the_gc_cadence() -> Result<()> {
        let local = Endpoint::embedded("mem://").connect().await?;
        let remote = Endpoint::embedded("mem://").connect().await?;

        let on = SyncConfig {
            gc_every: 1,
            ..mem_cfg()
        };
        collect_garbage(&on, Side::own(&local), Side::own(&remote), 1).await; // 1 % 1 == 0 → runs

        let every_two = SyncConfig {
            gc_every: 2,
            ..mem_cfg()
        };
        collect_garbage(&every_two, Side::own(&local), Side::own(&remote), 1).await; // 1 % 2 != 0 → skipped

        let off = SyncConfig {
            gc_every: 0,
            ..mem_cfg()
        };
        collect_garbage(&off, Side::own(&local), Side::own(&remote), 1).await; // disabled
        Ok(())
    }

    // A failing remote sends the supervisor into the backoff path; shutdown during
    // backoff still exits cleanly.
    #[tokio::test]
    async fn run_backs_off_on_connect_failure_then_stops() -> Result<()> {
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
        stop_within(&tx, handle, Duration::from_secs(5)).await
    }
}
