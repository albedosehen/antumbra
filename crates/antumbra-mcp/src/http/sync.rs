//! The collector, inside the server.
//!
//! A server on an embedded store is that store's one writer, so `antumbra
//! sync`, which opens the store itself, cannot run beside it: the machine
//! stopped serving to sync. Told a remote (`--sync-remote`), the server runs
//! the same collector on the connection it already holds.
//!
//! That connection is shared. Embedded, every request signs in on it as its
//! own user under one lock, and the server's other owner work takes that lock
//! and the owner's view for a step at a time. A pass does the same
//! ([`OwnerTurn`]): it holds the store to read it and to write a batch of
//! rows to it, and lets it go before it waits on the remote. A request waits
//! for one of those steps, never for the network.
//!
//! What arrives needs nothing more from the server. Routing and recall read
//! the store on every call, a replicated memory is announced to live
//! propagation as a local write is, and an expert trained elsewhere is pulled
//! when it is first served.
//!
//! As owner, like the collector the CLI runs: it carries every replicated
//! table whole, with the remote's root credentials.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{watch, OwnedMutexGuard};

use antumbra_sync::{Endpoint, Hold, SyncConfig};

use super::HttpState;

/// The remote this server's store is kept in step with, and how often.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Settings {
    /// The remote SurrealDB, e.g. `ws://host:8000/rpc`.
    pub remote: String,
    /// The remote's root user and password; both, or the remote is
    /// unauthenticated.
    pub credentials: Option<(String, String)>,
    /// The time between two passes.
    pub interval: Duration,
}

impl Settings {
    /// The collector's configuration for these settings. Its local endpoint
    /// is never connected: the pass runs on the store the server holds.
    pub(super) fn config(&self) -> SyncConfig {
        let remote = match &self.credentials {
            Some((user, pass)) => Endpoint::authoritative(&self.remote, user, pass),
            None => Endpoint::embedded(&self.remote),
        };
        SyncConfig::new(Endpoint::embedded("mem://"), remote).with_interval(self.interval)
    }

    /// The remote as it is safe to print: an address may carry credentials
    /// before its host, and a log is read by more people than a flag.
    pub(super) fn shown(&self) -> &str {
        match self.remote.split_once("://") {
            Some((_, rest)) => rest.rsplit_once('@').map_or(rest, |(_, host)| host),
            None => "the remote",
        }
    }
}

/// The server's one connection, taken for a step of a pass as its other owner
/// work takes it: the lock every request signs in under, then the owner's
/// view. Dropping what it returns lets the next request sign in.
pub(super) struct OwnerTurn {
    pub(super) state: Arc<HttpState>,
}

impl Hold for OwnerTurn {
    // Requests write the store between two steps of a pass.
    const SHARED: bool = true;
    type Held = OwnedMutexGuard<()>;

    async fn hold(&self) -> antumbra_core::Result<Self::Held> {
        let guard = self.state.auth.clone().lock_owned().await;
        self.state.store.signin_root().await?;
        Ok(guard)
    }
}

/// Keep the server's store in step with `settings`'s remote for as long as
/// the server runs; with none, do nothing. A remote that is down or refuses
/// is said on stderr and tried again, and never stops the server answering.
pub(super) fn spawn(state: Arc<HttpState>, settings: Option<Settings>) {
    let Some(settings) = settings else {
        return;
    };
    eprintln!(
        "antumbra-mcp: keeping this store in step with {} every {}s",
        settings.shown(),
        settings.interval.as_secs()
    );
    tokio::spawn(async move {
        // Never signalled: the loop ends with the runtime, as the server's
        // other background passes do. Kept here so the loop does not read a
        // dropped sender as a stop.
        let (_running, stop) = watch::channel(false);
        let hold = OwnerTurn {
            state: state.clone(),
        };
        if let Err(e) = antumbra_sync::run_on(&state.store, &hold, settings.config(), stop).await {
            eprintln!("antumbra-mcp: sync stopped: {e}");
        }
    });
}
