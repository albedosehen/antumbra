//! How a collector takes hold of a store for one step of a pass.
//!
//! A store the collector opened for itself needs nothing: [`Own`]. A store it
//! shares is another matter. A server on an embedded store has one connection,
//! which every request signs in on as its own user under one lock; a collector
//! running inside that server takes the lock and the owner's view for each
//! step that touches the store, and lets both go before it waits on the other
//! store, so a request waits for a local read or a batch of writes and never
//! for the network.

use std::future::Future;

use antumbra_core::Result;

/// How a pass takes hold of one of its two stores.
pub trait Hold: Send + Sync {
    /// Whether anything else writes the store between two steps of a pass. A
    /// pass then writes it a batch at a time, and writes a row only if the
    /// store has not been given a newer one since the pass read it.
    const SHARED: bool;

    /// What is kept until the step is done, and let go when it is dropped.
    type Held: Send;

    /// Take hold of the store, ready for the collector's reads and writes.
    fn hold(&self) -> impl Future<Output = Result<Self::Held>> + Send;
}

/// A store that is the collector's own: nothing to hold, and nothing else
/// writing it.
#[derive(Debug, Clone, Copy, Default)]
pub struct Own;

impl Hold for Own {
    const SHARED: bool = false;
    type Held = ();

    async fn hold(&self) -> Result<()> {
        Ok(())
    }
}
