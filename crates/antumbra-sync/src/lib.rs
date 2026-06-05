//! Collector/sync (R-1): keep an edge device's local embedded penumbra and a
//! remote authoritative store in agreement by periodic **bidirectional
//! last-write-wins** reconciliation.
//!
//! The data movement lives here (R-1). Live, push-on-change propagation over
//! SSE/LIVE SELECT is the separate R-2 step; this cut converges on a cadence.
//!
//! - [`config`] -- the two endpoints and the reconcile/backoff timing.
//! - [`table`] -- which tables replicate and each one's version field.
//! - [`reconcile`] -- the last-write-wins pass over a pair of stores.
//! - [`worker`] -- the supervised loop: connect, reconcile on a cadence,
//!   reconnect with backoff, shut down cleanly.

pub mod config;
pub mod reconcile;
pub mod table;
pub mod worker;

pub use config::{Endpoint, SyncConfig};
pub use reconcile::{reconcile_all, reconcile_table, ReconcileStats};
pub use table::{TableSpec, PENUMBRA_TABLES};
pub use worker::run;
