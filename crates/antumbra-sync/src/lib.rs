//! Collector/sync (R-1): keep an edge device's local embedded penumbra and a
//! remote authoritative store in agreement by periodic **bidirectional
//! last-write-wins** reconciliation.
//!
//! The cadence-based data movement is R-1; the live, push-on-change engine is
//! R-2. Both live here. R-2's last mile -- delivering a change to a subscriber's
//! MCP client over the SSE stream -- belongs to the MCP transport (it carries the
//! ADR-0015 stateless-vs-streaming tension); this crate produces the routed
//! [`propagate::MemoryChange`] events for it to deliver.
//!
//! - [`config`] -- the two endpoints and the reconcile/backoff timing.
//! - [`table`] -- which tables replicate and each one's version field.
//! - [`reconcile`] -- the last-write-wins pass over a pair of stores (R-1).
//! - [`worker`] -- the supervised loop: connect, reconcile on a cadence,
//!   reconnect with backoff, shut down cleanly (R-1).
//! - [`propagate`] -- watch the change feed and resolve each shared-memory
//!   change to its audience (owner + grantees) (R-2 engine).

pub mod config;
pub mod gc;
pub mod propagate;
pub mod reconcile;
pub mod table;
pub mod worker;

pub use config::{Endpoint, SyncConfig};
pub use propagate::{audience, resolve_change, watch_shared_memories, MemoryChange};
pub use reconcile::{reconcile_all, reconcile_all_since, reconcile_table, Cursors, ReconcileStats};
pub use table::{TableSpec, PENUMBRA_TABLES};
pub use worker::run;
