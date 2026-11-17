//! Last-write-wins reconciliation between two stores.
//!
//! For each table we list every row from both sides, pair them by record id, and
//! move the newer version to the staler side (or copy a row the other side is
//! missing). "Newer" is decided in Rust by parsing the table's RFC3339 version
//! field -- no datetime comparison crosses into a query, sidestepping SurrealDB's
//! string/datetime coercion entirely.
//!
//! Convergence and loopback: a write is only propagated when it is *strictly*
//! newer than the target's copy. After one pass both sides hold the same version,
//! so the next pass finds them equal and does nothing -- the bidirectional flow
//! settles instead of echoing. Deletes propagate as tombstones (a soft-delete is
//! the newest version of its row, see [`crate::table`]); the grace-windowed purge
//! is what finally removes them.
//!
//! Incremental cursors: a long-running collector tracks a per-table high-water
//! mark ([`Cursors`]) and each cycle fetches only rows past it (minus a small
//! lookback window), instead of scanning the whole table. The asymmetry that
//! makes this correct: a row appearing in only one side's window is necessarily
//! newer than the other side's copy (which is at or below the window floor), so
//! it wins without a cross-side comparison; rows changed on both sides land in
//! both windows and are compared directly. The one-shot [`reconcile_all`] stays a
//! full scan.

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde_json::Value;
use std::collections::BTreeMap;
use std::time::Duration;

use antumbra_core::Result;
use antumbra_store::repo::sync as row_repo;
use antumbra_store::Store;

use crate::hold::{Hold, Own};
use crate::scope::Scope;
use crate::table::TableSpec;

/// Per-cycle counts: rows written to the remote (`pushed`) and to the local
/// (`pulled`).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ReconcileStats {
    pub pushed: usize,
    pub pulled: usize,
    /// Rows refused on a table this collector believed it had already narrowed.
    /// Apart from `pushed` because the engine refuses silently, and apart from
    /// `declined` because **this should be zero**: a refusal here means the
    /// replication policy and the engine ACL disagree, which is a bug in one.
    pub refused: usize,
    /// Rows refused on a table whose scope the engine decides. Expected, not a
    /// fault: `memory_edge` carries no compartment, so whether an edge belongs
    /// here is a property of memories it only references.
    pub declined: usize,
}

impl ReconcileStats {
    pub fn total(&self) -> usize {
        self.pushed + self.pulled
    }

    fn add(&mut self, other: ReconcileStats) {
        self.pushed += other.pushed;
        self.pulled += other.pulled;
        self.refused += other.refused;
        self.declined += other.declined;
    }
}

/// The row's version timestamp (the LWW tiebreaker): the table's declared field,
/// falling back to `created_at`. `None` if neither parses -- such a row can only
/// be seeded where absent, never win a conflict.
fn version(row: &Value, spec: &TableSpec) -> Option<DateTime<Utc>> {
    let parse = |field: &str| -> Option<DateTime<Utc>> {
        let s = row.get(field)?.as_str()?;
        DateTime::parse_from_rfc3339(s)
            .ok()
            .map(|d| d.with_timezone(&Utc))
    };
    parse(spec.version_field).or_else(|| parse("created_at"))
}

/// `true` if `candidate` is strictly newer than `current` (so it should win), or
/// `current` has no comparable version while `candidate` does.
fn is_newer(candidate: &Value, current: &Value, spec: &TableSpec) -> bool {
    match (version(candidate, spec), version(current, spec)) {
        (Some(c), Some(cur)) => c > cur,
        (Some(_), None) => true,
        _ => false,
    }
}

/// Per-table high-water marks for incremental reconciliation: the newest row
/// version (an RFC3339 string) reconciled so far on each table. A long-running
/// collector holds one across cycles, so each cycle fetches only rows past the
/// mark. It is reset (empty) on a reconnect -- which triggers one full scan, the
/// backstop that re-syncs anything that changed while disconnected.
#[derive(Debug, Default, Clone)]
pub struct Cursors {
    hwm: BTreeMap<String, String>,
}

impl Cursors {
    pub fn new() -> Self {
        Self::default()
    }

    /// The current watermark for `table` (empty until its first pass).
    fn get(&self, table: &str) -> String {
        self.hwm.get(table).cloned().unwrap_or_default()
    }

    /// Advance the watermark for `table` to `candidate` if it is the newer string.
    fn advance(&mut self, table: &str, candidate: String) {
        let entry = self.hwm.entry(table.to_string()).or_default();
        if candidate > *entry {
            *entry = candidate;
        }
    }
}

/// Lower a watermark by `lookback` (the CDC "delay" window): the next cycle re-
/// includes rows within `lookback` of the mark, so a write that landed with a
/// slightly stale timestamp (clock skew, a late commit) is not skipped. Re-
/// reconciling already-settled rows is a harmless no-op under idempotent LWW. An
/// empty or unparseable mark stays empty (a full scan).
fn lookback_floor(hwm: &str, lookback: Duration) -> String {
    if hwm.is_empty() {
        return String::new();
    }
    match (
        DateTime::parse_from_rfc3339(hwm),
        ChronoDuration::from_std(lookback),
    ) {
        (Ok(ts), Ok(delta)) => ts
            .with_timezone(&Utc)
            .checked_sub_signed(delta)
            .map(|t| t.to_rfc3339())
            .unwrap_or_else(|| hwm.to_string()),
        _ => hwm.to_string(),
    }
}

/// One side of a pass: a store, and how the pass takes hold of it for a step.
///
/// A pass over a table is five steps: read this side, read the other, decide,
/// write the other, write this one. Each read and each batch of writes takes
/// hold of its own side for as long as it lasts and of nothing else, so a
/// store the collector shares with other work is held while the collector
/// uses it and never while it waits on the other store.
pub struct Side<'a, H: Hold> {
    pub store: &'a Store,
    pub hold: &'a H,
}

impl<H: Hold> Clone for Side<'_, H> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<H: Hold> Copy for Side<'_, H> {}

impl<'a> Side<'a, Own> {
    /// A store that is the collector's own: nothing to hold.
    pub fn own(store: &'a Store) -> Self {
        Side { store, hold: &Own }
    }
}

/// How many rows are written to a shared store before it is let go and taken
/// again: what waits for the store waits for one batch, however long the pass.
const BATCH: usize = 64;

/// One side's rows past `since`, by record id, read while that side is held.
async fn read<H: Hold>(
    side: Side<'_, H>,
    spec: &TableSpec,
    since: &str,
) -> Result<BTreeMap<String, Value>> {
    let _held = side.hold.hold().await?;
    Ok(index_by_id(
        row_repo::list_rows_since(side.store, spec.name, spec.version_field, since).await?,
    ))
}

/// What a pass moves, decided from two already-indexed row sets: the rows for
/// the remote, then the rows for the local. A row only one side has is seeded
/// the other way; a shared row's strictly-newer version wins on both.
fn decide<'a>(
    spec: &TableSpec,
    local_rows: &'a BTreeMap<String, Value>,
    remote_rows: &'a BTreeMap<String, Value>,
) -> (Vec<&'a Value>, Vec<&'a Value>) {
    let (mut to_remote, mut to_local) = (Vec::new(), Vec::new());
    for (id, lrow) in local_rows {
        match remote_rows.get(id) {
            None => to_remote.push(lrow),
            Some(rrow) if is_newer(lrow, rrow, spec) => to_remote.push(lrow),
            Some(rrow) if is_newer(rrow, lrow, spec) => to_local.push(rrow),
            Some(_) => {}
        }
    }
    for (id, rrow) in remote_rows {
        if !local_rows.contains_key(id) {
            to_local.push(rrow);
        }
    }
    (to_remote, to_local)
}

/// Whether `row` still belongs on `store`: it has no record of that id, or the
/// one it has is older. Asked while a shared store is held, of a row decided
/// on from a read made before the store was let go: something else may have
/// written the record since, and a newer write is not replaced by an older one.
async fn still_wins(store: &Store, spec: &TableSpec, row: &Value) -> Result<bool> {
    let Some(id) = row_repo::row_id(row) else {
        return Ok(true);
    };
    Ok(match row_repo::get_row(store, &id).await? {
        Some(current) => is_newer(row, &current, spec),
        None => true,
    })
}

/// Write `rows` to `side`, held for the writing: how many landed, and how many
/// the engine refused. A shared store is written a batch at a time and let go
/// between two, and each row is written only if it [`still_wins`].
///
/// A refusal is counted, never confused with a write. The engine persists
/// nothing and reports no error when a session may read a row it may not
/// write, so this is the only place the difference is visible at all.
async fn write<H: Hold>(
    side: Side<'_, H>,
    spec: &TableSpec,
    rows: &[&Value],
) -> Result<(usize, usize)> {
    let (mut landed, mut refused) = (0, 0);
    let batch = if H::SHARED { BATCH } else { rows.len().max(1) };
    for rows in rows.chunks(batch) {
        let _held = side.hold.hold().await?;
        for row in rows {
            if H::SHARED && !still_wins(side.store, spec, row).await? {
                continue;
            }
            match row_repo::put_row(side.store, row).await? {
                row_repo::Written::Yes => landed += 1,
                row_repo::Written::Refused => refused += 1,
                row_repo::Written::NoId => {}
            }
        }
    }
    Ok((landed, refused))
}

/// Reconcile one table starting from watermark `cursor` (empty = full scan), with
/// a `lookback` delay window. Returns what moved and the new watermark: the newest
/// version seen on either side this pass, never below `cursor`.
async fn reconcile_table_since<L: Hold, R: Hold>(
    local: Side<'_, L>,
    remote: Side<'_, R>,
    spec: &TableSpec,
    cursor: &str,
    lookback: Duration,
    scope: Option<&Scope>,
) -> Result<(ReconcileStats, String)> {
    let since = lookback_floor(cursor, lookback);
    let local_rows = read(local, spec, &since).await?;
    let remote_rows = read(remote, spec, &since).await?;
    // Narrow before comparing, not after: a row outside this user's scope is
    // not a row that is missing from the other side, so it must not look like
    // one. Filtering afterwards would leave the pair logic deciding to push
    // rows it had already agreed not to carry. The same holds for a row the
    // table cannot pair at all, which is left out as owner too.
    //
    // The watermark is taken from the UNNARROWED rows below, on purpose. The
    // cursor tracks how far this pass read, not how much it carried; advancing
    // it only past rows in scope would re-read everything else forever.
    let local_kept = narrow(&local_rows, spec, scope);
    let remote_kept = narrow(&remote_rows, spec, scope);
    let (to_remote, to_local) = decide(spec, &local_kept, &remote_kept);
    let (pushed, refused_there) = write(remote, spec, &to_remote).await?;
    let (pulled, refused_here) = write(local, spec, &to_local).await?;
    let mut stats = ReconcileStats {
        pushed,
        pulled,
        ..ReconcileStats::default()
    };
    if spec.replicate.refusal_is_expected() {
        stats.declined = refused_there + refused_here;
    } else {
        stats.refused = refused_there + refused_here;
    }

    let mut hwm = cursor.to_string();
    for rows in [&local_rows, &remote_rows] {
        for row in rows.values() {
            if let Some(v) = row.get(spec.version_field).and_then(Value::as_str) {
                if v > hwm.as_str() {
                    hwm = v.to_string();
                }
            }
        }
    }
    Ok((stats, hwm))
}

/// The rows a collector carries from `spec`'s table: those the table can pair
/// across two stores, and under a `scope` only those in it.
fn narrow(
    rows: &BTreeMap<String, Value>,
    spec: &TableSpec,
    scope: Option<&Scope>,
) -> BTreeMap<String, Value> {
    rows.iter()
        .filter(|(_, row)| spec.carries(row))
        .filter(|(_, row)| scope.is_none_or(|scope| spec.replicate.admits(scope, row)))
        .map(|(id, row)| (id.clone(), row.clone()))
        .collect()
}

/// Reconcile one table across both stores, last-write-wins (a full scan). Returns
/// what moved. For the incremental, cursor-tracked path the collector runs, see
/// [`reconcile_all_since`].
pub async fn reconcile_table(
    local: &Store,
    remote: &Store,
    spec: &TableSpec,
) -> Result<ReconcileStats> {
    let (local, remote) = (Side::own(local), Side::own(remote));
    Ok(
        reconcile_table_since(local, remote, spec, "", Duration::ZERO, None)
            .await?
            .0,
    )
}

/// Reconcile every table, in the given order (a full scan over each). The one-shot
/// `sync --once` path.
pub async fn reconcile_all(
    local: &Store,
    remote: &Store,
    tables: &[TableSpec],
) -> Result<ReconcileStats> {
    reconcile_all_scoped(local, remote, tables, None).await
}

/// [`reconcile_all`], narrowed to one user's fabric.
pub async fn reconcile_all_scoped(
    local: &Store,
    remote: &Store,
    tables: &[TableSpec],
    scope: Option<&Scope>,
) -> Result<ReconcileStats> {
    let (local, remote) = (Side::own(local), Side::own(remote));
    let mut stats = ReconcileStats::default();
    for spec in tables {
        stats.add(
            reconcile_table_since(local, remote, spec, "", Duration::ZERO, scope)
                .await?
                .0,
        );
    }
    Ok(stats)
}

/// Reconcile every table incrementally: each is fetched from its watermark in
/// `cursors` (minus the `lookback` window) and its watermark advanced past what
/// was seen. The first call (empty cursors) is a full scan; later calls move only
/// what changed since. Returns what moved this pass.
pub async fn reconcile_all_since(
    local: &Store,
    remote: &Store,
    tables: &[TableSpec],
    cursors: &mut Cursors,
    lookback: Duration,
) -> Result<ReconcileStats> {
    reconcile_all_since_scoped(local, remote, tables, cursors, lookback, None).await
}

/// [`reconcile_all_since`], narrowed to one user's fabric. `None` is owner mode
/// over the whole tenant, which is what [`reconcile_all_since`] passes.
pub async fn reconcile_all_since_scoped(
    local: &Store,
    remote: &Store,
    tables: &[TableSpec],
    cursors: &mut Cursors,
    lookback: Duration,
    scope: Option<&Scope>,
) -> Result<ReconcileStats> {
    let (local, remote) = (Side::own(local), Side::own(remote));
    reconcile_all_since_on(local, remote, tables, cursors, lookback, scope).await
}

/// [`reconcile_all_since_scoped`] over two [`Side`]s, each held only for the
/// steps that use it: the pass a collector runs on a store it shares.
pub async fn reconcile_all_since_on<L: Hold, R: Hold>(
    local: Side<'_, L>,
    remote: Side<'_, R>,
    tables: &[TableSpec],
    cursors: &mut Cursors,
    lookback: Duration,
    scope: Option<&Scope>,
) -> Result<ReconcileStats> {
    let mut stats = ReconcileStats::default();
    for spec in tables {
        let cursor = cursors.get(spec.name);
        let (s, hwm) = reconcile_table_since(local, remote, spec, &cursor, lookback, scope).await?;
        cursors.advance(spec.name, hwm);
        stats.add(s);
    }
    Ok(stats)
}

fn index_by_id(rows: Vec<Value>) -> BTreeMap<String, Value> {
    rows.into_iter()
        .filter_map(|row| row_repo::row_id(&row).map(|id| (id, row)))
        .collect()
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod expert_tests;

#[cfg(test)]
mod fabric_tests;

#[cfg(test)]
mod refusal_tests;

#[cfg(test)]
mod scoped_tests;

#[cfg(test)]
mod shared_tests;
