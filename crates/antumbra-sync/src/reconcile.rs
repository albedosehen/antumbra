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

/// The last-write-wins pass over two already-indexed row sets (the shared core of
/// the full and incremental paths). A row only one side has is seeded the other
/// way; a shared row's strictly-newer version wins on both.
async fn reconcile_indexed(
    local: &Store,
    remote: &Store,
    spec: &TableSpec,
    local_rows: &BTreeMap<String, Value>,
    remote_rows: &BTreeMap<String, Value>,
) -> Result<ReconcileStats> {
    let mut stats = ReconcileStats::default();
    // A refusal is counted, never confused with a write. The engine persists
    // nothing and reports no error when a session may read a row it may not
    // write, so this is the only place the difference is visible at all.
    let expected = spec.replicate.refusal_is_expected();
    let record =
        |stats: &mut ReconcileStats, written: row_repo::Written, pulled: bool| match written {
            row_repo::Written::Yes if pulled => stats.pulled += 1,
            row_repo::Written::Yes => stats.pushed += 1,
            row_repo::Written::Refused if expected => stats.declined += 1,
            row_repo::Written::Refused => stats.refused += 1,
            row_repo::Written::NoId => {}
        };
    for (id, lrow) in local_rows {
        match remote_rows.get(id) {
            None => {
                let written = row_repo::put_row(remote, lrow).await?;
                record(&mut stats, written, false);
            }
            Some(rrow) => {
                if is_newer(lrow, rrow, spec) {
                    let written = row_repo::put_row(remote, lrow).await?;
                    record(&mut stats, written, false);
                } else if is_newer(rrow, lrow, spec) {
                    let written = row_repo::put_row(local, rrow).await?;
                    record(&mut stats, written, true);
                }
            }
        }
    }
    for (id, rrow) in remote_rows {
        if !local_rows.contains_key(id) {
            let written = row_repo::put_row(local, rrow).await?;
            record(&mut stats, written, true);
        }
    }
    Ok(stats)
}

/// Reconcile one table starting from watermark `cursor` (empty = full scan), with
/// a `lookback` delay window. Returns what moved and the new watermark: the newest
/// version seen on either side this pass, never below `cursor`.
async fn reconcile_table_since(
    local: &Store,
    remote: &Store,
    spec: &TableSpec,
    cursor: &str,
    lookback: Duration,
    scope: Option<&Scope>,
) -> Result<(ReconcileStats, String)> {
    let since = lookback_floor(cursor, lookback);
    let local_rows =
        index_by_id(row_repo::list_rows_since(local, spec.name, spec.version_field, &since).await?);
    let remote_rows = index_by_id(
        row_repo::list_rows_since(remote, spec.name, spec.version_field, &since).await?,
    );
    // Narrow before comparing, not after: a row outside this user's scope is
    // not a row that is missing from the other side, so it must not look like
    // one. Filtering afterwards would leave the pair logic deciding to push
    // rows it had already agreed not to carry.
    //
    // The watermark is taken from the UNNARROWED rows below, on purpose. The
    // cursor tracks how far this pass read, not how much it carried; advancing
    // it only past rows in scope would re-read everything else forever.
    let (local_kept, remote_kept) = match scope {
        Some(scope) => (
            narrow(&local_rows, spec, scope),
            narrow(&remote_rows, spec, scope),
        ),
        None => (local_rows.clone(), remote_rows.clone()),
    };
    let stats = reconcile_indexed(local, remote, spec, &local_kept, &remote_kept).await?;

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

/// The rows a scoped collector carries from `spec`'s table.
fn narrow(
    rows: &BTreeMap<String, Value>,
    spec: &TableSpec,
    scope: &Scope,
) -> BTreeMap<String, Value> {
    rows.iter()
        .filter(|(_, row)| spec.replicate.admits(scope, row))
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
mod fabric_tests;

#[cfg(test)]
mod refusal_tests;

#[cfg(test)]
mod scoped_tests;
