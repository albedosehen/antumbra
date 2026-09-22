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
mod tests {
    use super::*;
    use antumbra_core::{Memory, MemoryId, MemoryNetwork, TenantId};
    use antumbra_store::repo::memory;
    use antumbra_store::EMBED_DIM;
    use chrono::Duration as ChronoDuration;

    use crate::scope::Replicate;

    const MEMORY: &TableSpec = &TableSpec {
        name: "memory",
        version_field: "updated_at",
        replicate: Replicate::Owned(|scope: &Scope, row| scope.holds_memory(row)),
    };

    async fn mem_store() -> Store {
        Store::connect_memory(EMBED_DIM).await.unwrap()
    }

    fn mem(id: &str, tenant: &TenantId, content: &str, at: DateTime<Utc>) -> Memory {
        Memory::new(id, tenant.clone(), MemoryNetwork::World, content, 0.9, at)
    }

    // A row each side is missing flows the other way: local-only pushes, remote-
    // only pulls, and a second pass is a no-op (converged + idempotent).
    #[tokio::test]
    async fn seeds_both_directions_then_settles() {
        let (local, remote) = (mem_store().await, mem_store().await);
        let tenant = TenantId::new("t");
        let now = Utc::now();
        memory::upsert(
            &local,
            &mem("aaaaaaaa-0000-0000-0000-000000000001", &tenant, "left", now),
        )
        .await
        .unwrap();
        memory::upsert(
            &remote,
            &mem(
                "bbbbbbbb-0000-0000-0000-000000000002",
                &tenant,
                "right",
                now,
            ),
        )
        .await
        .unwrap();

        let stats = reconcile_table(&local, &remote, MEMORY).await.unwrap();
        assert_eq!(
            stats,
            ReconcileStats {
                pushed: 1,
                pulled: 1,
                refused: 0,
                declined: 0
            }
        );

        // Both stores now hold both memories.
        assert_eq!(memory::list(&local, &tenant).await.unwrap().len(), 2);
        assert_eq!(memory::list(&remote, &tenant).await.unwrap().len(), 2);

        // Converged: the next pass moves nothing.
        let again = reconcile_table(&local, &remote, MEMORY).await.unwrap();
        assert_eq!(again, ReconcileStats::default());
    }

    // The strictly-newer version of a shared record wins on both sides.
    #[tokio::test]
    async fn last_write_wins_on_conflict() {
        let (local, remote) = (mem_store().await, mem_store().await);
        let tenant = TenantId::new("t");
        let id = "cccccccc-0000-0000-0000-000000000003";
        let t0 = Utc::now();
        let t1 = t0 + ChronoDuration::seconds(5);

        // Same record, divergent: remote's copy is newer.
        memory::upsert(&local, &mem(id, &tenant, "stale", t0))
            .await
            .unwrap();
        memory::upsert(&remote, &mem(id, &tenant, "fresh", t1))
            .await
            .unwrap();

        let stats = reconcile_table(&local, &remote, MEMORY).await.unwrap();
        assert_eq!(
            stats,
            ReconcileStats {
                pushed: 0,
                pulled: 1,
                refused: 0,
                declined: 0
            },
            "newer remote pulled to local"
        );

        let mid = MemoryId::new(id);
        let on_local = memory::get(&local, &tenant, &mid).await.unwrap().unwrap();
        let on_remote = memory::get(&remote, &tenant, &mid).await.unwrap().unwrap();
        assert_eq!(on_local.content, "fresh", "local took the newer version");
        assert_eq!(on_remote.content, "fresh", "remote unchanged");
    }

    // A forget (tombstone) is a newer version, so it propagates to the other side
    // and the trace does not resurrect; the pass then converges.
    #[tokio::test]
    async fn a_tombstone_propagates_and_does_not_resurrect() {
        let (local, remote) = (mem_store().await, mem_store().await);
        let tenant = TenantId::new("t");
        let id = "dddddddd-0000-0000-0000-000000000004";
        let mid = MemoryId::new(id);
        let t0 = Utc::now();

        // Both sides hold the live trace.
        memory::upsert(&local, &mem(id, &tenant, "live", t0))
            .await
            .unwrap();
        memory::upsert(&remote, &mem(id, &tenant, "live", t0))
            .await
            .unwrap();

        // Local forgets it (a tombstone, newer than remote's live copy).
        memory::soft_delete(&local, &tenant, &mid, t0 + ChronoDuration::seconds(5))
            .await
            .unwrap();

        let stats = reconcile_table(&local, &remote, MEMORY).await.unwrap();
        assert_eq!(
            stats,
            ReconcileStats {
                pushed: 1,
                pulled: 0,
                refused: 0,
                declined: 0
            },
            "tombstone pushed"
        );

        // Forgotten on both sides; the live copy did not resurrect it on local.
        assert!(memory::get(&remote, &tenant, &mid).await.unwrap().is_none());
        assert!(memory::get(&local, &tenant, &mid).await.unwrap().is_none());

        // Converged.
        assert_eq!(
            reconcile_table(&local, &remote, MEMORY).await.unwrap(),
            ReconcileStats::default()
        );
    }

    // Security: a revoke on one store propagates to the other (a stale live grant
    // cannot keep the grantee in). The revocation tombstone out-versions the live
    // copy (bumped updated_at) and wins.
    #[tokio::test]
    async fn a_revoke_propagates_and_cannot_be_out_voted_by_a_stale_grant() {
        use antumbra_core::{Capability, Compartment, CompartmentId, Origin, UserId};
        use antumbra_store::repo::compartment;

        const GRANT: &TableSpec = &TableSpec {
            name: "grant",
            version_field: "updated_at",
            replicate: Replicate::Owned(|scope: &Scope, row| scope.grants_own_compartment(row)),
        };
        let (local, remote) = (mem_store().await, mem_store().await);
        let tenant = TenantId::new("t");
        let comp = CompartmentId::new("comp-g");
        let bob = UserId::new("bob");
        let t0 = Utc::now();

        let new_comp = || Compartment {
            id: comp.clone(),
            tenant: tenant.clone(),
            owner: UserId::new("alice"),
            name: "shared".into(),
            origin: Origin::User,
            created_at: t0,
            updated_at: t0,
            deleted_at: None,
        };
        let grant = antumbra_core::Grant::new(
            tenant.clone(),
            comp.clone(),
            bob.clone(),
            Capability::Reference,
            UserId::new("alice"),
            t0,
        );
        // Both sides start with the compartment + the live grant.
        for s in [&local, &remote] {
            compartment::create(s, &new_comp()).await.unwrap();
            compartment::grant(s, &grant).await.unwrap();
        }

        // Local revokes bob (a newer version than remote's still-live grant).
        compartment::revoke(
            &local,
            &tenant,
            &comp,
            &bob,
            t0 + ChronoDuration::seconds(5),
        )
        .await
        .unwrap();

        let stats = reconcile_table(&local, &remote, GRANT).await.unwrap();
        assert_eq!(
            stats,
            ReconcileStats {
                pushed: 1,
                pulled: 0,
                refused: 0,
                declined: 0
            },
            "revocation pushed"
        );

        // Remote no longer lists bob as a grantee (revoked everywhere).
        assert!(
            compartment::list_grants(&remote, &tenant, &comp)
                .await
                .unwrap()
                .is_empty(),
            "the revocation reached the remote: bob is no longer a live grantee"
        );
        // And it does not resurrect from the stale side.
        assert_eq!(
            reconcile_table(&local, &remote, GRANT).await.unwrap(),
            ReconcileStats::default()
        );
    }

    // Incremental cursors: the first pass (empty cursors) is a full scan that
    // seeds the remote and sets the watermark; a later write past the watermark is
    // the only thing the next pass fetches and pushes, and the pass then settles.
    #[tokio::test]
    async fn incremental_pass_moves_only_what_changed_since_the_watermark() {
        let (local, remote) = (mem_store().await, mem_store().await);
        let tenant = TenantId::new("t");
        let t0 = Utc::now();
        let t1 = t0 + ChronoDuration::seconds(30);
        let mut cursors = Cursors::new();
        let tables = &[*MEMORY];

        // First pass: a full scan seeds A onto the remote.
        memory::upsert(
            &local,
            &mem("aaaaaaaa-0000-0000-0000-0000000000a1", &tenant, "A", t0),
        )
        .await
        .unwrap();
        let s = reconcile_all_since(&local, &remote, tables, &mut cursors, Duration::ZERO)
            .await
            .unwrap();
        assert_eq!(
            s,
            ReconcileStats {
                pushed: 1,
                pulled: 0,
                refused: 0,
                declined: 0
            }
        );

        // A new local write past the watermark is the only row the next pass moves.
        memory::upsert(
            &local,
            &mem("bbbbbbbb-0000-0000-0000-0000000000b2", &tenant, "B", t1),
        )
        .await
        .unwrap();
        let s = reconcile_all_since(&local, &remote, tables, &mut cursors, Duration::ZERO)
            .await
            .unwrap();
        assert_eq!(
            s,
            ReconcileStats {
                pushed: 1,
                pulled: 0,
                refused: 0,
                declined: 0
            },
            "only B moved"
        );
        assert_eq!(memory::list(&remote, &tenant).await.unwrap().len(), 2);

        // Converged: nothing new past the watermark.
        let s = reconcile_all_since(&local, &remote, tables, &mut cursors, Duration::ZERO)
            .await
            .unwrap();
        assert_eq!(s, ReconcileStats::default());
    }

    // The asymmetry that keeps incremental reconcile correct: a row updated on
    // ONLY the remote past the watermark appears in just the remote's window, yet
    // is still pulled to the local (its copy sits at or below the floor, so the
    // remote's is provably newer -- no cross-side compare needed).
    #[tokio::test]
    async fn an_update_on_one_side_only_still_propagates_under_a_cursor() {
        let (local, remote) = (mem_store().await, mem_store().await);
        let tenant = TenantId::new("t");
        let id = "cccccccc-0000-0000-0000-0000000000c3";
        let mid = MemoryId::new(id);
        let t0 = Utc::now();
        let t2 = t0 + ChronoDuration::seconds(30);
        let mut cursors = Cursors::new();
        let tables = &[*MEMORY];

        // Seed A on both sides, advancing the watermark to t0.
        memory::upsert(&local, &mem(id, &tenant, "v0", t0))
            .await
            .unwrap();
        reconcile_all_since(&local, &remote, tables, &mut cursors, Duration::ZERO)
            .await
            .unwrap();

        // The remote alone updates A past the watermark.
        memory::upsert(&remote, &mem(id, &tenant, "v2", t2))
            .await
            .unwrap();
        let s = reconcile_all_since(&local, &remote, tables, &mut cursors, Duration::ZERO)
            .await
            .unwrap();
        assert_eq!(
            s,
            ReconcileStats {
                pushed: 0,
                pulled: 1,
                refused: 0,
                declined: 0
            },
            "remote-only update pulled"
        );
        assert_eq!(
            memory::get(&local, &tenant, &mid)
                .await
                .unwrap()
                .unwrap()
                .content,
            "v2"
        );
    }

    // The lookback window lowers the query floor (so a slightly-stale write is not
    // skipped) but never raises it; an empty or bad mark stays a full scan.
    #[test]
    fn lookback_floor_subtracts_the_window() {
        let t = "2026-06-05T12:00:30+00:00";
        let lowered = lookback_floor(t, Duration::from_secs(5));
        let expected = DateTime::parse_from_rfc3339(t).unwrap().with_timezone(&Utc)
            - ChronoDuration::seconds(5);
        assert_eq!(lowered, expected.to_rfc3339());
        assert!(lowered.as_str() < t, "the floor is below the mark");
        // No window: the mark is unchanged. Empty/garbage: a full scan.
        assert_eq!(lookback_floor(t, Duration::ZERO), t);
        assert_eq!(lookback_floor("", Duration::from_secs(5)), "");
        assert_eq!(
            lookback_floor("not-a-date", Duration::from_secs(5)),
            "not-a-date"
        );
    }

    // A compartment deletion (tombstone) propagates and does not resurrect from a
    // replica still holding the live row.
    #[tokio::test]
    async fn a_compartment_deletion_propagates_and_does_not_resurrect() {
        use antumbra_core::{Compartment, CompartmentId, UserId};
        use antumbra_store::repo::compartment;

        const COMPARTMENT: &TableSpec = &TableSpec {
            name: "compartment",
            version_field: "updated_at",
            replicate: Replicate::Owned(|scope: &Scope, row| scope.owns_compartment(row)),
        };
        let (local, remote) = (mem_store().await, mem_store().await);
        let tenant = TenantId::new("t");
        let id = CompartmentId::new("comp-d");
        let t0 = Utc::now();

        for s in [&local, &remote] {
            compartment::create(
                s,
                &Compartment::new(id.clone(), tenant.clone(), UserId::new("alice"), "c", t0),
            )
            .await
            .unwrap();
        }
        compartment::delete(&local, &tenant, &id, t0 + ChronoDuration::seconds(5))
            .await
            .unwrap();

        let stats = reconcile_table(&local, &remote, COMPARTMENT).await.unwrap();
        assert_eq!(
            stats,
            ReconcileStats {
                pushed: 1,
                pulled: 0,
                refused: 0,
                declined: 0
            }
        );
        assert!(
            compartment::get(&remote, &tenant, &id)
                .await
                .unwrap()
                .is_none(),
            "deletion reached remote"
        );
        assert_eq!(
            reconcile_table(&local, &remote, COMPARTMENT).await.unwrap(),
            ReconcileStats::default()
        );
    }
}

#[cfg(test)]
mod fabric_tests {
    use super::*;
    use antumbra_core::{
        CompartmentId, DeviceProfile, DeviceRole, GenesisRequest, GenesisStatus, TenantId, UserId,
    };
    use antumbra_store::repo::{device, genesis};
    use antumbra_store::EMBED_DIM;
    use chrono::Duration as ChronoDuration;

    use crate::table::PENUMBRA_TABLES;

    async fn node() -> Result<Store> {
        Store::connect_memory(EMBED_DIM).await
    }

    fn tenant() -> TenantId {
        TenantId::new("ws:t")
    }

    fn user() -> UserId {
        UserId::new("user:a")
    }

    /// ADR-0017 A2's delivery. The laptop cannot train and leaves a request; the
    /// rig takes it. Without the fabric tables replicating, both halves work
    /// perfectly and no run ever crosses between the two machines.
    #[tokio::test]
    async fn a_genesis_run_crosses_from_the_node_that_asked_to_the_one_that_can() -> Result<()> {
        let laptop = node().await?;
        let rig = node().await?;
        let now = Utc::now();

        // Each machine knows only itself to begin with.
        device::upsert(
            &laptop,
            &DeviceProfile::new(
                tenant(),
                user(),
                "her-laptop",
                "cpu",
                DeviceRole::Memory,
                now,
            ),
        )
        .await?;
        device::upsert(
            &rig,
            &DeviceProfile::new(
                tenant(),
                user(),
                "the-rig",
                "cuda",
                DeviceRole::Genesis,
                now,
            ),
        )
        .await?;
        assert!(
            device::genesis_for_user(&laptop, &tenant(), &user())
                .await?
                .is_none(),
            "before replication the laptop believes it is alone"
        );

        reconcile_all(&laptop, &rig, PENUMBRA_TABLES).await?;

        // Now the laptop can see where the work belongs.
        assert_eq!(
            device::genesis_for_user(&laptop, &tenant(), &user())
                .await?
                .map(|d| d.host),
            Some("the-rig".to_string())
        );

        // It leaves a run, which reaches the rig on the next pass.
        genesis::ask(
            &laptop,
            &GenesisRequest::new(
                tenant(),
                user(),
                CompartmentId::new("comp:rust"),
                "her-laptop",
                "the-rig",
                now,
            ),
        )
        .await?;
        assert!(genesis::list_open_for_user(&rig, &tenant(), &user())
            .await?
            .is_empty());

        reconcile_all(&laptop, &rig, PENUMBRA_TABLES).await?;
        let waiting = genesis::list_open_for_user(&rig, &tenant(), &user()).await?;
        assert_eq!(
            waiting
                .iter()
                .map(|r| (r.compartment.as_str(), r.from_host.as_str()))
                .collect::<Vec<_>>(),
            vec![("comp:rust", "her-laptop")]
        );

        // The rig runs it and closes it; the laptop learns the outcome.
        genesis::set_status(
            &rig,
            &waiting[0],
            GenesisStatus::Done,
            now + ChronoDuration::minutes(5),
        )
        .await?;
        reconcile_all(&laptop, &rig, PENUMBRA_TABLES).await?;
        assert!(
            genesis::list_open_for_user(&laptop, &tenant(), &user())
                .await?
                .is_empty(),
            "the asking node sees its run finished"
        );
        Ok(())
    }

    /// `updated_at` on these two is load-bearing, not incidental. A claim is an
    /// in-place mutation, so last-write-wins has to order it above the pending
    /// row it replaces -- otherwise a stale pending row wins the reconcile and
    /// the same run is handed to a second trainer.
    #[tokio::test]
    async fn a_claim_out_versions_the_pending_row_it_replaces() -> Result<()> {
        let laptop = node().await?;
        let rig = node().await?;
        let asked = Utc::now();
        let request = GenesisRequest::new(
            tenant(),
            user(),
            CompartmentId::new("comp:rust"),
            "her-laptop",
            "the-rig",
            asked,
        );
        genesis::ask(&laptop, &request).await?;
        reconcile_all(&laptop, &rig, PENUMBRA_TABLES).await?;

        // The rig claims it, later.
        let taken = genesis::list_open_for_user(&rig, &tenant(), &user()).await?;
        genesis::set_status(
            &rig,
            &taken[0],
            GenesisStatus::Claimed,
            asked + ChronoDuration::minutes(1),
        )
        .await?;

        // Reconciling both ways must not resurrect the pending row on either
        // side, however many passes run.
        for _ in 0..3 {
            reconcile_all(&laptop, &rig, PENUMBRA_TABLES).await?;
        }
        for (name, store) in [("laptop", &laptop), ("rig", &rig)] {
            let seen = genesis::list_open_for_user(store, &tenant(), &user()).await?;
            assert_eq!(seen.len(), 1, "{name}");
            assert_eq!(
                seen[0].status,
                GenesisStatus::Claimed,
                "{name} must not resurrect the pending row"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod refusal_tests {
    use super::*;
    use antumbra_core::{DeviceProfile, DeviceRole, TenantId, UserId};
    use antumbra_store::repo::{device, principal};
    use antumbra_store::EMBED_DIM;

    use crate::scope::Replicate;
    use crate::table::PENUMBRA_TABLES;

    const DEVICE: &TableSpec = &TableSpec {
        name: "device_profile",
        version_field: "updated_at",
        replicate: Replicate::Owned(|scope: &Scope, row| scope.is_own_user(row)),
    };

    fn tenant() -> TenantId {
        TenantId::new("ws:org")
    }

    /// Two members of one tenant, provisioned on both stores so either can sign
    /// a session in.
    async fn org() -> Result<(Store, Store, UserId, UserId)> {
        let local = Store::connect_memory(EMBED_DIM).await?;
        let remote = Store::connect_memory(EMBED_DIM).await?;
        let lily = UserId::new("user:lily");
        let oslo = UserId::new("user:oslo");
        for store in [&local, &remote] {
            principal::provision(store, &tenant(), &lily).await?;
            principal::provision(store, &tenant(), &oslo).await?;
        }
        Ok((local, remote, lily, oslo))
    }

    /// The test ADR-0017 increment 5 asks for, and the reason the record-session
    /// design is not sufficient on its own.
    ///
    /// `device_profile` is tenant-readable and own-write. Under lily's session
    /// the collector can see oslo's node and cannot write it. The engine
    /// refuses by persisting nothing and without an error, so before this the
    /// row was counted pushed and never landed -- every cycle, forever.
    #[tokio::test]
    async fn a_row_the_session_may_read_and_not_write_is_refused_not_pushed() -> Result<()> {
        let (local, remote, _lily, oslo) = org().await?;
        let now = Utc::now();
        // Oslo's machine, on the local store only.
        device::upsert(
            &local,
            &DeviceProfile::new(
                tenant(),
                oslo.clone(),
                "oslos-rig",
                "cuda",
                DeviceRole::Genesis,
                now,
            ),
        )
        .await?;

        // As owner, it replicates: this is today's collector, and the control
        // that proves the refusal below is about the session, not the row.
        let owner_pass = reconcile_table(&local, &remote, DEVICE).await?;
        assert_eq!((owner_pass.pushed, owner_pass.refused), (1, 0));

        // Now the same row, from a store where it has not landed, under lily.
        let (local, remote, lily, oslo) = org().await?;
        device::upsert(
            &local,
            &DeviceProfile::new(
                tenant(),
                oslo.clone(),
                "oslos-rig",
                "cuda",
                DeviceRole::Genesis,
                now,
            ),
        )
        .await?;
        local.signin(&tenant(), &lily).await?;
        remote.signin(&tenant(), &lily).await?;

        let scoped = reconcile_table(&local, &remote, DEVICE).await?;
        assert_eq!(
            (scoped.pushed, scoped.refused),
            (0, 1),
            "lily may read oslo's node and may not write it"
        );
        // And the engine really did refuse: nothing crossed.
        remote.invalidate().await?;
        assert!(
            device::list_for_user(&remote, &tenant(), &oslo)
                .await?
                .is_empty(),
            "the row did not land, which is what `refused` is reporting"
        );
        Ok(())
    }

    /// The other half: a row the session owns crosses normally, so `refused` is
    /// reporting the permission boundary and not simply every write under a
    /// record session.
    #[tokio::test]
    async fn a_row_the_session_owns_still_crosses() -> Result<()> {
        let (local, remote, lily, _oslo) = org().await?;
        let now = Utc::now();
        device::upsert(
            &local,
            &DeviceProfile::new(
                tenant(),
                lily.clone(),
                "her-laptop",
                "cpu",
                DeviceRole::Memory,
                now,
            ),
        )
        .await?;
        local.signin(&tenant(), &lily).await?;
        remote.signin(&tenant(), &lily).await?;

        let stats = reconcile_all(&local, &remote, PENUMBRA_TABLES).await?;
        assert_eq!(
            (stats.pushed, stats.refused),
            (1, 0),
            "her own node is hers to replicate"
        );
        remote.invalidate().await?;
        assert_eq!(
            device::list_for_user(&remote, &tenant(), &lily)
                .await?
                .len(),
            1
        );
        Ok(())
    }
}

#[cfg(test)]
mod scoped_tests {
    use super::*;
    use antumbra_core::{Compartment, DeviceProfile, DeviceRole, TenantId, UserId};
    use antumbra_store::repo::{compartment, device, principal};
    use antumbra_store::EMBED_DIM;

    use crate::config::Fabric;
    use crate::table::PENUMBRA_TABLES;

    fn tenant() -> TenantId {
        TenantId::new("ws:org")
    }

    /// Two members, each with a compartment and a machine, on the local store.
    async fn org() -> Result<(Store, Store, UserId, UserId)> {
        let local = Store::connect_memory(EMBED_DIM).await?;
        let remote = Store::connect_memory(EMBED_DIM).await?;
        let lily = UserId::new("user:lily");
        let oslo = UserId::new("user:oslo");
        for store in [&local, &remote] {
            principal::provision(store, &tenant(), &lily).await?;
            principal::provision(store, &tenant(), &oslo).await?;
        }
        let now = Utc::now();
        for (id, owner) in [("comp:hers", &lily), ("comp:his", &oslo)] {
            compartment::create(
                &local,
                &Compartment::new(id, tenant(), (*owner).clone(), id, now),
            )
            .await?;
        }
        for (host, owner, role) in [
            ("her-laptop", &lily, DeviceRole::Memory),
            ("his-rig", &oslo, DeviceRole::Genesis),
        ] {
            device::upsert(
                &local,
                &DeviceProfile::new(tenant(), (*owner).clone(), host, "cpu", role, now),
            )
            .await?;
        }
        Ok((local, remote, lily, oslo))
    }

    /// The whole point of increment 5. A collector scoped to lily carries her
    /// compartment and her machine, leaves oslo's behind, and refuses nothing --
    /// `refused` is the number that must be zero, because a refusal on a table
    /// the policy already narrowed means the policy and the ACL disagree.
    #[tokio::test]
    async fn a_scoped_collector_carries_one_fabric_and_refuses_nothing() -> Result<()> {
        let (local, remote, lily, oslo) = org().await?;
        let fabric = Fabric::new("ws:org", "user:lily");
        fabric.bind(&local).await?;
        fabric.bind(&remote).await?;
        let scope = Scope::resolve(&local, &fabric).await?;

        let stats = reconcile_all_scoped(&local, &remote, PENUMBRA_TABLES, Some(&scope)).await?;
        assert_eq!(
            stats.refused, 0,
            "a refusal here means the policy and the engine disagree"
        );
        assert!(stats.pushed > 0, "her own rows crossed");

        // Hers landed; his did not.
        remote.invalidate().await?;
        assert_eq!(
            device::list_for_user(&remote, &tenant(), &lily)
                .await?
                .len(),
            1,
            "her machine is in her fabric"
        );
        assert!(
            device::list_for_user(&remote, &tenant(), &oslo)
                .await?
                .is_empty(),
            "his machine is not"
        );
        remote.signin(&tenant(), &oslo).await?;
        assert!(
            compartment::get(
                &remote,
                &tenant(),
                &antumbra_core::CompartmentId::new("comp:his")
            )
            .await?
            .is_none(),
            "his compartment was never carried into her fabric"
        );
        Ok(())
    }

    /// The control. Without a scope the same pass is tenant-wide, which is what
    /// every deployment does today -- so the narrowing is doing the work, not
    /// some accident of the fixture.
    #[tokio::test]
    async fn an_unscoped_collector_still_carries_the_whole_tenant() -> Result<()> {
        let (local, remote, _lily, oslo) = org().await?;
        let stats = reconcile_all(&local, &remote, PENUMBRA_TABLES).await?;
        assert_eq!(stats.refused, 0);
        assert_eq!(
            device::list_for_user(&remote, &tenant(), &oslo)
                .await?
                .len(),
            1,
            "owner mode carries his machine too"
        );
        Ok(())
    }
}
