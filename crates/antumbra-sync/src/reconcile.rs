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

use crate::table::TableSpec;

/// Per-cycle counts: rows written to the remote (`pushed`) and to the local
/// (`pulled`).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ReconcileStats {
    pub pushed: usize,
    pub pulled: usize,
}

impl ReconcileStats {
    pub fn total(&self) -> usize {
        self.pushed + self.pulled
    }

    fn add(&mut self, other: ReconcileStats) {
        self.pushed += other.pushed;
        self.pulled += other.pulled;
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
    for (id, lrow) in local_rows {
        match remote_rows.get(id) {
            None => {
                if row_repo::put_row(remote, lrow).await? {
                    stats.pushed += 1;
                }
            }
            Some(rrow) => {
                if is_newer(lrow, rrow, spec) {
                    if row_repo::put_row(remote, lrow).await? {
                        stats.pushed += 1;
                    }
                } else if is_newer(rrow, lrow, spec) && row_repo::put_row(local, rrow).await? {
                    stats.pulled += 1;
                }
            }
        }
    }
    for (id, rrow) in remote_rows {
        if !local_rows.contains_key(id) && row_repo::put_row(local, rrow).await? {
            stats.pulled += 1;
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
) -> Result<(ReconcileStats, String)> {
    let since = lookback_floor(cursor, lookback);
    let local_rows =
        index_by_id(row_repo::list_rows_since(local, spec.name, spec.version_field, &since).await?);
    let remote_rows = index_by_id(
        row_repo::list_rows_since(remote, spec.name, spec.version_field, &since).await?,
    );
    let stats = reconcile_indexed(local, remote, spec, &local_rows, &remote_rows).await?;

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

/// Reconcile one table across both stores, last-write-wins (a full scan). Returns
/// what moved. For the incremental, cursor-tracked path the collector runs, see
/// [`reconcile_all_since`].
pub async fn reconcile_table(
    local: &Store,
    remote: &Store,
    spec: &TableSpec,
) -> Result<ReconcileStats> {
    Ok(
        reconcile_table_since(local, remote, spec, "", Duration::ZERO)
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
    let mut stats = ReconcileStats::default();
    for spec in tables {
        stats.add(reconcile_table(local, remote, spec).await?);
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
    let mut stats = ReconcileStats::default();
    for spec in tables {
        let cursor = cursors.get(spec.name);
        let (s, hwm) = reconcile_table_since(local, remote, spec, &cursor, lookback).await?;
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

    const MEMORY: &TableSpec = &TableSpec {
        name: "memory",
        version_field: "updated_at",
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
                pulled: 1
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
                pulled: 1
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
                pulled: 0
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
                pulled: 0
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
                pulled: 0
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
                pulled: 0
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
                pulled: 1
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
                pulled: 0
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
