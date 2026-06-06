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
//! settles instead of echoing. Deletes are not propagated in this cut (no
//! tombstones); a row removed on one side is re-seeded from the other.

use chrono::{DateTime, Utc};
use serde_json::Value;
use std::collections::BTreeMap;

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

/// Reconcile one table across both stores, last-write-wins. Returns what moved.
pub async fn reconcile_table(
    local: &Store,
    remote: &Store,
    spec: &TableSpec,
) -> Result<ReconcileStats> {
    let local_rows = index_by_id(row_repo::list_rows(local, spec.name).await?);
    let remote_rows = index_by_id(row_repo::list_rows(remote, spec.name).await?);
    let mut stats = ReconcileStats::default();

    for (id, lrow) in &local_rows {
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
    for (id, rrow) in &remote_rows {
        if !local_rows.contains_key(id) && row_repo::put_row(local, rrow).await? {
            stats.pulled += 1;
        }
    }
    Ok(stats)
}

/// Reconcile every table, in the given order.
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
        memory::upsert(&local, &mem("aaaaaaaa-0000-0000-0000-000000000001", &tenant, "left", now))
            .await
            .unwrap();
        memory::upsert(&remote, &mem("bbbbbbbb-0000-0000-0000-000000000002", &tenant, "right", now))
            .await
            .unwrap();

        let stats = reconcile_table(&local, &remote, MEMORY).await.unwrap();
        assert_eq!(stats, ReconcileStats { pushed: 1, pulled: 1 });

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
        memory::upsert(&local, &mem(id, &tenant, "stale", t0)).await.unwrap();
        memory::upsert(&remote, &mem(id, &tenant, "fresh", t1)).await.unwrap();

        let stats = reconcile_table(&local, &remote, MEMORY).await.unwrap();
        assert_eq!(stats, ReconcileStats { pushed: 0, pulled: 1 }, "newer remote pulled to local");

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
        memory::upsert(&local, &mem(id, &tenant, "live", t0)).await.unwrap();
        memory::upsert(&remote, &mem(id, &tenant, "live", t0)).await.unwrap();

        // Local forgets it (a tombstone, newer than remote's live copy).
        memory::soft_delete(&local, &tenant, &mid, t0 + ChronoDuration::seconds(5))
            .await
            .unwrap();

        let stats = reconcile_table(&local, &remote, MEMORY).await.unwrap();
        assert_eq!(stats, ReconcileStats { pushed: 1, pulled: 0 }, "tombstone pushed");

        // Forgotten on both sides; the live copy did not resurrect it on local.
        assert!(memory::get(&remote, &tenant, &mid).await.unwrap().is_none());
        assert!(memory::get(&local, &tenant, &mid).await.unwrap().is_none());

        // Converged.
        assert_eq!(
            reconcile_table(&local, &remote, MEMORY).await.unwrap(),
            ReconcileStats::default()
        );
    }
}
