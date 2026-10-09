//! Tombstone garbage collection. Soft-deletes (forgotten memories, deleted
//! compartments, revoked grants) leave a tombstone row so the deletion propagates
//! under last-write-wins instead of resurrecting from a replica. Those tombstones
//! must eventually be hard-removed or they accumulate forever -- but only once
//! every replica has seen them, so GC runs on a **grace window wider than the sync
//! interval** (the `gc_grace_seconds` pattern). The collector spans tenants as an
//! owner/root session, so it is the natural place to run this; see [`crate::worker`].

use chrono::{DateTime, Utc};

use antumbra_core::Result;
use antumbra_store::repo::{compartment, memory};
use antumbra_store::Store;

/// Hard-remove every tombstone on `store` whose deletion is older than
/// `older_than` (memory + compartment + grant). Returns the total rows purged.
/// Idempotent and safe to run on a cadence; a tombstone within the grace window
/// is left alone so a replica that has not yet reconciled it cannot resurrect it.
pub async fn purge_store(store: &Store, older_than: DateTime<Utc>) -> Result<usize> {
    let mut purged = memory::purge(store, older_than).await?;
    purged += compartment::purge_compartments(store, older_than).await?;
    purged += compartment::purge_grants(store, older_than).await?;
    Ok(purged)
}

#[cfg(test)]
mod tests {
    use super::*;
    use antumbra_core::{Memory, MemoryNetwork, TenantId};
    use antumbra_store::repo::memory as memrepo;
    use antumbra_store::EMBED_DIM;
    use chrono::Duration;

    // A forgotten memory's tombstone is purged once past the grace window, and
    // left alone within it -- across the tables purge_store covers.
    #[tokio::test]
    async fn purge_store_removes_tombstones_past_the_grace_window() {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        let tenant = TenantId::new("t");
        let now = Utc::now();
        let m = Memory::new(
            "aaaaaaaa-0000-0000-0000-000000000001",
            tenant.clone(),
            MemoryNetwork::World,
            "forget me",
            0.8,
            now,
        );
        memrepo::upsert(&store, &m).await.unwrap();
        memrepo::soft_delete(&store, &tenant, &m.id, now)
            .await
            .unwrap();

        // Within the grace window: retained (a replica might not have seen it).
        assert_eq!(
            purge_store(&store, now - Duration::days(1)).await.unwrap(),
            0
        );
        // Past the grace window: purged.
        assert_eq!(
            purge_store(&store, now + Duration::seconds(1))
                .await
                .unwrap(),
            1
        );
    }
}
