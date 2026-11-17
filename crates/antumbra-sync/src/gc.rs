//! Tombstone garbage collection. Soft-deletes (forgotten memories, deleted
//! compartments, revoked grants, deleted experts) leave a tombstone row so the
//! deletion propagates
//! under last-write-wins instead of resurrecting from a replica. Those tombstones
//! must eventually be hard-removed or they accumulate forever -- but only once
//! every replica has seen them, so GC runs on a **grace window wider than the sync
//! interval** (the `gc_grace_seconds` pattern). The collector spans tenants as an
//! owner/root session, so it is the natural place to run this; see [`crate::worker`].

use chrono::{DateTime, Utc};

use antumbra_core::Result;
use antumbra_store::repo::{compartment, expert, memory};
use antumbra_store::Store;

/// Hard-remove every tombstone on `store` whose deletion is older than
/// `older_than` (memory + compartment + grant + expert). Returns the total rows
/// purged.
/// Idempotent and safe to run on a cadence; a tombstone within the grace window
/// is left alone so a replica that has not yet reconciled it cannot resurrect it.
pub async fn purge_store(store: &Store, older_than: DateTime<Utc>) -> Result<usize> {
    let mut purged = memory::purge(store, older_than).await?;
    purged += compartment::purge_compartments(store, older_than).await?;
    purged += compartment::purge_grants(store, older_than).await?;
    purged += expert::purge(store, older_than).await?;
    Ok(purged)
}

#[cfg(test)]
mod tests {
    use super::*;
    use antumbra_core::{Expert, ExpertId, Generation, Memory, MemoryNetwork, TenantId};
    use antumbra_store::repo::memory as memrepo;
    use antumbra_store::testing::mem_store;
    use chrono::Duration;

    // A forgotten memory's tombstone is purged once past the grace window, and
    // left alone within it -- across the tables purge_store covers.
    #[tokio::test]
    async fn purge_store_removes_tombstones_past_the_grace_window() -> Result<()> {
        let store = mem_store().await?;
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
        memrepo::upsert(&store, &m).await?;
        memrepo::soft_delete(&store, &tenant, &m.id, now).await?;

        // Within the grace window: retained (a replica might not have seen it).
        assert_eq!(purge_store(&store, now - Duration::days(1)).await?, 0);
        // Past the grace window: purged.
        assert_eq!(purge_store(&store, now + Duration::seconds(1)).await?, 1);
        Ok(())
    }

    // A deleted expert's tombstone goes the same way, and a live expert is
    // never touched.
    #[tokio::test]
    async fn purge_store_removes_a_deleted_expert_past_the_grace_window() -> Result<()> {
        let store = mem_store().await?;
        let now = Utc::now();
        let minted = |id: &str| Expert {
            id: ExpertId::new(id),
            name: id.to_string(),
            base_model: "base".into(),
            artifact_uri: "adapters/x.safetensors".into(),
            capability_card: serde_json::json!({}),
            capability_vec: None,
            fitness: 1.0,
            frozen_at: Some(now),
            generation: Generation::ZERO,
            owner: None,
            compartment: None,
            placed_on: None,
            created_at: now,
        };
        expert::insert(&store, &minted("expert:kept")).await?;
        expert::insert(&store, &minted("expert:gone")).await?;
        expert::delete(&store, &ExpertId::new("expert:gone")).await?;

        assert_eq!(purge_store(&store, now - Duration::days(1)).await?, 0);
        assert_eq!(
            purge_store(&store, Utc::now() + Duration::seconds(1)).await?,
            1
        );
        let left = expert::list(&store).await?;
        assert_eq!(left.len(), 1);
        assert_eq!(left[0].id.as_str(), "expert:kept");
        Ok(())
    }
}
