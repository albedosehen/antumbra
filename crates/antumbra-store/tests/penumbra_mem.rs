//! Penumbra memory tenant-isolation regression (mirrors the data-plane POC's
//! `test_regression_tenant_isolation`): no workspace may recall, list, read by
//! id, or delete another workspace's traces, even when the other tenant's
//! trace is the nearest vector match.

use anyhow::Result;
use chrono::Utc;

use antumbra_core::{CompartmentId, ExpertId, Memory, MemoryId, MemoryNetwork, TenantId};
use antumbra_store::repo::memory;
use antumbra_store::testing;

#[tokio::test]
async fn list_by_compartment_gathers_only_that_compartment() -> Result<()> {
    let store = testing::mem_store_of(4).await?;
    let ws = TenantId::new("ws:a");
    let mk = |id: &str, comp: Option<&str>| {
        let m = Memory::new(id, "ws:a", MemoryNetwork::World, "c", 0.8, Utc::now())
            .with_embedding(vec![1.0, 0.0, 0.0, 0.0]);
        match comp {
            Some(c) => m.in_compartment(c),
            None => m,
        }
    };
    memory::upsert(&store, &mk("memory:1", Some("comp:x"))).await?;
    memory::upsert(&store, &mk("memory:2", Some("comp:y"))).await?;
    memory::upsert(&store, &mk("memory:3", None)).await?;

    let got = memory::list_by_compartment(&store, &ws, &CompartmentId::new("comp:x")).await?;
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].id, MemoryId::new("memory:1"));
    Ok(())
}

fn trace(id: &str, ws: &str, net: MemoryNetwork, content: &str, embed: Vec<f32>) -> Memory {
    Memory::new(id, ws, net, content, 0.8, Utc::now()).with_embedding(embed)
}

#[tokio::test]
async fn list_and_recall_are_tenant_scoped() -> Result<()> {
    let store = testing::mem_store_of(4).await?;
    let alpha = TenantId::new("ws:alpha");
    let beta = TenantId::new("ws:beta");

    // Alpha owns one trace; beta owns a trace whose embedding is the CLOSEST to
    // the query; if isolation leaked, alpha's recall would surface beta's.
    memory::upsert(
        &store,
        &trace(
            "memory:a1",
            "ws:alpha",
            MemoryNetwork::World,
            "alpha fact",
            vec![1.0, 0.0, 0.0, 0.0],
        ),
    )
    .await?;
    memory::upsert(
        &store,
        &trace(
            "memory:b1",
            "ws:beta",
            MemoryNetwork::World,
            "beta secret",
            vec![0.95, 0.05, 0.0, 0.0],
        ),
    )
    .await?;

    // list is tenant-filtered.
    assert_eq!(memory::list(&store, &alpha).await?.len(), 1);
    assert_eq!(memory::list(&store, &beta).await?.len(), 1);

    // recall for alpha, with a query nearest to beta's trace, must return ONLY
    // alpha's trace; beta's closer vector is never a candidate.
    let hits = memory::recall(&store, &alpha, &[0.95, 0.05, 0.0, 0.0], 5, None).await?;
    assert_eq!(hits.len(), 1, "cross-tenant recall leak");
    assert_eq!(hits[0].id, MemoryId::new("memory:a1"));
    assert_eq!(hits[0].content, "alpha fact");
    Ok(())
}

#[tokio::test]
async fn get_by_id_is_tenant_checked() -> Result<()> {
    let store = testing::mem_store_of(4).await?;
    let alpha = TenantId::new("ws:alpha");
    let beta = TenantId::new("ws:beta");
    memory::upsert(
        &store,
        &trace(
            "memory:owned",
            "ws:beta",
            MemoryNetwork::Bank,
            "beta only",
            vec![0.0, 1.0, 0.0, 0.0],
        ),
    )
    .await?;

    let id = MemoryId::new("memory:owned");
    // The owner sees it; another tenant does not, even with the exact id.
    assert!(memory::get(&store, &beta, &id).await?.is_some());
    assert!(
        memory::get(&store, &alpha, &id).await?.is_none(),
        "cross-tenant read leak"
    );
    Ok(())
}

#[tokio::test]
async fn delete_cannot_cross_tenants() -> Result<()> {
    let store = testing::mem_store_of(4).await?;
    let alpha = TenantId::new("ws:alpha");
    let beta = TenantId::new("ws:beta");
    let id = MemoryId::new("memory:keep");
    memory::upsert(
        &store,
        &trace(
            "memory:keep",
            "ws:beta",
            MemoryNetwork::World,
            "beta keeps this",
            vec![1.0, 0.0, 0.0, 0.0],
        ),
    )
    .await?;

    // Alpha's delete of beta's id is a no-op; beta still has it.
    memory::delete(&store, &alpha, &id).await?;
    assert!(memory::get(&store, &beta, &id).await?.is_some());

    // The owner can delete it.
    memory::delete(&store, &beta, &id).await?;
    assert!(memory::get(&store, &beta, &id).await?.is_none());
    Ok(())
}

#[tokio::test]
async fn recall_filters_by_network() -> Result<()> {
    let store = testing::mem_store_of(4).await?;
    let ws = TenantId::new("ws:alpha");
    memory::upsert(
        &store,
        &trace(
            "memory:w",
            "ws:alpha",
            MemoryNetwork::World,
            "a world fact",
            vec![1.0, 0.0, 0.0, 0.0],
        ),
    )
    .await?;
    memory::upsert(
        &store,
        &trace(
            "memory:o",
            "ws:alpha",
            MemoryNetwork::Opinion,
            "an opinion",
            vec![1.0, 0.0, 0.0, 0.0],
        ),
    )
    .await?;

    let world = memory::recall(
        &store,
        &ws,
        &[1.0, 0.0, 0.0, 0.0],
        5,
        Some(MemoryNetwork::World),
    )
    .await?;
    assert_eq!(world.len(), 1);
    assert_eq!(world[0].network, MemoryNetwork::World);
    Ok(())
}

#[tokio::test]
async fn reinforce_and_consolidate_persist() -> Result<()> {
    let store = testing::mem_store_of(4).await?;
    let ws = TenantId::new("ws:alpha");
    let id = MemoryId::new("memory:r");
    memory::upsert(
        &store,
        &trace(
            "memory:r",
            "ws:alpha",
            MemoryNetwork::World,
            "reinforce me",
            vec![1.0, 0.0, 0.0, 0.0],
        ),
    )
    .await?;

    let Some(reinforced) = memory::reinforce(&store, &ws, &id, Utc::now()).await? else {
        anyhow::bail!("reinforce found no memory");
    };
    assert_eq!(reinforced.reinforcement, 1);
    assert!(reinforced.confidence > 0.8);

    let Some(consolidated) = memory::mark_consolidated(
        &store,
        &ws,
        &id,
        ExpertId::new("expert:consolidated-world"),
        Utc::now(),
    )
    .await?
    else {
        anyhow::bail!("mark_consolidated found no memory");
    };
    assert!(consolidated.is_consolidated());

    // Persisted: reload reflects both the reinforcement and the graduation link.
    let Some(reloaded) = memory::get(&store, &ws, &id).await? else {
        anyhow::bail!("the memory does not read back");
    };
    assert_eq!(reloaded.reinforcement, 1);
    assert_eq!(
        reloaded.consolidated_expert,
        Some(ExpertId::new("expert:consolidated-world"))
    );
    Ok(())
}
