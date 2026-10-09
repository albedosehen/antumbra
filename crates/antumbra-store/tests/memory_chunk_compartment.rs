//! The chunk index keeps its memory's privacy: a piece of a memory
//! in a private compartment is invisible to another member of the workspace,
//! by the chunk search and by the recall it feeds, until the memory moves to
//! the shared pool and its chunks are refiled. No member can write a chunk.

use chrono::Utc;

use antumbra_core::{Compartment, CompartmentId, Memory, MemoryNetwork, Result, TenantId, UserId};
use antumbra_store::repo::{compartment, memory, memory_chunk, principal};
use antumbra_store::Store;

const DIM: usize = 8;

fn axis(axis: usize) -> Vec<f32> {
    let mut v = vec![0.0; DIM];
    v[axis] = 1.0;
    v
}

/// What `user` finds near `query`: the memories the chunk search alone
/// reaches, and those recall returns.
async fn found_by(
    store: &Store,
    tenant: &TenantId,
    user: &UserId,
    query: &[f32],
) -> Result<(Vec<String>, Vec<String>)> {
    store.signin(tenant, user).await?;
    let mut pieces: Vec<String> = memory_chunk::nearest(store, tenant, query, 5, None)
        .await?
        .into_iter()
        .map(|(m, _)| m)
        .collect();
    pieces.dedup();
    let recalled = memory::recall_hybrid(store, tenant, "", query, 5, None, &[])
        .await?
        .into_iter()
        .map(|m| m.id.as_str().to_string())
        .collect();
    store.invalidate().await?;
    Ok((pieces, recalled))
}

#[tokio::test]
async fn a_private_memorys_pieces_are_found_only_by_its_audience() -> Result<()> {
    let store = Store::connect_memory(DIM).await?;
    let tenant = TenantId::new("ws:org");
    let lily = UserId::new("user:lily");
    let oslo = UserId::new("user:oslo");
    let lilys = CompartmentId::new("comp:lily-private");
    principal::provision(&store, &tenant, &lily).await?;
    principal::provision(&store, &tenant, &oslo).await?;
    compartment::create(
        &store,
        &Compartment::new(
            lilys.clone(),
            tenant.clone(),
            lily.clone(),
            "lily private",
            Utc::now(),
        ),
    )
    .await?;
    // Its whole vector points away from the query; one of its pieces is on it.
    let mut review = Memory::new(
        "memory:review",
        tenant.clone(),
        MemoryNetwork::World,
        "lily's salary review",
        0.8,
        Utc::now(),
    )
    .with_embedding(axis(0))
    .in_compartment(lilys.clone());
    memory::upsert(&store, &review).await?;
    memory_chunk::replace(&store, &review, "h", vec![axis(1), axis(5)]).await?;

    let lilys_view = found_by(&store, &tenant, &lily, &axis(5)).await?;
    assert_eq!(lilys_view.0, ["memory:review"]);
    assert_eq!(lilys_view.1, ["memory:review"]);
    assert_eq!(
        found_by(&store, &tenant, &oslo, &axis(5)).await?,
        (vec![], vec![]),
        "a member must not find another member's private memory by a piece of it"
    );

    // No member can plant or move a chunk; the engine persists nothing.
    store.signin(&tenant, &oslo).await?;
    let planted = Memory::new(
        "memory:planted",
        tenant.clone(),
        MemoryNetwork::World,
        "oslo was here",
        0.8,
        Utc::now(),
    );
    memory_chunk::replace(&store, &planted, "h", vec![axis(5)])
        .await
        .ok();
    store.invalidate().await?;
    assert!(!memory_chunk::indexed(&store, &tenant)
        .await?
        .contains_key("memory:planted"));

    // Moved to the shared pool, and refiled: now oslo finds it.
    review.compartment = None;
    memory::upsert(&store, &review).await?;
    memory_chunk::refile(&store, &review).await?;
    assert_eq!(
        found_by(&store, &tenant, &oslo, &axis(5)).await?.1,
        ["memory:review"]
    );
    Ok(())
}
