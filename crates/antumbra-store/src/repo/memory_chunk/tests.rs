use super::*;
use crate::repo::memory;
use crate::schema::EMBED_DIM;
use antumbra_core::{CompartmentId, MemoryNetwork};

/// A unit vector along `axis`, so similarities are exact and the test reads
/// as geometry rather than as an embedder.
fn axis(axis: usize) -> Vec<f32> {
    let mut v = vec![0.0; EMBED_DIM];
    v[axis] = 1.0;
    v
}

fn memory(id: &str, tenant: &str, content: &str, whole: Vec<f32>) -> Memory {
    Memory::new(
        id,
        TenantId::new(tenant),
        MemoryNetwork::World,
        content,
        0.8,
        chrono::Utc::now(),
    )
    .with_embedding(whole)
}

/// Chunks are replaced whole, read back by their memory, found by their
/// vector, and dropped with their memory; another workspace never sees them.
#[tokio::test]
async fn chunks_are_replaced_found_and_removed_by_their_memory() -> Result<()> {
    let store = Store::connect_memory(EMBED_DIM).await?;
    let tenant = TenantId::new("ws:a");
    let m = memory("memory:long", "ws:a", "a long memory", axis(0))
        .in_compartment(CompartmentId::new("comp:a"));
    replace(&store, &m, "hash-1", vec![axis(1), axis(2), axis(3)]).await?;
    replace(&store, &m, "hash-2", vec![axis(4), axis(5)]).await?;

    let held = indexed(&store, &tenant).await?;
    assert_eq!(
        held.get("memory:long"),
        Some(&Indexed {
            content_hash: "hash-2".into(),
            compartment: Some("comp:a".into()),
            network: MemoryNetwork::World,
        }),
        "the second cut replaced the first"
    );
    let among = ["memory:long", "memory:uncut"].map(String::from);
    assert_eq!(
        indexed_among(&store, &tenant, &among).await?,
        held,
        "read by key, the same as read whole"
    );
    assert!(indexed_among(&store, &TenantId::new("ws:b"), &among)
        .await?
        .is_empty());
    let near = nearest(&store, &tenant, &axis(5), 5, None).await?;
    assert_eq!(
        near[0],
        ("memory:long".to_string(), axis(5)),
        "nearest piece first"
    );
    assert!(
        nearest(&store, &TenantId::new("ws:b"), &axis(5), 5, None)
            .await?
            .is_empty(),
        "another workspace finds nothing"
    );

    remove(&store, &tenant, "memory:long").await?;
    assert!(indexed(&store, &tenant).await?.is_empty());
    Ok(())
}

/// A memory that moves keeps its chunks' vectors: refiling rewrites where
/// they are filed and nothing else.
#[tokio::test]
async fn a_moved_memory_is_refiled_with_its_vectors() -> Result<()> {
    let store = Store::connect_memory(EMBED_DIM).await?;
    let tenant = TenantId::new("ws:a");
    let mut m = memory("memory:long", "ws:a", "a long memory", axis(0))
        .in_compartment(CompartmentId::new("comp:a"));
    replace(&store, &m, "hash", vec![axis(1), axis(2)]).await?;
    assert!(indexed(&store, &tenant).await?["memory:long"].filed_as(&m));

    m.compartment = None;
    m.network = MemoryNetwork::Bank;
    assert!(!indexed(&store, &tenant).await?["memory:long"].filed_as(&m));
    refile(&store, &m).await?;
    let held = &indexed(&store, &tenant).await?["memory:long"];
    assert!(held.filed_as(&m), "{held:?}");
    assert_eq!(held.content_hash, "hash");
    let near = nearest(&store, &tenant, &axis(2), 5, Some(MemoryNetwork::Bank)).await?;
    assert_eq!(near.len(), 2, "both pieces kept: {near:?}");
    assert_eq!(near[0], ("memory:long".to_string(), axis(2)));
    assert!(
        nearest(&store, &tenant, &axis(2), 5, Some(MemoryNetwork::World))
            .await?
            .is_empty()
    );
    Ok(())
}

/// A memory whose whole vector points elsewhere is found by a piece of it:
/// the fused dense leg returns it where the whole-memory leg alone does not,
/// and a memory with no chunks is still found by its whole vector.
#[tokio::test]
async fn the_dense_leg_finds_a_memory_by_a_piece_of_it() -> Result<()> {
    let store = Store::connect_memory(EMBED_DIM).await?;
    let tenant = TenantId::new("ws:a");
    // The memory the query is about: its whole vector says something else.
    let target = memory("memory:target", "ws:a", "notes", axis(0));
    memory::upsert(&store, &target).await?;
    replace(&store, &target, "h", vec![axis(0), axis(7)]).await?;
    // Distractors whose whole vectors sit nearer the query than the target's.
    for i in 1..=6 {
        let mut v = axis(7);
        v[10 + i] = 0.9;
        memory::upsert(
            &store,
            &memory(&format!("memory:near-{i}"), "ws:a", "other notes", v),
        )
        .await?;
    }
    // One with no chunks at all, found by its whole vector as before.
    memory::upsert(
        &store,
        &memory("memory:unchunked", "ws:a", "plain", axis(9)),
    )
    .await?;

    let query = axis(7);
    let alone = memory::recall(&store, &tenant, &query, 3, None).await?;
    assert!(
        !alone.iter().any(|m| m.id.as_str() == "memory:target"),
        "the whole-memory leg alone misses it"
    );
    let fused = memory::recall_hybrid(&store, &tenant, "", &query, 3, None, &[]).await?;
    assert!(
        fused.iter().any(|m| m.id.as_str() == "memory:target"),
        "a piece of it is nearest the query: {:?}",
        fused.iter().map(|m| m.id.as_str()).collect::<Vec<_>>()
    );
    let plain = memory::recall_hybrid(&store, &tenant, "", &axis(9), 1, None, &[]).await?;
    assert_eq!(plain[0].id.as_str(), "memory:unchunked");

    // Forgotten, it drops out even before its chunks are removed.
    memory::soft_delete(&store, &tenant, &target.id, chrono::Utc::now()).await?;
    let after = memory::recall_hybrid(&store, &tenant, "", &query, 3, None, &[]).await?;
    assert!(!after.iter().any(|m| m.id.as_str() == "memory:target"));
    Ok(())
}

/// What the chunker reads: every memory of a workspace, or those changed
/// since a time, forgotten ones flagged; and every workspace that has any.
#[tokio::test]
async fn the_chunker_reads_what_changed_and_every_workspace() -> Result<()> {
    let store = Store::connect_memory(EMBED_DIM).await?;
    let tenant = TenantId::new("ws:a");
    let before = chrono::Utc::now() - chrono::Duration::seconds(5);
    memory::upsert(&store, &memory("memory:one", "ws:a", "one", axis(1))).await?;
    memory::upsert(&store, &memory("memory:two", "ws:b", "two", axis(2))).await?;
    let gone = memory("memory:gone", "ws:a", "gone", axis(3));
    memory::upsert(&store, &gone).await?;
    memory::soft_delete(&store, &tenant, &gone.id, chrono::Utc::now()).await?;

    let all = sources(&store, &tenant, None).await?;
    let mut seen: Vec<(String, bool)> = all
        .iter()
        .map(|s| (s.memory.id.as_str().to_string(), s.forgotten))
        .collect();
    seen.sort();
    assert_eq!(
        seen,
        [("memory:gone".into(), true), ("memory:one".into(), false)]
    );
    assert!(
        all.iter().all(|s| s.memory.embedding.is_none()),
        "no embeddings read"
    );
    assert_eq!(sources(&store, &tenant, Some(before)).await?.len(), 2);
    let later = chrono::Utc::now() + chrono::Duration::seconds(5);
    assert!(sources(&store, &tenant, Some(later)).await?.is_empty());

    let mut tenants: Vec<String> = tenants(&store)
        .await?
        .into_iter()
        .map(|t| t.as_str().to_string())
        .collect();
    tenants.sort();
    assert_eq!(tenants, ["ws:a", "ws:b"]);
    Ok(())
}

/// The chunk leg fetches the memories only it found by their keys: the live
/// ones of the workspace, never a forgotten one, another workspace's, or one
/// that is not there.
#[tokio::test]
async fn the_chunk_leg_fetches_its_memories_by_key() -> Result<()> {
    let store = Store::connect_memory(EMBED_DIM).await?;
    let tenant = TenantId::new("ws:a");
    for (id, ws) in [
        ("memory:one", "ws:a"),
        ("memory:two", "ws:a"),
        ("memory:gone", "ws:a"),
        ("memory:theirs", "ws:b"),
    ] {
        memory::upsert(&store, &memory(id, ws, id, axis(1))).await?;
    }
    memory::soft_delete(
        &store,
        &tenant,
        &antumbra_core::MemoryId::new("memory:gone"),
        chrono::Utc::now(),
    )
    .await?;
    let keys: Vec<String> = [
        "memory:one",
        "memory:two",
        "memory:gone",
        "memory:theirs",
        "memory:absent",
    ]
    .map(String::from)
    .to_vec();
    let mut got: Vec<String> = memory::get_many(&store, &tenant, &keys)
        .await?
        .into_iter()
        .map(|m| {
            assert!(m.embedding.is_some(), "with the vector the leg scores");
            m.id.as_str().to_string()
        })
        .collect();
    got.sort();
    assert_eq!(got, ["memory:one", "memory:two"]);
    assert!(memory::get_many(&store, &tenant, &[]).await?.is_empty());
    Ok(())
}
