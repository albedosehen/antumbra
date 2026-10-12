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

/// A chunk search the engine stops at its timeout finds nothing, rather than
/// failing the recall it is one leg of; within its time it finds the pieces.
#[tokio::test]
async fn a_chunk_search_stopped_at_its_timeout_finds_nothing_rather_than_failing() -> Result<()> {
    let store = Store::connect_memory(EMBED_DIM).await?;
    let tenant = TenantId::new("ws:a");
    for i in 0..40 {
        let m = memory(&format!("memory:m{i}"), "ws:a", "a long memory", axis(i));
        replace(
            &store,
            &m,
            "hash",
            (0..6).map(|j| axis((i + j) % 300)).collect(),
        )
        .await?;
    }
    let found =
        nearest_scored_within(&store, &tenant, &axis(3), 10, None, Duration::from_secs(10)).await?;
    assert!(!found.is_empty(), "within its time it finds the pieces");

    let stopped =
        nearest_scored_within(&store, &tenant, &axis(3), 10, None, Duration::from_nanos(1)).await;
    assert!(
        matches!(&stopped, Ok(p) if p.is_empty()),
        "stopped, it finds nothing and does not fail: {stopped:?}"
    );
    assert!(
        SEARCH_TIMEOUT >= Duration::from_millis(600),
        "above the owner's slowest day"
    );
    Ok(())
}

#[test]
fn a_timeout_is_told_from_other_errors() {
    // What the engine says, verbatim (SurrealDB 3, embedded).
    assert!(timed_out(
        "query error: The query was not executed because it exceeded the timeout: 1ns"
    ));
    assert!(timed_out("Query timed out"));
    assert!(!timed_out(
        "There was a problem with the database: Not enough permissions"
    ));
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

/// A member who can read little of a workspace is searched exactly: its own
/// compartment, one granted to it, one offered to the hive and the shared
/// pool, nearest first, and never a piece of another member's private
/// compartment, even when the list of compartments names it. A session that
/// can read more than the exact search takes is walked, and remembered.
#[tokio::test]
async fn a_member_who_reads_little_is_searched_exactly_and_sees_only_what_it_may() -> Result<()> {
    use crate::repo::{compartment, hive, principal};
    use antumbra_core::{
        Capability, Compartment, Grant, Hive, HiveMembership, HiveOffer, OfferStatus, OfferedKind,
    };
    let store = Store::connect_memory(EMBED_DIM).await?;
    let tenant = TenantId::new("ws:exact");
    let (owner, member) = (UserId::new("user:owner"), UserId::new("user:member"));
    principal::provision(&store, &tenant, &owner).await?;
    principal::provision(&store, &tenant, &member).await?;
    let now = chrono::Utc::now();
    for (id, who) in [
        ("comp:owner-private", &owner),
        ("comp:owner-granted", &owner),
        ("comp:owner-hive", &owner),
        ("comp:member", &member),
    ] {
        let c = Compartment::new(
            CompartmentId::new(id),
            tenant.clone(),
            who.clone(),
            "c",
            now,
        );
        compartment::create(&store, &c).await?;
    }
    compartment::grant(
        &store,
        &Grant {
            tenant: tenant.clone(),
            compartment: CompartmentId::new("comp:owner-granted"),
            grantee: member.clone(),
            capability: Capability::Reference,
            granted_by: owner.clone(),
            created_at: now,
            updated_at: now,
            deleted_at: None,
        },
    )
    .await?;
    hive::set_enabled(&store, &Hive::new(tenant.clone(), true, now)).await?;
    hive::set_membership(
        &store,
        &HiveMembership::new(tenant.clone(), owner.clone(), true, now),
    )
    .await?;
    let mut offer = HiveOffer::new(
        tenant.clone(),
        OfferedKind::Compartment,
        "comp:owner-hive",
        owner.clone(),
        now,
    );
    offer.status = OfferStatus::Accepted;
    hive::offer(&store, &offer).await?;
    // A memory of two pieces in each place, on axes of their own.
    for (id, place, at) in [
        ("memory:private", Some("comp:owner-private"), 10),
        ("memory:granted", Some("comp:owner-granted"), 20),
        ("memory:hive", Some("comp:owner-hive"), 30),
        ("memory:mine", Some("comp:member"), 40),
        ("memory:pool", None, 50),
    ] {
        let mut m = memory(id, "ws:exact", "a long memory", axis(at));
        if let Some(c) = place {
            m = m.in_compartment(CompartmentId::new(c));
        }
        replace(&store, &m, "hash", vec![axis(at), axis(at + 1)]).await?;
    }
    let long = Duration::from_secs(10);

    store.signin(&tenant, &member).await?;
    assert_eq!(store.session_user(), Some(member.clone()));
    let readable = readable_compartments(&store, &tenant, &member).await?;
    assert_eq!(
        readable,
        ["comp:member", "comp:owner-granted", "comp:owner-hive"]
    );
    let found = search(&store, &tenant, &axis(21), 5, None, long, 100).await?;
    assert_eq!(
        found.len(),
        8,
        "every piece it can read: four memories of two"
    );
    assert_eq!(
        found[0],
        ("memory:granted".to_string(), 1.0),
        "nearest first"
    );
    assert!(
        found.iter().all(|(m, _)| m != "memory:private"),
        "{found:?}"
    );
    assert!(!walks(&walk_key(&tenant, &member, &readable)));
    // The list decides what is cheap to read, never what is visible.
    let forged = vec!["comp:owner-private".to_string()];
    let read = exact(&store, &tenant, &axis(10), 20, None, &forged, long, 100)
        .await?
        .expect("within the cap");
    assert!(read.iter().all(|(m, _)| m != "memory:private"), "{read:?}");

    store.signin(&tenant, &owner).await?;
    let key = walk_key(
        &tenant,
        &owner,
        &readable_compartments(&store, &tenant, &owner).await?,
    );
    assert!(!walks(&key));
    let walked = search(&store, &tenant, &axis(10), 5, None, long, 4).await?;
    assert!(
        walks(&key),
        "more than the exact search takes: walked, and remembered"
    );
    assert_eq!(walked[0].0, "memory:private");

    store.signin_root().await?;
    assert_eq!(
        store.session_user(),
        None,
        "the owner view searches by the walk"
    );
    Ok(())
}
