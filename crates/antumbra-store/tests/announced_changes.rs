//! An embedded store announces its memory writes in-process, which is how it
//! propagates live without a LIVE query: each write's row, with the action a
//! LIVE watch would report, from the store and from every clone made after
//! it asked.

use antumbra_core::{CompartmentId, Memory, MemoryNetwork, TenantId};
use antumbra_store::repo::memory;
use antumbra_store::repo::sync::ChangeAction;
use antumbra_store::{Store, EMBED_DIM};

#[tokio::test]
async fn each_memory_write_is_announced_with_its_row() {
    let mut store = Store::connect_memory(EMBED_DIM).await.unwrap();
    let mut changes = store.announce_changes();
    // A clone made after the call announces on the same channel, as the
    // serving connection cloned from the server's store does.
    let serving = store.clone();
    let tenant = TenantId::new("t");
    let now = chrono::Utc::now();
    let m = Memory::new(
        "11111111-2222-3333-4444-555555555555",
        tenant.clone(),
        MemoryNetwork::World,
        "a shared note",
        0.5,
        now,
    )
    .in_compartment(CompartmentId::new("comp-1"));

    memory::upsert(&serving, &m).await.unwrap();
    memory::upsert(&serving, &m).await.unwrap();
    memory::reinforce(&serving, &tenant, &m.id, now)
        .await
        .unwrap();
    memory::soft_delete(&serving, &tenant, &m.id, now)
        .await
        .unwrap();
    // Removing the tombstone says nothing: its forget was the announcement.
    memory::delete(&store, &tenant, &m.id).await.unwrap();
    // A live memory removed outright is announced as a delete.
    let gone = Memory::new(
        "66666666-7777-8888-9999-000000000000",
        tenant.clone(),
        MemoryNetwork::World,
        "a note removed outright",
        0.5,
        now,
    )
    .in_compartment(CompartmentId::new("comp-1"));
    memory::upsert(&serving, &gone).await.unwrap();
    memory::delete(&serving, &tenant, &gone.id).await.unwrap();

    let mut seen = Vec::new();
    while let Ok(event) = changes.try_recv() {
        assert_eq!(event.row["tenant_id"], "t");
        assert_eq!(event.row["compartment"], "comp-1");
        let tombstone = event.row.get("deleted_at").is_some_and(|v| !v.is_null());
        let key = event.row["key"].as_str().unwrap_or_default().to_string();
        seen.push((event.action, tombstone, key == m.id.as_str()));
    }
    assert_eq!(
        seen,
        [
            (ChangeAction::Create, false, true),
            (ChangeAction::Update, false, true),
            (ChangeAction::Update, false, true),
            // The forget is the tombstone write, which routes as a delete.
            (ChangeAction::Update, true, true),
            (ChangeAction::Create, false, false),
            (ChangeAction::Delete, false, false),
        ]
    );
}
