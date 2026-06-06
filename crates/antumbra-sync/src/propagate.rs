//! Live propagation (R-2): turn a change to a *shared* memory into a
//! [`MemoryChange`] addressed to everyone who may see it, so a grantee's agent
//! learns of new/planned memories without polling.
//!
//! This is the engine half -- it watches the store's change feed and resolves
//! each change's audience. Delivering the change to a subscriber's transport (an
//! MCP notification over the SSE stream) is the consumer's job; keeping the two
//! apart lets the routing be tested without a live MCP client.

use tokio::sync::mpsc;

use antumbra_core::{CompartmentId, MemoryId, Result, TenantId, UserId};
use antumbra_store::repo::compartment;
use antumbra_store::repo::sync::{self as rows, ChangeAction, ChangeEvent};
use antumbra_store::Store;

/// Everyone who may see a compartment's memories: its owner plus every grantee.
pub async fn audience(
    store: &Store,
    tenant: &TenantId,
    compartment: &CompartmentId,
) -> Result<Vec<UserId>> {
    let mut who: Vec<UserId> = Vec::new();
    if let Some(c) = compartment::get(store, tenant, compartment).await? {
        who.push(c.owner);
    }
    for grant in compartment::list_grants(store, tenant, compartment).await? {
        if !who.iter().any(|u| u.as_str() == grant.grantee.as_str()) {
            who.push(grant.grantee);
        }
    }
    Ok(who)
}

/// A change to a shared (compartmentalized) memory, addressed to its audience.
#[derive(Debug, Clone)]
pub struct MemoryChange {
    pub action: ChangeAction,
    pub tenant: TenantId,
    pub compartment: CompartmentId,
    pub memory: MemoryId,
    pub recipients: Vec<UserId>,
}

/// Watch the `memory` table and emit a [`MemoryChange`] for every change to a
/// *compartmentalized* memory, resolved to its audience. Tenant-wide
/// (un-compartmentalized) writes are skipped -- R-2 is shared-compartment
/// awareness. Deletes carry no compartment in the notification payload, so they
/// are not routed in this cut (a known gap, like delete sync in R-1).
///
/// A background task owns the subscription and exits when the feed ends or the
/// returned receiver is dropped.
pub async fn watch_shared_memories(store: &Store) -> Result<mpsc::Receiver<MemoryChange>> {
    let mut feed = rows::watch_table(store, "memory").await?;
    let (tx, rx) = mpsc::channel(64);
    let store = store.clone();
    tokio::spawn(async move {
        while let Some(event) = feed.recv().await {
            match resolve_change(&store, &event).await {
                Some(change) => {
                    if tx.send(change).await.is_err() {
                        break; // receiver dropped
                    }
                }
                None => continue, // not a compartment change, or unresolved
            }
        }
    });
    Ok(rx)
}

/// Parse one change-feed event and resolve its audience, or `None` if it is not
/// a routable shared-compartment change. Public so a transport that owns the
/// change feed directly (e.g. the MCP server, which must resolve under its own
/// connection lock in owner mode) can reuse the exact routing.
pub async fn resolve_change(store: &Store, event: &ChangeEvent) -> Option<MemoryChange> {
    let row = &event.row;
    let tenant = TenantId::new(row.get("tenant_id")?.as_str()?);
    let compartment = CompartmentId::new(row.get("compartment")?.as_str()?);
    let memory = MemoryId::new(row.get("key")?.as_str()?);
    let recipients = audience(store, &tenant, &compartment).await.ok()?;
    // A forget is a tombstone write (an update carrying `deleted_at`), which still
    // has the compartment -- so unlike a hard delete it routes. Surface it to the
    // agent as a delete rather than the raw update action.
    let action = if row.get("deleted_at").and_then(|v| v.as_str()).is_some() {
        ChangeAction::Delete
    } else {
        event.action
    };
    Some(MemoryChange {
        action,
        tenant,
        compartment,
        memory,
        recipients,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use antumbra_core::{
        Capability, Compartment, CompartmentId, Grant, Memory, MemoryNetwork, Origin, TenantId,
        UserId,
    };
    use antumbra_store::repo::memory;
    use antumbra_store::EMBED_DIM;

    // A write into a shared compartment is routed to the owner and every grantee.
    #[tokio::test]
    async fn a_shared_write_reaches_owner_and_grantees() {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        let tenant = TenantId::new("t");
        let comp = CompartmentId::new("comp-1");
        let now = chrono::Utc::now();

        // alice owns the compartment; bob is a grantee.
        compartment::create(
            &store,
            &Compartment {
                id: comp.clone(),
                tenant: tenant.clone(),
                owner: UserId::new("alice"),
                name: "shared".into(),
                origin: Origin::User,
                created_at: now,
            },
        )
        .await
        .unwrap();
        compartment::grant(
            &store,
            &Grant {
                tenant: tenant.clone(),
                compartment: comp.clone(),
                grantee: UserId::new("bob"),
                capability: Capability::Reference,
                granted_by: UserId::new("alice"),
                created_at: now,
            },
        )
        .await
        .unwrap();

        let mut rx = watch_shared_memories(&store).await.unwrap();

        let m = Memory::new(
            "aaaaaaaa-0000-0000-0000-00000000000a",
            tenant.clone(),
            MemoryNetwork::World,
            "shared note",
            0.9,
            now,
        )
        .in_compartment(comp.clone());
        memory::upsert(&store, &m).await.unwrap();

        let change = tokio::time::timeout(std::time::Duration::from_secs(3), rx.recv())
            .await
            .expect("a change arrives")
            .expect("channel open");
        assert_eq!(change.action, ChangeAction::Create);
        assert_eq!(change.compartment.as_str(), "comp-1");
        let names: Vec<&str> = change.recipients.iter().map(UserId::as_str).collect();
        assert!(names.contains(&"alice"), "owner notified: {names:?}");
        assert!(names.contains(&"bob"), "grantee notified: {names:?}");
    }

    // A forget on a shared memory routes a *Delete* change to the audience -- the
    // tombstone update still carries the compartment, so unlike a hard delete it
    // is routable.
    #[tokio::test]
    async fn a_forget_routes_a_delete_to_the_audience() {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        let tenant = TenantId::new("t");
        let comp = CompartmentId::new("comp-2");
        let now = chrono::Utc::now();
        compartment::create(
            &store,
            &Compartment {
                id: comp.clone(),
                tenant: tenant.clone(),
                owner: UserId::new("alice"),
                name: "shared".into(),
                origin: Origin::User,
                created_at: now,
            },
        )
        .await
        .unwrap();
        compartment::grant(
            &store,
            &Grant {
                tenant: tenant.clone(),
                compartment: comp.clone(),
                grantee: UserId::new("bob"),
                capability: Capability::Reference,
                granted_by: UserId::new("alice"),
                created_at: now,
            },
        )
        .await
        .unwrap();

        let mut rx = watch_shared_memories(&store).await.unwrap();
        let m = Memory::new(
            "eeeeeeee-0000-0000-0000-00000000000e",
            tenant.clone(),
            MemoryNetwork::World,
            "to be forgotten",
            0.9,
            now,
        )
        .in_compartment(comp.clone());
        memory::upsert(&store, &m).await.unwrap();
        memory::soft_delete(&store, &tenant, &m.id, now).await.unwrap();

        // The create then the delete arrive; read until the Delete is seen.
        let deleted = tokio::time::timeout(std::time::Duration::from_secs(3), async {
            while let Some(change) = rx.recv().await {
                if change.action == ChangeAction::Delete {
                    return Some(change);
                }
            }
            None
        })
        .await
        .expect("a change arrives before timeout")
        .expect("a delete change is routed");
        assert_eq!(deleted.memory.as_str(), "eeeeeeee-0000-0000-0000-00000000000e");
        let names: Vec<&str> = deleted.recipients.iter().map(UserId::as_str).collect();
        assert!(names.contains(&"alice") && names.contains(&"bob"), "{names:?}");
    }
}
