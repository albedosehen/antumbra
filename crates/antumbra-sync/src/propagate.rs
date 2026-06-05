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
            match resolve(&store, &event).await {
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

/// Parse a memory change row and resolve its audience, or `None` if it is not a
/// routable shared-compartment change.
async fn resolve(store: &Store, event: &ChangeEvent) -> Option<MemoryChange> {
    let row = &event.row;
    let tenant = TenantId::new(row.get("tenant_id")?.as_str()?);
    let compartment = CompartmentId::new(row.get("compartment")?.as_str()?);
    let memory = MemoryId::new(row.get("key")?.as_str()?);
    let recipients = audience(store, &tenant, &compartment).await.ok()?;
    Some(MemoryChange {
        action: event.action,
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
}
