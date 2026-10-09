//! Live-propagation delivery: a registry of each identity's open MCP
//! sessions, and the fan-out that pushes a shared-memory change to them as a
//! server-initiated notification over the SSE stream.
//!
//! Server push requires the client to hold an open GET/SSE stream, which rmcp
//! only offers in stateful mode (see `http::server_config`). When a client
//! initializes, the server captures its [`Peer`] here, keyed by the JWT identity;
//! the change watcher then notifies every recipient's live peers.
//!
//! Delivery rides `notifications/message` (the logging channel), which MCP
//! deprecated in SEP-2577; rmcp 3 marks every use accordingly. The wire still
//! carries it and no replacement server-push channel has shipped, so this
//! module keeps the channel -- dropping live propagation to dodge a
//! deprecation would be backwards. The allow is module-wide because this
//! module IS the deprecated channel; it leaves with the migration.
#![allow(deprecated)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use rmcp::model::{LoggingLevel, LoggingMessageNotificationParam};
use rmcp::service::{Peer, RoleServer};
use serde_json::json;
use tokio::sync::Mutex;

use antumbra_sync::MemoryChange;

use crate::auth::Identity;

/// A process-unique id for one open session, so a closed transport can be pruned
/// without disturbing concurrently-registered sessions of the same identity.
type SessionId = u64;

/// Each identity's open sessions: its tagged server peers.
type Sessions = HashMap<Identity, Vec<(SessionId, Peer<RoleServer>)>>;

/// Maps a JWT identity to its currently-connected server peers (a client may
/// hold more than one session). Cloneable; all clones share one map and id
/// counter.
#[derive(Clone, Default)]
pub struct PeerRegistry {
    peers: Arc<Mutex<Sessions>>,
    next_id: Arc<AtomicU64>,
}

impl PeerRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a freshly-initialized session's peer for an identity.
    pub async fn register(&self, identity: Identity, peer: Peer<RoleServer>) {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.peers
            .lock()
            .await
            .entry(identity)
            .or_default()
            .push((id, peer));
    }

    /// Push `change` to every live session of each of its recipients, as a
    /// `notifications/message` carrying the structured change. Returns how many
    /// sessions were notified, and prunes peers whose transport has closed.
    ///
    /// Delivery snapshots the target peers under the lock, then releases it
    /// before awaiting the per-peer sends: holding the registry lock across a
    /// send would let one stuck SSE consumer stall every other session's
    /// registration and the next change's fan-out. Peers are cheap cloneable
    /// handles. Failed sends are pruned by session id afterwards, so a peer that
    /// a concurrent `register` appended in the meantime is left untouched.
    pub async fn notify(&self, change: &MemoryChange) -> usize {
        let param = change_notification(change);
        let targets: Vec<(Identity, u64, Peer<RoleServer>)> = {
            let map = self.peers.lock().await;
            let mut targets = Vec::new();
            for user in &change.recipients {
                let identity = Identity {
                    tenant: change.tenant.as_str().to_string(),
                    user: user.as_str().to_string(),
                };
                if let Some(peers) = map.get(&identity) {
                    for (id, peer) in peers {
                        targets.push((identity.clone(), *id, peer.clone()));
                    }
                }
            }
            targets
        };

        let mut delivered = 0;
        let mut dead: Vec<(Identity, u64)> = Vec::new();
        for (identity, id, peer) in targets {
            if peer.notify_logging_message(param.clone()).await.is_ok() {
                delivered += 1;
            } else {
                dead.push((identity, id));
            }
        }

        if !dead.is_empty() {
            let mut map = self.peers.lock().await;
            for (identity, id) in dead {
                if let Some(peers) = map.get_mut(&identity) {
                    peers.retain(|(pid, _)| *pid != id);
                    if peers.is_empty() {
                        map.remove(&identity);
                    }
                }
            }
        }
        delivered
    }
}

/// The wire form of a memory change: a structured `notifications/message` an
/// agent can act on. The `type` discriminator marks it as a penumbra event.
fn change_notification(change: &MemoryChange) -> LoggingMessageNotificationParam {
    // rmcp 3 made the param non-exhaustive; the constructor + builder is the
    // supported way to shape it.
    LoggingMessageNotificationParam::new(
        LoggingLevel::Info,
        json!({
            "type": "antumbra/memory_changed",
            "action": format!("{:?}", change.action).to_lowercase(),
            "tenant": change.tenant.as_str(),
            "compartment": change.compartment.as_str(),
            "memory": change.memory.as_str(),
        }),
    )
    .with_logger("antumbra/penumbra")
}

#[cfg(test)]
mod tests {
    use super::*;
    use antumbra_store::repo::sync::ChangeAction;

    #[test]
    fn change_notification_carries_the_routable_fields() {
        use antumbra_core::{CompartmentId, MemoryId, TenantId, UserId};
        let change = MemoryChange {
            action: ChangeAction::Create,
            tenant: TenantId::new("ws:t"),
            compartment: CompartmentId::new("comp:1"),
            memory: MemoryId::new("mem:9"),
            recipients: vec![UserId::new("alice"), UserId::new("bob")],
        };
        let param = change_notification(&change);
        assert_eq!(param.data["type"], "antumbra/memory_changed");
        assert_eq!(param.data["action"], "create");
        assert_eq!(param.data["compartment"], "comp:1");
        assert_eq!(param.data["memory"], "mem:9");
    }

    // An identity with no registered session is simply skipped (no panic, none
    // delivered) -- the common case for a change whose recipients are offline.
    #[tokio::test]
    async fn notify_with_no_sessions_delivers_none() {
        use antumbra_core::{CompartmentId, MemoryId, TenantId, UserId};
        let registry = PeerRegistry::new();
        let change = MemoryChange {
            action: ChangeAction::Update,
            tenant: TenantId::new("ws:t"),
            compartment: CompartmentId::new("comp:1"),
            memory: MemoryId::new("mem:9"),
            recipients: vec![UserId::new("ghost")],
        };
        assert_eq!(registry.notify(&change).await, 0);
    }
}
