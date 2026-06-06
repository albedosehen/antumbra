//! Live-propagation delivery (R-2): a registry of each identity's open MCP
//! sessions, and the fan-out that pushes a shared-memory change to them as a
//! server-initiated notification over the SSE stream.
//!
//! Server push requires the client to hold an open GET/SSE stream, which rmcp
//! only offers in stateful mode (see `http::server_config`). When a client
//! initializes, the server captures its [`Peer`] here, keyed by the JWT identity;
//! the change watcher then notifies every recipient's live peers.

use std::collections::HashMap;
use std::sync::Arc;

use rmcp::model::{LoggingLevel, LoggingMessageNotificationParam};
use rmcp::service::{Peer, RoleServer};
use serde_json::json;
use tokio::sync::Mutex;

use antumbra_sync::MemoryChange;

use crate::auth::Identity;

/// Maps a JWT identity to its currently-connected server peers (a client may
/// hold more than one session). Cloneable; all clones share one map.
#[derive(Clone, Default)]
pub struct PeerRegistry {
    peers: Arc<Mutex<HashMap<Identity, Vec<Peer<RoleServer>>>>>,
}

impl PeerRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a freshly-initialized session's peer for an identity.
    pub async fn register(&self, identity: Identity, peer: Peer<RoleServer>) {
        self.peers.lock().await.entry(identity).or_default().push(peer);
    }

    /// Push `change` to every live session of each of its recipients, as a
    /// `notifications/message` carrying the structured change. Returns how many
    /// sessions were notified, and prunes peers whose transport has closed.
    pub async fn notify(&self, change: &MemoryChange) -> usize {
        let param = change_notification(change);
        let mut delivered = 0;
        let mut map = self.peers.lock().await;
        for user in &change.recipients {
            let identity = Identity {
                tenant: change.tenant.as_str().to_string(),
                user: user.as_str().to_string(),
            };
            let Some(peers) = map.get_mut(&identity) else {
                continue;
            };
            let mut live = Vec::with_capacity(peers.len());
            for peer in std::mem::take(peers) {
                if peer.notify_logging_message(param.clone()).await.is_ok() {
                    delivered += 1;
                    live.push(peer); // keep only peers still connected
                }
            }
            *peers = live;
        }
        delivered
    }
}

/// The wire form of a memory change: a structured `notifications/message` an
/// agent can act on. The `type` discriminator marks it as a penumbra event.
fn change_notification(change: &MemoryChange) -> LoggingMessageNotificationParam {
    LoggingMessageNotificationParam {
        level: LoggingLevel::Info,
        logger: Some("antumbra/penumbra".to_string()),
        data: json!({
            "type": "antumbra/memory_changed",
            "action": format!("{:?}", change.action).to_lowercase(),
            "tenant": change.tenant.as_str(),
            "compartment": change.compartment.as_str(),
            "memory": change.memory.as_str(),
        }),
    }
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
