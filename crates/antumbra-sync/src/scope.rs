//! What a user's nodes carry (each user has their own fabric), which is
//! narrower than what the user may read.
//!
//! The ACL and this are different statements and must not be collapsed into
//! one. `MEMORY_SELECT_RULE` says what a session may **see**, live, through the
//! engine. This says what that user's other machines get a **copy** of. A grant
//! is a live read, not a license to take another member's private compartment
//! home on a laptop.
//!
//! The narrower policy is also the implementable one: on five of the six
//! replicated tables the engine's write rule is narrower than its read rule, so
//! a collector that replicated everything it could read would have its writes
//! refused silently. Own-plus-shared-pool is exactly what the session can write
//! back, which is what makes read scope and write scope the same set.

use std::collections::HashSet;

use serde_json::Value;

use antumbra_core::Result;
use antumbra_store::repo::compartment;
use antumbra_store::Store;

use crate::config::Fabric;

/// One user's replication scope, resolved once per pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scope {
    user: String,
    /// The compartments this user owns. Read once: a compartment created
    /// mid-pass is picked up on the next one, which is the same freshness every
    /// other part of a cadence-based collector has.
    owned: HashSet<String>,
}

impl Scope {
    /// Resolve `fabric`'s scope from `store`, which must already be signed in as
    /// that user (the owned set is read through the engine).
    pub async fn resolve(store: &Store, fabric: &Fabric) -> Result<Self> {
        let owned = compartment::list_owned(store, &fabric.tenant, &fabric.user)
            .await?
            .into_iter()
            .map(|c| c.id.as_str().to_string())
            .collect();
        Ok(Scope {
            user: fabric.user.as_str().to_string(),
            owned,
        })
    }

    fn field<'a>(row: &'a Value, name: &str) -> Option<&'a str> {
        row.get(name)?.as_str()
    }

    /// The row's compartment, or `None` for the shared tenant pool. Absent and
    /// null mean the same thing here: nothing filed it anywhere.
    fn compartment_of(row: &Value) -> Option<&str> {
        Self::field(row, "compartment")
    }

    pub fn owns_compartment(&self, row: &Value) -> bool {
        Self::field(row, "owner").is_some_and(|owner| owner == self.user)
    }

    pub fn is_own_user(&self, row: &Value) -> bool {
        Self::field(row, "user").is_some_and(|user| user == self.user)
    }

    /// A grant belongs to the fabric of whoever owns the compartment it is on,
    /// not the grantee's: the owner is the one whose machines need it to resolve
    /// their own ACL, and the grantee reads through the engine rather than from
    /// a copy.
    pub fn grants_own_compartment(&self, row: &Value) -> bool {
        Self::field(row, "compartment").is_some_and(|c| self.owned.contains(c))
    }

    /// Own compartments plus the shared pool.
    pub fn holds_memory(&self, row: &Value) -> bool {
        match Self::compartment_of(row) {
            None => true,
            Some(c) => self.owned.contains(c),
        }
    }
}

/// How a table's rows are narrowed when a collector is scoped to one user.
///
/// Deliberately not `PartialEq`: one variant carries a function pointer, and
/// comparing those compares addresses, which say nothing useful.
#[derive(Debug, Clone, Copy)]
pub enum Replicate {
    /// Rows this user's nodes carry, decided from the row itself.
    Owned(fn(&Scope, &Value) -> bool),
    /// The row does not carry enough to decide, so the engine decides at write
    /// time. `memory_edge` is the case: it carries `from_id` and `to_id` and no
    /// compartment, so whether it belongs here is a property of memories it
    /// only references. A refusal on such a table is the system working, not a
    /// fault, and is counted apart from one on an `Owned` table -- where a
    /// refusal means the policy and the ACL disagree, which is a bug.
    EngineDecides,
}

impl Replicate {
    /// Whether to offer this row to the other store at all.
    pub fn admits(self, scope: &Scope, row: &Value) -> bool {
        match self {
            Replicate::Owned(predicate) => predicate(scope, row),
            Replicate::EngineDecides => true,
        }
    }

    /// Whether a refusal on this table is expected.
    pub fn refusal_is_expected(self) -> bool {
        matches!(self, Replicate::EngineDecides)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antumbra_core::{Compartment, TenantId, UserId};
    use antumbra_store::repo::principal;
    use antumbra_store::EMBED_DIM;
    use chrono::Utc;
    use serde_json::json;

    async fn lilys_scope() -> Result<Scope> {
        let store = Store::connect_memory(EMBED_DIM).await?;
        let tenant = TenantId::new("ws:org");
        let lily = UserId::new("user:lily");
        let oslo = UserId::new("user:oslo");
        principal::provision(&store, &tenant, &lily).await?;
        principal::provision(&store, &tenant, &oslo).await?;
        for (id, owner) in [("comp:hers", &lily), ("comp:his", &oslo)] {
            compartment::create(
                &store,
                &Compartment::new(id, tenant.clone(), owner.clone(), id, Utc::now()),
            )
            .await?;
        }
        Scope::resolve(&store, &Fabric::new("ws:org", "user:lily")).await
    }

    #[tokio::test]
    async fn a_users_nodes_carry_their_own_compartments_and_the_shared_pool() -> Result<()> {
        let scope = lilys_scope().await?;

        // Memory: hers, and the pool, and not his.
        assert!(scope.holds_memory(&json!({ "compartment": "comp:hers" })));
        assert!(!scope.holds_memory(&json!({ "compartment": "comp:his" })));
        // The shared pool, however the engine spelled "nothing filed it".
        assert!(
            scope.holds_memory(&json!({ "content": "pooled" })),
            "absent"
        );
        assert!(scope.holds_memory(&json!({ "compartment": null })), "null");

        // Compartments and grants follow the owner, not the grantee: the owner's
        // machines need them to resolve their own ACL, and a grantee reads
        // through the engine rather than from a copy taken home.
        assert!(scope.owns_compartment(&json!({ "owner": "user:lily" })));
        assert!(!scope.owns_compartment(&json!({ "owner": "user:oslo" })));
        assert!(scope.grants_own_compartment(&json!({ "compartment": "comp:hers" })));
        assert!(!scope.grants_own_compartment(&json!({ "compartment": "comp:his" })));

        // The fabric tables are keyed on the user directly.
        assert!(scope.is_own_user(&json!({ "user": "user:lily" })));
        assert!(!scope.is_own_user(&json!({ "user": "user:oslo" })));
        Ok(())
    }

    /// A `reference` grant makes another member's compartment readable. It does
    /// not make it something this user's laptop takes a copy of -- which is both
    /// the privacy answer and, not by coincidence, exactly what the session can
    /// write back.
    #[tokio::test]
    async fn a_grant_is_a_live_read_not_a_license_to_copy() -> Result<()> {
        let scope = lilys_scope().await?;
        assert!(
            !scope.holds_memory(&json!({ "compartment": "comp:his" })),
            "oslo granting lily a read does not put his memories on her machines"
        );
        Ok(())
    }

    /// The row that cannot answer for itself. An edge carries `from_id` and
    /// `to_id` and no compartment, so the policy cannot judge it and the engine
    /// must -- which is why a refusal there is expected rather than a fault.
    #[test]
    fn an_edge_is_left_to_the_engine() {
        assert!(Replicate::EngineDecides.refusal_is_expected());
        assert!(!Replicate::Owned(|_, _| true).refusal_is_expected());
    }
}
