//! The device registry: a user's machines, as they describe themselves.
//!
//! A node registers itself here at startup and re-registers whenever what it
//! can do changes. The row is keyed per (tenant, user, host), so re-registering
//! updates rather than accumulating, and two people on one machine keep two
//! rows. Dispatch reads it to answer one question: of this user's machines,
//! which one can train.
//!
//! The engine enforces the write side (`DEVICE_PERMS`): a session writes only
//! its own user's rows, so no one can re-declare another member's laptop a
//! trainer and have genesis routed to it.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use surql::query::builder::Query;
use surql::query::crud::{query_records, upsert_record};
use surql::types::operators::{and_, eq};
use surql::types::RecordID;

use antumbra_core::{DeviceProfile, DeviceRole, Result, TenantId, UserId};

use crate::dto::parse_dt;
use crate::error::map;
use crate::store::Store;

const DEVICE: &str = "device_profile";

#[derive(Serialize, Deserialize)]
struct DeviceRow {
    key: String,
    tenant_id: String,
    user: String,
    host: String,
    backend: String,
    // Absent when the node did not know, which is not the same as zero, so the
    // field is left off the row rather than written as 0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    vram_mib: Option<u64>,
    // Kept as the wire string rather than the enum so a row written by a newer
    // node, in a role this version has never heard of, does not fail the whole
    // listing out from under the roles it does understand.
    role: String,
    updated_at: String,
}

impl DeviceRow {
    fn from_domain(d: &DeviceProfile) -> Self {
        DeviceRow {
            key: d.id.clone(),
            tenant_id: d.tenant_id.as_str().to_string(),
            user: d.user.as_str().to_string(),
            host: d.host.clone(),
            backend: d.backend.clone(),
            vram_mib: d.vram_mib,
            role: d.role.as_str().to_string(),
            updated_at: d.updated_at.to_rfc3339(),
        }
    }

    /// `None` when the row names a role this version cannot route to. Such a
    /// node is not an error, it is simply one this build has no dispatch for.
    fn into_domain(self) -> Result<Option<DeviceProfile>> {
        let Ok(role) = self.role.parse::<DeviceRole>() else {
            return Ok(None);
        };
        Ok(Some(DeviceProfile {
            id: self.key,
            tenant_id: TenantId::new(self.tenant_id),
            user: UserId::new(self.user),
            host: self.host,
            backend: self.backend,
            vram_mib: self.vram_mib,
            role,
            updated_at: parse_dt(&self.updated_at)?,
        }))
    }
}

/// The record key, which is the profile id with its table prefix removed (the
/// id carries it so the domain type reads as a record address everywhere else).
fn record_key(profile: &DeviceProfile) -> &str {
    profile
        .id
        .split_once(':')
        .map(|(_, key)| key)
        .unwrap_or(profile.id.as_str())
}

/// Register a node, or update what it can do. Idempotent per (tenant, user,
/// host): the same machine coming back keeps its row.
pub async fn upsert(store: &Store, profile: &DeviceProfile) -> Result<()> {
    let rid = RecordID::<()>::new(DEVICE, record_key(profile)).map_err(map)?;
    let data: Value = serde_json::to_value(DeviceRow::from_domain(profile))?;
    upsert_record(store.client(), &rid, data)
        .await
        .map_err(map)?;
    Ok(())
}

/// Every machine a user has registered in a tenant, newest registration first.
/// Rows naming an unknown role are left out (see [`DeviceRow::into_domain`]).
pub async fn list_for_user(
    store: &Store,
    tenant: &TenantId,
    user: &UserId,
) -> Result<Vec<DeviceProfile>> {
    let query = Query::new()
        .select(None)
        .from_table(DEVICE)
        .map_err(map)?
        .where_(and_(
            eq("tenant_id", tenant.as_str()),
            eq("user", user.as_str()),
        ));
    let rows: Vec<DeviceRow> = query_records(store.client(), &query).await.map_err(map)?;
    let mut found: Vec<DeviceProfile> = rows
        .into_iter()
        .map(DeviceRow::into_domain)
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect();
    found.sort_by_key(|d| std::cmp::Reverse(d.updated_at));
    Ok(found)
}

/// The machine of this user's that can train, or `None` if they have none and
/// genesis has to be refused or escalated. If they have run more than one, the
/// one that registered most recently wins: a fabric is a moving set of
/// machines, and the freshest registration is the one most likely to still be
/// up.
pub async fn genesis_for_user(
    store: &Store,
    tenant: &TenantId,
    user: &UserId,
) -> Result<Option<DeviceProfile>> {
    Ok(list_for_user(store, tenant, user)
        .await?
        .into_iter()
        .find(DeviceProfile::is_genesis))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::EMBED_DIM;

    async fn store() -> Result<Store> {
        Store::connect_memory(EMBED_DIM).await
    }

    fn profile(user: &str, host: &str, role: DeviceRole) -> DeviceProfile {
        DeviceProfile::new(
            TenantId::new("ws:t"),
            UserId::new(user),
            host,
            "cpu",
            role,
            chrono::Utc::now(),
        )
    }

    #[tokio::test]
    async fn a_node_re_registering_updates_its_row_rather_than_adding_one() -> Result<()> {
        let s = store().await?;
        let tenant = TenantId::new("ws:t");
        let user = UserId::new("user:a");

        upsert(&s, &profile("user:a", "laptop", DeviceRole::Memory)).await?;
        let promoted = profile("user:a", "laptop", DeviceRole::Genesis).with_vram(24_576);
        upsert(&s, &promoted).await?;

        let listed = list_for_user(&s, &tenant, &user).await?;
        assert_eq!(listed.len(), 1, "one machine, one row");
        assert_eq!(listed[0].role, DeviceRole::Genesis);
        assert_eq!(listed[0].vram_mib, Some(24_576));
        Ok(())
    }

    #[tokio::test]
    async fn the_genesis_node_is_the_one_that_can_train_and_only_that_users() -> Result<()> {
        let s = store().await?;
        let tenant = TenantId::new("ws:t");
        let alice = UserId::new("user:alice");
        let bob = UserId::new("user:bob");

        upsert(&s, &profile("user:alice", "laptop", DeviceRole::Memory)).await?;
        upsert(
            &s,
            &profile("user:alice", "workstation", DeviceRole::Genesis),
        )
        .await?;
        // Bob's trainer is not a machine alice's work may be dispatched to.
        upsert(&s, &profile("user:bob", "bobs-rig", DeviceRole::Genesis)).await?;

        let found = genesis_for_user(&s, &tenant, &alice).await?;
        assert_eq!(
            found.map(|d| d.host),
            Some("workstation".to_string()),
            "dispatch picks the user's own trainer"
        );
        assert_eq!(list_for_user(&s, &tenant, &alice).await?.len(), 2);
        assert_eq!(
            genesis_for_user(&s, &tenant, &bob).await?.map(|d| d.host),
            Some("bobs-rig".to_string())
        );

        // A user with only memory nodes has nowhere to train.
        upsert(&s, &profile("user:cass", "chromebook", DeviceRole::Memory)).await?;
        assert!(genesis_for_user(&s, &tenant, &UserId::new("user:cass"))
            .await?
            .is_none());
        Ok(())
    }

    #[tokio::test]
    async fn a_role_this_build_cannot_route_to_is_skipped_not_fatal() -> Result<()> {
        let s = store().await?;
        let tenant = TenantId::new("ws:t");
        let user = UserId::new("user:a");

        upsert(&s, &profile("user:a", "laptop", DeviceRole::Memory)).await?;
        // What a newer node would write: a role this version has never heard of.
        crate::repo::sync::put_row(
            &s,
            &serde_json::json!({
                "id": "device_profile:from_the_future",
                "key": "device_profile:from_the_future",
                "tenant_id": "ws:t",
                "user": "user:a",
                "host": "tomorrow",
                "backend": "tpu",
                "role": "distiller",
                "updated_at": chrono::Utc::now().to_rfc3339(),
            }),
        )
        .await?;

        let listed = list_for_user(&s, &tenant, &user).await?;
        assert_eq!(
            listed.iter().map(|d| d.host.as_str()).collect::<Vec<_>>(),
            vec!["laptop"],
            "the unknown node is not routable, and does not take the known one down with it"
        );
        Ok(())
    }
}
