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

use surql::query::builder::Query;
use surql::query::crud::query_records;
use surql::types::operators::{and_, eq};

use antumbra_core::{DeviceProfile, DeviceRole, Result, TenantId, UserId};

use crate::dto::parse_dt;
use crate::error::map;
use crate::store::Store;

const DEVICE: &str = crate::tables::DEVICE_PROFILE;

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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    seen_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    os: Option<String>,
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
            seen_at: d.seen_at.map(|at| at.to_rfc3339()),
            name: d.name.clone(),
            os: d.os.clone(),
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
            seen_at: self.seen_at.as_deref().map(parse_dt).transpose()?,
            name: self.name,
            os: self.os,
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
///
/// What a registration does not say stays as the row has it: the name its
/// user gave it, when it was last seen, and its operating system when this
/// registration names none. A registration is what the machine says of
/// itself, and none of those is its to take away.
pub async fn upsert(store: &Store, profile: &DeviceProfile) -> Result<()> {
    let mut row = DeviceRow::from_domain(profile);
    if let Some(had) = store
        .get_row::<DeviceRow>(DEVICE, record_key(profile))
        .await?
    {
        row.name = row.name.or(had.name);
        row.seen_at = row.seen_at.or(had.seen_at);
        row.os = row.os.or(had.os);
    }
    store.put(DEVICE, record_key(profile), &row).await
}

fn vars<const N: usize>(
    pairs: [(&str, serde_json::Value); N],
) -> std::collections::BTreeMap<String, serde_json::Value> {
    pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

/// Note that a call named one of a user's machines: its row's `seen_at`
/// moves to `now`, and nothing else of it. Whether there was a row to move:
/// a machine that never registered is not made one by being named.
pub async fn saw(
    store: &Store,
    tenant: &TenantId,
    user: &UserId,
    host: &str,
    now: chrono::DateTime<chrono::Utc>,
) -> Result<bool> {
    let moved: Vec<serde_json::Value> = store
        .query_rows(
            &format!(
                "UPDATE {DEVICE} SET seen_at = $now \
                 WHERE tenant_id = $tenant AND user = $user AND string::lowercase(host) = $host RETURN AFTER"
            ),
            vars([
                ("now", serde_json::json!(now.to_rfc3339())),
                ("tenant", serde_json::json!(tenant.as_str())),
                ("user", serde_json::json!(user.as_str())),
                ("host", serde_json::json!(host.trim().to_lowercase())),
            ]),
        )
        .await?;
    Ok(!moved.is_empty())
}

/// Note that the server on `host` is up: every row a server wrote for that
/// machine, whoever's fabric it is in, is seen at `now`. A row a session
/// wrote for a machine of that name is a client's and is left. How many.
pub async fn saw_server(
    store: &Store,
    host: &str,
    now: chrono::DateTime<chrono::Utc>,
) -> Result<usize> {
    let moved: Vec<serde_json::Value> = store
        .query_rows(
            &format!(
                "UPDATE {DEVICE} SET seen_at = $now \
                 WHERE string::lowercase(host) = $host AND backend != $client RETURN AFTER"
            ),
            vars([
                ("now", serde_json::json!(now.to_rfc3339())),
                ("host", serde_json::json!(host.trim().to_lowercase())),
                (
                    "client",
                    serde_json::json!(antumbra_core::device::CLIENT_BACKEND),
                ),
            ]),
        )
        .await?;
    Ok(moved.len())
}

/// Give one of a user's machines a name, or take it away (`None`). Whether
/// they have a machine by that host.
pub async fn rename(
    store: &Store,
    tenant: &TenantId,
    user: &UserId,
    host: &str,
    name: Option<&str>,
) -> Result<bool> {
    let set = match name {
        Some(_) => "SET name = $name",
        None => "UNSET name",
    };
    let moved: Vec<serde_json::Value> = store
        .query_rows(
            &format!(
                "UPDATE {DEVICE} {set} \
                 WHERE tenant_id = $tenant AND user = $user AND string::lowercase(host) = $host RETURN AFTER"
            ),
            vars([
                ("name", serde_json::json!(name)),
                ("tenant", serde_json::json!(tenant.as_str())),
                ("user", serde_json::json!(user.as_str())),
                ("host", serde_json::json!(host.trim().to_lowercase())),
            ]),
        )
        .await?;
    Ok(!moved.is_empty())
}

/// Every machine a user has registered in a tenant, the most recently seen first.
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
    found.sort_by_key(|d| std::cmp::Reverse(d.last_seen()));
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
    use crate::testing::mem_store;

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
        let s = mem_store().await?;
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
        let s = mem_store().await?;
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
        let s = mem_store().await?;
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

    /// A call that names a machine moves its `seen_at` and nothing else; a
    /// machine that never registered is not made one; a name is given and
    /// taken away; and registering again keeps the name, the operating
    /// system and when it was last seen.
    #[tokio::test]
    async fn a_device_is_seen_named_and_keeps_both_when_it_registers_again() -> Result<()> {
        let s = mem_store().await?;
        let (t, u) = (TenantId::new("ws:t"), UserId::new("user:a"));
        let registered = chrono::Utc::now() - chrono::Duration::hours(11);
        let mut laptop = DeviceProfile::client(t.clone(), u.clone(), "laptop", registered)
            .with_os(Some("macOS 15.6".into()));
        upsert(&s, &laptop).await?;
        let one = || async {
            Ok::<_, antumbra_core::AntumbraError>(list_for_user(&s, &t, &u).await?.remove(0))
        };
        assert_eq!(one().await?.last_seen(), registered);

        let now = registered + chrono::Duration::hours(11);
        assert!(saw(&s, &t, &u, " Laptop ", now).await?);
        let seen = one().await?;
        assert_eq!(
            (seen.updated_at, seen.seen_at, seen.last_seen()),
            (registered, Some(now), now)
        );
        assert_eq!(
            (seen.backend.as_str(), seen.os.as_deref()),
            ("client", Some("macOS 15.6"))
        );
        // Never registered, another user's, another workspace's: no row moves or is made.
        assert!(!saw(&s, &t, &u, "desktop", now).await?);
        assert!(!saw(&s, &t, &UserId::new("user:b"), "laptop", now).await?);
        assert!(!saw(&s, &TenantId::new("ws:other"), &u, "laptop", now).await?);
        assert_eq!(list_for_user(&s, &t, &u).await?.len(), 1);

        assert!(rename(&s, &t, &u, "laptop", Some("Work laptop")).await?);
        assert!(!rename(&s, &t, &u, "desktop", Some("x")).await?);
        assert_eq!(one().await?.name.as_deref(), Some("Work laptop"));

        // A new session registers it again and names no operating system.
        laptop.updated_at = now + chrono::Duration::minutes(5);
        laptop.os = None;
        upsert(&s, &laptop).await?;
        let again = one().await?;
        assert_eq!(again.name.as_deref(), Some("Work laptop"));
        assert_eq!(again.os.as_deref(), Some("macOS 15.6"));
        assert_eq!(again.seen_at, Some(now));
        assert_eq!(again.last_seen(), now + chrono::Duration::minutes(5));

        assert!(rename(&s, &t, &u, "laptop", None).await?);
        assert_eq!(one().await?.name, None);
        Ok(())
    }

    /// The server's own rows move when it notes itself up, and a client's
    /// row for a machine of the same name does not.
    #[tokio::test]
    async fn a_server_notes_itself_up_on_its_own_rows() -> Result<()> {
        let s = mem_store().await?;
        let then = chrono::Utc::now() - chrono::Duration::days(2);
        let t = TenantId::new("ws:t");
        upsert(
            &s,
            &DeviceProfile::new(
                t.clone(),
                UserId::new("user:a"),
                "Hub",
                "cuda",
                DeviceRole::Genesis,
                then,
            ),
        )
        .await?;
        upsert(
            &s,
            &DeviceProfile::new(
                t.clone(),
                UserId::new("user:b"),
                "Hub",
                "cuda",
                DeviceRole::Genesis,
                then,
            ),
        )
        .await?;
        upsert(
            &s,
            &DeviceProfile::client(t.clone(), UserId::new("user:c"), "hub", then),
        )
        .await?;
        let now = chrono::Utc::now();
        assert_eq!(saw_server(&s, "hub", now).await?, 2);
        for user in ["user:a", "user:b"] {
            assert_eq!(
                list_for_user(&s, &t, &UserId::new(user)).await?[0].last_seen(),
                now
            );
        }
        assert_eq!(
            list_for_user(&s, &t, &UserId::new("user:c")).await?[0].last_seen(),
            then
        );
        Ok(())
    }
}
