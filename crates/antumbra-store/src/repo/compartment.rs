//! Compartment + grant repositories (the latent-spaces and their sharing).
//!
//! Tenant-isolated like the rest of the Penumbra. The fine-grained read ACL on
//! *memory* (own + granted compartments) is enforced by the engine via the
//! `memory` table's permission rule, which reads these two tables; here we just
//! provide tenant-scoped CRUD. No raw SurrealQL; surql-rs builders only.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use surql::query::builder::Query;
use surql::query::crud::{delete_records, query_records, upsert_record};
use surql::types::operators::{and_, eq};
use surql::types::RecordID;

use antumbra_core::{
    Capability, Compartment, CompartmentId, Grant, Origin, Result, TenantId, UserId,
};

use crate::dto::parse_dt;
use crate::error::map;
use crate::store::Store;

const COMPARTMENT: &str = "compartment";
const GRANT: &str = "grant";

#[derive(Serialize, Deserialize)]
struct CompartmentRow {
    key: String,
    tenant_id: String,
    owner: String,
    name: String,
    origin: Origin,
    created_at: String,
    #[serde(default)]
    updated_at: Option<String>,
    // The deletion tombstone. Absent (NONE) for a live compartment so the engine
    // ACL's owner subquery (`deleted_at IS NONE`) admits it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    deleted_at: Option<String>,
}

impl CompartmentRow {
    fn from_domain(c: &Compartment) -> Self {
        CompartmentRow {
            key: c.id.as_str().to_string(),
            tenant_id: c.tenant.as_str().to_string(),
            owner: c.owner.as_str().to_string(),
            name: c.name.clone(),
            origin: c.origin,
            created_at: c.created_at.to_rfc3339(),
            updated_at: Some(c.updated_at.to_rfc3339()),
            deleted_at: c.deleted_at.map(|t| t.to_rfc3339()),
        }
    }

    fn into_domain(self) -> Result<Compartment> {
        let created_at = parse_dt(&self.created_at)?;
        let updated_at = self
            .updated_at
            .as_deref()
            .map(parse_dt)
            .transpose()?
            .unwrap_or(created_at);
        Ok(Compartment {
            id: CompartmentId::new(self.key),
            tenant: TenantId::new(self.tenant_id),
            owner: UserId::new(self.owner),
            name: self.name,
            origin: self.origin,
            created_at,
            updated_at,
            deleted_at: self.deleted_at.as_deref().map(parse_dt).transpose()?,
        })
    }
}

fn sanitize(s: &str) -> String {
    s.replace([':', '/', '\\', '|', ' '], "_")
}

/// Create or update a compartment (addressed by its id).
pub async fn create(store: &Store, compartment: &Compartment) -> Result<()> {
    let rid = RecordID::<()>::new(COMPARTMENT, sanitize(compartment.id.as_str()).as_str())
        .map_err(map)?;
    let data: Value = serde_json::to_value(CompartmentRow::from_domain(compartment))?;
    upsert_record(store.client(), &rid, data)
        .await
        .map_err(map)?;
    Ok(())
}

/// The compartments a user owns in a tenant.
pub async fn list_owned(
    store: &Store,
    tenant: &TenantId,
    owner: &UserId,
) -> Result<Vec<Compartment>> {
    let query = Query::new()
        .select(None)
        .from_table(COMPARTMENT)
        .map_err(map)?
        .where_(and_(
            eq("tenant_id", tenant.as_str()),
            eq("owner", owner.as_str()),
        ));
    let rows: Vec<CompartmentRow> = query_records(store.client(), &query).await.map_err(map)?;
    rows.into_iter()
        .filter(|r| r.deleted_at.is_none())
        .map(CompartmentRow::into_domain)
        .collect()
}

/// Every live compartment named `name`, in every workspace: for a keeper that
/// runs in owner mode over every user's own compartment of one kind.
pub async fn list_named(store: &Store, name: &str) -> Result<Vec<Compartment>> {
    let query = Query::new()
        .select(None)
        .from_table(COMPARTMENT)
        .map_err(map)?
        .where_(eq("name", name));
    let rows: Vec<CompartmentRow> = query_records(store.client(), &query).await.map_err(map)?;
    rows.into_iter()
        .filter(|r| r.deleted_at.is_none())
        .map(CompartmentRow::into_domain)
        .collect()
}

/// Fetch one **live** compartment by id within a tenant (e.g. to read its `owner`
/// for audience resolution). `None` if absent, deleted, or owned by another
/// tenant.
pub async fn get(
    store: &Store,
    tenant: &TenantId,
    id: &CompartmentId,
) -> Result<Option<Compartment>> {
    let query = Query::new()
        .select(None)
        .from_table(COMPARTMENT)
        .map_err(map)?
        .where_(and_(
            eq("tenant_id", tenant.as_str()),
            eq("key", id.as_str()),
        ));
    let rows: Vec<CompartmentRow> = query_records(store.client(), &query).await.map_err(map)?;
    rows.into_iter()
        .find(|r| r.deleted_at.is_none())
        .map(CompartmentRow::into_domain)
        .transpose()
}

/// Delete a compartment as a **tombstone** (owner-scoped), so the deletion
/// propagates across the fleet instead of resurrecting from a replica that still
/// has the live row. Its memories become invisible at once where the engine ACL
/// runs (the owner subquery excludes `deleted_at` rows). A no-op if the
/// compartment is absent or already deleted. `now` stamps the deletion. Use
/// [`purge_compartments`] to hard-remove tombstones past a grace window.
pub async fn delete(
    store: &Store,
    tenant: &TenantId,
    id: &CompartmentId,
    now: chrono::DateTime<chrono::Utc>,
) -> Result<()> {
    if let Some(mut c) = get(store, tenant, id).await? {
        c.soft_delete(now);
        create(store, &c).await?;
    }
    Ok(())
}

/// Hard-remove compartment tombstones deleted before `older_than` (the grace
/// window), so they do not accumulate. Run wider than the sync interval so every
/// replica saw the deletion first (resurrection-safe GC). Returns the count.
pub async fn purge_compartments(
    store: &Store,
    older_than: chrono::DateTime<chrono::Utc>,
) -> Result<usize> {
    let query = Query::new()
        .select(None)
        .from_table(COMPARTMENT)
        .map_err(map)?;
    let rows: Vec<CompartmentRow> = query_records(store.client(), &query).await.map_err(map)?;
    let mut purged = 0;
    for row in rows {
        let Some(ts) = row.deleted_at.as_deref() else {
            continue;
        };
        if parse_dt(ts).map(|t| t < older_than).unwrap_or(false) {
            let condition = and_(
                eq("key", row.key.as_str()),
                eq("tenant_id", row.tenant_id.as_str()),
            );
            delete_records(store.client(), COMPARTMENT, Some(&condition))
                .await
                .map_err(map)?;
            purged += 1;
        }
    }
    Ok(purged)
}

#[derive(Serialize, Deserialize)]
struct GrantRow {
    tenant_id: String,
    compartment: String,
    grantee: String,
    capability: Capability,
    granted_by: String,
    created_at: String,
    // Defaulted for rows written before grants carried a version; falls back to
    // created_at on read.
    #[serde(default)]
    updated_at: Option<String>,
    // The revocation tombstone. Absent (NONE) for a live grant so the engine ACL
    // subquery (`deleted_at IS NONE`) admits it; an RFC3339 timestamp once revoked.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    deleted_at: Option<String>,
}

impl GrantRow {
    fn from_domain(g: &Grant) -> Self {
        GrantRow {
            tenant_id: g.tenant.as_str().to_string(),
            compartment: g.compartment.as_str().to_string(),
            grantee: g.grantee.as_str().to_string(),
            capability: g.capability,
            granted_by: g.granted_by.as_str().to_string(),
            created_at: g.created_at.to_rfc3339(),
            updated_at: Some(g.updated_at.to_rfc3339()),
            deleted_at: g.deleted_at.map(|t| t.to_rfc3339()),
        }
    }

    fn into_domain(self) -> Result<Grant> {
        let created_at = parse_dt(&self.created_at)?;
        let updated_at = self
            .updated_at
            .as_deref()
            .map(parse_dt)
            .transpose()?
            .unwrap_or(created_at);
        Ok(Grant {
            tenant: TenantId::new(self.tenant_id),
            compartment: CompartmentId::new(self.compartment),
            grantee: UserId::new(self.grantee),
            capability: self.capability,
            granted_by: UserId::new(self.granted_by),
            created_at,
            updated_at,
            deleted_at: self.deleted_at.as_deref().map(parse_dt).transpose()?,
        })
    }
}

fn grant_key(g: &Grant) -> String {
    sanitize(&format!(
        "{}|{}|{}",
        g.tenant.as_str(),
        g.compartment.as_str(),
        g.grantee.as_str()
    ))
}

/// Grant (or update) a capability on a compartment to a user. Idempotent per
/// (tenant, compartment, grantee).
pub async fn grant(store: &Store, grant: &Grant) -> Result<()> {
    let rid = RecordID::<()>::new(GRANT, grant_key(grant).as_str()).map_err(map)?;
    let data: Value = serde_json::to_value(GrantRow::from_domain(grant))?;
    upsert_record(store.client(), &rid, data)
        .await
        .map_err(map)?;
    Ok(())
}

/// Revoke a user's grant on a compartment as a **tombstone** (not a hard delete),
/// so the revocation propagates across the fleet instead of leaving a stale grant
/// that keeps the grantee in (the security gap). Access ends at once where the
/// engine ACL runs; its grant subquery already excludes `deleted_at` rows. A
/// no-op if there is no live grant. `now` stamps the revocation.
pub async fn revoke(
    store: &Store,
    tenant: &TenantId,
    compartment: &CompartmentId,
    grantee: &UserId,
    now: chrono::DateTime<chrono::Utc>,
) -> Result<()> {
    if let Some(mut g) = live_grant(store, tenant, compartment, grantee).await? {
        g.revoke(now);
        grant(store, &g).await?;
    }
    Ok(())
}

/// The single **live** grant for (tenant, compartment, grantee), or `None` if
/// absent or already revoked.
async fn live_grant(
    store: &Store,
    tenant: &TenantId,
    compartment: &CompartmentId,
    grantee: &UserId,
) -> Result<Option<Grant>> {
    let query = Query::new()
        .select(None)
        .from_table(GRANT)
        .map_err(map)?
        .where_(and_(
            and_(
                eq("tenant_id", tenant.as_str()),
                eq("compartment", compartment.as_str()),
            ),
            eq("grantee", grantee.as_str()),
        ));
    let rows: Vec<GrantRow> = query_records(store.client(), &query).await.map_err(map)?;
    rows.into_iter()
        .find(|r| r.deleted_at.is_none())
        .map(GrantRow::into_domain)
        .transpose()
}

/// Hard-remove revoked-grant tombstones older than `older_than` (the grace
/// window), so they do not accumulate. Run wider than the sync interval so every
/// replica saw the revocation first (resurrection-safe GC). Returns the count.
pub async fn purge_grants(
    store: &Store,
    older_than: chrono::DateTime<chrono::Utc>,
) -> Result<usize> {
    let query = Query::new().select(None).from_table(GRANT).map_err(map)?;
    let rows: Vec<GrantRow> = query_records(store.client(), &query).await.map_err(map)?;
    let mut purged = 0;
    for row in rows {
        let Some(ts) = row.deleted_at.as_deref() else {
            continue;
        };
        if parse_dt(ts).map(|t| t < older_than).unwrap_or(false) {
            let condition = and_(
                and_(
                    eq("tenant_id", row.tenant_id.as_str()),
                    eq("compartment", row.compartment.as_str()),
                ),
                eq("grantee", row.grantee.as_str()),
            );
            delete_records(store.client(), GRANT, Some(&condition))
                .await
                .map_err(map)?;
            purged += 1;
        }
    }
    Ok(purged)
}

/// The (live) grants a user has received in a tenant. Revoked grants are hidden.
pub async fn list_for_grantee(
    store: &Store,
    tenant: &TenantId,
    grantee: &UserId,
) -> Result<Vec<Grant>> {
    let query = Query::new()
        .select(None)
        .from_table(GRANT)
        .map_err(map)?
        .where_(and_(
            eq("tenant_id", tenant.as_str()),
            eq("grantee", grantee.as_str()),
        ));
    let rows: Vec<GrantRow> = query_records(store.client(), &query).await.map_err(map)?;
    rows.into_iter()
        .filter(|r| r.deleted_at.is_none())
        .map(GrantRow::into_domain)
        .collect()
}

/// Whether `user` may write into `compartment`: they own it, or hold a live
/// `link` grant on it (a `reference` grant reads and no more). The app-side
/// mirror of the engine's write rule, for a caller that has to know *before* it
/// acts. An ingest archives the original before it writes a chunk, and under a
/// record session the engine's refusal of that chunk is silent, so by the time
/// anyone noticed, the original would already sit in the archive under a
/// compartment its author could not write to.
pub async fn can_write(
    store: &Store,
    tenant: &TenantId,
    user: &UserId,
    compartment: &CompartmentId,
) -> Result<bool> {
    // Absent or deleted: nobody writes there.
    let Some(found) = get(store, tenant, compartment).await? else {
        return Ok(false);
    };
    if found.owner == *user {
        return Ok(true);
    }
    Ok(list_for_grantee(store, tenant, user)
        .await?
        .iter()
        .any(|g| g.compartment == *compartment && g.capability.allows_link()))
}

/// Every **live** grant on a compartment (its grantees) -- the other half, with
/// the owner, of who may see the compartment's memories. Used to fan a live
/// change out to its audience (R-2). Revoked grants are excluded.
pub async fn list_grants(
    store: &Store,
    tenant: &TenantId,
    compartment: &CompartmentId,
) -> Result<Vec<Grant>> {
    let query = Query::new()
        .select(None)
        .from_table(GRANT)
        .map_err(map)?
        .where_(and_(
            eq("tenant_id", tenant.as_str()),
            eq("compartment", compartment.as_str()),
        ));
    let rows: Vec<GrantRow> = query_records(store.client(), &query).await.map_err(map)?;
    rows.into_iter()
        .filter(|r| r.deleted_at.is_none())
        .map(GrantRow::into_domain)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::EMBED_DIM;
    use antumbra_core::Capability;

    async fn store() -> Store {
        Store::connect_memory(EMBED_DIM).await.unwrap()
    }

    #[tokio::test]
    async fn compartment_create_get_list_delete() {
        let s = store().await;
        let t = TenantId::new("t");
        let alice = UserId::new("alice");
        let now = chrono::Utc::now();
        let c = Compartment::new("comp:1", t.clone(), alice.clone(), "c", now);

        create(&s, &c).await.unwrap();
        assert_eq!(get(&s, &t, &c.id).await.unwrap().unwrap().owner, alice);
        assert_eq!(list_owned(&s, &t, &alice).await.unwrap().len(), 1);
        // A different tenant cannot see it.
        assert!(get(&s, &TenantId::new("other"), &c.id)
            .await
            .unwrap()
            .is_none());

        // Delete is a tombstone: hidden from get/list_owned, but the row remains
        // (so the deletion can propagate), then a past-grace purge removes it.
        delete(&s, &t, &c.id, now).await.unwrap();
        assert!(get(&s, &t, &c.id).await.unwrap().is_none());
        assert!(list_owned(&s, &t, &alice).await.unwrap().is_empty());
        assert_eq!(
            crate::repo::sync::list_rows(&s, "compartment")
                .await
                .unwrap()
                .len(),
            1,
            "tombstone row retained for propagation"
        );
        delete(&s, &t, &c.id, now).await.unwrap(); // re-delete is a no-op
        assert_eq!(
            purge_compartments(&s, now - chrono::Duration::days(1))
                .await
                .unwrap(),
            0
        );
        assert_eq!(
            purge_compartments(&s, now + chrono::Duration::seconds(1))
                .await
                .unwrap(),
            1
        );
    }

    #[tokio::test]
    async fn grant_revoke_is_a_tombstone_then_purges() {
        let s = store().await;
        let t = TenantId::new("t");
        let comp = CompartmentId::new("comp:1");
        let bob = UserId::new("bob");
        let now = chrono::Utc::now();
        let g = Grant::new(
            t.clone(),
            comp.clone(),
            bob.clone(),
            Capability::Reference,
            UserId::new("alice"),
            now,
        );

        grant(&s, &g).await.unwrap();
        assert_eq!(list_grants(&s, &t, &comp).await.unwrap().len(), 1);
        assert_eq!(list_for_grantee(&s, &t, &bob).await.unwrap().len(), 1);

        // Revoke: the live grant is hidden, but the tombstone row remains.
        revoke(&s, &t, &comp, &bob, now).await.unwrap();
        assert!(
            list_grants(&s, &t, &comp).await.unwrap().is_empty(),
            "revoked grant hidden"
        );
        assert!(list_for_grantee(&s, &t, &bob).await.unwrap().is_empty());
        // Re-revoke is a no-op (no live grant to revoke).
        revoke(&s, &t, &comp, &bob, now).await.unwrap();

        // Re-granting un-revokes (a fresh, newer grant overwrites the tombstone).
        let g2 = Grant::new(
            t.clone(),
            comp.clone(),
            bob.clone(),
            Capability::Link,
            UserId::new("alice"),
            now,
        );
        grant(&s, &g2).await.unwrap();
        assert_eq!(
            list_grants(&s, &t, &comp).await.unwrap()[0].capability,
            Capability::Link
        );

        // Revoke again, then purge past the grace window removes the tombstone.
        revoke(&s, &t, &comp, &bob, now).await.unwrap();
        assert_eq!(
            purge_grants(&s, now - chrono::Duration::days(1))
                .await
                .unwrap(),
            0,
            "within grace"
        );
        assert_eq!(
            purge_grants(&s, now + chrono::Duration::seconds(1))
                .await
                .unwrap(),
            1,
            "past grace"
        );
    }
}
