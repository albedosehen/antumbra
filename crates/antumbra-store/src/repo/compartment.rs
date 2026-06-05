//! Compartment + grant repositories (the latent-spaces and their sharing).
//!
//! Tenant-isolated like the rest of the Penumbra. The fine-grained read ACL on
//! *memory* (own + granted compartments) is enforced by the engine via the
//! `memory` table's permission rule, which reads these two tables; here we just
//! provide tenant-scoped CRUD. No raw SurrealQL — surql-rs builders only.

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
        }
    }

    fn into_domain(self) -> Result<Compartment> {
        Ok(Compartment {
            id: CompartmentId::new(self.key),
            tenant: TenantId::new(self.tenant_id),
            owner: UserId::new(self.owner),
            name: self.name,
            origin: self.origin,
            created_at: parse_dt(&self.created_at)?,
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
    upsert_record(store.client(), &rid, data).await.map_err(map)?;
    Ok(())
}

/// The compartments a user owns in a tenant.
pub async fn list_owned(store: &Store, tenant: &TenantId, owner: &UserId) -> Result<Vec<Compartment>> {
    let query = Query::new()
        .select(None)
        .from_table(COMPARTMENT)
        .map_err(map)?
        .where_(and_(eq("tenant_id", tenant.as_str()), eq("owner", owner.as_str())));
    let rows: Vec<CompartmentRow> = query_records(store.client(), &query).await.map_err(map)?;
    rows.into_iter().map(CompartmentRow::into_domain).collect()
}

/// Fetch one compartment by id within a tenant (e.g. to read its `owner` for
/// audience resolution). `None` if absent or owned by another tenant.
pub async fn get(
    store: &Store,
    tenant: &TenantId,
    id: &CompartmentId,
) -> Result<Option<Compartment>> {
    let query = Query::new()
        .select(None)
        .from_table(COMPARTMENT)
        .map_err(map)?
        .where_(and_(eq("tenant_id", tenant.as_str()), eq("key", id.as_str())));
    let rows: Vec<CompartmentRow> = query_records(store.client(), &query).await.map_err(map)?;
    rows.into_iter().next().map(CompartmentRow::into_domain).transpose()
}

/// Delete a compartment (owner-scoped; the engine bars non-owners on memory).
pub async fn delete(store: &Store, tenant: &TenantId, id: &CompartmentId) -> Result<()> {
    let condition = and_(eq("key", id.as_str()), eq("tenant_id", tenant.as_str()));
    delete_records(store.client(), COMPARTMENT, Some(&condition))
        .await
        .map_err(map)?;
    Ok(())
}

#[derive(Serialize, Deserialize)]
struct GrantRow {
    tenant_id: String,
    compartment: String,
    grantee: String,
    capability: Capability,
    granted_by: String,
    created_at: String,
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
        }
    }

    fn into_domain(self) -> Result<Grant> {
        Ok(Grant {
            tenant: TenantId::new(self.tenant_id),
            compartment: CompartmentId::new(self.compartment),
            grantee: UserId::new(self.grantee),
            capability: self.capability,
            granted_by: UserId::new(self.granted_by),
            created_at: parse_dt(&self.created_at)?,
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
    upsert_record(store.client(), &rid, data).await.map_err(map)?;
    Ok(())
}

/// Revoke a user's grant on a compartment.
pub async fn revoke(
    store: &Store,
    tenant: &TenantId,
    compartment: &CompartmentId,
    grantee: &UserId,
) -> Result<()> {
    let condition = and_(
        and_(eq("tenant_id", tenant.as_str()), eq("compartment", compartment.as_str())),
        eq("grantee", grantee.as_str()),
    );
    delete_records(store.client(), GRANT, Some(&condition))
        .await
        .map_err(map)?;
    Ok(())
}

/// The grants a user has received in a tenant.
pub async fn list_for_grantee(store: &Store, tenant: &TenantId, grantee: &UserId) -> Result<Vec<Grant>> {
    let query = Query::new()
        .select(None)
        .from_table(GRANT)
        .map_err(map)?
        .where_(and_(eq("tenant_id", tenant.as_str()), eq("grantee", grantee.as_str())));
    let rows: Vec<GrantRow> = query_records(store.client(), &query).await.map_err(map)?;
    rows.into_iter().map(GrantRow::into_domain).collect()
}

/// Every grant on a compartment (its grantees) -- the other half, with the
/// owner, of who may see the compartment's memories. Used to fan a live change
/// out to its audience (R-2).
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
    rows.into_iter().map(GrantRow::into_domain).collect()
}
