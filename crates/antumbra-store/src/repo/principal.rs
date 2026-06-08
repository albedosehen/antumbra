//! Tenant principals: the records the `tenant` record-access SIGNIN resolves so
//! a session's `$auth.tenant` is bound (enabling engine-enforced PERMISSIONS).
//!
//! A principal is provisioned by the owner/root; the access SIGNIN matches on
//! the `tenant` field, so the record key is just a sanitized, unique id.

use serde_json::json;

use surql::query::crud::upsert_record;
use surql::types::RecordID;

use antumbra_core::{Result, TenantId, UserId};

use crate::error::map;
use crate::store::Store;

const TABLE: &str = "principal";

/// Create (or refresh) the principal for a `(tenant, user)`. Idempotent. The
/// record-access SIGNIN resolves this so `$auth` carries both `tenant` and
/// `user`.
pub async fn provision(store: &Store, tenant: &TenantId, user: &UserId) -> Result<()> {
    let key = format!("{}|{}", tenant.as_str(), user.as_str()).replace([':', '/', '\\', '|'], "_");
    let rid = RecordID::<()>::new(TABLE, key.as_str()).map_err(map)?;
    upsert_record(
        store.client(),
        &rid,
        json!({ "tenant": tenant.as_str(), "user": user.as_str() }),
    )
    .await
    .map_err(map)?;
    Ok(())
}
