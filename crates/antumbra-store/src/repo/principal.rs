//! Tenant principals: the records the `tenant` record-access SIGNIN resolves so
//! a session's `$auth.tenant` is bound (enabling engine-enforced PERMISSIONS).
//!
//! A principal is provisioned by the owner/root; the access SIGNIN matches on
//! the `tenant` field, so the record key is just a sanitized, unique id.

use serde::Deserialize;
use serde_json::json;

use surql::query::builder::Query;
use surql::query::crud::query_records;
use surql::types::operators::eq;

use antumbra_core::{AntumbraError, Result, TenantId, UserId};

use crate::error::map;
use crate::store::Store;

const TABLE: &str = "principal";

fn key(tenant: &TenantId, user: &UserId) -> String {
    format!("{}|{}", tenant.as_str(), user.as_str()).replace([':', '/', '\\', '|'], "_")
}

/// The workspaces `user` already belongs to.
pub async fn workspaces_of(store: &Store, user: &UserId) -> Result<Vec<TenantId>> {
    #[derive(Deserialize)]
    struct Row {
        tenant: String,
    }
    let query = Query::new()
        .select(Some(vec!["tenant".to_string()]))
        .from_table(TABLE)
        .map_err(map)?
        .where_(eq("user", user.as_str()));
    let rows: Vec<Row> = query_records(store.client(), &query).await.map_err(map)?;
    Ok(rows.into_iter().map(|r| TenantId::new(r.tenant)).collect())
}

/// Create the principal for a `(tenant, user)`. Idempotent. The record-access
/// SIGNIN resolves this so `$auth` carries both `tenant` and `user`.
///
/// A user id names one user in one workspace: a user already provisioned in
/// another workspace is refused, so a rule that matches on the user alone (the
/// expert table's read rule) never matches another workspace's rows. System
/// identities ([`antumbra_core::ids::SYSTEM_USERS`]) may belong to many. A
/// principal that already exists is left as it is, refused or not, so a pair
/// provisioned before this rule keeps working.
pub async fn provision(store: &Store, tenant: &TenantId, user: &UserId) -> Result<()> {
    let key = key(tenant, user);
    let existing: Option<serde_json::Value> = store.get_row(TABLE, key.as_str()).await?;
    if existing.is_some() {
        return Ok(());
    }
    if !user.is_system() {
        // Which workspace is not said: the caller has no business knowing it.
        if workspaces_of(store, user)
            .await?
            .iter()
            .any(|t| t != tenant)
        {
            return Err(AntumbraError::rejected(format!(
                "user id {} already belongs to another workspace; a user id names one user in one workspace",
                user.as_str()
            )));
        }
    }
    store
        .put(
            TABLE,
            key.as_str(),
            &json!({ "tenant": tenant.as_str(), "user": user.as_str() }),
        )
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::EMBED_DIM;
    use antumbra_core::ids::GITHUB_SYSTEM_USER;

    #[tokio::test]
    async fn a_user_id_belongs_to_one_workspace_and_a_system_identity_to_many() -> Result<()> {
        let store = Store::connect_memory(EMBED_DIM).await?;
        let (a, b) = (TenantId::new("ws:a"), TenantId::new("ws:b"));
        let ada = UserId::new("user:ada");
        provision(&store, &a, &ada).await?;
        provision(&store, &a, &ada).await?;
        let refused = provision(&store, &b, &ada).await.unwrap_err().to_string();
        assert!(
            refused.contains("already belongs to another workspace"),
            "{refused}"
        );
        assert!(
            !refused.contains("ws:a"),
            "the other workspace is not named: {refused}"
        );
        assert_eq!(workspaces_of(&store, &ada).await?, std::slice::from_ref(&a));

        let github = UserId::new(GITHUB_SYSTEM_USER);
        provision(&store, &a, &github).await?;
        provision(&store, &b, &github).await?;
        let mut both = workspaces_of(&store, &github).await?;
        both.sort_by(|x, y| x.as_str().cmp(y.as_str()));
        assert_eq!(both, [a.clone(), b.clone()]);
        Ok(())
    }

    /// A user provisioned in two workspaces before the rule keeps working in
    /// both: provisioning an existing pair again is left as it is.
    #[tokio::test]
    async fn a_pair_provisioned_before_the_rule_keeps_working() -> Result<()> {
        let store = Store::connect_memory(EMBED_DIM).await?;
        let owner = UserId::new("user:owner");
        for tenant in ["ws:one", "ws:two"] {
            let tenant = TenantId::new(tenant);
            store
                .put(
                    TABLE,
                    key(&tenant, &owner).as_str(),
                    &json!({ "tenant": tenant.as_str(), "user": owner.as_str() }),
                )
                .await?;
        }
        provision(&store, &TenantId::new("ws:one"), &owner).await?;
        provision(&store, &TenantId::new("ws:two"), &owner).await?;
        assert!(provision(&store, &TenantId::new("ws:three"), &owner)
            .await
            .is_err());
        Ok(())
    }
}
