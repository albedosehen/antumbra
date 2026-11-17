//! Tenant principals: the records the `tenant` record-access SIGNIN resolves so
//! a session's `$auth.tenant` is bound (enabling engine-enforced PERMISSIONS).
//!
//! A principal is provisioned by the owner/root; the access SIGNIN matches on
//! the `tenant` field, so the record key is just a sanitized, unique id.

use serde::Deserialize;
use serde_json::json;

use surql::query::builder::Query;
use surql::query::crud::query_records;
use surql::types::operators::{and_, eq};

use antumbra_core::{AntumbraError, Result, TenantId, UserId};

use crate::error::map;
use crate::store::Store;

const TABLE: &str = crate::tables::PRINCIPAL;

/// The record key of a pair. Each part is escaped on its own (the characters a
/// record key cannot carry, and `_` itself, as `_` and two hex digits) and the
/// two are joined with `__`, so two pairs never share a key: `ws:a` + `user:b`
/// and `ws_a_user` + `b` were one key when the separator was replaced after
/// the join. Rows keyed the old way keep working: nothing reads a principal by
/// its key, and [`provision`] looks for a pair by its fields.
fn key(tenant: &TenantId, user: &UserId) -> String {
    format!("{}__{}", escaped(tenant.as_str()), escaped(user.as_str()))
}

fn escaped(part: &str) -> String {
    let mut out = String::with_capacity(part.len());
    for c in part.chars() {
        match c {
            ':' | '/' | '\\' | '|' | '_' | ' ' => out.push_str(&format!("_{:02x}", u32::from(c))),
            _ => out.push(c),
        }
    }
    out
}

/// Whether a principal for the pair exists, by its fields: the key scheme has
/// changed once, and a row keyed the old way is still the pair's.
async fn exists(store: &Store, tenant: &TenantId, user: &UserId) -> Result<bool> {
    let query = Query::new()
        .select(Some(vec!["user".to_string()]))
        .from_table(TABLE)
        .map_err(map)?
        .where_(and_(
            eq("tenant", tenant.as_str()),
            eq("user", user.as_str()),
        ));
    let rows: Vec<serde_json::Value> = query_records(store.client(), &query).await.map_err(map)?;
    Ok(!rows.is_empty())
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
    if exists(store, tenant, user).await? {
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
            key(tenant, user).as_str(),
            &json!({ "tenant": tenant.as_str(), "user": user.as_str() }),
        )
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::mem_store;
    use antumbra_core::ids::GITHUB_SYSTEM_USER;

    #[tokio::test]
    async fn a_user_id_belongs_to_one_workspace_and_a_system_identity_to_many() -> Result<()> {
        let store = mem_store().await?;
        let (a, b) = (TenantId::new("ws:a"), TenantId::new("ws:b"));
        let ada = UserId::new("user:ada");
        provision(&store, &a, &ada).await?;
        provision(&store, &a, &ada).await?;
        let Err(refused) = provision(&store, &b, &ada).await else {
            return Err(AntumbraError::other(
                "a user id was provisioned into a second workspace",
            ));
        };
        let refused = refused.to_string();
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

    /// Two pairs that differ only in where the separator falls never share a
    /// record: `ws:a` + `user:b` and `ws_a_user` + `b` are two principals.
    #[tokio::test]
    async fn distinct_pairs_never_share_a_record() -> Result<()> {
        let store = mem_store().await?;
        let (first_t, first_u) = (TenantId::new("ws:a"), UserId::new("user:b"));
        let (second_t, second_u) = (TenantId::new("ws_a_user"), UserId::new("b"));
        provision(&store, &first_t, &first_u).await?;
        provision(&store, &second_t, &second_u).await?;
        assert_ne!(key(&first_t, &first_u), key(&second_t, &second_u));
        assert_eq!(
            workspaces_of(&store, &second_u).await?,
            std::slice::from_ref(&second_t),
            "the second pair has a principal of its own"
        );
        assert_eq!(
            workspaces_of(&store, &first_u).await?,
            std::slice::from_ref(&first_t)
        );
        // Both sign in: the record access resolves each pair to its own row.
        store.signin(&first_t, &first_u).await?;
        store.invalidate().await?;
        store.signin(&second_t, &second_u).await?;
        store.invalidate().await?;
        Ok(())
    }

    /// A user provisioned in two workspaces before the rule keeps working in
    /// both: provisioning an existing pair again is left as it is.
    #[tokio::test]
    async fn a_pair_provisioned_before_the_rule_keeps_working() -> Result<()> {
        let store = mem_store().await?;
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
