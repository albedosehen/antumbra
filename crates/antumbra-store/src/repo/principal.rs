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

use antumbra_core::membership::{self, Member, Role};
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
///
/// A member who was removed is refused: provisioning is what a signed token's
/// first call does, and a removal that the next call undid would be none.
///
/// A new principal is given a role: the first person in a workspace is its
/// owner and everyone after is a member. A system identity is nobody's owner.
pub async fn provision(store: &Store, tenant: &TenantId, user: &UserId) -> Result<()> {
    if let Some(row) = row_of(store, tenant, user).await? {
        if row.removed() {
            return Err(AntumbraError::rejected(
                "you were removed from this workspace; ask one of its admins to let you back in",
            ));
        }
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
    let mut row = json!({ "tenant": tenant.as_str(), "user": user.as_str() });
    if !user.is_system() {
        let first = people(store, tenant).await?.is_empty();
        row["role"] = json!(if first { Role::Owner } else { Role::Member }.as_str());
        row["joined_at"] = json!(chrono::Utc::now().to_rfc3339());
    }
    store.put(TABLE, key(tenant, user).as_str(), &row).await
}

/// A principal's row, as far as membership reads it. A row from before roles
/// has none of the optional fields.
#[derive(Deserialize)]
struct Row {
    user: String,
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    joined_at: Option<String>,
    #[serde(default)]
    status: Option<String>,
}

impl Row {
    fn removed(&self) -> bool {
        self.status.as_deref() == Some(REMOVED)
    }

    fn role(&self) -> Option<Role> {
        self.role.as_deref().and_then(|r| r.parse().ok())
    }
}

/// The `status` of a principal whose member was removed. The row stays, so
/// that a token they still hold is refused and not provisioned afresh.
const REMOVED: &str = "removed";

fn pair(tenant: &TenantId, user: &UserId) -> std::collections::BTreeMap<String, serde_json::Value> {
    std::collections::BTreeMap::from([
        ("tenant".to_string(), json!(tenant.as_str())),
        ("user".to_string(), json!(user.as_str())),
    ])
}

/// The pair's row, by its fields (a row keyed the old way is still the
/// pair's).
async fn row_of(store: &Store, tenant: &TenantId, user: &UserId) -> Result<Option<Row>> {
    let rows: Vec<Row> = store
        .query_rows(
            &format!("SELECT user, role, joined_at, status FROM {TABLE} WHERE tenant = $tenant AND user = $user"),
            pair(tenant, user),
        )
        .await?;
    Ok(rows.into_iter().next())
}

/// Every person who is in the workspace now: not a system identity, not
/// removed.
async fn people(store: &Store, tenant: &TenantId) -> Result<Vec<Row>> {
    let rows: Vec<Row> = store
        .query_rows(
            &format!("SELECT user, role, joined_at, status FROM {TABLE} WHERE tenant = $tenant"),
            std::collections::BTreeMap::from([("tenant".to_string(), json!(tenant.as_str()))]),
        )
        .await?;
    Ok(rows
        .into_iter()
        .filter(|r| !r.removed() && !UserId::new(r.user.clone()).is_system())
        .collect())
}

/// Write fields onto the pair's row, wherever it is keyed.
async fn set(
    store: &Store,
    tenant: &TenantId,
    user: &UserId,
    fields: serde_json::Value,
) -> Result<()> {
    let mut vars = pair(tenant, user);
    vars.insert("fields".to_string(), fields);
    store
        .query_rows::<serde_json::Value>(
            &format!("UPDATE {TABLE} MERGE $fields WHERE tenant = $tenant AND user = $user"),
            vars,
        )
        .await?;
    Ok(())
}

/// A member's role in a workspace, `None` for someone who is not in it (or
/// was removed).
///
/// A workspace from before roles has none recorded, and gets its owner from
/// whoever ran it:
///
/// - **A workspace with no owner is taken by whoever holds the server
///   operator's token,** whatever role their own row has recorded. Without
///   this a workspace whose members were all recorded as members had no way
///   to an owner short of editing the store: nobody could change a role, and
///   an operator's new identity was one more member.
/// - **Someone alone in a workspace with no role recorded is its owner.**
/// - **Anyone else with no role recorded is a member, and it is recorded
///   only once the workspace has an owner,** who can then change it. Until
///   then it is an answer and not a record: an ask is often a read (a page
///   listing members), and a read that fixed the first asker as a member
///   closed the workspace to the person it belonged to.
pub async fn role_of(
    store: &Store,
    tenant: &TenantId,
    user: &UserId,
    runs_the_server: bool,
) -> Result<Option<Role>> {
    let Some(row) = row_of(store, tenant, user).await? else {
        return Ok(None);
    };
    if row.removed() || user.is_system() {
        return Ok(None);
    }
    let recorded = row.role();
    match recorded {
        Some(Role::Owner) => return Ok(recorded),
        // The common ask, answered from the caller's own row.
        Some(_) if !runs_the_server => return Ok(recorded),
        _ => {}
    }
    let everyone = people(store, tenant).await?;
    let owned = everyone.iter().any(|r| r.role() == Some(Role::Owner));
    if !owned && (runs_the_server || (recorded.is_none() && everyone.len() == 1)) {
        set(store, tenant, user, json!({ "role": Role::Owner.as_str() })).await?;
        return Ok(Some(Role::Owner));
    }
    if recorded.is_none() && owned {
        set(
            store,
            tenant,
            user,
            json!({ "role": Role::Member.as_str() }),
        )
        .await?;
    }
    Ok(Some(recorded.unwrap_or(Role::Member)))
}

/// Whether the pair is a member now: provisioned, and not removed.
pub async fn is_member(store: &Store, tenant: &TenantId, user: &UserId) -> Result<bool> {
    Ok(row_of(store, tenant, user)
        .await?
        .is_some_and(|row| !row.removed()))
}

/// The workspace's members, owners first, then admins, then by user id. A
/// member with no role recorded yet is listed as a member.
pub async fn members(store: &Store, tenant: &TenantId) -> Result<Vec<Member>> {
    let mut all: Vec<Member> = people(store, tenant)
        .await?
        .into_iter()
        .map(|r| Member {
            role: r.role().unwrap_or(Role::Member),
            joined_at: r
                .joined_at
                .as_deref()
                .and_then(|at| chrono::DateTime::parse_from_rfc3339(at).ok())
                .map(|at| at.with_timezone(&chrono::Utc)),
            user: UserId::new(r.user),
        })
        .collect();
    let rank = |role: Role| match role {
        Role::Owner => 0,
        Role::Admin => 1,
        Role::Member => 2,
    };
    all.sort_by(|a, b| {
        rank(a.role)
            .cmp(&rank(b.role))
            .then_with(|| a.user.as_str().cmp(b.user.as_str()))
    });
    Ok(all)
}

/// The role `user` holds, or a refusal that says they are not in it.
async fn standing(store: &Store, tenant: &TenantId, user: &UserId) -> Result<Role> {
    role_of(store, tenant, user, false).await?.ok_or_else(|| {
        AntumbraError::rejected(format!(
            "{} is not a member of this workspace",
            user.as_str()
        ))
    })
}

/// The workspace keeps an owner: refused when `target` is its last and would
/// stop being one.
async fn keeps_an_owner(store: &Store, tenant: &TenantId, target: &UserId) -> Result<()> {
    let others = people(store, tenant)
        .await?
        .into_iter()
        .filter(|r| r.role() == Some(Role::Owner) && r.user != target.as_str())
        .count();
    if others == 0 {
        return Err(AntumbraError::rejected(
            "a workspace keeps an owner: make someone else an owner first",
        ));
    }
    Ok(())
}

/// Give `target` the role `to`, as `actor` ([`membership::may_set_role`]).
pub async fn set_role(
    store: &Store,
    tenant: &TenantId,
    actor: &UserId,
    target: &UserId,
    to: Role,
) -> Result<()> {
    let (acting, held) = (
        standing(store, tenant, actor).await?,
        standing(store, tenant, target).await?,
    );
    membership::may_set_role(acting, held, to)?;
    if held == Role::Owner {
        keeps_an_owner(store, tenant, target).await?;
    }
    set(store, tenant, target, json!({ "role": to.as_str() })).await
}

/// Remove `target` from the workspace, as `actor` ([`membership::may_remove`]),
/// or leave it: a member removes themselves whatever their role, unless they
/// are its last owner.
/// Their row stays, marked, so a token they still hold is refused and not
/// provisioned afresh; what they wrote stays where it is.
pub async fn remove(
    store: &Store,
    tenant: &TenantId,
    actor: &UserId,
    target: &UserId,
) -> Result<()> {
    let (acting, held) = (
        standing(store, tenant, actor).await?,
        standing(store, tenant, target).await?,
    );
    // Anyone may leave; removing someone else goes by role.
    if actor != target {
        membership::may_remove(acting, held)?;
    }
    if held == Role::Owner {
        keeps_an_owner(store, tenant, target).await?;
    }
    set(
        store,
        tenant,
        target,
        json!({
            "status": REMOVED,
            "removed_at": chrono::Utc::now().to_rfc3339(),
            "removed_by": actor.as_str(),
        }),
    )
    .await
}

/// Let a removed member back in, as a member. Refused for anyone but an admin
/// or an owner.
pub async fn readmit(
    store: &Store,
    tenant: &TenantId,
    actor: &UserId,
    target: &UserId,
) -> Result<()> {
    if !standing(store, tenant, actor).await?.admits() {
        return Err(AntumbraError::rejected(
            "only an admin or an owner of this workspace lets someone back in",
        ));
    }
    match row_of(store, tenant, target).await? {
        Some(row) if row.removed() => {
            set(
                store,
                tenant,
                target,
                json!({ "status": "member", "role": Role::Member.as_str() }),
            )
            .await
        }
        _ => Err(AntumbraError::rejected(format!(
            "{} was not removed from this workspace",
            target.as_str()
        ))),
    }
}

/// Every (workspace, user) whose member was removed: what a server refuses
/// tokens for.
pub async fn removed(store: &Store) -> Result<Vec<(TenantId, UserId)>> {
    #[derive(Deserialize)]
    struct Gone {
        tenant: String,
        user: String,
    }
    let rows: Vec<Gone> = store
        .query_rows(
            &format!("SELECT tenant, user FROM {TABLE} WHERE status = $status"),
            std::collections::BTreeMap::from([("status".to_string(), json!(REMOVED))]),
        )
        .await?;
    Ok(rows
        .into_iter()
        .map(|g| (TenantId::new(g.tenant), UserId::new(g.user)))
        .collect())
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

    /// The first person in a workspace owns it and the next is a member; a
    /// system identity is neither listed nor an owner.
    #[tokio::test]
    async fn the_first_person_in_a_workspace_is_its_owner() -> Result<()> {
        let store = mem_store().await?;
        let t = TenantId::new("ws:team");
        let (github, ada, bo) = (
            UserId::new(GITHUB_SYSTEM_USER),
            UserId::new("user:ada"),
            UserId::new("user:bo"),
        );
        provision(&store, &t, &github).await?;
        provision(&store, &t, &ada).await?;
        provision(&store, &t, &bo).await?;
        let listed: Vec<(String, Role)> = members(&store, &t)
            .await?
            .into_iter()
            .map(|m| (m.user.as_str().to_string(), m.role))
            .collect();
        assert_eq!(
            listed,
            [
                ("user:ada".to_string(), Role::Owner),
                ("user:bo".to_string(), Role::Member)
            ]
        );
        assert_eq!(role_of(&store, &t, &github, true).await?, None);
        assert_eq!(
            role_of(&store, &t, &UserId::new("user:nobody"), true).await?,
            None
        );
        Ok(())
    }

    /// A workspace from before roles gets its owner from whoever ran it: the
    /// one holding the server operator's token, or the only person in it.
    #[tokio::test]
    async fn members_from_before_roles_are_given_one_when_asked() -> Result<()> {
        let store = mem_store().await?;
        let t = TenantId::new("ws:old");
        let before = |user: &str| {
            let (store, t, user) = (store.clone(), t.clone(), UserId::new(user));
            async move {
                store
                    .put(
                        TABLE,
                        key(&t, &user).as_str(),
                        &json!({ "tenant": t.as_str(), "user": user.as_str() }),
                    )
                    .await
            }
        };
        before("user:ada").await?;
        before("user:bo").await?;
        let (ada, bo) = (UserId::new("user:ada"), UserId::new("user:bo"));
        // Two people and no owner: one who does not run the server is a member.
        assert_eq!(role_of(&store, &t, &bo, false).await?, Some(Role::Member));
        assert_eq!(role_of(&store, &t, &ada, true).await?, Some(Role::Owner));
        // Once there is an owner, running the server makes no second one.
        before("user:cy").await?;
        assert_eq!(
            role_of(&store, &t, &UserId::new("user:cy"), true).await?,
            Some(Role::Member)
        );

        let alone = TenantId::new("ws:solo");
        store
            .put(
                TABLE,
                "solo",
                &json!({ "tenant": "ws:solo", "user": "user:default" }),
            )
            .await?;
        assert_eq!(
            role_of(&store, &alone, &UserId::new("user:default"), false).await?,
            Some(Role::Owner)
        );
        Ok(())
    }

    /// A member leaves by removing themselves, whatever their role, and is
    /// then refused as anyone removed is; the last owner cannot leave.
    #[tokio::test]
    async fn a_member_leaves_and_the_last_owner_does_not() -> Result<()> {
        let store = mem_store().await?;
        let t = TenantId::new("ws:crew");
        let (ada, bo) = (UserId::new("user:ada"), UserId::new("user:bo"));
        provision(&store, &t, &ada).await?;
        provision(&store, &t, &bo).await?;
        // A member cannot remove the owner, and can leave.
        assert!(remove(&store, &t, &bo, &ada).await.is_err());
        remove(&store, &t, &bo, &bo).await?;
        assert!(!is_member(&store, &t, &bo).await?);
        assert!(provision(&store, &t, &bo).await.is_err());
        // The workspace keeps its owner.
        let Err(kept) = remove(&store, &t, &ada, &ada).await else {
            return Err(AntumbraError::other("the last owner left"));
        };
        assert!(kept.is_rejection(), "{kept}");
        assert!(is_member(&store, &t, &ada).await?);
        Ok(())
    }

    /// A workspace from before roles, with several people and no owner, as
    /// one was found after an upgrade: its first asker is told member and no
    /// more is recorded, so the one who runs the server can still be made its
    /// owner; and a workspace already left with members only is taken by
    /// whoever holds the operator's token, a new identity among them.
    #[tokio::test]
    async fn a_workspace_with_no_owner_is_not_closed_by_being_asked() -> Result<()> {
        let store = mem_store().await?;
        let t = TenantId::new("ws:old");
        let put = |user: &str, role: Option<Role>| {
            let (store, t, user) = (store.clone(), t.clone(), UserId::new(user));
            async move {
                let mut row = json!({ "tenant": t.as_str(), "user": user.as_str() });
                if let Some(role) = role {
                    row["role"] = json!(role.as_str());
                }
                store.put(TABLE, key(&t, &user).as_str(), &row).await
            }
        };
        let recorded = |user: &str| {
            let (store, t, user) = (store.clone(), t.clone(), UserId::new(user));
            async move {
                Ok::<_, AntumbraError>(row_of(&store, &t, &user).await?.and_then(|r| r.role()))
            }
        };
        let (own, check) = (UserId::new("user:default"), UserId::new("user:check"));
        put("user:default", None).await?;
        put("user:check", None).await?;

        // A read by the workspace's real owner, with an ordinary token: a
        // member for now, and nothing is written.
        assert_eq!(role_of(&store, &t, &own, false).await?, Some(Role::Member));
        assert_eq!(recorded("user:default").await?, None);
        // The same person with the operator's token is its owner.
        assert_eq!(role_of(&store, &t, &own, true).await?, Some(Role::Owner));
        assert_eq!(recorded("user:default").await?, Some(Role::Owner));
        // Now that it has one, the other is recorded as a member when asked.
        assert_eq!(
            role_of(&store, &t, &check, false).await?,
            Some(Role::Member)
        );
        assert_eq!(recorded("user:check").await?, Some(Role::Member));
        // And an operator's token makes no second owner.
        assert_eq!(role_of(&store, &t, &check, true).await?, Some(Role::Member));

        // A workspace the earlier rule left with members only.
        let stuck = TenantId::new("ws:stuck");
        for user in ["user:a", "user:b"] {
            store
                .put(
                    TABLE,
                    &format!("stuck-{user}"),
                    &json!({ "tenant": "ws:stuck", "user": user, "role": "member" }),
                )
                .await?;
        }
        let a = UserId::new("user:a");
        assert_eq!(
            role_of(&store, &stuck, &a, false).await?,
            Some(Role::Member)
        );
        // A new identity with the operator's token: provisioned a member,
        // and the workspace's owner when it asks, since there is none.
        let operator = UserId::new("user:operator");
        provision(&store, &stuck, &operator).await?;
        assert_eq!(
            role_of(&store, &stuck, &operator, true).await?,
            Some(Role::Owner)
        );
        set_role(&store, &stuck, &operator, &a, Role::Owner).await?;
        assert_eq!(role_of(&store, &stuck, &a, false).await?, Some(Role::Owner));
        // An ordinary token never takes a workspace that others are in.
        let other = TenantId::new("ws:two");
        for user in ["user:x", "user:y"] {
            store
                .put(
                    TABLE,
                    &format!("two-{user}"),
                    &json!({ "tenant": "ws:two", "user": user, "role": "member" }),
                )
                .await?;
        }
        assert_eq!(
            role_of(&store, &other, &UserId::new("user:x"), false).await?,
            Some(Role::Member)
        );
        Ok(())
    }

    /// Roles are an owner's to change, a workspace keeps an owner, and a
    /// removed member is refused until let back in.
    #[tokio::test]
    async fn roles_change_and_members_leave_by_the_rules() -> Result<()> {
        let store = mem_store().await?;
        let t = TenantId::new("ws:team");
        let (ada, bo, cy) = (
            UserId::new("user:ada"),
            UserId::new("user:bo"),
            UserId::new("user:cy"),
        );
        for user in [&ada, &bo, &cy] {
            provision(&store, &t, user).await?;
        }
        let refused = |r: Result<()>| r.err().map(|e| e.to_string()).unwrap_or_default();

        // A member changes nothing; the owner makes an admin.
        assert!(
            refused(set_role(&store, &t, &bo, &cy, Role::Admin).await).contains("only an owner")
        );
        set_role(&store, &t, &ada, &bo, Role::Admin).await?;
        // An admin removes a member, not the owner.
        assert!(refused(remove(&store, &t, &bo, &ada).await).contains("only an owner"));
        remove(&store, &t, &bo, &cy).await?;
        assert_eq!(role_of(&store, &t, &cy, false).await?, None);
        assert!(refused(provision(&store, &t, &cy).await).contains("removed from this workspace"));
        assert_eq!(removed(&store).await?, [(t.clone(), cy.clone())]);
        // The engine refuses them too: a removed member's row does not sign in.
        assert!(
            store.signin(&t, &cy).await.is_err(),
            "a removed member signed in"
        );
        store.signin(&t, &bo).await?;
        store.invalidate().await?;
        store.signin_root().await?;
        assert_eq!(members(&store, &t).await?.len(), 2);

        // The last owner neither steps down nor leaves.
        assert!(
            refused(set_role(&store, &t, &ada, &ada, Role::Member).await)
                .contains("keeps an owner")
        );
        assert!(refused(remove(&store, &t, &ada, &ada).await).contains("keeps an owner"));
        set_role(&store, &t, &ada, &bo, Role::Owner).await?;
        set_role(&store, &t, &bo, &ada, Role::Member).await?;

        // Back in, as a member, by an admin or an owner only.
        assert!(refused(readmit(&store, &t, &ada, &cy).await).contains("only an admin"));
        readmit(&store, &t, &bo, &cy).await?;
        provision(&store, &t, &cy).await?;
        assert_eq!(role_of(&store, &t, &cy, false).await?, Some(Role::Member));
        assert!(removed(&store).await?.is_empty());
        Ok(())
    }
}
