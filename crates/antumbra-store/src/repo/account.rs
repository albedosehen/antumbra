//! Accounts: the hosted-onboarding identity map (the control plane and product surface): a verified external
//! login (an OAuth `provider:subject`, or `email:<addr>`) → the `(tenant, user)`
//! it owns. Created at signup; read at login. Keyed by the canonical subject.
//! Owner-only: the `account` table has no `PERMISSIONS`, so a tenant session is
//! denied; only the control plane's owner connection touches it (this row maps a
//! login to a tenant and must never be tenant-readable). surql-rs `crud`; no raw
//! SurrealQL.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use surql::query::builder::Query;
use surql::query::crud::{create_record, query_records};
use surql::types::operators::eq;

use antumbra_core::{AntumbraError, Result};

use crate::error::map;
use crate::store::Store;

const TABLE: &str = crate::tables::ACCOUNT;
const MEMBERSHIP: &str = crate::tables::ACCOUNT_MEMBERSHIP;

/// A hosted account: a login identity bound to the workspace it owns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Account {
    /// The canonical, unique login key, e.g. `github:12345` or `email:a@b.com`.
    pub subject: String,
    pub tenant: String,
    pub user: String,
    #[serde(default)]
    pub email: Option<String>,
    #[serde(default)]
    pub display: Option<String>,
    pub created_at: DateTime<Utc>,
}

fn key(subject: &str) -> String {
    subject.replace([':', '/', '\\', '|', ' ', '@'], "_")
}

/// Persist a new account (signup). Keyed by subject, and a creation rather
/// than a write: a second account for the same login is refused, so a signup
/// that lost a race, or a replayed one, cannot overwrite the first.
pub async fn create(store: &Store, account: &Account) -> Result<()> {
    let mut row = serde_json::to_value(account)?;
    row["id"] = serde_json::json!(key(&account.subject));
    match create_record(store.client(), TABLE, row).await {
        Ok(_) => Ok(()),
        Err(e) if e.to_string().contains("already exists") => Err(AntumbraError::rejected(
            "an account for this login already exists",
        )),
        Err(e) => Err(map(e)),
    }
}

/// Look up the account for a verified login subject (login).
pub async fn get_by_subject(store: &Store, subject: &str) -> Result<Option<Account>> {
    store.get_row(TABLE, key(subject).as_str()).await
}

/// One workspace an account is in, and who the account is there. A user id
/// names one user in one workspace, so the same login is a different user in
/// each workspace it joins.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Membership {
    /// The account's login key.
    pub subject: String,
    pub tenant: String,
    pub user: String,
    pub joined_at: DateTime<Utc>,
}

/// Record that an account is in a workspace. One row for an account in a
/// workspace: said again, the first stands, whoever it names there.
pub async fn add_membership(store: &Store, membership: &Membership) -> Result<()> {
    let mut row = serde_json::to_value(membership)?;
    row["id"] = serde_json::json!(format!(
        "{}__{}",
        key(&membership.subject),
        key(&membership.tenant)
    ));
    match create_record(store.client(), MEMBERSHIP, row).await {
        Ok(_) => Ok(()),
        Err(e) if e.to_string().contains("already exists") => Ok(()),
        Err(e) => Err(map(e)),
    }
}

/// The workspaces an account is in, the first joined first. An account from
/// before memberships were recorded has no row for the workspace it was made
/// with, so that one is always among them, first.
pub async fn memberships(store: &Store, account: &Account) -> Result<Vec<Membership>> {
    let query = Query::new()
        .select(None)
        .from_table(MEMBERSHIP)
        .map_err(map)?
        .where_(eq("subject", account.subject.as_str()));
    let mut all: Vec<Membership> = query_records(store.client(), &query).await.map_err(map)?;
    all.sort_by(|a, b| {
        a.joined_at
            .cmp(&b.joined_at)
            .then_with(|| a.tenant.cmp(&b.tenant))
    });
    if !all.iter().any(|m| m.tenant == account.tenant) {
        all.insert(
            0,
            Membership {
                subject: account.subject.clone(),
                tenant: account.tenant.clone(),
                user: account.user.clone(),
                joined_at: account.created_at,
            },
        );
    }
    Ok(all)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::mem_store;

    /// A signup creates and never overwrites: a second account for the same
    /// login (a signup that lost a race, or a replayed one) is refused and the
    /// first stands.
    #[tokio::test]
    async fn a_second_account_for_the_same_login_is_refused() -> Result<()> {
        let store = crate::testing::mem_store().await?;
        let first = Account {
            subject: "email:dana@example.com".into(),
            tenant: "ws:first".into(),
            user: "user:dana".into(),
            email: Some("dana@example.com".into()),
            display: None,
            created_at: Utc::now(),
        };
        create(&store, &first).await?;
        let second = Account {
            tenant: "ws:second".into(),
            ..first.clone()
        };
        let Err(refused) = create(&store, &second).await else {
            return Err(AntumbraError::other(
                "a second account for the login was written",
            ));
        };
        assert!(refused.is_rejection(), "{refused}");
        let Some(kept) = get_by_subject(&store, &first.subject).await? else {
            return Err(AntumbraError::other("the first account vanished"));
        };
        assert_eq!(kept.tenant, "ws:first");
        Ok(())
    }

    #[tokio::test]
    async fn create_then_lookup_by_subject() -> Result<()> {
        let s = mem_store().await?;
        assert!(get_by_subject(&s, "github:1").await?.is_none());
        create(
            &s,
            &Account {
                subject: "github:1".into(),
                tenant: "ws:abc".into(),
                user: "user:owner".into(),
                email: Some("a@b.com".into()),
                display: Some("Ada".into()),
                created_at: Utc::now(),
            },
        )
        .await?;
        let Some(got) = get_by_subject(&s, "github:1").await? else {
            return Err(antumbra_core::AntumbraError::other(
                "the account did not read back",
            ));
        };
        assert_eq!(got.tenant, "ws:abc");
        assert_eq!(got.user, "user:owner");
        assert_eq!(got.email.as_deref(), Some("a@b.com"));
        Ok(())
    }

    /// An account is in the workspace it was made with whether or not a row
    /// says so; each workspace it joins is one row, said once; and another
    /// account's workspaces are not among them.
    #[tokio::test]
    async fn an_account_lists_the_workspaces_it_is_in() -> Result<()> {
        let s = mem_store().await?;
        let made = Utc::now() - chrono::Duration::days(30);
        let dana = Account {
            subject: "email:dana@example.com".into(),
            tenant: "ws:first".into(),
            user: "user:owner".into(),
            email: Some("dana@example.com".into()),
            display: None,
            created_at: made,
        };
        create(&s, &dana).await?;
        // From before memberships: the workspace it was made with, alone.
        let first = memberships(&s, &dana).await?;
        assert_eq!(first.len(), 1);
        assert_eq!(
            (first[0].tenant.as_str(), first[0].user.as_str()),
            ("ws:first", "user:owner")
        );
        assert_eq!(first[0].joined_at, made);

        let row = |subject: &str, tenant: &str, user: &str, days: i64| Membership {
            subject: subject.into(),
            tenant: tenant.into(),
            user: user.into(),
            joined_at: made + chrono::Duration::days(days),
        };
        add_membership(&s, &row(&dana.subject, "ws:first", "user:owner", 0)).await?;
        add_membership(&s, &row(&dana.subject, "ws:crew", "user:dana", 9)).await?;
        add_membership(&s, &row(&dana.subject, "ws:lab", "user:dana-l", 3)).await?;
        // Said again as someone else: the first stands.
        add_membership(&s, &row(&dana.subject, "ws:crew", "user:other", 12)).await?;
        add_membership(&s, &row("email:eli@example.com", "ws:elis", "user:eli", 1)).await?;

        let all = memberships(&s, &dana).await?;
        let seen: Vec<(&str, &str)> = all
            .iter()
            .map(|m| (m.tenant.as_str(), m.user.as_str()))
            .collect();
        assert_eq!(
            seen,
            [
                ("ws:first", "user:owner"),
                ("ws:lab", "user:dana-l"),
                ("ws:crew", "user:dana")
            ]
        );
        Ok(())
    }
}
