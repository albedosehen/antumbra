//! Accounts: the hosted-onboarding identity map (the control plane and product surface): a verified external
//! login (an OAuth `provider:subject`, or `email:<addr>`) → the `(tenant, user)`
//! it owns. Created at signup; read at login. Keyed by the canonical subject.
//! Owner-only: the `account` table has no `PERMISSIONS`, so a tenant session is
//! denied; only the control plane's owner connection touches it (this row maps a
//! login to a tenant and must never be tenant-readable). surql-rs `crud`; no raw
//! SurrealQL.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use surql::query::crud::create_record;

use antumbra_core::{AntumbraError, Result};

use crate::error::map;
use crate::store::Store;

const TABLE: &str = crate::tables::ACCOUNT;

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
}
