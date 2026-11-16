//! Accounts: the hosted-onboarding identity map (the control plane and product surface): a verified external
//! login (an OAuth `provider:subject`, or `email:<addr>`) → the `(tenant, user)`
//! it owns. Created at signup; read at login. Keyed by the canonical subject.
//! Owner-only: the `account` table has no `PERMISSIONS`, so a tenant session is
//! denied; only the control plane's owner connection touches it (this row maps a
//! login to a tenant and must never be tenant-readable). surql-rs `crud`; no raw
//! SurrealQL.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use antumbra_core::Result;

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

/// Persist a new account (signup). Keyed by subject; idempotent on it.
pub async fn create(store: &Store, account: &Account) -> Result<()> {
    store
        .put(TABLE, key(&account.subject).as_str(), account)
        .await
}

/// Look up the account for a verified login subject (login).
pub async fn get_by_subject(store: &Store, subject: &str) -> Result<Option<Account>> {
    store.get_row(TABLE, key(subject).as_str()).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::EMBED_DIM;

    #[tokio::test]
    async fn create_then_lookup_by_subject() {
        let s = Store::connect_memory(EMBED_DIM).await.unwrap();
        assert!(get_by_subject(&s, "github:1").await.unwrap().is_none());
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
        .await
        .unwrap();
        let got = get_by_subject(&s, "github:1").await.unwrap().unwrap();
        assert_eq!(got.tenant, "ws:abc");
        assert_eq!(got.user, "user:owner");
        assert_eq!(got.email.as_deref(), Some("a@b.com"));
    }
}
