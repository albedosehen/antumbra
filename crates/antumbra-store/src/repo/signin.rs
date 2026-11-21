//! A sign-in waiting to be collected: a machine asked to sign in to an
//! account, the account's address followed the link, and the machine that
//! asked collects a token with the secret whose digest is kept here. Made
//! when the link is followed and not before, so nothing is kept for an
//! address nobody verified. Owner-only (no `PERMISSIONS`), like the rest of
//! the control plane's tables.

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use surql::query::crud::delete_records;
use surql::types::operators::lt;

use antumbra_core::Result;

use crate::error::map;
use crate::store::Store;

const TABLE: &str = crate::tables::SIGNIN;

/// One sign-in, between its link being followed and its token collected.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Signin {
    pub key: String,
    /// The account's login key, as the followed link verified it.
    pub subject: String,
    /// The digest of the secret the token is collected with.
    pub secret: String,
    /// The workspace asked for with the link, when one was.
    #[serde(default)]
    pub workspace: Option<String>,
    pub expires_at: DateTime<Utc>,
    #[serde(default)]
    pub collected: bool,
}

/// Keep a sign-in whose link was just followed.
pub async fn open(store: &Store, signin: &Signin) -> Result<()> {
    store.put(TABLE, &signin.key, signin).await
}

pub async fn get(store: &Store, id: &str) -> Result<Option<Signin>> {
    store.get_row(TABLE, id).await
}

/// Take a sign-in for collecting. `true` for the one caller that takes it:
/// two machines racing with the same secret cannot both be given a token.
pub async fn take(store: &Store, id: &str) -> Result<bool> {
    let vars: BTreeMap<String, Value> = [("id".to_string(), json!(id))].into_iter().collect();
    let taken: Vec<Value> = store
        .query_rows(
            &format!(
                "UPDATE type::record('{TABLE}', $id) SET collected = true \
                 WHERE collected != true RETURN AFTER"
            ),
            vars,
        )
        .await?;
    Ok(!taken.is_empty())
}

/// Drop sign-ins past their time, collected or not.
pub async fn sweep_expired(store: &Store, now: DateTime<Utc>) -> Result<()> {
    let cutoff = now.to_rfc3339();
    delete_records(
        store.client(),
        TABLE,
        Some(&lt("expires_at", cutoff.as_str())),
    )
    .await
    .map_err(map)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::mem_store;

    /// A sign-in is kept, read, taken by one caller only, and gone once its
    /// time has passed.
    #[tokio::test]
    async fn a_sign_in_is_taken_once_and_swept_when_past() -> Result<()> {
        let s = mem_store().await?;
        let now = Utc::now();
        let signin = Signin {
            key: "s1".into(),
            subject: "email:dana@example.com".into(),
            secret: "digest".into(),
            workspace: Some("ws:crew".into()),
            expires_at: now + chrono::Duration::minutes(15),
            collected: false,
        };
        open(&s, &signin).await?;
        assert_eq!(get(&s, "s1").await?, Some(signin));
        assert!(get(&s, "nope").await?.is_none());
        assert!(take(&s, "s1").await?, "the first takes it");
        assert!(!take(&s, "s1").await?, "the second does not");
        assert!(!take(&s, "nope").await?);
        sweep_expired(&s, now).await?;
        assert!(get(&s, "s1").await?.is_some(), "not past yet");
        sweep_expired(&s, now + chrono::Duration::minutes(16)).await?;
        assert!(get(&s, "s1").await?.is_none());
        Ok(())
    }
}
