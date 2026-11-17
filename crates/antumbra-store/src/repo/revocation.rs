//! Revoked tokens: the denylist a networked server checks a token's `jti`
//! against, so a token saved on a lost or shared machine can be ended before
//! it expires without rotating the signing key for everyone. A row lives as
//! long as the token it names would have (its `expires_at`), then the sweep
//! drops it: a token past its expiry is refused by the verifier anyway.
//! Owner-only (no `PERMISSIONS`, so a tenant session is denied): the operator's
//! CLI writes it, the server reads it in the owner view.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use surql::query::crud::delete_records;
use surql::types::operators::lt;

use antumbra_core::Result;

use crate::error::map;
use crate::store::Store;

const TABLE: &str = crate::tables::REVOKED_TOKEN;

/// One revoked token, by the id its claims carry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Revocation {
    pub jti: String,
    /// When the token would have expired on its own: the row is kept until
    /// then and swept after.
    pub expires_at: DateTime<Utc>,
    pub revoked_at: DateTime<Utc>,
    /// Why, in the operator's words, when they gave any.
    #[serde(default)]
    pub note: Option<String>,
}

fn key(jti: &str) -> String {
    jti.replace([':', '/', '\\', '|', ' '], "_")
}

/// Revoke the token `jti`, kept on the list until `expires_at`. Idempotent:
/// revoking a token twice records the later call.
pub async fn revoke(
    store: &Store,
    jti: &str,
    expires_at: DateTime<Utc>,
    note: Option<String>,
) -> Result<()> {
    let row = Revocation {
        jti: jti.to_string(),
        expires_at,
        revoked_at: Utc::now(),
        note,
    };
    store.put(TABLE, key(jti).as_str(), &row).await
}

/// Whether `jti` is on the list.
pub async fn is_revoked(store: &Store, jti: &str) -> Result<bool> {
    Ok(store
        .get_row::<Revocation>(TABLE, key(jti).as_str())
        .await?
        .is_some())
}

/// Every revoked id whose token has not yet expired at `now`: what a server
/// loads into its denylist.
pub async fn live(store: &Store, now: DateTime<Utc>) -> Result<Vec<String>> {
    let rows: Vec<Revocation> = store.read_paged(TABLE, None, None).await?;
    Ok(rows
        .into_iter()
        .filter(|r| r.expires_at >= now)
        .map(|r| r.jti)
        .collect())
}

/// Drop the rows whose token has expired on its own: the verifier refuses it
/// already, so the row is dead weight. Timestamps are UTC RFC3339, so the
/// lexicographic `<` matches chronological order (the idiom the other
/// ledgers use).
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

    #[tokio::test]
    async fn a_revoked_id_is_listed_until_its_token_would_have_expired() -> Result<()> {
        let store = mem_store().await?;
        let now = Utc::now();
        revoke(
            &store,
            "abc123",
            now + chrono::Duration::days(3),
            Some("laptop lost".into()),
        )
        .await?;
        revoke(&store, "old456", now - chrono::Duration::minutes(1), None).await?;
        assert!(is_revoked(&store, "abc123").await?);
        assert!(!is_revoked(&store, "never").await?);
        let mut listed = live(&store, now).await?;
        listed.sort();
        assert_eq!(listed, ["abc123"], "an expired token is not worth listing");
        sweep_expired(&store, now).await?;
        assert!(
            !is_revoked(&store, "old456").await?,
            "the expired row is swept"
        );
        assert!(is_revoked(&store, "abc123").await?, "the live row stays");
        Ok(())
    }
}
