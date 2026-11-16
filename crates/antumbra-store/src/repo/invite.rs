//! Invite codes: the signup gate for hosted onboarding (the control plane and product surface). An operator
//! mints a single-use code; a signup redeems it (consumed by deletion). Keyed by
//! the code. Owner-only: the `invite_code` table has no `PERMISSIONS`, so a
//! tenant session is denied; only the control plane's owner connection touches
//! it. Built on surql-rs `crud`; no raw SurrealQL.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use surql::query::builder::Query;
use surql::query::crud::{delete_record, query_records};
use surql::types::RecordID;

use antumbra_core::{AntumbraError, Result};

use crate::error::map;
use crate::store::Store;

const TABLE: &str = "invite_code";

/// A mint-once signup invite.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Invite {
    pub code: String,
    /// When the invite stops being valid; `None` = no expiry.
    #[serde(default)]
    pub expires_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
}

impl Invite {
    /// Whether the invite is still redeemable at `now` (unexpired or eternal).
    pub fn is_usable_at(&self, now: DateTime<Utc>) -> bool {
        self.expires_at.is_none_or(|exp| exp >= now)
    }
}

fn key(code: &str) -> String {
    code.replace([':', '/', '\\', '|', ' '], "_")
}

fn record_id(code: &str) -> Result<RecordID> {
    RecordID::<()>::new(TABLE, key(code).as_str()).map_err(map)
}

/// Mint an invite (operator action). Idempotent on the code.
pub async fn mint(store: &Store, code: &str, expires_at: Option<DateTime<Utc>>) -> Result<()> {
    let invite = Invite {
        code: code.to_string(),
        expires_at,
        created_at: Utc::now(),
    };
    store.put(TABLE, key(code).as_str(), &invite).await
}

/// Read an invite by code, if it exists.
pub async fn get(store: &Store, code: &str) -> Result<Option<Invite>> {
    store.get_row(TABLE, key(code).as_str()).await
}

/// Every outstanding (un-redeemed) invite: the operator's audit view.
pub async fn all(store: &Store) -> Result<Vec<Invite>> {
    let query = Query::new().select(None).from_table(TABLE).map_err(map)?;
    query_records(store.client(), &query).await.map_err(map)
}

/// Revoke an outstanding invite (operator action). Returns whether the code
/// existed; revoking an unknown/already-redeemed code is a quiet no-op.
pub async fn revoke(store: &Store, code: &str) -> Result<bool> {
    let existed = get(store, code).await?.is_some();
    if existed {
        delete_record(store.client(), &record_id(code)?)
            .await
            .map_err(map)?;
    }
    Ok(existed)
}

/// Redeem (consume) an invite: it must exist and be unexpired. The row is
/// deleted, so a code is single-use. Rejects if the code is unknown or expired.
pub async fn redeem(store: &Store, code: &str, now: DateTime<Utc>) -> Result<()> {
    let invite = get(store, code)
        .await?
        .ok_or_else(|| AntumbraError::rejected("unknown invite code"))?;
    if !invite.is_usable_at(now) {
        // Tidy up the dead code on the way out.
        let _ = delete_record(store.client(), &record_id(code)?).await;
        return Err(AntumbraError::rejected("invite code expired"));
    }
    delete_record(store.client(), &record_id(code)?)
        .await
        .map_err(map)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::mem_store;

    #[tokio::test]
    async fn mint_then_redeem_is_single_use() -> Result<()> {
        let s = mem_store().await?;
        mint(&s, "code-abc", None).await?;
        assert!(get(&s, "code-abc").await?.is_some());
        redeem(&s, "code-abc", Utc::now()).await?;
        // Consumed: gone, and a second redeem fails.
        assert!(get(&s, "code-abc").await?.is_none());
        assert!(redeem(&s, "code-abc", Utc::now()).await.is_err());
        // An unknown code is rejected.
        assert!(redeem(&s, "never-minted", Utc::now()).await.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn an_expired_invite_is_rejected() -> Result<()> {
        let s = mem_store().await?;
        let now = Utc::now();
        mint(&s, "code-exp", Some(now - chrono::Duration::minutes(1))).await?;
        let Err(err) = redeem(&s, "code-exp", now).await else {
            return Err(AntumbraError::other("an expired invite was redeemed"));
        };
        assert!(err.is_rejection(), "a caller fault, not an internal one");
        // And `is_usable_at` reports the same verdict without consuming.
        mint(&s, "code-live", Some(now + chrono::Duration::minutes(5))).await?;
        let Some(live) = get(&s, "code-live").await? else {
            return Err(AntumbraError::other("the live invite did not read back"));
        };
        assert!(live.is_usable_at(now));
        assert!(!live.is_usable_at(now + chrono::Duration::minutes(6)));
        Ok(())
    }

    #[tokio::test]
    async fn list_and_revoke_manage_outstanding_invites() -> Result<()> {
        let s = mem_store().await?;
        mint(&s, "a", None).await?;
        mint(&s, "b", Some(Utc::now() + chrono::Duration::days(7))).await?;
        let mut codes: Vec<String> = all(&s).await?.into_iter().map(|i| i.code).collect();
        codes.sort();
        assert_eq!(codes, ["a", "b"]);

        assert!(revoke(&s, "a").await?, "revoked");
        assert!(!revoke(&s, "a").await?, "second revoke is a no-op");
        assert!(redeem(&s, "a", Utc::now()).await.is_err(), "a is dead");
        assert_eq!(all(&s).await?.len(), 1);
        Ok(())
    }
}
