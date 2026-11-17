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

const TABLE: &str = crate::tables::INVITE_CODE;

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
/// deleted, so a code is single-use. Rejects if the code is unknown, expired
/// or already used.
///
/// One statement checks and consumes, so two callers racing on a code cannot
/// both pass the check before either consumes it: an unexpired row comes back
/// deleted, anything else comes back empty. The invite is the one hand-authored
/// statement in this module, since the builders have no `RETURN BEFORE`.
pub async fn redeem(store: &Store, code: &str, now: DateTime<Utc>) -> Result<()> {
    let vars = |now: &str| {
        std::collections::BTreeMap::from([
            ("key".to_string(), serde_json::json!(key(code))),
            ("now".to_string(), serde_json::json!(now)),
        ])
    };
    let now = now.to_rfc3339();
    let consumed: Vec<serde_json::Value> = store
        .query_rows(
            &format!(
                "DELETE type::record('{TABLE}', $key) \
                 WHERE expires_at IS NONE OR expires_at IS NULL \
                    OR type::datetime(expires_at) >= type::datetime($now) \
                 RETURN BEFORE"
            ),
            vars(&now),
        )
        .await?;
    if consumed.is_empty() {
        // An expired code is dead weight: tidied on the way out, and whether
        // one was there is not said. The sweep's own outcome does not matter.
        store
            .query_rows::<serde_json::Value>(
                &format!(
                    "DELETE type::record('{TABLE}', $key) \
                     WHERE type::datetime(expires_at) < type::datetime($now)"
                ),
                vars(&now),
            )
            .await
            .ok();
        return Err(AntumbraError::rejected(
            "invite code unknown, expired, or already used",
        ));
    }
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

    /// A single-use code redeemed twice at once is redeemed once: the check
    /// and the consume are one statement, so two callers racing on it cannot
    /// both pass the check before either consumes.
    #[tokio::test]
    async fn two_redemptions_of_one_code_at_once_succeed_once() -> Result<()> {
        let s = mem_store().await?;
        mint(&s, "code-race", None).await?;
        let now = Utc::now();
        let (first, second) =
            tokio::join!(redeem(&s, "code-race", now), redeem(&s, "code-race", now));
        let succeeded = usize::from(first.is_ok()) + usize::from(second.is_ok());
        assert_eq!(succeeded, 1, "{first:?} / {second:?}");
        assert!(get(&s, "code-race").await?.is_none());
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
