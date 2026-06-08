//! Invite codes — the signup gate for hosted onboarding (ADR-0016). An operator
//! mints a single-use code; a signup redeems it (consumed by deletion). Keyed by
//! the code. Owner-only: the `invite_code` table has no `PERMISSIONS`, so a
//! tenant session is denied — only the control plane's owner connection touches
//! it. Built on surql-rs `crud`; no raw SurrealQL.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use surql::query::crud::{delete_record, get_record, upsert_record};
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

fn record_id(code: &str) -> Result<RecordID> {
    let key = code.replace([':', '/', '\\', '|', ' '], "_");
    RecordID::<()>::new(TABLE, key.as_str()).map_err(map)
}

/// Mint an invite (operator action). Idempotent on the code.
pub async fn mint(store: &Store, code: &str, expires_at: Option<DateTime<Utc>>) -> Result<()> {
    let rid = record_id(code)?;
    let invite = Invite {
        code: code.to_string(),
        expires_at,
        created_at: Utc::now(),
    };
    upsert_record(store.client(), &rid, serde_json::to_value(&invite)?)
        .await
        .map_err(map)?;
    Ok(())
}

/// Read an invite by code, if it exists.
pub async fn get(store: &Store, code: &str) -> Result<Option<Invite>> {
    let rid = record_id(code)?;
    match get_record(store.client(), &rid).await.map_err(map)? {
        Some(value) => Ok(Some(serde_json::from_value(value)?)),
        None => Ok(None),
    }
}

/// Redeem (consume) an invite: it must exist and be unexpired. The row is
/// deleted, so a code is single-use. Errors if the code is unknown or expired.
pub async fn redeem(store: &Store, code: &str, now: DateTime<Utc>) -> Result<()> {
    let invite = get(store, code)
        .await?
        .ok_or_else(|| AntumbraError::other("unknown invite code"))?;
    if invite.expires_at.is_some_and(|exp| exp < now) {
        // Tidy up the dead code on the way out.
        let _ = delete_record(store.client(), &record_id(code)?).await;
        return Err(AntumbraError::other("invite code expired"));
    }
    delete_record(store.client(), &record_id(code)?)
        .await
        .map_err(map)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::EMBED_DIM;

    #[tokio::test]
    async fn mint_then_redeem_is_single_use() {
        let s = Store::connect_memory(EMBED_DIM).await.unwrap();
        mint(&s, "code-abc", None).await.unwrap();
        assert!(get(&s, "code-abc").await.unwrap().is_some());
        redeem(&s, "code-abc", Utc::now()).await.unwrap();
        // Consumed: gone, and a second redeem fails.
        assert!(get(&s, "code-abc").await.unwrap().is_none());
        assert!(redeem(&s, "code-abc", Utc::now()).await.is_err());
        // An unknown code is rejected.
        assert!(redeem(&s, "never-minted", Utc::now()).await.is_err());
    }

    #[tokio::test]
    async fn an_expired_invite_is_rejected() {
        let s = Store::connect_memory(EMBED_DIM).await.unwrap();
        let now = Utc::now();
        mint(&s, "code-exp", Some(now - chrono::Duration::minutes(1)))
            .await
            .unwrap();
        assert!(redeem(&s, "code-exp", now).await.is_err());
    }
}
