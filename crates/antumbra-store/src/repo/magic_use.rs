//! Consumed magic-link ids: the single-use ledger behind passwordless login.
//! A verified link's `jti` is written through here exactly once; a second
//! write means the link is being replayed and the caller must refuse it.
//! Owner-only (no `PERMISSIONS`, so tenant sessions are denied), like the rest
//! of the control-plane tables. Built on surql-rs `crud`; no raw SurrealQL.

use chrono::{DateTime, Utc};
use serde_json::json;

use surql::query::crud::{create_record, delete_records};
use surql::types::operators::lt;

use antumbra_core::Result;

use crate::error::map;
use crate::store::Store;

const TABLE: &str = "magic_link_use";

/// Mark a link id consumed. `true` = first use (proceed); `false` = already
/// consumed (the link is being replayed; refuse it). Atomic: a keyed `CREATE`
/// errors on a duplicate id, so two concurrent clicks cannot both win.
pub async fn consume(store: &Store, jti: &str, expires_at: DateTime<Utc>) -> Result<bool> {
    let row = json!({
        "id": jti.replace([':', '/', '\\', '|', ' '], "_"),
        "jti": jti,
        "expires_at": expires_at.to_rfc3339(),
        "used_at": Utc::now().to_rfc3339(),
    });
    match create_record(store.client(), TABLE, row).await {
        Ok(_) => Ok(true),
        Err(e) if e.to_string().contains("already exists") => Ok(false),
        Err(e) => Err(map(e)),
    }
}

/// Drop ledger rows whose link has itself expired: the token no longer
/// verifies, so the replay guard is dead weight. Timestamps are UTC RFC3339,
/// so the lexicographic `<` matches chronological order (the same idiom as
/// [`crate::repo::memory::purge`]). Call opportunistically; the ledger then
/// never outgrows the magic-link TTL window.
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
    use crate::schema::EMBED_DIM;

    #[tokio::test]
    async fn a_link_id_consumes_exactly_once() {
        let s = Store::connect_memory(EMBED_DIM).await.unwrap();
        let exp = Utc::now() + chrono::Duration::minutes(15);
        assert!(consume(&s, "jti-1", exp).await.unwrap(), "first use wins");
        assert!(!consume(&s, "jti-1", exp).await.unwrap(), "replay refused");
        // A different id is unaffected.
        assert!(consume(&s, "jti-2", exp).await.unwrap());
    }

    #[tokio::test]
    async fn sweeping_drops_only_expired_entries() {
        let s = Store::connect_memory(EMBED_DIM).await.unwrap();
        let now = Utc::now();
        assert!(consume(&s, "dead", now - chrono::Duration::minutes(1))
            .await
            .unwrap());
        assert!(consume(&s, "live", now + chrono::Duration::minutes(15))
            .await
            .unwrap());
        sweep_expired(&s, now).await.unwrap();
        // The expired row is gone (its id consumes afresh); the live row holds.
        assert!(consume(&s, "dead", now + chrono::Duration::minutes(15))
            .await
            .unwrap());
        assert!(!consume(&s, "live", now + chrono::Duration::minutes(15))
            .await
            .unwrap());
    }
}
