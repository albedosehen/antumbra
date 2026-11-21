//! What a person decided about bringing each of their memories to a workspace
//! ([`antumbra_core::elevation`]).
//!
//! One row per (tenant, user, memory), the user's own to write (the table
//! carries a device row's permissions). It is kept on the person's own
//! server: what they declined is theirs to know and nobody else's.

use std::collections::{BTreeMap, HashMap};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;

use antumbra_core::elevation::Choice;
use antumbra_core::{Result, TenantId, UserId};

use crate::dto::parse_dt;
use crate::store::Store;

const TABLE: &str = crate::tables::MEMORY_CHOICE;

#[derive(Serialize, Deserialize)]
struct Row {
    tenant_id: String,
    user: String,
    memory: String,
    choice: String,
    decided_at: String,
}

/// A record key for (user, memory): what a key can hold of each, with their
/// lengths so two pairs never share one.
fn key(user: &UserId, memory: &str) -> String {
    let plain = |s: &str| -> String {
        s.chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect()
    };
    format!(
        "{}_{}_{}_{}",
        user.as_str().len(),
        plain(user.as_str()),
        memory.len(),
        plain(memory)
    )
}

/// Record one decision for each of `memories`, replacing what was decided
/// before.
pub async fn decide(
    store: &Store,
    tenant: &TenantId,
    user: &UserId,
    memories: &[String],
    choice: Choice,
    now: DateTime<Utc>,
) -> Result<()> {
    for memory in memories {
        store
            .put(
                TABLE,
                &key(user, memory),
                &Row {
                    tenant_id: tenant.as_str().to_string(),
                    user: user.as_str().to_string(),
                    memory: memory.clone(),
                    choice: choice.as_str().to_string(),
                    decided_at: now.to_rfc3339(),
                },
            )
            .await?;
    }
    Ok(())
}

/// Forget what was decided about each of `memories`: they are undecided
/// again.
pub async fn clear(
    store: &Store,
    tenant: &TenantId,
    user: &UserId,
    memories: &[String],
) -> Result<()> {
    if memories.is_empty() {
        return Ok(());
    }
    store
        .query_rows::<serde_json::Value>(
            &format!(
                "DELETE {TABLE} WHERE tenant_id = $tenant AND user = $user AND memory IN $memories"
            ),
            BTreeMap::from([
                ("tenant".to_string(), json!(tenant.as_str())),
                ("user".to_string(), json!(user.as_str())),
                ("memories".to_string(), json!(memories)),
            ]),
        )
        .await?;
    Ok(())
}

/// Everything a person has decided, by memory.
pub async fn of(
    store: &Store,
    tenant: &TenantId,
    user: &UserId,
) -> Result<HashMap<String, (Choice, DateTime<Utc>)>> {
    let rows: Vec<Row> = store
        .query_rows(
            &format!("SELECT tenant_id, user, memory, choice, decided_at FROM {TABLE} WHERE tenant_id = $tenant AND user = $user"),
            BTreeMap::from([
                ("tenant".to_string(), json!(tenant.as_str())),
                ("user".to_string(), json!(user.as_str())),
            ]),
        )
        .await?;
    let mut all = HashMap::with_capacity(rows.len());
    for row in rows {
        // A choice this version does not know is left out, not an error.
        if let Ok(choice) = row.choice.parse::<Choice>() {
            all.insert(row.memory, (choice, parse_dt(&row.decided_at)?));
        }
    }
    Ok(all)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::mem_store;

    #[tokio::test]
    async fn a_decision_is_kept_replaced_and_cleared_for_its_own_user() -> Result<()> {
        let s = mem_store().await?;
        let t = TenantId::new("ws:t");
        let (dana, eli) = (UserId::new("user:dana"), UserId::new("user:eli"));
        let ids = |names: &[&str]| names.iter().map(|n| n.to_string()).collect::<Vec<_>>();
        let now = Utc::now();

        decide(
            &s,
            &t,
            &dana,
            &ids(&["memory:a", "memory:b", "memory:c"]),
            Choice::Chosen,
            now,
        )
        .await?;
        decide(&s, &t, &dana, &ids(&["memory:b"]), Choice::Never, now).await?;
        decide(&s, &t, &eli, &ids(&["memory:a"]), Choice::Declined, now).await?;
        let hers = of(&s, &t, &dana).await?;
        assert_eq!(hers.len(), 3);
        assert_eq!(hers.get("memory:a").map(|c| c.0), Some(Choice::Chosen));
        assert_eq!(hers.get("memory:b").map(|c| c.0), Some(Choice::Never));

        clear(&s, &t, &dana, &ids(&["memory:a", "memory:zzz"])).await?;
        let hers = of(&s, &t, &dana).await?;
        assert_eq!(hers.len(), 2);
        assert!(!hers.contains_key("memory:a"));
        // Another person's decision about the same memory is their own.
        assert_eq!(
            of(&s, &t, &eli).await?.get("memory:a").map(|c| c.0),
            Some(Choice::Declined)
        );
        assert!(of(&s, &TenantId::new("ws:other"), &dana).await?.is_empty());
        Ok(())
    }
}
