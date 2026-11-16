//! A workspace's memory knobs ([`MemoryPolicy`]): one row per tenant, which the
//! server reads on every recall and an operator sets from the console while it
//! runs. A workspace with no row runs on the defaults.
//!
//! Readable by the workspace's own sessions, so the server can read it on the
//! caller's connection; written only by the owner (an operator's console
//! action), so no agent's token can retune the workspace. Built on surql-rs
//! builders; no raw SurrealQL.

use surql::query::builder::Query;
use surql::query::crud::first;
use surql::types::operators::eq;

use antumbra_core::{MemoryPolicy, Result, TenantId};

use crate::error::map;
use crate::store::Store;

const TABLE: &str = crate::tables::MEMORY_POLICY;

#[derive(serde::Serialize, serde::Deserialize)]
struct PolicyRow {
    tenant_id: String,
    #[serde(default)]
    policy: MemoryPolicy,
}

fn key(tenant: &TenantId) -> String {
    tenant.as_str().replace([':', '/', '\\', '|', ' '], "_")
}

/// The workspace's knobs, or the defaults when none were ever set. Always
/// clamped into range.
pub async fn get(store: &Store, tenant: &TenantId) -> Result<MemoryPolicy> {
    let query = Query::new()
        .select(None)
        .from_table(TABLE)
        .map_err(map)?
        .where_(eq("tenant_id", tenant.as_str()));
    let row: Option<PolicyRow> = first(store.client(), &query).await.map_err(map)?;
    Ok(row.map(|r| r.policy).unwrap_or_default().clamped())
}

/// Replace the workspace's knobs, clamped into range. Needs the owner's
/// connection.
pub async fn set(store: &Store, tenant: &TenantId, policy: &MemoryPolicy) -> Result<MemoryPolicy> {
    let policy = policy.clone().clamped();
    let row = PolicyRow {
        tenant_id: tenant.as_str().to_string(),
        policy: policy.clone(),
    };
    store.put(TABLE, key(tenant).as_str(), &row).await?;
    Ok(policy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::mem_store;

    #[tokio::test]
    async fn a_workspace_runs_on_the_defaults_until_its_knobs_are_set() -> Result<()> {
        let s = mem_store().await?;
        let ws = TenantId::new("ws:a");
        assert_eq!(get(&s, &ws).await?, MemoryPolicy::default());

        let turned = MemoryPolicy {
            bank_half_life_days: 30.0,
            strength_weight: 7.0,
            ..MemoryPolicy::default()
        };
        let stored = set(&s, &ws, &turned).await?;
        assert_eq!(stored.strength_weight, 1.0, "stored clamped");
        let back = get(&s, &ws).await?;
        assert_eq!(back.bank_half_life_days, 30.0);
        assert_eq!(back, stored);
        // Another workspace keeps its own.
        assert_eq!(
            get(&s, &TenantId::new("ws:b")).await?,
            MemoryPolicy::default()
        );
        Ok(())
    }
}
