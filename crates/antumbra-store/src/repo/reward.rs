//! Reward-signal repository (the critic for credit assignment). `RewardSignal` has no reserved `id`
//! field, so it persists directly (no DTO). Source-tagged and auditable.

use surql::query::builder::Query;
use surql::query::crud::{create_records, query_records};
use surql::types::operators::eq;

use antumbra_core::{Result, RewardSignal, RunId};

use crate::error::map;
use crate::store::Store;

const TABLE: &str = "reward_signal";

/// Append a batch of signals in one round-trip (one row each, source-tagged).
pub async fn insert_many(store: &Store, signals: &[RewardSignal]) -> Result<()> {
    if signals.is_empty() {
        return Ok(());
    }
    let rows = signals
        .iter()
        .map(serde_json::to_value)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    create_records(store.client(), TABLE, rows)
        .await
        .map_err(map)?;
    Ok(())
}

/// Every signal recorded for a run, in insertion order.
pub async fn list_by_run(store: &Store, run_id: &RunId) -> Result<Vec<RewardSignal>> {
    let query = Query::new()
        .select(None)
        .from_table(TABLE)
        .map_err(map)?
        .where_(eq("run_id", run_id.as_str()));
    query_records(store.client(), &query).await.map_err(map)
}
