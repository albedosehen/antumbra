//! Reward-signal repository (the critic for credit assignment). `RewardSignal` has no reserved `id`
//! field, so it persists directly (no DTO). Source-tagged and auditable.

use surql::query::builder::Query;
use surql::query::crud::{create_record, query_records};
use surql::types::operators::eq;

use antumbra_core::{Result, RewardSignal, RunId};

use crate::error::map;
use crate::store::Store;

const TABLE: &str = "reward_signal";

/// Append a batch of signals (one row each, source-tagged).
pub async fn insert_many(store: &Store, signals: &[RewardSignal]) -> Result<()> {
    for signal in signals {
        create_record(store.client(), TABLE, serde_json::to_value(signal)?)
            .await
            .map_err(map)?;
    }
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
