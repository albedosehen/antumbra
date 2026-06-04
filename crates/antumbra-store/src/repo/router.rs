//! Learned-router repository (ADR-0009). A singleton per store: the gate's
//! trained metric + centroids, retrained whenever the population changes. No
//! raw SurrealQL — surql-rs `crud` helpers only.

use surql::query::crud::{get_record, upsert_record};
use surql::types::RecordID;

use antumbra_core::{LearnedRouter, Result};

use crate::error::map;
use crate::store::Store;

const TABLE: &str = "learned_router";
const KEY: &str = "active";

fn record_id() -> Result<RecordID> {
    RecordID::<()>::new(TABLE, KEY).map_err(map)
}

/// Persist (replace) the active learned router for this population.
pub async fn save(store: &Store, router: &LearnedRouter) -> Result<()> {
    let id = record_id()?;
    let data = serde_json::to_value(router)?;
    upsert_record(store.client(), &id, data).await.map_err(map)?;
    Ok(())
}

/// Load the active learned router, if one has been trained. Returns `None`
/// (not an error) before the first `gate-train`, when the table does not exist.
pub async fn load(store: &Store) -> Result<Option<LearnedRouter>> {
    let id = record_id()?;
    match get_record(store.client(), &id).await {
        Ok(Some(value)) => Ok(Some(serde_json::from_value(value)?)),
        Ok(None) => Ok(None),
        Err(e) if e.to_string().contains("does not exist") => Ok(None),
        Err(e) => Err(map(e)),
    }
}
