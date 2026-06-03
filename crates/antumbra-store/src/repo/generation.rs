//! Generation-head repository — the durable loop checkpoint. ADR-0008.
//!
//! A singleton-per-run row, addressed by a stable record id (`RecordID` auto-
//! escapes the run key), upserted via surql-rs's `crud` helpers. No raw
//! SurrealQL.

use surql::query::crud::{get_record, upsert_record};
use surql::types::RecordID;

use antumbra_core::generational::GenerationHead;
use antumbra_core::{Result, RunId};

use crate::error::map;
use crate::store::Store;

const TABLE: &str = "generation_head";

fn record_id(run_id: &RunId) -> Result<RecordID> {
    RecordID::<()>::new(TABLE, run_id.as_str()).map_err(map)
}

/// Upsert the loop head; this single write IS the checkpoint.
pub async fn save_head(store: &Store, head: &GenerationHead) -> Result<()> {
    let id = record_id(&head.run_id)?;
    let data = serde_json::to_value(head)?;
    upsert_record(store.client(), &id, data)
        .await
        .map_err(map)?;
    Ok(())
}

/// Load the loop head for a run, if one has been checkpointed.
pub async fn load_head(store: &Store, run_id: &RunId) -> Result<Option<GenerationHead>> {
    let id = record_id(run_id)?;
    match get_record(store.client(), &id).await.map_err(map)? {
        Some(value) => Ok(Some(serde_json::from_value(value)?)),
        None => Ok(None),
    }
}
