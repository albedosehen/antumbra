//! Shadow repository (the penumbra): shadow models hold the plasticity.
//!
//! Shadows mutate through their lifecycle, so they are addressed by a stable
//! record id and written with `crud::upsert_record`. No raw SurrealQL.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use surql::query::crud::{get_record, upsert_record};
use surql::types::operators::eq;
use surql::types::RecordID;

use antumbra_core::{ExpertId, Generation, Result, Shadow, ShadowId, ShadowStatus};

use crate::dto::parse_dt;
use crate::error::map;
use crate::store::Store;

const TABLE: &str = "shadow";

#[derive(Serialize, Deserialize)]
struct ShadowRow {
    key: String,
    #[serde(default)]
    parent_expert: Option<String>,
    #[serde(default)]
    adapter_uri: Option<String>,
    status: ShadowStatus,
    generation: u32,
    #[serde(default)]
    reward_curve: Vec<f32>,
    created_at: String,
}

impl ShadowRow {
    fn from_domain(s: &Shadow) -> Self {
        ShadowRow {
            key: s.id.as_str().to_string(),
            parent_expert: s.parent_expert.as_ref().map(|e| e.as_str().to_string()),
            adapter_uri: s.adapter_uri.clone(),
            status: s.status,
            generation: s.generation.0,
            reward_curve: s.reward_curve.clone(),
            created_at: s.created_at.to_rfc3339(),
        }
    }

    fn into_domain(self) -> Result<Shadow> {
        Ok(Shadow {
            id: ShadowId::new(self.key),
            parent_expert: self.parent_expert.map(ExpertId::new),
            adapter_uri: self.adapter_uri,
            status: self.status,
            generation: Generation(self.generation),
            reward_curve: self.reward_curve,
            created_at: parse_dt(&self.created_at)?,
        })
    }
}

/// Insert or replace a shadow (status transitions re-upsert the full row).
pub async fn upsert(store: &Store, shadow: &Shadow) -> Result<()> {
    let id = RecordID::<()>::new(TABLE, shadow.id.as_str()).map_err(map)?;
    let data: Value = serde_json::to_value(ShadowRow::from_domain(shadow))?;
    upsert_record(store.client(), &id, data)
        .await
        .map_err(map)?;
    Ok(())
}

/// Fetch one shadow by id.
pub async fn get(store: &Store, id: &ShadowId) -> Result<Option<Shadow>> {
    let rid = RecordID::<()>::new(TABLE, id.as_str()).map_err(map)?;
    match get_record(store.client(), &rid).await.map_err(map)? {
        Some(value) => Ok(Some(
            serde_json::from_value::<ShadowRow>(value)?.into_domain()?,
        )),
        None => Ok(None),
    }
}

/// All shadows currently in a given lifecycle state.
pub async fn list_by_status(store: &Store, status: ShadowStatus) -> Result<Vec<Shadow>> {
    // Paged by id (see `Store::read_paged`); the existing status filter is
    // preserved. Order is unspecified, so no re-sort.
    let filter = eq("status", status.as_str());
    let rows: Vec<ShadowRow> = store.read_paged(TABLE, None, Some(&filter)).await?;
    rows.into_iter().map(ShadowRow::into_domain).collect()
}

/// Every shadow ever spawned (the penumbra's full lineage, in-flight and retired).
/// The console reads this to show recent training activity.
pub async fn list(store: &Store) -> Result<Vec<Shadow>> {
    // Paged by id (see `Store::read_paged`): the full lineage grows unbounded
    // over the loop's lifetime. Order is unspecified, so no re-sort.
    let rows: Vec<ShadowRow> = store.read_paged(TABLE, None, None).await?;
    rows.into_iter().map(ShadowRow::into_domain).collect()
}
