//! Failure-boundary repository (the antumbra / inhibitory store): the counterfactual boundary of competence.
//!
//! Boundaries accrue and may have their confidence updated, so they are
//! addressed by a stable record id and upserted. Context KNN uses surql-rs's
//! vector-search builder.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use surql::query::builder::Query;
use surql::query::crud::{delete_record, query_records, upsert_record};
use surql::query::helpers::VectorDistanceType;
use surql::types::RecordID;

use antumbra_core::{BoundaryId, FailureBoundary, Generation, Grain, Result};

use crate::dto::parse_dt;
use crate::error::map;
use crate::store::Store;

const TABLE: &str = "failure_boundary";

#[derive(Serialize, Deserialize)]
struct BoundaryRow {
    key: String,
    behavior: String,
    fail_context: Value,
    #[serde(default)]
    near_ok_context: Option<Value>,
    #[serde(default)]
    governing_features: Vec<String>,
    #[serde(default)]
    grain: Option<Grain>,
    #[serde(default)]
    context_vec: Option<Vec<f32>>,
    #[serde(default)]
    ok_context_vec: Option<Vec<f32>>,
    #[serde(default)]
    confidence: f32,
    generation: u32,
    created_at: String,
}

impl BoundaryRow {
    fn from_domain(b: &FailureBoundary) -> Self {
        BoundaryRow {
            key: b.id.as_str().to_string(),
            behavior: b.behavior.clone(),
            fail_context: b.fail_context.clone(),
            near_ok_context: b.near_ok_context.clone(),
            governing_features: b.governing_features.clone(),
            grain: b.grain,
            context_vec: b.context_vec.clone(),
            ok_context_vec: b.ok_context_vec.clone(),
            confidence: b.confidence,
            generation: b.generation.0,
            created_at: b.created_at.to_rfc3339(),
        }
    }

    fn into_domain(self) -> Result<FailureBoundary> {
        Ok(FailureBoundary {
            id: BoundaryId::new(self.key),
            behavior: self.behavior,
            fail_context: self.fail_context,
            near_ok_context: self.near_ok_context,
            governing_features: self.governing_features,
            grain: self.grain,
            context_vec: self.context_vec,
            ok_context_vec: self.ok_context_vec,
            confidence: self.confidence,
            generation: Generation(self.generation),
            created_at: parse_dt(&self.created_at)?,
        })
    }
}

/// Insert or replace a boundary.
pub async fn upsert(store: &Store, boundary: &FailureBoundary) -> Result<()> {
    let id = RecordID::<()>::new(TABLE, boundary.id.as_str()).map_err(map)?;
    let data = serde_json::to_value(BoundaryRow::from_domain(boundary))?;
    upsert_record(store.client(), &id, data)
        .await
        .map_err(map)?;
    Ok(())
}

/// Remove a boundary, e.g. retired once a captured expert resolves its region.
pub async fn delete(store: &Store, id: &BoundaryId) -> Result<()> {
    let rid = RecordID::<()>::new(TABLE, id.as_str()).map_err(map)?;
    delete_record(store.client(), &rid).await.map_err(map)?;
    Ok(())
}

/// List all boundaries.
pub async fn list(store: &Store) -> Result<Vec<FailureBoundary>> {
    let query = Query::new().select(None).from_table(TABLE).map_err(map)?;
    let rows: Vec<BoundaryRow> = query_records(store.client(), &query).await.map_err(map)?;
    rows.into_iter().map(BoundaryRow::into_domain).collect()
}

/// The `k` boundaries whose context is nearest `query` (inhibitory-penalty
/// lookup via routing-as-retrieval), through surql-rs's vector-search builder.
pub async fn knn_by_context(
    store: &Store,
    query: &[f32],
    k: usize,
) -> Result<Vec<FailureBoundary>> {
    let vector: Vec<f64> = query.iter().map(|&x| f64::from(x)).collect();
    let q = Query::new()
        .select(None)
        .from_table(TABLE)
        .map_err(map)?
        .vector_search(
            "context_vec",
            vector,
            k as i64,
            VectorDistanceType::Cosine,
            None,
        )
        .map_err(map)?;
    let rows: Vec<BoundaryRow> = query_records(store.client(), &q).await.map_err(map)?;
    rows.into_iter().map(BoundaryRow::into_domain).collect()
}
