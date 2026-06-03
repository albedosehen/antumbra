//! Evaluation-run repository — one row per measured run. ADR-0007.
//! `EvaluationRun` has no reserved `id` field, so it persists directly.

use surql::query::builder::Query;
use surql::query::crud::{create_record, first, query_records};
use surql::types::operators::eq;

use antumbra_core::{EvaluationRun, Result, SubjectKind};

use crate::error::map;
use crate::store::Store;

const TABLE: &str = "evaluation_run";

/// Record a measured run.
pub async fn insert(store: &Store, run: &EvaluationRun) -> Result<()> {
    create_record(store.client(), TABLE, serde_json::to_value(run)?)
        .await
        .map_err(map)?;
    Ok(())
}

/// Every run for a subject.
pub async fn list_for_subject(
    store: &Store,
    kind: SubjectKind,
    subject_id: &str,
) -> Result<Vec<EvaluationRun>> {
    let query = Query::new()
        .select(None)
        .from_table(TABLE)
        .map_err(map)?
        .where_(eq("subject_kind", kind.as_str()))
        .where_(eq("subject_id", subject_id));
    query_records(store.client(), &query).await.map_err(map)
}

/// The most recent run for a subject (by `created_at`).
pub async fn latest_for_subject(
    store: &Store,
    kind: SubjectKind,
    subject_id: &str,
) -> Result<Option<EvaluationRun>> {
    let query = Query::new()
        .select(None)
        .from_table(TABLE)
        .map_err(map)?
        .where_(eq("subject_kind", kind.as_str()))
        .where_(eq("subject_id", subject_id))
        .order_by("created_at", "DESC")
        .map_err(map)?;
    first(store.client(), &query).await.map_err(map)
}
