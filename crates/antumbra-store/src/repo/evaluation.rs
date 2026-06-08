//! Evaluation-run repository: one row per measured run.
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

/// The most recent runs across all subjects, newest first: the owner/console
/// view (parallels [`crate::repo::memory::all_unscoped`]).
pub async fn recent_unscoped(store: &Store) -> Result<Vec<EvaluationRun>> {
    let query = Query::new()
        .select(None)
        .from_table(TABLE)
        .map_err(map)?
        .order_by("created_at", "DESC")
        .map_err(map)?;
    query_records(store.client(), &query).await.map_err(map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::EMBED_DIM;
    use antumbra_core::evaluation::EvalStatus;
    use antumbra_core::RunId;
    use chrono::Utc;

    fn make(id: &str, status: EvalStatus, at: chrono::DateTime<Utc>) -> EvaluationRun {
        EvaluationRun {
            run_id: RunId::new(id),
            subject_kind: SubjectKind::Expert,
            subject_id: "expert:1".into(),
            corpus_task_id: "task:1".into(),
            status,
            metrics: None,
            regression_fingerprint: None,
            created_at: at,
        }
    }

    #[tokio::test]
    async fn recent_unscoped_reads_all_newest_first() {
        let s = Store::connect_memory(EMBED_DIM).await.unwrap();
        let now = Utc::now();
        insert(
            &s,
            &make(
                "run:old",
                EvalStatus::Success,
                now - chrono::Duration::seconds(60),
            ),
        )
        .await
        .unwrap();
        insert(&s, &make("run:new", EvalStatus::Failure, now))
            .await
            .unwrap();

        let runs = recent_unscoped(&s).await.unwrap();
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].run_id.as_str(), "run:new", "newest first");
        assert_eq!(runs[0].status, EvalStatus::Failure);
    }
}
