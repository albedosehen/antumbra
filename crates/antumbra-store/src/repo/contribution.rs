//! Contribution repository (ADR-0022 S-5): each expert's leave-one-out
//! contribution, one row per expert per measured generation of a run, keyed so
//! a generation measured again after a crash replaces its rows rather than
//! adding second ones. The history the loop's detectors read.
//!
//! Kept apart from `evaluation_run`, whose latest row for an expert is the
//! freeze baseline the byte-identity tripwire checks against.

use surql::query::builder::Query;
use surql::query::crud::{delete_record, query_records, upsert_record};
use surql::types::operators::eq;
use surql::types::RecordID;

use antumbra_core::{BaselineRecord, ContributionRecord, ExpertId, Result, RoutingOutcome, RunId};

use crate::error::map;
use crate::store::Store;

const TABLE: &str = "contribution";
const BASELINE: &str = "population_baseline";
const OUTCOME: &str = "routing_outcome";

fn key(record: &ContributionRecord) -> String {
    format!(
        "{}@{}@g{}",
        record.expert, record.run_id, record.generation.0
    )
}

/// Record one expert's contribution in one generation, replacing what the same
/// generation recorded for it before.
pub async fn upsert(store: &Store, record: &ContributionRecord) -> Result<()> {
    let id = RecordID::<()>::new(TABLE, key(record).as_str()).map_err(map)?;
    upsert_record(store.client(), &id, serde_json::to_value(record)?)
        .await
        .map_err(map)?;
    Ok(())
}

/// Every contribution recorded for an expert, oldest generation first.
pub async fn history(store: &Store, expert: &ExpertId) -> Result<Vec<ContributionRecord>> {
    let query = Query::new()
        .select(None)
        .from_table(TABLE)
        .map_err(map)?
        .where_(eq("expert", expert.as_str()))
        .order_by("generation", "ASC")
        .map_err(map)?;
    query_records(store.client(), &query).await.map_err(map)
}

/// Every contribution one run recorded, oldest generation first.
pub async fn list_for_run(store: &Store, run_id: &RunId) -> Result<Vec<ContributionRecord>> {
    let query = Query::new()
        .select(None)
        .from_table(TABLE)
        .map_err(map)?
        .where_(eq("run_id", run_id.as_str()))
        .order_by("generation", "ASC")
        .map_err(map)?;
    query_records(store.client(), &query).await.map_err(map)
}

/// Record a task's clear winner, replacing the one recorded for the same task
/// before: the latest measurement is the one that describes the population as
/// it now is.
pub async fn upsert_outcome(store: &Store, outcome: &RoutingOutcome) -> Result<()> {
    let id = RecordID::<()>::new(OUTCOME, outcome.task.as_str()).map_err(map)?;
    upsert_record(store.client(), &id, serde_json::to_value(outcome)?)
        .await
        .map_err(map)?;
    Ok(())
}

/// Forget a task's clear winner: the latest measurement found none.
pub async fn clear_outcome(store: &Store, task: &str) -> Result<()> {
    let id = RecordID::<()>::new(OUTCOME, task).map_err(map)?;
    delete_record(store.client(), &id).await.map_err(map)?;
    Ok(())
}

/// Every task's latest clear winner, by task id.
pub async fn outcomes(store: &Store) -> Result<Vec<RoutingOutcome>> {
    let query = Query::new()
        .select(None)
        .from_table(OUTCOME)
        .map_err(map)?
        .order_by("task", "ASC")
        .map_err(map)?;
    query_records(store.client(), &query).await.map_err(map)
}

/// Record a generation's population-against-best-expert comparison,
/// replacing what the same generation recorded before.
pub async fn upsert_baseline(store: &Store, record: &BaselineRecord) -> Result<()> {
    let key = format!("{}@g{}", record.run_id, record.generation.0);
    let id = RecordID::<()>::new(BASELINE, key.as_str()).map_err(map)?;
    upsert_record(store.client(), &id, serde_json::to_value(record)?)
        .await
        .map_err(map)?;
    Ok(())
}

/// Every comparison one run recorded, oldest generation first: the rolling
/// comparison the record asks to be reported.
pub async fn baselines_for_run(store: &Store, run_id: &RunId) -> Result<Vec<BaselineRecord>> {
    let query = Query::new()
        .select(None)
        .from_table(BASELINE)
        .map_err(map)?
        .where_(eq("run_id", run_id.as_str()))
        .order_by("generation", "ASC")
        .map_err(map)?;
    query_records(store.client(), &query).await.map_err(map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::EMBED_DIM;
    use antumbra_core::Generation;
    use chrono::Utc;

    fn record(expert: &str, generation: u32, with: f32) -> ContributionRecord {
        ContributionRecord {
            expert: ExpertId::new(expert),
            run_id: RunId::new("run:c"),
            generation: Generation(generation),
            routed: 3,
            tasks: 9,
            with: Some(with),
            without: Some(0.2),
            seeds: 2,
            affinity: Some(0.9),
            at: Utc::now(),
        }
    }

    #[tokio::test]
    async fn a_generation_measured_again_replaces_its_row() -> Result<()> {
        let s = Store::connect_memory(EMBED_DIM).await?;
        upsert(&s, &record("expert:a", 0, 0.5)).await?;
        upsert(&s, &record("expert:a", 1, 0.6)).await?;
        upsert(&s, &record("expert:a", 1, 0.7)).await?;
        upsert(&s, &record("expert:b", 1, 0.9)).await?;
        let a = history(&s, &ExpertId::new("expert:a")).await?;
        let seen: Vec<(u32, Option<f32>)> = a.iter().map(|r| (r.generation.0, r.with)).collect();
        assert_eq!(seen, [(0, Some(0.5)), (1, Some(0.7))]);
        assert_eq!(list_for_run(&s, &RunId::new("run:c")).await?.len(), 3);
        Ok(())
    }

    #[tokio::test]
    async fn a_baseline_is_kept_per_generation_and_replaced_when_measured_again() -> Result<()> {
        let s = Store::connect_memory(EMBED_DIM).await?;
        let at = |generation: u32, population: f32| BaselineRecord {
            run_id: RunId::new("run:b"),
            generation: Generation(generation),
            tasks: 8,
            population,
            best: Some(ExpertId::new("expert:a")),
            best_alone: Some(0.6),
            oracle: None,
            oracle_cross_fitted: false,
            seeds: 2,
            at: Utc::now(),
        };
        upsert_baseline(&s, &at(0, 0.7)).await?;
        upsert_baseline(&s, &at(1, 0.5)).await?;
        upsert_baseline(&s, &at(1, 0.65)).await?;
        let rolling = baselines_for_run(&s, &RunId::new("run:b")).await?;
        let deltas: Vec<Option<f32>> = rolling.iter().map(BaselineRecord::delta).collect();
        assert_eq!(rolling.len(), 2);
        assert!((deltas[0].unwrap_or(0.0) - 0.1).abs() < 1e-6);
        assert!((deltas[1].unwrap_or(0.0) - 0.05).abs() < 1e-6);
        Ok(())
    }
}
