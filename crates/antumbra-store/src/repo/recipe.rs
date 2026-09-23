//! Recipe repository (ADR-0022 S-1): one row per run of a training recipe,
//! keyed by the shadow that ran it, so re-running a generation after a crash
//! replaces its row rather than adding a second.
//!
//! The rows are what makes the recipe search auditable and resumable: which
//! settings trained which shadow, what that shadow scored and on how many
//! evaluations, and which recipe it descends from. No raw SurrealQL.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use surql::query::builder::Query;
use surql::query::crud::{get_record, query_records, upsert_record};
use surql::types::operators::eq;
use surql::types::RecordID;

use antumbra_core::{Generation, RecipeRecord, Result, RunId, ShadowId, TrainingRecipe};

use crate::dto::parse_dt;
use crate::error::map;
use crate::store::Store;

const TABLE: &str = "recipe";

#[derive(Serialize, Deserialize)]
struct RecipeRow {
    key: String,
    run_id: String,
    generation: u32,
    recipe: TrainingRecipe,
    #[serde(default)]
    parent: Option<String>,
    #[serde(default)]
    partition_seed: Option<u64>,
    fitness_mean: f32,
    #[serde(default)]
    fitness_variance: Option<f32>,
    evaluations: u32,
    created_at: String,
}

impl RecipeRow {
    fn from_domain(r: &RecipeRecord) -> Self {
        RecipeRow {
            key: r.shadow.as_str().to_string(),
            run_id: r.run_id.as_str().to_string(),
            generation: r.generation.0,
            recipe: r.recipe,
            parent: r.parent.as_ref().map(|p| p.as_str().to_string()),
            partition_seed: r.partition_seed,
            fitness_mean: r.fitness_mean,
            fitness_variance: r.fitness_variance,
            evaluations: r.evaluations,
            created_at: r.created_at.to_rfc3339(),
        }
    }

    fn into_domain(self) -> Result<RecipeRecord> {
        Ok(RecipeRecord {
            shadow: ShadowId::new(self.key),
            run_id: RunId::new(self.run_id),
            generation: Generation(self.generation),
            recipe: self.recipe,
            parent: self.parent.map(ShadowId::new),
            partition_seed: self.partition_seed,
            fitness_mean: self.fitness_mean,
            fitness_variance: self.fitness_variance,
            evaluations: self.evaluations,
            created_at: parse_dt(&self.created_at)?,
        })
    }
}

/// Record a run of a recipe, replacing any row the same shadow left before.
pub async fn upsert(store: &Store, record: &RecipeRecord) -> Result<()> {
    let id = RecordID::<()>::new(TABLE, record.shadow.as_str()).map_err(map)?;
    let data: Value = serde_json::to_value(RecipeRow::from_domain(record))?;
    upsert_record(store.client(), &id, data)
        .await
        .map_err(map)?;
    Ok(())
}

/// The row a shadow left, if it trained under a recorded recipe.
pub async fn get(store: &Store, shadow: &ShadowId) -> Result<Option<RecipeRecord>> {
    let rid = RecordID::<()>::new(TABLE, shadow.as_str()).map_err(map)?;
    match get_record(store.client(), &rid).await.map_err(map)? {
        Some(value) => Ok(Some(
            serde_json::from_value::<RecipeRow>(value)?.into_domain()?,
        )),
        None => Ok(None),
    }
}

/// Every recipe one run trained under, in generation order: the search's
/// history, as a resumed run reads it.
pub async fn list_for_run(store: &Store, run_id: &RunId) -> Result<Vec<RecipeRecord>> {
    let query = Query::new()
        .select(None)
        .from_table(TABLE)
        .map_err(map)?
        .where_(eq("run_id", run_id.as_str()))
        .order_by("generation", "ASC")
        .map_err(map)?;
    let rows: Vec<RecipeRow> = query_records(store.client(), &query).await.map_err(map)?;
    rows.into_iter().map(RecipeRow::into_domain).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::EMBED_DIM;
    use chrono::Utc;

    fn record(run: &str, generation: u32, fitness: f32) -> RecipeRecord {
        RecipeRecord {
            shadow: ShadowId::new(format!("{run}:g{generation}")),
            run_id: RunId::new(run),
            generation: Generation(generation),
            recipe: TrainingRecipe {
                learning_rate: 1e-4,
                batch_size: 1,
                kl_beta: 0.04,
            },
            parent: generation
                .checked_sub(1)
                .map(|g| ShadowId::new(format!("{run}:g{g}"))),
            partition_seed: Some(u64::MAX / 3),
            fitness_mean: fitness,
            fitness_variance: None,
            evaluations: 1,
            created_at: Utc::now(),
        }
    }

    #[tokio::test]
    async fn a_row_round_trips_by_its_shadow() -> Result<()> {
        let s = Store::connect_memory(EMBED_DIM).await?;
        let r = record("run:a", 1, 0.5);
        upsert(&s, &r).await?;
        let back = get(&s, &r.shadow).await?.expect("stored");
        assert_eq!(
            (
                back.generation,
                back.parent,
                back.partition_seed,
                back.recipe
            ),
            (r.generation, r.parent, r.partition_seed, r.recipe)
        );
        assert_eq!(back.fitness_mean, 0.5);
        assert!(get(&s, &ShadowId::new("run:a:g9")).await?.is_none());
        Ok(())
    }

    /// A generation re-run after a crash replaces its row: the history holds
    /// one row per shadow, however many times it was written.
    #[tokio::test]
    async fn rewriting_a_shadows_row_replaces_it() -> Result<()> {
        let s = Store::connect_memory(EMBED_DIM).await?;
        upsert(&s, &record("run:a", 0, 0.2)).await?;
        upsert(&s, &record("run:a", 0, 0.3)).await?;
        let rows = list_for_run(&s, &RunId::new("run:a")).await?;
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].fitness_mean, 0.3);
        Ok(())
    }

    #[tokio::test]
    async fn a_runs_history_reads_in_generation_order_and_alone() -> Result<()> {
        let s = Store::connect_memory(EMBED_DIM).await?;
        for (run, generation) in [("run:a", 2), ("run:a", 0), ("run:b", 1), ("run:a", 1)] {
            upsert(&s, &record(run, generation, 0.1)).await?;
        }
        let rows = list_for_run(&s, &RunId::new("run:a")).await?;
        let generations: Vec<u32> = rows.iter().map(|r| r.generation.0).collect();
        assert_eq!(generations, [0, 1, 2]);
        Ok(())
    }
}
