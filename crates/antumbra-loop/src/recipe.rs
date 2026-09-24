//! What the loop records of the recipe each shadow trained under (ADR-0022
//! S-1): the ledger the recipe search reads, one row per shadow.
//!
//! Only the trainer's echo is recorded. A row naming the recipe the loop asked
//! for, when the trainer used another or would not say, would describe a run
//! that did not happen, and a search ranked on such rows would be tuning
//! settings nobody trained under.

use chrono::Utc;

use antumbra_core::ports::TrainOutcome;
use antumbra_core::slice::Holdout;
use antumbra_core::{Generation, RecipeRecord, Result, RunId, ShadowId, TrainingRecipe};
use antumbra_store::repo::recipe;

use crate::GenerationLoop;

/// One shadow's run, as its recipe row needs it.
pub(crate) struct Trained<'a> {
    pub run_id: &'a RunId,
    pub generation: Generation,
    pub shadow: &'a ShadowId,
    /// The recipe the shadow was asked to train under, `None` for the
    /// trainer's own.
    pub asked: Option<TrainingRecipe>,
    /// The row this one descends from.
    pub parent: Option<ShadowId>,
    pub holdout: Option<&'a Holdout>,
    pub outcome: &'a TrainOutcome,
}

impl GenerationLoop<'_> {
    /// Record the recipe a shadow trained under, as its trainer reported it,
    /// and return it.
    ///
    /// A trainer that reports none leaves no row. One that trained under
    /// something other than what was asked is recorded as what it did, and the
    /// difference is said. The partition seed is kept only when the trainer
    /// confirmed the holdout: then the fitness is over that split's visible
    /// tasks alone, and a ranking must never set it against fitness read under
    /// another split, or under none.
    pub(crate) async fn record_recipe(&self, run: Trained<'_>) -> Result<Option<TrainingRecipe>> {
        let Some(ran) = run.outcome.recipe else {
            if run.asked.is_some() {
                eprintln!(
                    "recipe: the trainer did not say which recipe {} trained under, so it leaves \
                     no recipe row",
                    run.shadow
                );
            }
            return Ok(None);
        };
        if let Some(asked) = run.asked.filter(|asked| *asked != ran) {
            eprintln!(
                "recipe: {} asked for {asked:?} but trained under {ran:?}; recorded as trained",
                run.shadow
            );
        }
        let partition_seed = run
            .holdout
            .filter(|asked| run.outcome.holdout.as_ref() == Some(*asked))
            .map(|asked| asked.partition.seed);
        recipe::upsert(
            self.store,
            &RecipeRecord {
                shadow: run.shadow.clone(),
                run_id: run.run_id.clone(),
                generation: run.generation,
                recipe: ran,
                parent: run.parent,
                partition_seed,
                fitness_mean: run.outcome.final_fitness,
                fitness_variance: None,
                evaluations: 1,
                created_at: Utc::now(),
            },
        )
        .await?;
        Ok(Some(ran))
    }
}
