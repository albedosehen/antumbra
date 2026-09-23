//! What the loop records of the recipe each generation trained under
//! (ADR-0022 S-1): the ledger the recipe search reads, one row per shadow.
//!
//! Only the trainer's echo is recorded. A row naming the recipe the loop asked
//! for, when the trainer used another or would not say, would describe a run
//! that did not happen, and a search ranked on such rows would be tuning
//! settings nobody trained under.

use chrono::Utc;

use antumbra_core::ports::TrainOutcome;
use antumbra_core::slice::Holdout;
use antumbra_core::{Generation, RecipeRecord, Result, RunId, TrainingRecipe};
use antumbra_store::repo::recipe;

use crate::{shadow_id, GenerationLoop};

impl GenerationLoop<'_> {
    /// Record the recipe this generation trained under, as its trainer
    /// reported it, and return it.
    ///
    /// A trainer that reports none leaves no row. One that trained under
    /// something other than what was asked is recorded as what it did, and the
    /// difference is said. The row descends from the previous generation's, so
    /// the lineage of a recipe can be walked back through the run. The
    /// partition seed is kept only when the trainer confirmed the holdout: then
    /// the fitness is over that split's visible tasks alone, and a ranking must
    /// never set it against fitness read under another split, or under none.
    pub(crate) async fn record_recipe(
        &self,
        run_id: &RunId,
        generation: Generation,
        holdout: Option<&Holdout>,
        outcome: &TrainOutcome,
    ) -> Result<Option<TrainingRecipe>> {
        let Some(ran) = outcome.recipe else {
            if self.cfg.recipe.is_some() {
                eprintln!(
                    "recipe: the trainer did not say which recipe generation {} trained under, so \
                     it leaves no recipe row",
                    generation.0
                );
            }
            return Ok(None);
        };
        if let Some(asked) = self.cfg.recipe.filter(|asked| *asked != ran) {
            eprintln!(
                "recipe: generation {} asked for {asked:?} but trained under {ran:?}; recorded as \
                 trained",
                generation.0
            );
        }
        let parent = match generation.0.checked_sub(1) {
            Some(previous) => recipe::get(self.store, &shadow_id(run_id, Generation(previous)))
                .await?
                .map(|row| row.shadow),
            None => None,
        };
        let partition_seed = holdout
            .filter(|asked| outcome.holdout.as_ref() == Some(*asked))
            .map(|asked| asked.partition.seed);
        recipe::upsert(
            self.store,
            &RecipeRecord {
                shadow: shadow_id(run_id, generation),
                run_id: run_id.clone(),
                generation,
                recipe: ran,
                parent,
                partition_seed,
                fitness_mean: outcome.final_fitness,
                fitness_variance: None,
                evaluations: 1,
                created_at: Utc::now(),
            },
        )
        .await?;
        Ok(Some(ran))
    }
}
