//! The training recipe: the settings a shadow trains under that
//! are worth varying between shadows, and the row each run of one leaves.
//!
//! The record splits `LoopConfig` into a searched recipe and the unsearched
//! policy that governs the search, and it names the axes: learning rate on a
//! log scale, the anti-collapse reward weights, and batch size. Everything else
//! about training stays fixed for a run. Rank most of all: exploitation copies
//! weights, and LoRA weights are not transferable across rank, so a rank
//! change paired with a copy corrupts it silently. Alpha is fixed at twice the
//! rank and every linear layer is targeted, both settled by Biderman et al.
//! (2024), so neither is here either.
//!
//! Every recipe that runs becomes a [`RecipeRecord`]: the generation that ran
//! it, what it was measured on, its fitness and how many evaluations that
//! fitness rests on, and the recipe it descends from, so the search is
//! auditable and resumable like everything else in the loop.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::{AntumbraError, Result};
use crate::ids::{Generation, RunId, ShadowId};

/// The searched settings a shadow trains under.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TrainingRecipe {
    /// The optimizer's step size. Once it is tuned, LoRA variants land within
    /// a point or two of each other, which is why it leads the search.
    pub learning_rate: f64,
    /// Verified winners per optimizer step. 1 steps on each in turn; n > 1
    /// steps on the mean loss of n at a time, which denoises the gradient and
    /// tolerates a higher learning rate. Every example in a step keeps its
    /// forward graph until the backward, so memory grows with n: 4 is what fits
    /// beside the frozen 1.5B base on a 24 GB card.
    pub batch_size: u32,
    /// The anti-collapse weight: how hard a policy-gradient run is held near
    /// the base (GRPO's KL-to-reference penalty). RAFT has no such term, since
    /// it avoids collapse by training on verified winners alone, and ignores it.
    pub kl_beta: f64,
}

impl TrainingRecipe {
    /// Refuse a recipe no run could actually train under.
    pub fn validate(&self) -> Result<()> {
        if !(self.learning_rate.is_finite() && self.learning_rate > 0.0) {
            return Err(AntumbraError::other(format!(
                "a recipe's learning rate must be positive and finite, not {}",
                self.learning_rate
            )));
        }
        if self.batch_size == 0 {
            return Err(AntumbraError::other(
                "a recipe's batch size is at least one example",
            ));
        }
        if !(self.kl_beta.is_finite() && self.kl_beta >= 0.0) {
            return Err(AntumbraError::other(format!(
                "a recipe's KL weight must be non-negative and finite, not {}",
                self.kl_beta
            )));
        }
        Ok(())
    }
}

/// One run of a recipe, as the loop records it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecipeRecord {
    /// The shadow that trained under the recipe. One shadow runs one recipe,
    /// so the shadow names the row.
    pub shadow: ShadowId,
    pub run_id: RunId,
    pub generation: Generation,
    pub recipe: TrainingRecipe,
    /// The row this recipe descends from: the recipe the run trained under the
    /// generation before. `None` for a run's first.
    #[serde(default)]
    pub parent: Option<ShadowId>,
    /// The partition the fitness was measured under, so a fitness
    /// read under one split is never ranked against one read under another.
    /// `None` when nothing was held out.
    #[serde(default)]
    pub partition_seed: Option<u64>,
    /// Mean fitness over the evaluations below.
    pub fitness_mean: f32,
    /// Sample variance of those evaluations, `None` until there are two: one
    /// evaluation says nothing about its own noise.
    #[serde(default)]
    pub fitness_variance: Option<f32>,
    /// How many evaluations the mean rests on. Ranking is shrunk toward the
    /// generational mean in proportion to it, so a lucky single evaluation
    /// cannot win.
    pub evaluations: u32,
    pub created_at: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recipe() -> TrainingRecipe {
        TrainingRecipe {
            learning_rate: 1e-4,
            batch_size: 1,
            kl_beta: 0.04,
        }
    }

    #[test]
    fn a_trainable_recipe_validates() {
        assert!(recipe().validate().is_ok());
    }

    #[test]
    fn an_untrainable_recipe_is_refused() {
        let bad = [
            TrainingRecipe {
                learning_rate: 0.0,
                ..recipe()
            },
            TrainingRecipe {
                learning_rate: f64::NAN,
                ..recipe()
            },
            TrainingRecipe {
                batch_size: 0,
                ..recipe()
            },
            TrainingRecipe {
                kl_beta: -0.1,
                ..recipe()
            },
            TrainingRecipe {
                kl_beta: f64::INFINITY,
                ..recipe()
            },
        ];
        for r in bad {
            assert!(r.validate().is_err(), "{r:?}");
        }
    }

    #[test]
    fn a_record_round_trips_and_older_rows_read_without_the_optional_fields() {
        let record = RecipeRecord {
            shadow: ShadowId::new("run:g1"),
            run_id: RunId::new("run"),
            generation: Generation(1),
            recipe: recipe(),
            parent: Some(ShadowId::new("run:g0")),
            partition_seed: Some(7),
            fitness_mean: 0.5,
            fitness_variance: None,
            evaluations: 1,
            created_at: Utc::now(),
        };
        let wire = serde_json::to_value(&record).unwrap();
        let back: RecipeRecord = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(back, record);
        let mut bare = wire;
        for field in ["parent", "partition_seed", "fitness_variance"] {
            bare.as_object_mut().unwrap().remove(field);
        }
        let read: RecipeRecord = serde_json::from_value(bare).unwrap();
        assert_eq!((read.parent, read.partition_seed), (None, None));
    }
}
