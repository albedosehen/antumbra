//! What a generation trains (ADR-0022 S-1). Without a search, one shadow under
//! the configured recipe, as every generation always has. With one, a cohort:
//! the search proposes a recipe per member, every member trains from the base
//! under its own, and the best of them is the one the generation carries
//! forward. Only the recipe propagates. No member starts from another's
//! weights, so each graduate is a skill of its own rather than a refinement of
//! the last one.

use chrono::Utc;

use antumbra_core::generational::{GenerationHead, LoopState};
use antumbra_core::ports::{RemeasureRequest, Remeasurement, TrainOutcome, TrainRequest};
use antumbra_core::slice::Holdout;
use antumbra_core::{
    AntumbraError, Generation, RecipeRecord, Result, RunId, Shadow, ShadowId, ShadowStatus,
    TrainingRecipe,
};
use antumbra_store::repo::{recipe, shadow};

use crate::recipe::Trained;
use crate::{search, shadow_id, GenerationLoop};

/// One trained member of a generation's cohort.
pub(crate) struct Member {
    pub shadow: Shadow,
    pub outcome: TrainOutcome,
    /// The recipe it trained under, as its trainer reported it.
    pub recipe: Option<TrainingRecipe>,
    pub slow: bool,
}

/// A cohort member as the generation report shows it.
#[derive(Debug, Clone, PartialEq)]
pub struct CohortMember {
    pub shadow: ShadowId,
    /// The recipe it trained under, as its trainer reported it.
    pub recipe: Option<TrainingRecipe>,
    pub fitness: f32,
    /// Whether it is in the slow cohort, whose recipes are held for longer.
    pub slow: bool,
}

/// One member a generation will train: who, under what, descending from what.
struct Planned {
    id: ShadowId,
    recipe: Option<TrainingRecipe>,
    parent: Option<ShadowId>,
    slow: bool,
}

/// How graduation is re-measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Remeasure {
    /// Evaluations, each under its own fresh seed. The record asks for at
    /// least three.
    pub repeats: u32,
}

impl Default for Remeasure {
    fn default() -> Self {
        Self { repeats: 3 }
    }
}

/// The seeds a generation's re-measurement draws from: derived from the run,
/// the generation and the repeat, so a resumed run re-measures exactly what a
/// continuous one would, and never the stream training drew from.
pub(crate) fn remeasure_seeds(run_id: &RunId, generation: Generation, repeats: u32) -> Vec<u64> {
    use sha2::{Digest, Sha256};
    (0..repeats)
        .map(|k| {
            let digest = Sha256::digest(format!("remeasure:{run_id}:{}:{k}", generation.0));
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(&digest[..8]);
            u64::from_le_bytes(bytes)
        })
        .collect()
}

/// A member of a searched generation's cohort: `{run}:g{n}:s{i}`.
fn member_id(run_id: &RunId, generation: Generation, index: usize) -> ShadowId {
    ShadowId::new(format!("{}:s{index}", shadow_id(run_id, generation)))
}

/// What the member at `index` ran before `generation`, read from its rows in
/// `history`: the recipe of its previous run, and how many consecutive
/// generations it had run it. Rows are the record, so a resumed run sees the
/// same slots a continuous one would.
fn slot_before(
    history: &[RecipeRecord],
    run_id: &RunId,
    generation: Generation,
    index: usize,
) -> Option<search::Slot> {
    let ran = |g: u32| {
        let id = member_id(run_id, Generation(g), index);
        history.iter().find(|r| r.shadow == id).map(|r| r.recipe)
    };
    let recipe = ran(generation.0.checked_sub(1)?)?;
    let mut held = 1;
    while let Some(g) = generation.0.checked_sub(held + 1) {
        if ran(g) != Some(recipe) {
            break;
        }
        held += 1;
    }
    Some(search::Slot { recipe, held })
}

impl GenerationLoop<'_> {
    /// Who trains this generation, under what, and what each descends from.
    async fn plan(&self, run_id: &RunId, generation: Generation) -> Result<Vec<Planned>> {
        let Some(policy) = &self.cfg.search else {
            let parent = match generation.0.checked_sub(1) {
                Some(previous) => recipe::get(self.store, &shadow_id(run_id, Generation(previous)))
                    .await?
                    .map(|row| row.shadow),
                None => None,
            };
            return Ok(vec![Planned {
                id: shadow_id(run_id, generation),
                recipe: self.cfg.recipe,
                parent,
                slow: false,
            }]);
        };
        let anchor = self.cfg.recipe.ok_or_else(|| {
            AntumbraError::other(
                "a searched run starts from a recipe: set LoopConfig::recipe to the one the \
                 trainer is configured with",
            )
        })?;
        // Rows from earlier generations measured under this partition, or
        // under none when there is none: fitness read under another split is
        // not comparable. Rows this generation left before a crash are not
        // history, so a resumed run proposes what a continuous one would.
        let seed = self.cfg.partition.map(|p| p.seed);
        let history: Vec<_> = recipe::list_for_run(self.store, run_id)
            .await?
            .into_iter()
            .filter(|row| row.generation < generation && row.partition_seed == seed)
            .collect();
        let parent = search::incumbent(&history, policy.prior_weight).map(|row| row.shadow.clone());
        let slots: Vec<_> = (0..policy.cohort)
            .map(|i| slot_before(&history, run_id, generation, i))
            .collect();
        Ok(
            search::propose(policy, &history, generation.0, anchor, &slots)
                .into_iter()
                .enumerate()
                .filter_map(|(i, r)| {
                    Some(Planned {
                        id: member_id(run_id, generation, i),
                        recipe: Some(r?),
                        parent: parent.clone(),
                        slow: policy.is_slow(i),
                    })
                })
                .collect(),
        )
    }

    /// Spawn and train every member of this generation, recording each one's
    /// recipe, and move the loop to `Explore` on the way. Members come back in
    /// cohort order.
    pub(crate) async fn train_cohort(
        &self,
        head: &mut GenerationHead,
        holdout: Option<Holdout>,
        focus: &[String],
    ) -> Result<Vec<Member>> {
        let run_id = head.run_id.clone();
        let generation = head.generation;
        let plan = self.plan(&run_id, generation).await?;
        let mut shadows = Vec::with_capacity(plan.len());
        for planned in &plan {
            let sh = Shadow::spawn(planned.id.clone(), generation, None, Utc::now());
            shadow::upsert(self.store, &sh).await?;
            shadows.push(sh);
        }
        self.advance(head, LoopState::Explore).await?;
        let mut members = Vec::with_capacity(plan.len());
        for (planned, mut sh) in plan.into_iter().zip(shadows) {
            let Planned {
                id,
                recipe: asked,
                parent,
                slow,
            } = planned;
            sh.advance_to(ShadowStatus::Exploring)?;
            shadow::upsert(self.store, &sh).await?;
            let outcome = self
                .trainer
                .train_shadow(TrainRequest {
                    shadow: id.clone(),
                    base_model: self.cfg.base_model.clone(),
                    corpus_task_ids: Vec::new(),
                    holdout,
                    recipe: asked,
                    focus: focus.to_vec(),
                })
                .await?;
            let recipe = self
                .record_recipe(Trained {
                    run_id: &run_id,
                    generation,
                    shadow: &id,
                    asked,
                    parent,
                    holdout: holdout.as_ref(),
                    outcome: &outcome,
                })
                .await?;
            sh.adapter_uri = Some(outcome.adapter_uri.clone());
            sh.reward_curve = outcome.reward_curve.clone();
            members.push(Member {
                shadow: sh,
                outcome,
                recipe,
                slow,
            });
        }
        Ok(members)
    }

    /// Take the best member out of the cohort and settle the rest: they were
    /// scored and lost, so they are pruned. Ties go to the earlier member: the
    /// first runs the incumbent's recipe, unless a member holding its recipe
    /// already does.
    pub(crate) async fn select(
        &self,
        mut members: Vec<Member>,
    ) -> Result<(Member, Vec<CohortMember>)> {
        let cohort = members
            .iter()
            .map(|m| CohortMember {
                shadow: m.shadow.id.clone(),
                recipe: m.recipe,
                fitness: m.outcome.final_fitness,
                slow: m.slow,
            })
            .collect();
        let best = members
            .iter()
            .enumerate()
            .max_by(|(i, a), (j, b)| {
                a.outcome
                    .final_fitness
                    .total_cmp(&b.outcome.final_fitness)
                    .then_with(|| j.cmp(i))
            })
            .map(|(i, _)| i)
            .ok_or_else(|| AntumbraError::other("a generation trained no shadow"))?;
        let winner = members.swap_remove(best);
        for mut loser in members {
            loser.shadow.advance_to(ShadowStatus::Scoring)?;
            loser.shadow.advance_to(ShadowStatus::Pruned)?;
            shadow::upsert(self.store, &loser.shadow).await?;
        }
        Ok((winner, cohort))
    }

    /// What graduation is judged on, and the re-measurement behind it when
    /// there is one. With `LoopConfig::remeasure`, the carried-forward shadow
    /// is evaluated again under fresh seeds, on the held-out slice its trainer
    /// confirmed withholding, and the score is the mean. A generation that
    /// already failed on a shortcut is not re-measured, since nothing it scores
    /// can graduate it.
    pub(crate) async fn judge(
        &self,
        run_id: &RunId,
        generation: Generation,
        shadow: &ShadowId,
        outcome: &TrainOutcome,
        cohort: &[CohortMember],
        shortcut: bool,
    ) -> Result<(f32, Option<Remeasurement>)> {
        let Some(plan) = self.cfg.remeasure.filter(|_| !shortcut) else {
            return Ok((self.graduation_score(outcome.final_fitness, cohort), None));
        };
        let remeasured = self
            .trainer
            .remeasure(RemeasureRequest {
                shadow: shadow.clone(),
                base_model: self.cfg.base_model.clone(),
                adapter_uri: outcome.adapter_uri.clone(),
                // What the trainer confirmed it withheld: a holdout it ignored
                // left the held-out tasks learned from, so they are not fresh.
                holdout: outcome.holdout,
                seeds: remeasure_seeds(run_id, generation, plan.repeats),
            })
            .await?;
        let score = remeasured.mean().unwrap_or(0.0);
        Ok((score, Some(remeasured)))
    }

    /// The number the graduation threshold is applied to without a
    /// re-measurement. Alone, a shadow's fitness. The best of a cohort was
    /// chosen for scoring well on noisy fitness, which the optimizer's curse
    /// says overstates it, so its fitness is shrunk toward the cohort's mean.
    /// That can only make graduation harder than the raw score would.
    pub(crate) fn graduation_score(&self, winner: f32, cohort: &[CohortMember]) -> f32 {
        match &self.cfg.search {
            Some(policy) if cohort.len() > 1 => {
                let mean = cohort.iter().map(|m| m.fitness).sum::<f32>() / cohort.len() as f32;
                search::shrink(winner, 1, mean, policy.prior_weight) as f32
            }
            _ => winner,
        }
    }
}
