//! A searched generation (the recipe search, run as a cohort): the search
//! proposes a recipe per cohort member, every member trains from the base
//! under its own, the best is carried forward, and only its recipe propagates.

use std::sync::Mutex;

use async_trait::async_trait;

use antumbra_core::ports::{TrainOutcome, TrainRequest, Trainer};
use antumbra_core::testing::{FixedEmbedder, ScriptedTrainer};
use antumbra_core::{Result, RunId, ShadowId, ShadowStatus, TrainingRecipe};
use antumbra_loop::search::SearchPolicy;
use antumbra_loop::{GenerationLoop, GenerationReport, LoopConfig};
use antumbra_store::repo::{recipe, shadow};
use antumbra_store::Store;

const START: TrainingRecipe = TrainingRecipe {
    learning_rate: 1e-4,
    batch_size: 1,
    kl_beta: 0.04,
};

/// Scores a run by its recipe alone: best near a learning rate of 3e-4 and a
/// batch of 4. Keeps every request it was sent.
#[derive(Default)]
struct RecipeSensitive {
    asked: Mutex<Vec<(ShadowId, Option<TrainingRecipe>)>>,
}

fn quality(r: &TrainingRecipe) -> f32 {
    let off = r.learning_rate.log10() - 3e-4f64.log10();
    let batch = match r.batch_size {
        4 => 0.0,
        2 => 0.05,
        _ => 0.15,
    };
    (0.9 - 0.3 * off * off - batch).clamp(0.0, 1.0) as f32
}

#[async_trait]
impl Trainer for RecipeSensitive {
    async fn train_shadow(&self, req: TrainRequest) -> Result<TrainOutcome> {
        if let Ok(mut asked) = self.asked.lock() {
            asked.push((req.shadow.clone(), req.recipe));
        }
        let fitness = req.recipe.map_or(0.0, |r| quality(&r));
        let outcome = ScriptedTrainer {
            final_fitness: fitness,
            curve: vec![fitness],
            ..ScriptedTrainer::graduating()
        }
        .train_shadow(req)
        .await?;
        Ok(outcome)
    }
}

fn searched(cohort: usize) -> LoopConfig {
    LoopConfig {
        recipe: Some(START),
        search: Some(SearchPolicy {
            cohort,
            ..SearchPolicy::default()
        }),
        ..LoopConfig::default()
    }
}

async fn run(
    store: &Store,
    trainer: &dyn Trainer,
    cfg: LoopConfig,
    generations: u32,
) -> Result<Vec<GenerationReport>> {
    let embedder = FixedEmbedder::new(8);
    GenerationLoop::new(store, trainer, &embedder, cfg)
        .run_until(&RunId::new("run:cohort"), generations)
        .await
}

#[tokio::test]
async fn a_searched_generation_trains_its_cohort_and_carries_the_best() -> Result<()> {
    let store = Store::connect_memory(8).await?;
    let trainer = RecipeSensitive::default();
    let reports = run(&store, &trainer, searched(4), 1).await?;
    let report = &reports[0];

    assert_eq!(report.cohort.len(), 4);
    let recipes: Vec<TrainingRecipe> = report.cohort.iter().filter_map(|m| m.recipe).collect();
    assert_eq!(recipes.len(), 4);
    assert_eq!(
        recipes[0], START,
        "the run's starting recipe leads generation 0"
    );
    for (i, r) in recipes.iter().enumerate() {
        assert!(
            recipes[i + 1..].iter().all(|other| other != r),
            "all different"
        );
    }
    let best = report
        .cohort
        .iter()
        .max_by(|a, b| a.fitness.total_cmp(&b.fitness))
        .expect("members");
    assert_eq!(
        report.shadow, best.shadow,
        "the best member is carried forward"
    );
    assert_eq!(report.fitness, best.fitness);

    // Every member trained from the base under the recipe it was given, and
    // left a recipe row and a shadow; the ones that lost were pruned.
    let asked = trainer.asked.lock().unwrap().clone();
    assert_eq!(asked.len(), 4);
    let rows = recipe::list_for_run(&store, &RunId::new("run:cohort")).await?;
    assert_eq!(rows.len(), 4);
    for member in &report.cohort {
        let sh = shadow::get(&store, &member.shadow)
            .await?
            .expect("a shadow");
        if member.shadow != report.shadow {
            assert_eq!(sh.status, ShadowStatus::Pruned, "{}", member.shadow);
        }
    }
    Ok(())
}

/// Only the recipe propagates: the next generation starts from the winner's
/// recipe, and nothing else of the winner's is handed on.
#[tokio::test]
async fn the_winning_recipe_leads_the_next_cohort() -> Result<()> {
    let store = Store::connect_memory(8).await?;
    let trainer = RecipeSensitive::default();
    let reports = run(&store, &trainer, searched(4), 2).await?;
    assert_eq!(reports[1].cohort[0].recipe, reports[0].recipe);
    let rows = recipe::list_for_run(&store, &RunId::new("run:cohort")).await?;
    let gen1: Vec<_> = rows.iter().filter(|r| r.generation.0 == 1).collect();
    assert_eq!(gen1.len(), 4);
    assert!(
        gen1.iter()
            .all(|r| r.parent.as_ref() == Some(&reports[0].shadow)),
        "each member of generation 1 descends from generation 0's best"
    );
    Ok(())
}

/// Scores its runs in a fixed order, whatever the recipe.
struct InOrder(Mutex<Vec<f32>>);

#[async_trait]
impl Trainer for InOrder {
    async fn train_shadow(&self, req: TrainRequest) -> Result<TrainOutcome> {
        let fitness = self.0.lock().map(|mut f| f.remove(0)).unwrap_or(0.0);
        ScriptedTrainer {
            final_fitness: fitness,
            curve: vec![fitness],
            ..ScriptedTrainer::graduating()
        }
        .train_shadow(req)
        .await
    }
}

/// The best of a cohort was chosen for scoring well, so until graduation
/// re-measures, its score is shrunk toward the cohort's mean: 0.9 beside
/// three 0.1s (mean 0.3) graduates as 0.6, which clears 0.5 and not 0.7.
#[tokio::test]
async fn a_cohorts_best_graduates_on_its_shrunk_score() -> Result<()> {
    for (threshold, expected) in [(0.5, true), (0.7, false)] {
        let store = Store::connect_memory(8).await?;
        let trainer = InOrder(Mutex::new(vec![0.1, 0.9, 0.1, 0.1]));
        let cfg = LoopConfig {
            graduate_threshold: threshold,
            ..searched(4)
        };
        let report = run(&store, &trainer, cfg, 1).await?.remove(0);
        assert_eq!(report.fitness, 0.9);
        assert!(
            (report.graduation_score - 0.6).abs() < 1e-6,
            "{}",
            report.graduation_score
        );
        assert_eq!(report.graduated, expected, "threshold {threshold}");
    }
    Ok(())
}

#[tokio::test]
async fn an_unsearched_generation_scores_its_one_shadow_as_it_always_did() -> Result<()> {
    let store = Store::connect_memory(8).await?;
    let report = run(
        &store,
        &ScriptedTrainer::graduating(),
        LoopConfig::default(),
        1,
    )
    .await?
    .remove(0);
    assert_eq!(report.cohort.len(), 1);
    assert_eq!(report.shadow, ShadowId::new("run:cohort:g0"));
    assert_eq!(report.graduation_score, report.fitness);
    Ok(())
}

#[tokio::test]
async fn a_searched_run_needs_a_starting_recipe() -> Result<()> {
    let store = Store::connect_memory(8).await?;
    let cfg = LoopConfig {
        recipe: None,
        ..searched(4)
    };
    assert!(run(&store, &RecipeSensitive::default(), cfg, 1)
        .await
        .is_err());
    Ok(())
}

/// A run stopped between generations and resumed proposes the cohort a run
/// that never stopped would have, because proposals come from the stored
/// history and the seed.
#[tokio::test]
async fn a_resumed_run_proposes_what_a_continuous_one_would() -> Result<()> {
    let recipes = |reports: &[GenerationReport]| -> Vec<Option<TrainingRecipe>> {
        reports[1].cohort.iter().map(|m| m.recipe).collect()
    };
    let continuous = Store::connect_memory(8).await?;
    let straight = run(&continuous, &RecipeSensitive::default(), searched(4), 2).await?;

    let resumed = Store::connect_memory(8).await?;
    run(&resumed, &RecipeSensitive::default(), searched(4), 1).await?;
    let mut after = run(&resumed, &RecipeSensitive::default(), searched(4), 2).await?;
    after.insert(0, straight[0].clone());
    assert_eq!(recipes(&after), recipes(&straight));
    Ok(())
}

/// A generation that crashed partway left rows for the members it had
/// trained. They are not history: re-run, the generation proposes what it
/// would have, rather than treating its own half-finished members as
/// evidence (here, a leftover row scoring a perfect 1.0).
#[tokio::test]
async fn a_half_finished_generation_is_not_its_own_history() -> Result<()> {
    let continuous = Store::connect_memory(8).await?;
    let straight = run(&continuous, &RecipeSensitive::default(), searched(4), 2).await?;

    let crashed = Store::connect_memory(8).await?;
    run(&crashed, &RecipeSensitive::default(), searched(4), 1).await?;
    recipe::upsert(
        &crashed,
        &antumbra_core::RecipeRecord {
            shadow: ShadowId::new("run:cohort:g1:s3"),
            run_id: RunId::new("run:cohort"),
            generation: antumbra_core::Generation(1),
            recipe: TrainingRecipe {
                learning_rate: 1e-5,
                batch_size: 1,
                kl_beta: 0.04,
            },
            parent: None,
            partition_seed: None,
            fitness_mean: 1.0,
            fitness_variance: None,
            evaluations: 1,
            created_at: chrono::Utc::now(),
        },
    )
    .await?;
    let rerun = run(&crashed, &RecipeSensitive::default(), searched(4), 2).await?;
    let proposed = |r: &GenerationReport| -> Vec<Option<TrainingRecipe>> {
        r.cohort.iter().map(|m| m.recipe).collect()
    };
    assert_eq!(proposed(&rerun[0]), proposed(&straight[1]));
    Ok(())
}

fn with_slow(interval: u32) -> LoopConfig {
    LoopConfig {
        recipe: Some(START),
        search: Some(SearchPolicy {
            cohort: 3,
            slow: 1,
            slow_interval: interval,
            ..SearchPolicy::default()
        }),
        ..LoopConfig::default()
    }
}

/// The slow member keeps its recipe for its interval, read back from the
/// loop's own rows, and is proposed afresh after it. The report says which
/// member is slow.
#[tokio::test]
async fn a_slow_member_holds_its_recipe_for_its_interval() -> Result<()> {
    let store = Store::connect_memory(8).await?;
    let reports = run(&store, &RecipeSensitive::default(), with_slow(2), 3).await?;
    let slow: Vec<Option<TrainingRecipe>> = reports
        .iter()
        .map(|r| {
            let slow: Vec<_> = r.cohort.iter().filter(|m| m.slow).collect();
            assert_eq!(slow.len(), 1, "one slow member");
            assert!(slow[0].shadow.as_str().ends_with(":s2"), "the last");
            slow[0].recipe
        })
        .collect();
    assert_eq!(slow[0], slow[1], "held for its interval of two");
    assert_ne!(slow[1], slow[2], "then proposed afresh");
    Ok(())
}

/// A run resumed partway holds and proposes what a continuous one would,
/// because what each member held is read from the stored rows.
#[tokio::test]
async fn a_resumed_run_holds_what_a_continuous_one_would() -> Result<()> {
    let recipes = |r: &GenerationReport| -> Vec<Option<TrainingRecipe>> {
        r.cohort.iter().map(|m| m.recipe).collect()
    };
    let continuous = Store::connect_memory(8).await?;
    let straight = run(&continuous, &RecipeSensitive::default(), with_slow(3), 3).await?;
    let resumed = Store::connect_memory(8).await?;
    run(&resumed, &RecipeSensitive::default(), with_slow(3), 2).await?;
    let after = run(&resumed, &RecipeSensitive::default(), with_slow(3), 3).await?;
    assert_eq!(after.len(), 1);
    assert_eq!(recipes(&after[0]), recipes(&straight[2]));
    Ok(())
}
