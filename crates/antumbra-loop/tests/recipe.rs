//! The recipe ledger (ADR-0022 S-1): every generation records the recipe its
//! shadow trained under, as the trainer reported it, descending from the one
//! before, and measured under the partition the trainer confirmed.

use async_trait::async_trait;

use antumbra_core::ports::{TrainOutcome, TrainRequest, Trainer};
use antumbra_core::slice::Partition;
use antumbra_core::testing::{FixedEmbedder, ScriptedTrainer, SCRIPTED_RECIPE};
use antumbra_core::{Result, RunId, ShadowId, TrainingRecipe};
use antumbra_loop::{GenerationLoop, GenerationReport, LoopConfig};
use antumbra_store::repo::recipe;
use antumbra_store::Store;

const ASKED: TrainingRecipe = TrainingRecipe {
    learning_rate: 3e-4,
    batch_size: 4,
    kl_beta: 0.1,
};

async fn run(
    store: &Store,
    trainer: &dyn Trainer,
    cfg: LoopConfig,
    generations: u32,
) -> Result<Vec<GenerationReport>> {
    let embedder = FixedEmbedder::new(8);
    GenerationLoop::new(store, trainer, &embedder, cfg)
        .run_until(&RunId::new("run:recipe"), generations)
        .await
}

#[tokio::test]
async fn each_generation_leaves_a_row_descending_from_the_last() -> Result<()> {
    let store = Store::connect_memory(8).await?;
    let reports = run(
        &store,
        &ScriptedTrainer::graduating(),
        LoopConfig::default(),
        3,
    )
    .await?;
    assert!(reports.iter().all(|r| r.recipe == Some(SCRIPTED_RECIPE)));

    let rows = recipe::list_for_run(&store, &RunId::new("run:recipe")).await?;
    let lineage: Vec<(String, Option<String>)> = rows
        .iter()
        .map(|r| {
            (
                r.shadow.to_string(),
                r.parent.as_ref().map(ShadowId::to_string),
            )
        })
        .collect();
    assert_eq!(
        lineage,
        [
            ("run:recipe:g0".to_string(), None),
            (
                "run:recipe:g1".to_string(),
                Some("run:recipe:g0".to_string())
            ),
            (
                "run:recipe:g2".to_string(),
                Some("run:recipe:g1".to_string())
            ),
        ]
    );
    for row in &rows {
        assert_eq!(
            row.recipe, SCRIPTED_RECIPE,
            "the trainer's own, as reported"
        );
        assert_eq!((row.fitness_mean, row.evaluations), (0.9, 1));
        assert_eq!(row.fitness_variance, None, "one evaluation has no variance");
    }
    Ok(())
}

#[tokio::test]
async fn the_loops_recipe_is_asked_for_and_recorded_as_run() -> Result<()> {
    let store = Store::connect_memory(8).await?;
    let cfg = LoopConfig {
        recipe: Some(ASKED),
        ..LoopConfig::default()
    };
    let reports = run(&store, &ScriptedTrainer::graduating(), cfg, 1).await?;
    assert_eq!(reports[0].recipe, Some(ASKED));
    let row = recipe::get(&store, &ShadowId::new("run:recipe:g0"))
        .await?
        .expect("a row for the generation");
    assert_eq!(row.recipe, ASKED);
    Ok(())
}

#[tokio::test]
async fn a_trainer_that_does_not_say_leaves_no_row() -> Result<()> {
    let store = Store::connect_memory(8).await?;
    let trainer = ScriptedTrainer {
        ignores_recipe: true,
        ..ScriptedTrainer::graduating()
    };
    let cfg = LoopConfig {
        recipe: Some(ASKED),
        ..LoopConfig::default()
    };
    let reports = run(&store, &trainer, cfg, 1).await?;
    assert_eq!(reports[0].recipe, None);
    assert!(recipe::list_for_run(&store, &RunId::new("run:recipe"))
        .await?
        .is_empty());
    Ok(())
}

/// Trains under its own recipe whatever it is asked, and says so.
struct Headstrong {
    inner: ScriptedTrainer,
    own: TrainingRecipe,
}

#[async_trait]
impl Trainer for Headstrong {
    async fn train_shadow(&self, req: TrainRequest) -> Result<TrainOutcome> {
        let outcome = self.inner.train_shadow(req).await?;
        Ok(TrainOutcome {
            recipe: Some(self.own),
            ..outcome
        })
    }
}

/// The row says what trained, not what was wanted: a search ranking rows by
/// the asked-for recipe would credit settings no shadow used.
#[tokio::test]
async fn a_run_under_another_recipe_is_recorded_as_it_ran() -> Result<()> {
    let store = Store::connect_memory(8).await?;
    let trainer = Headstrong {
        inner: ScriptedTrainer::graduating(),
        own: SCRIPTED_RECIPE,
    };
    let cfg = LoopConfig {
        recipe: Some(ASKED),
        ..LoopConfig::default()
    };
    let reports = run(&store, &trainer, cfg, 1).await?;
    assert_eq!(reports[0].recipe, Some(SCRIPTED_RECIPE));
    let row = recipe::get(&store, &ShadowId::new("run:recipe:g0"))
        .await?
        .expect("a row for the generation");
    assert_eq!(row.recipe, SCRIPTED_RECIPE);
    Ok(())
}

/// The seed is kept only when the trainer confirmed the holdout, because only
/// then is the fitness over that split's visible tasks.
#[tokio::test]
async fn the_partition_seed_is_kept_only_under_a_confirmed_holdout() -> Result<()> {
    let partition = Partition::new(0.2, 0.1, 7)?;
    let cfg = LoopConfig {
        partition: Some(partition),
        ..LoopConfig::default()
    };
    let seed_of = |store: Store| async move {
        recipe::get(&store, &ShadowId::new("run:recipe:g0"))
            .await
            .map(|row| row.and_then(|r| r.partition_seed))
    };

    let confirmed = Store::connect_memory(8).await?;
    run(&confirmed, &ScriptedTrainer::graduating(), cfg.clone(), 1).await?;
    assert_eq!(seed_of(confirmed).await?, Some(7));

    let ignored = Store::connect_memory(8).await?;
    let trainer = ScriptedTrainer {
        ignores_holdout: true,
        ..ScriptedTrainer::graduating()
    };
    run(&ignored, &trainer, cfg, 1).await?;
    assert_eq!(seed_of(ignored).await?, None);
    Ok(())
}
