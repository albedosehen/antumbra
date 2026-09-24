//! Graduation on a re-measurement (ADR-0022 S-1's fourth constraint): the
//! carried-forward shadow is evaluated again under fresh seeds, on the
//! held-out slice its trainer confirmed withholding, and the threshold applies
//! to the mean rather than to the training fitness a search ranked by.

use std::sync::Mutex;

use async_trait::async_trait;

use antumbra_core::ports::{
    RemeasureRequest, Remeasurement, TaskOutcome, TrainOutcome, TrainRequest, Trainer,
};
use antumbra_core::slice::Partition;
use antumbra_core::testing::{FixedEmbedder, ScriptedTrainer};
use antumbra_core::{Result, RunId, SubjectKind};
use antumbra_loop::{GenerationLoop, GenerationReport, LoopConfig, Remeasure};
use antumbra_store::repo::evaluation;
use antumbra_store::Store;

/// A scripted trainer that keeps every re-measurement it was asked for.
struct Recording {
    inner: ScriptedTrainer,
    asked: Mutex<Vec<RemeasureRequest>>,
}

impl Recording {
    fn new(inner: ScriptedTrainer) -> Self {
        Self {
            inner,
            asked: Mutex::new(Vec::new()),
        }
    }

    fn asked(&self) -> Vec<RemeasureRequest> {
        self.asked.lock().map(|a| a.clone()).unwrap_or_default()
    }
}

#[async_trait]
impl Trainer for Recording {
    async fn train_shadow(&self, req: TrainRequest) -> Result<TrainOutcome> {
        self.inner.train_shadow(req).await
    }

    async fn remeasure(&self, req: RemeasureRequest) -> Result<Remeasurement> {
        if let Ok(mut asked) = self.asked.lock() {
            asked.push(req.clone());
        }
        self.inner.remeasure(req).await
    }
}

fn scoring(fitness: f32, remeasured: &[f32]) -> ScriptedTrainer {
    ScriptedTrainer {
        final_fitness: fitness,
        remeasured: Some(remeasured.to_vec()),
        ..ScriptedTrainer::graduating()
    }
}

fn remeasuring() -> LoopConfig {
    LoopConfig {
        remeasure: Some(Remeasure::default()),
        ..LoopConfig::default()
    }
}

async fn run(
    store: &Store,
    trainer: &dyn Trainer,
    cfg: LoopConfig,
    run: &str,
    generations: u32,
) -> Result<Vec<GenerationReport>> {
    let embedder = FixedEmbedder::new(8);
    GenerationLoop::new(store, trainer, &embedder, cfg)
        .run_until(&RunId::new(run), generations)
        .await
}

/// A training fitness of 0.9 does not graduate a shadow that re-measures at
/// 0.3, and one that re-measures at 0.7 does.
#[tokio::test]
async fn graduation_is_judged_on_the_remeasured_mean() -> Result<()> {
    for (rates, expected) in [(vec![0.2, 0.3, 0.4], false), (vec![0.6, 0.7, 0.8], true)] {
        let store = Store::connect_memory(8).await?;
        let report = run(&store, &scoring(0.9, &rates), remeasuring(), "run:r", 1)
            .await?
            .remove(0);
        let mean = rates.iter().sum::<f32>() / 3.0;
        assert!((report.graduation_score - mean).abs() < 1e-6);
        assert_eq!(report.graduated, expected, "{rates:?}");
        assert_eq!(
            report.remeasured.map(|m| m.pass_rates.len()),
            Some(3),
            "three repeats by default"
        );
    }
    Ok(())
}

/// The seeds are the run's own for that generation: three different ones,
/// different again the next generation, and the same when the run is started
/// over, so a resumed run re-measures exactly what a continuous one would.
#[tokio::test]
async fn the_seeds_are_fresh_per_generation_and_repeat_on_a_rerun() -> Result<()> {
    let first = Recording::new(scoring(0.9, &[0.7]));
    run(
        &Store::connect_memory(8).await?,
        &first,
        remeasuring(),
        "run:s",
        2,
    )
    .await?;
    let again = Recording::new(scoring(0.9, &[0.7]));
    run(
        &Store::connect_memory(8).await?,
        &again,
        remeasuring(),
        "run:s",
        2,
    )
    .await?;

    let seeds =
        |t: &Recording| -> Vec<Vec<u64>> { t.asked().into_iter().map(|r| r.seeds).collect() };
    let (a, b) = (seeds(&first), seeds(&again));
    assert_eq!(a, b, "a rerun draws the same seeds");
    assert_eq!(a.len(), 2);
    assert_eq!(a[0].len(), 3);
    assert!(a[0][0] != a[0][1] && a[0][1] != a[0][2] && a[0][0] != a[0][2]);
    assert!(
        a[0].iter().all(|s| !a[1].contains(s)),
        "each generation its own"
    );
    Ok(())
}

/// The held-out slice is fresh only if the trainer really withheld it; one
/// that ignored the holdout learned those tasks, so it re-measures what it
/// trained on instead.
#[tokio::test]
async fn the_held_out_slice_is_used_only_when_its_holdout_was_confirmed() -> Result<()> {
    let cfg = LoopConfig {
        partition: Some(Partition::default()),
        ..remeasuring()
    };
    let confirmed = Recording::new(scoring(0.9, &[0.7]));
    run(
        &Store::connect_memory(8).await?,
        &confirmed,
        cfg.clone(),
        "run:h",
        1,
    )
    .await?;
    assert!(confirmed.asked()[0].holdout.is_some());

    let ignored = Recording::new(ScriptedTrainer {
        ignores_holdout: true,
        ..scoring(0.9, &[0.7])
    });
    run(&Store::connect_memory(8).await?, &ignored, cfg, "run:h", 1).await?;
    assert!(ignored.asked()[0].holdout.is_none());
    Ok(())
}

#[tokio::test]
async fn a_trainer_that_cannot_remeasure_fails_the_generation() -> Result<()> {
    let store = Store::connect_memory(8).await?;
    let trainer = ScriptedTrainer::graduating();
    assert!(run(&store, &trainer, remeasuring(), "run:n", 1)
        .await
        .is_err());
    Ok(())
}

/// A generation that passed an impossible task fails whole, so there is
/// nothing to re-measure it for.
#[tokio::test]
async fn a_generation_failed_by_a_shortcut_is_not_remeasured() -> Result<()> {
    let trainer = Recording::new(ScriptedTrainer {
        per_task: vec![
            TaskOutcome {
                task_id: "task:0".into(),
                passed: true,
                size: 10,
                impossible: false,
            },
            TaskOutcome {
                task_id: "unsatisfiable".into(),
                passed: true,
                size: 10,
                impossible: true,
            },
        ],
        ..scoring(0.9, &[0.9])
    });
    let cfg = LoopConfig {
        partition: Some(Partition::default()),
        ..remeasuring()
    };
    let report = run(&Store::connect_memory(8).await?, &trainer, cfg, "run:x", 1)
        .await?
        .remove(0);
    assert!(!report.graduated);
    assert!(report.remeasured.is_none());
    assert!(trainer.asked().is_empty());
    Ok(())
}

#[tokio::test]
async fn the_evaluation_row_keeps_the_remeasurement() -> Result<()> {
    let store = Store::connect_memory(8).await?;
    run(
        &store,
        &scoring(0.9, &[0.6, 0.7, 0.8]),
        remeasuring(),
        "run:e",
        1,
    )
    .await?;
    let rows = evaluation::list_for_run(&store, &RunId::new("run:e"), SubjectKind::Shadow).await?;
    let metrics = rows[0].metrics.clone().unwrap_or_default();
    assert_eq!(
        metrics["remeasured"]["pass_rates"].as_array().map(Vec::len),
        Some(3)
    );
    Ok(())
}
