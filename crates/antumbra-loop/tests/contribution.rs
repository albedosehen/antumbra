//! Leave-one-out contribution through the loop (ADR-0022 S-5): mask each shared
//! expert, route the live tasks again, score both ways, record the difference.

use std::sync::Mutex;

use async_trait::async_trait;
use chrono::Utc;

use antumbra_core::ports::{
    Embedder, EvaluateRequest, TaskPrompt, TaskScores, TrainOutcome, TrainRequest, Trainer,
};
use antumbra_core::slice::Holdout;
use antumbra_core::testing::ScriptedTrainer;
use antumbra_core::{Expert, ExpertId, ExpertStatus, Generation, Result, RunId, TransitionCause};
use antumbra_loop::{ContributionPolicy, GenerationLoop, LoopConfig};
use antumbra_store::repo::{contribution, expert, lifecycle};
use antumbra_store::Store;

const DIM: usize = 4;

/// Puts a text on the axis of the first skill word it names.
struct Axes;

#[async_trait]
impl Embedder for Axes {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut v = vec![0.0f32; DIM];
        for (i, word) in ["alpha", "beta", "gamma", "delta"].iter().enumerate() {
            if text.contains(word) {
                v[i] = 1.0;
                break;
            }
        }
        Ok(v)
    }
    fn dim(&self) -> usize {
        DIM
    }
}

fn axis(i: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM];
    v[i] = 1.0;
    v
}

fn shared(id: &str, capability: Vec<f32>) -> Expert {
    Expert {
        id: ExpertId::new(id),
        name: id.into(),
        base_model: "code-base".into(),
        artifact_uri: format!("adapters/{id}"),
        capability_card: serde_json::json!({}),
        capability_vec: Some(capability),
        fitness: 1.0,
        frozen_at: Some(Utc::now()),
        generation: Generation::ZERO,
        owner: None,
        compartment: None,
        placed_on: None,
        created_at: Utc::now(),
    }
}

/// Trains nothing that graduates, and scores tasks by who serves them: the
/// alpha expert solves alpha tasks, the beta expert is no better than the
/// base, and the base model alone gets half of everything.
#[derive(Default)]
struct Scorer {
    asked: Mutex<Vec<EvaluateRequest>>,
}

#[async_trait]
impl Trainer for Scorer {
    async fn train_shadow(&self, req: TrainRequest) -> Result<TrainOutcome> {
        ScriptedTrainer::collapsing().train_shadow(req).await
    }

    async fn live_tasks(&self, _holdout: Option<Holdout>) -> Result<Vec<TaskPrompt>> {
        Ok(["alpha one", "alpha two", "beta one", "gamma one"]
            .iter()
            .map(|p| TaskPrompt {
                id: p.replace(' ', "-"),
                prompt: p.to_string(),
                region: p.split(' ').next().unwrap_or_default().to_string(),
            })
            .collect())
    }

    async fn evaluate(&self, req: EvaluateRequest) -> Result<TaskScores> {
        let scores = req
            .task_ids
            .iter()
            .map(|t| {
                let s = match req.adapter_uri.as_deref() {
                    Some("adapters/expert:alpha") if t.starts_with("alpha") => 1.0,
                    _ => 0.5,
                };
                (t.clone(), s)
            })
            .collect();
        if let Ok(mut asked) = self.asked.lock() {
            asked.push(req);
        }
        Ok(TaskScores { scores })
    }
}

async fn population() -> Result<Store> {
    let store = Store::connect_memory(DIM).await?;
    expert::insert(&store, &shared("expert:alpha", axis(0))).await?;
    expert::insert(&store, &shared("expert:beta", axis(1))).await?;
    expert::insert(&store, &shared("expert:unused", axis(3))).await?;
    // Would take the gamma task, but is dormant.
    expert::insert(&store, &shared("expert:gamma", axis(2))).await?;
    lifecycle::transition(
        &store,
        &ExpertId::new("expert:gamma"),
        ExpertStatus::Dormant,
        TransitionCause::Operator { note: None },
        None,
    )
    .await?;
    Ok(store)
}

fn every(n: u32) -> LoopConfig {
    LoopConfig {
        contribution: Some(ContributionPolicy {
            every: n,
            seeds: 2,
            max_tasks: 16,
            baseline: false,
        }),
        ..LoopConfig::default()
    }
}

#[tokio::test]
async fn each_expert_is_measured_against_the_population_without_it() -> Result<()> {
    let store = population().await?;
    let trainer = Scorer::default();
    let reports = GenerationLoop::new(&store, &trainer, &Axes, every(1))
        .run_until(&RunId::new("run:loo"), 1)
        .await?;
    let found = |id: &str| {
        reports[0]
            .contribution
            .iter()
            .find(|c| c.expert.as_str() == id)
            .cloned()
    };
    let alpha = found("expert:alpha").expect("measured");
    assert_eq!((alpha.routed, alpha.tasks), (2, 4));
    assert_eq!((alpha.with, alpha.without), (Some(1.0), Some(0.5)));
    assert_eq!(
        alpha.delta(),
        Some(0.5),
        "it beats the base it falls back to"
    );
    let beta = found("expert:beta").expect("measured");
    assert_eq!(
        (beta.routed, beta.delta()),
        (1, Some(0.0)),
        "routed to, adds nothing"
    );
    let unused = found("expert:unused").expect("measured");
    assert!(unused.is_unused() && unused.delta().is_none());
    assert!(
        found("expert:gamma").is_none(),
        "a dormant expert is not measured"
    );

    // Each adapter was scored once, the base alone included, under one set of
    // seeds for both sides.
    let asked = trainer.asked.lock().unwrap().clone();
    assert_eq!(asked.len(), 3, "{asked:?}");
    assert!(asked
        .iter()
        .all(|r| r.seeds == asked[0].seeds && r.seeds.len() == 2));
    let base = asked
        .iter()
        .find(|r| r.adapter_uri.is_none())
        .expect("the base");
    assert_eq!(base.task_ids, ["alpha-one", "alpha-two", "beta-one"]);

    let stored = contribution::history(&store, &ExpertId::new("expert:alpha")).await?;
    assert_eq!(stored, vec![alpha]);
    Ok(())
}

/// The routed population against its best single expert, on the same tasks:
/// here routing adds nothing over sending everything to the alpha expert,
/// and the record says so.
#[tokio::test]
async fn the_population_is_compared_with_its_best_single_expert() -> Result<()> {
    let store = population().await?;
    let trainer = Scorer::default();
    let cfg = LoopConfig {
        contribution: Some(ContributionPolicy {
            every: 1,
            seeds: 2,
            max_tasks: 16,
            baseline: true,
        }),
        ..LoopConfig::default()
    };
    let reports = GenerationLoop::new(&store, &trainer, &Axes, cfg)
        .run_until(&RunId::new("run:base"), 1)
        .await?;
    let b = reports[0].baseline.clone().expect("compared");
    assert_eq!(b.tasks, 4);
    assert_eq!(b.population, 0.75, "alpha 1, 1; beta 0.5; the base 0.5");
    assert_eq!(b.best, Some(ExpertId::new("expert:alpha")));
    assert_eq!(b.best_alone, Some(0.75));
    assert_eq!(b.delta(), Some(0.0));
    let kept = contribution::baselines_for_run(&store, &RunId::new("run:base")).await?;
    assert_eq!(kept, vec![b]);
    // Each adapter was still scored once, over everything it was needed for.
    let asked = trainer.asked.lock().unwrap().clone();
    assert_eq!(asked.len(), 4, "alpha, beta, unused and the base");
    assert!(asked
        .iter()
        .filter(|r| r.adapter_uri.is_some())
        .all(|r| r.task_ids.len() == 4));
    Ok(())
}

#[tokio::test]
async fn contribution_is_measured_on_its_schedule_and_off_by_default() -> Result<()> {
    let store = population().await?;
    let trainer = Scorer::default();
    let reports = GenerationLoop::new(&store, &trainer, &Axes, every(2))
        .run_until(&RunId::new("run:loo"), 2)
        .await?;
    assert!(!reports[0].contribution.is_empty());
    assert!(
        reports[1].contribution.is_empty(),
        "generation 1 is not due"
    );

    let store = population().await?;
    let trainer = Scorer::default();
    let reports = GenerationLoop::new(&store, &trainer, &Axes, LoopConfig::default())
        .run_until(&RunId::new("run:off"), 1)
        .await?;
    assert!(reports[0].contribution.is_empty());
    assert!(trainer.asked.lock().unwrap().is_empty());
    Ok(())
}

/// Every generation's measurement draws the same seeds, so an unchanged
/// population measures the same and what the grow step credits is change,
/// not seed noise.
#[tokio::test]
async fn every_generation_is_measured_under_the_same_seeds() -> Result<()> {
    let store = population().await?;
    let trainer = Scorer::default();
    GenerationLoop::new(&store, &trainer, &Axes, every(1))
        .run_until(&RunId::new("run:paired"), 2)
        .await?;
    let asked = trainer.asked.lock().unwrap().clone();
    assert!(asked.len() >= 6, "two generations measured: {asked:?}");
    assert!(asked.iter().all(|r| r.seeds == asked[0].seeds));
    Ok(())
}
