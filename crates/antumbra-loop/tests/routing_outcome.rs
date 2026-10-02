//! The gate learns where each live task should have gone (ADR-0024 D-1): a
//! contribution measurement that scores every expert on every live task keeps
//! each task's clear winner, and the learned router is retrained with the won
//! tasks as exemplars of their winners.

use std::sync::Mutex;

use async_trait::async_trait;
use chrono::Utc;

use antumbra_core::ports::{
    Embedder, EvaluateRequest, TaskPrompt, TaskScores, TrainOutcome, TrainRequest, Trainer,
};
use antumbra_core::slice::Holdout;
use antumbra_core::testing::ScriptedTrainer;
use antumbra_core::{
    Expert, ExpertId, Generation, LearnedRouter, Result, RouterExpert, RoutingOutcome, RunId,
};
use antumbra_loop::{ContributionPolicy, GenerationLoop, LoopConfig};
use antumbra_store::repo::{contribution, expert};
use antumbra_store::Store;

const DIM: usize = 4;

/// The exemplars a router was trained on.
type Exemplars = Vec<(ExpertId, Vec<f32>)>;

/// Alpha texts on one axis, beta on another, and "mixed" between, nearer
/// alpha.
struct Places;

#[async_trait]
impl Embedder for Places {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        Ok(if text.contains("mixed") {
            vec![0.9, 0.3, 0.0, 0.0]
        } else if text.contains("beta") {
            vec![0.0, 1.0, 0.0, 0.0]
        } else {
            vec![1.0, 0.0, 0.0, 0.0]
        })
    }
    fn dim(&self) -> usize {
        DIM
    }
}

/// Graduates nothing; scores the beta expert 1.0 everywhere, and the alpha
/// expert 1.0 on alpha, 0.5 on beta and nothing on the mixed task; trains a
/// centroid router and keeps what it was trained on.
#[derive(Default)]
struct Scorer {
    trained_on: Mutex<Vec<Exemplars>>,
}

#[async_trait]
impl Trainer for Scorer {
    async fn train_shadow(&self, req: TrainRequest) -> Result<TrainOutcome> {
        ScriptedTrainer::collapsing().train_shadow(req).await
    }

    async fn live_tasks(&self, _holdout: Option<Holdout>) -> Result<Vec<TaskPrompt>> {
        Ok(["alpha one", "beta one", "mixed one"]
            .iter()
            .map(|p| TaskPrompt {
                id: p.replace(' ', "-"),
                prompt: p.to_string(),
                region: "r".into(),
            })
            .collect())
    }

    async fn evaluate(&self, req: EvaluateRequest) -> Result<TaskScores> {
        let scores = req
            .task_ids
            .iter()
            .map(|t| {
                let s = match (req.adapter_uri.as_deref(), t.as_str()) {
                    (Some("adapters/expert:beta"), _) => 1.0,
                    (Some("adapters/expert:alpha"), "alpha-one") => 1.0,
                    (Some("adapters/expert:alpha"), "beta-one") => 0.5,
                    _ => 0.0,
                };
                (t.clone(), s)
            })
            .collect();
        Ok(TaskScores { scores })
    }

    async fn train_router(&self, exemplars: &[(ExpertId, Vec<f32>)]) -> Result<LearnedRouter> {
        self.trained_on.lock().unwrap().push(exemplars.to_vec());
        let mut experts: Vec<RouterExpert> = Vec::new();
        for (id, _) in exemplars {
            if experts.iter().any(|e| &e.id == id) {
                continue;
            }
            let mut centroid = [0.0f32; DIM];
            for (_, v) in exemplars.iter().filter(|(of, _)| of == id) {
                centroid.iter_mut().zip(v).for_each(|(c, x)| *c += x);
            }
            let norm = centroid.iter().map(|c| c * c).sum::<f32>().sqrt();
            experts.push(RouterExpert {
                id: id.clone(),
                centroid: centroid.iter().map(|c| c / norm).collect(),
            });
        }
        Ok(LearnedRouter {
            weights: vec![1.0; DIM],
            experts,
            temperature: 0.1,
            floor: 0.0,
        })
    }
}

fn shared(id: &str, exemplar: &str, capability: Vec<f32>) -> Expert {
    Expert {
        id: ExpertId::new(format!("expert:{id}")),
        name: id.into(),
        base_model: "code-base".into(),
        artifact_uri: format!("adapters/expert:{id}"),
        capability_card: serde_json::json!({ "exemplars": [exemplar] }),
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

#[tokio::test]
async fn the_gate_is_retrained_on_each_tasks_clear_winner() -> Result<()> {
    let store = Store::connect_memory(DIM).await?;
    expert::insert(
        &store,
        &shared("alpha", "alpha task", vec![1.0, 0.0, 0.0, 0.0]),
    )
    .await?;
    expert::insert(
        &store,
        &shared("beta", "beta task", vec![0.0, 1.0, 0.0, 0.0]),
    )
    .await?;
    // A win recorded under an earlier population, on a task this one ties on.
    contribution::upsert_outcome(
        &store,
        &RoutingOutcome {
            task: "alpha-one".into(),
            prompt: "alpha one".into(),
            winner: ExpertId::new("expert:alpha"),
            score: 1.0,
            runner_up: 0.5,
            run_id: RunId::new("run:earlier"),
            generation: Generation::ZERO,
            at: Utc::now(),
        },
    )
    .await?;
    let trainer = Scorer::default();
    let cfg = LoopConfig {
        contribution: Some(ContributionPolicy {
            every: 1,
            seeds: 2,
            max_tasks: 16,
            baseline: true,
        }),
        route_on_outcomes: true,
        ..LoopConfig::default()
    };
    let reports = GenerationLoop::new(&store, &trainer, &Places, cfg)
        .run_until(&RunId::new("run:outcomes"), 1)
        .await?;

    // Beta beat alpha on two tasks. The alpha task is a tie now, so the win
    // recorded on it before is forgotten.
    assert_eq!(reports[0].routing_outcomes, 2);
    let won: Vec<(String, String)> = contribution::outcomes(&store)
        .await?
        .into_iter()
        .map(|o| (o.task, o.winner.as_str().to_string()))
        .collect();
    assert_eq!(
        won,
        vec![
            ("beta-one".to_string(), "expert:beta".to_string()),
            ("mixed-one".to_string(), "expert:beta".to_string()),
        ]
    );
    // The gate was retrained with the won tasks as the winner's exemplars.
    let last = trainer.trained_on.lock().unwrap().last().cloned().unwrap();
    assert!(last
        .iter()
        .any(|(id, v)| id.as_str() == "expert:beta" && v == &vec![0.9, 0.3, 0.0, 0.0]));
    Ok(())
}

/// By default the winners are recorded and the gate does not learn them: on
/// the comparison the record set (ADR-0024 D-1), the outcome-trained router
/// scored no higher on the withheld tasks.
#[tokio::test]
async fn by_default_the_winners_are_recorded_and_not_routed_on() -> Result<()> {
    let store = Store::connect_memory(DIM).await?;
    expert::insert(
        &store,
        &shared("alpha", "alpha task", vec![1.0, 0.0, 0.0, 0.0]),
    )
    .await?;
    expert::insert(
        &store,
        &shared("beta", "beta task", vec![0.0, 1.0, 0.0, 0.0]),
    )
    .await?;
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
    let reports = GenerationLoop::new(&store, &trainer, &Places, cfg)
        .run_until(&RunId::new("run:outcomes-off"), 1)
        .await?;
    assert_eq!(reports[0].routing_outcomes, 2);
    assert_eq!(contribution::outcomes(&store).await?.len(), 2);
    let trained = trainer.trained_on.lock().unwrap();
    assert!(
        trained
            .iter()
            .flatten()
            .all(|(_, v)| v != &vec![0.9, 0.3, 0.0, 0.0]),
        "no router learned a won task"
    );
    Ok(())
}
