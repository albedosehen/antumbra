//! Leave-one-in admission through the loop (ADR-0022 S-5): a candidate that
//! beats the generalist on its own tasks is still turned away when adding it
//! makes the gate escalate others, because the population does worse on the
//! live tasks with it than without it. A trainer that can retrain the learned
//! router judges it under that router instead, which routes the tasks between
//! the two rather than escalating them.

use async_trait::async_trait;
use chrono::Utc;

use antumbra_core::ports::{
    Embedder, EvaluateRequest, TaskPrompt, TaskScores, TrainOutcome, TrainRequest, Trainer,
};
use antumbra_core::slice::Holdout;
use antumbra_core::testing::ScriptedTrainer;
use antumbra_core::{
    cosine_similarity, Expert, ExpertId, Generation, LearnedRouter, Result, RouterExpert, RunId,
};
use antumbra_loop::{Admission, AdmissionPolicy, GenerationLoop, LoopConfig};
use antumbra_store::repo::{expert, router};
use antumbra_store::Store;

const DIM: usize = 4;
const GENERALIST: [f32; DIM] = [1.0, 0.5, 0.0, 0.0];
const SPECIALIST: [f32; DIM] = [0.9, 1.0, 0.0, 0.0];

/// "beta" tasks sit near the specialist; "mixed" ones sit between the two,
/// where the gate's top-two margin is under its threshold once both are in.
struct Places;

#[async_trait]
impl Embedder for Places {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        Ok(if text.contains("specialist") {
            SPECIALIST.to_vec()
        } else if text.contains("beta") {
            vec![0.0, 1.0, 0.0, 0.0]
        } else if text.contains("mixed") {
            vec![0.7, 0.7, 0.0, 0.0]
        } else {
            vec![1.0, 0.0, 0.0, 0.0]
        })
    }
    fn dim(&self) -> usize {
        DIM
    }
}

/// Graduates the specialist; the generalist scores 0.8 everywhere, the
/// specialist 1.0 everywhere, the base model nothing.
struct Specialist;

#[async_trait]
impl Trainer for Specialist {
    async fn train_shadow(&self, req: TrainRequest) -> Result<TrainOutcome> {
        ScriptedTrainer {
            capability_exemplars: vec!["specialist exemplar".into()],
            ..ScriptedTrainer::graduating()
        }
        .train_shadow(req)
        .await
    }

    async fn live_tasks(&self, _holdout: Option<Holdout>) -> Result<Vec<TaskPrompt>> {
        let task = |id: &str, prompt: &str| TaskPrompt {
            id: id.into(),
            prompt: prompt.into(),
            region: prompt.split(' ').next().unwrap_or_default().into(),
        };
        Ok(vec![
            task("b1", "beta one"),
            task("b2", "beta two"),
            task("m1", "mixed one"),
            task("m2", "mixed two"),
            task("m3", "mixed three"),
            task("m4", "mixed four"),
        ])
    }

    async fn evaluate(&self, req: EvaluateRequest) -> Result<TaskScores> {
        let s = match req.adapter_uri.as_deref() {
            Some("adapters/generalist") => 0.8,
            None => 0.0,
            Some(_) => 1.0,
        };
        Ok(TaskScores {
            scores: req.task_ids.iter().map(|t| (t.clone(), s)).collect(),
        })
    }
}

/// The specialist's trainer, able to retrain the learned router too: each
/// expert's centroid is the mean of its exemplars, and a task is covered when
/// it sits within 60 degrees of one.
struct Routing;

#[async_trait]
impl Trainer for Routing {
    async fn train_shadow(&self, req: TrainRequest) -> Result<TrainOutcome> {
        Specialist.train_shadow(req).await
    }

    async fn live_tasks(&self, holdout: Option<Holdout>) -> Result<Vec<TaskPrompt>> {
        Specialist.live_tasks(holdout).await
    }

    async fn evaluate(&self, req: EvaluateRequest) -> Result<TaskScores> {
        Specialist.evaluate(req).await
    }

    async fn train_router(&self, exemplars: &[(ExpertId, Vec<f32>)]) -> Result<LearnedRouter> {
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
            floor: 0.5,
        })
    }
}

async fn population(card: serde_json::Value) -> Result<Store> {
    let store = Store::connect_memory(DIM).await?;
    expert::insert(
        &store,
        &Expert {
            id: ExpertId::new("expert:generalist"),
            name: "generalist".into(),
            base_model: "code-base".into(),
            artifact_uri: "adapters/generalist".into(),
            capability_card: card,
            capability_vec: Some(GENERALIST.to_vec()),
            fitness: 1.0,
            frozen_at: Some(Utc::now()),
            generation: Generation::ZERO,
            owner: None,
            compartment: None,
            placed_on: None,
            created_at: Utc::now(),
        },
    )
    .await?;
    Ok(store)
}

fn admitting() -> LoopConfig {
    LoopConfig {
        admission: Some(AdmissionPolicy::default()),
        ..LoopConfig::default()
    }
}

#[tokio::test]
async fn a_candidate_that_makes_the_gate_escalate_others_is_turned_away() -> Result<()> {
    let store = population(serde_json::json!({})).await?;
    let reports = GenerationLoop::new(&store, &Specialist, &Places, admitting())
        .run_until(&RunId::new("run:loi"), 1)
        .await?;
    // With it, the two beta tasks go to it (1.0 each) and the four mixed ones
    // escalate to the base model (nothing); without it, all six go to the
    // generalist (0.8 each).
    assert_eq!(
        reports[0].admission,
        Some(Admission::Outserved {
            tasks: 6,
            escalated: 4,
            candidate: 2.0 / 6.0,
            serving: 0.8,
        })
    );
    assert!(!reports[0].graduated);
    assert!(router::load(&store).await?.is_none());
    Ok(())
}

#[tokio::test]
async fn under_a_retrained_router_the_tasks_between_the_two_are_routed() -> Result<()> {
    let generalist = ExpertId::new("expert:generalist");
    let store = population(serde_json::json!({"exemplars": ["generalist exemplar"]})).await?;
    let reports = GenerationLoop::new(&store, &Routing, &Places, admitting())
        .run_until(&RunId::new("run:loi-routed"), 1)
        .await?;
    // The retrained router sends the beta and the mixed tasks to the
    // specialist (1.0 each) instead of escalating the mixed ones: better than
    // the generalist's 0.8, so it joins.
    let similarity = cosine_similarity(&SPECIALIST, &GENERALIST);
    match &reports[0].admission {
        Some(Admission::Admitted {
            nearest: Some((id, s)),
        }) => {
            assert_eq!(id, &generalist);
            assert!((s - similarity).abs() < 1e-4, "{s} vs {similarity}");
        }
        other => panic!("not admitted: {other:?}"),
    }
    assert!(reports[0].graduated);
    // Serving routes under the router admission was judged with, stored as
    // soon as the specialist joined.
    let stored = router::load(&store).await?.expect("the gate was retrained");
    assert_eq!(stored.experts.len(), 2);
    assert!(stored.experts.iter().any(|e| e.id == generalist));
    Ok(())
}
