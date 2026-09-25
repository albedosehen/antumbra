//! Leave-one-in admission through the loop (ADR-0022 S-5): a candidate that
//! beats the generalist on its own tasks is still turned away when adding it
//! makes the gate escalate others, because the population does worse on the
//! live tasks with it than without it.

use async_trait::async_trait;
use chrono::Utc;

use antumbra_core::ports::{
    Embedder, EvaluateRequest, TaskPrompt, TaskScores, TrainOutcome, TrainRequest, Trainer,
};
use antumbra_core::slice::Holdout;
use antumbra_core::testing::ScriptedTrainer;
use antumbra_core::{Expert, ExpertId, Generation, Result, RunId};
use antumbra_loop::{Admission, AdmissionPolicy, GenerationLoop, LoopConfig};
use antumbra_store::repo::expert;
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

#[tokio::test]
async fn a_candidate_that_makes_the_gate_escalate_others_is_turned_away() -> Result<()> {
    let store = Store::connect_memory(DIM).await?;
    expert::insert(
        &store,
        &Expert {
            id: ExpertId::new("expert:generalist"),
            name: "generalist".into(),
            base_model: "code-base".into(),
            artifact_uri: "adapters/generalist".into(),
            capability_card: serde_json::json!({}),
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
    let cfg = LoopConfig {
        admission: Some(AdmissionPolicy::default()),
        ..LoopConfig::default()
    };
    let reports = GenerationLoop::new(&store, &Specialist, &Places, cfg)
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
    Ok(())
}
