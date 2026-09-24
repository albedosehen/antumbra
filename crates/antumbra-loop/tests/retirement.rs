//! Retirement as the loop's job, through the loop (ADR-0022 S-5): an expert
//! whose leave-one-out contribution stays at nothing is demoted once that
//! persists, with its measurements as the evidence; one that contributes is
//! not; and a revive starts its stream afresh.

use async_trait::async_trait;
use chrono::Utc;

use antumbra_core::ports::{
    Embedder, EvaluateRequest, TaskPrompt, TaskScores, TrainOutcome, TrainRequest, Trainer,
};
use antumbra_core::slice::Holdout;
use antumbra_core::testing::ScriptedTrainer;
use antumbra_core::{Expert, ExpertId, ExpertStatus, Generation, Result, RunId, TransitionCause};
use antumbra_loop::{ContributionPolicy, GenerationLoop, LoopConfig, RetirementPolicy};
use antumbra_store::repo::{expert, lifecycle};
use antumbra_store::Store;

const DIM: usize = 4;

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

/// Graduates nothing. The alpha expert solves its tasks; the beta expert does
/// no better on its tasks than the base model, which gets half of everything.
struct Scorer;

#[async_trait]
impl Trainer for Scorer {
    async fn train_shadow(&self, req: TrainRequest) -> Result<TrainOutcome> {
        ScriptedTrainer::collapsing().train_shadow(req).await
    }

    async fn live_tasks(&self, _holdout: Option<Holdout>) -> Result<Vec<TaskPrompt>> {
        Ok(["alpha one", "alpha two", "beta one", "beta two"]
            .iter()
            .map(|p| TaskPrompt {
                id: p.replace(' ', "-"),
                prompt: p.to_string(),
                region: p.split(' ').next().unwrap_or_default().to_string(),
            })
            .collect())
    }

    async fn evaluate(&self, req: EvaluateRequest) -> Result<TaskScores> {
        let alpha = req.adapter_uri.as_deref() == Some("adapters/expert:alpha");
        Ok(TaskScores {
            scores: req
                .task_ids
                .iter()
                .map(|t| (t.clone(), if alpha { 1.0 } else { 0.5 }))
                .collect(),
        })
    }
}

fn shared(id: &str, axis: usize) -> Expert {
    let mut v = vec![0.0f32; DIM];
    v[axis] = 1.0;
    Expert {
        id: ExpertId::new(id),
        name: id.into(),
        base_model: "code-base".into(),
        artifact_uri: format!("adapters/{id}"),
        capability_card: serde_json::json!({}),
        capability_vec: Some(v),
        fitness: 1.0,
        frozen_at: Some(Utc::now()),
        generation: Generation::ZERO,
        owner: None,
        compartment: None,
        placed_on: None,
        created_at: Utc::now(),
    }
}

fn retiring() -> LoopConfig {
    LoopConfig {
        contribution: Some(ContributionPolicy {
            every: 1,
            seeds: 1,
            max_tasks: 8,
            baseline: false,
        }),
        retirement: Some(RetirementPolicy::default()),
        ..LoopConfig::default()
    }
}

#[tokio::test]
async fn a_persistently_useless_expert_is_demoted_and_a_useful_one_is_not() -> Result<()> {
    let store = Store::connect_memory(DIM).await?;
    expert::insert(&store, &shared("expert:alpha", 0)).await?;
    expert::insert(&store, &shared("expert:beta", 1)).await?;
    let (alpha, beta) = (ExpertId::new("expert:alpha"), ExpertId::new("expert:beta"));
    let run = RunId::new("run:retire");
    let lp = GenerationLoop::new(&store, &Scorer, &Axes, retiring());

    let reports = lp.run_until(&run, 3).await?;
    assert!(reports[0].detection.demoted.is_empty() && reports[1].detection.demoted.is_empty());
    let demoted = &reports[2].detection.demoted;
    assert_eq!(demoted.len(), 1, "{demoted:?}");
    assert_eq!(demoted[0].expert, beta);
    assert_eq!(
        (demoted[0].from, demoted[0].to),
        (ExpertStatus::Active, ExpertStatus::Dormant)
    );
    assert_eq!(
        demoted[0].cause,
        TransitionCause::Stale {
            generations: vec![Generation(0), Generation(1), Generation(2)],
            contributions: vec![0.0, 0.0, 0.0],
        },
        "demoted on the evidence of its three measurements"
    );
    assert_eq!(
        lifecycle::status_of(&store, &alpha).await?,
        ExpertStatus::Active
    );
    assert_eq!(
        lifecycle::status_of(&store, &beta).await?,
        ExpertStatus::Dormant
    );

    // A person revives it: its stream starts afresh, and one more measurement
    // at nothing is not yet persistent.
    lifecycle::transition(
        &store,
        &beta,
        ExpertStatus::Active,
        TransitionCause::Operator { note: None },
        None,
    )
    .await?;
    let more = lp.run_until(&run, 4).await?;
    assert!(more[0].detection.demoted.is_empty());
    assert_eq!(
        lifecycle::status_of(&store, &beta).await?,
        ExpertStatus::Active
    );
    Ok(())
}

#[tokio::test]
async fn without_a_policy_nothing_is_demoted() -> Result<()> {
    let store = Store::connect_memory(DIM).await?;
    expert::insert(&store, &shared("expert:beta", 1)).await?;
    let cfg = LoopConfig {
        retirement: None,
        ..retiring()
    };
    let reports = GenerationLoop::new(&store, &Scorer, &Axes, cfg)
        .run_until(&RunId::new("run:keep"), 3)
        .await?;
    assert!(reports.iter().all(|r| r.detection.demoted.is_empty()));
    assert_eq!(
        lifecycle::status_of(&store, &ExpertId::new("expert:beta")).await?,
        ExpertStatus::Active
    );
    Ok(())
}
