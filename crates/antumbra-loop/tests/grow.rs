//! The grow step through the loop (ADR-0022 S-3): each generation learns from
//! the region the latest census makes most learnable, gated, discounted for
//! redundancy with recent choices, with an unfiltered share of the whole
//! visible slice; and the census is taken even before there is an expert.

use std::sync::Mutex;

use async_trait::async_trait;

use antumbra_core::ports::{
    Embedder, EvaluateRequest, TaskPrompt, TaskScores, TrainOutcome, TrainRequest, Trainer,
};
use antumbra_core::slice::Holdout;
use antumbra_core::testing::ScriptedTrainer;
use antumbra_core::{Generation, Result, RunId};
use antumbra_loop::{ContributionPolicy, GenerationLoop, GrowPolicy, LoopConfig};
use antumbra_store::repo::grow;
use antumbra_store::Store;

const DIM: usize = 4;
const REGIONS: [(&str, f32); 4] = [("half", 0.5), ("third", 0.4), ("easy", 0.9), ("never", 0.0)];

/// Puts each region on its own axis.
struct Axes;

#[async_trait]
impl Embedder for Axes {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut v = vec![0.0f32; DIM];
        if let Some(i) = REGIONS.iter().position(|(r, _)| text.starts_with(r)) {
            v[i] = 1.0;
        }
        Ok(v)
    }
    fn dim(&self) -> usize {
        DIM
    }
}

/// Four tasks a region; the base model passes each at its region's rate.
/// Graduates nothing, and keeps the focus of every run it was asked for.
#[derive(Default)]
struct Regions {
    focus: Mutex<Vec<Vec<String>>>,
}

#[async_trait]
impl Trainer for Regions {
    async fn train_shadow(&self, req: TrainRequest) -> Result<TrainOutcome> {
        if let Ok(mut seen) = self.focus.lock() {
            seen.push(req.focus.clone());
        }
        ScriptedTrainer::collapsing().train_shadow(req).await
    }

    async fn live_tasks(&self, _holdout: Option<Holdout>) -> Result<Vec<TaskPrompt>> {
        Ok(REGIONS
            .iter()
            .flat_map(|(r, _)| {
                (0..4).map(move |i| TaskPrompt {
                    id: format!("{r}-{i}"),
                    prompt: format!("{r} task {i}"),
                    region: r.to_string(),
                })
            })
            .collect())
    }

    async fn evaluate(&self, req: EvaluateRequest) -> Result<TaskScores> {
        Ok(TaskScores {
            scores: req
                .task_ids
                .iter()
                .map(|t| {
                    let rate = REGIONS
                        .iter()
                        .find(|(r, _)| t.starts_with(r))
                        .map_or(0.0, |(_, p)| *p);
                    (t.clone(), rate)
                })
                .collect(),
        })
    }
}

fn growing() -> LoopConfig {
    LoopConfig {
        contribution: Some(ContributionPolicy {
            every: 1,
            seeds: 1,
            max_tasks: 64,
            baseline: true,
        }),
        grow: Some(GrowPolicy::default()),
        ..LoopConfig::default()
    }
}

#[tokio::test]
async fn each_generation_learns_from_the_region_the_census_makes_most_learnable() -> Result<()> {
    let store = Store::connect_memory(DIM).await?;
    let trainer = Regions::default();
    let run = RunId::new("run:grow");
    let reports = GenerationLoop::new(&store, &trainer, &Axes, growing())
        .run_until(&run, 3)
        .await?;

    // Generation 0 has no census yet: it learns from everything, and takes
    // the first census, of the base model alone.
    let first = reports[0].growth.as_ref().expect("decided");
    assert!(first.record.chosen.is_none() && first.focus.is_empty());
    let census: Vec<(&str, f32)> = reports[0]
        .census
        .iter()
        .map(|c| (c.region.as_str(), c.acceptability))
        .collect();
    assert_eq!(
        census,
        [("easy", 0.9), ("half", 0.5), ("never", 0.0), ("third", 0.4)]
    );

    // Generation 1 learns from the half-solved region, and a quarter of what
    // it learns is sampled from the rest.
    let second = reports[1].growth.as_ref().expect("decided");
    assert_eq!(second.record.chosen.as_deref(), Some("half"));
    assert_eq!((second.record.focus, second.record.unfiltered), (6, 2));
    let never = second
        .record
        .candidates
        .iter()
        .find(|c| c.region == "never");
    assert!(never.is_some_and(|c| !c.admitted), "the gate keeps it out");
    let asked = trainer.focus.lock().unwrap().clone();
    assert!(asked[0].is_empty());
    assert_eq!(asked[1].iter().filter(|t| t.starts_with("half")).count(), 4);

    // Generation 2: the half region, just chosen, gives up half its
    // learnability, so the next most learnable is chosen; and the first
    // choice's realized credit is recorded.
    let third = reports[2].growth.as_ref().expect("decided");
    assert_eq!(third.record.chosen.as_deref(), Some("third"));
    assert_eq!(
        third.record.credit,
        Some(0.0),
        "nothing graduated, nothing moved"
    );
    assert!(third.diversity.entropy > 0.0);

    let kept = grow::history(&store, &run).await?;
    assert_eq!(kept.len(), 3);
    assert_eq!(kept[2].census_generation, Some(Generation(1)));
    Ok(())
}

#[tokio::test]
async fn without_a_policy_every_run_learns_from_everything() -> Result<()> {
    let store = Store::connect_memory(DIM).await?;
    let trainer = Regions::default();
    let cfg = LoopConfig {
        grow: None,
        ..growing()
    };
    let reports = GenerationLoop::new(&store, &trainer, &Axes, cfg)
        .run_until(&RunId::new("run:plain"), 2)
        .await?;
    assert!(reports.iter().all(|r| r.growth.is_none()));
    assert!(trainer.focus.lock().unwrap().iter().all(Vec::is_empty));
    Ok(())
}
