//! Conservative merging through the loop: the most similar pair
//! of active shared experts is merged only when their adapters are siblings and
//! the merge scores at least as well as the better of them; both are then
//! archived as redundant with the merged expert, so the merge is undoable.

use async_trait::async_trait;
use chrono::Utc;

use antumbra_core::ports::{
    Embedder, EvaluateRequest, MergeOutcome, MergeRequest, TaskPrompt, TaskScores, TrainOutcome,
    TrainRequest, Trainer,
};
use antumbra_core::slice::Holdout;
use antumbra_core::testing::ScriptedTrainer;
use antumbra_core::{Expert, ExpertId, ExpertStatus, Generation, Result, RunId, TransitionCause};
use antumbra_loop::{GenerationLoop, LoopConfig, Merge, MergePolicy};
use antumbra_store::repo::{expert, lifecycle};
use antumbra_store::Store;

const DIM: usize = 4;

struct Zero;

#[async_trait]
impl Embedder for Zero {
    async fn embed(&self, _text: &str) -> Result<Vec<f32>> {
        Ok(vec![0.0; DIM])
    }
    fn dim(&self) -> usize {
        DIM
    }
}

/// Graduates nothing; merges with a scripted overlap; scores the merged
/// adapter at `merged`, the first expert at 0.6 and the second at 0.5.
struct Merger {
    retained: f32,
    merged: f32,
}

#[async_trait]
impl Trainer for Merger {
    async fn train_shadow(&self, req: TrainRequest) -> Result<TrainOutcome> {
        ScriptedTrainer::collapsing().train_shadow(req).await
    }

    async fn live_tasks(&self, _holdout: Option<Holdout>) -> Result<Vec<TaskPrompt>> {
        Ok((0..3)
            .map(|i| TaskPrompt {
                id: format!("t{i}"),
                prompt: format!("task {i}"),
                region: "tasks".into(),
            })
            .collect())
    }

    async fn evaluate(&self, req: EvaluateRequest) -> Result<TaskScores> {
        let uri = req.adapter_uri.unwrap_or_default();
        let s = if uri.contains("merged") {
            self.merged
        } else if uri.ends_with("expert:one") {
            0.6
        } else {
            0.5
        };
        Ok(TaskScores {
            scores: req.task_ids.iter().map(|t| (t.clone(), s)).collect(),
        })
    }

    async fn merge(&self, _req: MergeRequest) -> Result<MergeOutcome> {
        Ok(MergeOutcome {
            rank: 16,
            retained: self.retained,
        })
    }
}

fn shared(id: &str, capability: Vec<f32>) -> Expert {
    Expert {
        id: ExpertId::new(id),
        name: id.into(),
        base_model: "code-base".into(),
        artifact_uri: format!("adapters/{id}"),
        capability_card: serde_json::json!({ "exemplars": [format!("{id} exemplar")] }),
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

async fn siblings() -> Result<Store> {
    let store = Store::connect_memory(DIM).await?;
    expert::insert(&store, &shared("expert:one", vec![1.0, 0.0, 0.0, 0.0])).await?;
    expert::insert(&store, &shared("expert:two", vec![0.95, 0.31, 0.0, 0.0])).await?;
    Ok(store)
}

async fn run(store: &Store, trainer: &Merger) -> Result<Option<Merge>> {
    let cfg = LoopConfig {
        merge: Some(MergePolicy::default()),
        ..LoopConfig::default()
    };
    let mut reports = GenerationLoop::new(store, trainer, &Zero, cfg)
        .run_until(&RunId::new("run:merge"), 1)
        .await?;
    Ok(reports.remove(0).merge)
}

async fn status(store: &Store, id: &str) -> Result<ExpertStatus> {
    lifecycle::status_of(store, &ExpertId::new(id)).await
}

#[tokio::test]
async fn siblings_that_merge_at_no_cost_become_one_and_are_archived() -> Result<()> {
    let store = siblings().await?;
    let merge = run(
        &store,
        &Merger {
            retained: 0.97,
            merged: 0.6,
        },
    )
    .await?;
    let Some(Merge::Merged {
        into,
        pair,
        retained,
        merged,
        better,
        ..
    }) = merge
    else {
        panic!("merged, got {merge:?}");
    };
    assert_eq!(into, ExpertId::new("expert:run:merge:g0:merged"));
    assert_eq!(
        pair,
        (ExpertId::new("expert:one"), ExpertId::new("expert:two"))
    );
    assert_eq!((retained, merged, better), (0.97, 0.6, 0.6));
    for id in ["expert:one", "expert:two"] {
        assert_eq!(status(&store, id).await?, ExpertStatus::Archived);
        let moved = lifecycle::history(&store, &ExpertId::new(id)).await?;
        assert_eq!(
            moved[0].cause,
            TransitionCause::Redundant { of: into.clone() }
        );
    }
    let routable: Vec<ExpertId> = lifecycle::routable(&store)
        .await?
        .into_iter()
        .map(|e| e.id)
        .collect();
    assert_eq!(routable, std::slice::from_ref(&into));
    let card = expert::get(&store, &into)
        .await?
        .expect("the merged expert");
    assert_eq!(card.capability_card["merged_from"][1], "expert:two");
    assert_eq!(
        card.capability_card["exemplars"].as_array().map(Vec::len),
        Some(2)
    );
    Ok(())
}

#[tokio::test]
async fn adapters_that_are_not_siblings_are_left_apart() -> Result<()> {
    let store = siblings().await?;
    let merge = run(
        &store,
        &Merger {
            retained: 0.6,
            merged: 0.9,
        },
    )
    .await?;
    assert!(
        matches!(merge, Some(Merge::NotSiblings { .. })),
        "{merge:?}"
    );
    assert_eq!(status(&store, "expert:one").await?, ExpertStatus::Active);
    assert_eq!(status(&store, "expert:two").await?, ExpertStatus::Active);
    assert_eq!(lifecycle::routable(&store).await?.len(), 2);
    Ok(())
}

#[tokio::test]
async fn a_merge_that_scores_below_the_better_original_is_not_made() -> Result<()> {
    let store = siblings().await?;
    let merge = run(
        &store,
        &Merger {
            retained: 0.97,
            merged: 0.55,
        },
    )
    .await?;
    let Some(Merge::Costly { merged, better, .. }) = merge else {
        panic!("costly, got {merge:?}");
    };
    assert_eq!((merged, better), (0.55, 0.6));
    assert_eq!(lifecycle::routable(&store).await?.len(), 2);
    Ok(())
}

#[tokio::test]
async fn without_a_policy_nothing_merges() -> Result<()> {
    let store = siblings().await?;
    let trainer = Merger {
        retained: 1.0,
        merged: 1.0,
    };
    let reports = GenerationLoop::new(&store, &trainer, &Zero, LoopConfig::default())
        .run_until(&RunId::new("run:merge"), 1)
        .await?;
    assert!(reports[0].merge.is_none());
    assert_eq!(lifecycle::routable(&store).await?.len(), 2);
    Ok(())
}
