//! Admission gating through the loop (ADR-0022 S-5): a graduate that
//! duplicates an active shared expert joins only by beating it head to head,
//! and then replaces it; one that cannot is not admitted.

use async_trait::async_trait;
use chrono::Utc;

use antumbra_core::ports::{
    Embedder, EvaluateRequest, TaskPrompt, TaskScores, TrainOutcome, TrainRequest, Trainer,
};
use antumbra_core::slice::Holdout;
use antumbra_core::testing::ScriptedTrainer;
use antumbra_core::{
    Expert, ExpertId, ExpertStatus, Generation, Result, RunId, ShadowStatus, TransitionCause,
};
use antumbra_loop::{Admission, AdmissionPolicy, GenerationLoop, LoopConfig};
use antumbra_store::repo::{expert, lifecycle, shadow};
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

/// Graduates a shadow whose exemplars are all `skill` tasks, and scores the
/// adapters it is asked about: the old expert at `old`, the base model at
/// nothing, and anything else (the candidate) at `new`.
struct Graduating {
    skill: &'static str,
    old: f32,
    new: f32,
}

#[async_trait]
impl Trainer for Graduating {
    async fn train_shadow(&self, req: TrainRequest) -> Result<TrainOutcome> {
        ScriptedTrainer {
            capability_exemplars: vec![format!("{} task", self.skill)],
            ..ScriptedTrainer::graduating()
        }
        .train_shadow(req)
        .await
    }

    async fn live_tasks(&self, _holdout: Option<Holdout>) -> Result<Vec<TaskPrompt>> {
        Ok((0..4)
            .map(|i| TaskPrompt {
                id: format!("t{i}"),
                prompt: format!("{} {i}", self.skill),
                region: self.skill.to_string(),
            })
            .collect())
    }

    async fn evaluate(&self, req: EvaluateRequest) -> Result<TaskScores> {
        let s = match req.adapter_uri.as_deref() {
            Some("adapters/expert:old") => self.old,
            None => 0.0,
            Some(_) => self.new,
        };
        Ok(TaskScores {
            scores: req.task_ids.iter().map(|t| (t.clone(), s)).collect(),
        })
    }
}

async fn with_an_alpha_expert() -> Result<Store> {
    let mut v = vec![0.0f32; DIM];
    v[0] = 1.0;
    with_an_old_expert(v).await
}

/// An old expert whose capability is `v`.
async fn with_an_old_expert(v: Vec<f32>) -> Result<Store> {
    let store = Store::connect_memory(DIM).await?;
    expert::insert(
        &store,
        &Expert {
            id: ExpertId::new("expert:old"),
            name: "old".into(),
            base_model: "code-base".into(),
            artifact_uri: "adapters/expert:old".into(),
            capability_card: serde_json::json!({}),
            capability_vec: Some(v),
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

fn gated() -> LoopConfig {
    LoopConfig {
        admission: Some(AdmissionPolicy::default()),
        ..LoopConfig::default()
    }
}

async fn run(store: &Store, trainer: &Graduating) -> Result<antumbra_loop::GenerationReport> {
    let mut reports = GenerationLoop::new(store, trainer, &Axes, gated())
        .run_until(&RunId::new("run:admit"), 1)
        .await?;
    Ok(reports.remove(0))
}

fn candidate() -> ExpertId {
    ExpertId::new("expert:run:admit:g0")
}

#[tokio::test]
async fn a_distinct_skill_is_admitted() -> Result<()> {
    let store = with_an_alpha_expert().await?;
    let trainer = Graduating {
        skill: "beta",
        old: 0.9,
        new: 0.1,
    };
    let report = run(&store, &trainer).await?;
    assert!(report.graduated);
    assert!(matches!(report.admission, Some(Admission::Admitted { .. })));
    assert!(expert::get(&store, &candidate()).await?.is_some());
    assert_eq!(
        lifecycle::status_of(&store, &ExpertId::new("expert:old")).await?,
        ExpertStatus::Active
    );
    Ok(())
}

#[tokio::test]
async fn a_twin_that_scores_better_replaces_the_one_it_duplicates() -> Result<()> {
    let store = with_an_alpha_expert().await?;
    let trainer = Graduating {
        skill: "alpha",
        old: 0.5,
        new: 0.75,
    };
    let report = run(&store, &trainer).await?;
    assert!(report.graduated);
    let Some(Admission::Superseded {
        archived,
        similarity,
        candidate: mine,
        incumbent,
    }) = report.admission
    else {
        panic!("superseded, got {:?}", report.admission);
    };
    assert_eq!(archived, ExpertId::new("expert:old"));
    assert!(similarity > 0.99);
    assert_eq!((mine, incumbent), (0.75, 0.5));
    let old = ExpertId::new("expert:old");
    assert_eq!(
        lifecycle::status_of(&store, &old).await?,
        ExpertStatus::Archived
    );
    let moved = lifecycle::history(&store, &old).await?;
    assert_eq!(
        moved[0].cause,
        TransitionCause::Redundant { of: candidate() },
        "archived as redundant with the one that replaced it"
    );
    assert_eq!(moved[0].generation, Some(Generation(0)));
    let routable: Vec<ExpertId> = lifecycle::routable(&store)
        .await?
        .into_iter()
        .map(|e| e.id)
        .collect();
    assert_eq!(routable, [candidate()]);
    Ok(())
}

#[tokio::test]
async fn a_twin_that_scores_no_better_is_not_admitted() -> Result<()> {
    let store = with_an_alpha_expert().await?;
    let trainer = Graduating {
        skill: "alpha",
        old: 0.5,
        new: 0.5,
    };
    let report = run(&store, &trainer).await?;
    assert!(!report.graduated, "cleared graduation, but not admission");
    assert!(matches!(
        report.admission,
        Some(Admission::Rejected {
            candidate: Some(_),
            incumbent: Some(_),
            ..
        })
    ));
    assert!(expert::get(&store, &candidate()).await?.is_none());
    let sh = shadow::get(&store, &report.shadow)
        .await?
        .expect("a shadow");
    assert_eq!(sh.status, ShadowStatus::Pruned);
    assert_eq!(
        lifecycle::status_of(&store, &ExpertId::new("expert:old")).await?,
        ExpertStatus::Active
    );
    Ok(())
}

/// A candidate that duplicates nothing, but does worse on the tasks it trained
/// for than the generalist the gate routes them to, is not admitted; one that
/// does better is.
#[tokio::test]
async fn a_candidate_must_beat_what_already_serves_its_tasks() -> Result<()> {
    let generalist = vec![0.7, 0.7, 0.0, 0.0];
    let store = with_an_old_expert(generalist.clone()).await?;
    let worse = Graduating {
        skill: "beta",
        old: 0.9,
        new: 0.4,
    };
    let report = run(&store, &worse).await?;
    assert!(!report.graduated);
    assert_eq!(
        report.admission,
        Some(Admission::Outserved {
            tasks: 4,
            candidate: 0.4,
            serving: 0.9,
        })
    );
    assert!(expert::get(&store, &candidate()).await?.is_none());

    let store = with_an_old_expert(generalist).await?;
    let better = Graduating {
        skill: "beta",
        old: 0.9,
        new: 0.95,
    };
    let report = run(&store, &better).await?;
    assert!(report.graduated);
    assert!(matches!(report.admission, Some(Admission::Admitted { .. })));
    Ok(())
}

/// Without a policy, a twin joins as every graduate always has.
#[tokio::test]
async fn without_a_policy_every_graduate_joins() -> Result<()> {
    let store = with_an_alpha_expert().await?;
    let trainer = Graduating {
        skill: "alpha",
        old: 0.9,
        new: 0.1,
    };
    let reports = GenerationLoop::new(&store, &trainer, &Axes, LoopConfig::default())
        .run_until(&RunId::new("run:admit"), 1)
        .await?;
    assert!(reports[0].graduated && reports[0].admission.is_none());
    assert_eq!(lifecycle::routable(&store).await?.len(), 2);
    Ok(())
}
