//! The byte-identity tripwire through retirement: a demoted or
//! archived expert keeps its weights, and they are still held to their
//! freeze; only a deleted expert, whose weights may be gone, is not checked.

use chrono::Utc;

use antumbra_core::testing::{FixedEmbedder, ScriptedTrainer};
use antumbra_core::{
    EvalStatus, EvaluationRun, Expert, ExpertId, ExpertStatus, Generation, Result, RunId,
    SubjectKind, TransitionCause,
};
use antumbra_loop::{GenerationLoop, LoopConfig};
use antumbra_store::repo::{evaluation, expert, lifecycle};
use antumbra_store::Store;

/// A frozen expert whose freeze baseline no longer matches its artifact.
async fn drifted(store: &Store, run: &RunId, name: &str) -> Result<ExpertId> {
    let id = ExpertId::new(format!("expert:{name}"));
    let now = Utc::now();
    expert::insert(
        store,
        &Expert {
            id: id.clone(),
            name: name.into(),
            base_model: "code-base".into(),
            artifact_uri: format!("memory://adapter/{name}"),
            capability_card: serde_json::json!({}),
            capability_vec: None,
            fitness: 0.9,
            frozen_at: Some(now),
            generation: Generation::ZERO,
            owner: None,
            compartment: None,
            placed_on: None,
            created_at: now,
        },
    )
    .await?;
    evaluation::insert(
        store,
        &EvaluationRun {
            run_id: run.clone(),
            subject_kind: SubjectKind::Expert,
            subject_id: id.to_string(),
            corpus_task_id: "freeze:g0".into(),
            status: EvalStatus::Success,
            metrics: None,
            regression_fingerprint: Some("sha256:the-bytes-it-was-frozen-with".into()),
            created_at: now,
        },
    )
    .await?;
    Ok(id)
}

#[tokio::test]
async fn demoted_experts_stay_under_the_tripwire_and_only_deleted_ones_leave_it() -> Result<()> {
    let store = Store::connect_memory(8).await?;
    let run = RunId::new("run:lifecycle");
    let operator = || TransitionCause::Operator { note: None };
    let dormant = drifted(&store, &run, "dormant").await?;
    let archived = drifted(&store, &run, "archived").await?;
    let deleted = drifted(&store, &run, "deleted").await?;
    let covering = drifted(&store, &run, "covering").await?;
    lifecycle::transition(&store, &dormant, ExpertStatus::Dormant, operator(), None).await?;
    lifecycle::transition(&store, &archived, ExpertStatus::Archived, operator(), None).await?;
    lifecycle::transition(&store, &deleted, ExpertStatus::Dormant, operator(), None).await?;
    lifecycle::transition(
        &store,
        &deleted,
        ExpertStatus::Deleted,
        TransitionCause::Redundant {
            of: covering.clone(),
        },
        Some(Generation(2)),
    )
    .await?;

    let trainer = ScriptedTrainer::graduating();
    let embedder = FixedEmbedder::new(8);
    let lp = GenerationLoop::new(&store, &trainer, &embedder, LoopConfig::default());
    let mut caught = lp.check_no_forgetting(&run).await?;
    caught.sort_by(|a, b| a.as_str().cmp(b.as_str()));
    assert_eq!(caught, [archived, covering, dormant]);
    Ok(())
}
