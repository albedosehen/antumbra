//! Round-trip the shadow, reward, boundary, and evaluation repositories
//! against an embedded SurrealDB.

use anyhow::Result;
use chrono::{Duration, Utc};

use antumbra_core::{
    BoundaryId, EvalStatus, EvaluationRun, Expert, ExpertId, FailureBoundary, Generation,
    GenerationHead, Grain, LoopState, RewardSignal, RunId, Shadow, ShadowId, ShadowStatus,
    SubjectKind,
};
use antumbra_store::repo::{boundary, evaluation, expert, generation, reward, shadow};
use antumbra_store::testing;

#[tokio::test]
async fn shadow_lifecycle_persists() -> Result<()> {
    let store = testing::mem_store_of(4).await?;
    let mut s = Shadow::spawn(
        ShadowId::new("shadow:g0"),
        Generation::ZERO,
        Some(ExpertId::new("expert:a")),
        Utc::now(),
    );
    shadow::upsert(&store, &s).await?;

    let Some(got) = shadow::get(&store, &s.id).await? else {
        anyhow::bail!("the shadow does not read back");
    };
    assert_eq!(got.status, ShadowStatus::Spawning);
    assert_eq!(got.parent_expert, Some(ExpertId::new("expert:a")));

    s.advance_to(ShadowStatus::Exploring)?;
    shadow::upsert(&store, &s).await?;
    assert_eq!(
        shadow::list_by_status(&store, ShadowStatus::Exploring)
            .await?
            .len(),
        1
    );
    assert!(shadow::list_by_status(&store, ShadowStatus::Spawning)
        .await?
        .is_empty());
    Ok(())
}

#[tokio::test]
async fn reward_signals_persist_per_run() -> Result<()> {
    let store = testing::mem_store_of(4).await?;
    let run = RunId::new("run:1");
    let now = Utc::now();
    let signals = vec![
        RewardSignal::verifier(run.clone(), 0, "tests", 1.0, now),
        RewardSignal::critic(run.clone(), 0, "critic", 0.5, now),
    ];
    reward::insert_many(&store, &signals).await?;

    let loaded = reward::list_by_run(&store, &run).await?;
    assert_eq!(loaded.len(), 2);
    assert_eq!(loaded.iter().filter(|s| s.is_verifier()).count(), 1);
    Ok(())
}

#[tokio::test]
async fn boundary_persists_and_recalls_by_context() -> Result<()> {
    let store = testing::mem_store_of(4).await?;
    let b = FailureBoundary {
        id: BoundaryId::new("b:1"),
        behavior: "npm install".into(),
        fail_context: serde_json::json!({ "runtime": "deno" }),
        near_ok_context: Some(serde_json::json!({ "runtime": "node" })),
        governing_features: vec!["runtime".into()],
        grain: Some(Grain::Project),
        context_vec: Some(vec![1.0, 0.0, 0.0, 0.0]),
        ok_context_vec: Some(vec![0.0, 1.0, 0.0, 0.0]),
        confidence: 0.8,
        generation: Generation::ZERO,
        owner: None,
        created_at: Utc::now(),
    };
    boundary::upsert(&store, &b).await?;

    assert_eq!(boundary::list(&store).await?.len(), 1);
    let near = boundary::knn_by_context(&store, &[0.9, 0.1, 0.0, 0.0], 1).await?;
    assert_eq!(near.len(), 1);
    assert!(near[0].is_actionable());
    Ok(())
}

#[tokio::test]
async fn evaluation_runs_track_latest() -> Result<()> {
    let store = testing::mem_store_of(4).await?;
    let earlier = Utc::now() - Duration::seconds(10);
    let later = Utc::now();

    for (ts, fp) in [(earlier, "aaa"), (later, "bbb")] {
        evaluation::insert(
            &store,
            &EvaluationRun {
                run_id: RunId::new("run:eval"),
                subject_kind: SubjectKind::Expert,
                subject_id: "expert:a".into(),
                corpus_task_id: "task:1".into(),
                status: EvalStatus::Success,
                metrics: None,
                regression_fingerprint: Some(fp.into()),
                created_at: ts,
            },
        )
        .await?;
    }

    let all = evaluation::list_for_subject(&store, SubjectKind::Expert, "expert:a").await?;
    assert_eq!(all.len(), 2);
    let Some(latest) =
        evaluation::latest_for_subject(&store, SubjectKind::Expert, "expert:a").await?
    else {
        anyhow::bail!("no latest evaluation for the subject");
    };
    assert_eq!(latest.regression_fingerprint.as_deref(), Some("bbb"));
    Ok(())
}

#[tokio::test]
async fn expert_population_crud_and_knn() -> Result<()> {
    let store = testing::mem_store_of(4).await?;
    let now = Utc::now();
    let mk = |key: &str, vec: Vec<f32>| Expert {
        id: ExpertId::new(key),
        name: key.into(),
        base_model: "base".into(),
        artifact_uri: format!("mem://{key}"),
        capability_card: serde_json::Value::Null,
        capability_vec: Some(vec),
        fitness: 1.0,
        frozen_at: Some(now),
        generation: Generation::ZERO,
        owner: None,
        compartment: None,
        placed_on: None,
        created_at: now,
    };
    expert::insert(&store, &mk("expert:a", vec![1.0, 0.0, 0.0, 0.0])).await?;
    expert::insert(&store, &mk("expert:b", vec![0.0, 1.0, 0.0, 0.0])).await?;

    assert_eq!(expert::list(&store).await?.len(), 2);
    let Some(got) = expert::get(&store, &ExpertId::new("expert:a")).await? else {
        anyhow::bail!("expert:a does not read back");
    };
    assert_eq!(got.name, "expert:a");
    assert!(expert::get(&store, &ExpertId::new("expert:missing"))
        .await?
        .is_none());

    let near = expert::knn_by_capability(&store, &[0.9, 0.1, 0.0, 0.0], 1).await?;
    assert_eq!(near.len(), 1);
    assert_eq!(near[0].id, ExpertId::new("expert:a"));
    Ok(())
}

#[tokio::test]
async fn generation_head_round_trips_and_resumes() -> Result<()> {
    let store = testing::mem_store_of(4).await?;
    let run = RunId::new("run:head");
    assert!(generation::load_head(&store, &run).await?.is_none());

    let mut head = GenerationHead::new(run.clone(), Utc::now());
    generation::save_head(&store, &head).await?;
    head.advance_to(LoopState::Explore, Utc::now())?;
    generation::save_head(&store, &head).await?;

    let Some(loaded) = generation::load_head(&store, &run).await? else {
        anyhow::bail!("the generation head does not read back");
    };
    assert_eq!(loaded.state, LoopState::Explore);
    assert_eq!(loaded.generation, Generation::ZERO);
    Ok(())
}
