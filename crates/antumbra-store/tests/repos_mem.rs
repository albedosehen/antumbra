//! Round-trip the shadow, reward, boundary, and evaluation repositories
//! against an embedded SurrealDB.

use chrono::{Duration, Utc};

use antumbra_core::{
    BoundaryId, EvalStatus, EvaluationRun, Expert, ExpertId, FailureBoundary, Generation,
    GenerationHead, Grain, LoopState, RewardSignal, RunId, Shadow, ShadowId, ShadowStatus,
    SubjectKind,
};
use antumbra_store::repo::{boundary, evaluation, expert, generation, reward, shadow};
use antumbra_store::Store;

#[tokio::test]
async fn shadow_lifecycle_persists() {
    let store = Store::connect_memory(4).await.unwrap();
    let mut s = Shadow::spawn(
        ShadowId::new("shadow:g0"),
        Generation::ZERO,
        Some(ExpertId::new("expert:a")),
        Utc::now(),
    );
    shadow::upsert(&store, &s).await.unwrap();

    let got = shadow::get(&store, &s.id).await.unwrap().unwrap();
    assert_eq!(got.status, ShadowStatus::Spawning);
    assert_eq!(got.parent_expert, Some(ExpertId::new("expert:a")));

    s.advance_to(ShadowStatus::Exploring).unwrap();
    shadow::upsert(&store, &s).await.unwrap();
    assert_eq!(
        shadow::list_by_status(&store, ShadowStatus::Exploring)
            .await
            .unwrap()
            .len(),
        1
    );
    assert!(shadow::list_by_status(&store, ShadowStatus::Spawning)
        .await
        .unwrap()
        .is_empty());
}

#[tokio::test]
async fn reward_signals_persist_per_run() {
    let store = Store::connect_memory(4).await.unwrap();
    let run = RunId::new("run:1");
    let now = Utc::now();
    let signals = vec![
        RewardSignal::verifier(run.clone(), 0, "tests", 1.0, now),
        RewardSignal::critic(run.clone(), 0, "critic", 0.5, now),
    ];
    reward::insert_many(&store, &signals).await.unwrap();

    let loaded = reward::list_by_run(&store, &run).await.unwrap();
    assert_eq!(loaded.len(), 2);
    assert_eq!(loaded.iter().filter(|s| s.is_verifier()).count(), 1);
}

#[tokio::test]
async fn boundary_persists_and_recalls_by_context() {
    let store = Store::connect_memory(4).await.unwrap();
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
        created_at: Utc::now(),
    };
    boundary::upsert(&store, &b).await.unwrap();

    assert_eq!(boundary::list(&store).await.unwrap().len(), 1);
    let near = boundary::knn_by_context(&store, &[0.9, 0.1, 0.0, 0.0], 1)
        .await
        .unwrap();
    assert_eq!(near.len(), 1);
    assert!(near[0].is_actionable());
}

#[tokio::test]
async fn evaluation_runs_track_latest() {
    let store = Store::connect_memory(4).await.unwrap();
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
        .await
        .unwrap();
    }

    let all = evaluation::list_for_subject(&store, SubjectKind::Expert, "expert:a")
        .await
        .unwrap();
    assert_eq!(all.len(), 2);
    let latest = evaluation::latest_for_subject(&store, SubjectKind::Expert, "expert:a")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(latest.regression_fingerprint.as_deref(), Some("bbb"));
}

#[tokio::test]
async fn expert_population_crud_and_knn() {
    let store = Store::connect_memory(4).await.unwrap();
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
    expert::insert(&store, &mk("expert:a", vec![1.0, 0.0, 0.0, 0.0]))
        .await
        .unwrap();
    expert::insert(&store, &mk("expert:b", vec![0.0, 1.0, 0.0, 0.0]))
        .await
        .unwrap();

    assert_eq!(expert::list(&store).await.unwrap().len(), 2);
    let got = expert::get(&store, &ExpertId::new("expert:a"))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(got.name, "expert:a");
    assert!(expert::get(&store, &ExpertId::new("expert:missing"))
        .await
        .unwrap()
        .is_none());

    let near = expert::knn_by_capability(&store, &[0.9, 0.1, 0.0, 0.0], 1)
        .await
        .unwrap();
    assert_eq!(near.len(), 1);
    assert_eq!(near[0].id, ExpertId::new("expert:a"));
}

#[tokio::test]
async fn generation_head_round_trips_and_resumes() {
    let store = Store::connect_memory(4).await.unwrap();
    let run = RunId::new("run:head");
    assert!(generation::load_head(&store, &run).await.unwrap().is_none());

    let mut head = GenerationHead::new(run.clone(), Utc::now());
    generation::save_head(&store, &head).await.unwrap();
    head.advance_to(LoopState::Explore, Utc::now()).unwrap();
    generation::save_head(&store, &head).await.unwrap();

    let loaded = generation::load_head(&store, &run).await.unwrap().unwrap();
    assert_eq!(loaded.state, LoopState::Explore);
    assert_eq!(loaded.generation, Generation::ZERO);
}
