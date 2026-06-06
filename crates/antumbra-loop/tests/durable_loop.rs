//! ADR-0008 validation: the loop grows the population across generations,
//! survives a simulated kill/restart by resuming from the persisted head,
//! prunes (does not grow) when shadows collapse, and writes its full lineage
//! (shadows, rewards, evaluations, boundaries) to the substrate.

use antumbra_core::generational::LoopState;
use antumbra_core::ports::Embedder;
use antumbra_core::testing::{FixedEmbedder, ScriptedTrainer};
use antumbra_core::{
    EvalStatus, EvaluationRun, Expert, ExpertId, Generation, RunId, ShadowStatus, SubjectKind,
};
use antumbra_loop::{GenerationLoop, LoopConfig};
use antumbra_store::repo::{boundary, evaluation, expert, reward, shadow};
use antumbra_store::Store;
use chrono::Utc;

#[tokio::test]
async fn grows_population_and_resumes_after_restart() {
    let store = Store::connect_memory(8).await.expect("connect");
    let trainer = ScriptedTrainer::graduating();
    let embedder = FixedEmbedder::new(8);
    let cfg = LoopConfig {
        graduate_threshold: 0.5,
        base_model: "code-base".into(),
    };
    let run = RunId::new("run:loop");

    // First "process": run one generation, then drop the loop (simulated kill).
    {
        let lp = GenerationLoop::new(&store, &trainer, &embedder, cfg.clone());
        let mut head = lp.resume_or_init(&run).await.unwrap();
        assert_eq!(head.state, LoopState::Grow);
        assert_eq!(head.generation, Generation::ZERO);

        let report = lp.run_generation(&mut head).await.unwrap();
        assert!(report.graduated);
        assert_eq!(head.state, LoopState::Grow);
        assert_eq!(head.generation, Generation(1));
    }

    // The population + lineage persisted independently of the loop object.
    assert_eq!(expert::list(&store).await.unwrap().len(), 1);
    assert_eq!(
        shadow::list_by_status(&store, ShadowStatus::Graduated)
            .await
            .unwrap()
            .len(),
        1
    );
    assert!(!reward::list_by_run(&store, &run).await.unwrap().is_empty());

    // Second "process": a fresh loop resumes from the persisted head.
    {
        let lp = GenerationLoop::new(&store, &trainer, &embedder, cfg.clone());
        let head = lp.resume_or_init(&run).await.unwrap();
        assert_eq!(head.generation, Generation(1)); // resumed, not reset
        assert_eq!(head.state, LoopState::Grow);

        let reports = lp.run_until(&run, 2).await.unwrap();
        assert_eq!(reports.len(), 1);
    }

    assert_eq!(expert::list(&store).await.unwrap().len(), 2);
    assert_eq!(
        shadow::list_by_status(&store, ShadowStatus::Graduated)
            .await
            .unwrap()
            .len(),
        2
    );
}

#[tokio::test]
async fn capability_vector_is_learned_from_solved_exemplars() {
    let store = Store::connect_memory(8).await.expect("connect");
    let exemplars = vec!["reverse a string".to_string(), "format text".to_string()];
    let trainer = ScriptedTrainer::graduating_with_exemplars(exemplars.clone());
    let embedder = FixedEmbedder::new(8);
    let run = RunId::new("run:cap");

    let lp = GenerationLoop::new(&store, &trainer, &embedder, LoopConfig::default());
    let mut head = lp.resume_or_init(&run).await.unwrap();
    let report = lp.run_generation(&mut head).await.unwrap();
    assert!(report.graduated);

    // The expert's capability vector must be the centroid of the embedded
    // solved-task prompts -- learned from evaluated behavior, not a label.
    let mut expected = [0.0f32; 8];
    for ex in &exemplars {
        let v = embedder.embed(ex).await.unwrap();
        for (e, x) in expected.iter_mut().zip(v.iter()) {
            *e += *x;
        }
    }
    expected
        .iter_mut()
        .for_each(|e| *e /= exemplars.len() as f32);

    let experts = expert::list(&store).await.unwrap();
    assert_eq!(experts.len(), 1);
    let cap = experts[0]
        .capability_vec
        .as_ref()
        .expect("graduated expert has a capability vector");
    assert_eq!(cap.len(), 8);
    for (c, e) in cap.iter().zip(expected.iter()) {
        assert!((c - e).abs() < 1e-6, "cap {c} != expected centroid {e}");
    }
}

// The no-forgetting tripwire (ADR-0001): across generations, every frozen expert
// stays byte-identical to its freeze baseline, so each generation reports zero
// regressions -- the freeze holds.
#[tokio::test]
async fn the_freeze_holds_across_generations() {
    let store = Store::connect_memory(8).await.expect("connect");
    let trainer = ScriptedTrainer::graduating();
    let embedder = FixedEmbedder::new(8);
    let run = RunId::new("run:hold");

    let lp = GenerationLoop::new(&store, &trainer, &embedder, LoopConfig::default());
    let reports = lp.run_until(&run, 2).await.unwrap();

    assert_eq!(reports.len(), 2);
    assert!(
        reports.iter().all(|r| r.regressions.is_empty()),
        "frozen experts must not drift from their freeze baseline"
    );
    assert_eq!(expert::list(&store).await.unwrap().len(), 2);
}

// The kill criterion fires: a frozen expert whose current artifact no longer
// matches its freeze-baseline fingerprint (the adapter was re-pointed / corrupted
// after it froze) is caught by the tripwire and reported.
#[tokio::test]
async fn a_drifted_frozen_expert_trips_the_no_forgetting_check() {
    let store = Store::connect_memory(8).await.expect("connect");
    let trainer = ScriptedTrainer::graduating();
    let embedder = FixedEmbedder::new(8);
    let run = RunId::new("run:nf");
    let lp = GenerationLoop::new(&store, &trainer, &embedder, LoopConfig::default());
    let now = Utc::now();

    // A frozen expert pointing at one artifact...
    let expert = Expert {
        id: ExpertId::new("expert:e1"),
        name: "e1".into(),
        base_model: "code-base".into(),
        artifact_uri: "memory://adapter/live".into(),
        capability_card: serde_json::json!({}),
        capability_vec: None,
        fitness: 0.9,
        frozen_at: Some(now),
        generation: Generation::ZERO,
        owner: None,
        compartment: None,
        created_at: now,
    };
    expert::insert(&store, &expert).await.unwrap();
    // ...but whose freeze baseline recorded a DIFFERENT fingerprint.
    evaluation::insert(
        &store,
        &EvaluationRun {
            run_id: run.clone(),
            subject_kind: SubjectKind::Expert,
            subject_id: "expert:e1".into(),
            corpus_task_id: "freeze:g0".into(),
            status: EvalStatus::Success,
            metrics: None,
            regression_fingerprint: Some("uri:memory://adapter/original".into()),
            created_at: now,
        },
    )
    .await
    .unwrap();

    let regressions = lp.check_no_forgetting(&run).await.unwrap();
    assert_eq!(
        regressions,
        vec![ExpertId::new("expert:e1")],
        "the drifted frozen expert is caught by the kill criterion"
    );
}

#[tokio::test]
async fn collapsing_shadows_are_pruned_and_logged() {
    let store = Store::connect_memory(8).await.expect("connect");
    let trainer = ScriptedTrainer::collapsing(); // final_fitness 0.0
    let embedder = FixedEmbedder::new(8);
    let run = RunId::new("run:prune");

    let lp = GenerationLoop::new(&store, &trainer, &embedder, LoopConfig::default());
    let mut head = lp.resume_or_init(&run).await.unwrap();
    let report = lp.run_generation(&mut head).await.unwrap();

    assert!(!report.graduated);
    assert_eq!(report.fitness, 0.0);
    assert_eq!(head.generation, Generation(1));

    // nothing graduated into the population
    assert!(expert::list(&store).await.unwrap().is_empty());
    // the shadow is pruned
    assert_eq!(
        shadow::list_by_status(&store, ShadowStatus::Pruned)
            .await
            .unwrap()
            .len(),
        1
    );
    // and an open-negative boundary was logged (recorded but NOT actionable)
    let boundaries = boundary::list(&store).await.unwrap();
    assert_eq!(boundaries.len(), 1);
    assert!(!boundaries[0].is_actionable());
    // the failure is captured as an evaluation run
    let evals = evaluation::list_for_subject(&store, SubjectKind::Shadow, report.shadow.as_str())
        .await
        .unwrap();
    assert_eq!(evals.len(), 1);
}
