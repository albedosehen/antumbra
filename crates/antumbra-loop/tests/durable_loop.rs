//! Durable generational loop validation: the loop grows the population across generations,
//! survives a simulated kill/restart by resuming from the persisted head,
//! prunes (does not grow) when shadows collapse, and writes its full lineage
//! (shadows, rewards, evaluations, boundaries) to the substrate.

use antumbra_core::generational::LoopCommand;
use antumbra_core::generational::LoopState;
use antumbra_core::ports::Embedder;
use antumbra_core::testing::{FixedEmbedder, ScriptedTrainer};
use antumbra_core::{
    BoundaryFinding, BoundaryId, EvalStatus, EvaluationRun, Expert, ExpertId, FailureBoundary,
    Generation, Grain, RunId, ShadowStatus, SubjectKind,
};
use antumbra_loop::{GenerationLoop, LoopConfig};
use antumbra_store::repo::{
    boundary, evaluation, expert, generation, loop_control, reward, shadow,
};
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

// A cooperative halt of the durable loop: an operator sets the loop control to Halt; the
// runner stops at the next generation boundary, checkpoints the head as Paused,
// and consumes the signal. Re-running resumes from that checkpoint.
#[tokio::test]
async fn an_operator_halt_stops_the_loop_at_a_generation_boundary() {
    let store = Store::connect_memory(8).await.expect("connect");
    let trainer = ScriptedTrainer::graduating();
    let embedder = FixedEmbedder::new(8);
    let run = RunId::new("run:halt");

    // Halt requested before the run starts → it stops immediately, no generations.
    loop_control::set(&store, &run, LoopCommand::Halt)
        .await
        .unwrap();
    let lp = GenerationLoop::new(&store, &trainer, &embedder, LoopConfig::default());
    let reports = lp.run_until(&run, 3).await.unwrap();
    assert!(reports.is_empty(), "halted before any generation ran");

    // The head is checkpointed Paused and the control was consumed (back to Run).
    let head = generation::load_head(&store, &run).await.unwrap().unwrap();
    assert_eq!(head.state, LoopState::Paused);
    assert_eq!(head.generation, Generation::ZERO);
    assert_eq!(
        loop_control::load(&store, &run).await.unwrap(),
        LoopCommand::Run,
        "the halt was consumed"
    );

    // Re-running with no control resumes out of Paused and makes progress.
    let reports = lp.run_until(&run, 1).await.unwrap();
    assert_eq!(reports.len(), 1, "resumed and ran one generation");
    assert_eq!(expert::list(&store).await.unwrap().len(), 1);
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

// The no-forgetting tripwire (the frozen-expert population): across generations, every frozen expert
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

// A graduating expert retires the boundaries it now covers (retire-on-correction lifecycle):
// by its capability vector it sits inside the failure region a scope marked, so
// the gap is filled and the boundary must stop gating routing. A boundary it does
// not cover is left intact.
#[tokio::test]
async fn a_graduating_expert_retires_the_boundaries_it_covers() {
    let store = Store::connect_memory(8).await.expect("connect");
    let embedder = FixedEmbedder::new(8);
    let exemplar = "reverse a string";
    let cap = embedder.embed(exemplar).await.unwrap();
    let far = embedder.embed("unrelated gardening prose").await.unwrap();
    let now = Utc::now();

    // Covered: the expert's capability == this boundary's failure context (and is
    // far from C'), so it sits closer to C than C' -> resolved -> retired.
    let covered = FailureBoundary {
        id: BoundaryId::new("boundary:covered"),
        behavior: "reverse".into(),
        fail_context: serde_json::json!({ "x": 1 }),
        near_ok_context: Some(serde_json::json!({ "x": 2 })),
        governing_features: vec!["x".into()],
        grain: Some(Grain::Project),
        context_vec: Some(cap.clone()),
        ok_context_vec: Some(far.clone()),
        confidence: 0.9,
        generation: Generation::ZERO,
        created_at: now,
    };
    // Uncovered: capability is closer to C' than to C -> not covered -> retained.
    let uncovered = FailureBoundary {
        id: BoundaryId::new("boundary:uncovered"),
        context_vec: Some(far.clone()),
        ok_context_vec: Some(cap.clone()),
        ..covered.clone()
    };
    boundary::upsert(&store, &covered).await.unwrap();
    boundary::upsert(&store, &uncovered).await.unwrap();

    // Graduate an expert whose capability vector is the centroid of `exemplar`.
    let trainer = ScriptedTrainer::graduating_with_exemplars(vec![exemplar.to_string()]);
    let run = RunId::new("run:retire");
    let lp = GenerationLoop::new(&store, &trainer, &embedder, LoopConfig::default());
    let mut head = lp.resume_or_init(&run).await.unwrap();
    lp.run_generation(&mut head).await.unwrap();

    let remaining = boundary::list(&store).await.unwrap();
    assert_eq!(remaining.len(), 1, "the covered boundary was retired");
    assert_eq!(remaining[0].id.as_str(), "boundary:uncovered");
}

// A capture run that surfaces a verified correction's contrastive pair persists
// an ACTIONABLE counterfactual boundary: a C' was recovered, and the loop embeds
// both contexts so the relative-margin inhibition can fire -- unlike the
// open-negative a prune logs.
#[tokio::test]
async fn a_captured_correction_persists_an_actionable_boundary() {
    let store = Store::connect_memory(8).await.expect("connect");
    let finding = BoundaryFinding {
        behavior: "add a dep".into(),
        governing_feature: "runtime".into(),
        fail_context: serde_json::json!({ "runtime": "deno" }),
        near_ok_context: serde_json::json!({ "runtime": "node" }),
    };
    let trainer = ScriptedTrainer::graduating_with_boundary(finding);
    let embedder = FixedEmbedder::new(8);
    let run = RunId::new("run:cap");

    let lp = GenerationLoop::new(&store, &trainer, &embedder, LoopConfig::default());
    let mut head = lp.resume_or_init(&run).await.unwrap();
    lp.run_generation(&mut head).await.unwrap();

    let boundaries = boundary::list(&store).await.unwrap();
    assert_eq!(boundaries.len(), 1);
    let b = &boundaries[0];
    assert!(b.is_actionable(), "a recovered C' makes it actionable");
    assert!(
        b.context_vec.is_some() && b.ok_context_vec.is_some(),
        "both contexts embedded, so inhibition can compare against task vectors"
    );
    assert_eq!(b.governing_features, vec!["runtime".to_string()]);
    assert_eq!(
        b.near_ok_context,
        Some(serde_json::json!({ "runtime": "node" }))
    );
    // The two context embeddings differ (C vs C'), so the relative margin is
    // meaningful rather than degenerate.
    assert_ne!(b.context_vec, b.ok_context_vec);
}
