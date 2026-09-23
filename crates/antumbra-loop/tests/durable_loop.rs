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
        ..LoopConfig::default()
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
        placed_on: None,
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

/// ADR-0022 step 2. A generation is read through the standing instruments, and
/// the slice a task lands in comes from its id alone -- so nothing the loop
/// decides can move a task across the anchor.
#[tokio::test]
async fn a_generation_is_measured_through_the_instruments() -> antumbra_core::Result<()> {
    use antumbra_core::ports::TaskOutcome;
    use antumbra_eclipse::Slice;

    let store = Store::connect_memory(8).await?;
    let partition = antumbra_eclipse::Partition::default();
    // Tasks chosen so the default partition puts them on both sides of the
    // anchor; the ids are the ones eclipse's own pinning test names.
    let per_task = vec![
        TaskOutcome {
            task_id: "task:0".into(),
            passed: true,
            size: 10,
            impossible: false,
        },
        TaskOutcome {
            task_id: "task:1".into(),
            passed: false,
            size: 10,
            impossible: false,
        },
        TaskOutcome {
            task_id: "task:19".into(),
            passed: true,
            size: 10,
            impossible: false,
        },
    ];
    assert_eq!(partition.of("task:0"), Slice::Visible);
    assert_eq!(partition.of("task:1"), Slice::HeldOut);
    assert_eq!(partition.of("task:19"), Slice::Audit);

    let trainer = ScriptedTrainer {
        per_task: per_task.clone(),
        ..ScriptedTrainer::graduating()
    };
    let embedder = FixedEmbedder::new(8);
    let cfg = LoopConfig {
        partition: Some(partition),
        ..LoopConfig::default()
    };
    let lp = GenerationLoop::new(&store, &trainer, &embedder, cfg);
    let run = RunId::new("run:instrumented");
    let mut head = lp.resume_or_init(&run).await?;
    let report = lp.run_generation(&mut head).await?;

    let measured = report
        .instruments
        .ok_or_else(|| antumbra_core::AntumbraError::other("the generation was not measured"))?;
    // The visible task passed and the held-out one did not, which is the gap
    // the record calls the primary hacking alarm.
    assert_eq!(measured.widest_gap().map(|(_, w)| w), Some(1.0));
    // The audit slice is counted apart from anything that chooses.
    assert_eq!(measured.audit.rate(), Some(1.0));
    assert!(!measured.failed(), "no impossible task was passed");
    Ok(())
}

/// Three per-task results, one in each slice the default partition draws:
/// the visible task passes, the held-out one fails, the audited one passes.
fn one_task_per_slice() -> Vec<antumbra_core::ports::TaskOutcome> {
    [("task:0", true), ("task:1", false), ("task:19", true)]
        .into_iter()
        .map(|(id, passed)| antumbra_core::ports::TaskOutcome {
            task_id: id.into(),
            passed,
            size: 10,
            impossible: false,
        })
        .collect()
}

/// Run one generation of `trainer` under `partition` and return its report.
async fn one_generation(
    trainer: &ScriptedTrainer,
    partition: Option<antumbra_eclipse::Partition>,
    run: &str,
) -> antumbra_core::Result<antumbra_loop::GenerationReport> {
    let store = Store::connect_memory(8).await?;
    let embedder = FixedEmbedder::new(8);
    let cfg = LoopConfig {
        partition,
        ..LoopConfig::default()
    };
    let lp = GenerationLoop::new(&store, trainer, &embedder, cfg);
    let mut head = lp.resume_or_init(&RunId::new(run)).await?;
    lp.run_generation(&mut head).await
}

/// A trainer that reports only aggregate fitness leaves the generation
/// unmeasured, and the loop says so rather than inventing a report from one
/// number. That distinction is the whole reason the instruments exist.
#[tokio::test]
async fn a_generation_with_no_per_task_results_is_not_measured() -> antumbra_core::Result<()> {
    let report = one_generation(
        &ScriptedTrainer::graduating(),
        Some(antumbra_eclipse::Partition::default()),
        "run:unmeasured",
    )
    .await?;
    assert!(
        report.instruments.is_none(),
        "a report synthesised from aggregate fitness would look like a measurement"
    );
    Ok(())
}

/// With no partition nothing was held out, so per-task results are a list of
/// learned tasks, and slicing it would invent a gap between them.
#[tokio::test]
async fn a_generation_with_nothing_held_out_is_not_measured() -> antumbra_core::Result<()> {
    let trainer = ScriptedTrainer {
        per_task: one_task_per_slice(),
        ..ScriptedTrainer::graduating()
    };
    let report = one_generation(&trainer, None, "run:nothing-held").await?;
    assert!(report.instruments.is_none());
    Ok(())
}

/// The defect this guards against shipped once: the loop sliced per-task
/// results by the partition while the trainer had learned from every task, so
/// every persisted gap was a difference between two sets of trained tasks. A
/// trainer that does not confirm the holdout it enforced gets no instruments.
#[tokio::test]
async fn a_trainer_that_ignores_the_holdout_is_not_measured() -> antumbra_core::Result<()> {
    let trainer = ScriptedTrainer {
        per_task: one_task_per_slice(),
        ignores_holdout: true,
        ..ScriptedTrainer::graduating()
    };
    let report = one_generation(
        &trainer,
        Some(antumbra_eclipse::Partition::default()),
        "run:ignored",
    )
    .await?;
    assert!(
        report.instruments.is_none(),
        "results from a run that learned from its held-out tasks are not a gap"
    );
    Ok(())
}

/// The instruments are persisted beside the fitness they qualify, with the
/// seed they were read under.
#[tokio::test]
async fn the_measurement_is_stored_with_the_score_it_qualifies() -> antumbra_core::Result<()> {
    let store = Store::connect_memory(8).await?;
    let trainer = ScriptedTrainer {
        per_task: one_task_per_slice(),
        ..ScriptedTrainer::graduating()
    };
    let embedder = FixedEmbedder::new(8);
    let partition = antumbra_eclipse::Partition::new(0.20, 0.10, 0)?;
    let cfg = LoopConfig {
        partition: Some(partition),
        ..LoopConfig::default()
    };
    let lp = GenerationLoop::new(&store, &trainer, &embedder, cfg);
    let run = RunId::new("run:stored");
    let mut head = lp.resume_or_init(&run).await?;
    lp.run_generation(&mut head).await?;

    let rows = evaluation::list_for_subject(&store, SubjectKind::Shadow, "run:stored:g0").await?;
    let metrics = rows
        .first()
        .and_then(|r| r.metrics.clone())
        .ok_or_else(|| antumbra_core::AntumbraError::other("no evaluation row"))?;
    assert_eq!(metrics["partition_seed"], serde_json::json!(0));
    assert_eq!(
        metrics["instruments"]["audit"]["measured"],
        serde_json::json!(1)
    );
    Ok(())
}

/// Run generations `from..to` of one run. Each is trained by a scripted trainer
/// whose fitness climbs three points a generation, and whose audited task
/// (`task:19` under the default partition) passes from `audit_passes_from` on.
async fn climbing_run(
    store: &Store,
    run: &RunId,
    partition: antumbra_eclipse::Partition,
    generations: std::ops::Range<u32>,
    audit_passes_from: Option<u32>,
) -> antumbra_core::Result<Vec<antumbra_loop::GenerationReport>> {
    let embedder = FixedEmbedder::new(8);
    let mut reports = Vec::new();
    for g in generations {
        let audit_passes = audit_passes_from.is_some_and(|from| g >= from);
        let trainer = ScriptedTrainer {
            final_fitness: 0.40 + 0.03 * g as f32,
            per_task: [
                ("task:0", true),
                ("task:1", true),
                ("task:19", audit_passes),
            ]
            .into_iter()
            .map(|(id, passed)| antumbra_core::ports::TaskOutcome {
                task_id: id.into(),
                passed,
                size: 10,
                impossible: false,
            })
            .collect(),
            ..ScriptedTrainer::graduating()
        };
        let cfg = LoopConfig {
            partition: Some(partition),
            ..LoopConfig::default()
        };
        let lp = GenerationLoop::new(store, &trainer, &embedder, cfg);
        let mut head = lp.resume_or_init(run).await?;
        reports.push(lp.run_generation(&mut head).await?);
    }
    Ok(reports)
}

fn audit_measured(report: &antumbra_loop::GenerationReport) -> Option<u32> {
    report.instruments.as_ref().map(|m| m.audit.measured)
}

/// ADR-0022 reads the audit slice every k generations, not every one. Off
/// schedule the audit task is not measured at all, which is different from
/// measured and failed.
#[tokio::test]
async fn the_audit_slice_is_measured_on_its_schedule() -> antumbra_core::Result<()> {
    let store = Store::connect_memory(8).await?;
    let partition = antumbra_eclipse::Partition::default();
    let reports = climbing_run(&store, &RunId::new("run:k"), partition, 0..3, Some(0)).await?;
    // Default k is 2: generations 0 and 2 are audited, 1 is not.
    let audited: Vec<Option<u32>> = reports.iter().map(audit_measured).collect();
    assert_eq!(audited, [Some(1), Some(0), Some(1)]);
    // The held-out slice is measured every generation regardless.
    assert!(reports.iter().all(|r| r
        .instruments
        .as_ref()
        .is_some_and(|m| m.gaps.iter().any(|g| g.held_out.measured > 0))));
    Ok(())
}

/// S-1's kill criterion, read by the loop: ten generations of a search score
/// climbing while the audit slice stays where it was.
#[tokio::test]
async fn a_climb_the_audit_slice_does_not_follow_reads_as_overtuning() -> antumbra_core::Result<()>
{
    use antumbra_eclipse::Trend;
    let store = Store::connect_memory(8).await?;
    let run = RunId::new("run:overtuned");
    let partition = antumbra_eclipse::Partition::default();
    let reports = climbing_run(&store, &run, partition, 0..10, None).await?;

    let trends: Vec<Option<Trend>> = reports.iter().map(|r| r.trend).collect();
    // Nine generations are not a window: an unasked question is not a clean
    // bill, so they read as inconclusive rather than carrying.
    assert!(trends[..9].iter().all(|t| *t == Some(Trend::Inconclusive)));
    assert_eq!(trends[9], Some(Trend::Overtuning));

    // And the reading is stored with the generation it was taken at.
    let rows =
        evaluation::list_for_subject(&store, SubjectKind::Shadow, "run:overtuned:g9").await?;
    let trend = rows
        .first()
        .and_then(|r| r.metrics.as_ref())
        .map(|m| m["trend"].clone());
    assert_eq!(trend, Some(serde_json::json!("overtuning")));
    Ok(())
}

/// The same climb, with the audit slice following it, is a real gain.
#[tokio::test]
async fn a_climb_the_audit_slice_follows_reads_as_carrying() -> antumbra_core::Result<()> {
    let store = Store::connect_memory(8).await?;
    let partition = antumbra_eclipse::Partition::default();
    let reports =
        climbing_run(&store, &RunId::new("run:honest"), partition, 0..10, Some(4)).await?;
    assert_eq!(
        reports.last().and_then(|r| r.trend),
        Some(antumbra_eclipse::Trend::Carrying)
    );
    Ok(())
}

/// A reseed repartitions the corpus, so generations measured under the old
/// seed read a different audit slice and do not count toward the new trend.
#[tokio::test]
async fn a_reseed_starts_the_trend_over() -> antumbra_core::Result<()> {
    let store = Store::connect_memory(8).await?;
    let run = RunId::new("run:reseeded");
    let before = antumbra_eclipse::Partition::default();
    climbing_run(&store, &run, before, 0..10, None).await?;
    let after = antumbra_eclipse::Partition::new(before.held_out, before.audit, 7)?;
    let reports = climbing_run(&store, &run, after, 10..11, None).await?;
    assert_eq!(
        reports.first().and_then(|r| r.trend),
        Some(antumbra_eclipse::Trend::Inconclusive)
    );
    Ok(())
}

/// One visible task that passes, plus an impossible task that passes or fails.
fn with_impossible(impossible_passed: bool) -> ScriptedTrainer {
    ScriptedTrainer {
        per_task: vec![
            antumbra_core::ports::TaskOutcome {
                task_id: "task:0".into(),
                passed: true,
                size: 10,
                impossible: false,
            },
            antumbra_core::ports::TaskOutcome {
                task_id: "unsatisfiable".into(),
                passed: impossible_passed,
                size: 10,
                impossible: true,
            },
        ],
        ..ScriptedTrainer::graduating()
    }
}

/// ADR-0022: a pass on an impossible task is proof of a shortcut, so the
/// generation fails whole -- however well it scored -- and nothing graduates.
#[tokio::test]
async fn a_generation_that_passes_an_impossible_task_does_not_graduate() -> antumbra_core::Result<()>
{
    let store = Store::connect_memory(8).await?;
    let partition = antumbra_eclipse::Partition::default();
    let cfg = LoopConfig {
        partition: Some(partition),
        ..LoopConfig::default()
    };
    let embedder = FixedEmbedder::new(8);
    let trainer = with_impossible(true);
    let lp = GenerationLoop::new(&store, &trainer, &embedder, cfg);
    let mut head = lp.resume_or_init(&RunId::new("run:shortcut")).await?;
    let report = lp.run_generation(&mut head).await?;

    assert!(report.fitness >= 0.5, "it scored well enough to graduate");
    assert!(!report.graduated, "and a shortcut stopped it anyway");
    let measured = report
        .instruments
        .ok_or_else(|| antumbra_core::AntumbraError::other("not measured"))?;
    assert_eq!(
        measured.impossible_passed,
        vec!["unsatisfiable".to_string()]
    );
    assert!(expert::list(&store).await?.is_empty());
    Ok(())
}

/// Failing an impossible task is the expected outcome and costs nothing.
#[tokio::test]
async fn failing_an_impossible_task_is_what_should_happen() -> antumbra_core::Result<()> {
    let store = Store::connect_memory(8).await?;
    let cfg = LoopConfig {
        partition: Some(antumbra_eclipse::Partition::default()),
        ..LoopConfig::default()
    };
    let embedder = FixedEmbedder::new(8);
    let trainer = with_impossible(false);
    let lp = GenerationLoop::new(&store, &trainer, &embedder, cfg);
    let mut head = lp
        .resume_or_init(&RunId::new("run:honest-impossible"))
        .await?;
    let report = lp.run_generation(&mut head).await?;
    assert!(report.graduated);
    let measured = report
        .instruments
        .ok_or_else(|| antumbra_core::AntumbraError::other("not measured"))?;
    assert_eq!(measured.impossible_measured, 1);
    Ok(())
}
