//! ADR-0008 validation: the loop grows the population across generations,
//! survives a simulated kill/restart by resuming from the persisted head,
//! prunes (does not grow) when shadows collapse, and writes its full lineage
//! (shadows, rewards, evaluations, boundaries) to the substrate.

use antumbra_core::generational::LoopState;
use antumbra_core::testing::{FixedEmbedder, ScriptedTrainer};
use antumbra_core::{Generation, RunId, ShadowStatus, SubjectKind};
use antumbra_loop::{GenerationLoop, LoopConfig};
use antumbra_store::repo::{boundary, evaluation, expert, reward, shadow};
use antumbra_store::Store;

#[tokio::test]
async fn grows_population_and_resumes_after_restart() {
    let store = Store::connect_memory(8).await.expect("connect");
    let trainer = ScriptedTrainer::graduating();
    let embedder = FixedEmbedder::new(8);
    let cfg = LoopConfig {
        graduate_threshold: 0.5,
        base_model: "code-base".into(),
        max_steps: 4,
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
