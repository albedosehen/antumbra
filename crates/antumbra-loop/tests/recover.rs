//! A run the process left mid-generation goes on. Found by the first GPU run
//! of the recipe search: a cohort member ran out of GPU memory in `Explore`,
//! and the run could never be resumed, because every generation opens by
//! moving to `Explore` and a head already there refused.

use chrono::Utc;

use antumbra_core::generational::LoopState;
use antumbra_core::testing::{FixedEmbedder, ScriptedTrainer};
use antumbra_core::{ExpertId, Generation, Result, RunId};
use antumbra_loop::{GenerationLoop, LoopConfig};
use antumbra_store::repo::{expert, generation};
use antumbra_store::Store;

/// Leave `run`'s head in `state` at `generation`, as a process killed there
/// would have.
async fn interrupted(store: &Store, run: &RunId, at: u32, state: LoopState) -> Result<()> {
    let mut head = antumbra_core::GenerationHead::new(run.clone(), Utc::now());
    head.generation = Generation(at);
    head.state = state;
    generation::save_head(store, &head).await
}

#[tokio::test]
async fn a_generation_interrupted_while_training_is_run_again() -> Result<()> {
    let store = Store::connect_memory(8).await?;
    let run = RunId::new("run:explore");
    interrupted(&store, &run, 0, LoopState::Explore).await?;
    let trainer = ScriptedTrainer::graduating();
    let embedder = FixedEmbedder::new(8);
    let reports = GenerationLoop::new(&store, &trainer, &embedder, LoopConfig::default())
        .run_until(&run, 1)
        .await?;
    assert_eq!(reports.len(), 1);
    assert_eq!(
        reports[0].generation,
        Generation(0),
        "generation 0 ran again"
    );
    let head = generation::load_head(&store, &run).await?.expect("a head");
    assert_eq!(
        (head.generation, head.state),
        (Generation(1), LoopState::Grow)
    );
    Ok(())
}

/// Everything a generation decides is written before `Consolidate`, so one
/// interrupted there is finished, not trained again.
#[tokio::test]
async fn a_generation_interrupted_after_its_decision_is_finished() -> Result<()> {
    let store = Store::connect_memory(8).await?;
    let run = RunId::new("run:consolidate");
    interrupted(&store, &run, 0, LoopState::Consolidate).await?;
    let trainer = ScriptedTrainer::graduating();
    let embedder = FixedEmbedder::new(8);
    let reports = GenerationLoop::new(&store, &trainer, &embedder, LoopConfig::default())
        .run_until(&run, 1)
        .await?;
    assert!(reports.is_empty(), "nothing was trained again");
    let head = generation::load_head(&store, &run).await?.expect("a head");
    assert_eq!(head.generation, Generation(1));
    Ok(())
}

/// A generation that graduated and was then interrupted before its head moved
/// on graduates again when it is run again, and the population holds one
/// expert for it, not two.
#[tokio::test]
async fn a_generation_run_again_after_graduating_leaves_one_expert() -> Result<()> {
    let store = Store::connect_memory(8).await?;
    let run = RunId::new("run:twice");
    let trainer = ScriptedTrainer::graduating();
    let embedder = FixedEmbedder::new(8);
    let lp = GenerationLoop::new(&store, &trainer, &embedder, LoopConfig::default());
    lp.run_until(&run, 1).await?;
    interrupted(&store, &run, 0, LoopState::Decide).await?;
    lp.run_until(&run, 1).await?;
    let id = ExpertId::new(format!("expert:{run}:g0"));
    let copies = expert::list(&store)
        .await?
        .into_iter()
        .filter(|e| e.id == id)
        .count();
    assert_eq!(copies, 1);
    Ok(())
}
