//! The critic's per-generation reading reaches the generation's report
//! (ADR-0022 S-2): what the trainer saw of its critic against the verifier and
//! the twin, every generation, rather than once at training time.

use antumbra_core::critic::watch;
use antumbra_core::testing::{FixedEmbedder, ScriptedTrainer};
use antumbra_core::{Result, RunId};
use antumbra_loop::{GenerationLoop, LoopConfig};
use antumbra_store::Store;

#[tokio::test]
async fn each_generation_reports_how_its_critic_read() -> Result<()> {
    let store = Store::connect_memory(8).await?;
    let read = watch(
        &[0.9, 0.2, 0.8, 0.1],
        &[true, false, true, false],
        Some(&[0.7, 0.3, 0.9, 0.2]),
    );
    let trainer = ScriptedTrainer {
        critic_watch: Some(read.clone()),
        ..ScriptedTrainer::graduating()
    };
    let reports = GenerationLoop::new(
        &store,
        &trainer,
        &FixedEmbedder::new(8),
        LoopConfig::default(),
    )
    .run_until(&RunId::new("run:watched"), 1)
    .await?;
    assert_eq!(reports[0].critic, Some(read));

    let plain = ScriptedTrainer::graduating();
    let reports = GenerationLoop::new(
        &store,
        &plain,
        &FixedEmbedder::new(8),
        LoopConfig::default(),
    )
    .run_until(&RunId::new("run:unwatched"), 1)
    .await?;
    assert_eq!(reports[0].critic, None);
    Ok(())
}
