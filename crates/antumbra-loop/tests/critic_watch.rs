//! The critic's per-generation reading reaches the generation's report: what
//! the trainer saw of its critic against the verifier and the twin, every
//! generation, rather than once at training time. Once the reading says the
//! critic no longer tracks the verifier, the run sets it aside and trains on
//! the verifier's reward alone.

use std::sync::Mutex;

use async_trait::async_trait;

use antumbra_core::critic::{watch, CriticWatch, Fallback};
use antumbra_core::ports::{TrainOutcome, TrainRequest, Trainer};
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

/// Reads its critic from a script, one watch a generation, and keeps whether
/// each request asked for the verifier's reward alone. A request that did is
/// trained without the critic, so it reports no watch.
struct Scripted {
    watches: Mutex<Vec<CriticWatch>>,
    verifier_only: Mutex<Vec<bool>>,
}

#[async_trait]
impl Trainer for Scripted {
    async fn train_shadow(&self, req: TrainRequest) -> Result<TrainOutcome> {
        self.verifier_only.lock().unwrap().push(req.verifier_only);
        let next = {
            let mut watches = self.watches.lock().unwrap();
            (!watches.is_empty()).then(|| watches.remove(0))
        };
        ScriptedTrainer {
            critic_watch: next.filter(|_| !req.verifier_only),
            ..ScriptedTrainer::collapsing()
        }
        .train_shadow(req)
        .await
    }
}

fn read(correlation: f32) -> CriticWatch {
    CriticWatch {
        n: 100,
        correlation: Some(correlation),
        ece: None,
        recalibrated_ece: None,
        twin_agreement: None,
    }
}

#[tokio::test]
async fn a_critic_that_stops_tracking_the_verifier_is_set_aside() -> Result<()> {
    let store = Store::connect_memory(8).await?;
    let trainer = Scripted {
        watches: Mutex::new(vec![read(0.4), read(-0.2), read(0.5)]),
        verifier_only: Mutex::default(),
    };
    let reports = GenerationLoop::new(
        &store,
        &trainer,
        &FixedEmbedder::new(8),
        LoopConfig::default(),
    )
    .run_until(&RunId::new("run:set-aside"), 3)
    .await?;
    let fallbacks: Vec<Option<Fallback>> = reports.iter().map(|r| r.critic_fallback).collect();
    assert_eq!(
        fallbacks,
        vec![
            None,
            Some(Fallback::Uncorrelated { correlation: -0.2 }),
            None
        ]
    );
    // The generation after the one that set it aside trained without it.
    assert_eq!(
        *trainer.verifier_only.lock().unwrap(),
        vec![false, false, true]
    );
    assert_eq!(reports[2].critic, None);
    Ok(())
}
