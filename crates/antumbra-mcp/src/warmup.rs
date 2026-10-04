//! Warming the models a recall waits on, once, when the server starts.
//!
//! The first embedding and the first rerank after a restart pay for loading
//! weights and setting up GPU kernels, and the first caller is usually the
//! session-start hook, which gives up after five seconds. Measured on kuskokwim
//! on 2026-10-04: the first recall after a deploy took 5.8 s, the next ones
//! 0.15 to 0.17 s. So the server spends that time itself, in the background,
//! before anyone asks.
//!
//! The reranker is a separate service that may still be loading when the
//! server starts, so its warm-up retries until it answers. Once it does, the
//! model it serves is asked again: the floor's calibration was chosen at
//! startup, possibly from configuration alone, and a calibration fitted to
//! another model's scores is not a floor.

use std::sync::Arc;
use std::time::{Duration, Instant};

use antumbra_core::platt::Platt;
use antumbra_core::ports::{Embedder, RelevanceScorer};

/// How often, and for how long, to try a reranker that is not up yet. Its
/// container allows itself three minutes to load a model on a cold volume.
const RERANK_RETRY: Duration = Duration::from_secs(5);
const RERANK_PATIENCE: Duration = Duration::from_secs(300);

/// The rerank endpoint, and the calibration the floor started with, for the
/// check once the endpoint answers.
pub(crate) struct Recheck {
    pub url: String,
    pub key: Option<String>,
    pub calibration: Option<Platt>,
}

/// Warm the embedder and, if there is one, the reranker, in the background.
pub(crate) fn spawn(
    embedder: Arc<dyn Embedder>,
    scorer: Option<Arc<dyn RelevanceScorer>>,
    recheck: Option<Recheck>,
) {
    tokio::spawn(async move {
        let started = Instant::now();
        match embedder.embed("warm up").await {
            Ok(_) => eprintln!(
                "antumbra-mcp: embedder warm in {} ms",
                started.elapsed().as_millis()
            ),
            Err(e) => eprintln!("antumbra-mcp: embedder warm-up failed: {e}"),
        }
        let Some(scorer) = scorer else {
            return;
        };
        let started = Instant::now();
        let texts = ["warm up".to_string()];
        loop {
            let call = Instant::now();
            match scorer.relevance("warm up", &texts).await {
                Ok(_) => {
                    eprintln!(
                        "antumbra-mcp: reranker warm after {} s (its first answer took {} ms)",
                        started.elapsed().as_secs(),
                        call.elapsed().as_millis()
                    );
                    break;
                }
                Err(_) if started.elapsed() < RERANK_PATIENCE => {
                    tokio::time::sleep(RERANK_RETRY).await;
                }
                Err(e) => {
                    eprintln!(
                        "antumbra-mcp: reranker not answering after {} s, recall runs in fused order until it does: {e}",
                        started.elapsed().as_secs()
                    );
                    return;
                }
            }
        }
        let Some(recheck) = recheck else {
            return;
        };
        let served = tokio::task::spawn_blocking(move || {
            antumbra_rerank::served_model(&recheck.url, recheck.key.as_deref())
        })
        .await
        .ok()
        .flatten();
        if let Some(model) = served {
            match antumbra_rerank::floor::recheck(&model, recheck.calibration) {
                Some(warning) => eprintln!("antumbra-mcp: WARNING: {warning}"),
                None => {
                    eprintln!("antumbra-mcp: the reranker serves {model}, as the floor expects")
                }
            }
        }
    });
}
