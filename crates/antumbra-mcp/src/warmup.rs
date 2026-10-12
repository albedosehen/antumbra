//! Warming the models a recall waits on, once, when the server starts.
//!
//! The first embedding and the first rerank after a restart pay for loading
//! weights and setting up GPU kernels, and the first caller is usually the
//! session-start hook, which gives up after five seconds. Measured on kuskokwim
//! on 2026-10-04: the first recall after a deploy took 5.8 s, the next ones
//! 0.15 to 0.17 s. So the server spends that time itself, before anyone asks.
//!
//! A networked server warms its embedder before it listens. Warmed in the
//! background, a call that came in meanwhile still waited for it: measured on
//! kuskokwim on 2026-10-12, a recall 3 s after a restart spent 3.8 s embedding
//! its query, finishing in the same millisecond as the warm-up, against 16 ms
//! warm. Every recall and every store embeds, so until the embedder answers
//! the server has little it can answer quickly, and a client that finds the
//! port closed for those seconds is refused at once instead of left waiting.
//! The wait is bounded, for an embedder behind an endpoint that is down.
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

/// How long a networked server waits for its embedder's first answer before
/// it listens anyway.
pub(crate) const EMBED_PATIENCE: Duration = Duration::from_secs(20);

/// Warm the embedder now, waiting at most `patience` for its first answer.
/// Whether it answered.
pub(crate) async fn embedder_now(embedder: &dyn Embedder, patience: Duration) -> bool {
    let started = Instant::now();
    match tokio::time::timeout(patience, embedder.embed("warm up")).await {
        Ok(Ok(_)) => {
            eprintln!(
                "antumbra-mcp: embedder warm in {} ms",
                started.elapsed().as_millis()
            );
            true
        }
        Ok(Err(e)) => {
            eprintln!("antumbra-mcp: embedder warm-up failed: {e}");
            false
        }
        Err(_) => {
            eprintln!(
                "antumbra-mcp: embedder not answering after {} s, serving anyway",
                patience.as_secs()
            );
            false
        }
    }
}

/// Warm the embedder, unless `embedder` is `None` because it is warm already,
/// and, if there is one, the reranker, in the background.
pub(crate) fn spawn(
    embedder: Option<Arc<dyn Embedder>>,
    scorer: Option<Arc<dyn RelevanceScorer>>,
    recheck: Option<Recheck>,
) {
    tokio::spawn(async move {
        if let Some(embedder) = embedder {
            let started = Instant::now();
            match embedder.embed("warm up").await {
                Ok(_) => eprintln!(
                    "antumbra-mcp: embedder warm in {} ms",
                    started.elapsed().as_millis()
                ),
                Err(e) => eprintln!("antumbra-mcp: embedder warm-up failed: {e}"),
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use antumbra_core::testing::FixedEmbedder;
    use antumbra_core::AntumbraError;

    /// Answers after `delay`, or refuses.
    struct Slow {
        delay: Duration,
        refuse: bool,
    }

    #[async_trait::async_trait]
    impl Embedder for Slow {
        async fn embed(&self, _text: &str) -> antumbra_core::Result<Vec<f32>> {
            tokio::time::sleep(self.delay).await;
            if self.refuse {
                return Err(AntumbraError::Other("down".into()));
            }
            Ok(vec![0.0; 4])
        }

        fn dim(&self) -> usize {
            4
        }
    }

    #[tokio::test]
    async fn the_embedder_is_warm_when_it_answers_within_the_bound() {
        assert!(embedder_now(&FixedEmbedder::new(8), Duration::from_secs(5)).await);
        let refusing = Slow {
            delay: Duration::ZERO,
            refuse: true,
        };
        assert!(!embedder_now(&refusing, Duration::from_secs(5)).await);
    }

    #[tokio::test]
    async fn an_embedder_slower_than_the_bound_does_not_hold_the_server() {
        let slow = Slow {
            delay: Duration::from_secs(30),
            refuse: false,
        };
        let started = Instant::now();
        assert!(!embedder_now(&slow, Duration::from_millis(200)).await);
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "{:?}",
            started.elapsed()
        );
        assert!(EMBED_PATIENCE <= Duration::from_secs(30));
    }
}
