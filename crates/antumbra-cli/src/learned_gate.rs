//! The learned gate, kept current by the commands that change the population:
//! trained over the experts the gate may route to, and re-frozen while they
//! hold still (ADR-0022 S-5).

use antumbra_core::ports::Embedder;
use antumbra_loop::gate_exemplars;
use antumbra_store::repo::lifecycle;
use antumbra_store::Store;

/// What a refresh did to the learned router.
pub(crate) enum RouterRefresh {
    /// Retrained, because the population it routes over changed.
    Trained(antumbra_core::LearnedRouter),
    /// Left frozen: the experts it routes over are the ones it was trained
    /// over.
    Unchanged,
    /// Cleared: too few routable experts to need one.
    Cleared,
}

/// Keep the learned gate current without letting it drift (ADR-0022 S-5: the
/// gate re-opens for training when the population changes, then closes).
/// Retrains only when the experts the gate may route to, those with
/// exemplars, are not the ones the stored router was trained over. This is the
/// self-maintaining gate: `train`/`teach` call it so routing stays current
/// without a manual `gate-train`, which always retrains.
pub(crate) async fn refresh_router(
    store: &Store,
    embedder: &dyn Embedder,
    epochs: usize,
) -> anyhow::Result<RouterRefresh> {
    let routable = lifecycle::routable(store).await?;
    let covered = routable
        .iter()
        .filter(|e| !e.exemplars().is_empty())
        .map(|e| &e.id);
    if let Some(current) = antumbra_store::repo::router::load(store).await? {
        if current.trained_over(covered) {
            return Ok(RouterRefresh::Unchanged);
        }
    }
    Ok(match train_router(store, embedder, epochs).await? {
        Some(router) => RouterRefresh::Trained(router),
        None => RouterRefresh::Cleared,
    })
}

/// Train the learned router over the exemplars of the experts the gate may
/// route to (the active ones, ADR-0022 S-5) and persist it (the learned gate).
/// Returns the router, or `None` when too few are routable to need one (<2
/// experts/exemplars), in which case any router left from before is cleared,
/// so routing falls back to the heuristic gate over the population as it now
/// is.
pub(crate) async fn train_router(
    store: &Store,
    embedder: &dyn Embedder,
    epochs: usize,
) -> anyhow::Result<Option<antumbra_core::LearnedRouter>> {
    let experts = lifecycle::routable(store).await?;
    let Some(exemplars) = gate_exemplars(&experts, embedder).await? else {
        antumbra_store::repo::router::clear(store).await?;
        return Ok(None);
    };
    let router = antumbra_train::train_learned_router(&exemplars, epochs)?;
    antumbra_store::repo::router::save(store, &router).await?;
    Ok(Some(router))
}
