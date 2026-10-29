//! The operator console's reads and actions ([`antumbra_core::console`]),
//! shared by the server, which serves them to `antumbra tui`, and the
//! console's own in-memory demo, so both read and act alike.
//!
//! [`workspace`] reads through whatever connection it is given: on a token's
//! scoped connection the engine's permissions keep it to that workspace's
//! memories and edges and the experts it may see. [`substrate`] and [`apply`]
//! need the owner's connection, since a workspace session may read none of the
//! substrate's own tables.

use chrono::Utc;

use antumbra_core::console::{Action, Snapshot};
use antumbra_core::generational::LoopCommand;
use antumbra_core::{BoundaryId, ExpertId, Result, ShadowId, ShadowStatus};

use crate::repo::{
    boundary, edge, evaluation, expert, generation, loop_control, memory, router, shadow,
};
use crate::store::Store;

/// The workspace's part of a snapshot: the experts, memories and edges
/// `store` may read. The memories come without their embeddings, grouped by
/// network and strongest first within each.
pub async fn workspace(store: &Store) -> Result<Snapshot> {
    let experts = expert::list(store).await?;
    let mut memories = memory::all_unscoped_lite(store).await?;
    memories.sort_by(|a, b| {
        a.network
            .as_str()
            .cmp(b.network.as_str())
            .then(b.confidence.total_cmp(&a.confidence))
    });
    let edges = edge::all_unscoped(store).await?;
    Ok(Snapshot {
        experts,
        memories,
        edges,
        ..Snapshot::default()
    })
}

/// Fill in the substrate's part of `snapshot`, read as the owner, and mark it
/// an operator's.
pub async fn substrate(store: &Store, snapshot: &mut Snapshot) -> Result<()> {
    snapshot.boundaries = boundary::list(store).await?;
    let mut shadows = shadow::list(store).await?;
    shadows.sort_by_key(|s| std::cmp::Reverse(s.created_at));
    snapshot.shadows = shadows;
    snapshot.loop_heads = generation::all_heads(store).await?;
    snapshot.loop_halt_pending = match snapshot.loop_heads.first() {
        Some(head) => loop_control::load(store, &head.run_id).await? == LoopCommand::Halt,
        None => false,
    };
    snapshot.evals = evaluation::recent_unscoped(store).await?;
    snapshot.router = router::load(store).await?;
    snapshot.operator = true;
    Ok(())
}

/// The whole snapshot from one owner connection, for the console's demo,
/// which reads its own store.
pub async fn everything(store: &Store) -> Result<Snapshot> {
    let mut snapshot = workspace(store).await?;
    substrate(store, &mut snapshot).await?;
    Ok(snapshot)
}

/// Carry out an operator's action, as the owner. Returns whether its target
/// exists; an absent one is left alone.
pub async fn apply(store: &Store, action: &Action) -> Result<bool> {
    match action {
        Action::PruneShadow { shadow } => set_shadow(store, shadow, ShadowStatus::Pruned).await,
        Action::GraduateShadow { shadow } => {
            set_shadow(store, shadow, ShadowStatus::Graduated).await
        }
        Action::FreezeExpert { expert } => set_frozen(store, expert, true).await,
        Action::ThawExpert { expert } => set_frozen(store, expert, false).await,
        Action::DeleteBoundary { boundary } => delete_boundary(store, boundary).await,
        Action::HaltLoop { run } => {
            let exists = generation::load_head(store, run).await?.is_some();
            if exists {
                loop_control::set(store, run, LoopCommand::Halt).await?;
            }
            Ok(exists)
        }
        Action::CancelHalt { run } => {
            loop_control::clear(store, run).await?;
            Ok(generation::load_head(store, run).await?.is_some())
        }
    }
}

async fn set_shadow(store: &Store, id: &ShadowId, status: ShadowStatus) -> Result<bool> {
    let Some(mut s) = shadow::get(store, id).await? else {
        return Ok(false);
    };
    s.status = status;
    shadow::upsert(store, &s).await?;
    Ok(true)
}

async fn set_frozen(store: &Store, id: &ExpertId, freeze: bool) -> Result<bool> {
    if expert::get(store, id).await?.is_none() {
        return Ok(false);
    }
    expert::set_frozen(store, id, freeze.then(Utc::now)).await?;
    Ok(true)
}

async fn delete_boundary(store: &Store, id: &BoundaryId) -> Result<bool> {
    let exists = boundary::list(store)
        .await?
        .iter()
        .any(|b| b.id.as_str() == id.as_str());
    if exists {
        boundary::delete(store, id).await?;
    }
    Ok(exists)
}

#[cfg(test)]
mod tests;
