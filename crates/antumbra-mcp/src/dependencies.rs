//! Writing the dependency graph: a claim recorded as a memory in
//! the workspace's shared pool, reinforced when it is already there, and
//! retracted when its source stops making it. Shared by `record_dependency`
//! and the GitHub App's manifest reading, so an edge recorded either way is
//! the same memory.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};

use antumbra_core::depgraph::Claim;
use antumbra_core::ports::Embedder;
use antumbra_core::{GitProvenance, Memory, MemoryId, MemoryNetwork, TenantId, UserId};
use antumbra_store::repo::memory;
use antumbra_store::Store;

/// What recording a claim did.
pub(crate) struct Recorded {
    pub id: MemoryId,
    /// False when the claim was already recorded and was reinforced instead.
    pub created: bool,
    pub confidence: f32,
    pub reinforcement: u32,
}

/// Record `claim` in `tenant`, by `author` on `host`, with `anchor` (the file
/// that says it, at the commit it was read) as part of its evidence. Seen
/// again, the latest evidence replaces the old and the edge is reinforced,
/// which is what keeps it from fading.
pub(crate) async fn record(
    store: &Store,
    embedder: &dyn Embedder,
    tenant: &TenantId,
    (author, host): (&UserId, &str),
    claim: &Claim,
    anchor: Option<&GitProvenance>,
    now: DateTime<Utc>,
) -> Result<Recorded> {
    let evidence = claim.evidence(anchor);
    let content = claim.content();
    let id = MemoryId::new(claim.memory_id(tenant));

    if let Some(mut m) = memory::get(store, tenant, &id).await? {
        if m.content != content {
            m.embedding = Some(embedder.embed(&content).await?);
            m.content = content;
        }
        m.evidence = evidence;
        memory::upsert(store, &m).await?;
        let reinforced = memory::reinforce(store, tenant, &id, now)
            .await?
            .context("the edge vanished while it was reinforced")?;
        return Ok(Recorded {
            id,
            created: false,
            confidence: reinforced.confidence,
            reinforcement: reinforced.reinforcement,
        });
    }

    let embedding = embedder.embed(&content).await?;
    let m = Memory::new(
        id.clone(),
        tenant.clone(),
        MemoryNetwork::World,
        content,
        claim.source.confidence(),
        now,
    )
    .with_embedding(embedding)
    // The tenant's shared pool, not the author's default compartment: a
    // dependency is the workspace's knowledge, and every member's blast
    // radius reads it.
    .by(author.clone(), host)
    .with_evidence(evidence);
    memory::upsert(store, &m).await?;
    if memory::get(store, tenant, &id).await?.is_none() {
        anyhow::bail!("the dependency did not land in the workspace's shared pool");
    }
    Ok(Recorded {
        id,
        created: true,
        confidence: claim.source.confidence(),
        reinforcement: 0,
    })
}

/// Retract a claim its source no longer makes. The memory is forgotten as a
/// tombstone, so the forgetting reaches every replica, and recording the
/// claim again later brings it back. Whether there was one to retract.
pub(crate) async fn retract(
    store: &Store,
    tenant: &TenantId,
    claim: &Claim,
    now: DateTime<Utc>,
) -> Result<bool> {
    let id = MemoryId::new(claim.memory_id(tenant));
    Ok(memory::soft_delete(store, tenant, &id, now)
        .await?
        .is_some())
}
