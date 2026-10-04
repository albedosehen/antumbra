//! The chunk index's keeper (ADR-0025): one background task that cuts every
//! memory into the pieces `memory_chunk` holds, embeds them, and keeps them
//! current.
//!
//! Memories arrive by many paths (the agent's tools, the CLI's intakes, the
//! GitHub App, sync between nodes) and the store embeds nothing, so the index
//! is derived here, after the fact, rather than by each writer. The first pass
//! reads every memory; each later one reads those changed since the one before.
//! Every hour a pass reads every memory again, for one synced in under an older
//! `updated_at` and for the chunks of a memory purged since.
//!
//! Database work runs in owner mode under the auth lock, a memory at a time,
//! and embedding runs outside it, so a pass never holds a request up for longer
//! than one memory's writes.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Utc};
use futures::{StreamExt, TryStreamExt};

use antumbra_core::chunk::{cut_hash, split, MEMORY_CHUNK_CHARS};
use antumbra_core::ports::Embedder;
use antumbra_core::TenantId;
use antumbra_store::repo::memory_chunk::{self, ChunkSource, Indexed};

use super::HttpState;

/// The pause between passes.
const INTERVAL: Duration = Duration::from_secs(60);
/// Every this many passes reads every memory, not only the changed ones.
const FULL_EVERY: u32 = 60;
/// How far before a pass began the next one reads from: a write that landed
/// while the pass ran can carry an earlier `updated_at` than its start.
const OVERLAP: chrono::Duration = chrono::Duration::minutes(2);
/// Memories in a row the embedder may fail before the pass stops: past this
/// it is the embedder that is down, not a memory it cannot read.
const FAILURES_IN_A_ROW: usize = 3;
/// A long pass says how far it has got every this many memories cut.
const PROGRESS_EVERY: usize = 500;

/// What a pass did.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(super) struct Tally {
    /// Memories cut, for the first time or again.
    pub cut: usize,
    /// Pieces embedded for them.
    pub pieces: usize,
    /// Memories whose chunks moved with them, keeping their vectors.
    pub refiled: usize,
    /// Memories whose chunks were dropped.
    pub dropped: usize,
    /// Memories the embedder failed; the next full pass tries them again.
    pub failed: usize,
}

/// What a memory's chunks need.
#[derive(Debug, PartialEq, Eq)]
enum Work {
    Keep,
    Drop,
    Refile,
    Cut { hash: String, pieces: Vec<String> },
}

/// What `source`'s chunks need, given what the index `held` for it.
fn work(source: &ChunkSource, held: Option<&Indexed>) -> Work {
    let none = if held.is_some() {
        Work::Drop
    } else {
        Work::Keep
    };
    if source.forgotten {
        return none;
    }
    let pieces = split(&source.memory.content, MEMORY_CHUNK_CHARS);
    // A memory that fits in one piece is read whole by its own vector.
    if pieces.len() < 2 {
        return none;
    }
    let hash = cut_hash(&source.memory.content);
    match held {
        Some(h) if h.content_hash == hash && h.filed_as(&source.memory) => Work::Keep,
        Some(h) if h.content_hash == hash => Work::Refile,
        _ => Work::Cut { hash, pieces },
    }
}

/// Start the keeper, embedding up to `in_flight` pieces at once; zero leaves
/// the index as it is, and recall reads whole memories for anything not in it.
pub(super) fn spawn(state: Arc<HttpState>, in_flight: usize) {
    if in_flight == 0 {
        eprintln!("antumbra-mcp: chunk index not kept (ANTUMBRA_CHUNK_IN_FLIGHT=0)");
        return;
    }
    tokio::spawn(async move {
        let mut since: Option<DateTime<Utc>> = None;
        let mut passes: u32 = 0;
        loop {
            let full = since.is_none() || passes.is_multiple_of(FULL_EVERY);
            let began = Utc::now();
            let clock = Instant::now();
            match pass(&state, if full { None } else { since }, in_flight).await {
                Ok(t) => {
                    if full || t != Tally::default() {
                        eprintln!(
                            "antumbra-mcp: chunk index, {} pass in {:.1}s: {} memories cut into {} \
                             pieces, {} refiled, {} dropped, {} the embedder failed",
                            if full { "full" } else { "incremental" },
                            clock.elapsed().as_secs_f32(),
                            t.cut,
                            t.pieces,
                            t.refiled,
                            t.dropped,
                            t.failed,
                        );
                    }
                    since = Some(began - OVERLAP);
                }
                // The watermark stays, so the next pass reads the same memories;
                // what this one finished it finds already done.
                Err(e) => eprintln!("antumbra-mcp: chunk index pass stopped: {e:#}"),
            }
            passes = passes.wrapping_add(1);
            tokio::time::sleep(INTERVAL).await;
        }
    });
}

/// One pass over every workspace: the memories changed since `since`, or all
/// of them when `None`.
pub(super) async fn pass(
    state: &HttpState,
    since: Option<DateTime<Utc>>,
    in_flight: usize,
) -> Result<Tally> {
    let tenants = {
        let _guard = state.auth.lock().await;
        state.store.signin_root().await?;
        memory_chunk::tenants(&state.store).await?
    };
    let mut tally = Tally::default();
    for tenant in &tenants {
        workspace(state, tenant, since, in_flight.max(1), &mut tally)
            .await
            .with_context(|| format!("workspace {}", tenant.as_str()))?;
    }
    Ok(tally)
}

async fn workspace(
    state: &HttpState,
    tenant: &TenantId,
    since: Option<DateTime<Utc>>,
    in_flight: usize,
    tally: &mut Tally,
) -> Result<()> {
    let (sources, mut held, embedder) = {
        let _guard = state.auth.lock().await;
        state.store.signin_root().await?;
        let sources = memory_chunk::sources(&state.store, tenant, since).await?;
        if sources.is_empty() && since.is_some() {
            return Ok(());
        }
        // A full pass needs every memory's chunks, to find those whose memory
        // is gone; a pass over what changed needs only theirs.
        let held = if since.is_none() {
            memory_chunk::indexed(&state.store, tenant).await?
        } else {
            let ids: Vec<String> = sources
                .iter()
                .map(|s| s.memory.id.as_str().to_string())
                .collect();
            memory_chunk::indexed_among(&state.store, tenant, &ids).await?
        };
        (sources, held, state.embedder_for(tenant).await)
    };
    let mut failures = 0;
    for source in &sources {
        let id = source.memory.id.as_str();
        match work(source, held.remove(id).as_ref()) {
            Work::Keep => {}
            Work::Drop => {
                let _guard = state.auth.lock().await;
                state.store.signin_root().await?;
                memory_chunk::remove(&state.store, tenant, id).await?;
                tally.dropped += 1;
            }
            Work::Refile => {
                let _guard = state.auth.lock().await;
                state.store.signin_root().await?;
                memory_chunk::refile(&state.store, &source.memory).await?;
                tally.refiled += 1;
            }
            Work::Cut { hash, pieces } => {
                let count = pieces.len();
                match embed(&embedder, pieces, in_flight).await {
                    Ok(vectors) => {
                        failures = 0;
                        let _guard = state.auth.lock().await;
                        state.store.signin_root().await?;
                        memory_chunk::replace(&state.store, &source.memory, &hash, vectors).await?;
                        tally.cut += 1;
                        tally.pieces += count;
                        if tally.cut.is_multiple_of(PROGRESS_EVERY) {
                            eprintln!(
                                "antumbra-mcp: chunk index, {} memories cut so far",
                                tally.cut
                            );
                        }
                    }
                    Err(e) => {
                        failures += 1;
                        tally.failed += 1;
                        if failures >= FAILURES_IN_A_ROW {
                            bail!("the embedder failed {failures} memories in a row, the last {id}: {e}");
                        }
                        eprintln!(
                            "antumbra-mcp: chunk index skipped {id}, the embedder failed: {e}"
                        );
                    }
                }
            }
        }
    }
    // A full pass read every memory, so what is left belongs to none: a
    // memory purged since its chunks were cut.
    if since.is_none() {
        for id in held.keys() {
            let _guard = state.auth.lock().await;
            state.store.signin_root().await?;
            memory_chunk::remove(&state.store, tenant, id).await?;
            tally.dropped += 1;
        }
    }
    Ok(())
}

/// Each piece's vector, in order, up to `in_flight` requests at once.
async fn embed(
    embedder: &Arc<dyn Embedder>,
    pieces: Vec<String>,
    in_flight: usize,
) -> Result<Vec<Vec<f32>>> {
    // Each request owns its piece and its embedder handle: futures borrowing
    // them fail the compiler's higher-ranked lifetime check in the spawned task.
    let vectors = futures::stream::iter(pieces)
        .map(|piece| {
            let embedder = embedder.clone();
            async move { embedder.embed(&piece).await }
        })
        .buffered(in_flight)
        .try_collect()
        .await?;
    Ok(vectors)
}

#[cfg(test)]
mod tests;
