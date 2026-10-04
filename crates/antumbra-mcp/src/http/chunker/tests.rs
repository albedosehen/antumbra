use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;

use antumbra_core::testing::FixedEmbedder;
use antumbra_core::{AntumbraError, CompartmentId, Memory, MemoryId, MemoryNetwork};
use antumbra_store::repo::memory;
use antumbra_store::{Store, EMBED_DIM};

use super::*;
use crate::http::tests::state_embedding;

/// Embeds as [`FixedEmbedder`] does, counting the texts, and fails on any
/// text with "unreadable" in it.
struct Counting {
    inner: FixedEmbedder,
    calls: AtomicUsize,
}

#[async_trait]
impl Embedder for Counting {
    async fn embed(&self, text: &str) -> antumbra_core::Result<Vec<f32>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if text.contains("unreadable") {
            return Err(AntumbraError::Other("the embedder refused".into()));
        }
        self.inner.embed(text).await
    }

    fn dim(&self) -> usize {
        self.inner.dim()
    }
}

fn counting() -> Arc<Counting> {
    Arc::new(Counting {
        inner: FixedEmbedder::new(EMBED_DIM),
        calls: AtomicUsize::new(0),
    })
}

async fn keeper(embedder: Arc<Counting>) -> Arc<HttpState> {
    state_embedding(Store::connect_memory(EMBED_DIM).await.unwrap(), embedder).await
}

/// `words` numbered words after `tag`: long enough for several pieces.
fn long(tag: &str, words: usize) -> String {
    (0..words)
        .map(|i| format!("{tag}{i:04}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn note(id: &str, content: &str) -> Memory {
    Memory::new(
        id,
        TenantId::new("ws:a"),
        MemoryNetwork::World,
        content,
        0.8,
        Utc::now(),
    )
}

async fn held(state: &HttpState) -> std::collections::HashMap<String, Indexed> {
    memory_chunk::indexed(&state.store, &TenantId::new("ws:a"))
        .await
        .unwrap()
}

/// The first pass cuts every long memory and leaves a short one whole; the
/// next finds nothing to do and embeds nothing.
#[tokio::test]
async fn a_pass_cuts_the_long_memories_and_the_next_has_nothing_to_do() {
    let embedder = counting();
    let state = keeper(embedder.clone()).await;
    let text = long("alpha", 200);
    let pieces = split(&text, MEMORY_CHUNK_CHARS).len();
    assert!(pieces > 2, "{pieces}");
    memory::upsert(&state.store, &note("memory:long", &text))
        .await
        .unwrap();
    memory::upsert(&state.store, &note("memory:short", "a short memory"))
        .await
        .unwrap();

    let first = pass(&state, None, 4).await.unwrap();
    assert_eq!(
        first,
        Tally {
            cut: 1,
            pieces,
            ..Tally::default()
        }
    );
    let index = held(&state).await;
    assert_eq!(index.len(), 1, "the short memory is read whole");
    assert_eq!(index["memory:long"].content_hash, cut_hash(&text));
    let embedded = embedder.calls.load(Ordering::SeqCst);
    assert_eq!(embedded, pieces);

    assert_eq!(pass(&state, None, 4).await.unwrap(), Tally::default());
    let later = Utc::now() + chrono::Duration::minutes(1);
    assert_eq!(
        pass(&state, Some(later), 4).await.unwrap(),
        Tally::default()
    );
    assert_eq!(embedder.calls.load(Ordering::SeqCst), embedded);
}

/// The pass after an edit re-cuts it, refiles a moved memory without
/// embedding it again, and drops a forgotten one and one grown short; the full
/// pass drops the chunks of a memory purged before any pass saw it go.
#[tokio::test]
async fn passes_follow_edits_moves_forgetting_and_purges() {
    let embedder = counting();
    let state = keeper(embedder.clone()).await;
    let tenant = TenantId::new("ws:a");
    for id in ["edited", "moved", "forgotten", "purged", "shortened"] {
        let m = note(&format!("memory:{id}"), &long(id, 150));
        memory::upsert(&state.store, &m).await.unwrap();
    }
    assert_eq!(pass(&state, None, 2).await.unwrap().cut, 5);
    let since = Utc::now();
    let embedded = embedder.calls.load(Ordering::SeqCst);

    let edited = note("memory:edited", &long("rewritten", 150));
    memory::upsert(&state.store, &edited).await.unwrap();
    let mut moved = memory::get(&state.store, &tenant, &MemoryId::new("memory:moved"))
        .await
        .unwrap()
        .unwrap()
        .in_compartment(CompartmentId::new("comp:elsewhere"));
    moved.updated_at = Utc::now();
    memory::upsert(&state.store, &moved).await.unwrap();
    memory::delete(&state.store, &tenant, &MemoryId::new("memory:purged"))
        .await
        .unwrap();
    memory::soft_delete(
        &state.store,
        &tenant,
        &MemoryId::new("memory:forgotten"),
        Utc::now(),
    )
    .await
    .unwrap();
    memory::upsert(&state.store, &note("memory:shortened", "now short"))
        .await
        .unwrap();

    let after = pass(&state, Some(since), 2).await.unwrap();
    let edited_pieces = split(&edited.content, MEMORY_CHUNK_CHARS).len();
    assert_eq!(
        after,
        Tally {
            cut: 1,
            pieces: edited_pieces,
            refiled: 1,
            dropped: 2,
            failed: 0,
        }
    );
    assert_eq!(
        embedder.calls.load(Ordering::SeqCst),
        embedded + edited_pieces,
        "only the edited memory was embedded again"
    );
    let index = held(&state).await;
    assert_eq!(
        index["memory:edited"].content_hash,
        cut_hash(&edited.content)
    );
    assert!(index["memory:moved"].filed_as(&moved));
    assert!(!index.contains_key("memory:forgotten"));
    assert!(!index.contains_key("memory:shortened"));
    assert!(
        index.contains_key("memory:purged"),
        "no incremental pass saw it"
    );

    let full = pass(&state, None, 2).await.unwrap();
    assert_eq!(
        full,
        Tally {
            dropped: 1,
            ..Tally::default()
        }
    );
    assert!(!held(&state).await.contains_key("memory:purged"));
}

/// A memory the embedder cannot read is skipped and counted, and the pass goes
/// on; an embedder that fails memory after memory stops the pass.
#[tokio::test]
async fn an_embedder_failure_skips_a_memory_and_a_down_embedder_stops_the_pass() {
    let state = keeper(counting()).await;
    memory::upsert(&state.store, &note("memory:fine", &long("fine", 150)))
        .await
        .unwrap();
    memory::upsert(&state.store, &note("memory:odd", &long("unreadable", 150)))
        .await
        .unwrap();
    let t = pass(&state, None, 4).await.unwrap();
    assert_eq!((t.cut, t.failed), (1, 1));
    assert!(held(&state).await.contains_key("memory:fine"));

    let state = keeper(counting()).await;
    for i in 0..FAILURES_IN_A_ROW {
        let m = note(&format!("memory:odd-{i}"), &long("unreadable", 150));
        memory::upsert(&state.store, &m).await.unwrap();
    }
    let e = pass(&state, None, 4).await.unwrap_err();
    assert!(format!("{e:#}").contains("in a row"), "{e:#}");
}
