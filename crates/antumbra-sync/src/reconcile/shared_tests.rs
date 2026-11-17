use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use super::*;
use antumbra_core::{AntumbraError, Memory, MemoryId, MemoryNetwork, TenantId};
use antumbra_store::repo::memory;
use antumbra_store::testing::mem_store;
use chrono::Duration as ChronoDuration;

use crate::scope::Replicate;

const MEMORY: &[TableSpec] = &[TableSpec {
    name: "memory",
    version_field: "updated_at",
    replicate: Replicate::Owned(|scope: &Scope, row| scope.holds_memory(row)),
    carried: None,
}];

fn tenant() -> TenantId {
    TenantId::new("t")
}

fn mem(id: &str, content: &str, at: DateTime<Utc>) -> Memory {
    Memory::new(id, tenant(), MemoryNetwork::World, content, 0.9, at)
}

async fn content_of(store: &Store, id: &str) -> Result<String> {
    memory::get(store, &tenant(), &MemoryId::new(id))
        .await?
        .map(|m| m.content)
        .ok_or_else(|| AntumbraError::other(format!("{id} is missing")))
}

/// A store other work shares, as a server's is: it says while it is held, and
/// counts how often it was.
struct Shared {
    held: Arc<AtomicBool>,
    holds: AtomicUsize,
}

impl Shared {
    fn new() -> Self {
        Shared {
            held: Arc::new(AtomicBool::new(false)),
            holds: AtomicUsize::new(0),
        }
    }
}

/// Lets the store go when the step is done.
struct Let(Arc<AtomicBool>);

impl Drop for Let {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

impl Hold for Shared {
    const SHARED: bool = true;
    type Held = Let;

    async fn hold(&self) -> Result<Let> {
        self.holds.fetch_add(1, Ordering::SeqCst);
        self.held.store(true, Ordering::SeqCst);
        Ok(Let(self.held.clone()))
    }
}

/// The other store of the pass, reached over a network in life. Each step on
/// it first checks that the shared store has been let go, then does whatever
/// the test left for the moment the pass is away.
struct Away {
    shared: Arc<AtomicBool>,
    meanwhile: tokio::sync::Mutex<Option<(Store, Memory)>>,
}

impl Away {
    fn from(shared: &Shared) -> Self {
        Away {
            shared: shared.held.clone(),
            meanwhile: tokio::sync::Mutex::new(None),
        }
    }
}

impl Hold for Away {
    const SHARED: bool = false;
    type Held = ();

    async fn hold(&self) -> Result<()> {
        if self.shared.load(Ordering::SeqCst) {
            return Err(AntumbraError::other(
                "the shared store was still held when the pass went to the other one",
            ));
        }
        if let Some((store, written)) = self.meanwhile.lock().await.take() {
            memory::upsert(&store, &written).await?;
        }
        Ok(())
    }
}

/// The reason the pass is taken apart. A request on the server waits for the
/// shared store, so the store is let go before every step on the other one,
/// with rows moving both ways.
#[tokio::test]
async fn a_shared_store_is_let_go_before_every_step_on_the_other() -> Result<()> {
    let (local, remote) = (mem_store().await?, mem_store().await?);
    let now = Utc::now();
    memory::upsert(
        &local,
        &mem("aaaaaaaa-0000-0000-0000-000000000001", "left", now),
    )
    .await?;
    memory::upsert(
        &remote,
        &mem("bbbbbbbb-0000-0000-0000-000000000002", "right", now),
    )
    .await?;
    let shared = Shared::new();
    let away = Away::from(&shared);

    let stats = reconcile_all_since_on(
        Side {
            store: &local,
            hold: &shared,
        },
        Side {
            store: &remote,
            hold: &away,
        },
        MEMORY,
        &mut Cursors::new(),
        Duration::ZERO,
        None,
    )
    .await?;

    assert_eq!((stats.pushed, stats.pulled), (1, 1));
    assert_eq!(memory::list(&local, &tenant()).await?.len(), 2);
    assert_eq!(memory::list(&remote, &tenant()).await?.len(), 2);
    assert!(!shared.held.load(Ordering::SeqCst), "and not held after");
    // Once to read it and once to write the row that was pulled.
    assert_eq!(shared.holds.load(Ordering::SeqCst), 2);
    Ok(())
}

/// What letting the store go costs, and what pays for it. Between a pass's
/// read and its write, a request may write the same record. The pass decided
/// to pull the remote's copy from what it read; the record it finds when it
/// comes back is newer than that copy, and is kept.
#[tokio::test]
async fn a_record_written_while_the_pass_was_away_is_not_replaced_by_an_older_one() -> Result<()> {
    let (local, remote) = (mem_store().await?, mem_store().await?);
    let id = "cccccccc-0000-0000-0000-000000000003";
    let t0 = Utc::now();
    memory::upsert(&local, &mem(id, "as the pass read it", t0)).await?;
    memory::upsert(
        &remote,
        &mem(id, "the remote's", t0 + ChronoDuration::seconds(5)),
    )
    .await?;
    let shared = Shared::new();
    let away = Away::from(&shared);
    let later = mem(id, "written meanwhile", t0 + ChronoDuration::seconds(9));
    *away.meanwhile.lock().await = Some((local.clone(), later));
    let sides = || {
        (
            Side {
                store: &local,
                hold: &shared,
            },
            Side {
                store: &remote,
                hold: &away,
            },
        )
    };

    let (here, there) = sides();
    let stats = reconcile_all_since_on(
        here,
        there,
        MEMORY,
        &mut Cursors::new(),
        Duration::ZERO,
        None,
    )
    .await?;
    assert_eq!(
        (stats.pulled, stats.pushed),
        (0, 0),
        "the older copy was not written"
    );
    assert_eq!(content_of(&local, id).await?, "written meanwhile");

    // And the next pass carries the newer write to the remote.
    let (here, there) = sides();
    let stats = reconcile_all_since_on(
        here,
        there,
        MEMORY,
        &mut Cursors::new(),
        Duration::ZERO,
        None,
    )
    .await?;
    assert_eq!((stats.pulled, stats.pushed), (0, 1));
    assert_eq!(content_of(&remote, id).await?, "written meanwhile");
    Ok(())
}

/// A first pass can pull a whole table. The shared store is let go between
/// batches, so what waits for it waits for one batch and not for the table.
#[tokio::test]
async fn a_shared_store_is_written_a_batch_at_a_time() -> Result<()> {
    let (local, remote) = (mem_store().await?, mem_store().await?);
    let now = Utc::now();
    let rows = BATCH * 2 + 10;
    for i in 0..rows {
        let id = format!("dddddddd-0000-0000-0000-{i:012}");
        memory::upsert(&remote, &mem(&id, "pulled", now)).await?;
    }
    let shared = Shared::new();
    let away = Away::from(&shared);

    let stats = reconcile_all_since_on(
        Side {
            store: &local,
            hold: &shared,
        },
        Side {
            store: &remote,
            hold: &away,
        },
        MEMORY,
        &mut Cursors::new(),
        Duration::ZERO,
        None,
    )
    .await?;

    assert_eq!(stats.pulled, rows);
    assert_eq!(memory::list(&local, &tenant()).await?.len(), rows);
    // One read, then three batches of writes.
    assert_eq!(shared.holds.load(Ordering::SeqCst), 4);
    Ok(())
}

/// A pass with nothing to write to the shared store takes it once, to read.
#[tokio::test]
async fn a_pass_with_nothing_to_pull_takes_the_shared_store_only_to_read() -> Result<()> {
    let (local, remote) = (mem_store().await?, mem_store().await?);
    memory::upsert(
        &local,
        &mem("eeeeeeee-0000-0000-0000-000000000005", "left", Utc::now()),
    )
    .await?;
    let shared = Shared::new();
    let away = Away::from(&shared);

    let stats = reconcile_all_since_on(
        Side {
            store: &local,
            hold: &shared,
        },
        Side {
            store: &remote,
            hold: &away,
        },
        MEMORY,
        &mut Cursors::new(),
        Duration::ZERO,
        None,
    )
    .await?;

    assert_eq!((stats.pushed, stats.pulled), (1, 0));
    assert_eq!(shared.holds.load(Ordering::SeqCst), 1);
    Ok(())
}
