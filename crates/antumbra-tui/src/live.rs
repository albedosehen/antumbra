//! Live store watching: subscribe to the tables the console shows so an external
//! write (a captured memory, a graduated expert, a loop step) refreshes the view
//! within a frame instead of waiting for the periodic reload tick. Best-effort: a
//! connection that cannot serve live queries falls back to the periodic reload
//! alone. This matters most against a shared `ws://` server, where the MCP server
//! and the loop write the store the console is watching.

use antumbra_store::repo::sync::{watch_table, ChangeEvent};
use antumbra_store::Store;
use tokio::sync::mpsc;

/// The tables whose changes the console reflects (population, memory, loop, evals).
const WATCHED: &[&str] = &[
    "expert",
    "shadow",
    "failure_boundary",
    "memory",
    "memory_edge",
    "generation_head",
    "evaluation_run",
    "learned_router",
    "loop_control",
];

/// Subscribe to live changes on every watched table. Best-effort: a table whose
/// subscription cannot start (e.g. over an `http` connection) is skipped, leaving
/// the periodic reload as the refresh path, so this never fails the console.
pub async fn watch_store(store: &Store) -> Vec<mpsc::Receiver<ChangeEvent>> {
    let mut rxs = Vec::with_capacity(WATCHED.len());
    for table in WATCHED {
        if let Ok(rx) = watch_table(store, table).await {
            rxs.push(rx);
        }
    }
    rxs
}

/// Drain every pending notification without blocking; `true` if any table changed
/// since the last drain, so the caller should reload. A disconnected watcher just
/// drains empty.
pub fn drained_change(rxs: &mut [mpsc::Receiver<ChangeEvent>]) -> bool {
    let mut changed = false;
    for rx in rxs.iter_mut() {
        while rx.try_recv().is_ok() {
            changed = true;
        }
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use antumbra_core::{Expert, ExpertId, Generation};
    use antumbra_store::repo::expert;
    use antumbra_store::EMBED_DIM;
    use chrono::Utc;

    fn demo_expert() -> Expert {
        Expert {
            id: ExpertId::new("expert:live"),
            name: "live".into(),
            base_model: "base".into(),
            artifact_uri: "mem://live".into(),
            capability_card: serde_json::json!({}),
            capability_vec: Some(vec![0.0; EMBED_DIM]),
            fitness: 1.0,
            frozen_at: Some(Utc::now()),
            generation: Generation::ZERO,
            owner: None,
            compartment: None,
            placed_on: None,
            created_at: Utc::now(),
        }
    }

    #[tokio::test]
    async fn a_write_surfaces_as_a_drained_change() {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        let mut rxs = watch_store(&store).await;
        assert!(!rxs.is_empty(), "the embedded store serves live queries");
        // Nothing written yet, so nothing to drain.
        assert!(!drained_change(&mut rxs), "no change before any write");

        expert::insert(&store, &demo_expert()).await.unwrap();
        // The notification is delivered by a background task; poll briefly for it.
        let mut saw = false;
        for _ in 0..100 {
            if drained_change(&mut rxs) {
                saw = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert!(saw, "the expert write surfaced as a live change");
    }
}
