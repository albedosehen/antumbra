//! Bookkeeping for the autonomous consolidation trigger, and the one rule about
//! where a train may run.
//!
//! Both exist because of what the deployed loop did the first time it fired on
//! the GPU node (EXP-022). The train ran inside `tokio::spawn`, and a train is
//! minutes of synchronous compute wrapped in an `async fn`: it pinned a runtime
//! worker, the store's connection driver queued behind it, and every tool call
//! that touches the store hung until the train ended. A `store_memory` sent
//! during the train timed out and was lost. Separately, a memory that arrived
//! while its compartment was training was never looked at again, because the
//! in-flight guard swallowed the write and nothing re-checked afterwards.

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::sync::Arc;

/// Per-compartment state shared by every session of a server. The HTTP surface
/// builds a fresh `McpServer` per request, so this is what lets concurrent
/// writes to one compartment collapse into one train.
#[derive(Debug, Default)]
pub struct ConsolidationState {
    /// Compartments with a consolidation running or cooling down.
    inflight: HashSet<String>,
    /// Compartments written to while in flight: run once more when it ends.
    dirty: HashSet<String>,
    /// The last gate summary said per compartment, so an unchanged "nothing
    /// graduates" is said once and not on every write.
    last_report: HashMap<String, String>,
}

/// The shared handle: one per server, cloned into each session.
pub type SharedConsolidation = Arc<tokio::sync::Mutex<ConsolidationState>>;

#[cfg_attr(not(feature = "models"), allow(dead_code))]
impl ConsolidationState {
    /// Claim a compartment. `false` means one is already in flight; the write is
    /// remembered so the run in flight is followed by another.
    pub fn begin(&mut self, compartment: &str) -> bool {
        if self.inflight.contains(compartment) {
            self.dirty.insert(compartment.to_string());
            return false;
        }
        self.inflight.insert(compartment.to_string());
        true
    }

    /// A run ended. `true` means a write arrived meanwhile, so the caller runs
    /// again and keeps the claim; `false` releases it.
    pub fn finish(&mut self, compartment: &str) -> bool {
        if self.dirty.remove(compartment) {
            return true;
        }
        self.inflight.remove(compartment);
        false
    }

    /// Whether `summary` differs from the last one said for this compartment,
    /// recording it if so.
    pub fn changed(&mut self, compartment: &str, summary: &str) -> bool {
        if self.last_report.get(compartment).map(String::as_str) == Some(summary) {
            return false;
        }
        self.last_report
            .insert(compartment.to_string(), summary.to_string());
        true
    }

    /// Forget what was last said, so the next held-back run speaks again. Called
    /// after a mint: the compartment's situation has changed.
    pub fn forget_report(&mut self, compartment: &str) {
        self.last_report.remove(compartment);
    }
}

/// Run a future that mixes awaits with long synchronous compute (a model load, a
/// training loop, a generation) on the blocking pool instead of a runtime worker.
/// The future still awaits the store normally: its I/O is driven by the runtime,
/// which stays free because the compute is no longer sitting on one of its
/// workers.
pub fn spawn_heavy<F>(future: F) -> tokio::task::JoinHandle<F::Output>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    let runtime = tokio::runtime::Handle::current();
    tokio::task::spawn_blocking(move || runtime.block_on(future))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn a_second_claim_is_refused_and_remembered() {
        let mut state = ConsolidationState::default();
        assert!(state.begin("comp:a"));
        assert!(!state.begin("comp:a"), "already in flight");
        assert!(state.begin("comp:b"), "another compartment is independent");
        // The refused write makes the finished run go again, exactly once.
        assert!(state.finish("comp:a"));
        assert!(!state.begin("comp:a"), "still claimed during the rerun");
        assert!(state.finish("comp:a"), "the rerun was written to as well");
        assert!(!state.finish("comp:a"), "quiet now: released");
        assert!(state.begin("comp:a"), "and claimable again");
    }

    #[test]
    fn a_quiet_run_releases_its_claim() {
        let mut state = ConsolidationState::default();
        assert!(state.begin("comp:a"));
        assert!(!state.finish("comp:a"));
        assert!(state.begin("comp:a"));
    }

    #[test]
    fn an_unchanged_report_is_said_once() {
        let mut state = ConsolidationState::default();
        let said = "0 of 400 graduate (400 under-reinforced)";
        assert!(state.changed("comp:a", said));
        assert!(!state.changed("comp:a", said));
        assert!(state.changed("comp:b", said), "per compartment");
        assert!(state.changed("comp:a", "0 of 401 graduate (401 under-reinforced)"));
        state.forget_report("comp:a");
        assert!(state.changed("comp:a", "0 of 401 graduate (401 under-reinforced)"));
    }

    /// The regression this module exists for. One runtime worker, and a task that
    /// computes synchronously for a while: other work must still run promptly.
    /// Spawned with `tokio::spawn`, the second task waits out the whole compute.
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn heavy_work_does_not_stall_the_runtime() -> anyhow::Result<()> {
        let compute = Duration::from_millis(900);
        let (computing, started) = std::sync::mpsc::channel();
        let heavy = spawn_heavy(async move {
            computing.send(())?;
            std::thread::sleep(compute);
            Ok::<_, std::sync::mpsc::SendError<()>>(7)
        });
        // Wait until the heavy task is inside its synchronous section. Not a
        // timer: the lone worker also drives the timers, so with the worker
        // pinned a sleep here would only fire once the compute was over, and the
        // measurement below would start after the stall it is meant to catch.
        started.recv()?;

        let asked = Instant::now();
        let light = tokio::spawn(async { 1 }).await?;
        let waited = asked.elapsed();

        assert_eq!(light, 1);
        assert!(
            waited < compute / 3,
            "a light task waited {waited:?} behind {compute:?} of compute"
        );
        assert_eq!(heavy.await??, 7, "and the heavy task still finishes");
        Ok(())
    }

    /// Awaits inside heavy work resolve: the blocking thread drives the future,
    /// the runtime drives its timers and I/O.
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn heavy_work_can_still_await() -> anyhow::Result<()> {
        let out = spawn_heavy(async {
            tokio::time::sleep(Duration::from_millis(20)).await;
            "done"
        })
        .await?;
        assert_eq!(out, "done");
        Ok(())
    }
}
