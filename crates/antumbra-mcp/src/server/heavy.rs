//! The one rule about where a train or a generation may run.
//!
//! It exists because of what the deployed loop did the first time it trained
//! on the GPU node. The train ran inside `tokio::spawn`, and a train
//! is minutes of synchronous compute wrapped in an `async fn`: it pinned a
//! runtime worker, the store's connection driver queued behind it, and every
//! tool call that touches the store hung until the train ended. A
//! `store_memory` sent during the train timed out and was lost.

use std::future::Future;

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
