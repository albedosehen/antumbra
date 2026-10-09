//! The critic's standing fallback (the learned critic's kill criterion). Once
//! the critic's rank correlation with the verifier is no longer positive, or
//! its agreement with its twin has declined across generations, the run trains
//! on the verifier's reward alone from the next generation on, and stays there.
//!
//! The watches are the run's as this process has seen them, so a resumed run
//! reads its critic afresh.

use antumbra_core::critic::{fallback, CriticWatch, Fallback};

use crate::GenerationLoop;

/// What the run has read of its critic.
#[derive(Debug, Default)]
pub(crate) struct CriticRun {
    watches: Vec<CriticWatch>,
    set_aside: Option<Fallback>,
}

impl GenerationLoop<'_> {
    /// Whether this run's critic has been set aside, so its shadows train on
    /// the verifier's reward alone.
    pub(crate) fn verifier_only(&self) -> bool {
        self.critic
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .set_aside
            .is_some()
    }

    /// Add a generation's watch to the run's, and set the critic aside if the
    /// fallback fires. Returns why, in the generation it fires.
    pub(crate) fn watch_critic(&self, watch: Option<&CriticWatch>) -> Option<Fallback> {
        let mut run = self.critic.lock().unwrap_or_else(|e| e.into_inner());
        if run.set_aside.is_some() {
            return None;
        }
        run.watches.push(watch?.clone());
        run.set_aside = fallback(&run.watches);
        run.set_aside
    }
}
