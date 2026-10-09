//! The reward gate over the verifier namespace.
//!
//! A task's `verify` is either a spec written into the task, which is authored
//! and runs as it is, or a reference to a verifier in the namespace,
//! `{"verifier": "verifier:..."}`. A reference is resolved on every check and
//! runs only while that verifier may grant reward for the task. Otherwise the
//! task cannot earn reward, as a task with no spec cannot. So a quarantine
//! stops a verifier's reward from the next check on, mid-run included.

use std::sync::Arc;

use async_trait::async_trait;

use antumbra_core::ports::{TrustedVerifiers, Verifier, VerifierVerdict, VerifyRequest};
use antumbra_core::{named_verifier, Result};

pub struct Governed<V> {
    inner: V,
    registry: Arc<dyn TrustedVerifiers>,
}

impl<V> Governed<V> {
    pub fn new(inner: V, registry: Arc<dyn TrustedVerifiers>) -> Self {
        Governed { inner, registry }
    }
}

#[async_trait]
impl<V: Verifier> Verifier for Governed<V> {
    async fn verify(&self, req: &VerifyRequest) -> Result<VerifierVerdict> {
        let named = req.artifact.get("verify").and_then(named_verifier);
        let Some(named) = named else {
            return self.inner.verify(req).await;
        };
        let task = req
            .artifact
            .get("task")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        match self.registry.trusted_spec(&named, task).await? {
            Some(spec) => {
                let mut resolved = req.clone();
                resolved.artifact["verify"] = spec;
                self.inner.verify(&resolved).await
            }
            None => Ok(VerifierVerdict {
                passed: false,
                value: 0.0,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Mutex;

    use antumbra_core::{RunId, VerifierId};

    use super::*;
    use crate::CommandVerifier;

    /// Verifier specs by id, each for one task, removable mid-run.
    #[derive(Default)]
    struct Namespace(Mutex<HashMap<String, (String, serde_json::Value)>>);

    #[async_trait]
    impl TrustedVerifiers for Namespace {
        async fn trusted_spec(
            &self,
            id: &VerifierId,
            task: &str,
        ) -> Result<Option<serde_json::Value>> {
            Ok(self
                .0
                .lock()
                .unwrap()
                .get(id.as_str())
                .filter(|(t, _)| t == task)
                .map(|(_, spec)| spec.clone()))
        }
    }

    fn req(task: &str, completion: &str, verify: serde_json::Value) -> VerifyRequest {
        VerifyRequest {
            run_id: RunId::new("r"),
            step_idx: 0,
            dimension: "exec".into(),
            artifact: serde_json::json!({ "task": task, "completion": completion, "verify": verify }),
        }
    }

    fn named(id: &str) -> serde_json::Value {
        serde_json::json!({ "verifier": id })
    }

    #[tokio::test]
    async fn a_spec_in_the_task_runs_as_it_is() {
        let gate = Governed::new(CommandVerifier, Arc::new(Namespace::default()));
        let spec = serde_json::json!({ "contains_all": ["x"] });
        assert!(
            gate.verify(&req("t", "x", spec.clone()))
                .await
                .unwrap()
                .passed
        );
        assert!(!gate.verify(&req("t", "y", spec)).await.unwrap().passed);
    }

    #[tokio::test]
    async fn a_named_verifier_grants_only_while_trusted_and_only_for_its_task() {
        let namespace = Arc::new(Namespace::default());
        namespace.0.lock().unwrap().insert(
            "verifier:v".into(),
            ("t".into(), serde_json::json!({ "contains_all": ["x"] })),
        );
        let gate = Governed::new(CommandVerifier, namespace.clone());
        assert!(
            gate.verify(&req("t", "x", named("verifier:v")))
                .await
                .unwrap()
                .passed
        );
        assert!(
            !gate
                .verify(&req("t", "y", named("verifier:v")))
                .await
                .unwrap()
                .passed
        );
        assert!(
            !gate
                .verify(&req("u", "x", named("verifier:v")))
                .await
                .unwrap()
                .passed
        );
        assert!(
            !gate
                .verify(&req("t", "x", named("verifier:unknown")))
                .await
                .unwrap()
                .passed
        );
        // Quarantined between two checks: the next one grants nothing.
        namespace.0.lock().unwrap().clear();
        assert!(
            !gate
                .verify(&req("t", "x", named("verifier:v")))
                .await
                .unwrap()
                .passed
        );
    }

    #[tokio::test]
    async fn without_the_gate_a_named_verifier_grants_nothing() {
        // The spec a reference carries is not one CommandVerifier can run.
        assert!(
            !CommandVerifier
                .verify(&req("t", "x", named("verifier:v")))
                .await
                .unwrap()
                .passed
        );
    }
}
