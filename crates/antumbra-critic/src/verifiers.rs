//! Concrete verifiers (ADR-0003): the environment is the reward.
//!
//! [`CommandVerifier`] runs an external command and treats exit-code 0 as a
//! pass — a test suite that goes green, a build that succeeds, an exec check
//! that returns clean. The command/args/cwd come from the request's `verify`
//! artifact (supplied per corpus task); the candidate completion is exposed to
//! the command via the `ANTUMBRA_COMPLETION` environment variable. This is the
//! ground-truth signal the critic densifies but never overrides.

use std::process::Command;

use async_trait::async_trait;

use antumbra_core::ports::{Verifier, VerifierVerdict, VerifyRequest};
use antumbra_core::{AntumbraError, Result};

#[derive(Debug, Default, Clone)]
pub struct CommandVerifier;

fn fail() -> VerifierVerdict {
    VerifierVerdict {
        passed: false,
        value: 0.0,
    }
}

#[async_trait]
impl Verifier for CommandVerifier {
    async fn verify(&self, req: &VerifyRequest) -> Result<VerifierVerdict> {
        let Some(spec) = req.artifact.get("verify") else {
            // No verify spec on this task -> cannot earn reward.
            return Ok(fail());
        };
        let Some(program) = spec.get("program").and_then(|v| v.as_str()) else {
            return Err(AntumbraError::other("verify spec missing `program`"));
        };
        let args: Vec<&str> = spec
            .get("args")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_str()).collect())
            .unwrap_or_default();
        let completion = req
            .artifact
            .get("completion")
            .and_then(|v| v.as_str())
            .unwrap_or_default();

        let mut cmd = Command::new(program);
        cmd.args(&args).env("ANTUMBRA_COMPLETION", completion);
        if let Some(cwd) = spec.get("cwd").and_then(|v| v.as_str()) {
            cmd.current_dir(cwd);
        }

        let status = cmd
            .status()
            .map_err(|e| AntumbraError::other(format!("verify spawn `{program}`: {e}")))?;
        Ok(if status.success() {
            VerifierVerdict {
                passed: true,
                value: 1.0,
            }
        } else {
            fail()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antumbra_core::RunId;

    fn req(verify: serde_json::Value, completion: &str) -> VerifyRequest {
        VerifyRequest {
            run_id: RunId::new("r"),
            step_idx: 0,
            dimension: "exec".into(),
            artifact: serde_json::json!({ "completion": completion, "verify": verify }),
        }
    }

    #[cfg(windows)]
    fn spec(code: u8) -> serde_json::Value {
        serde_json::json!({ "program": "cmd", "args": ["/C", format!("exit {code}")] })
    }
    #[cfg(not(windows))]
    fn spec(code: u8) -> serde_json::Value {
        serde_json::json!({ "program": "sh", "args": ["-c", format!("exit {code}")] })
    }

    #[tokio::test]
    async fn exit_zero_passes_nonzero_fails() {
        let v = CommandVerifier;
        assert!(v.verify(&req(spec(0), "x")).await.unwrap().passed);
        assert!(!v.verify(&req(spec(1), "x")).await.unwrap().passed);
    }

    #[tokio::test]
    async fn missing_spec_does_not_pass() {
        let v = CommandVerifier;
        let r = VerifyRequest {
            run_id: RunId::new("r"),
            step_idx: 0,
            dimension: "x".into(),
            artifact: serde_json::json!({ "completion": "x" }),
        };
        assert!(!v.verify(&r).await.unwrap().passed);
    }
}
