//! Concrete verifiers (ADR-0003): the environment is the reward.
//!
//! [`CommandVerifier`] runs an external command and treats exit-code 0 as a
//! pass: a test suite that goes green, a build that succeeds, an exec check
//! that returns clean. The command/args/cwd come from the request's `verify`
//! artifact (supplied per corpus task); the candidate completion is exposed to
//! the command via the `ANTUMBRA_COMPLETION` environment variable. This is the
//! ground-truth signal the critic densifies but never overrides.

use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use tokio::process::Command;
use tokio::time::timeout;

use antumbra_core::ports::{Verifier, VerifierVerdict, VerifyRequest};
use antumbra_core::{AntumbraError, Result};

/// Hard cap on a verify subprocess: model-generated code is untrusted and may
/// loop forever or block on input, so a timeout (and null stdin) is mandatory.
const VERIFY_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug, Default, Clone)]
pub struct CommandVerifier;

/// Resolve a verifier `program` to an executable. `python`/`python3` honor the
/// `ANTUMBRA_PYTHON` env var when it is set and non-empty, so a verifier that
/// shells out to `python` works even when a shadowing interpreter is first on
/// `PATH` (e.g. the Windows Store `python.exe` alias, which is not a real
/// interpreter). Everything else is passed through unchanged.
fn resolve_program(program: &str) -> String {
    if matches!(program, "python" | "python3") {
        if let Ok(p) = std::env::var("ANTUMBRA_PYTHON") {
            if !p.trim().is_empty() {
                return p;
            }
        }
    }
    program.to_string()
}

/// Extract the first fenced code block (```` ```lang\n...\n``` ````), or return
/// the trimmed text if there is no fence. Models frequently wrap code in
/// fences; the runnable code is inside.
pub fn extract_code_block(text: &str) -> &str {
    if let Some(start) = text.find("```") {
        let after = &text[start + 3..];
        let body_start = after.find('\n').map(|i| i + 1).unwrap_or(0);
        let body = &after[body_start..];
        if let Some(end) = body.find("```") {
            return body[..end].trim();
        }
        return body.trim();
    }
    text.trim()
}

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
        let raw_completion = req
            .artifact
            .get("completion")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        // Optionally pull the code out of a markdown fence first.
        let completion = if spec
            .get("extract_code")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            extract_code_block(raw_completion)
        } else {
            raw_completion
        };

        // In-process rule (no external process / no Python): the completion must
        // contain every listed substring. A real, deterministic reward.
        if let Some(subs) = spec.get("contains_all").and_then(|v| v.as_array()) {
            let passed = subs
                .iter()
                .filter_map(|s| s.as_str())
                .all(|needle| completion.contains(needle));
            return Ok(if passed {
                VerifierVerdict {
                    passed: true,
                    value: 1.0,
                }
            } else {
                fail()
            });
        }

        // Otherwise run an external command; exit-code 0 = pass.
        let Some(program) = spec.get("program").and_then(|v| v.as_str()) else {
            return Ok(fail());
        };
        let args: Vec<&str> = spec
            .get("args")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_str()).collect())
            .unwrap_or_default();

        let mut cmd = Command::new(resolve_program(program));
        cmd.args(&args)
            .env("ANTUMBRA_COMPLETION", completion)
            .stdin(Stdio::null())
            .kill_on_drop(true);
        if let Some(cwd) = spec.get("cwd").and_then(|v| v.as_str()) {
            cmd.current_dir(cwd);
        }

        // Async wait under a timeout: a runaway or blocking program is a
        // failure, never a hang, and never blocks the runtime thread.
        let mut child = cmd
            .spawn()
            .map_err(|e| AntumbraError::other(format!("verify spawn `{program}`: {e}")))?;
        let passed = match timeout(VERIFY_TIMEOUT, child.wait()).await {
            Ok(status) => status
                .map_err(|e| AntumbraError::other(format!("verify wait `{program}`: {e}")))?
                .success(),
            Err(_elapsed) => {
                let _ = child.kill().await;
                false
            }
        };
        Ok(if passed {
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

    #[cfg(windows)]
    fn runaway() -> serde_json::Value {
        serde_json::json!({ "program": "ping", "args": ["-t", "127.0.0.1"] })
    }
    #[cfg(not(windows))]
    fn runaway() -> serde_json::Value {
        serde_json::json!({ "program": "sh", "args": ["-c", "sleep 60"] })
    }

    #[tokio::test]
    async fn exit_zero_passes_nonzero_fails() {
        let v = CommandVerifier;
        assert!(v.verify(&req(spec(0), "x")).await.unwrap().passed);
        assert!(!v.verify(&req(spec(1), "x")).await.unwrap().passed);
    }

    // Untrusted generated code can run forever; the verifier must kill it and
    // report failure, not hang. Waits the full VERIFY_TIMEOUT, so it is opt-in.
    #[tokio::test]
    #[ignore = "exercises the verify timeout (~10s)"]
    async fn runaway_command_times_out_as_failure() {
        let v = CommandVerifier;
        assert!(!v.verify(&req(runaway(), "x")).await.unwrap().passed);
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

    #[tokio::test]
    async fn contains_all_rule_passes_only_when_all_present() {
        let v = CommandVerifier;
        let spec = serde_json::json!({ "contains_all": ["def add", "return a + b"] });
        let good = req(spec.clone(), "def add(a, b):\n    return a + b\n");
        let bad = req(spec, "def sub(a, b):\n    return a - b\n");
        assert!(v.verify(&good).await.unwrap().passed);
        assert!(!v.verify(&bad).await.unwrap().passed);
    }

    #[test]
    fn extract_code_block_handles_fences_and_plain() {
        assert_eq!(
            extract_code_block("```python\ndef f():\n    pass\n```"),
            "def f():\n    pass"
        );
        assert_eq!(extract_code_block("no fence here"), "no fence here");
        assert_eq!(extract_code_block("```\nx = 1\n```"), "x = 1");
    }

    #[test]
    fn resolve_program_honors_antumbra_python_override() {
        // No env -> pass through.
        std::env::remove_var("ANTUMBRA_PYTHON");
        assert_eq!(resolve_program("python"), "python");

        // Set -> python/python3 resolve to it; other programs are untouched.
        std::env::set_var("ANTUMBRA_PYTHON", "/real/python.exe");
        assert_eq!(resolve_program("python"), "/real/python.exe");
        assert_eq!(resolve_program("python3"), "/real/python.exe");
        assert_eq!(resolve_program("cmd"), "cmd");

        // Empty/whitespace is ignored (treated as unset).
        std::env::set_var("ANTUMBRA_PYTHON", "  ");
        assert_eq!(resolve_program("python"), "python");
        std::env::remove_var("ANTUMBRA_PYTHON");
    }

    // The extract_code path inside verify(): the rule matches the code pulled from
    // a markdown fence, not the surrounding prose.
    #[tokio::test]
    async fn extract_code_then_contains_all() {
        let v = CommandVerifier;
        let spec = serde_json::json!({
            "extract_code": true,
            "contains_all": ["return a + b"],
        });
        let r = req(
            spec,
            "Here you go:\n```python\ndef add(a, b):\n    return a + b\n```",
        );
        assert!(v.verify(&r).await.unwrap().passed);
    }

    // A spec with neither a rule nor a program cannot earn reward.
    #[tokio::test]
    async fn spec_without_program_or_rule_fails() {
        let v = CommandVerifier;
        let r = req(serde_json::json!({ "extract_code": false }), "x");
        assert!(!v.verify(&r).await.unwrap().passed);
    }
}
