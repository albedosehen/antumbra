//! The real `AcceptabilityProbe` (ADR-0004): **generate-then-verify**.
//!
//! Hold a behavior fixed, render it for a candidate context, **serve** a
//! completion, and let a **verifier** judge it. This retires the keystone's
//! last fake: acceptability is decided by actually producing the behavior and
//! checking it against the environment, not asserted. It is generic over the
//! `Serve` and `Verifier` ports, so it composes the candle server
//! (`CandleServe`) with the environment verifier (`CommandVerifier`) in
//! production, and deterministic fakes in tests.

use async_trait::async_trait;
use serde_json::{json, Value};

use antumbra_core::ports::{AcceptabilityProbe, ActRequest, Serve, Verifier, VerifyRequest};
use antumbra_core::{Result, RunId};

/// Probes acceptability by serving the behavior in a context and verifying it.
///
/// Generation is stochastic, so a single sample is a noisy acceptability test (a
/// model that *can* satisfy a context still fails on some draws). The probe is
/// therefore **best-of-K**: a context is acceptable if any of `samples` served
/// completions verifies. This is the kill-criterion fix from ADR-0004: raise
/// `samples` until single-draw noise stops flipping the boundary.
pub struct GenerateVerifyProbe<S: Serve, V: Verifier> {
    serve: S,
    verifier: V,
    samples: usize,
}

impl<S: Serve, V: Verifier> GenerateVerifyProbe<S, V> {
    pub fn new(serve: S, verifier: V) -> Self {
        Self {
            serve,
            verifier,
            samples: 1,
        }
    }

    /// Set the best-of-K sample budget per acceptability check (min 1).
    pub fn with_samples(mut self, samples: usize) -> Self {
        self.samples = samples.max(1);
        self
    }

    /// Render the fixed behavior plus the context (minus its `verify` spec) into
    /// a prompt. The context's other fields steer what the model produces, so a
    /// single-feature change to the context changes the served behavior.
    fn render(behavior: &str, context: &Value) -> String {
        let mut prompt = format!("# {behavior}\n");
        if let Some(obj) = context.as_object() {
            for (key, value) in obj {
                if key == "verify" {
                    continue;
                }
                let rendered = value
                    .as_str()
                    .map(str::to_string)
                    .unwrap_or_else(|| value.to_string());
                prompt.push_str(&format!("# {key}: {rendered}\n"));
            }
        }
        prompt
    }
}

#[async_trait]
impl<S: Serve, V: Verifier> AcceptabilityProbe for GenerateVerifyProbe<S, V> {
    async fn acceptable(&self, behavior: &str, context: &Value) -> Result<bool> {
        let prompt = Self::render(behavior, context);
        let verify = context.get("verify").cloned().unwrap_or(Value::Null);
        for sample in 0..self.samples {
            let output = self
                .serve
                .act(ActRequest {
                    task_id: "probe".into(),
                    prompt: prompt.clone(),
                    adapters: Vec::new(),
                })
                .await?;
            let request = VerifyRequest {
                run_id: RunId::new("probe"),
                step_idx: sample as u32,
                dimension: "acceptability".into(),
                artifact: json!({ "completion": output.final_output, "verify": verify }),
            };
            if self.verifier.verify(&request).await?.passed {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antumbra_boundary::find_scope;
    use antumbra_core::ports::{ActOutput, StepOutput, VerifierVerdict};

    /// Serves the celsius->fahrenheit formula only when the prompt asks for that
    /// direction; otherwise the inverse. Stands in for `CandleServe`.
    struct ConvertServe;

    #[async_trait]
    impl Serve for ConvertServe {
        async fn act(&self, req: ActRequest) -> Result<ActOutput> {
            let code = if req.prompt.contains("celsius_to_fahrenheit") {
                "def convert(x):\n    return x * 9 / 5 + 32\n"
            } else {
                "def convert(x):\n    return (x - 32) * 5 / 9\n"
            };
            Ok(ActOutput {
                steps: vec![StepOutput {
                    step_idx: 0,
                    content: code.to_string(),
                }],
                final_output: code.to_string(),
            })
        }
    }

    /// The environment truth: convert(100) must be 212 (celsius->fahrenheit).
    /// Stands in for `CommandVerifier` running the generated code.
    struct C2fVerifier;

    #[async_trait]
    impl Verifier for C2fVerifier {
        async fn verify(&self, req: &VerifyRequest) -> Result<VerifierVerdict> {
            let completion = req
                .artifact
                .get("completion")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            let passed = completion.contains("9 / 5 + 32");
            Ok(VerifierVerdict {
                passed,
                value: if passed { 1.0 } else { 0.0 },
            })
        }
    }

    #[tokio::test]
    async fn acceptability_is_decided_by_generate_then_verify() {
        let probe = GenerateVerifyProbe::new(ConvertServe, C2fVerifier);
        let verify = json!({});
        let f2c = json!({ "target": "fahrenheit_to_celsius", "verify": verify });
        let c2f = json!({ "target": "celsius_to_fahrenheit", "verify": verify });
        // The behavior is acceptable only in the context that makes the served
        // code pass the environment check.
        assert!(!probe.acceptable("Write convert(x)", &f2c).await.unwrap());
        assert!(probe.acceptable("Write convert(x)", &c2f).await.unwrap());
    }

    #[tokio::test]
    async fn keystone_search_recovers_c_prime_with_the_real_probe() {
        // find_scope drives the real probe: hold the behavior fixed, vary the
        // governing feature, and recover the context where serving passes.
        let probe = GenerateVerifyProbe::new(ConvertServe, C2fVerifier);
        let fail = json!({ "target": "fahrenheit_to_celsius", "verify": json!({}) });
        let candidates = vec![("target".to_string(), vec![json!("celsius_to_fahrenheit")])];
        let finding = find_scope("Write convert(x)", &fail, &candidates, &probe)
            .await
            .unwrap()
            .expect("a context change flips acceptability");
        assert_eq!(finding.governing_feature, "target");
        assert_eq!(
            finding.near_ok_context["target"],
            json!("celsius_to_fahrenheit")
        );
    }
}
