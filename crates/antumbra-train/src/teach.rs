//! Capture: internalize a *given, verified* correction into the composed model.
//!
//! RAFT (`raft.rs`) discovers a skill from the model's own verified-correct
//! samples. Capture is the other intake path into the same population: a
//! correction supplied from outside (a human's "no, this project uses deno")
//! is checked by the verifier and, if it holds, fine-tuned into a frozen expert.
//! Discovery and capture are two ways competence enters the brain; both are
//! gated on ground truth, neither trusts unverified teacher text.

use antumbra_core::ports::{TrainOutcome, Verifier, VerifyRequest};
use antumbra_core::{governing_feature_from_pair, BoundaryFinding, Result, RunId};
use serde_json::json;

use crate::config::RaftConfig;
use crate::consolidate::interleave_replay;
use crate::eval::eval_pass_rate;
use crate::model::{CausalLm, CorpusTask, SftExample};

fn verify_request(
    run_id: &RunId,
    idx: usize,
    task: &CorpusTask,
    completion: &str,
) -> VerifyRequest {
    VerifyRequest {
        run_id: run_id.clone(),
        step_idx: idx as u32,
        dimension: "exec".into(),
        artifact: json!({
            "task": task.id,
            "completion": completion,
            "marker": completion,
            "verify": task.verify,
        }),
    }
}

/// Verify each supplied correction, fine-tune the adapter on the ones that hold
/// for `cfg.rounds` epochs, then measure whether the expert now reproduces them.
/// `final_fitness` is that internalized pass-rate: the honest "did it stick".
///
/// `replay` is a rehearsal buffer of already-consolidated `prompt -> behavior`
/// pairs (EXP-021): when `cfg.replay_ratio > 0` it is interleaved into every SFT
/// round so consolidating new memories does not clobber old skills. Plain
/// capture passes an empty buffer (replay off).
pub async fn capture_corrections(
    model: &mut (dyn CausalLm + Send),
    verifier: &dyn Verifier,
    tasks: &[CorpusTask],
    run_id: &RunId,
    cfg: &RaftConfig,
    replay: &[SftExample],
) -> Result<TrainOutcome> {
    // A correction we cannot check is not trusted into the population.
    let mut winners: Vec<SftExample> = Vec::new();
    let mut solved: Vec<String> = Vec::new();
    // A correction that also carries a contrastive scope becomes an actionable
    // boundary -- but only once verified, so an unchecked "rule" never gates
    // routing (the same ground-truth gate the population itself sits behind).
    let mut findings: Vec<BoundaryFinding> = Vec::new();
    for (i, task) in tasks.iter().enumerate() {
        let Some(correction) = task.completion.as_deref() else {
            continue;
        };
        let req = verify_request(run_id, i, task, correction);
        if verifier.verify(&req).await?.passed {
            winners.push(SftExample {
                prompt: task.prompt.clone(),
                completion: correction.to_string(),
            });
            if !solved.contains(&task.prompt) {
                solved.push(task.prompt.clone());
            }
            if let Some(scope) = &task.scope {
                // The governing feature is supplied, or inferred from the one key
                // that differs between C and C'. If it cannot be named (zero or
                // several keys differ), no boundary is emitted -- the same
                // open-negative discipline as a search that finds no single scope.
                let feature = scope.governing_feature.clone().or_else(|| {
                    governing_feature_from_pair(&scope.fail_context, &scope.near_ok_context)
                });
                if let Some(governing_feature) = feature {
                    findings.push(BoundaryFinding {
                        behavior: task.prompt.clone(),
                        governing_feature,
                        fail_context: scope.fail_context.clone(),
                        near_ok_context: scope.near_ok_context.clone(),
                    });
                }
            }
        }
    }

    // Nothing verified means nothing to capture: the batch will be empty (replay
    // is scaled by the winner count), so training is a no-op and the saved adapter
    // stays at its untrained init -- a silent failure that looks like graduation
    // succeeded. Surface it, and name the usual Windows cause (the `python`
    // verifier resolving to the Store stub; set ANTUMBRA_PYTHON).
    let with_corrections = tasks.iter().filter(|t| t.completion.is_some()).count();
    if winners.is_empty() && with_corrections > 0 {
        eprintln!(
            "warning: 0 of {with_corrections} correction(s) verified for {} -- nothing \
             captured, the adapter will be UNTRAINED. Is the verifier runnable? (on Windows, \
             set ANTUMBRA_PYTHON to a real interpreter)",
            run_id.as_str()
        );
    }

    // Interleave the rehearsal buffer once; reused each round. With replay off
    // (empty buffer or ratio 0) this is exactly the winners.
    let batch = interleave_replay(&winners, replay, cfg.replay_ratio);
    for _ in 0..cfg.rounds.max(1) {
        if !batch.is_empty() {
            model.sft_step(&batch).await?;
        }
    }

    // Honest fitness: does the expert now generate the correction on its own?
    let learned = eval_pass_rate(model, verifier, tasks, run_id, cfg.samples_per_task).await?;

    let safe = run_id.as_str().replace([':', '/', '\\'], "_");
    let adapter_uri = format!("{}/{safe}.safetensors", cfg.adapter_dir);
    model.save_adapter(&adapter_uri)?;

    Ok(TrainOutcome {
        adapter_uri,
        reward_curve: vec![learned.pass_rate],
        final_fitness: learned.pass_rate,
        capability_exemplars: solved,
        boundary_findings: findings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::CausalLm;
    use antumbra_core::testing::MarkerVerifier;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// Emits the base prior until it is taught, then emits the correction,
    /// so the post-capture eval reflects the internalized behavior.
    struct Learner {
        taught: AtomicBool,
    }

    #[async_trait]
    impl CausalLm for Learner {
        async fn generate(&mut self, _prompt: &str, n: usize) -> Result<Vec<String>> {
            let out = if self.taught.load(Ordering::SeqCst) {
                "deno install"
            } else {
                "npm install"
            };
            Ok(vec![out.to_string(); n])
        }
        async fn sft_step(&mut self, batch: &[SftExample]) -> Result<f32> {
            // Learns only from the supplied (verified) correction.
            if batch.iter().any(|e| e.completion.contains("deno")) {
                self.taught.store(true, Ordering::SeqCst);
            }
            Ok(0.0)
        }
        fn save_adapter(&self, _path: &str) -> Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn captures_and_internalizes_a_verified_correction() {
        let mut lm = Learner {
            taught: AtomicBool::new(false),
        };
        let verifier = MarkerVerifier {
            expect: "deno install".into(),
        };
        let tasks = vec![CorpusTask::new("p", "add a dep").with_completion("deno install")];
        let cfg = RaftConfig {
            rounds: 2,
            samples_per_task: 4,
            ..RaftConfig::default()
        };
        let out = capture_corrections(&mut lm, &verifier, &tasks, &RunId::new("cap:g0"), &cfg, &[])
            .await
            .unwrap();
        // The correction verified, was internalized, and the expert now emits it.
        assert_eq!(out.final_fitness, 1.0);
        assert_eq!(out.capability_exemplars, vec!["add a dep"]);
    }

    #[tokio::test]
    async fn an_unverifiable_correction_is_not_trusted() {
        let mut lm = Learner {
            taught: AtomicBool::new(false),
        };
        // The verifier wants "deno" but the supplied correction is npm -> rejected.
        let verifier = MarkerVerifier {
            expect: "deno install".into(),
        };
        let tasks = vec![CorpusTask::new("p", "add a dep").with_completion("npm install")];
        let cfg = RaftConfig {
            rounds: 2,
            samples_per_task: 4,
            ..RaftConfig::default()
        };
        let out = capture_corrections(&mut lm, &verifier, &tasks, &RunId::new("cap:g1"), &cfg, &[])
            .await
            .unwrap();
        assert_eq!(out.final_fitness, 0.0);
        assert!(out.capability_exemplars.is_empty());
    }

    #[tokio::test]
    async fn a_verified_correction_with_a_scope_yields_an_actionable_finding() {
        let mut lm = Learner {
            taught: AtomicBool::new(false),
        };
        let verifier = MarkerVerifier {
            expect: "deno install".into(),
        };
        // The correction verifies AND names where it applies: npm-style installs
        // are wrong in a deno repo (C), fine in a node repo (C').
        let tasks = vec![CorpusTask::new("p", "add a dep")
            .with_completion("deno install")
            .with_scope(
                "runtime",
                serde_json::json!({ "runtime": "deno" }),
                serde_json::json!({ "runtime": "node" }),
            )];
        let cfg = RaftConfig {
            rounds: 1,
            samples_per_task: 2,
            ..RaftConfig::default()
        };
        let out = capture_corrections(&mut lm, &verifier, &tasks, &RunId::new("cap:b"), &cfg, &[])
            .await
            .unwrap();
        assert_eq!(out.boundary_findings.len(), 1);
        let f = &out.boundary_findings[0];
        assert_eq!(f.behavior, "add a dep");
        assert_eq!(f.governing_feature, "runtime");
        assert_eq!(f.fail_context["runtime"], serde_json::json!("deno"));
        assert_eq!(f.near_ok_context["runtime"], serde_json::json!("node"));
    }

    #[tokio::test]
    async fn an_inferred_scope_names_the_one_differing_feature() {
        let mut lm = Learner {
            taught: AtomicBool::new(false),
        };
        let verifier = MarkerVerifier {
            expect: "deno install".into(),
        };
        // No governing feature supplied: it is inferred from the one key (runtime)
        // that differs between C and C'.
        let tasks = vec![CorpusTask::new("p", "add a dep")
            .with_completion("deno install")
            .with_inferred_scope(
                serde_json::json!({ "runtime": "deno", "task": "install" }),
                serde_json::json!({ "runtime": "node", "task": "install" }),
            )];
        let cfg = RaftConfig {
            rounds: 1,
            samples_per_task: 2,
            ..RaftConfig::default()
        };
        let out = capture_corrections(&mut lm, &verifier, &tasks, &RunId::new("cap:i"), &cfg, &[])
            .await
            .unwrap();
        assert_eq!(out.boundary_findings.len(), 1);
        assert_eq!(out.boundary_findings[0].governing_feature, "runtime");
    }

    #[tokio::test]
    async fn an_ambiguous_inferred_scope_yields_no_boundary() {
        let mut lm = Learner {
            taught: AtomicBool::new(false),
        };
        let verifier = MarkerVerifier {
            expect: "deno install".into(),
        };
        // Two keys differ -> no single feature names the scope -> no boundary,
        // even though the correction itself verifies.
        let tasks = vec![CorpusTask::new("p", "add a dep")
            .with_completion("deno install")
            .with_inferred_scope(
                serde_json::json!({ "runtime": "deno", "task": "install" }),
                serde_json::json!({ "runtime": "node", "task": "build" }),
            )];
        let cfg = RaftConfig {
            rounds: 1,
            samples_per_task: 2,
            ..RaftConfig::default()
        };
        let out = capture_corrections(&mut lm, &verifier, &tasks, &RunId::new("cap:i2"), &cfg, &[])
            .await
            .unwrap();
        assert!(out.boundary_findings.is_empty());
    }

    #[tokio::test]
    async fn an_unverified_correction_yields_no_boundary() {
        let mut lm = Learner {
            taught: AtomicBool::new(false),
        };
        // The verifier wants deno; the supplied correction is npm -> rejected, so
        // even with a scope it is not trusted into a boundary (ground-truth gate).
        let verifier = MarkerVerifier {
            expect: "deno install".into(),
        };
        let tasks = vec![CorpusTask::new("p", "add a dep")
            .with_completion("npm install")
            .with_scope(
                "runtime",
                serde_json::json!({ "runtime": "deno" }),
                serde_json::json!({ "runtime": "node" }),
            )];
        let cfg = RaftConfig {
            rounds: 1,
            samples_per_task: 2,
            ..RaftConfig::default()
        };
        let out = capture_corrections(&mut lm, &verifier, &tasks, &RunId::new("cap:b2"), &cfg, &[])
            .await
            .unwrap();
        assert!(out.boundary_findings.is_empty());
    }

    /// Records every prompt it is fine-tuned on, so a test can assert the
    /// rehearsal buffer actually reached the SFT batch.
    struct Recorder {
        seen: std::sync::Mutex<Vec<String>>,
    }

    #[async_trait]
    impl CausalLm for Recorder {
        async fn generate(&mut self, _prompt: &str, n: usize) -> Result<Vec<String>> {
            Ok(vec!["deno install".to_string(); n])
        }
        async fn sft_step(&mut self, batch: &[SftExample]) -> Result<f32> {
            let mut seen = self.seen.lock().unwrap();
            seen.extend(batch.iter().map(|e| e.prompt.clone()));
            Ok(0.0)
        }
        fn save_adapter(&self, _path: &str) -> Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn replay_buffer_is_rehearsed_during_consolidation() {
        let mut lm = Recorder {
            seen: std::sync::Mutex::new(Vec::new()),
        };
        let verifier = MarkerVerifier {
            expect: "deno install".into(),
        };
        let tasks = vec![CorpusTask::new("p", "new project").with_completion("deno install")];
        // A previously-consolidated skill to rehearse so it is not clobbered.
        let replay = vec![SftExample {
            prompt: "old skill".into(),
            completion: "reverse a string".into(),
        }];
        let cfg = RaftConfig {
            rounds: 1,
            samples_per_task: 2,
            replay_ratio: 1.0,
            ..RaftConfig::default()
        };
        capture_corrections(
            &mut lm,
            &verifier,
            &tasks,
            &RunId::new("cons:g0"),
            &cfg,
            &replay,
        )
        .await
        .unwrap();
        let seen = lm.seen.lock().unwrap();
        // Both the new memory and the rehearsed old skill were trained on.
        assert!(seen.iter().any(|p| p == "new project"));
        assert!(seen.iter().any(|p| p == "old skill"));
    }
}
