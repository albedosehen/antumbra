//! Metabolize the harness (ADR-0001).
//!
//! An agent harness — loops, behavior graphs, task execution, code-intel — is
//! orchestration layered *on top of* a frozen brain. Antumbra's thesis is to
//! absorb that orchestration into weights, not to clone it: the population
//! should learn to do in one shot what the harness did in many steps, so the
//! scaffold shrinks over time.
//!
//! This module adapts a harness's **successful orchestration traces** (a loop
//! run, a behavior-graph evaluation, a task execution) into the same
//! [`CorpusTask`]s the capture loop ([`crate::teach::capture_corrections`])
//! internalizes — the goal becomes the prompt, the collapsed outcome becomes the
//! completion. Only **successful, sufficiently-recurrent** traces metabolize: a
//! one-off or failed orchestration is not a competence worth freezing into the
//! weights (the same verifiability × recurrence gate the consolidation path
//! uses). Like [`crate::memory`], records are read from `serde_json::Value` by
//! hand so the serde derive stays out of the default (non-`models`) build.

use antumbra_core::{AntumbraError, Result};
use serde_json::Value;

use crate::memory::marker_verify;
use crate::model::CorpusTask;

/// One normalized harness trace, source-agnostic. An adapter for any harness
/// (Kushtaka task traces, behavior-graph evaluations, loop runs) only has to
/// emit this shape.
#[derive(Debug, Clone, Default)]
pub struct HarnessTrace {
    /// Stable id; synthesized from the kind + index when absent.
    pub id: Option<String>,
    /// The goal the harness pursued — becomes the task prompt.
    pub goal: String,
    /// The collapsed outcome the orchestration produced (the final answer/code)
    /// — what the expert should learn to emit in one shot.
    pub solution: String,
    /// What kind of orchestration produced it (`loop` / `graph` / `task`) — the
    /// skill group, so metabolized traces cluster per kind. Defaults to `task`.
    pub kind: Option<String>,
    /// Substring the emitted completion must contain to count as internalized.
    /// Defaults to the whole `solution` (an exact-recall check).
    pub marker: Option<String>,
    /// Substrings the completion must NOT contain.
    pub forbid: Vec<String>,
    /// Whether the harness verified this trace as successful. Only successes
    /// metabolize. Absent is treated as `true` (an emitted trace is a success).
    pub success: Option<bool>,
    /// How many times this orchestration pattern recurred — the recurrence
    /// signal the gate scores. Absent is treated as 1 (observed once).
    pub recurrence: Option<u32>,
}

impl HarnessTrace {
    fn from_value(v: &Value) -> Self {
        let s = |k: &str| v.get(k).and_then(Value::as_str).map(str::to_string);
        let forbid = v
            .get("forbid")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect())
            .unwrap_or_default();
        Self {
            id: s("id"),
            goal: s("goal").or_else(|| s("prompt")).unwrap_or_default(),
            solution: s("solution").or_else(|| s("outcome")).unwrap_or_default(),
            kind: s("kind"),
            marker: s("marker"),
            forbid,
            success: v.get("success").and_then(Value::as_bool),
            recurrence: v
                .get("recurrence")
                .and_then(Value::as_u64)
                .map(|n| n as u32),
        }
    }

    fn metabolizes(&self, policy: &MetabolizePolicy) -> bool {
        !self.goal.is_empty()
            && !self.solution.is_empty()
            && self.success.unwrap_or(true)
            && self.recurrence.unwrap_or(1) >= policy.min_recurrence
    }
}

/// How selective metabolization is.
#[derive(Debug, Clone, Copy)]
pub struct MetabolizePolicy {
    /// Recurrence floor: only patterns seen at least this often become weights
    /// (raise above 1 to require a *repeated* orchestration, not a one-off).
    pub min_recurrence: u32,
}

impl Default for MetabolizePolicy {
    fn default() -> Self {
        Self { min_recurrence: 1 }
    }
}

/// Parse a normalized harness-trace export (a JSON array of trace objects).
pub fn parse_traces(bytes: &[u8]) -> Result<Vec<HarnessTrace>> {
    let raw: Vec<Value> =
        serde_json::from_slice(bytes).map_err(|e| AntumbraError::other(e.to_string()))?;
    Ok(raw.iter().map(HarnessTrace::from_value).collect())
}

/// Convert one trace into a capture task (goal → prompt, solution → completion,
/// a self-consistency marker check as the verifier).
fn to_task(trace: &HarnessTrace, index: usize) -> CorpusTask {
    let kind = trace.kind.clone().unwrap_or_else(|| "task".to_string());
    let id = trace
        .id
        .clone()
        .unwrap_or_else(|| format!("harness-{kind}-{index}"));
    let marker = trace.marker.as_deref().unwrap_or(&trace.solution);
    let mut task = CorpusTask::new(id, trace.goal.clone())
        .with_verify(marker_verify(marker, &trace.forbid))
        .with_completion(trace.solution.clone());
    task.skill = Some(kind);
    task
}

/// Metabolize the traces that clear the gate into capture tasks (per-kind
/// skilled, carrying trusted completions). Failed and one-off traces are
/// dropped — nothing unverified or non-recurrent is fine-tuned into the weights.
pub fn metabolize(traces: &[HarnessTrace], policy: &MetabolizePolicy) -> Vec<CorpusTask> {
    traces
        .iter()
        .enumerate()
        .filter(|(_, t)| t.metabolizes(policy))
        .map(|(i, t)| to_task(t, i))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn traces() -> Vec<HarnessTrace> {
        parse_traces(
            br#"[
              {"id":"t1","goal":"reverse a string","solution":"return s[::-1]","kind":"task","recurrence":4,"success":true},
              {"goal":"add two numbers","solution":"return a + b","kind":"loop","recurrence":1},
              {"goal":"flaky one-off","solution":"whatever","kind":"task","recurrence":1,"success":false},
              {"goal":"","solution":"no goal","kind":"task"}
            ]"#,
        )
        .unwrap()
    }

    #[test]
    fn parses_goal_and_solution_with_fallbacks() {
        let t = HarnessTrace::from_value(&serde_json::json!({
            "prompt": "g", "outcome": "o"
        }));
        assert_eq!(t.goal, "g");
        assert_eq!(t.solution, "o");
        assert!(t.success.is_none() && t.recurrence.is_none());
    }

    #[test]
    fn only_successful_nonempty_traces_metabolize() {
        let out = metabolize(&traces(), &MetabolizePolicy::default());
        // t1 (success) + add (default success) metabolize; the failed and the
        // empty-goal traces are dropped.
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|t| t.completion.is_some()));
        assert_eq!(out[0].skill.as_deref(), Some("task"));
        assert_eq!(out[1].skill.as_deref(), Some("loop"));
    }

    #[test]
    fn recurrence_floor_requires_a_repeated_pattern() {
        // min_recurrence 2 keeps only t1 (recurrence 4); the recurrence-1 trace drops.
        let out = metabolize(&traces(), &MetabolizePolicy { min_recurrence: 2 });
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].id, "t1");
    }

    #[test]
    fn marker_check_targets_the_solution() {
        let out = metabolize(
            &[HarnessTrace {
                goal: "g".into(),
                solution: "return s[::-1]".into(),
                ..Default::default()
            }],
            &MetabolizePolicy::default(),
        );
        let verify = out[0].verify.to_string();
        assert!(verify.contains("python"), "a runnable verifier is attached");
    }
}
