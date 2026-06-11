//! Metabolize the harness into the population of small frozen experts.
//!
//! An agent harness (loops, behavior graphs, task execution, code-intel) is
//! orchestration layered *on top of* a frozen brain. Antumbra's thesis is to
//! absorb that orchestration into weights, not to clone it: the population
//! should learn to do in one shot what the harness did in many steps, so the
//! scaffold shrinks over time.
//!
//! This module adapts a harness's **successful orchestration traces** (a loop
//! run, a behavior-graph evaluation, a task execution) into the same
//! [`CorpusTask`]s the capture loop ([`crate::teach::capture_corrections`])
//! internalizes: the goal becomes the prompt, the collapsed outcome becomes the
//! completion. Only **successful, sufficiently-recurrent** traces metabolize: a
//! one-off or failed orchestration is not a competence worth freezing into the
//! weights (the same verifiability × recurrence gate the consolidation path
//! uses). Like [`crate::memory`], records are read from `serde_json::Value` by
//! hand so the serde derive stays out of the default (non-`models`) build.
//!
//! **Structure-aware metabolization.** A behavior graph (or a loop run) is not
//! just its collapsed answer; it is a *decomposition* into sub-goals. When a
//! trace carries its steps, each step is metabolized as its own capture task
//! alongside the whole, so the expert learns the intermediate sub-skills, not
//! only the final output. This turns the trace's *outcome* supervision into
//! *process* supervision, which is the more informative signal for multi-step
//! competence (Structured Agent Distillation, arXiv:2505.13820; the
//! success × recurrence gate is the selectivity that hindsight-distillation work
//! finds necessary, arXiv:2605.19447).

use antumbra_core::{AntumbraError, Result};
use serde_json::Value;

use crate::memory::marker_verify;
use crate::model::CorpusTask;

/// The first present string field among `keys` (the tolerant field-extractor that
/// lets one adapter read every harness tool's payload without a per-tool schema).
fn first_str(v: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|k| v.get(*k).and_then(Value::as_str).map(str::to_string))
}

/// A string array field (e.g. `forbid`), or empty when absent/mistyped.
fn str_array(v: &Value, key: &str) -> Vec<String> {
    v.get(key)
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// One step of a structured orchestration: a behavior-graph node, a loop
/// iteration, or a sub-task: its own goal and the outcome it produced.
/// Metabolizing these alongside the collapsed whole turns the trace's *outcome*
/// supervision into *process* supervision, so the expert learns the decomposition.
#[derive(Debug, Clone, Default)]
pub struct HarnessStep {
    /// A stable id edges reference. Absent, a step is addressed by `step{index}`.
    pub id: Option<String>,
    pub goal: String,
    pub solution: String,
    pub marker: Option<String>,
    pub forbid: Vec<String>,
}

impl HarnessStep {
    fn from_value(v: &Value) -> Self {
        Self {
            id: first_str(v, &["id", "node_id", "step_id"]),
            goal: first_str(v, &["goal", "prompt", "name", "label", "description"])
                .unwrap_or_default(),
            solution: first_str(
                v,
                &["solution", "outcome", "result", "output", "final_output"],
            )
            .unwrap_or_default(),
            marker: first_str(v, &["marker"]),
            forbid: str_array(v, "forbid"),
        }
    }

    /// A step is metabolizable only when it carries both a sub-goal and an outcome
    /// (an empty node, a bare branch/marker, is scaffolding, not a sub-skill).
    fn is_metabolizable(&self) -> bool {
        !self.goal.is_empty() && !self.solution.is_empty()
    }
}

/// A directed edge of a behavior graph: orchestration flows from one step to the
/// next, optionally under a branch `condition`. Metabolizing edges teaches the
/// *ordering / branching* (produce `to` after `from`), the dataflow the isolated
/// per-step captures do not carry. `from`/`to` reference a step's id (or its
/// synthesized `step{index}`).
#[derive(Debug, Clone, Default)]
pub struct HarnessEdge {
    pub from: String,
    pub to: String,
    pub condition: Option<String>,
}

impl HarnessEdge {
    fn from_value(v: &Value) -> Option<Self> {
        Some(Self {
            from: first_str(v, &["from", "source", "src"])?,
            to: first_str(v, &["to", "target", "dst"])?,
            condition: first_str(v, &["condition", "cond", "when"]),
        })
    }
}

/// One normalized harness trace, source-agnostic. An adapter for any harness
/// (task traces, behavior-graph evaluations, loop runs) only has to emit this
/// shape.
#[derive(Debug, Clone, Default)]
pub struct HarnessTrace {
    /// Stable id; synthesized from the kind + index when absent.
    pub id: Option<String>,
    /// The goal the harness pursued; becomes the task prompt.
    pub goal: String,
    /// The collapsed outcome the orchestration produced (the final answer/code):
    /// what the expert should learn to emit in one shot.
    pub solution: String,
    /// What kind of orchestration produced it (`loop` / `graph` / `task`): the
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
    /// How many times this orchestration pattern recurred: the recurrence
    /// signal the gate scores. Absent is treated as 1 (observed once).
    pub recurrence: Option<u32>,
    /// The orchestration's decomposition (graph nodes / loop iterations / sub-
    /// tasks). Each becomes its own capture task under [`MetabolizePolicy`]'s
    /// `include_steps`, so the expert learns the process, not only the outcome.
    pub steps: Vec<HarnessStep>,
    /// The graph's directed edges (ordering / branching between steps). Each
    /// becomes a transition capture task under `include_steps`, so the expert
    /// learns the dataflow, not only the isolated sub-skills.
    pub edges: Vec<HarnessEdge>,
}

impl HarnessTrace {
    fn from_value(v: &Value) -> Self {
        let steps = ["steps", "nodes", "iterations", "subtasks"]
            .iter()
            .find_map(|k| v.get(*k).and_then(Value::as_array))
            .map(|a| {
                a.iter()
                    .map(HarnessStep::from_value)
                    .filter(HarnessStep::is_metabolizable)
                    .collect()
            })
            .unwrap_or_default();
        Self {
            id: first_str(v, &["id", "trace_id", "task_id", "run_id"]),
            goal: first_str(v, &["goal", "prompt", "task", "objective"]).unwrap_or_default(),
            solution: first_str(
                v,
                &["solution", "outcome", "result", "final_output", "answer"],
            )
            .unwrap_or_default(),
            kind: first_str(v, &["kind", "type", "source"]),
            marker: first_str(v, &["marker"]),
            forbid: str_array(v, "forbid"),
            success: v.get("success").and_then(Value::as_bool).or_else(|| {
                match first_str(v, &["status", "state"]).as_deref() {
                    Some("success" | "succeeded" | "completed" | "passed" | "ok") => Some(true),
                    Some("failed" | "failure" | "error" | "aborted") => Some(false),
                    _ => None,
                }
            }),
            recurrence: v
                .get("recurrence")
                .or_else(|| v.get("count"))
                .and_then(Value::as_u64)
                .map(|n| n as u32),
            steps,
            edges: v
                .get("edges")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(HarnessEdge::from_value).collect())
                .unwrap_or_default(),
        }
    }

    fn metabolizes(&self, policy: &MetabolizePolicy) -> bool {
        !self.goal.is_empty()
            && !self.solution.is_empty()
            && self.success.unwrap_or(true)
            && self.recurrence.unwrap_or(1) >= policy.min_recurrence
    }

    /// The trace's skill kind and base id (synthesizing a stable id when absent).
    fn identity(&self, index: usize) -> (String, String) {
        let kind = self.kind.clone().unwrap_or_else(|| "task".to_string());
        let id = self
            .id
            .clone()
            .unwrap_or_else(|| format!("harness-{kind}-{index}"));
        (kind, id)
    }
}

/// How selective metabolization is.
#[derive(Debug, Clone, Copy)]
pub struct MetabolizePolicy {
    /// Recurrence floor: only patterns seen at least this often become weights
    /// (raise above 1 to require a *repeated* orchestration, not a one-off).
    pub min_recurrence: u32,
    /// Also metabolize each trace's steps (its decomposition) as their own capture
    /// tasks (process supervision, not just the collapsed outcome). On by default;
    /// disable to learn only the one-shot collapse.
    pub include_steps: bool,
}

impl Default for MetabolizePolicy {
    fn default() -> Self {
        Self {
            min_recurrence: 1,
            include_steps: true,
        }
    }
}

/// Parse a normalized harness-trace export. Accepts a JSON array of trace objects
/// or an object wrapping the array (see [`normalize_traces`]).
pub fn parse_traces(bytes: &[u8]) -> Result<Vec<HarnessTrace>> {
    let v: Value =
        serde_json::from_slice(bytes).map_err(|e| AntumbraError::other(e.to_string()))?;
    Ok(normalize_traces(&v))
}

/// Normalize any harness export into traces. Tolerant of shape: a top-level
/// array, an object carrying the array under a common key (`tasks`, `traces`,
/// `evaluations`, `runs`, `results`, `data`, `memories`, `items`), or a single
/// trace object. Field names are matched with fallbacks (goal|prompt|task,
/// solution|outcome|result|…, steps|nodes|iterations), so one adapter covers a
/// task list / task-trace / behavior-graph-evaluations / loop-run payload
/// without a per-tool schema. Non-object/array values yield none.
pub fn normalize_traces(value: &Value) -> Vec<HarnessTrace> {
    let items: Vec<&Value> = match value {
        Value::Array(a) => a.iter().collect(),
        Value::Object(_) => [
            "tasks",
            "traces",
            "evaluations",
            "runs",
            "results",
            "data",
            "memories",
            "items",
        ]
        .iter()
        .find_map(|k| value.get(*k).and_then(Value::as_array))
        .map(|a| a.iter().collect())
        .unwrap_or_else(|| vec![value]),
        _ => vec![],
    };
    items.into_iter().map(HarnessTrace::from_value).collect()
}

/// Build the capture task for a trace's collapsed whole (do the orchestration in
/// one shot): goal → prompt, solution → completion, a marker check as the verifier.
fn whole_task(trace: &HarnessTrace, kind: &str, id: &str) -> CorpusTask {
    let marker = trace.marker.as_deref().unwrap_or(&trace.solution);
    let mut task = CorpusTask::new(id.to_string(), trace.goal.clone())
        .with_verify(marker_verify(marker, &trace.forbid))
        .with_completion(trace.solution.clone());
    task.skill = Some(kind.to_string());
    task
}

/// Build the capture task for one step of the decomposition. Shares the trace's
/// skill (a step is a facet of the same orchestration competence) with a distinct
/// `{id}#step{n}` id, so the expert learns the sub-skill under the same expert.
fn step_task(step: &HarnessStep, kind: &str, base_id: &str, n: usize) -> CorpusTask {
    let marker = step.marker.as_deref().unwrap_or(&step.solution);
    let mut task = CorpusTask::new(format!("{base_id}#step{n}"), step.goal.clone())
        .with_verify(marker_verify(marker, &step.forbid))
        .with_completion(step.solution.clone());
    task.skill = Some(kind.to_string());
    task
}

/// Build the capture task for one edge: teach what follows what (the ordering /
/// branch). The prompt frames the `to` step as the continuation of `from` (under
/// the branch `condition`, when present); the completion is `to`'s outcome, so
/// the expert learns the dataflow, not just the sub-skills in isolation.
fn edge_task(
    from: &HarnessStep,
    to: &HarnessStep,
    edge: &HarnessEdge,
    kind: &str,
    base_id: &str,
) -> CorpusTask {
    let marker = to.marker.as_deref().unwrap_or(&to.solution);
    let cond = edge
        .condition
        .as_deref()
        .map(|c| format!(" (when {c})"))
        .unwrap_or_default();
    let prompt = format!(
        "After: {}{cond}\nwhich produced:\n{}\n\nNext: {}",
        from.goal, from.solution, to.goal
    );
    let mut task = CorpusTask::new(format!("{base_id}#edge:{}->{}", edge.from, edge.to), prompt)
        .with_verify(marker_verify(marker, &to.forbid))
        .with_completion(to.solution.clone());
    task.skill = Some(kind.to_string());
    task
}

/// Metabolize the traces that clear the gate into capture tasks (per-kind
/// skilled, carrying trusted completions). Failed and one-off traces are
/// dropped; nothing unverified or non-recurrent is fine-tuned into the weights.
/// For a trace that carries its decomposition, the whole is metabolized *plus*
/// each step (process supervision), unless [`MetabolizePolicy::include_steps`] is
/// off.
pub fn metabolize(traces: &[HarnessTrace], policy: &MetabolizePolicy) -> Vec<CorpusTask> {
    let mut out = Vec::new();
    for (i, trace) in traces.iter().enumerate() {
        if !trace.metabolizes(policy) {
            continue;
        }
        let (kind, id) = trace.identity(i);
        out.push(whole_task(trace, &kind, &id));
        if policy.include_steps {
            for (n, step) in trace.steps.iter().enumerate() {
                out.push(step_task(step, &kind, &id, n));
            }
            // Transition tasks for the graph's edges (ordering / branching),
            // resolving each edge's endpoints to steps (by id or `step{index}`);
            // an edge naming a dropped/missing step is skipped.
            if !trace.edges.is_empty() {
                let by_id: std::collections::HashMap<String, &HarnessStep> = trace
                    .steps
                    .iter()
                    .enumerate()
                    .map(|(n, s)| (s.id.clone().unwrap_or_else(|| format!("step{n}")), s))
                    .collect();
                for edge in &trace.edges {
                    if let (Some(from), Some(to)) = (by_id.get(&edge.from), by_id.get(&edge.to)) {
                        out.push(edge_task(from, to, edge, &kind, &id));
                    }
                }
            }
        }
    }
    out
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
        assert!(t.steps.is_empty());
    }

    #[test]
    fn only_successful_nonempty_traces_metabolize() {
        let out = metabolize(&traces(), &MetabolizePolicy::default());
        // t1 (success) + add (default success) metabolize; the failed and the
        // empty-goal traces are dropped. Neither carries steps, so two tasks.
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|t| t.completion.is_some()));
        assert_eq!(out[0].skill.as_deref(), Some("task"));
        assert_eq!(out[1].skill.as_deref(), Some("loop"));
    }

    #[test]
    fn recurrence_floor_requires_a_repeated_pattern() {
        // min_recurrence 2 keeps only t1 (recurrence 4); the recurrence-1 trace drops.
        let out = metabolize(
            &traces(),
            &MetabolizePolicy {
                min_recurrence: 2,
                include_steps: true,
            },
        );
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

    // Structure-aware: a behavior graph carries its decomposition; the whole AND
    // each step metabolize, so the expert learns the process, not just the outcome.
    #[test]
    fn a_graph_metabolizes_its_decomposition_as_steps() {
        let trace = HarnessTrace::from_value(&serde_json::json!({
            "id": "g1",
            "goal": "fetch, parse, and summarize a page",
            "solution": "summary text",
            "kind": "graph",
            "success": true,
            "nodes": [
                {"goal": "fetch the page", "outcome": "html bytes"},
                {"goal": "parse the html", "result": "dom tree"},
                {"name": "bare branch with no outcome"}
            ]
        }));
        assert_eq!(
            trace.steps.len(),
            2,
            "the empty (outcome-less) node is dropped"
        );

        let out = metabolize(&[trace], &MetabolizePolicy::default());
        // whole + 2 steps.
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].id, "g1");
        assert_eq!(out[1].id, "g1#step0");
        assert_eq!(out[2].id, "g1#step1");
        // steps are facets of the same skill, carry their own completion.
        assert!(out.iter().all(|t| t.skill.as_deref() == Some("graph")));
        assert_eq!(out[1].completion.as_deref(), Some("html bytes"));
        assert_eq!(out[2].prompt, "parse the html");
    }

    // Turning steps off learns only the one-shot collapse (outcome supervision).
    #[test]
    fn include_steps_off_drops_the_decomposition() {
        let trace = HarnessTrace::from_value(&serde_json::json!({
            "id": "g1", "goal": "g", "solution": "s", "kind": "graph",
            "steps": [{"goal": "sub", "solution": "out"}]
        }));
        let out = metabolize(
            &[trace],
            &MetabolizePolicy {
                min_recurrence: 1,
                include_steps: false,
            },
        );
        assert_eq!(out.len(), 1, "only the collapsed whole");
        assert_eq!(out[0].id, "g1");
    }

    // A failed trace contributes nothing; neither its whole nor its steps are
    // frozen (you don't internalize the decomposition of a broken orchestration).
    #[test]
    fn a_failed_trace_metabolizes_neither_whole_nor_steps() {
        let trace = HarnessTrace::from_value(&serde_json::json!({
            "goal": "g", "solution": "s", "kind": "graph", "success": false,
            "steps": [{"goal": "sub", "solution": "out"}]
        }));
        assert!(metabolize(&[trace], &MetabolizePolicy::default()).is_empty());
    }

    #[test]
    fn edges_metabolize_into_transition_tasks() {
        let trace = HarnessTrace::from_value(&serde_json::json!({
            "goal": "fetch then summarize", "solution": "pipeline", "kind": "graph",
            "success": true, "recurrence": 5,
            "nodes": [
                {"goal": "fetch html", "solution": "requests.get(url).text", "marker": "requests.get"},
                {"goal": "summarize text", "solution": "text[:280]", "marker": "text["}
            ],
            "edges": [{"from": "step0", "to": "step1", "condition": "fetch ok"}]
        }));
        let tasks = metabolize(&[trace], &MetabolizePolicy::default());
        assert_eq!(tasks.len(), 4, "whole + 2 steps + 1 edge transition");
        let edge = tasks
            .iter()
            .find(|t| t.id.contains("#edge:step0->step1"))
            .expect("an edge transition task is metabolized");
        assert!(edge.prompt.contains("Next: summarize text"));
        assert!(edge.prompt.contains("when fetch ok"), "the branch condition rides the prompt");
        assert_eq!(edge.completion.as_deref(), Some("text[:280]"));
        assert_eq!(edge.skill.as_deref(), Some("graph"));
    }

    #[test]
    fn an_edge_to_a_dropped_or_missing_step_is_skipped() {
        let trace = HarnessTrace::from_value(&serde_json::json!({
            "goal": "g", "solution": "s", "kind": "graph", "success": true, "recurrence": 5,
            "nodes": [{"goal": "a", "solution": "out_a"}],
            "edges": [{"from": "step0", "to": "step9"}]
        }));
        let tasks = metabolize(&[trace], &MetabolizePolicy::default());
        assert_eq!(tasks.len(), 2, "whole + 1 step; the edge to a missing step is dropped");
        assert!(!tasks.iter().any(|t| t.id.contains("#edge")));
    }

    // A status string stands in for an explicit success bool (a harness task state).
    #[test]
    fn status_string_maps_to_success() {
        let done = HarnessTrace::from_value(&serde_json::json!({
            "goal": "g", "solution": "s", "status": "completed"
        }));
        let failed = HarnessTrace::from_value(&serde_json::json!({
            "goal": "g", "solution": "s", "status": "failed"
        }));
        assert_eq!(done.success, Some(true));
        assert_eq!(failed.success, Some(false));
        assert_eq!(
            metabolize(&[done, failed], &MetabolizePolicy::default()).len(),
            1
        );
    }

    // The normalizer accepts every harness-tool envelope: a bare array, an object
    // wrapping the list under a tool-specific key, or one trace object.
    #[test]
    fn normalizer_tolerates_each_tool_envelope() {
        // list_tasks-style: { tasks: [...] }
        let tasks = normalize_traces(&serde_json::json!({
            "count": 2,
            "tasks": [
                {"task_id": "k1", "prompt": "do x", "outcome": "did x"},
                {"id": "k2", "goal": "do y", "result": "did y"}
            ]
        }));
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].id.as_deref(), Some("k1"));
        assert_eq!(tasks[0].goal, "do x");

        // behavior-graph evaluations: { runs: [...] } with a count-recurrence.
        let runs = normalize_traces(&serde_json::json!({
            "runs": [{"run_id": "r1", "objective": "g", "final_output": "o", "count": 5}]
        }));
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].recurrence, Some(5));
        assert_eq!(runs[0].id.as_deref(), Some("r1"));

        // a single trace object (get_task_trace) and the empty / wrong-typed cases.
        assert_eq!(
            normalize_traces(&serde_json::json!({"goal": "g", "solution": "s"})).len(),
            1
        );
        assert!(normalize_traces(&serde_json::json!({})).len() == 1); // self as one
        assert!(normalize_traces(&serde_json::json!([])).is_empty());
        assert!(normalize_traces(&serde_json::json!("nope")).is_empty());
    }
}
