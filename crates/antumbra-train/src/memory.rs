//! Bootstrap from an existing memory corpus (the capture intake, ADR-0004/0009).
//!
//! People already hold verified competence in their agents' memory stores —
//! Kushtaka, a qdrant collection, a json file, surrealdb. A memory earned its
//! place by working in production and being reinforced; that reinforcement is
//! the reward signal RLVR would otherwise have to rediscover from a cold start.
//! This module adapts a *normalized* memory export into the same [`CorpusTask`]s
//! the capture loop ([`crate::teach::capture_corrections`]) internalizes, so a
//! population can start from lived experience instead of the high-variance RAFT
//! discovery (the bootstrap problem of EXP-019).
//!
//! Trust is tiered by the memory's own confidence, mirroring the two intake
//! paths:
//!   - a reinforced memory (`confidence >= capture_threshold`) becomes a
//!     **capture** — provenance is its verifier, and the loop still checks the
//!     behavior actually stuck;
//!   - a weak memory becomes a RAFT **seed** — a hypothesis to be confirmed by
//!     experience (it carries no trusted `completion`, only the prompt + check),
//!     so nothing unverified is fine-tuned into the weights.
//!
//! A memory's `scope`/`network` becomes the expert **skill**, so imported
//! memories cluster into per-skill specialists the same way `populate` grows
//! them — the rule/fact duality of a memory store maps onto the standing /
//! contextual expert split.

use antumbra_core::{AntumbraError, Result};
use serde_json::Value;

use crate::model::CorpusTask;

/// One normalized memory, source-agnostic. An adapter for any backing store
/// (Kushtaka, qdrant, surrealdb, a json file) only has to emit this shape. Like
/// [`crate::corpus`], it is read from a `serde_json::Value` by hand to keep the
/// serde derive out of the default (non-`models`) build.
#[derive(Debug, Clone, Default)]
pub struct MemoryRecord {
    /// The behavior or fact to internalize — what the expert should emit.
    pub content: String,
    /// The situation/cue that should elicit `content`. Synthesized from
    /// `content` when absent.
    pub prompt: Option<String>,
    /// Skill group: a domain scope (e.g. `package-manager`). Falls back to
    /// `network`, then `id`.
    pub scope: Option<String>,
    /// The memory network it came from (world/bank/opinion); a coarse skill.
    pub network: Option<String>,
    /// Substring the emitted completion must contain to count as internalized.
    /// Defaults to the whole `content` (an exact-recall check).
    pub marker: Option<String>,
    /// Substrings the completion must NOT contain — typically the base prior
    /// this memory corrects (e.g. forbid `npm` when teaching `deno`).
    pub forbid: Vec<String>,
    /// The memory's own confidence / reinforcement strength in `[0, 1]`. Drives
    /// the capture-vs-seed tier. Absent is treated as fully trusted (1.0): a
    /// store that does not track confidence is taken at its word.
    pub confidence: Option<f32>,
    /// How many times the memory was reinforced / accessed — the *recurrence*
    /// signal the consolidation gate scores (EXP-021). Absent is treated as 0.
    pub reinforcement: Option<u32>,
    /// `true` if the fact changes over time (current branch, today's deploy
    /// state). Volatile memories never graduate into frozen weights — they stay
    /// in the store. Absent is treated as stable.
    pub volatile: Option<bool>,
    /// Explicit verifiability override for the consolidation gate. Absent lets
    /// the gate derive it (an `opinion` is treated as unverifiable unless its
    /// confidence clears the provenance tier; everything else is verifiable).
    pub verifiable: Option<bool>,
    /// Stable id for the task; synthesized from the skill + index when absent.
    pub id: Option<String>,
}

impl MemoryRecord {
    /// Read a normalized memory from a JSON object, by hand (no serde derive).
    fn from_value(v: &Value) -> Self {
        let s = |k: &str| v.get(k).and_then(Value::as_str).map(str::to_string);
        let forbid = v
            .get("forbid")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        Self {
            content: s("content").unwrap_or_default(),
            prompt: s("prompt"),
            scope: s("scope"),
            network: s("network"),
            marker: s("marker"),
            forbid,
            confidence: v.get("confidence").and_then(Value::as_f64).map(|x| x as f32),
            reinforcement: v
                .get("reinforcement")
                .and_then(Value::as_u64)
                .map(|x| x as u32),
            volatile: v.get("volatile").and_then(Value::as_bool),
            verifiable: v.get("verifiable").and_then(Value::as_bool),
            id: s("id"),
        }
    }
}

/// How much to trust an imported memory.
#[derive(Debug, Clone, Copy)]
pub struct ImportPolicy {
    /// Memories at or above this confidence are captured (trusted on import);
    /// below it they are emitted as RAFT seeds to be confirmed by experience.
    pub capture_threshold: f32,
}

impl Default for ImportPolicy {
    fn default() -> Self {
        Self {
            capture_threshold: 0.5,
        }
    }
}

/// Which intake path a record was routed to, for honest reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Intake {
    /// Trusted on import — fine-tuned directly (carries a `completion`).
    Capture,
    /// A hypothesis to be discovered/confirmed by RAFT (no trusted completion).
    Seed,
}

/// A converted task plus the path it took, so the caller can split captures
/// from seeds without re-deriving the tier.
#[derive(Debug, Clone)]
pub struct ImportedTask {
    pub task: CorpusTask,
    pub intake: Intake,
}

/// Parse a normalized memory export (a JSON array of memory objects).
pub fn parse_export(bytes: &[u8]) -> Result<Vec<MemoryRecord>> {
    let raw: Vec<Value> =
        serde_json::from_slice(bytes).map_err(|e| AntumbraError::other(e.to_string()))?;
    Ok(raw.iter().map(MemoryRecord::from_value).collect())
}

/// Build a self-consistency verifier: pass when the completion contains
/// `marker` and none of `forbid`, compared case-insensitively. The marker and
/// forbid list are JSON-encoded into the python source, which is also valid
/// python literal syntax — so arbitrary memory text cannot break or inject the
/// check. This is the same marker check the teach corpora use; it confirms the
/// expert *internalized* the memory, not that the memory is true (the store's
/// reinforcement already settled truth).
fn marker_verify(marker: &str, forbid: &[String]) -> serde_json::Value {
    let m = serde_json::to_string(&marker.to_lowercase()).unwrap_or_else(|_| "\"\"".into());
    let f: Vec<String> = forbid.iter().map(|s| s.to_lowercase()).collect();
    let f = serde_json::to_string(&f).unwrap_or_else(|_| "[]".into());
    let src = format!(
        "import os,sys; c=os.environ.get('ANTUMBRA_COMPLETION','').lower(); \
         sys.exit(0 if ({m} in c and not any(x in c for x in {f})) else 1)"
    );
    serde_json::json!({
        "program": "python",
        "extract_code": false,
        "args": ["-c", src],
    })
}

/// The skill group a record belongs to: its `scope`, then `network`, then `id`.
fn skill_of(record: &MemoryRecord, index: usize) -> String {
    record
        .scope
        .clone()
        .or_else(|| record.network.clone())
        .or_else(|| record.id.clone())
        .unwrap_or_else(|| format!("memory-{index}"))
}

/// Convert one normalized memory into a corpus task under `policy`.
pub fn to_task(record: &MemoryRecord, index: usize, policy: &ImportPolicy) -> ImportedTask {
    let skill = skill_of(record, index);
    let id = record
        .id
        .clone()
        .unwrap_or_else(|| format!("{skill}-{index}"));
    let prompt = record
        .prompt
        .clone()
        .unwrap_or_else(|| format!("# Given what you have learned:\n{}\n", record.content));
    let marker = record.marker.as_deref().unwrap_or(&record.content);
    let verify = marker_verify(marker, &record.forbid);

    let trusted = record.confidence.unwrap_or(1.0) >= policy.capture_threshold;
    let mut task = CorpusTask::new(id, prompt).with_verify(verify);
    task.skill = Some(skill);
    if trusted {
        task = task.with_completion(record.content.clone());
        ImportedTask {
            task,
            intake: Intake::Capture,
        }
    } else {
        // A weak memory is a seed: keep the check, drop the trusted answer so
        // RAFT must rediscover and verify it before anything is internalized.
        ImportedTask {
            task,
            intake: Intake::Seed,
        }
    }
}

/// Convert a whole normalized export into tasks, tiered by `policy`.
pub fn import(records: &[MemoryRecord], policy: &ImportPolicy) -> Vec<ImportedTask> {
    records
        .iter()
        .enumerate()
        .map(|(i, r)| to_task(r, i, policy))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(content: &str, confidence: Option<f32>) -> MemoryRecord {
        MemoryRecord {
            content: content.into(),
            prompt: None,
            scope: Some("package-manager".into()),
            network: Some("opinion".into()),
            marker: Some("deno".into()),
            forbid: vec!["npm".into()],
            confidence,
            id: None,
            ..Default::default()
        }
    }

    #[test]
    fn reinforced_memory_becomes_a_trusted_capture() {
        let imported = to_task(&record("use deno install", Some(0.9)), 0, &ImportPolicy::default());
        assert_eq!(imported.intake, Intake::Capture);
        assert_eq!(imported.task.completion.as_deref(), Some("use deno install"));
        assert_eq!(imported.task.skill(), "package-manager");
    }

    #[test]
    fn weak_memory_becomes_a_seed_with_no_trusted_completion() {
        let imported = to_task(&record("use deno install", Some(0.2)), 0, &ImportPolicy::default());
        assert_eq!(imported.intake, Intake::Seed);
        assert!(imported.task.completion.is_none());
        // The check survives so RAFT can verify a discovered sample.
        assert_eq!(imported.task.verify["program"], "python");
    }

    #[test]
    fn missing_confidence_is_trusted() {
        let imported = to_task(&record("use deno install", None), 0, &ImportPolicy::default());
        assert_eq!(imported.intake, Intake::Capture);
    }

    #[test]
    fn marker_check_is_injection_safe_and_case_insensitive() {
        // A marker with quotes/newlines must not break the python source.
        let verify = marker_verify("use \"deno\"\nnow", &["NPM".into()]);
        let src = verify["args"][1].as_str().unwrap();
        assert!(src.contains("\\\"deno\\\""));
        // forbid is lowercased.
        assert!(src.contains("npm"));
        assert!(!src.contains("NPM"));
    }

    #[test]
    fn parse_export_reads_a_normalized_array() {
        let bytes = br#"[{"content":"use deno","scope":"pm","confidence":0.8}]"#;
        let records = parse_export(bytes).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].content, "use deno");
        let imported = import(&records, &ImportPolicy::default());
        assert_eq!(imported[0].intake, Intake::Capture);
        assert_eq!(imported[0].task.skill(), "pm");
    }
}
