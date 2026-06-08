//! Evaluation runs: one row per measured run (the reuse of kushtaka's
//! strongest idea). ADR-0007. The `regression_fingerprint` is what makes the
//! ADR-0001/0002 no-forgetting invariant *checkable*: a frozen expert's
//! fingerprint on its corpus must not change when the population grows.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::ids::RunId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SubjectKind {
    Expert,
    Shadow,
    Router,
    Composed,
}

impl SubjectKind {
    /// Lowercase wire form (matches serde), for typed store filters.
    pub fn as_str(self) -> &'static str {
        match self {
            SubjectKind::Expert => "expert",
            SubjectKind::Shadow => "shadow",
            SubjectKind::Router => "router",
            SubjectKind::Composed => "composed",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EvalStatus {
    Pending,
    Running,
    Success,
    Failure,
    Error,
}

impl EvalStatus {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            EvalStatus::Success | EvalStatus::Failure | EvalStatus::Error
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationRun {
    pub run_id: RunId,
    pub subject_kind: SubjectKind,
    pub subject_id: String,
    pub corpus_task_id: String,
    pub status: EvalStatus,
    #[serde(default)]
    pub metrics: Option<serde_json::Value>,
    /// `sha256(canonical output)`, the no-forgetting tripwire.
    #[serde(default)]
    pub regression_fingerprint: Option<String>,
    pub created_at: DateTime<Utc>,
}

impl EvaluationRun {
    /// Whether two runs of the same subject on the same corpus task produced
    /// byte-identical canonical output. A `false` here for a *frozen* expert is
    /// the ADR-0001 kill criterion firing.
    pub fn fingerprint_matches(&self, other: &EvaluationRun) -> bool {
        match (&self.regression_fingerprint, &other.regression_fingerprint) {
            (Some(a), Some(b)) => a == b,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(fp: Option<&str>) -> EvaluationRun {
        EvaluationRun {
            run_id: RunId::new("run:1"),
            subject_kind: SubjectKind::Expert,
            subject_id: "expert:1".into(),
            corpus_task_id: "task:1".into(),
            status: EvalStatus::Success,
            metrics: None,
            regression_fingerprint: fp.map(str::to_string),
            created_at: Utc::now(),
        }
    }

    #[test]
    fn fingerprint_comparison() {
        assert!(run(Some("abc")).fingerprint_matches(&run(Some("abc"))));
        assert!(!run(Some("abc")).fingerprint_matches(&run(Some("xyz"))));
        assert!(!run(None).fingerprint_matches(&run(Some("abc"))));
    }
}
