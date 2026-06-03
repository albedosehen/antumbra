//! The frozen-expert population (the umbra). ADR-0001.
//!
//! In v0 an expert *is* a frozen LoRA adapter over the shared, code-capable
//! base. The weights live on disk (`artifact_uri`); this row is metadata plus
//! the learned capability vector used for routing-as-retrieval (ADR-0005).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::ids::{ExpertId, Generation};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Expert {
    pub id: ExpertId,
    pub name: String,
    /// The shared base this adapter rides on (e.g. a code-tuned 7-8B).
    pub base_model: String,
    /// Path to the adapter artifact (gguf / safetensors / lora).
    pub artifact_uri: String,
    /// Structured "what I do" card.
    #[serde(default)]
    pub capability_card: serde_json::Value,
    /// Learned routing vector; co-learned from evaluated behaviour (ADR-0005).
    #[serde(default)]
    pub capability_vec: Option<Vec<f32>>,
    #[serde(default)]
    pub fitness: f32,
    /// `Some` once the artifact is sealed read-only; freezing is permanent.
    #[serde(default)]
    pub frozen_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub generation: Generation,
    pub created_at: DateTime<Utc>,
}

impl Expert {
    /// Freezing is the load-bearing invariant of ADR-0001: a frozen expert is
    /// never written again.
    pub fn is_frozen(&self) -> bool {
        self.frozen_at.is_some()
    }

    /// Cosine similarity of this expert's capability vector to a query vector,
    /// or `None` if the expert has not yet been embedded. Used by the gate
    /// (ADR-0005) for coverage scoring.
    pub fn capability_similarity(&self, query: &[f32]) -> Option<f32> {
        self.capability_vec
            .as_deref()
            .map(|cap| cosine_similarity(cap, query))
    }
}

/// Cosine similarity, defined to `0.0` for a zero-norm or length-mismatched
/// vector so callers never see a `NaN` leak into routing scores.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_handles_degenerate_inputs() {
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]), 1.0);
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 0.0]), 0.0);
        let orth = cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]);
        assert!(orth.abs() < 1e-6);
    }
}
