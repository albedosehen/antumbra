//! The frozen-expert population (the umbra): a population of small frozen
//! experts, the umbra ideal.
//!
//! In v0 an expert *is* a frozen LoRA adapter over the shared, code-capable
//! base. The weights live on disk (`artifact_uri`); this row is metadata plus
//! the learned capability vector used for routing-as-retrieval.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::ids::{CompartmentId, ExpertId, Generation, UserId};

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
    /// Learned routing vector; co-learned from evaluated behaviour.
    #[serde(default)]
    pub capability_vec: Option<Vec<f32>>,
    #[serde(default)]
    pub fitness: f32,
    /// `Some` once the artifact is sealed read-only; freezing is permanent.
    #[serde(default)]
    pub frozen_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub generation: Generation,
    /// The owning user for a **private** expert (consolidated from a private
    /// compartment). `None` = a shared expert in the common umbra
    /// (readable by every tenant session, by multi-tenant isolation).
    #[serde(default)]
    pub owner: Option<UserId>,
    /// The source compartment a private expert was consolidated from.
    #[serde(default)]
    pub compartment: Option<CompartmentId>,
    /// The node whose disk holds `artifact_uri` (ADR-0017 A2). `None` for an
    /// expert minted before placement was recorded, which is read as "here",
    /// because until a user's nodes shared a store there was only ever one
    /// machine it could have been on.
    ///
    /// This matters because the row travels and the weights do not. ADR-0017
    /// keeps adapters out of sync scope deliberately, so once a user's fabric
    /// reconciles, every node learns about every expert while exactly one of
    /// them can actually open the file. A node that registered them all would
    /// route to an adapter it does not have and fail at serve time, on a path
    /// that looks perfectly valid in the row.
    ///
    /// A field rather than the `placed_on` graph edge ADR-0007 sketched:
    /// placement is one-to-one in v1, because ADR-0017 defers shipping
    /// adapters to every genesis node, and an edge earns its keep when a
    /// relation is many-to-many or traversed. It is neither yet. Shipping an
    /// adapter to a second node is what would turn this back into an edge.
    #[serde(default)]
    pub placed_on: Option<String>,
    pub created_at: DateTime<Utc>,
}

impl Expert {
    /// Freezing is the load-bearing invariant of the frozen-expert population: a
    /// frozen expert is never written again.
    pub fn is_frozen(&self) -> bool {
        self.frozen_at.is_some()
    }

    /// A private expert is owned by a user (vs a shared expert, `owner = None`).
    pub fn is_private(&self) -> bool {
        self.owner.is_some()
    }

    /// Whether `host` is the node that can actually open this expert's adapter.
    ///
    /// An unplaced expert is servable anywhere, which is the only reading that
    /// keeps every expert minted before placement existed working: there was
    /// one machine then, so "unknown" and "here" were the same answer. The cost
    /// of that choice is that an old expert stays registered on a node that
    /// cannot open it, which is the behaviour those nodes have today anyway.
    pub fn is_placed_on(&self, host: &str) -> bool {
        match &self.placed_on {
            Some(node) => node == host,
            None => true,
        }
    }

    /// Cosine similarity of this expert's capability vector to a query vector,
    /// or `None` if the expert has not yet been embedded. Used by the gate
    /// for coverage scoring.
    pub fn capability_similarity(&self, query: &[f32]) -> Option<f32> {
        self.capability_vec
            .as_deref()
            .map(|cap| cosine_similarity(cap, query))
    }

    /// The texts on its capability card the learned router trains on. None
    /// for a card that lists no exemplars.
    pub fn exemplars(&self) -> Vec<String> {
        self.capability_card
            .get("exemplars")
            .and_then(|v| v.as_array())
            .map(|xs| {
                xs.iter()
                    .filter_map(|x| x.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
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

    fn placement_expert(placed_on: Option<&str>) -> Expert {
        Expert {
            id: ExpertId::new("expert:one"),
            name: "one".into(),
            base_model: "base".into(),
            artifact_uri: "adapters/one.safetensors".into(),
            capability_card: serde_json::Value::Null,
            capability_vec: None,
            fitness: 1.0,
            frozen_at: None,
            generation: Generation::ZERO,
            owner: None,
            compartment: None,
            placed_on: placed_on.map(str::to_owned),
            created_at: Utc::now(),
        }
    }

    /// The row travels and the weights do not. Once a user's nodes reconcile,
    /// every node learns about every expert while exactly one holds each file.
    #[test]
    fn an_expert_is_servable_only_where_its_adapter_actually_is() {
        let on_the_rig = placement_expert(Some("the-rig"));
        assert!(on_the_rig.is_placed_on("the-rig"));
        assert!(
            !on_the_rig.is_placed_on("her-laptop"),
            "the laptop would route to a path it cannot open"
        );
    }

    /// Every expert minted before placement was recorded. There was one machine
    /// then, so "unknown" and "here" were the same answer, and reading it any
    /// other way would un-serve a population that works today.
    #[test]
    fn an_unplaced_expert_is_servable_anywhere() {
        let old = placement_expert(None);
        assert!(old.is_placed_on("the-rig"));
        assert!(old.is_placed_on("her-laptop"));
        assert!(old.is_placed_on("anywhere-at-all"));
    }

    /// The host name is compared whole. A node is not "the same machine" as one
    /// whose name it happens to start with.
    #[test]
    fn exemplars_are_read_from_the_card() {
        let mut e = placement_expert(None);
        assert!(e.exemplars().is_empty());
        e.capability_card = serde_json::json!({"exemplars": ["a", 1, "b"]});
        assert_eq!(e.exemplars(), vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn placement_is_not_a_prefix_match() {
        let node = placement_expert(Some("rig"));
        assert!(node.is_placed_on("rig"));
        assert!(!node.is_placed_on("rig-2"));
        assert!(!node.is_placed_on("the-rig"));
        assert!(!node.is_placed_on(""));
    }
}
