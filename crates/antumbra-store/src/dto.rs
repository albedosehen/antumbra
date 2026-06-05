//! Storage DTOs. These decouple the domain types from SurrealDB quirks: the
//! domain `id` becomes a `key` column (the reserved `id` is SurrealDB's record
//! id), and timestamps round-trip as RFC3339 strings.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use antumbra_core::ids::{CompartmentId, ExpertId, Generation, UserId};
use antumbra_core::{AntumbraError, Expert, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ExpertRow {
    pub key: String,
    pub name: String,
    pub base_model: String,
    pub artifact_uri: String,
    #[serde(default)]
    pub capability_card: Value,
    #[serde(default)]
    pub capability_vec: Option<Vec<f32>>,
    #[serde(default)]
    pub fitness: f32,
    #[serde(default)]
    pub frozen_at: Option<String>,
    #[serde(default)]
    pub generation: u32,
    // Omitted when None so the engine sees `owner = NONE` for shared experts
    // (the shared-umbra read rule, ADR-0013).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compartment: Option<String>,
    pub created_at: String,
}

impl ExpertRow {
    pub(crate) fn from_domain(e: &Expert) -> Self {
        ExpertRow {
            key: e.id.as_str().to_string(),
            name: e.name.clone(),
            base_model: e.base_model.clone(),
            artifact_uri: e.artifact_uri.clone(),
            capability_card: e.capability_card.clone(),
            capability_vec: e.capability_vec.clone(),
            fitness: e.fitness,
            frozen_at: e.frozen_at.map(|t| t.to_rfc3339()),
            generation: e.generation.0,
            owner: e.owner.as_ref().map(|u| u.as_str().to_string()),
            compartment: e.compartment.as_ref().map(|c| c.as_str().to_string()),
            created_at: e.created_at.to_rfc3339(),
        }
    }

    pub(crate) fn into_domain(self) -> Result<Expert> {
        Ok(Expert {
            id: ExpertId::new(self.key),
            name: self.name,
            base_model: self.base_model,
            artifact_uri: self.artifact_uri,
            capability_card: self.capability_card,
            capability_vec: self.capability_vec,
            fitness: self.fitness,
            frozen_at: self.frozen_at.as_deref().map(parse_dt).transpose()?,
            generation: Generation(self.generation),
            owner: self.owner.map(UserId::new),
            compartment: self.compartment.map(CompartmentId::new),
            created_at: parse_dt(&self.created_at)?,
        })
    }
}

pub(crate) fn parse_dt(s: &str) -> Result<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .map(|d| d.with_timezone(&Utc))
        .map_err(|e| AntumbraError::other(format!("bad datetime {s:?}: {e}")))
}
