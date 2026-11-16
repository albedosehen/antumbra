//! The memory_edges resource.
//!
//! One entity per file: everything here is that entity, and nothing here is
//! anything else. `super::contract` puts them together, because what kayak
//! validates is the whole.

use kayak::{FieldExposure, Resource};
use surql::schema::{datetime_field, float_field, string_field, FieldDefinition};

use super::{built, listing};

/// The penumbra graph: typed, directed edges between memories. List-only,
/// because an edge row carries no identity on the wire -- its record key is
/// synthesised from (tenant, from, to, type) for idempotent re-relating and
/// never serialized back, so there is nothing for a by-instance GET to bind.
/// Declaring [`Identity::Absent`] says so instead of promising a field the
/// service never sends.
pub(super) fn resource() -> Resource {
    Resource {
        // tenant_id rides every read; both composite indexes lead with it,
        // which is what makes from_id and to_id valid filter claims.
        pinned: vec!["tenant_id".into()],
        ..listing(
            "memory_edges",
            "memory_edge",
            vec![
                FieldExposure::column("from_id"),
                FieldExposure::column("to_id"),
                FieldExposure::column("edge_type"),
                FieldExposure::column("weight"),
                FieldExposure::column("created_at"),
            ],
            &["from_id", "to_id"],
            &["created_at"],
        )
    }
}

/// The wire columns of the `memory_edge` table, typed for kayak. Mirrors
/// `EdgeRow` in antumbra-store/src/repo/edge.rs.
pub(super) fn columns() -> Vec<FieldDefinition> {
    vec![
        built(string_field("tenant_id")),
        built(string_field("from_id")),
        built(string_field("to_id")),
        built(string_field("edge_type")),
        built(float_field("weight")),
        built(datetime_field("created_at")),
    ]
}
