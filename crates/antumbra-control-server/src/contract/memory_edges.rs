//! The memory_edges resource.
//!
//! One entity per file: everything here is that entity, and nothing here is
//! anything else. `super::contract` puts them together, because what kayak
//! validates is the whole.

use kayak::{FieldExposure, Identity, Resource, ResourceFaces};
use surql::schema::{datetime_field, float_field, string_field, FieldDefinition};

use super::built;

/// The penumbra graph: typed, directed edges between memories. List-only,
/// because an edge row carries no identity on the wire -- its record key is
/// synthesised from (tenant, from, to, type) for idempotent re-relating and
/// never serialized back, so there is nothing for a by-instance GET to bind.
/// Declaring [`Identity::Absent`] says so instead of promising a field the
/// service never sends.
pub(super) fn resource() -> Resource {
    Resource {
        name: "memory_edges".into(),
        table: "memory_edge".into(),
        faces: ResourceFaces::LIST_ONLY,
        identity: Identity::Absent,
        fields: vec![
            FieldExposure::column("from_id"),
            FieldExposure::column("to_id"),
            FieldExposure::column("edge_type"),
            FieldExposure::column("weight"),
            FieldExposure::column("created_at"),
        ],
        // tenant_id rides every read; both composite indexes lead with it,
        // which is what makes from_id and to_id honest filter claims.
        pinned: vec!["tenant_id".into()],
        pinned_either: vec![],
        filterable: vec!["from_id".into(), "to_id".into()],
        filter_options: Default::default(),
        sortable: vec!["created_at".into()],
        max_page_size: 100,
        graphql: None,
        watchable: false,
        reads_require: vec![],
        rate_class: None,
        sub_resources: vec![],
        content: None,
        actions: vec![],
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
