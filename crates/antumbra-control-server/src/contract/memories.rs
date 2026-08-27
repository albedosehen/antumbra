//! The memories resource.
//!
//! One entity per file: everything here is that entity, and nothing here is
//! anything else. `super::contract` puts them together, because what kayak
//! validates is the whole.

use std::collections::BTreeMap;

use kayak::{FieldExposure, Identity, Resource, ResourceFaces};
use surql::schema::{
    bool_field, datetime_field, float_field, int_field, string_field, FieldDefinition,
};

use super::built;

/// The penumbra: soft, editable memory traces, tenant-scoped and reinforced
/// over use. Read-only in this slice -- writes stay on the MCP tools, which
/// carry the compartment/grant reasoning the engine ACL expects.
pub(super) fn resource() -> Resource {
    Resource {
        name: "memories".into(),
        table: "memory".into(),
        // A browsable collection: paged and reachable by key.
        faces: ResourceFaces::ALL,
        // The DTO layer maps the domain id to a `key` column (the reserved
        // `id` is SurrealDB's record id and never reaches the wire), so the
        // wire names an instance by `key` -- see `MemoryRow` in
        // antumbra-store/src/repo/memory.rs.
        identity: Identity::Column("key".into()),
        fields: vec![
            FieldExposure::column("key"),
            FieldExposure::column("network"),
            FieldExposure::column("content"),
            FieldExposure::column("confidence"),
            FieldExposure::column("reinforcement"),
            FieldExposure::column("volatile"),
            FieldExposure::column("compartment"),
            FieldExposure::column("author"),
            FieldExposure::column("author_host"),
            FieldExposure::column("created_at"),
            FieldExposure::column("updated_at"),
            // The embedding stays off the wire: 384 floats of recall
            // machinery per row is noise to every API consumer, and the KNN
            // paths that need it never leave the store.
        ],
        // tenant_id is server-bound on every read (the engine ACL enforces
        // it; the repo filters on it as the second layer). Pinning it is what
        // lets the prefix rule credit memory_tenant_network_idx for the
        // network filter and seat the plain listing on memory_tenant_key_uq.
        pinned: vec!["tenant_id".into()],
        pinned_either: vec![],
        filterable: vec!["network".into()],
        // The closed set the domain enum asserts (MemoryNetwork), so a caller
        // narrowing by network picks from what exists rather than guessing at
        // its spelling.
        filter_options: BTreeMap::from([(
            "network".to_owned(),
            ["world", "bank", "opinion"]
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
        )]),
        // memory_updated_at_idx leads with updated_at, the collector's own
        // watermark order; created_at is claimed by no index and so not here.
        sortable: vec!["updated_at".into()],
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

/// The wire columns of the `memory` table, typed for kayak. Mirrors what
/// `MemoryRow` actually writes -- the store keeps the table SCHEMALESS, so
/// this vocabulary is declared here, beside the resource that exposes it.
pub(super) fn columns() -> Vec<FieldDefinition> {
    vec![
        built(string_field("key")),
        built(string_field("tenant_id")),
        built(string_field("network")),
        built(string_field("content")),
        built(float_field("confidence")),
        built(int_field("reinforcement")),
        built(bool_field("volatile")),
        built(string_field("compartment").nullable(true)),
        built(string_field("author").nullable(true)),
        built(string_field("author_host").nullable(true)),
        built(datetime_field("created_at")),
        built(datetime_field("updated_at")),
    ]
}
