//! The experts resource.
//!
//! One entity per file: everything here is that entity, and nothing here is
//! anything else. `super::contract` puts them together, because what janus
//! validates is the whole.

use janus::{FieldExposure, Identity, Resource, ResourceFaces};
use surql::schema::{
    datetime_field, field, float_field, int_field, string_field, FieldDefinition, FieldType,
};

use super::built;

/// The umbra: the population of small frozen experts. Shared experts have no
/// owner; private ones consolidated from a compartment carry one, and the
/// engine's read rule decides which a session sees. No tenant pin -- the
/// population is shared across tenants by design.
pub(super) fn resource() -> Resource {
    Resource {
        name: "experts".into(),
        table: "expert".into(),
        faces: ResourceFaces::ALL,
        // The domain ExpertId persists as `key` (expert_key_uq).
        identity: Identity::Column("key".into()),
        fields: vec![
            FieldExposure::column("key"),
            FieldExposure::column("name"),
            FieldExposure::column("base_model"),
            FieldExposure::column("artifact_uri"),
            FieldExposure::column("capability_card"),
            FieldExposure::column("fitness"),
            FieldExposure::column("frozen_at"),
            FieldExposure::column("generation"),
            FieldExposure::column("owner"),
            FieldExposure::column("compartment"),
            FieldExposure::column("created_at"),
            // capability_vec stays off the wire: the routing vector is 384
            // floats of gate machinery, and the HNSW index that reads it
            // never leaves the store.
        ],
        pinned: vec![],
        pinned_either: vec![],
        // expert_key_uq is the only ordering index, so key is the one honest
        // filter and the one honest sort.
        filterable: vec!["key".into()],
        filter_options: Default::default(),
        sortable: vec!["key".into()],
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

/// The wire columns of the `expert` table, typed for janus. Mirrors
/// `ExpertRow` in antumbra-store/src/dto.rs.
pub(super) fn columns() -> Vec<FieldDefinition> {
    vec![
        built(string_field("key")),
        built(string_field("name")),
        built(string_field("base_model")),
        built(string_field("artifact_uri")),
        // A free-form "what I do" card; any JSON the trainer wrote.
        built(field("capability_card", FieldType::Any)),
        built(float_field("fitness")),
        built(datetime_field("frozen_at").nullable(true)),
        built(int_field("generation")),
        built(string_field("owner").nullable(true)),
        built(string_field("compartment").nullable(true)),
        built(datetime_field("created_at")),
    ]
}
