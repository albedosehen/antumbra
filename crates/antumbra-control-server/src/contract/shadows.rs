//! The shadows resource.
//!
//! One entity per file: everything here is that entity, and nothing here is
//! anything else. `super::contract` puts them together, because what janus
//! validates is the whole.

use janus::{FieldExposure, Identity, Resource, ResourceFaces};
use surql::schema::{int_field, string_field, FieldDefinition};

use super::built;

/// The penumbra's training side: shadow models hold the plasticity while the
/// frozen population holds the competence. The exposure is deliberately the
/// lifecycle view -- key, status, generation -- because that is what an
/// operator watches; the artifact URIs and reward curves stay operational
/// until something outside the store needs them.
pub(super) fn resource() -> Resource {
    Resource {
        name: "shadows".into(),
        table: "shadow".into(),
        faces: ResourceFaces::ALL,
        // The domain ShadowId persists as `key` (shadow_key_uq).
        identity: Identity::Column("key".into()),
        fields: vec![
            FieldExposure::column("key"),
            FieldExposure::column("status"),
            FieldExposure::column("generation"),
        ],
        pinned: vec![],
        pinned_either: vec![],
        // shadow_key_uq covers key; shadow_status_idx is (status,
        // generation), covering both. key and status each lead an index, so
        // both sorts hold; generation is filterable but sits behind status,
        // so it is not a sort this contract claims.
        filterable: vec!["key".into(), "status".into(), "generation".into()],
        filter_options: Default::default(),
        sortable: vec!["key".into(), "status".into()],
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

/// The wire columns of the `shadow` table, typed for janus. A subset of
/// `ShadowRow` in antumbra-store/src/repo/shadow.rs: only what the resource
/// exposes and the indexes read.
pub(super) fn columns() -> Vec<FieldDefinition> {
    vec![
        built(string_field("key")),
        built(string_field("status")),
        built(int_field("generation")),
    ]
}
