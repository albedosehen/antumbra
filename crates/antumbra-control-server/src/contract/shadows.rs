//! The shadows resource.
//!
//! One entity per file: everything here is that entity, and nothing here is
//! anything else. `super::contract` puts them together, because what kayak
//! validates is the whole.

use kayak::{FieldExposure, Resource};
use surql::schema::{int_field, string_field, FieldDefinition};

use super::{built, read_only};

/// The penumbra's training side: shadow models hold the plasticity while the
/// frozen population holds the competence. The exposure is deliberately the
/// lifecycle view -- key, status, generation -- because that is what an
/// operator watches; the artifact URIs and reward curves stay operational
/// until something outside the store needs them.
pub(super) fn resource() -> Resource {
    // The domain ShadowId persists as `key` (shadow_key_uq). shadow_key_uq
    // covers key; shadow_status_idx is (status, generation), covering both.
    // key and status each lead an index, so both sorts hold; generation is
    // filterable but sits behind status, so it is not a sort this contract
    // claims.
    read_only(
        "shadows",
        antumbra_store::tables::SHADOW,
        "key",
        vec![
            FieldExposure::column("key"),
            FieldExposure::column("status"),
            FieldExposure::column("generation"),
        ],
        &["key", "status", "generation"],
        &["key", "status"],
    )
}

/// The wire columns of the `shadow` table, typed for kayak. A subset of
/// `ShadowRow` in antumbra-store/src/repo/shadow.rs: only what the resource
/// exposes and the indexes read.
pub(super) fn columns() -> Vec<FieldDefinition> {
    vec![
        built(string_field("key")),
        built(string_field("status")),
        built(int_field("generation")),
    ]
}
