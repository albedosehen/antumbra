//! The compartments resource.
//!
//! One entity per file: everything here is that entity, and nothing here is
//! anything else. `super::contract` puts them together, because what kayak
//! validates is the whole.

use kayak::{FieldExposure, Resource};
use surql::schema::{datetime_field, string_field, FieldDefinition};

use super::{built, read_only};

/// Compartments: the named latent-spaces of memory within a tenant, the unit
/// of organization and sharing. Read-only here; creation and sharing stay on
/// the MCP tools, where the engine's owner/grant rules are enforced.
pub(super) fn resource() -> Resource {
    Resource {
        // tenant_id is server-bound on every read; compartment_key_uq and
        // compartment_owner_idx both lead with it, so the plain listing
        // seeks and the owner filter is credited.
        pinned: vec!["tenant_id".into()],
        // The domain id persists as `key` (compartment_key_uq is over
        // tenant_id + key); the reserved SurrealDB `id` never reaches the
        // wire.
        ..read_only(
            "compartments",
            "compartment",
            "key",
            vec![
                FieldExposure::column("key"),
                FieldExposure::column("name"),
                FieldExposure::column("owner"),
                FieldExposure::column("created_at"),
                FieldExposure::column("updated_at"),
            ],
            &["owner"],
            &["updated_at"],
        )
    }
}

/// The wire columns of the `compartment` table, typed for kayak. Mirrors
/// `CompartmentRow` in antumbra-store/src/repo/compartment.rs; `updated_at`
/// is nullable because rows written before compartments carried a version
/// lack it (reads fall back to `created_at`).
pub(super) fn columns() -> Vec<FieldDefinition> {
    vec![
        built(string_field("key")),
        built(string_field("tenant_id")),
        built(string_field("owner")),
        built(string_field("name")),
        built(datetime_field("created_at")),
        built(datetime_field("updated_at").nullable(true)),
    ]
}
