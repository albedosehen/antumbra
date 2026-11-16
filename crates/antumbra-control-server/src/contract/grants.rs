//! The grants resource.
//!
//! One entity per file: everything here is that entity, and nothing here is
//! anything else. `super::contract` puts them together, because what kayak
//! validates is the whole.

use kayak::{FieldExposure, Resource};
use surql::schema::{datetime_field, string_field, FieldDefinition};

use super::{built, listing};

/// Capability grants: compartment sharing, intra-tenant and user-to-user.
/// List-only, because a grant row carries no identity on the wire -- like an
/// edge, its record key is synthesised from (tenant, compartment, grantee)
/// for idempotent re-granting and never serialized back. Granting and
/// revoking stay on the MCP tools, where the engine's owner rule fails a
/// forgery closed.
pub(super) fn resource() -> Resource {
    Resource {
        // tenant_id rides every read; grant_grantee_idx and
        // grant_compartment_idx both lead with it, crediting both filters.
        pinned: vec!["tenant_id".into()],
        ..listing(
            "grants",
            antumbra_store::tables::GRANT,
            vec![
                FieldExposure::column("grantee"),
                FieldExposure::column("compartment"),
                FieldExposure::column("capability"),
                FieldExposure::column("created_at"),
                FieldExposure::column("updated_at"),
            ],
            &["grantee", "compartment"],
            &["updated_at"],
        )
    }
}

/// The wire columns of the `grant` table, typed for kayak. Mirrors
/// `GrantRow` in antumbra-store/src/repo/compartment.rs; `updated_at` is
/// nullable for rows written before grants carried a version.
pub(super) fn columns() -> Vec<FieldDefinition> {
    vec![
        built(string_field("tenant_id")),
        built(string_field("compartment")),
        built(string_field("grantee")),
        built(string_field("capability")),
        built(string_field("granted_by")),
        built(datetime_field("created_at")),
        built(datetime_field("updated_at").nullable(true)),
    ]
}
