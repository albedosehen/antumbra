//! The grants resource.
//!
//! One entity per file: everything here is that entity, and nothing here is
//! anything else. `super::contract` puts them together, because what janus
//! validates is the whole.

use janus::{FieldExposure, Identity, Resource, ResourceFaces};
use surql::schema::{datetime_field, string_field, FieldDefinition};

use super::built;

/// Capability grants: compartment sharing, intra-tenant and user-to-user.
/// List-only, because a grant row carries no identity on the wire -- like an
/// edge, its record key is synthesised from (tenant, compartment, grantee)
/// for idempotent re-granting and never serialized back. Granting and
/// revoking stay on the MCP tools, where the engine's owner rule fails a
/// forgery closed.
pub(super) fn resource() -> Resource {
    Resource {
        name: "grants".into(),
        table: "grant".into(),
        faces: ResourceFaces::LIST_ONLY,
        identity: Identity::Absent,
        fields: vec![
            FieldExposure::column("grantee"),
            FieldExposure::column("compartment"),
            FieldExposure::column("capability"),
            FieldExposure::column("created_at"),
            FieldExposure::column("updated_at"),
        ],
        // tenant_id rides every read; grant_grantee_idx and
        // grant_compartment_idx both lead with it, crediting both filters.
        pinned: vec!["tenant_id".into()],
        pinned_either: vec![],
        filterable: vec!["grantee".into(), "compartment".into()],
        filter_options: Default::default(),
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

/// The wire columns of the `grant` table, typed for janus. Mirrors
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
