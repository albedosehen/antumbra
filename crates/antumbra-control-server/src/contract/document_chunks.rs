//! The document_chunks resource.
//!
//! One entity per file: everything here is that entity, and nothing here is
//! anything else. `super::contract` puts them together, because what janus
//! validates is the whole.

use janus::{FieldExposure, Identity, Resource, ResourceFaces};
use surql::schema::{datetime_field, int_field, string_field, FieldDefinition};

use super::built;

/// Knowledge-document chunks: reference material, a distinct type from
/// episodic memory but tenant-isolated the same way. Read-only; ingestion is
/// the MCP `store_document` path, which owns chunking and embedding.
pub(super) fn resource() -> Resource {
    Resource {
        name: "document_chunks".into(),
        table: "document_chunk".into(),
        faces: ResourceFaces::ALL,
        // Chunks are addressed by their globally-unique domain id, stored as
        // `key` by the DTO convention.
        identity: Identity::Column("key".into()),
        fields: vec![
            FieldExposure::column("title"),
            FieldExposure::column("source"),
            FieldExposure::column("ordinal"),
            FieldExposure::column("content"),
            // Document-of-record provenance: when a copal archive was
            // configured at ingest, the chunk names the archived file and
            // its content digest (see antumbra-mcp/src/copal.rs). Absent on
            // chunks ingested without one, like the row itself.
            FieldExposure::column("copal_file"),
            FieldExposure::column("copal_digest"),
            FieldExposure::column("created_at"),
            // The embedding stays off the wire for the same reason the
            // memory one does: recall machinery, not content.
        ],
        // tenant_id is server-bound; document_chunk_tenant_title_idx leads
        // with it, seating the plain listing and crediting the title filter.
        pinned: vec!["tenant_id".into()],
        pinned_either: vec![],
        filterable: vec!["title".into()],
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

/// The wire columns of the `document_chunk` table, typed for janus. Mirrors
/// `ChunkRow` in antumbra-store/src/repo/document.rs.
pub(super) fn columns() -> Vec<FieldDefinition> {
    vec![
        built(string_field("key")),
        built(string_field("tenant_id")),
        built(string_field("title")),
        built(string_field("source").nullable(true)),
        built(int_field("ordinal")),
        built(string_field("content")),
        built(string_field("copal_file").nullable(true)),
        built(string_field("copal_digest").nullable(true)),
        built(datetime_field("created_at")),
    ]
}
