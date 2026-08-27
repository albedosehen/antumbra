//! The orchestration_runs resource.
//!
//! One entity per file: everything here is that entity, and nothing here is
//! anything else. `super::contract` puts them together, because what janus
//! validates is the whole.

use janus::{FieldExposure, Identity, Resource, ResourceFaces};
use surql::schema::{array_field, datetime_field, int_field, string_field, FieldDefinition};

use super::built;

/// Durable orchestration runs: the status field IS the checkpoint, so a
/// crash resumes mid-task. The schema is ahead of the repo here -- no store
/// module persists this table yet -- so the exposures follow the schema's
/// own key convention (orun_key_uq) and the domain struct in
/// antumbra-core/src/orchestration.rs, the two things the eventual repo
/// layer is bound by.
pub(super) fn resource() -> Resource {
    Resource {
        name: "orchestration_runs".into(),
        table: "orchestration_run".into(),
        faces: ResourceFaces::ALL,
        // The domain RunId persists as `key` by the DTO convention the
        // unique index already assumes.
        identity: Identity::Column("key".into()),
        fields: vec![
            FieldExposure::column("key"),
            FieldExposure::column("task_id"),
            FieldExposure::column("round"),
            FieldExposure::column("status"),
            FieldExposure::column("chosen_experts"),
            FieldExposure::column("compose_strategy"),
            FieldExposure::column("updated_at"),
        ],
        pinned: vec![],
        pinned_either: vec![],
        // orun_key_uq covers key; orun_status_idx is (status, updated_at),
        // covering status. Each leads its index, so both sort claims hold.
        filterable: vec!["key".into(), "status".into()],
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

/// The wire columns of the `orchestration_run` table, typed for janus.
/// Mirrors `OrchestrationRun` in antumbra-core/src/orchestration.rs with the
/// id-to-key mapping every persisted entity gets.
pub(super) fn columns() -> Vec<FieldDefinition> {
    vec![
        built(string_field("key")),
        built(string_field("task_id")),
        built(int_field("round")),
        built(string_field("status")),
        built(array_field("chosen_experts")),
        built(string_field("compose_strategy").nullable(true)),
        built(datetime_field("updated_at")),
    ]
}
