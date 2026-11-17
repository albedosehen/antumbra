//! The orchestration_runs resource.
//!
//! One entity per file: everything here is that entity, and nothing here is
//! anything else. `super::contract` puts them together, because what kayak
//! validates is the whole.

use kayak::{FieldExposure, Resource};
use surql::schema::{array_field, datetime_field, int_field, string_field, FieldDefinition};

use super::{built, read_only};

/// Durable orchestration runs: the status field IS the checkpoint, so a
/// crash resumes mid-task. The schema is ahead of the repo here -- no store
/// module persists this table yet -- so the exposures follow the schema's
/// own key convention (orun_key_uq) and the domain struct in
/// antumbra-core/src/orchestration.rs, the two things the eventual repo
/// layer is bound by.
pub(super) fn resource() -> Resource {
    // The domain RunId persists as `key` by the DTO convention the unique
    // index already assumes. orun_key_uq covers key; orun_status_idx is
    // (status, updated_at), covering status. Each leads its index, so both
    // sort claims hold.
    read_only(
        "orchestration_runs",
        antumbra_store::tables::ORCHESTRATION_RUN,
        "key",
        vec![
            FieldExposure::column("key"),
            FieldExposure::column("task_id"),
            FieldExposure::column("round"),
            FieldExposure::column("status"),
            FieldExposure::column("chosen_experts"),
            FieldExposure::column("compose_strategy"),
            FieldExposure::column("updated_at"),
        ],
        &["key", "status"],
        &["key", "status"],
    )
}

/// The wire columns of the `orchestration_run` table, typed for kayak.
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
