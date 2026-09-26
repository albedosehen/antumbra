//! The evaluation_runs resource.
//!
//! One entity per file: everything here is that entity, and nothing here is
//! anything else. `super::contract` puts them together, because what kayak
//! validates is the whole.

use kayak::{FieldExposure, Identity, Resource, ResourceFaces};
use surql::schema::{datetime_field, field, string_field, FieldDefinition, FieldType};

use super::built;

/// The validation harness: one row per measured run. List-only, because the
/// repo reads are all listings (per subject, latest, recent) and the rows
/// carry no addressed identity on the wire -- `run_id` names the run that
/// was measured, not the measurement row, and no by-instance read exists to
/// promise.
pub(super) fn resource() -> Resource {
    Resource {
        name: "evaluation_runs".into(),
        table: "evaluation_run".into(),
        faces: ResourceFaces::LIST_ONLY,
        identity: Identity::Absent,
        fields: vec![
            FieldExposure::column("run_id"),
            FieldExposure::column("subject_kind"),
            FieldExposure::column("subject_id"),
            FieldExposure::column("corpus_task_id"),
            FieldExposure::column("status"),
            FieldExposure::column("metrics"),
            FieldExposure::column("regression_fingerprint"),
            FieldExposure::column("created_at"),
        ],
        pinned: vec![],
        pinned_either: vec![],
        // eval_subject_idx is (subject_kind, subject_id): both are indexed
        // filters, and subject_kind -- the leading column -- the one indexed
        // sort. created_at is claimed by no index, so the repo's
        // newest-first reads stay its own business rather than a sort this
        // contract promises.
        filterable: vec!["subject_kind".into(), "subject_id".into()],
        filter_options: Default::default(),
        sortable: vec!["subject_kind".into()],
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

/// The wire columns of the `evaluation_run` table, typed for kayak. The
/// domain `EvaluationRun` persists directly (it has no reserved `id` field),
/// so this mirrors the struct in antumbra-core/src/evaluation.rs.
pub(super) fn columns() -> Vec<FieldDefinition> {
    vec![
        built(string_field("run_id")),
        built(string_field("subject_kind")),
        built(string_field("subject_id")),
        built(string_field("corpus_task_id")),
        built(string_field("status")),
        // Free-form metric payload; absent until a run reports numbers.
        built(field("metrics", FieldType::Any).nullable(true)),
        built(string_field("regression_fingerprint").nullable(true)),
        built(datetime_field("created_at")),
    ]
}
