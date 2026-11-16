//! THE Antumbra contract: one contract, every face.
//!
//! What this assembles drives the checked-in artifacts (`docs/openapi.json`,
//! `docs/schema.graphql`) and the drift gate in `tests/contract.rs`, which
//! validates it against the store's real indexes -- dropping
//! `memory_updated_at_idx` breaks the `updated_at` sort claim and fails a
//! test in this repo, by name, before it becomes a production table scan.
//!
//! Each entity lives in its own file beside this one, so opening
//! `memories.rs` puts nothing in front of a reader except the memories
//! resource. This module holds what belongs to no single entity and hands
//! kayak the whole, because the checks that matter span it.
//!
//! Everything here is read-only and read-narrow on purpose: this slice
//! declares the faces the store can already serve from its indexes, and
//! nothing more. Writes stay on the MCP tools, where the engine's
//! compartment/grant ACL does the reasoning a bare POST could not.

use kayak::{Contract, ContractLimits, FieldExposure, Identity, Resource, ResourceFaces};
use surql::schema::{FieldBuilder, FieldDefinition, TableDefinition};

mod compartments;
mod document_chunks;
mod evaluation_runs;
mod experts;
mod grants;
mod memories;
mod memory_edges;
mod orchestration_runs;
mod shadows;

/// The wire contract, assembled from the entities beside this file.
pub fn contract() -> Contract {
    Contract {
        name: "antumbra".into(),
        version: "0.1.0".into(),
        ir_revision: 1,
        // What every antumbra surface already requires: `Authorization:
        // Bearer <jwt>` -- the control plane signs RS256 tokens and the
        // data-plane MCP server verifies them. Declared so the generated
        // clients send the credential the service actually checks, and so
        // the differ names any change to it as the break it would be.
        auth: kayak::AuthScheme::Bearer,
        // The default, stated because antumbra's routes will be versioned
        // and a reader should not have to know kayak's default to know
        // antumbra's paths.
        api_prefix: "/v1".into(),
        // Modest ceilings, declared here so they appear in the artifacts
        // and tightening them is a breaking change the differ names. No
        // watch ceiling: nothing is watchable in this slice, so a ceiling
        // would bound nothing.
        limits: Some(ContractLimits {
            max_depth: Some(10),
            max_complexity: Some(500),
            max_watches_per_principal: None,
        }),
        // No budgets yet: nothing references a rate class in this slice,
        // and an unreferenced budget is a promise about metering nothing.
        rate_classes: vec![],
        resources: vec![
            memories::resource(),
            memory_edges::resource(),
            compartments::resource(),
            grants::resource(),
            document_chunks::resource(),
            experts::resource(),
            evaluation_runs::resource(),
            orchestration_runs::resource(),
            shadows::resource(),
        ],
        queries: vec![],
    }
}

/// The schema the contract is validated and generated against: the store's
/// real table definitions with the wire columns typed in.
///
/// The store keeps every table SCHEMALESS by design (v0 leans on explicit
/// indexes, not field DDL), which leaves `TableDefinition::fields` empty --
/// and kayak resolves every exposed column against those fields, both to
/// refuse a column the table does not have and to type the OpenAPI/SDL
/// schemas. So the contract layer carries the missing half itself: each
/// entity file types exactly the columns its repo layer actually writes,
/// and this function lays them over the store's own definitions.
///
/// The indexes -- the load-bearing half of the gate, the thing every filter
/// and sort claim is checked against -- come from
/// [`antumbra_store::schema::tables`] untouched, so an index dropped or
/// demoted in the store still fails the gate here. Only the field lists are
/// contract-side, and they can drift only in the direction the gate cannot
/// see anyway on a schemaless table: a column the repo stopped writing.
// The panic is the gate: this runs only in the drift test, and a missing
// table must stop it by name rather than pass as a silently thinner schema.
#[allow(clippy::panic)]
pub fn schema() -> Vec<TableDefinition> {
    let mut tables = antumbra_store::schema::tables(antumbra_store::EMBED_DIM as u32);
    for (name, columns) in [
        ("memory", memories::columns()),
        ("memory_edge", memory_edges::columns()),
        ("compartment", compartments::columns()),
        ("grant", grants::columns()),
        ("document_chunk", document_chunks::columns()),
        ("expert", experts::columns()),
        ("evaluation_run", evaluation_runs::columns()),
        ("orchestration_run", orchestration_runs::columns()),
        ("shadow", shadows::columns()),
    ] {
        let table = tables
            .iter_mut()
            .find(|table| table.name == name)
            // A contract module naming a table the store dropped is exactly
            // the drift the gate exists to catch; stopping here names it.
            .unwrap_or_else(|| panic!("table {name} is missing from the store schema"));
        table.fields = columns;
    }
    tables
}

/// A read-only, browsable collection: paged and reachable by its `key`
/// column, filterable and sortable only where an index serves the claim,
/// and nothing more. Every resource in this slice is one of these or a
/// [`listing`]; what differs between them is what each file states, and the
/// eleven defaults that never differ are stated once, here. Writes stay on
/// the MCP tools, which carry the compartment and grant reasoning the engine
/// ACL expects.
fn read_only(
    name: &str,
    table: &str,
    key: &str,
    fields: Vec<FieldExposure>,
    filterable: &[&str],
    sortable: &[&str],
) -> Resource {
    Resource {
        faces: ResourceFaces::ALL,
        identity: Identity::Column(key.into()),
        ..listing(name, table, fields, filterable, sortable)
    }
}

/// A read-only listing with no identity of its own: rows reached only
/// through the list, filterable and sortable only where an index serves the
/// claim.
fn listing(
    name: &str,
    table: &str,
    fields: Vec<FieldExposure>,
    filterable: &[&str],
    sortable: &[&str],
) -> Resource {
    Resource {
        name: name.into(),
        table: table.into(),
        faces: ResourceFaces::LIST_ONLY,
        identity: Identity::Absent,
        fields,
        pinned: vec![],
        pinned_either: vec![],
        filterable: filterable.iter().map(|s| (*s).to_owned()).collect(),
        filter_options: Default::default(),
        sortable: sortable.iter().map(|s| (*s).to_owned()).collect(),
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

/// Finalize a field builder. The names in this module are static and valid,
/// so a refusal is a typo in a sibling file, worth stopping the build over;
/// reserved-word warnings are the schema layer's concern, not the wire's.
#[allow(clippy::expect_used)]
fn built(field: FieldBuilder) -> FieldDefinition {
    field
        .build_unchecked()
        .expect("contract column names are static and valid")
}
