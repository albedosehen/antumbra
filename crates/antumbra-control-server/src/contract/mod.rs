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

use kayak::{Contract, ContractLimits};
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

/// Finalise a field builder. The names in this module are static and valid,
/// so a refusal is a typo in a sibling file, worth stopping the build over;
/// reserved-word warnings are the schema layer's concern, not the wire's.
fn built(field: FieldBuilder) -> FieldDefinition {
    field
        .build_unchecked()
        .expect("contract column names are static and valid")
}
