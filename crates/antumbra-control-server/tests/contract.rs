//! The janus contract gate.
//!
//! THE contract lives in `antumbra_control_server::contract` and is validated
//! here over the store's REAL schema definitions (`tables(EMBED_DIM)`, the
//! same builders every deployment applies). Three failure classes become test
//! failures in this repo:
//!
//! 1. Contract-vs-schema drift: exposing a dropped column, or declaring a
//!    filter/sort no index can serve, fails validation with the offending
//!    name.
//! 2. Artifact drift: the generated artifacts (`docs/openapi.json`,
//!    `docs/schema.graphql`) must match their checked-in copies byte for byte
//!    (`JANUS_BLESS=1` re-blesses as an explicit step).
//! 3. Index regressions: dropping `memory_updated_at_idx` (or demoting a
//!    composite's prefix) breaks a sort claim and fails here.

use antumbra_control_server::contract::{contract, schema};
use janus::{generate_all, validate};

#[test]
fn contract_validates_against_the_real_schema() {
    let violations = validate(&contract(), &schema());
    assert_eq!(violations, vec![], "contract drifted from schema");
}

#[test]
fn generated_artifacts_match_the_checked_in_documents() {
    let artifacts =
        generate_all(&contract(), &schema(), &["openapi", "sdl"]).expect("contract generates");
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    for (filename, content) in &artifacts {
        let checked_in_path = format!("{root}/docs/{filename}");
        if std::env::var("JANUS_BLESS").is_ok() {
            std::fs::write(&checked_in_path, content).unwrap();
        }
        let checked_in = std::fs::read_to_string(&checked_in_path).unwrap_or_else(|_| {
            panic!("{checked_in_path} missing; run with JANUS_BLESS=1 to create")
        });
        assert_eq!(
            content.trim(),
            checked_in.trim(),
            "{filename} drifted from its checked-in copy; JANUS_BLESS=1 to re-bless",
        );
    }
}

#[test]
fn the_gate_actually_fires_on_an_unindexed_sort() {
    // Sanity that the gate is not vacuously green: an unindexable sort over
    // the REAL schema must be refused by name. `content` is a memory column
    // covered only by the BM25 full-text index, which serves no ORDER BY.
    let mut contract = contract();
    contract.resources[0].sortable.push("content".into());
    let violations = validate(&contract, &schema());
    assert!(
        violations.iter().any(|v| v.to_string().contains("content")),
        "expected a named refusal, got {violations:?}",
    );
}

#[test]
fn the_typed_columns_change_nothing_but_the_columns() {
    // The contract layer types wire columns over the store's SCHEMALESS
    // tables, and this pins down that it does ONLY that: same tables, same
    // indexes, in the same order. An index that could be edited on the way
    // through would quietly detach the gate from the schema it exists to
    // hold the contract to.
    let store = antumbra_store::schema::tables(antumbra_store::EMBED_DIM as u32);
    let typed = schema();
    assert_eq!(
        store.iter().map(|t| &t.name).collect::<Vec<_>>(),
        typed.iter().map(|t| &t.name).collect::<Vec<_>>(),
    );
    for (stored, typed) in store.iter().zip(&typed) {
        assert_eq!(stored.indexes, typed.indexes, "indexes of {}", stored.name);
        assert!(
            stored.fields.is_empty(),
            "{} grew store-side fields; the \
             contract overlay may now be shadowing them",
            stored.name
        );
    }
}
