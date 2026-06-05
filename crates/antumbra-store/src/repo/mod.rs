//! Repositories: thin, typed accessors over the surql-rs client. Each is a set
//! of free functions taking `&Store` (functional over a session handle, rather
//! than a per-table object graph). All access goes through surql-rs builders
//! and `crud` helpers — never raw SurrealQL.

pub mod boundary;
pub mod compartment;
pub mod edge;
pub mod evaluation;
pub mod expert;
pub mod generation;
pub mod memory;
pub mod principal;
pub mod reward;
pub mod router;
pub mod shadow;
