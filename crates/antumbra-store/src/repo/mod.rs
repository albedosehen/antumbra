//! Repositories: thin, typed accessors over the surql-rs client. Each is a set
//! of free functions taking `&Store` (functional over a session handle, rather
//! than a per-table object graph). All access goes through surql-rs builders
//! and `crud` helpers, never raw SurrealQL.

pub mod account;
pub mod boundary;
pub mod compartment;
pub mod contribution;
pub mod device;
pub mod document;
pub mod edge;
pub mod embedder_config;
pub mod evaluation;
pub mod expert;
pub mod generation;
pub mod genesis;
pub mod grow;
pub mod hive;
pub mod invite;
pub mod lifecycle;
pub mod loop_control;
pub mod magic_use;
pub mod manifest_set;
pub mod memory;
pub mod memory_chunk;
pub mod principal;
pub mod recipe;
pub mod reward;
pub mod router;
pub mod shadow;
pub mod sync;
pub mod verifier;
