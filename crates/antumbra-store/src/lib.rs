// Test code may unwrap until its sweep lands (#295 to #302); production code may not.
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]
//! # antumbra-store
//!
//! SurrealDB persistence for Antumbra via surql-rs (`oneiriq-surql`). The store
//! is the single unified substrate (SurrealDB): document, vector (HNSW), and the
//! durable flow state that makes the generational loop crash-resumable.

mod dto;
mod error;

#[cfg(feature = "clap")]
pub mod args;
pub mod fusion;
pub mod knn;
mod lexical;
pub mod repo;
pub mod schema;
pub mod store;
pub mod tables;

pub use schema::{EMBED_DIM, TENANT_SESSION};
pub use store::Store;

// Re-export the connection config so callers can point at a real server
// without depending on surql-rs directly.
pub use surql::connection::ConnectionConfig;
