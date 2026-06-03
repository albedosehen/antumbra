//! # antumbra-store
//!
//! SurrealDB persistence for Antumbra via surql-rs (`oneiriq-surql`). The store
//! is the single substrate of ADR-0007: document, vector (HNSW), and the
//! durable flow state that makes the generational loop crash-resumable.

mod dto;
mod error;

pub mod repo;
pub mod schema;
pub mod store;

pub use schema::EMBED_DIM;
pub use store::Store;

// Re-export the connection config so callers can point at a real server
// without depending on surql-rs directly.
pub use surql::connection::ConnectionConfig;
