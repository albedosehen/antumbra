//! The control plane's library face.
//!
//! The binary (`main.rs`) is the hosted signup/login/token shell; this lib
//! target exists because the kayak contract is *data*, and data wants to be
//! importable: the drift gate in `tests/contract.rs` validates it against the
//! store's real schema, and the checked-in artifacts (`docs/openapi.json`,
//! `docs/schema.graphql`) are generated from it. Nothing here executes -- the
//! contract is declared in this slice, not served -- and it sits behind the
//! `contract` feature, since the server's routes do not need kayak.

#[cfg(feature = "contract")]
pub mod contract;
