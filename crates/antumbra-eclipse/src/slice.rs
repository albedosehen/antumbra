//! The partition every instrument is built on: which tasks the loop may see,
//! which are frozen away from it, which exist only to be logged, and which
//! cannot be passed at all.
//!
//! Defined in `antumbra_core::slice` and re-exported here, because the trainer
//! has to enforce the same partition these instruments read, and the request
//! that carries it to the trainer is a core port type. A partition that only
//! labels outcomes after training on every task is not a partition.

pub use antumbra_core::slice::{Holdout, Partition, Slice};
