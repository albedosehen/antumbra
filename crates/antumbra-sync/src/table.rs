//! Which tables the collector replicates, the field on each that carries its
//! version (the last-write-wins tiebreaker), and what a user-scoped collector
//! carries from it.

use antumbra_store::tables;

use crate::scope::{Replicate, Scope};

/// A table to replicate and the row field whose RFC3339 timestamp orders writes
/// for last-write-wins. `memory` is mutated in place (reinforce, consolidate),
/// so its `updated_at` is load-bearing; the rest are effectively write-once, so
/// `created_at` is their version.
#[derive(Debug, Clone, Copy)]
pub struct TableSpec {
    pub name: &'static str,
    pub version_field: &'static str,
    /// What a user-scoped collector carries from this table. Ignored entirely
    /// when the collector runs as owner, which is every deployment today.
    pub replicate: Replicate,
}

impl TableSpec {
    const fn new(name: &'static str, version_field: &'static str, replicate: Replicate) -> Self {
        Self {
            name,
            version_field,
            replicate,
        }
    }
}

/// The penumbra tables, in dependency order: a compartment and its grants must
/// exist before the memories whose ACL resolves against them, so they replicate
/// first. Experts/adapters are large on-disk safetensors, not DB rows, so they
/// are out of scope for store sync.
pub const PENUMBRA_TABLES: &[TableSpec] = &[
    // compartment carries `updated_at` (bumped on delete) so a deletion
    // out-versions a stale live row and propagates under LWW.
    TableSpec::new(
        tables::COMPARTMENT,
        "updated_at",
        Replicate::Owned(|scope: &Scope, row| scope.owns_compartment(row)),
    ),
    // grant carries `updated_at` (bumped on revoke) so a revocation out-versions a
    // stale live grant and propagates under LWW.
    TableSpec::new(
        tables::GRANT,
        "updated_at",
        Replicate::Owned(|scope: &Scope, row| scope.grants_own_compartment(row)),
    ),
    TableSpec::new(
        tables::MEMORY,
        "updated_at",
        Replicate::Owned(|scope: &Scope, row| scope.holds_memory(row)),
    ),
    TableSpec::new(tables::MEMORY_EDGE, "created_at", Replicate::EngineDecides),
    // The fabric itself. These two are what make a user's nodes
    // more than a set of machines that happen to share a database.
    //
    // `device_profile` so a node can see the rest of the user's fabric at all:
    // the whole of `genesis_for_user` is a read of rows another machine wrote,
    // so without replication every node believes it is alone and nothing is
    // ever dispatched anywhere.
    //
    // `genesis_request` because the queue IS the delivery mechanism. A laptop
    // that cannot train writes a request and the trainer takes it; with the row
    // stranded on the laptop, the asking half works, the taking half works, and
    // no run ever crosses between them.
    //
    // Both carry `updated_at`, which is load-bearing rather than incidental: a
    // re-registration and a claim are both in-place mutations, so last-write-
    // wins has to order them. A claim that lost to a stale pending row would
    // hand the same run to a second trainer.
    TableSpec::new(
        tables::DEVICE_PROFILE,
        "updated_at",
        Replicate::Owned(|scope: &Scope, row| scope.is_own_user(row)),
    ),
    TableSpec::new(
        tables::GENESIS_REQUEST,
        "updated_at",
        Replicate::Owned(|scope: &Scope, row| scope.is_own_user(row)),
    ),
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Every table the sync replicates is one the schema defines: a renamed
    /// store table fails here by name rather than silently stopping its
    /// replication.
    #[test]
    fn every_replicated_table_is_one_the_schema_defines() {
        let defined: Vec<String> = antumbra_store::schema::tables(antumbra_store::EMBED_DIM as u32)
            .iter()
            .map(|t| t.name.clone())
            .collect();
        for spec in PENUMBRA_TABLES {
            assert!(
                defined.iter().any(|d| d == spec.name),
                "{} is replicated but not defined",
                spec.name
            );
        }
        assert!(PENUMBRA_TABLES.len() >= 6);
    }
}
