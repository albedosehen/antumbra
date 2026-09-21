//! Which tables the collector replicates, and the field on each that carries its
//! version (the last-write-wins tiebreaker).

/// A table to replicate and the row field whose RFC3339 timestamp orders writes
/// for last-write-wins. `memory` is mutated in place (reinforce, consolidate),
/// so its `updated_at` is load-bearing; the rest are effectively write-once, so
/// `created_at` is their version.
#[derive(Debug, Clone, Copy)]
pub struct TableSpec {
    pub name: &'static str,
    pub version_field: &'static str,
}

impl TableSpec {
    const fn new(name: &'static str, version_field: &'static str) -> Self {
        Self {
            name,
            version_field,
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
    TableSpec::new("compartment", "updated_at"),
    // grant carries `updated_at` (bumped on revoke) so a revocation out-versions a
    // stale live grant and propagates under LWW.
    TableSpec::new("grant", "updated_at"),
    TableSpec::new("memory", "updated_at"),
    TableSpec::new("memory_edge", "created_at"),
    // The fabric itself (ADR-0017 A2). These two are what make a user's nodes
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
    TableSpec::new("device_profile", "updated_at"),
    TableSpec::new("genesis_request", "updated_at"),
];
