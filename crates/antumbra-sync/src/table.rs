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
        Self { name, version_field }
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
];
