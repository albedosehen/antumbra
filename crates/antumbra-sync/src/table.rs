//! Which tables the collector replicates, the field on each that carries its
//! version (the last-write-wins tiebreaker), which of its rows can be carried
//! at all, and what a user-scoped collector carries from it.

use serde_json::Value;

use antumbra_store::repo::{expert, lifecycle};
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
    /// Which of the table's rows can be carried at all, as owner as well as
    /// under a scope. `None` is every row. A pass pairs a row on two stores by
    /// its record id, so a table that can hold a row whose record id is not
    /// the same on every store names the rows that can be paired here.
    pub carried: Option<fn(&Value) -> bool>,
}

impl TableSpec {
    const fn new(name: &'static str, version_field: &'static str, replicate: Replicate) -> Self {
        Self {
            name,
            version_field,
            replicate,
            carried: None,
        }
    }

    /// This table, carrying only the rows `carried` admits.
    const fn carrying(self, carried: fn(&Value) -> bool) -> Self {
        Self {
            name: self.name,
            version_field: self.version_field,
            replicate: self.replicate,
            carried: Some(carried),
        }
    }

    /// Whether a pass may carry `row`, whoever it runs as.
    pub fn carries(&self, row: &Value) -> bool {
        self.carried.is_none_or(|carried| carried(row))
    }
}

/// The replicated tables, in dependency order: a compartment and its grants
/// must exist before the memories whose ACL resolves against them, so they
/// replicate first. An expert's row is carried and its adapter is not: the
/// adapter is a large file on disk, which a server pulls for itself.
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
    // An expert's row, so a store that syncs learns an expert was trained:
    // what it is, whose it is, the machine that trained it (`placed_on`) and
    // the seal of its adapter. The adapter is a file and stays where it is. A
    // server that reads the row does not take the adapter for its own; it
    // pulls it from the server that trained it, or from the archive, when it
    // first serves the expert, and checks it against that seal.
    //
    // `updated_at` is set by the store on every write and only goes up, and a
    // deleted expert keeps its record as a tombstone, so a retraining and a
    // delete both cross as a newer version of one record.
    //
    // Only a row in its expert's own record is carried. One still under an
    // engine-assigned id, in a store written into by a server from before
    // records were named for their expert, is a different record on every
    // store and would land as a second one for an expert the other has.
    // Opening a store moves such rows, so the next connection carries them.
    //
    // After every table that was carried before it, though it needs only
    // `compartment`: nothing above resolves against an expert's row, and a
    // fault in a newer table then leaves the older ones reconciled.
    //
    // Under a scope a pass offers the user's standing experts, and the engine
    // refuses them: only an owner connection writes this table. That is
    // counted and said, as any refusal on a narrowed table is.
    TableSpec::new(
        tables::EXPERT,
        "updated_at",
        Replicate::Owned(|scope: &Scope, row| scope.holds_expert(row)),
    )
    .carrying(expert::is_own_record),
    // An expert's moves: where it stands. Status is not on the expert's row;
    // it is the last of these. A store that had the row and not the moves
    // read every expert as active, whatever the store that trained it had
    // decided since: it routed to a demoted expert and served an archived one.
    //
    // A move is written once and never rewritten or deleted, so the time it
    // was made, `at`, is its version, and it needs no tombstone. Its record is
    // named for its expert and its number in that expert's history, so two
    // stores that each made a move under one number while apart hold the
    // same record, and the later of the two wins on both.
    //
    // Only a move in its own record is carried, as with the expert: one still
    // under an engine-assigned id is left where it is until its store is
    // opened again, which moves it.
    //
    // After `expert`, whose row a move is about. Under a scope a pass offers
    // the moves of the user's own experts, and the engine refuses them as it
    // refuses the expert: only an owner connection writes either table.
    TableSpec::new(
        tables::EXPERT_TRANSITION,
        "at",
        Replicate::Owned(|scope: &Scope, row| scope.owns_move(row)),
    )
    .carrying(lifecycle::is_own_record),
];

#[cfg(test)]
mod tests {
    use super::*;
    use antumbra_core::Result;

    /// Every table the sync replicates is one the schema defines: a renamed
    /// store table fails here by name rather than silently stopping its
    /// replication.
    #[test]
    fn every_replicated_table_is_one_the_schema_defines() -> Result<()> {
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
        assert!(PENUMBRA_TABLES.len() >= 8);
        Ok(())
    }

    /// The expert's two tables are the ones whose rows are not all carried.
    /// The expert comes after the compartment a standing expert names, and
    /// its moves after the expert they are about.
    #[test]
    fn only_the_experts_tables_name_the_rows_they_carry() -> Result<()> {
        let position = |name: &str| PENUMBRA_TABLES.iter().position(|spec| spec.name == name);
        let narrowed: Vec<&str> = PENUMBRA_TABLES
            .iter()
            .filter(|spec| spec.carried.is_some())
            .map(|spec| spec.name)
            .collect();
        assert_eq!(narrowed, vec![tables::EXPERT, tables::EXPERT_TRANSITION]);
        assert!(position(tables::COMPARTMENT) < position(tables::EXPERT));
        assert!(position(tables::EXPERT) < position(tables::EXPERT_TRANSITION));
        assert_eq!(
            position(tables::EXPERT_TRANSITION),
            Some(PENUMBRA_TABLES.len() - 1)
        );

        // A table that names none carries any row.
        let memory = PENUMBRA_TABLES
            .iter()
            .find(|spec| spec.name == tables::MEMORY)
            .ok_or_else(|| antumbra_core::AntumbraError::other("memory is not replicated"))?;
        assert!(memory.carries(&serde_json::json!({ "id": "memory:x" })));
        Ok(())
    }
}
