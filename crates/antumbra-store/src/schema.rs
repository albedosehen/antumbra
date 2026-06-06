//! Schema as code (ADR-0007), defined entirely with surql-rs schema builders —
//! no hand-authored SurrealQL. v0 keeps tables `SCHEMALESS` and leans on
//! explicit unique + HNSW indexes; the DDL is *generated* by surql-rs.
//!
//! The HNSW dimension is parameterized so tests use small vectors while
//! production uses the 384-d all-MiniLM-L6-v2 convention (the real candle
//! embedder in antumbra-serve).

use surql::schema::access::{record_access, AccessDefinition, RecordAccessConfig};
use surql::schema::table::{
    hnsw_index, index, table_schema, unique_index, HnswDistanceType, MTreeVectorType,
    TableDefinition, TableMode,
};
use surql::schema::{generate_access_sql_with_options, generate_table_sql};

use antumbra_core::Result;

use crate::error::map;

/// Default embedding dimension (all-MiniLM-L6-v2). ADR-0007.
pub const EMBED_DIM: usize = 384;

/// Permissions for the shared-population tables (the umbra: experts, the learned
/// router, boundaries). Any authenticated tenant session may READ them — the
/// brain is shared across tenants — but only the owner/root may WRITE (a record
/// session is denied; the rootful owner connection bypasses the clause).
/// `WHERE true` / `WHERE false` are the always / never predicates. Private
/// per-tenant data (`memory`, `memory_edge`) uses tenant-scoped permissions
/// instead; the rest of the tables are owner-internal (no record access).
const SHARED_POPULATION_PERMS: [(&str, &str); 4] = [
    ("select", "true"),
    ("create", "false"),
    ("update", "false"),
    ("delete", "false"),
];

/// Expert population permissions (ADR-0013/0014): a session reads a **shared**
/// expert (`owner = NONE`, the common umbra) or one it **owns** (a private
/// expert consolidated from its compartment); only the owner/root writes (a
/// record session is denied; the rootful owner connection bypasses).
const EXPERT_PERMS: [(&str, &str); 4] = [
    ("select", "owner = NONE OR owner = $auth.user"),
    ("create", "false"),
    ("update", "false"),
    ("delete", "false"),
];

/// Tenant-scoped permissions: any authenticated session in the tenant may
/// read/write the row (the engine still bars cross-tenant access).
const TENANT_PERMS: [(&str, &str); 4] = [
    ("select", "tenant_id = $auth.tenant"),
    ("create", "tenant_id = $auth.tenant"),
    ("update", "tenant_id = $auth.tenant"),
    ("delete", "tenant_id = $auth.tenant"),
];

/// The compartment-aware read rule for `memory` (the grant-graph ACL): a memory
/// is visible to a session when it is in the session's tenant AND either it is
/// un-compartmentalized (the shared tenant pool), OR its compartment is owned by
/// `$auth.user`, OR its compartment is granted to `$auth.user`. Evaluated by the
/// engine per row via subqueries over the (tenant-readable) compartment/grant
/// tables, so a forgotten app filter cannot leak and a revoke takes effect at
/// once.
const MEMORY_SELECT_RULE: &str = "tenant_id = $auth.tenant AND (compartment = NONE \
     OR compartment IN (SELECT VALUE key FROM compartment WHERE owner = $auth.user AND deleted_at IS NONE) \
     OR compartment IN (SELECT VALUE compartment FROM grant WHERE grantee = $auth.user AND deleted_at IS NONE))";

/// The link-capability gate for `memory_edge` create/update: you may create an
/// edge only when its *target* memory is in a compartment you may LINK into —
/// the shared pool (un-compartmentalized), a compartment you own, or one granted
/// to you with the `link` capability (not mere `reference`). Read visibility of
/// the target is already enforced when an edge is resolved back to a memory.
const EDGE_LINK_RULE: &str = "tenant_id = $auth.tenant AND to_id IN (SELECT VALUE key FROM memory \
     WHERE compartment = NONE \
     OR compartment IN (SELECT VALUE key FROM compartment WHERE owner = $auth.user AND deleted_at IS NONE) \
     OR compartment IN (SELECT VALUE compartment FROM grant WHERE grantee = $auth.user AND capability = 'link' AND deleted_at IS NONE))";

/// The full table set, built with surql-rs builders.
pub fn tables(embed_dim: u32) -> Vec<TableDefinition> {
    vec![
        // Expert population (umbra). ADR-0001/0013/0014. Shared experts read by
        // all; private experts read only by their owner; owner writes.
        table_schema("expert")
            .with_mode(TableMode::Schemaless)
            .with_permissions(EXPERT_PERMS)
            .with_indexes([
                unique_index("expert_key_uq", ["key"]),
                hnsw_index(
                    "expert_cap_hnsw",
                    "capability_vec",
                    embed_dim,
                    HnswDistanceType::Cosine,
                    MTreeVectorType::F32,
                    None,
                    None,
                ),
            ]),
        // Shadow lifecycle (penumbra). ADR-0002.
        table_schema("shadow")
            .with_mode(TableMode::Schemaless)
            .with_indexes([
                unique_index("shadow_key_uq", ["key"]),
                index("shadow_status_idx", ["status", "generation"]),
            ]),
        // Critic signal (verifier-first). ADR-0003.
        table_schema("reward_signal")
            .with_mode(TableMode::Schemaless)
            .with_indexes([index("reward_run_idx", ["run_id", "step_idx"])]),
        // Inhibitory store (antumbra / keystone). ADR-0004. Shared population.
        table_schema("failure_boundary")
            .with_mode(TableMode::Schemaless)
            .with_permissions(SHARED_POPULATION_PERMS)
            .with_indexes([
                unique_index("fb_key_uq", ["key"]),
                hnsw_index(
                    "fb_ctx_hnsw",
                    "context_vec",
                    embed_dim,
                    HnswDistanceType::Cosine,
                    MTreeVectorType::F32,
                    None,
                    None,
                ),
            ]),
        // Durable orchestration. ADR-0005.
        table_schema("orchestration_run")
            .with_mode(TableMode::Schemaless)
            .with_indexes([
                unique_index("orun_key_uq", ["key"]),
                index("orun_status_idx", ["status", "updated_at"]),
            ]),
        // Durable generational loop head (the checkpoint). ADR-0008.
        table_schema("generation_head").with_mode(TableMode::Schemaless),
        // Validation harness. ADR-0007.
        table_schema("evaluation_run")
            .with_mode(TableMode::Schemaless)
            .with_indexes([index("eval_subject_idx", ["subject_kind", "subject_id"])]),
        // Placement registry (inert until the fleet wakes). ADR-0006.
        table_schema("device_profile")
            .with_mode(TableMode::Schemaless)
            .with_indexes([index("device_host_idx", ["host", "backend"])]),
        // Learned router singleton (ADR-0009). Shared population: tenants read
        // it to route a task across the shared experts; the owner trains/writes
        // it. Previously auto-created (which defaulted to deny for record
        // sessions); defined here so the read permission is explicit.
        table_schema("learned_router")
            .with_mode(TableMode::Schemaless)
            .with_permissions(SHARED_POPULATION_PERMS),
        // Penumbra memory (the soft, editable consolidation source). Tenant-
        // isolated the way the data-plane design intends: a `tenant_id` on every
        // row and an engine-enforced row-level `PERMISSIONS` clause comparing it
        // to `$auth.tenant`, so a handler that forgets its filter still cannot
        // leak — the engine refuses the read. The repo also carries an explicit
        // `WHERE tenant_id = ...` as the documented second layer. The clause
        // becomes load-bearing once a per-tenant `ScopeCredentials` session binds
        // `$auth.tenant`; a rootful schema/migration session bypasses it.
        table_schema("memory")
            .with_mode(TableMode::Schemaless)
            .with_permissions([
                ("select", MEMORY_SELECT_RULE),
                ("create", "tenant_id = $auth.tenant"),
                ("update", "tenant_id = $auth.tenant"),
                ("delete", "tenant_id = $auth.tenant"),
            ])
            .with_indexes([
                unique_index("memory_tenant_key_uq", ["tenant_id", "key"]),
                index("memory_tenant_network_idx", ["tenant_id", "network"]),
                hnsw_index(
                    "memory_embedding_hnsw",
                    "embedding",
                    embed_dim,
                    HnswDistanceType::Cosine,
                    MTreeVectorType::F32,
                    None,
                    None,
                ),
            ]),
        // Penumbra graph: typed, directed edges between memories
        // (references/supersedes/contradicts/follows/caused). Tenant-isolated
        // like `memory` (engine-enforced PERMISSIONS + the repo's explicit
        // filter). A plain table keyed by from/to/type, queried by clean
        // equality (native N-hop RELATION traversal is a later enhancement).
        table_schema("memory_edge")
            .with_mode(TableMode::Schemaless)
            .with_permissions([
                ("select", "tenant_id = $auth.tenant"),
                ("create", EDGE_LINK_RULE),
                ("update", EDGE_LINK_RULE),
                ("delete", "tenant_id = $auth.tenant"),
            ])
            .with_indexes([
                index("memory_edge_from_idx", ["tenant_id", "from_id"]),
                index("memory_edge_to_idx", ["tenant_id", "to_id"]),
            ]),
        // Compartments (the latent-spaces). Tenant-readable so the memory ACL's
        // subqueries resolve; ownership/sharing is carried in the rows (owner +
        // the grant table) and enforced by the memory rule.
        table_schema("compartment")
            .with_mode(TableMode::Schemaless)
            .with_permissions(TENANT_PERMS)
            .with_indexes([
                unique_index("compartment_key_uq", ["tenant_id", "key"]),
                index("compartment_owner_idx", ["tenant_id", "owner"]),
            ]),
        // Capability grants (intra-tenant, user-to-user). Tenant-readable so the
        // memory ACL can resolve `grantee = $auth.user`.
        table_schema("grant")
            .with_mode(TableMode::Schemaless)
            .with_permissions(TENANT_PERMS)
            .with_indexes([
                index("grant_grantee_idx", ["tenant_id", "grantee"]),
                index("grant_compartment_idx", ["tenant_id", "compartment"]),
            ]),
        // Principals: one record per (tenant, user). The record-access SIGNIN
        // resolves a principal so `$auth` carries both `$auth.tenant` (the hard
        // isolation key) and `$auth.user` (the compartment-ownership / sharing
        // actor). Provisioned by the owner/root; read only by the SIGNIN.
        table_schema("principal")
            .with_mode(TableMode::Schemaless)
            .with_indexes([unique_index("principal_tenant_user_uq", ["tenant", "user"])]),
    ]
}

/// The record-access method that binds `$auth.tenant` for a session. Signing in
/// with a `tenant` variable (via `ScopeCredentials`) resolves the matching
/// principal; `$auth` becomes that record, so the `memory` table's row-level
/// `PERMISSIONS ... WHERE tenant_id = $auth.tenant` are enforced by the engine
/// for that session. Root/owner sessions (no signin) bypass the clause and see
/// across tenants.
pub fn tenant_access() -> AccessDefinition {
    record_access(
        "tenant",
        RecordAccessConfig::new()
            .with_signin("SELECT * FROM principal WHERE tenant = $tenant AND user = $user"),
    )
    .with_session("1h")
}

/// The record-access method name (the `ac` in a scope signin).
pub const TENANT_ACCESS: &str = "tenant";

/// Validate every table and render the idempotent (`IF NOT EXISTS`) DDL the
/// builders generate. The returned statements are surql-rs output, not
/// hand-authored SurrealQL. Table-level `PERMISSIONS` render inline on the
/// `DEFINE TABLE` statement (the surql-rs fix; published 0.2.7 emitted a
/// malformed `DEFINE FIELD PERMISSIONS ...`, patched via the workspace's local
/// checkout until released).
pub fn schema_statements(embed_dim: u32) -> Result<Vec<String>> {
    let mut out = Vec::new();
    for table in tables(embed_dim) {
        table.validate().map_err(map)?;
        out.extend(generate_table_sql(&table, true));
    }
    // The tenant record-access method (binds $auth.tenant), idempotent so a
    // persistent store can re-apply the schema on every connect.
    out.extend(generate_access_sql_with_options(&tenant_access(), true).map_err(map)?);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_table_carries_engine_enforced_tenant_permissions() {
        let stmts = schema_statements(EMBED_DIM as u32).unwrap();
        let define = stmts
            .iter()
            .find(|s| s.starts_with("DEFINE TABLE") && s.contains(" memory "))
            .expect("memory DEFINE TABLE statement");
        // The clause renders inline on DEFINE TABLE (valid SurrealQL), not as a
        // malformed `DEFINE FIELD PERMISSIONS ...` statement. (Actions are
        // emitted in BTreeMap order, so don't assume select is first.)
        assert!(define.contains("PERMISSIONS FOR"));
        assert!(define.contains("FOR select WHERE tenant_id = $auth.tenant"));
        assert!(define.contains("FOR create WHERE tenant_id = $auth.tenant"));
        assert!(define.contains("FOR update WHERE tenant_id = $auth.tenant"));
        assert!(define.contains("FOR delete WHERE tenant_id = $auth.tenant"));
        assert!(!stmts.iter().any(|s| s.contains("DEFINE FIELD PERMISSIONS")));
    }
}
