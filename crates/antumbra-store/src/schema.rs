//! Schema as code (ADR-0007), defined entirely with surql-rs schema builders —
//! no hand-authored SurrealQL. v0 keeps tables `SCHEMALESS` and leans on
//! explicit unique + HNSW indexes; the DDL is *generated* by surql-rs.
//!
//! The HNSW dimension is parameterized so tests use small vectors while
//! production uses the 384-d all-MiniLM-L6-v2 convention (the real candle
//! embedder in antumbra-serve).

use surql::schema::generate_table_sql;
use surql::schema::table::{
    hnsw_index, index, table_schema, unique_index, HnswDistanceType, MTreeVectorType,
    TableDefinition, TableMode,
};

use antumbra_core::Result;

use crate::error::map;

/// Default embedding dimension (all-MiniLM-L6-v2). ADR-0007.
pub const EMBED_DIM: usize = 384;

/// The full table set, built with surql-rs builders.
pub fn tables(embed_dim: u32) -> Vec<TableDefinition> {
    vec![
        // Expert population (umbra). ADR-0001.
        table_schema("expert")
            .with_mode(TableMode::Schemaless)
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
        // Inhibitory store (antumbra / keystone). ADR-0004.
        table_schema("failure_boundary")
            .with_mode(TableMode::Schemaless)
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
        // Penumbra memory (the soft, editable consolidation source). Tenant-
        // isolated by a `tenant_id` on every row, a composite unique index so
        // the same key can exist per tenant, an HNSW recall index, and a
        // per-tenant network index. The intended engine guard is a row-level
        // `PERMISSIONS ... WHERE tenant_id = $auth.tenant` clause, but it is NOT
        // emitted yet: oneiriq-surql 0.2.7 has no working renderer for table
        // PERMISSIONS — both `to_surql_all_with_options` and `generate_table_sql`
        // (a thin wrapper over it) emit a malformed `DEFINE FIELD PERMISSIONS
        // ...`. Engine enforcement is blocked on fixing that crate. Until then
        // isolation is the repo's explicit `WHERE tenant_id = ...` plus the
        // per-record tenant re-check on point reads.
        table_schema("memory")
            .with_mode(TableMode::Schemaless)
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
    ]
}

/// Validate every table and render the idempotent (`IF NOT EXISTS`) DDL the
/// builders generate. The returned statements are surql-rs output, not
/// hand-authored SurrealQL. Uses `generate_table_sql` (not the table's
/// `to_surql_all_with_options`) because only the former renders table-level
/// `PERMISSIONS` correctly on oneiriq-surql 0.2.7 — the latter emits a malformed
/// `DEFINE FIELD PERMISSIONS ...` (defect to report upstream).
pub fn schema_statements(embed_dim: u32) -> Result<Vec<String>> {
    let mut out = Vec::new();
    for table in tables(embed_dim) {
        table.validate().map_err(map)?;
        out.extend(generate_table_sql(&table, true));
    }
    Ok(out)
}
