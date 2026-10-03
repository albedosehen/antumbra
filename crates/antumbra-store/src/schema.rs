//! Schema as code on SurrealDB as the unified substrate, defined entirely with surql-rs schema builders;
//! no hand-authored SurrealQL. v0 keeps tables `SCHEMALESS` and leans on
//! explicit unique + HNSW indexes; the DDL is *generated* by surql-rs.
//!
//! The HNSW dimension is parameterized so tests use small vectors while
//! production uses the 384-d all-MiniLM-L6-v2 convention (the real candle
//! embedder in antumbra-serve).

use surql::schema::access::{record_access, AccessDefinition, RecordAccessConfig};

/// How long a record (tenant) session lives once signed in. A connection kept
/// signed in for longer answers every query with "the session has expired", so
/// the servers re-sign in well before this (`antumbra-mcp`'s session keeper).
pub const TENANT_SESSION: std::time::Duration = std::time::Duration::from_secs(60 * 60);
use surql::schema::table::{
    bm25_index, hnsw_index, index, table_schema, unique_index, HnswDistanceType, MTreeVectorType,
    TableDefinition, TableMode,
};
use surql::schema::{
    generate_access_sql_with_options, generate_analyzer_sql_with_options, generate_table_sql,
    standard_analyzer, AnalyzerDefinition,
};

use antumbra_core::Result;

use crate::error::map;

/// Default embedding dimension (all-MiniLM-L6-v2).
pub const EMBED_DIM: usize = 384;

/// The full-text analyzer name shared by the `memory` and `document_chunk` BM25
/// indexes (the lexical/sparse leg of hybrid recall). A generalist tokenizer +
/// case/ascii folding -- no stemming -- so exact tokens (code identifiers, error
/// codes, tickers) match, which is precisely what dense vectors silently drop.
pub const CONTENT_ANALYZER: &str = "antumbra_content";

/// The `DEFINE ANALYZER` definition for [`CONTENT_ANALYZER`].
fn content_analyzer() -> AnalyzerDefinition {
    standard_analyzer(CONTENT_ANALYZER)
}

/// Permissions for the shared-population tables (the umbra: experts, the learned
/// router, boundaries). Any authenticated tenant session may READ them (the
/// brain is shared across tenants), but only the owner/root may WRITE (a record
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

/// Expert population permissions (multi-tenant isolation across compartments, the latent-spaces of memory): a session reads a **shared**
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

/// Only the **owner** of the referenced compartment may write the row. The
/// compartment table is tenant-readable, so the subquery resolves for any session,
/// but it returns the compartment id only when `$auth.user` owns it.
const COMPARTMENT_OWNER_RULE: &str = "tenant_id = $auth.tenant AND compartment IN \
    (SELECT VALUE key FROM compartment WHERE owner = $auth.user AND deleted_at IS NONE)";

/// Grant-table permissions. Any tenant member may **read** grants (so the memory
/// rule's `grantee = $auth.user` subquery resolves), but only the **owner of the
/// compartment** may create/update/delete a grant on it. Without the owner check
/// any tenant member could forge a grant to another user's private compartment
/// (the `share_compartment` tool accepts an arbitrary compartment id and runs
/// under the caller's scoped session) and then read it; engine-enforced here so a
/// non-owner's `share`/`revoke` fails closed regardless of the app layer.
const GRANT_PERMS: [(&str, &str); 4] = [
    ("select", "tenant_id = $auth.tenant"),
    ("create", COMPARTMENT_OWNER_RULE),
    ("update", COMPARTMENT_OWNER_RULE),
    ("delete", COMPARTMENT_OWNER_RULE),
];

/// The compartment-aware read rule for `memory` (the grant-graph ACL): a memory
/// is visible to a session when it is in the session's tenant AND either it is
/// un-compartmentalized (the shared tenant pool), OR its compartment is owned by
/// `$auth.user`, OR its compartment is granted to `$auth.user`. Evaluated by the
/// engine per row via subqueries over the (tenant-readable) compartment/grant
/// tables, so a forgotten app filter cannot leak and a revoke takes effect at
/// once.
/// The active-hive read branch (ADR-0017 B), one more OR in the same shape as
/// the owner and grant subqueries beside it. A memory is hive-visible when its
/// compartment has an **accepted** offer, from a member who **opted in**, in a
/// tenant whose hive the owner **enabled**.
///
/// All three conditions, every time. Dropping any one of them would make the
/// hive something other than what ADR-0017 B decided: without `accepted` a
/// member publishes unilaterally, without `opted_in` an owner conscripts a
/// member's memory by accepting an offer they have since withdrawn consent for,
/// and without `enabled` a member publishes into an org that never opened one.
///
/// Read only. The write rule is deliberately untouched: the hive is a shared
/// read layer, so a member seeing another's offered compartment cannot write
/// into it. That asymmetry is intentional here, unlike the ones ADR-0017
/// increment 5 had to remove -- and the replication scope already handles it,
/// because a user's nodes carry what they own rather than what they can read.
const HIVE_VISIBLE_RULE: &str = "compartment IN (SELECT VALUE subject_id FROM hive_offer \
     WHERE tenant_id = $auth.tenant AND subject_kind = 'compartment' AND status = 'accepted' \
     AND offered_by IN (SELECT VALUE user FROM hive_membership WHERE tenant_id = $auth.tenant AND opted_in = true) \
     AND $auth.tenant IN (SELECT VALUE tenant_id FROM hive WHERE enabled = true))";

/// What a session may read without the hive: the shared pool, its own
/// compartments, and the ones granted to it.
const MEMORY_SELECT_WITHOUT_HIVE: &str = "tenant_id = $auth.tenant AND (compartment = NONE \
     OR compartment IN (SELECT VALUE key FROM compartment WHERE owner = $auth.user AND deleted_at IS NONE) \
     OR compartment IN (SELECT VALUE compartment FROM grant WHERE grantee = $auth.user AND deleted_at IS NONE)";

/// The read rule the `memory` and `document_chunk` tables carry: the private
/// rule above, with the active hive OR'd in and the group closed.
///
/// Composed rather than written out twice. Two copies of an ACL is how the copy
/// that matters stops matching the one that is read, and this one is read on
/// every recall in the system.
fn memory_select_rule() -> String {
    format!("{MEMORY_SELECT_WITHOUT_HIVE} OR {HIVE_VISIBLE_RULE})")
}

/// The write rule for `memory` (create/update). A session may write a memory only
/// into a compartment it may contribute to: the shared pool (un-compartmentalized),
/// a compartment it owns, or one granted to it with the **`link`** capability (not
/// mere `reference`, which is read-only). Without this, the create rule would only
/// check the tenant, letting any tenant member inject a memory into another user's
/// private compartment (which the owner would then see as their own). This mirrors
/// `EDGE_LINK_RULE` for the write side of the graph (the dual of `GRANT_PERMS`).
const MEMORY_WRITE_RULE: &str = "tenant_id = $auth.tenant AND (compartment = NONE \
     OR compartment IN (SELECT VALUE key FROM compartment WHERE owner = $auth.user AND deleted_at IS NONE) \
     OR compartment IN (SELECT VALUE compartment FROM grant WHERE grantee = $auth.user AND capability = 'link' AND deleted_at IS NONE))";

/// `document_chunk` follows the compartment rules a memory does, because a
/// document is shared the way a memory is: through its compartment. It used to
/// carry [`TENANT_PERMS`], under which every document in a tenant was readable by
/// every member, however privately it was ingested.
///
/// Delete takes the write rule, not the tenant-wide predicate `memory` keeps:
/// ingest deletes a title's previous generation before writing the next, so a
/// tenant-wide delete would let any member erase another's private document by
/// naming its title.
fn document_perms() -> [(&'static str, String); 4] {
    [
        ("select", memory_select_rule()),
        ("create", MEMORY_WRITE_RULE.to_string()),
        ("update", MEMORY_WRITE_RULE.to_string()),
        ("delete", MEMORY_WRITE_RULE.to_string()),
    ]
}

/// A device profile is a machine describing itself into its owner's fabric
/// (ADR-0017). Read is tenant-wide, because dispatch has to be able to find the
/// user's genesis node from whichever node is asking. Write is the user's own:
/// a node registers only the row keyed to the session running on it, so one
/// tenant member cannot re-declare another's laptop a trainer and have work
/// routed to it. The table was previously owner-only, under which a node could
/// never register itself at all.
const DEVICE_PERMS: [(&str, &str); 4] = [
    ("select", "tenant_id = $auth.tenant"),
    ("create", "tenant_id = $auth.tenant AND user = $auth.user"),
    ("update", "tenant_id = $auth.tenant AND user = $auth.user"),
    ("delete", "tenant_id = $auth.tenant AND user = $auth.user"),
];

/// A genesis request is one node in a user's fabric asking another to run what
/// it cannot (ADR-0017 A2). Same shape as [`DEVICE_PERMS`], and for the same
/// reason: the asking node and the node that takes the work are two machines of
/// one user, so `user = $auth.user` lets the trainer claim a request its own
/// laptop wrote while barring another tenant member from touching it.
const GENESIS_REQUEST_PERMS: [(&str, &str); 4] = [
    ("select", "tenant_id = $auth.tenant"),
    ("create", "tenant_id = $auth.tenant AND user = $auth.user"),
    ("update", "tenant_id = $auth.tenant AND user = $auth.user"),
    ("delete", "tenant_id = $auth.tenant AND user = $auth.user"),
];

/// The tenant's hive gate (ADR-0017 B). Any member reads whether the hive is
/// open, because the read rule they are subject to depends on it; nobody with a
/// record session writes it. `false` on all three writes is what makes the
/// owner's decision the owner's: a record session is denied outright, and the
/// rootful owner bypasses permissions entirely.
const HIVE_PERMS: [(&str, &str); 4] = [
    ("select", "tenant_id = $auth.tenant"),
    ("create", "false"),
    ("update", "false"),
    ("delete", "false"),
];

/// What a repository's manifests declared, as the server last read them from
/// GitHub (`repo::manifest_set`). Members read it; only the rootful owner
/// writes it, so nobody can forge a manifest into the dependency graph's
/// evidence.
const MANIFEST_SET_PERMS: [(&str, &str); 4] = [
    ("select", "tenant_id = $auth.tenant"),
    ("create", "false"),
    ("update", "false"),
    ("delete", "false"),
];

/// The member's hive gate. A member reads who has joined -- the read rule needs
/// it, and a hive whose membership were secret could not be audited by the
/// people in it -- and writes only their own row. So an owner cannot opt a
/// member in on their behalf, which is the half of the two-gate design that
/// protects the member.
const HIVE_MEMBERSHIP_PERMS: [(&str, &str); 4] = [
    ("select", "tenant_id = $auth.tenant"),
    ("create", "tenant_id = $auth.tenant AND user = $auth.user"),
    ("update", "tenant_id = $auth.tenant AND user = $auth.user"),
    ("delete", "tenant_id = $auth.tenant AND user = $auth.user"),
];

/// Offer and curation, engine-enforced, and the dual of [`GRANT_PERMS`].
///
/// A member creates and withdraws their own offers. **Update is `false` for
/// every record session**, and that is the whole curation boundary: the flip to
/// `accepted` is the only thing that puts a subject in the active hive, so
/// denying update to members means contribution is theirs and curation is the
/// owner's, enforced rather than agreed. Without it a member could accept their
/// own offer and publish into the org unilaterally.
const HIVE_OFFER_PERMS: [(&str, &str); 4] = [
    ("select", "tenant_id = $auth.tenant"),
    (
        "create",
        "tenant_id = $auth.tenant AND offered_by = $auth.user",
    ),
    ("update", "false"),
    (
        "delete",
        "tenant_id = $auth.tenant AND offered_by = $auth.user",
    ),
];

/// The link-capability gate for `memory_edge` create/update: you may create an
/// edge only when its *target* memory is in a compartment you may LINK into:
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
        // Expert population (umbra), the population of small frozen experts, with multi-tenant isolation across compartments. Shared experts read by
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
        // Shadow lifecycle (penumbra): shadow models hold the plasticity.
        table_schema("shadow")
            .with_mode(TableMode::Schemaless)
            .with_indexes([
                unique_index("shadow_key_uq", ["key"]),
                index("shadow_status_idx", ["status", "generation"]),
            ]),
        // Critic signal (verifier-first): the critic for credit assignment.
        table_schema("reward_signal")
            .with_mode(TableMode::Schemaless)
            .with_indexes([index("reward_run_idx", ["run_id", "step_idx"])]),
        // Inhibitory store (antumbra): the counterfactual boundary of competence. Shared population.
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
        // Durable orchestration (the boundary-conditioned gate, routing-as-retrieval).
        table_schema("orchestration_run")
            .with_mode(TableMode::Schemaless)
            .with_indexes([
                unique_index("orun_key_uq", ["key"]),
                index("orun_status_idx", ["status", "updated_at"]),
            ]),
        // Durable generational loop head (the generation_head checkpoint).
        table_schema("generation_head").with_mode(TableMode::Schemaless),
        // Out-of-band loop control (operator graceful-stop signal) for the durable generational loop.
        table_schema("loop_control").with_mode(TableMode::Schemaless),
        // Hosted-onboarding control plane and product surface. No PERMISSIONS clause, so
        // both default to deny for record/tenant sessions; only the control
        // plane's owner connection reads/writes them (an account row maps a login
        // to a tenant and must never be tenant-readable).
        table_schema("invite_code").with_mode(TableMode::Schemaless),
        table_schema("account").with_mode(TableMode::Schemaless),
        // Consumed magic-link ids (keyed by `jti`): the single-use ledger a
        // verify writes through, with an `expires_at` matching the link's so
        // rows self-expire out at the next sweep. Owner-only, like the rest of
        // the control-plane tables.
        table_schema("magic_link_use").with_mode(TableMode::Schemaless),
        // Expert lifecycle (ADR-0022 S-5): every status change, appended and
        // numbered per expert. Readable where its expert is, through the same
        // owner rule, so no session sees an expert without its moves.
        table_schema("expert_transition")
            .with_mode(TableMode::Schemaless)
            .with_permissions(EXPERT_PERMS)
            .with_indexes([unique_index("xtrans_seq_uq", ["expert", "seq"])]),
        // Each shared expert's leave-one-out contribution per measured
        // generation (ADR-0022 S-5), the history retirement reads. Readable
        // like the shared population it measures.
        table_schema("contribution")
            .with_mode(TableMode::Schemaless)
            .with_permissions(SHARED_POPULATION_PERMS)
            .with_indexes([index("contribution_expert_idx", ["expert", "generation"])]),
        // The population against its single best expert, per measured
        // generation of a run (ADR-0022 S-5): the rolling comparison that is
        // reported whether or not it flatters the architecture.
        table_schema("population_baseline")
            .with_mode(TableMode::Schemaless)
            .with_permissions(SHARED_POPULATION_PERMS)
            .with_indexes([index("baseline_run_idx", ["run_id", "generation"])]),
        // Each live task's clear winner among the experts, from the latest
        // contribution measurement that scored them all (ADR-0024 D-1): what
        // the learned router trains on beside the capability exemplars.
        table_schema("routing_outcome")
            .with_mode(TableMode::Schemaless)
            .with_permissions(SHARED_POPULATION_PERMS)
            .with_indexes([index("routing_outcome_winner_idx", ["winner"])]),
        // The grow step (ADR-0022 S-3): the region census each contribution
        // measurement takes, and the decision each generation made from it.
        table_schema("region_census")
            .with_mode(TableMode::Schemaless)
            .with_indexes([index("census_run_idx", ["run_id", "generation"])]),
        table_schema("grow")
            .with_mode(TableMode::Schemaless)
            .with_indexes([index("grow_run_idx", ["run_id", "generation"])]),
        // The verifier namespace (ADR-0022 S-4): verifiers keyed by content
        // address, and the append-only measurements and state changes that
        // decide whether each may grant reward. No permissions clause: owner
        // only, so no tenant session and nothing that trains can write one.
        table_schema("verifier")
            .with_mode(TableMode::Schemaless)
            .with_indexes([unique_index("verifier_key_uq", ["key"])]),
        table_schema("verifier_transition")
            .with_mode(TableMode::Schemaless)
            .with_indexes([unique_index("vtrans_seq_uq", ["verifier", "seq"])]),
        table_schema("verifier_measurement")
            .with_mode(TableMode::Schemaless)
            .with_indexes([unique_index("vmeasure_seq_uq", ["verifier", "seq"])]),
        // Validation harness.
        table_schema("evaluation_run")
            .with_mode(TableMode::Schemaless)
            .with_indexes([index("eval_subject_idx", ["subject_kind", "subject_id"])]),
        // The recipe search (ADR-0022 S-1): one row per shadow's training
        // recipe, read back by run in generation order.
        table_schema("recipe")
            .with_mode(TableMode::Schemaless)
            .with_indexes([index("recipe_run_idx", ["run_id", "generation"])]),
        // A user's fabric: which of their machines an agent is running on, and
        // which one of them can train (ADR-0017). Keyed per (tenant, user, host),
        // so the user index is what dispatch looks their genesis node up by.
        table_schema("device_profile")
            .with_mode(TableMode::Schemaless)
            .with_permissions(DEVICE_PERMS)
            .with_indexes([
                index("device_host_idx", ["host", "backend"]),
                index("device_user_idx", ["user", "role"]),
            ]),
        // The tenant hive (ADR-0017 B): two gates and an offer ledger. One hive
        // row per tenant, so the gate cannot be ambiguous.
        table_schema("hive")
            .with_mode(TableMode::Schemaless)
            .with_permissions(HIVE_PERMS)
            .with_indexes([unique_index("hive_tenant_uq", ["tenant_id"])]),
        table_schema("hive_membership")
            .with_mode(TableMode::Schemaless)
            .with_permissions(HIVE_MEMBERSHIP_PERMS)
            .with_indexes([unique_index("hive_member_uq", ["tenant_id", "user"])]),
        table_schema("hive_offer")
            .with_mode(TableMode::Schemaless)
            .with_permissions(HIVE_OFFER_PERMS)
            .with_indexes([index(
                "hive_offer_idx",
                ["tenant_id", "subject_kind", "subject_id"],
            )]),
        // What a node could not run itself, left where the machine that can will
        // find it (ADR-0017 A2). Indexed by the pair a trainer looks it up on.
        table_schema("genesis_request")
            .with_mode(TableMode::Schemaless)
            .with_permissions(GENESIS_REQUEST_PERMS)
            .with_indexes([index("genesis_user_idx", ["user", "status"])]),
        // Each repository's manifests as last read through the GitHub App, one
        // row per (tenant, repository), so the declared edges into and out of
        // one repository can be worked out again from every other's last
        // reading (ADR-0019).
        table_schema("manifest_set")
            .with_mode(TableMode::Schemaless)
            .with_permissions(MANIFEST_SET_PERMS)
            .with_indexes([index("manifest_set_tenant_idx", ["tenant_id"])]),
        // Learned router singleton (the learned gate). Shared population: tenants read
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
        // leak; the engine refuses the read. The repo also carries an explicit
        // `WHERE tenant_id = ...` as the documented second layer. The clause
        // becomes load-bearing once a per-tenant `ScopeCredentials` session binds
        // `$auth.tenant`; a rootful schema/migration session bypasses it.
        table_schema("memory")
            .with_mode(TableMode::Schemaless)
            .with_permissions([
                ("select", memory_select_rule()),
                ("create", MEMORY_WRITE_RULE.to_string()),
                ("update", MEMORY_WRITE_RULE.to_string()),
                ("delete", "tenant_id = $auth.tenant".to_string()),
            ])
            .with_indexes([
                unique_index("memory_tenant_key_uq", ["tenant_id", "key"]),
                index("memory_tenant_network_idx", ["tenant_id", "network"]),
                // Supports the collector's incremental watermark filter
                // (`updated_at > since`) as a range scan (R-1).
                index("memory_updated_at_idx", ["updated_at"]),
                // Bounds the GC purge to a range scan over actual tombstones
                // (`deleted_at IS NOT NONE AND deleted_at < cutoff`) instead of a
                // full-table read.
                index("memory_deleted_at_idx", ["deleted_at"]),
                hnsw_index(
                    "memory_embedding_hnsw",
                    "embedding",
                    embed_dim,
                    HnswDistanceType::Cosine,
                    MTreeVectorType::F32,
                    None,
                    None,
                ),
                // BM25 full-text over content: the sparse leg of hybrid recall,
                // fused with the HNSW dense leg by Reciprocal Rank Fusion.
                bm25_index("memory_content_fts", ["content"], CONTENT_ANALYZER),
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
                index("memory_edge_created_at_idx", ["created_at"]),
            ]),
        // Knowledge documents (P-3): a document's embedded chunks, a distinct type
        // from episodic `memory` but isolated the same way (the compartment rule
        // in the engine + the repo's tenant filter). HNSW-indexed for semantic
        // recall over reference material.
        table_schema("document_chunk")
            .with_mode(TableMode::Schemaless)
            .with_permissions(document_perms())
            .with_indexes([
                index("document_chunk_tenant_title_idx", ["tenant_id", "title"]),
                index("document_chunk_created_at_idx", ["created_at"]),
                hnsw_index(
                    "document_chunk_embedding_hnsw",
                    "embedding",
                    embed_dim,
                    HnswDistanceType::Cosine,
                    MTreeVectorType::F32,
                    None,
                    None,
                ),
                // BM25 full-text over content (sparse leg of hybrid recall).
                bm25_index("document_chunk_content_fts", ["content"], CONTENT_ANALYZER),
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
                index("compartment_updated_at_idx", ["updated_at"]),
            ]),
        // Capability grants (intra-tenant, user-to-user). Tenant-readable so the
        // memory ACL can resolve `grantee = $auth.user`; only the compartment's
        // owner may write a grant on it (GRANT_PERMS) so a grant cannot be forged.
        table_schema("grant")
            .with_mode(TableMode::Schemaless)
            .with_permissions(GRANT_PERMS)
            .with_indexes([
                index("grant_grantee_idx", ["tenant_id", "grantee"]),
                index("grant_compartment_idx", ["tenant_id", "compartment"]),
                index("grant_updated_at_idx", ["updated_at"]),
            ]),
        // Principals: one record per (tenant, user). The record-access SIGNIN
        // resolves a principal so `$auth` carries both `$auth.tenant` (the hard
        // isolation key) and `$auth.user` (the compartment-ownership / sharing
        // actor). Provisioned by the owner/root; read only by the SIGNIN.
        table_schema("principal")
            .with_mode(TableMode::Schemaless)
            .with_indexes([unique_index("principal_tenant_user_uq", ["tenant", "user"])]),
        // Per-workspace embedder endpoint (hosted multi-tenant): one row per
        // tenant, tenant-scoped so a workspace sets only its own.
        table_schema("embedder_config")
            .with_mode(TableMode::Schemaless)
            .with_permissions(TENANT_PERMS)
            .with_indexes([unique_index("embedder_config_tenant_uq", ["tenant_id"])]),
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
    .with_session(format!("{}s", TENANT_SESSION.as_secs()))
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
    // The full-text analyzer must be defined before any FULLTEXT index that
    // references it (the `memory` / `document_chunk` BM25 indexes), so emit it
    // first. Idempotent (`IF NOT EXISTS`), re-applied on every connect.
    out.extend(generate_analyzer_sql_with_options(&content_analyzer(), true).map_err(map)?);
    let tables = tables(embed_dim);
    for table in &tables {
        table.validate().map_err(map)?;
        out.extend(generate_table_sql(table, true));
    }
    // `IF NOT EXISTS` skips a table that already exists, so on a database that
    // predates a change to a table's PERMISSIONS the old rule would stay in force
    // forever: a tightened rule would pass every test on a fresh store and never
    // reach a deployed one. Re-assert each table's own definition. This is the
    // `DEFINE TABLE` statement alone (mode, permissions), not its fields or
    // indexes, so no index is rebuilt and no row is touched.
    out.extend(tables.iter().map(TableDefinition::to_surql_overwrite));
    // The tenant record-access method (binds $auth.tenant), idempotent so a
    // persistent store can re-apply the schema on every connect.
    out.extend(generate_access_sql_with_options(&tenant_access(), true).map_err(map)?);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tenant_access_renders_its_session_duration() {
        let sql = generate_access_sql_with_options(&tenant_access(), true).unwrap();
        let rendered = sql.join("\n");
        assert!(rendered.contains("FOR SESSION 3600s"), "{rendered}");
    }

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
