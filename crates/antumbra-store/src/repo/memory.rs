//! Penumbra memory repository: tenant-isolated traces (the consolidation
//! source). ADR-0004/0009.
//!
//! Isolation is engine-enforced: the `memory` table carries a row-level
//! `PERMISSIONS ... WHERE tenant_id = $auth.tenant` clause, so once a per-tenant
//! `ScopeCredentials` session binds `$auth.tenant`, the engine refuses any read
//! whose `tenant_id` does not match; a forgotten filter cannot leak. This repo
//! *also* applies an explicit `WHERE tenant_id = ...` as the documented second
//! layer (defense-in-depth), and re-checks `tenant_id` on a point read by id (so
//! another tenant's trace reads as absent). Built on surql-rs builders + `crud`;
//! no raw SurrealQL.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use surql::query::builder::Query;
use surql::query::crud::{delete_records, get_record, merge_record, query_records, upsert_record};
use surql::query::expressions::{field, value};
use surql::query::helpers::VectorDistanceType;
use surql::types::operators::{and_, eq, is_none};
use surql::types::RecordID;

use antumbra_core::{
    CompartmentId, ExpertId, Memory, MemoryId, MemoryNetwork, MemoryStatus, Result, TenantId,
    UserId,
};

use crate::dto::parse_dt;
use crate::error::map;
use crate::store::Store;

const TABLE: &str = "memory";

#[derive(Serialize, Deserialize)]
struct MemoryRow {
    key: String,
    tenant_id: String,
    network: MemoryNetwork,
    content: String,
    #[serde(default)]
    embedding: Option<Vec<f32>>,
    #[serde(default)]
    confidence: f32,
    #[serde(default)]
    reinforcement: u32,
    #[serde(default)]
    evidence: Vec<String>,
    #[serde(default)]
    volatile: bool,
    #[serde(default)]
    consolidated_expert: Option<String>,
    // Absent (not null) when None so the engine sees `compartment = NONE` for
    // un-compartmentalized memories (the shared tenant pool).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    compartment: Option<String>,
    #[serde(default)]
    author: Option<String>,
    #[serde(default)]
    author_host: Option<String>,
    #[serde(default)]
    status: MemoryStatus,
    created_at: String,
    updated_at: String,
    // Tombstone marker. Absent (NONE) for a live trace so read paths can test it
    // cheaply; an RFC3339 timestamp once forgotten.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    deleted_at: Option<String>,
}

impl MemoryRow {
    fn from_domain(m: &Memory) -> Self {
        MemoryRow {
            key: m.id.as_str().to_string(),
            tenant_id: m.tenant.as_str().to_string(),
            network: m.network,
            content: m.content.clone(),
            embedding: m.embedding.clone(),
            confidence: m.confidence,
            reinforcement: m.reinforcement,
            evidence: m.evidence.clone(),
            volatile: m.volatile,
            consolidated_expert: m
                .consolidated_expert
                .as_ref()
                .map(|e| e.as_str().to_string()),
            compartment: m.compartment.as_ref().map(|c| c.as_str().to_string()),
            author: m.author.as_ref().map(|u| u.as_str().to_string()),
            author_host: m.author_host.clone(),
            status: m.status,
            created_at: m.created_at.to_rfc3339(),
            updated_at: m.updated_at.to_rfc3339(),
            deleted_at: m.deleted_at.map(|t| t.to_rfc3339()),
        }
    }

    fn into_domain(self) -> Result<Memory> {
        Ok(Memory {
            id: MemoryId::new(self.key),
            tenant: TenantId::new(self.tenant_id),
            network: self.network,
            content: self.content,
            embedding: self.embedding,
            confidence: self.confidence,
            reinforcement: self.reinforcement,
            evidence: self.evidence,
            volatile: self.volatile,
            consolidated_expert: self.consolidated_expert.map(ExpertId::new),
            compartment: self.compartment.map(CompartmentId::new),
            author: self.author.map(UserId::new),
            author_host: self.author_host,
            status: self.status,
            created_at: parse_dt(&self.created_at)?,
            updated_at: parse_dt(&self.updated_at)?,
            deleted_at: self.deleted_at.as_deref().map(parse_dt).transpose()?,
        })
    }
}

/// Insert or replace a memory (addressed by its globally-unique id).
pub async fn upsert(store: &Store, memory: &Memory) -> Result<()> {
    let id = RecordID::<()>::new(TABLE, memory.id.as_str()).map_err(map)?;
    let data: Value = serde_json::to_value(MemoryRow::from_domain(memory))?;
    upsert_record(store.client(), &id, data)
        .await
        .map_err(map)?;
    Ok(())
}

/// Fetch one memory by id, but only if it belongs to `tenant`; a trace owned by
/// another tenant reads as absent (the defensive isolation re-check).
pub async fn get(store: &Store, tenant: &TenantId, id: &MemoryId) -> Result<Option<Memory>> {
    let rid = RecordID::<()>::new(TABLE, id.as_str()).map_err(map)?;
    match get_record(store.client(), &rid).await.map_err(map)? {
        Some(value) => {
            let row: MemoryRow = serde_json::from_value(value)?;
            // Another tenant's trace, or a tombstone, reads as absent.
            if row.tenant_id != tenant.as_str() || row.deleted_at.is_some() {
                return Ok(None);
            }
            Ok(Some(row.into_domain()?))
        }
        None => Ok(None),
    }
}

/// All of a tenant's memories (tenant-filtered).
pub async fn list(store: &Store, tenant: &TenantId) -> Result<Vec<Memory>> {
    let query = Query::new()
        .select(None)
        .from_table(TABLE)
        .map_err(map)?
        .where_(eq("tenant_id", tenant.as_str()));
    let rows: Vec<MemoryRow> = query_records(store.client(), &query).await.map_err(map)?;
    rows.into_iter()
        .filter(|r| r.deleted_at.is_none()) // hide tombstones (forgotten traces)
        .map(MemoryRow::into_domain)
        .collect()
}

/// Every memory across all tenants, with NO tenant filter. As an owner/root
/// session this is the cross-tenant view (profiling / training across tenants);
/// as a tenant-authenticated session the engine's row-level PERMISSIONS still
/// scope the result to that tenant, which is precisely the engine-enforcement
/// guarantee (isolation holds even with no app-side WHERE).
pub async fn all_unscoped(store: &Store) -> Result<Vec<Memory>> {
    let query = Query::new().select(None).from_table(TABLE).map_err(map)?;
    let rows: Vec<MemoryRow> = query_records(store.client(), &query).await.map_err(map)?;
    rows.into_iter()
        .filter(|r| r.deleted_at.is_none()) // hide tombstones (forgotten traces)
        .map(MemoryRow::into_domain)
        .collect()
}

/// A compartment's memories (the corpus for per-compartment consolidation,
/// ADR-0014/0012). Tenant + compartment filtered; the engine ACL also applies
/// under a tenant session.
pub async fn list_by_compartment(
    store: &Store,
    tenant: &TenantId,
    compartment: &CompartmentId,
) -> Result<Vec<Memory>> {
    let query = Query::new()
        .select(None)
        .from_table(TABLE)
        .map_err(map)?
        .where_(and_(
            eq("tenant_id", tenant.as_str()),
            eq("compartment", compartment.as_str()),
        ));
    let rows: Vec<MemoryRow> = query_records(store.client(), &query).await.map_err(map)?;
    rows.into_iter()
        .filter(|r| r.deleted_at.is_none()) // hide tombstones (forgotten traces)
        .map(MemoryRow::into_domain)
        .collect()
}

/// A tenant's memories in one network (tenant + network filtered).
pub async fn list_by_network(
    store: &Store,
    tenant: &TenantId,
    network: MemoryNetwork,
) -> Result<Vec<Memory>> {
    let query = Query::new()
        .select(None)
        .from_table(TABLE)
        .map_err(map)?
        .where_(and_(
            eq("tenant_id", tenant.as_str()),
            eq("network", network.as_str()),
        ));
    let rows: Vec<MemoryRow> = query_records(store.client(), &query).await.map_err(map)?;
    rows.into_iter()
        .filter(|r| r.deleted_at.is_none()) // hide tombstones (forgotten traces)
        .map(MemoryRow::into_domain)
        .collect()
}

/// Semantic recall: the `k` nearest memories to `query` *within* `tenant` (and
/// optionally one network). The tenant filter is ANDed onto the KNN clause so
/// the candidate set never crosses a tenant boundary.
pub async fn recall(
    store: &Store,
    tenant: &TenantId,
    query: &[f32],
    k: usize,
    network: Option<MemoryNetwork>,
) -> Result<Vec<Memory>> {
    let vector: Vec<f64> = query.iter().map(|&x| f64::from(x)).collect();
    let condition = match network {
        Some(net) => and_(
            eq("tenant_id", tenant.as_str()),
            eq("network", net.as_str()),
        ),
        None => eq("tenant_id", tenant.as_str()),
    };
    let q = Query::new()
        .select(None)
        .from_table(TABLE)
        .map_err(map)?
        .where_(condition)
        .vector_search(
            "embedding",
            vector,
            k as i64,
            VectorDistanceType::Cosine,
            None,
        )
        .map_err(map)?;
    let rows: Vec<MemoryRow> = query_records(store.client(), &q).await.map_err(map)?;
    rows.into_iter()
        .filter(|r| r.deleted_at.is_none()) // hide tombstones (forgotten traces)
        .map(MemoryRow::into_domain)
        .collect()
}

/// Merge only the named fields of a memory row (a partial `UPDATE ... MERGE`),
/// leaving every other field as the engine already holds it. The read-modify-
/// write mutators below write *just* the fields they change this way rather than
/// rewriting the whole row, so a concurrent forget's `deleted_at` (set between
/// the read and this write) is never clobbered back to live: the resurrection-
/// safe alternative to a full-row upsert. The engine PERMISSIONS still gate the
/// update, exactly as the upsert did.
async fn merge_fields(store: &Store, id: &MemoryId, patch: Value) -> Result<()> {
    let rid = RecordID::<()>::new(TABLE, id.as_str()).map_err(map)?;
    merge_record(store.client(), &rid, patch)
        .await
        .map_err(map)?;
    Ok(())
}

/// Reinforce a trace (recurrence + confidence bump), tenant-checked. Returns the
/// updated memory, or `None` if it does not exist for this tenant or has been
/// forgotten.
///
/// One atomic, tombstone-guarded statement rather than a read-modify-write: the
/// increment and the confidence bump are computed server-side from the row's
/// *current* values (`reinforcement = reinforcement + 1`), so concurrent
/// reinforcements cannot lose an increment, and the `deleted_at IS NONE` guard
/// means a trace forgotten between any read and this write simply matches no row
/// -- it can never be resurrected. Mirrors `Memory::reinforce`; the confidence
/// formula is self-bounding for confidence in `[0, 1]`, so no clamp is needed.
pub async fn reinforce(
    store: &Store,
    tenant: &TenantId,
    id: &MemoryId,
    now: chrono::DateTime<chrono::Utc>,
) -> Result<Option<Memory>> {
    let target = RecordID::<()>::new(TABLE, id.as_str())
        .map_err(map)?
        .to_string();
    // confidence + (1 - confidence) * 0.25
    let confidence_bump = field("confidence") + (value(1) - field("confidence")) * 0.25;
    let q = Query::new()
        .update_set(target)
        .map_err(map)?
        .set_expr("reinforcement", field("reinforcement") + 1)
        .map_err(map)?
        .set_expr("confidence", confidence_bump)
        .map_err(map)?
        .set("updated_at", now.to_rfc3339())
        .map_err(map)?
        .where_(and_(
            eq("tenant_id", tenant.as_str()),
            is_none("deleted_at"),
        ))
        .return_after();
    let rows: Vec<MemoryRow> = query_records(store.client(), &q).await.map_err(map)?;
    rows.into_iter()
        .next()
        .map(MemoryRow::into_domain)
        .transpose()
}

/// Record that a trace graduated into the umbra as `expert`, tenant-checked.
pub async fn mark_consolidated(
    store: &Store,
    tenant: &TenantId,
    id: &MemoryId,
    expert: ExpertId,
    now: chrono::DateTime<chrono::Utc>,
) -> Result<Option<Memory>> {
    match get(store, tenant, id).await? {
        Some(mut m) => {
            m.mark_consolidated(expert, now);
            merge_fields(
                store,
                id,
                serde_json::json!({
                    "consolidated_expert": m.consolidated_expert.as_ref().map(ExpertId::as_str),
                    "updated_at": m.updated_at.to_rfc3339(),
                }),
            )
            .await?;
            Ok(Some(m))
        }
        None => Ok(None),
    }
}

/// Forget a trace as a **tombstone** (the deletion that propagates and routes):
/// mark it `deleted_at = now` and persist, so read paths hide it while sync (R-1)
/// and live propagation (R-2) carry the deletion to other replicas/grantees
/// instead of it resurfacing. Tenant-checked via `get`; a no-op (returns `None`)
/// if the trace is absent or already a tombstone. Use [`purge`] to hard-remove
/// tombstones once the propagation grace window has passed.
pub async fn soft_delete(
    store: &Store,
    tenant: &TenantId,
    id: &MemoryId,
    now: chrono::DateTime<chrono::Utc>,
) -> Result<Option<Memory>> {
    match get(store, tenant, id).await? {
        Some(mut m) => {
            m.soft_delete(now);
            merge_fields(
                store,
                id,
                serde_json::json!({
                    "deleted_at": m.deleted_at.map(|t| t.to_rfc3339()),
                    "updated_at": m.updated_at.to_rfc3339(),
                }),
            )
            .await?;
            Ok(Some(m))
        }
        None => Ok(None),
    }
}

/// Hard-remove tombstones forgotten before `older_than` (the grace window), so
/// they do not accumulate forever. Run it on a cadence with a window wider than
/// the sync interval, so every replica has seen the tombstone before it is
/// purged (resurrection-safe garbage collection). Returns how many were purged.
pub async fn purge(store: &Store, older_than: chrono::DateTime<chrono::Utc>) -> Result<usize> {
    let query = Query::new().select(None).from_table(TABLE).map_err(map)?;
    let rows: Vec<MemoryRow> = query_records(store.client(), &query).await.map_err(map)?;
    let mut purged = 0;
    for row in rows {
        let Some(ts) = row.deleted_at.as_deref() else {
            continue; // live trace
        };
        if parse_dt(ts).map(|t| t < older_than).unwrap_or(false) {
            let condition = and_(
                eq("key", row.key.as_str()),
                eq("tenant_id", row.tenant_id.as_str()),
            );
            delete_records(store.client(), TABLE, Some(&condition))
                .await
                .map_err(map)?;
            purged += 1;
        }
    }
    Ok(purged)
}

/// Hard-delete a trace, but only within the caller's tenant (the `tenant_id`
/// predicate is ANDed onto the key, so no cross-tenant delete). Bypasses the
/// tombstone path -- prefer [`soft_delete`] for user-facing forgets so the
/// deletion propagates; this is for purges and internal cleanup.
pub async fn delete(store: &Store, tenant: &TenantId, id: &MemoryId) -> Result<()> {
    let condition = and_(eq("key", id.as_str()), eq("tenant_id", tenant.as_str()));
    delete_records(store.client(), TABLE, Some(&condition))
        .await
        .map_err(map)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repo::sync as rows;
    use crate::schema::EMBED_DIM;

    // Forgetting hides the trace from every read path, but the row is RETAINED as
    // a tombstone (so sync can carry the deletion); a grace-windowed purge then
    // removes it for good.
    #[tokio::test]
    async fn soft_delete_hides_the_trace_then_purge_removes_it() {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        let tenant = TenantId::new("t");
        let now = chrono::Utc::now();
        let m = Memory::new(
            "11111111-0000-0000-0000-000000000001",
            tenant.clone(),
            MemoryNetwork::World,
            "remember me",
            0.8,
            now,
        );
        upsert(&store, &m).await.unwrap();
        assert_eq!(list(&store, &tenant).await.unwrap().len(), 1);

        // Forget: hidden from list + get, but the raw row stays (for propagation).
        assert!(soft_delete(&store, &tenant, &m.id, now)
            .await
            .unwrap()
            .is_some());
        assert!(
            list(&store, &tenant).await.unwrap().is_empty(),
            "hidden from list"
        );
        assert!(
            get(&store, &tenant, &m.id).await.unwrap().is_none(),
            "hidden from get"
        );
        assert_eq!(
            rows::list_rows(&store, "memory").await.unwrap().len(),
            1,
            "tombstone row retained so the deletion can propagate"
        );

        // Re-forget is a no-op (already a tombstone).
        assert!(soft_delete(&store, &tenant, &m.id, now)
            .await
            .unwrap()
            .is_none());

        // Purge with a cutoff after the deletion removes it for good.
        let purged = purge(&store, now + chrono::Duration::seconds(1))
            .await
            .unwrap();
        assert_eq!(purged, 1);
        assert!(
            rows::list_rows(&store, "memory").await.unwrap().is_empty(),
            "purged"
        );
        // A purge before the cutoff leaves live-window tombstones alone.
        upsert(&store, &m).await.unwrap();
        soft_delete(&store, &tenant, &m.id, now).await.unwrap();
        assert_eq!(
            purge(&store, now - chrono::Duration::days(1))
                .await
                .unwrap(),
            0,
            "within the grace window: not purged"
        );
    }

    // Reinforcement is one atomic, tombstone-guarded statement: the increment is
    // computed server-side (so it composes without losing an update), and a
    // forgotten trace is refused by the `deleted_at IS NONE` guard rather than
    // resurrected.
    #[tokio::test]
    async fn reinforce_is_atomic_and_tombstone_guarded() {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        let tenant = TenantId::new("t");
        let now = chrono::Utc::now();
        let m = Memory::new(
            "22222222-0000-0000-0000-000000000002",
            tenant.clone(),
            MemoryNetwork::World,
            "keep me",
            0.5,
            now,
        );
        upsert(&store, &m).await.unwrap();

        // The increment + confidence bump (0.5 -> 0.625) happen in the engine.
        let r1 = reinforce(&store, &tenant, &m.id, now)
            .await
            .unwrap()
            .expect("a live trace reinforces");
        assert_eq!(r1.reinforcement, 1);
        assert!((r1.confidence - 0.625).abs() < 1e-4, "confidence bumped");
        // Increments compose -- no lost update.
        let r2 = reinforce(&store, &tenant, &m.id, now)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(r2.reinforcement, 2);

        // A live trace owned by another tenant is not reinforceable (the tenant
        // guard, isolated here while the row is still live).
        assert!(
            reinforce(&store, &TenantId::new("other"), &m.id, now)
                .await
                .unwrap()
                .is_none(),
            "cross-tenant reinforce is refused"
        );

        // Forget, then a reinforcement is refused and does NOT resurrect.
        soft_delete(&store, &tenant, &m.id, now).await.unwrap();
        assert!(
            reinforce(&store, &tenant, &m.id, now)
                .await
                .unwrap()
                .is_none(),
            "reinforcing a forgotten trace is a no-op"
        );
        let rid = RecordID::<()>::new(TABLE, m.id.as_str()).unwrap();
        let row: MemoryRow =
            serde_json::from_value(get_record(store.client(), &rid).await.unwrap().unwrap())
                .unwrap();
        assert!(row.deleted_at.is_some(), "tombstone survives");
        assert_eq!(row.reinforcement, 2, "no phantom increment after forget");
    }
}
