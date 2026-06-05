//! Penumbra memory repository — tenant-isolated traces (the consolidation
//! source). ADR-0004/0009.
//!
//! Isolation is engine-enforced: the `memory` table carries a row-level
//! `PERMISSIONS ... WHERE tenant_id = $auth.tenant` clause, so once a per-tenant
//! `ScopeCredentials` session binds `$auth.tenant`, the engine refuses any read
//! whose `tenant_id` does not match — a forgotten filter cannot leak. This repo
//! *also* applies an explicit `WHERE tenant_id = ...` as the documented second
//! layer (defense-in-depth), and re-checks `tenant_id` on a point read by id (so
//! another tenant's trace reads as absent). Built on surql-rs builders + `crud`;
//! no raw SurrealQL.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use surql::query::builder::Query;
use surql::query::crud::{delete_records, get_record, query_records, upsert_record};
use surql::query::helpers::VectorDistanceType;
use surql::types::operators::{and_, eq};
use surql::types::RecordID;

use antumbra_core::{ExpertId, Memory, MemoryId, MemoryNetwork, Result, TenantId};

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
    created_at: String,
    updated_at: String,
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
            consolidated_expert: m.consolidated_expert.as_ref().map(|e| e.as_str().to_string()),
            created_at: m.created_at.to_rfc3339(),
            updated_at: m.updated_at.to_rfc3339(),
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
            created_at: parse_dt(&self.created_at)?,
            updated_at: parse_dt(&self.updated_at)?,
        })
    }
}

/// Insert or replace a memory (addressed by its globally-unique id).
pub async fn upsert(store: &Store, memory: &Memory) -> Result<()> {
    let id = RecordID::<()>::new(TABLE, memory.id.as_str()).map_err(map)?;
    let data: Value = serde_json::to_value(MemoryRow::from_domain(memory))?;
    upsert_record(store.client(), &id, data).await.map_err(map)?;
    Ok(())
}

/// Fetch one memory by id, but only if it belongs to `tenant` — a trace owned by
/// another tenant reads as absent (the defensive isolation re-check).
pub async fn get(store: &Store, tenant: &TenantId, id: &MemoryId) -> Result<Option<Memory>> {
    let rid = RecordID::<()>::new(TABLE, id.as_str()).map_err(map)?;
    match get_record(store.client(), &rid).await.map_err(map)? {
        Some(value) => {
            let row: MemoryRow = serde_json::from_value(value)?;
            if row.tenant_id != tenant.as_str() {
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
    rows.into_iter().map(MemoryRow::into_domain).collect()
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
    rows.into_iter().map(MemoryRow::into_domain).collect()
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
        Some(net) => and_(eq("tenant_id", tenant.as_str()), eq("network", net.as_str())),
        None => eq("tenant_id", tenant.as_str()),
    };
    let q = Query::new()
        .select(None)
        .from_table(TABLE)
        .map_err(map)?
        .where_(condition)
        .vector_search("embedding", vector, k as i64, VectorDistanceType::Cosine, None)
        .map_err(map)?;
    let rows: Vec<MemoryRow> = query_records(store.client(), &q).await.map_err(map)?;
    rows.into_iter().map(MemoryRow::into_domain).collect()
}

/// Reinforce a trace (recurrence + confidence bump), tenant-checked. Returns the
/// updated memory, or `None` if it does not exist for this tenant.
pub async fn reinforce(
    store: &Store,
    tenant: &TenantId,
    id: &MemoryId,
    now: chrono::DateTime<chrono::Utc>,
) -> Result<Option<Memory>> {
    match get(store, tenant, id).await? {
        Some(mut m) => {
            m.reinforce(now);
            upsert(store, &m).await?;
            Ok(Some(m))
        }
        None => Ok(None),
    }
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
            upsert(store, &m).await?;
            Ok(Some(m))
        }
        None => Ok(None),
    }
}

/// Delete a trace, but only within the caller's tenant (the `tenant_id`
/// predicate is ANDed onto the key, so no cross-tenant delete).
pub async fn delete(store: &Store, tenant: &TenantId, id: &MemoryId) -> Result<()> {
    let condition = and_(eq("key", id.as_str()), eq("tenant_id", tenant.as_str()));
    delete_records(store.client(), TABLE, Some(&condition))
        .await
        .map_err(map)?;
    Ok(())
}
