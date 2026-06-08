//! Penumbra graph repository — tenant-isolated typed edges between memories.
//!
//! Same isolation contract as `memory`: engine-enforced `PERMISSIONS` plus the
//! explicit `WHERE tenant_id = ...` second layer. Edges are keyed by
//! `(tenant, from, to, type)` so re-relating the same pair is idempotent.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use surql::query::builder::Query;
use surql::query::crud::{query_records, upsert_record};
use surql::types::operators::{and_, eq};
use surql::types::RecordID;

use antumbra_core::{EdgeType, MemoryEdge, MemoryId, Result, TenantId};

use crate::dto::parse_dt;
use crate::error::map;
use crate::store::Store;

const TABLE: &str = "memory_edge";

#[derive(Serialize, Deserialize)]
struct EdgeRow {
    tenant_id: String,
    from_id: String,
    to_id: String,
    edge_type: EdgeType,
    #[serde(default)]
    weight: f32,
    created_at: String,
}

impl EdgeRow {
    fn from_domain(e: &MemoryEdge) -> Self {
        EdgeRow {
            tenant_id: e.tenant.as_str().to_string(),
            from_id: e.from_id.as_str().to_string(),
            to_id: e.to_id.as_str().to_string(),
            edge_type: e.edge_type,
            weight: e.weight,
            created_at: e.created_at.to_rfc3339(),
        }
    }

    fn into_domain(self) -> Result<MemoryEdge> {
        Ok(MemoryEdge {
            tenant: TenantId::new(self.tenant_id),
            from_id: MemoryId::new(self.from_id),
            to_id: MemoryId::new(self.to_id),
            edge_type: self.edge_type,
            weight: self.weight,
            created_at: parse_dt(&self.created_at)?,
        })
    }
}

/// A stable, unique record key for an edge (so re-relating is idempotent).
fn edge_key(e: &MemoryEdge) -> String {
    format!(
        "{}|{}|{}|{}",
        e.tenant.as_str(),
        e.from_id.as_str(),
        e.to_id.as_str(),
        e.edge_type.as_str()
    )
    .replace([':', '/', '\\', '|', ' '], "_")
}

/// Create or update a directed edge (idempotent per from/to/type).
pub async fn relate(store: &Store, edge: &MemoryEdge) -> Result<()> {
    let rid = RecordID::<()>::new(TABLE, edge_key(edge).as_str()).map_err(map)?;
    let data: Value = serde_json::to_value(EdgeRow::from_domain(edge))?;
    upsert_record(store.client(), &rid, data)
        .await
        .map_err(map)?;
    Ok(())
}

/// Every edge across all tenants — the owner/console view of the whole Penumbra
/// graph (parallels [`crate::repo::memory::all_unscoped`]). For the operator
/// console, which reads the store as owner; not for tenant-scoped paths.
pub async fn all_unscoped(store: &Store) -> Result<Vec<MemoryEdge>> {
    let q = Query::new().select(None).from_table(TABLE).map_err(map)?;
    let rows: Vec<EdgeRow> = query_records(store.client(), &q).await.map_err(map)?;
    rows.into_iter().map(EdgeRow::into_domain).collect()
}

/// The edges leaving `from` within `tenant` (optionally one edge type).
pub async fn neighbors(
    store: &Store,
    tenant: &TenantId,
    from: &MemoryId,
    edge_type: Option<EdgeType>,
) -> Result<Vec<MemoryEdge>> {
    let cond = match edge_type {
        Some(t) => and_(
            and_(
                eq("tenant_id", tenant.as_str()),
                eq("from_id", from.as_str()),
            ),
            eq("edge_type", t.as_str()),
        ),
        None => and_(
            eq("tenant_id", tenant.as_str()),
            eq("from_id", from.as_str()),
        ),
    };
    let q = Query::new()
        .select(None)
        .from_table(TABLE)
        .map_err(map)?
        .where_(cond);
    let rows: Vec<EdgeRow> = query_records(store.client(), &q).await.map_err(map)?;
    rows.into_iter().map(EdgeRow::into_domain).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::EMBED_DIM;
    use chrono::Utc;

    #[tokio::test]
    async fn relate_is_idempotent_and_all_unscoped_reads_every_edge() {
        let s = Store::connect_memory(EMBED_DIM).await.unwrap();
        let now = Utc::now();
        let contradicts = MemoryEdge::new(
            "ws:1",
            "memory:a",
            "memory:b",
            EdgeType::Contradicts,
            0.8,
            now,
        );
        relate(&s, &contradicts).await.unwrap();
        // Re-relating the same (from,to,type) upserts in place, not duplicates.
        relate(&s, &contradicts).await.unwrap();
        relate(
            &s,
            &MemoryEdge::new(
                "ws:2",
                "memory:c",
                "memory:a",
                EdgeType::Supersedes,
                0.5,
                now,
            ),
        )
        .await
        .unwrap();

        let all = all_unscoped(&s).await.unwrap();
        assert_eq!(all.len(), 2, "two distinct edges across tenants");

        // The tenant-scoped neighbor read still sees only its own edge.
        let from_a = neighbors(&s, &TenantId::new("ws:1"), &MemoryId::new("memory:a"), None)
            .await
            .unwrap();
        assert_eq!(from_a.len(), 1);
        assert_eq!(from_a[0].edge_type, EdgeType::Contradicts);
    }
}
