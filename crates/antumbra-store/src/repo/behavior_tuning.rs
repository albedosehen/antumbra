//! The loop's readings of a workspace's behavior audits against the
//! regression confidence admission is held to: one row per reading, appended;
//! the latest is the confidence in force. Tenant-scoped like the behaviors,
//! so a workspace reads only its own. All access is via surql-rs builders.

use chrono::{DateTime, Utc};
use surql::query::builder::Query;
use surql::query::crud::{create_record, first, query_records};
use surql::types::operators::eq;

use antumbra_core::behavior::tuning::{Lean, Reading};
use antumbra_core::{Result, TenantId};

use crate::error::map;
use crate::store::Store;

const TABLE: &str = "behavior_tuning";

/// One reading, as stored.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TuningRow {
    pub tenant_id: String,
    pub at: DateTime<Utc>,
    /// Which way the pool leaned: `looser` or `stricter`.
    pub lean: String,
    /// The confidence in force when read.
    pub from: f64,
    /// The confidence after; the one in force until the next reading.
    pub to: f64,
    pub bets: usize,
    pub fresh: usize,
    pub base: usize,
    pub expert: usize,
    pub realized: f64,
}

impl TuningRow {
    /// `reading`, taken in `tenant` at `at`.
    pub fn of(tenant: &TenantId, reading: &Reading, at: DateTime<Utc>) -> TuningRow {
        TuningRow {
            tenant_id: tenant.as_str().to_string(),
            at,
            lean: reading.lean.as_str().to_string(),
            from: reading.from,
            to: reading.to,
            bets: reading.bets,
            fresh: reading.fresh,
            base: reading.base,
            expert: reading.expert,
            realized: reading.realized,
        }
    }

    /// The reading this row kept. `None` for a lean this build does not know.
    pub fn reading(&self) -> Option<Reading> {
        Some(Reading {
            lean: Lean::parse(&self.lean)?,
            from: self.from,
            to: self.to,
            bets: self.bets,
            fresh: self.fresh,
            base: self.base,
            expert: self.expert,
            realized: self.realized,
        })
    }
}

/// Keep a reading.
pub async fn append(store: &Store, row: &TuningRow) -> Result<()> {
    create_record(store.client(), TABLE, serde_json::to_value(row)?)
        .await
        .map_err(map)?;
    Ok(())
}

/// The workspace's latest reading, whose `to` is the confidence in force;
/// `None` before the first, when the recipe's stands.
pub async fn latest(store: &Store, tenant: &TenantId) -> Result<Option<TuningRow>> {
    let query = Query::new()
        .select(None)
        .from_table(TABLE)
        .map_err(map)?
        .where_(eq("tenant_id", tenant.as_str()))
        .order_by("at", "DESC")
        .map_err(map)?;
    first(store.client(), &query).await.map_err(map)
}

/// The workspace's readings, newest first, at most `limit`.
pub async fn recent(store: &Store, tenant: &TenantId, limit: usize) -> Result<Vec<TuningRow>> {
    let query = Query::new()
        .select(None)
        .from_table(TABLE)
        .map_err(map)?
        .where_(eq("tenant_id", tenant.as_str()))
        .order_by("at", "DESC")
        .map_err(map)?;
    let mut rows: Vec<TuningRow> = query_records(store.client(), &query).await.map_err(map)?;
    rows.truncate(limit);
    Ok(rows)
}
