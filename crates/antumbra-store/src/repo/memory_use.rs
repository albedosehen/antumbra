//! What the system has seen of each memory's use ([`MemoryUse`]): one row per
//! (workspace, memory), written by the server as it watches recall and the work
//! that follows, never by hand.
//!
//! The counts live apart from the `memory` row on purpose. A memory's
//! `updated_at` drives replication and live propagation, so counting a recall
//! on the memory itself would re-send the whole row, embedding and all, to
//! every device and grantee each time it was recalled. A count is what this
//! server observed, not part of what the memory says, so it stays here and is
//! not replicated. Tenant-scoped like the memory (engine PERMISSIONS plus the
//! repo's own filter). Built on surql-rs builders; no raw SurrealQL.

use std::collections::HashSet;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use surql::query::builder::Query;
use surql::query::crud::{create_record, query_records};
use surql::query::expressions::field;
use surql::types::operators::{and_, eq, inside};

use antumbra_core::{MemoryId, MemoryUse, Result, TenantId};

use crate::dto::parse_dt;
use crate::error::map;
use crate::store::Store;

const TABLE: &str = "memory_use";

/// Which counts one observation adds to. Each set flag adds one; `used`,
/// `recurred` and `good` also restart the memory's fade.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Seen {
    pub recalled: bool,
    pub judged: bool,
    pub used: bool,
    pub recurred: bool,
    pub good: bool,
    pub bad: bool,
}

impl Seen {
    pub const RECALLED: Seen = Seen {
        recalled: true,
        judged: false,
        used: false,
        recurred: false,
        good: false,
        bad: false,
    };
    pub const PASSED_OVER: Seen = Seen {
        judged: true,
        ..Seen::NOTHING
    };
    pub const USED: Seen = Seen {
        judged: true,
        used: true,
        ..Seen::NOTHING
    };
    /// Drawn on by work it was not judged against (an agent's own recall).
    pub const USED_UNJUDGED: Seen = Seen {
        used: true,
        ..Seen::NOTHING
    };
    pub const RECURRED: Seen = Seen {
        recurred: true,
        ..Seen::NOTHING
    };
    pub const GOOD: Seen = Seen {
        good: true,
        ..Seen::NOTHING
    };
    pub const BAD: Seen = Seen {
        bad: true,
        ..Seen::NOTHING
    };
    const NOTHING: Seen = Seen {
        recalled: false,
        judged: false,
        used: false,
        recurred: false,
        good: false,
        bad: false,
    };

    fn restarts_fade(self) -> bool {
        self.used || self.recurred || self.good
    }

    /// The counts, in row order, with whether this observation adds to each.
    fn counts(self) -> [(&'static str, bool); 6] {
        [
            ("recalled", self.recalled),
            ("judged", self.judged),
            ("used", self.used),
            ("recurred", self.recurred),
            ("good", self.good),
            ("bad", self.bad),
        ]
    }
}

#[derive(Serialize, Deserialize)]
struct UseRow {
    tenant_id: String,
    memory_id: String,
    #[serde(default)]
    recalled: u32,
    #[serde(default)]
    judged: u32,
    #[serde(default)]
    used: u32,
    #[serde(default)]
    recurred: u32,
    #[serde(default)]
    good: u32,
    #[serde(default)]
    bad: u32,
    #[serde(default)]
    last_recalled_at: Option<String>,
    #[serde(default)]
    last_used_at: Option<String>,
}

impl UseRow {
    fn into_domain(self) -> Result<MemoryUse> {
        let at = |s: Option<String>| s.as_deref().map(parse_dt).transpose();
        Ok(MemoryUse {
            memory: MemoryId::new(self.memory_id),
            recalled: self.recalled,
            judged: self.judged,
            used: self.used,
            recurred: self.recurred,
            good: self.good,
            bad: self.bad,
            last_recalled_at: at(self.last_recalled_at)?,
            last_used_at: at(self.last_used_at)?,
        })
    }
}

/// One record id per (tenant, memory), so a second first-sighting collides.
fn key(tenant: &TenantId, memory: &str) -> String {
    format!("{}|{memory}", tenant.as_str()).replace([':', '/', '\\', '|', ' '], "_")
}

fn ids_value(ids: &[MemoryId]) -> Vec<Value> {
    ids.iter()
        .map(|m| Value::String(m.as_str().to_string()))
        .collect()
}

/// Add one observation to each of `memories` in `tenant`. Atomic per row: the
/// counts are raised in the engine from the row's current values, so two
/// servers or two requests counting at once lose nothing, and a memory seen for
/// the first time gets its row.
pub async fn record(
    store: &Store,
    tenant: &TenantId,
    memories: &[MemoryId],
    seen: Seen,
    now: DateTime<Utc>,
) -> Result<()> {
    if memories.is_empty() {
        return Ok(());
    }
    let mut unique: Vec<MemoryId> = Vec::with_capacity(memories.len());
    let mut taken = HashSet::new();
    for m in memories {
        if taken.insert(m.as_str().to_string()) {
            unique.push(m.clone());
        }
    }
    let counted = raise(store, tenant, &unique, seen, now).await?;
    for memory in unique.iter().filter(|m| !counted.contains(m.as_str())) {
        if !first_sighting(store, tenant, memory, seen, now).await? {
            // Another writer created it between the two statements: count on
            // the row it made.
            raise(store, tenant, std::slice::from_ref(memory), seen, now).await?;
        }
    }
    Ok(())
}

/// Raise the counts on the rows that exist; returns the memories it raised.
async fn raise(
    store: &Store,
    tenant: &TenantId,
    memories: &[MemoryId],
    seen: Seen,
    now: DateTime<Utc>,
) -> Result<HashSet<String>> {
    let mut q = Query::new().update_set(TABLE).map_err(map)?;
    for (name, adds) in seen.counts() {
        if adds {
            q = q.set_expr(name, field(name) + 1).map_err(map)?;
        }
    }
    let stamp = now.to_rfc3339();
    if seen.recalled {
        q = q.set("last_recalled_at", stamp.clone()).map_err(map)?;
    }
    if seen.restarts_fade() {
        q = q.set("last_used_at", stamp).map_err(map)?;
    }
    let q = q
        .where_(and_(
            eq("tenant_id", tenant.as_str()),
            inside("memory_id", ids_value(memories)),
        ))
        .return_after();
    let rows: Vec<UseRow> = query_records(store.client(), &q).await.map_err(map)?;
    Ok(rows.into_iter().map(|r| r.memory_id).collect())
}

/// Create the row for a memory seen for the first time. `false` when it
/// already exists.
async fn first_sighting(
    store: &Store,
    tenant: &TenantId,
    memory: &MemoryId,
    seen: Seen,
    now: DateTime<Utc>,
) -> Result<bool> {
    let stamp = now.to_rfc3339();
    let mut row = serde_json::json!({
        "id": key(tenant, memory.as_str()),
        "tenant_id": tenant.as_str(),
        "memory_id": memory.as_str(),
    });
    for (name, adds) in seen.counts() {
        row[name] = Value::from(u32::from(adds));
    }
    if seen.recalled {
        row["last_recalled_at"] = Value::String(stamp.clone());
    }
    if seen.restarts_fade() {
        row["last_used_at"] = Value::String(stamp);
    }
    match create_record(store.client(), TABLE, row).await {
        Ok(_) => Ok(true),
        Err(e) if e.to_string().contains("already exists") => Ok(false),
        Err(e) => Err(map(e)),
    }
}

/// What has been seen of `memories` in `tenant`, for those seen at all.
pub async fn for_memories(
    store: &Store,
    tenant: &TenantId,
    memories: &[MemoryId],
) -> Result<Vec<MemoryUse>> {
    if memories.is_empty() {
        return Ok(Vec::new());
    }
    let q = Query::new()
        .select(None)
        .from_table(TABLE)
        .map_err(map)?
        .where_(and_(
            eq("tenant_id", tenant.as_str()),
            inside("memory_id", ids_value(memories)),
        ));
    let rows: Vec<UseRow> = query_records(store.client(), &q).await.map_err(map)?;
    rows.into_iter().map(UseRow::into_domain).collect()
}

/// Everything seen in `tenant`, for the console.
pub async fn all(store: &Store, tenant: &TenantId) -> Result<Vec<MemoryUse>> {
    let filter = eq("tenant_id", tenant.as_str());
    let rows: Vec<UseRow> = store.read_paged(TABLE, None, Some(&filter)).await?;
    rows.into_iter().map(UseRow::into_domain).collect()
}

#[cfg(test)]
mod tests;
