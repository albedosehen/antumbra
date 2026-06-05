//! Generic, table-agnostic row access for the collector/sync (R-1). The
//! collector replicates whole penumbra tables between a local embedded store and
//! a remote authoritative one, so it works on raw JSON rows rather than typed
//! DTOs -- one code path covers `memory`, `memory_edge`, `compartment`, `grant`.
//!
//! Everything goes through surql-rs builders / `crud` helpers (never raw
//! SurrealQL). Timestamps are already RFC3339 strings in the storage DTOs (see
//! `crate::dto`), so a row round-trips between stores without any datetime
//! coercion. Access runs as an owner/root session, spanning tenants; per-tenant
//! isolation is preserved by the `tenant_id` each row carries.

use serde_json::Value;
use surql::query::builder::Query;
use surql::query::crud::{query_records, upsert_record_target};

use antumbra_core::Result;

use crate::error::map;
use crate::store::Store;

/// Every row of `table`, as raw JSON. Each row carries its own SurrealDB record
/// `id`, which [`row_id`] reads back and [`put_row`] writes to the other store.
pub async fn list_rows(store: &Store, table: &str) -> Result<Vec<Value>> {
    let query = Query::new().select(None).from_table(table).map_err(map)?;
    let rows: Vec<Value> = query_records(store.client(), &query).await.map_err(map)?;
    Ok(rows)
}

/// Upsert a raw row (as returned by [`list_rows`]) into this store, recreating
/// the same record it carries (idempotent `UPSERT ... CONTENT`). The id is
/// addressed separately, so it is stripped from the written payload. Returns
/// `false` if the row has no usable id (skipped, not an error).
///
/// The row's own `id` target is reused **verbatim** rather than parsed back into
/// a `RecordID` and re-rendered: SurrealDB's v3 id escaping (e.g. `⟨`uuid`⟩` for
/// a hyphenated key) is not round-trip-stable through parse-then-display, so a
/// re-render would double-escape and address a different record. Both stores
/// render the same id identically, so the verbatim string also serves as the
/// cross-store match key ([`row_id`]).
pub async fn put_row(store: &Store, row: &Value) -> Result<bool> {
    let Some(target) = row_id(row) else {
        return Ok(false);
    };
    let mut data = row.clone();
    if let Value::Object(map) = &mut data {
        map.remove("id");
    }
    upsert_record_target(store.client(), &target, data)
        .await
        .map_err(map)?;
    Ok(true)
}

/// The stable match key for a row -- its record id target string, used both to
/// pair the same record across the two stores and as the upsert target.
/// Tolerates the driver rendering `id` as a `table:key` string or a `{ tb, id }`
/// object. `None` if the row has no recognizable id.
pub fn row_id(row: &Value) -> Option<String> {
    match row.get("id")? {
        Value::String(s) => Some(s.clone()),
        Value::Object(obj) => {
            let table = obj.get("tb").or_else(|| obj.get("table"))?.as_str()?;
            let key = match obj.get("id")? {
                Value::String(s) => s.clone(),
                Value::Number(n) => n.to_string(),
                other => other.as_str()?.to_string(),
            };
            Some(format!("{table}:{key}"))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repo::memory;
    use crate::schema::EMBED_DIM;
    use antumbra_core::{Memory, MemoryNetwork, TenantId};

    // A raw row read from one store and put into another must recreate the same
    // record verbatim -- same record id, and readable by the typed repo on the
    // far side. This is the whole generic-replication contract the collector
    // depends on, and it exercises whatever shape the driver renders `id` in.
    #[tokio::test]
    async fn a_raw_row_round_trips_between_two_stores() {
        let src = Store::connect_memory(EMBED_DIM).await.unwrap();
        let dst = Store::connect_memory(EMBED_DIM).await.unwrap();
        let now = chrono::Utc::now();
        let tenant = TenantId::new("tenant-x");
        let m = Memory::new(
            "11111111-2222-3333-4444-555555555555",
            tenant.clone(),
            MemoryNetwork::World,
            "deno install lodash",
            0.9,
            now,
        );
        memory::upsert(&src, &m).await.unwrap();

        let rows = list_rows(&src, "memory").await.unwrap();
        assert_eq!(rows.len(), 1, "source has the one memory");
        let row = &rows[0];
        let id = row_id(row).expect("row carries a record id");
        assert!(id.starts_with("memory:"), "id is a memory record: {id}");

        assert!(put_row(&dst, row).await.unwrap(), "row replicated");

        let dst_rows = list_rows(&dst, "memory").await.unwrap();
        assert_eq!(dst_rows.len(), 1, "destination got exactly one row");
        assert_eq!(
            row_id(&dst_rows[0]).as_deref(),
            Some(id.as_str()),
            "the same record id is preserved across stores"
        );

        // The far side's typed repo can read the replicated row back.
        let got = memory::get(&dst, &tenant, &m.id).await.unwrap();
        let got = got.expect("replicated memory is readable by the typed repo");
        assert_eq!(got.content, "deno install lodash");
        assert_eq!(got.confidence, 0.9);
    }
}
