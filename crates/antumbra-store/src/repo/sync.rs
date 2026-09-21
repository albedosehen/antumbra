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

use futures::StreamExt;
use serde_json::Value;
use surql::connection::streaming::LiveQuery;
use surql::query::crud::upsert_record_target;
use surql::types::operators::gt;
use surrealdb::types::Action;
use tokio::sync::mpsc;

use antumbra_core::Result;

use crate::error::map;
use crate::store::Store;

/// Every row of `table`, as raw JSON. Each row carries its own SurrealDB record
/// `id`, which [`row_id`] reads back and [`put_row`] writes to the other store.
pub async fn list_rows(store: &Store, table: &str) -> Result<Vec<Value>> {
    // Paged by id (see `Store::read_paged`): a full-table replication read is the
    // highest-risk frame -- `memory` alone is thousands of embedding-carrying
    // rows, and this path is table-agnostic so any large table flows through it.
    // The collector pairs rows by their carried record id, so id-order is fine.
    store.read_paged(table, None, None).await
}

/// The rows of `table` whose `version_field` is **strictly greater** than
/// `since` (an RFC3339 string), for the collector's incremental cursor: each
/// cycle fetches only what changed past its high-water mark instead of the whole
/// table. An empty `since` matches everything, so it degrades to [`list_rows`] (a
/// full scan) -- the first cycle, and the path that also returns rows with no
/// version field at all.
///
/// The comparison is a plain SurrealQL string `>` (the field is stored as an
/// RFC3339 string; the bound literal is a quoted string), so no datetime crosses
/// into the query. That is correct ordering because chrono's `to_rfc3339()` over
/// uniform-UTC stamps is lexicographically monotonic with time (fixed-width
/// fields; the fraction's `+` terminator sorts below any digit). Builders only.
pub async fn list_rows_since(
    store: &Store,
    table: &str,
    version_field: &str,
    since: &str,
) -> Result<Vec<Value>> {
    if since.is_empty() {
        return list_rows(store, table).await;
    }
    // Paged by id (see `Store::read_paged`); the incremental `version_field >
    // since` cursor is preserved. The collector pairs rows by id, so id-order is
    // fine.
    let filter = gt(version_field, since);
    store.read_paged(table, None, Some(&filter)).await
}

/// What became of a replicated row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Written {
    /// The row landed.
    Yes,
    /// The engine persisted nothing. Under a record session that means the
    /// caller may **read** this row and may not **write** it, which is not an
    /// error and must not be counted as a write: the engine refuses silently,
    /// so a caller that assumed success would report progress it never made,
    /// on every cycle, forever.
    Refused,
    /// The row carried no usable id. Skipped, not an error.
    NoId,
}

impl Written {
    pub fn landed(self) -> bool {
        matches!(self, Written::Yes)
    }
}

/// Upsert a raw row (as returned by [`list_rows`]) into this store, recreating
/// the same record it carries (idempotent `UPSERT ... CONTENT`). The id is
/// addressed separately, so it is stripped from the written payload.
///
/// The answer distinguishes a refusal from a success because the engine does
/// not. A disallowed write persists nothing and returns no row, exactly as a
/// permitted one that wrote nothing would look if we only checked for an
/// error -- so the returned record is the only evidence either way.
///
/// The row's own `id` target is reused **verbatim** rather than parsed back into
/// a `RecordID` and re-rendered: SurrealDB's v3 id escaping (e.g. `⟨`uuid`⟩` for
/// a hyphenated key) is not round-trip-stable through parse-then-display, so a
/// re-render would double-escape and address a different record. Both stores
/// render the same id identically, so the verbatim string also serves as the
/// cross-store match key ([`row_id`]).
pub async fn put_row(store: &Store, row: &Value) -> Result<Written> {
    let Some(target) = row_id(row) else {
        return Ok(Written::NoId);
    };
    let mut data = row.clone();
    if let Value::Object(map) = &mut data {
        map.remove("id");
    }
    let written = upsert_record_target(store.client(), &target, data)
        .await
        .map_err(map)?;
    Ok(match written {
        Value::Object(_) => Written::Yes,
        _ => Written::Refused,
    })
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

/// What happened to a watched row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeAction {
    Create,
    Update,
    Delete,
}

/// A live change on a watched table: the action and the affected row (its full
/// content for create/update; for a delete, whatever the engine reports).
#[derive(Debug, Clone)]
pub struct ChangeEvent {
    pub action: ChangeAction,
    pub row: Value,
}

/// Subscribe to every change on `table` (`LIVE SELECT`), forwarding each as a
/// [`ChangeEvent`] on the returned channel. The subscription is registered before
/// this returns, so writes that happen after the call are observed. A background
/// task owns the live stream and exits when the stream ends/errors, the query is
/// killed, or the receiver is dropped (which also drops the subscription,
/// sending `KILL` to the engine).
///
/// This is the R-2 change-feed primitive; routing a change to the right grantees
/// (by tenant/compartment) is layered on top by the caller. Requires a live-query
/// capable connection (`ws://` or embedded; not `http`).
pub async fn watch_table(store: &Store, table: &str) -> Result<mpsc::Receiver<ChangeEvent>> {
    let mut live: LiveQuery<Value> = LiveQuery::start(store.client(), table).await.map_err(map)?;
    let (tx, rx) = mpsc::channel(64);
    // Keep a store handle alive for the task's lifetime so the connection backing
    // the live stream is not dropped out from under it.
    let keep_alive = store.clone();
    tokio::spawn(async move {
        let _keep_alive = keep_alive;
        while let Some(item) = live.next().await {
            let Ok(notification) = item else {
                break; // stream error: end the watch
            };
            let action = match notification.action {
                Action::Create => ChangeAction::Create,
                Action::Update => ChangeAction::Update,
                Action::Delete => ChangeAction::Delete,
                _ => break, // Killed (or future variants): stop watching
            };
            let event = ChangeEvent {
                action,
                row: notification.data,
            };
            if tx.send(event).await.is_err() {
                break; // receiver dropped: stop watching
            }
        }
    });
    Ok(rx)
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

        assert!(put_row(&dst, row).await.unwrap().landed(), "row replicated");

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

    // The incremental cursor fetch returns only rows newer than the watermark
    // (string `>` on the RFC3339 version field), and an empty watermark returns
    // everything -- the full-scan bootstrap path.
    #[tokio::test]
    async fn list_rows_since_filters_by_version() {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        let tenant = TenantId::new("t");
        let t0 = chrono::Utc::now();
        let t1 = t0 + chrono::Duration::seconds(10);
        let older = mem("00000000-0000-0000-0000-0000000000a1", &tenant, "older", t0);
        let newer = mem("00000000-0000-0000-0000-0000000000a2", &tenant, "newer", t1);
        memory::upsert(&store, &older).await.unwrap();
        memory::upsert(&store, &newer).await.unwrap();

        // Empty watermark: a full scan returns both.
        assert_eq!(
            list_rows_since(&store, "memory", "updated_at", "")
                .await
                .unwrap()
                .len(),
            2
        );
        // Watermark at the older row's timestamp: only the strictly-newer row.
        let rows = list_rows_since(&store, "memory", "updated_at", &t0.to_rfc3339())
            .await
            .unwrap();
        assert_eq!(rows.len(), 1, "only the row newer than the watermark");
        assert_eq!(
            rows[0].get("content").and_then(Value::as_str),
            Some("newer")
        );
        // Watermark at or past the newest: nothing changed.
        assert!(
            list_rows_since(&store, "memory", "updated_at", &t1.to_rfc3339())
                .await
                .unwrap()
                .is_empty()
        );
    }

    fn mem(
        id: &str,
        tenant: &TenantId,
        content: &str,
        at: chrono::DateTime<chrono::Utc>,
    ) -> Memory {
        Memory::new(id, tenant.clone(), MemoryNetwork::World, content, 0.8, at)
    }

    // The change-feed watcher delivers a write made after the subscription is
    // registered -- proving LIVE SELECT works on the embedded engine (the R-2
    // foundation), and that the event carries the row.
    #[tokio::test]
    async fn watch_table_delivers_a_create() {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        let tenant = TenantId::new("tenant-x");
        let mut rx = watch_table(&store, "memory").await.unwrap();

        let now = chrono::Utc::now();
        let m = Memory::new(
            "99999999-0000-0000-0000-000000000009",
            tenant,
            MemoryNetwork::World,
            "watched write",
            0.8,
            now,
        );
        memory::upsert(&store, &m).await.unwrap();

        let event = tokio::time::timeout(std::time::Duration::from_secs(3), rx.recv())
            .await
            .expect("a change is delivered before the timeout")
            .expect("the watch channel stays open");
        assert_eq!(event.action, ChangeAction::Create);
        assert_eq!(
            event.row.get("content").and_then(Value::as_str),
            Some("watched write"),
            "the event carries the written row"
        );
    }

    // The deployment reality: the networked/embedded server holds ONE connection
    // and signs it in per request. The live-propagation watch is registered once
    // at startup (owner), then the connection signs in as a tenant to serve a
    // write. This proves the owner-registered subscription still delivers that
    // tenant-authored write -- i.e. per-request signin does not starve the feed.
    #[tokio::test]
    async fn watch_survives_a_tenant_signin() {
        use crate::repo::principal;
        use antumbra_core::UserId;

        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        let tenant = TenantId::new("ws:t");
        let user = UserId::new("user:u");
        principal::provision(&store, &tenant, &user).await.unwrap();

        // Subscription registered as the startup owner session (before any signin).
        let mut rx = watch_table(&store, "memory").await.unwrap();

        // A request binds the shared connection to the tenant, then writes.
        store.signin(&tenant, &user).await.unwrap();
        let now = chrono::Utc::now();
        let m = Memory::new(
            "bbbbbbbb-0000-0000-0000-00000000000b",
            tenant.clone(),
            MemoryNetwork::World,
            "written under signin",
            0.7,
            now,
        );
        memory::upsert(&store, &m).await.unwrap();

        let event = tokio::time::timeout(std::time::Duration::from_secs(3), rx.recv())
            .await
            .expect("the owner watch delivers a tenant-authored write")
            .expect("the watch channel stays open");
        assert_eq!(event.action, ChangeAction::Create);
        assert_eq!(
            event.row.get("content").and_then(Value::as_str),
            Some("written under signin")
        );
    }
}
