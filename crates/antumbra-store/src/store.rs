//! The connection handle. Wraps surql-rs's `DatabaseClient` and owns the
//! configured embedding dimension so repositories and schema agree.

use serde::de::DeserializeOwned;

use surql::connection::auth::{DatabaseCredentials, RootCredentials, ScopeCredentials};
use surql::connection::ConnectionConfig;
use surql::query::builder::Query;
use surql::query::crud::query_records;
use surql::types::operators::Operator;
use surql::DatabaseClient;

use antumbra_core::{AntumbraError, Result, TenantId, UserId};

use crate::error::map;
use crate::repo::sync::ChangeEvent;
use crate::schema::{self, EMBED_DIM, TENANT_ACCESS};

/// How many announced changes may wait for the consumer before more are
/// dropped. Announcements are notifications, so a consumer that far behind
/// loses some rather than stalling the writes.
const ANNOUNCED_BACKLOG: usize = 256;

#[derive(Clone)]
pub struct Store {
    client: DatabaseClient,
    embed_dim: usize,
    namespace: String,
    database: String,
    /// Database-level owner credentials (a `DEFINE USER ... ON DATABASE`
    /// user), when the deployment scopes this service below instance root.
    /// `None` = the classic shape: root credentials in the connection config,
    /// or an unauthenticated embedded store.
    db_owner: Option<(String, String)>,
    /// Where memory writes are announced in-process, once
    /// [`Store::announce_changes`] asks for it. Shared by every clone made
    /// after that call.
    changes: Option<tokio::sync::mpsc::Sender<ChangeEvent>>,
}

impl Store {
    /// Connect to an embedded in-memory database and apply the schema. Ideal
    /// for tests and ephemeral runs; nothing persists past process exit.
    pub async fn connect_memory(embed_dim: usize) -> Result<Self> {
        let config = ConnectionConfig::builder()
            .url("mem://")
            .namespace("antumbra")
            .database("main")
            .build()
            .map_err(map)?;
        Self::connect(config, embed_dim).await
    }

    /// Connect to a configured database (remote `ws://` or embedded
    /// `surrealkv://`) and apply the schema.
    pub async fn connect(config: ConnectionConfig, embed_dim: usize) -> Result<Self> {
        let store = Self::connect_without_schema(config, embed_dim).await?;
        store.ensure_schema().await?;
        Ok(store)
    }

    /// Connect **without** applying the schema. For a second, credential-less
    /// *serving* connection to an authenticated remote whose schema a root
    /// connection has already applied: an anonymous session cannot run `DEFINE`,
    /// and only needs to sign in per request as a record (which then scopes the
    /// engine ACL correctly, unlike a root connection, which bypasses it). See
    /// the R-6 fix in the MCP HTTP layer.
    pub async fn connect_without_schema(
        config: ConnectionConfig,
        embed_dim: usize,
    ) -> Result<Self> {
        let namespace = config.namespace().to_string();
        let database = config.database().to_string();
        let client = DatabaseClient::new(config).map_err(map)?;
        client.connect().await.map_err(map)?;
        Ok(Store {
            client,
            embed_dim,
            namespace,
            database,
            db_owner: None,
            changes: None,
        })
    }

    /// Connect as a **database-level** user (`DEFINE USER ... ON DATABASE`)
    /// instead of instance root: the least-privilege shape for a service that
    /// only ever works inside one database (the control plane). The config must
    /// be credential-less -- config credentials would make the client sign the
    /// session in at root level on connect -- so the connection comes up
    /// anonymous and is then signed in at database level. The credentials are
    /// kept so [`Store::signin_root`] can restore the owner view after a
    /// record-scoped `signin`. A `ROLES OWNER` database user can run the
    /// (database-level) schema DDL, so `ensure_schema` still applies.
    pub async fn connect_with_db_user(
        config: ConnectionConfig,
        db_user: &str,
        db_pass: &str,
        embed_dim: usize,
    ) -> Result<Self> {
        if config.username().is_some() || config.password().is_some() {
            return Err(AntumbraError::other(
                "connect_with_db_user requires a credential-less config; \
                 pass the database user via db_user/db_pass",
            ));
        }
        let mut store = Self::connect_without_schema(config, embed_dim).await?;
        store.db_owner = Some((db_user.to_string(), db_pass.to_string()));
        store.signin_root().await?;
        store.ensure_schema().await?;
        Ok(store)
    }

    /// Authenticate this session as `(tenant, user)` via the record-access
    /// method, so `$auth.tenant` (the hard isolation key) and `$auth.user` (the
    /// compartment-ownership / sharing actor) are bound and the engine enforces
    /// the row-level PERMISSIONS. Requires a provisioned principal. Root/owner
    /// sessions skip this and see across tenants.
    pub async fn signin(&self, tenant: &TenantId, user: &UserId) -> Result<()> {
        let creds = ScopeCredentials::new(&self.namespace, &self.database, TENANT_ACCESS)
            .with("tenant", tenant.as_str())
            .with("user", user.as_str());
        self.client.signin(&creds).await.map_err(map)?;
        Ok(())
    }

    /// Drop the current session's authentication, returning to the owner/root
    /// view (full access on the embedded engine).
    pub async fn invalidate(&self) -> Result<()> {
        self.client.invalidate().await.map_err(map)?;
        Ok(())
    }

    /// Return to the **owner** view for cross-tenant work (provisioning a
    /// principal, the live-propagation watcher). On a database-scoped store
    /// ([`Store::connect_with_db_user`]) the owner view is the database user;
    /// on an authenticated remote (`ws://` with root credentials) it re-signs-in
    /// as root, because there `invalidate` would drop to *anonymous*, which has
    /// no permissions. On an embedded/unauthenticated store (no configured
    /// credentials) it falls back to `invalidate` (anonymous *is* the owner
    /// there). Use this, not `invalidate`, whenever owner access is required on
    /// a real deployment.
    pub async fn signin_root(&self) -> Result<()> {
        if let Some((user, pass)) = &self.db_owner {
            let creds = DatabaseCredentials::new(&self.namespace, &self.database, user, pass);
            self.client.signin(&creds).await.map_err(map)?;
            return Ok(());
        }
        match (
            self.client.config().username(),
            self.client.config().password(),
        ) {
            (Some(user), Some(pass)) => {
                let creds = RootCredentials::new(user, pass);
                self.client.signin(&creds).await.map_err(map)?;
            }
            _ => self.client.invalidate().await.map_err(map)?,
        }
        Ok(())
    }

    /// Announce every memory write made through this store, and through the
    /// clones made from it afterwards, on the returned channel. This is live
    /// propagation for an embedded engine: its one connection is signed in
    /// per request, and SurrealDB drops a session's LIVE queries when the
    /// session signs in as another principal, so a LIVE watch there delivers
    /// nothing. A server watches its own LIVE feed instead and never calls
    /// this.
    pub fn announce_changes(&mut self) -> tokio::sync::mpsc::Receiver<ChangeEvent> {
        let (tx, rx) = tokio::sync::mpsc::channel(ANNOUNCED_BACKLOG);
        self.changes = Some(tx);
        rx
    }

    /// Whether writes through this store are announced.
    pub(crate) fn announcing(&self) -> bool {
        self.changes.is_some()
    }

    /// Announce one change. Never blocks the write it follows: with the
    /// consumer gone or [`ANNOUNCED_BACKLOG`] behind, the change is dropped.
    pub(crate) fn announce(&self, event: ChangeEvent) {
        if let Some(tx) = &self.changes {
            let _ = tx.try_send(event);
        }
    }

    /// Apply the schema. The statements are generated by surql-rs schema
    /// builders (`schema::schema_statements`), not hand-authored SurrealQL,
    /// and are idempotent (`IF NOT EXISTS`), so this is safe to call repeatedly.
    pub async fn ensure_schema(&self) -> Result<()> {
        let script = schema::schema_statements(self.embed_dim as u32)?.join("\n");
        self.client.query(&script).await.map_err(map)?;
        Ok(())
    }

    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    pub(crate) fn client(&self) -> &DatabaseClient {
        &self.client
    }

    /// Page through an otherwise-unbounded `SELECT` to avoid a single oversized
    /// WebSocket response. A full `memory` population (each row carrying an
    /// embedding) or a whole-table replication read is thousands of rows;
    /// returning them in one frame exceeds the `ws://` frame limit and resets the
    /// connection (surql-rs v0.28 exposes no max-frame-size knob, so paging is the
    /// only fix). Orders by `id` -- a stable total order, so LIMIT/OFFSET paging
    /// never drops or duplicates rows the way paging over a tied field
    /// (`created_at`/`updated_at`) would. Callers that need a particular order
    /// re-sort the accumulated rows in Rust afterward. `fields` is the projection
    /// (`None` selects every field); `filter` is the caller's existing `WHERE`
    /// predicate (its row-scoping is preserved verbatim).
    pub(crate) async fn read_paged<T: DeserializeOwned>(
        &self,
        table: &str,
        fields: Option<Vec<String>>,
        filter: Option<&Operator>,
    ) -> Result<Vec<T>> {
        self.read_in_pages_of(READ_PAGE_ROWS, table, fields, filter)
            .await
    }

    /// The rows of `table` among the record keys `keys` that belong to
    /// `tenant`, in no particular order: one direct lookup per key. A filter
    /// on a key field (`key INSIDE [...]`) reads every row of the workspace to
    /// answer instead, 178 ms against 4 ms for 150 memories on kuskokwim.
    /// `also` is any further condition on the row, in SurrealQL.
    pub(crate) async fn rows_by_key<T: DeserializeOwned>(
        &self,
        table: &'static str,
        keys: &[String],
        tenant: &TenantId,
        also: &str,
    ) -> Result<Vec<T>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        let also = if also.is_empty() {
            String::new()
        } else {
            format!(" AND ({also})")
        };
        let surql = format!(
            "SELECT * FROM $keys.map(|$k| type::record('{table}', $k))              WHERE tenant_id = $tenant{also}"
        );
        let vars = std::collections::BTreeMap::from([
            ("keys".to_string(), serde_json::json!(keys)),
            ("tenant".to_string(), serde_json::json!(tenant.as_str())),
        ]);
        let raw = self
            .client
            .query_with_vars(&surql, vars)
            .await
            .map_err(map)?;
        rows_of_first_statement(raw)
    }

    /// [`read_paged`](Self::read_paged) with the page size named: for a
    /// projection whose rows are far smaller than a memory's, and so fit many
    /// more to a page, and for testing paging itself with pages of a few rows.
    pub(crate) async fn read_in_pages_of<T: DeserializeOwned>(
        &self,
        page_rows: i64,
        table: &str,
        fields: Option<Vec<String>>,
        filter: Option<&Operator>,
    ) -> Result<Vec<T>> {
        let page_rows = page_rows.max(1);
        // SurrealDB v3 requires the `ORDER BY` idiom to appear in an explicit
        // projection: `SELECT a, b FROM t ORDER BY id` errors with "Missing
        // order idiom `id` in statement selection". We page by `id`, so ensure
        // it is selected when the caller passes a narrowed field list. (A `None`
        // projection is `SELECT *`, which already includes `id`.)
        let fields = fields.map(|mut f| {
            if !f.iter().any(|c| c == "id") {
                f.insert(0, "id".to_string());
            }
            f
        });
        let mut out: Vec<T> = Vec::new();
        let mut offset: i64 = 0;
        loop {
            let mut builder = Query::new()
                .select(fields.clone())
                .from_table(table)
                .map_err(map)?;
            if let Some(condition) = filter {
                builder = builder.where_(condition);
            }
            let query = builder
                .order_by("id", "ASC")
                .map_err(map)?
                .limit(page_rows)
                .map_err(map)?
                .offset(offset)
                .map_err(map)?;
            let page: Vec<T> = query_records(self.client(), &query).await.map_err(map)?;
            let n = page.len() as i64;
            out.extend(page);
            if n < page_rows {
                break;
            }
            offset += page_rows;
        }
        Ok(out)
    }
}

pub const DEFAULT_EMBED_DIM: usize = EMBED_DIM;

/// Rows per page in [`Store::read_paged`].
///
/// A memory row carries its embedding, 384 floats that serialize to about 5 KB
/// of JSON, plus its content, which runs to several KB. The first cut paged at
/// 20,000 rows, so a workspace of 5,672 memories still came back as one
/// response, and on kuskokwim that reset the connection: `list_memories`,
/// `record_merge` and every other whole-workspace read failed with
/// "Connection reset". At 500 rows a page is a few megabytes, well inside the
/// WebSocket limits, and 6,000 memories take twelve round trips.
const READ_PAGE_ROWS: i64 = 500;

/// The rows a one-statement raw query returned, as `T`: the response is one
/// entry per statement, each either the rows or `{ "result": rows }`.
fn rows_of_first_statement<T: DeserializeOwned>(raw: serde_json::Value) -> Result<Vec<T>> {
    use serde_json::Value;
    let first = match raw {
        Value::Array(mut statements) if !statements.is_empty() => statements.swap_remove(0),
        _ => return Ok(Vec::new()),
    };
    let rows = match first {
        Value::Object(mut statement) if statement.contains_key("result") => {
            statement.remove("result").unwrap_or(Value::Null)
        }
        rows => rows,
    };
    match rows {
        Value::Null => Ok(Vec::new()),
        rows => Ok(serde_json::from_value(rows)?),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pages of two over five rows return all five, each once: the loop stops
    /// on the short page, and ordering by id keeps offsets from skipping or
    /// repeating a row.
    #[tokio::test]
    async fn paging_returns_every_row_exactly_once() -> Result<()> {
        let store = Store::connect_memory(EMBED_DIM).await?;
        store
            .client()
            .query(
                "FOR $i IN [1, 2, 3, 4, 5] { CREATE type::record('paging_probe', $i) SET n = $i; };",
            )
            .await
            .map_err(map)?;
        for page_rows in [1, 2, 5, 6] {
            let rows: Vec<serde_json::Value> = store
                .read_in_pages_of(page_rows, "paging_probe", None, None)
                .await?;
            let mut seen: Vec<i64> = rows.iter().filter_map(|r| r["n"].as_i64()).collect();
            seen.sort_unstable();
            assert_eq!(seen, vec![1, 2, 3, 4, 5], "pages of {page_rows}");
        }
        Ok(())
    }

    /// Every (table, column, index) the schema builds an HNSW index
    /// over.
    const VECTOR_TABLES: [(&str, &str, &str); 4] = [
        ("expert", "capability_vec", "expert_cap_hnsw"),
        ("failure_boundary", "context_vec", "fb_ctx_hnsw"),
        ("memory", "embedding", "memory_embedding_hnsw"),
        (
            "document_chunk",
            "embedding",
            "document_chunk_embedding_hnsw",
        ),
    ];

    /// Whether the plan reaches `index` anywhere in its tree. Matched
    /// on the attribute rather than the operator name, so an operator
    /// being respelled does not turn this green by accident.
    fn reaches_index(node: &serde_json::Value, index: &str) -> bool {
        if node.pointer("/attributes/index").and_then(|v| v.as_str()) == Some(index) {
            return true;
        }
        node.get("children")
            .and_then(|v| v.as_array())
            .is_some_and(|kids| kids.iter().any(|kid| reaches_index(kid, index)))
    }

    fn plan_of(answer: &serde_json::Value) -> &serde_json::Value {
        answer
            .get(0)
            .filter(|v| v.get("operator").is_some())
            .unwrap_or_else(|| panic!("expected a plan, got {answer}"))
    }

    /// The recall paths reach their indexes, and the form they used to
    /// render did not.
    ///
    /// Every vector table here has carried an HNSW index since it was
    /// defined, and every recall path asked for neighbours with
    /// `<|k,COSINE|>` — the metric form, which makes the engine compare
    /// every row and ignore the index entirely. It answers correctly,
    /// so nothing ever failed; it just paid for four indexes and used
    /// none of them, which is invisible from the outside. `EXPLAIN`
    /// names the plan, and the plan is the whole difference.
    ///
    /// The metric form is pinned too. If a future engine starts
    /// routing it through the index, this says so by failing rather
    /// than leaving a justification standing that has stopped being
    /// true. The probe's vector is one element whatever the index's
    /// width: the planner resolves the index before it reads the
    /// literal.
    #[tokio::test]
    async fn every_vector_table_is_reachable_through_its_index() {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();

        for (table, column, index) in VECTOR_TABLES {
            let indexed = format!("SELECT * FROM {table} WHERE {column} <|4,64|> [0] EXPLAIN");
            let answer = store.client().query(&indexed).await.unwrap();
            assert!(
                reaches_index(plan_of(&answer), index),
                "{table}.{column} did not reach {index}: {answer}",
            );

            let exhaustive =
                format!("SELECT * FROM {table} WHERE {column} <|4,COSINE|> [0] EXPLAIN");
            let answer = store.client().query(&exhaustive).await.unwrap();
            assert!(
                !reaches_index(plan_of(&answer), index),
                "the metric form now reaches {index}, so the reason these repos \
                 render the indexed form has changed: {answer}",
            );
        }
    }

    #[tokio::test]
    async fn a_db_scoped_connect_refuses_config_credentials() {
        // Config credentials would sign the session in at ROOT on connect,
        // silently defeating the point of the database-scoped user.
        let config = ConnectionConfig::builder()
            .url("mem://")
            .namespace("antumbra")
            .database("main")
            .username("root")
            .password("root")
            .build()
            .unwrap();
        let Err(err) = Store::connect_with_db_user(config, "ctrl", "pw", EMBED_DIM).await else {
            panic!("config credentials must be refused");
        };
        assert!(err.to_string().contains("credential-less"));
    }

    #[tokio::test]
    async fn embedded_store_reports_its_dim_and_owner_access_falls_back() {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        assert_eq!(store.embed_dim(), EMBED_DIM);
        // On an embedded store there are no configured credentials, so owner
        // access (`signin_root`) takes the `invalidate` fallback; anonymous *is*
        // the owner there. Both are no-ops here but must not error.
        store.signin_root().await.unwrap();
        store.invalidate().await.unwrap();
        // Schema application is idempotent, safe to re-run.
        store.ensure_schema().await.unwrap();
    }
}
