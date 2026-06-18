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
use crate::schema::{self, EMBED_DIM, TENANT_ACCESS};

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
        const PAGE: i64 = 20_000;
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
                .limit(PAGE)
                .map_err(map)?
                .offset(offset)
                .map_err(map)?;
            let page: Vec<T> = query_records(self.client(), &query).await.map_err(map)?;
            let n = page.len() as i64;
            out.extend(page);
            if n < PAGE {
                break;
            }
            offset += PAGE;
        }
        Ok(out)
    }
}

pub const DEFAULT_EMBED_DIM: usize = EMBED_DIM;

#[cfg(test)]
mod tests {
    use super::*;

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
