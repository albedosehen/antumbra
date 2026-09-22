//! Collector configuration: the two endpoints (local embedded, remote
//! authoritative) and the reconcile cadence / reconnect backoff.

use std::time::Duration;

use surql::connection::ConnectionConfig;

use antumbra_core::{AntumbraError, Result, TenantId, UserId};
use antumbra_store::{Store, EMBED_DIM};

/// One side of the sync: a SurrealDB endpoint. `username`/`password` are the
/// root credentials the collector signs in with so it spans tenants (it
/// replicates every tenant's rows; per-tenant isolation is preserved by the
/// `tenant_id` each row carries). Embedded `surrealkv://` / `mem://` endpoints
/// are owner sessions and need no credentials.
#[derive(Debug, Clone)]
pub struct Endpoint {
    pub url: String,
    pub namespace: String,
    pub database: String,
    pub username: Option<String>,
    pub password: Option<String>,
}

impl Endpoint {
    /// An embedded/owner endpoint (no credentials).
    pub fn embedded(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            namespace: "antumbra".to_string(),
            database: "main".to_string(),
            username: None,
            password: None,
        }
    }

    /// A networked endpoint with root credentials.
    pub fn authoritative(
        url: impl Into<String>,
        username: impl Into<String>,
        password: impl Into<String>,
    ) -> Self {
        Self {
            url: url.into(),
            namespace: "antumbra".to_string(),
            database: "main".to_string(),
            username: Some(username.into()),
            password: Some(password.into()),
        }
    }

    fn connection_config(&self) -> Result<ConnectionConfig> {
        let mut builder = ConnectionConfig::builder()
            .url(&self.url)
            .namespace(&self.namespace)
            .database(&self.database);
        if let (Some(user), Some(pass)) = (&self.username, &self.password) {
            builder = builder.username(user).password(pass);
        }
        builder
            .build()
            .map_err(|e| AntumbraError::other(format!("sync endpoint config: {e}")))
    }

    /// Connect (and apply the schema) to this endpoint.
    pub async fn connect(&self) -> Result<Store> {
        Store::connect(self.connection_config()?, EMBED_DIM).await
    }
}

/// Whose fabric a collector replicates (ADR-0017 A1: "A tenant has many users,
/// and each user has their own fabric").
///
/// Set it and the collector signs both connections in as that user, so the
/// engine scopes what it can read and, more to the point, what it can write.
/// Leave it unset and the collector runs as owner over the whole tenant, which
/// is what every deployment does today.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fabric {
    pub tenant: TenantId,
    pub user: UserId,
}

impl Fabric {
    pub fn new(tenant: impl Into<String>, user: impl Into<String>) -> Self {
        Self {
            tenant: TenantId::new(tenant),
            user: UserId::new(user),
        }
    }

    /// Sign a connection in as this user. The principal must already exist --
    /// provisioning is owner work and a collector is not an owner, so a missing
    /// principal is a configuration error and should read as one.
    pub async fn bind(&self, store: &Store) -> Result<()> {
        store.signin(&self.tenant, &self.user).await
    }
}

/// The collector's full configuration.
#[derive(Debug, Clone)]
pub struct SyncConfig {
    pub local: Endpoint,
    pub remote: Endpoint,
    /// Whose fabric to replicate. `None` is owner mode over the whole tenant,
    /// which is what every deployment runs today.
    ///
    /// Set it and the collector carries one user's fabric: their own
    /// compartments and the shared pool, which is both the privacy answer and
    /// exactly what their session can write back. `ReconcileStats::refused`
    /// should stay at zero -- a refusal on a narrowed table means the
    /// replication policy and the engine ACL disagree.
    pub fabric: Option<Fabric>,
    /// Delay between reconcile cycles once connected.
    pub interval: Duration,
    /// Reconnect backoff bounds after a lost connection.
    pub min_backoff: Duration,
    pub max_backoff: Duration,
    /// The incremental-cursor lookback window (the CDC "delay"): each cycle re-
    /// includes rows whose version is within this much of the watermark, so a
    /// write stamped slightly in the past (clock skew, a late commit) is not
    /// skipped. Re-reconciling settled rows is a no-op, so this only costs a small
    /// overlapping fetch. Keep it well under `interval`.
    pub lookback: Duration,
    /// Tombstone garbage-collection grace window: a soft-deleted row is hard-
    /// purged only once its deletion is older than this, so every replica has
    /// reconciled the tombstone first (resurrection-safe). Keep it far wider than
    /// `interval`.
    pub gc_grace: Duration,
    /// Run tombstone GC every this many reconcile cycles (`0` disables it). At the
    /// default `interval` this is roughly hourly.
    pub gc_every: usize,
}

impl SyncConfig {
    pub fn new(local: Endpoint, remote: Endpoint) -> Self {
        Self {
            local,
            remote,
            fabric: None,
            interval: Duration::from_secs(15),
            min_backoff: Duration::from_millis(500),
            max_backoff: Duration::from_secs(30),
            lookback: Duration::from_secs(5),
            gc_grace: Duration::from_secs(24 * 60 * 60),
            gc_every: 240,
        }
    }

    pub fn with_interval(mut self, interval: Duration) -> Self {
        self.interval = interval;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_endpoint_carries_no_credentials() {
        let e = Endpoint::embedded("mem://");
        assert_eq!(e.url, "mem://");
        assert_eq!(e.namespace, "antumbra");
        assert_eq!(e.database, "main");
        assert!(e.username.is_none() && e.password.is_none());
        // Builds a valid connection config (no creds branch).
        assert!(e.connection_config().is_ok());
    }

    #[test]
    fn authoritative_endpoint_carries_root_credentials() {
        let e = Endpoint::authoritative("ws://h:8000/rpc", "root", "pw");
        assert_eq!(e.username.as_deref(), Some("root"));
        assert_eq!(e.password.as_deref(), Some("pw"));
        // Builds a valid connection config (creds branch).
        assert!(e.connection_config().is_ok());
    }

    #[tokio::test]
    async fn embedded_endpoint_connects_and_applies_schema() {
        let store = Endpoint::embedded("mem://").connect().await.unwrap();
        // A connected store can be queried (schema applied).
        assert_eq!(
            antumbra_store::repo::sync::list_rows(&store, "memory")
                .await
                .unwrap()
                .len(),
            0
        );
    }

    #[test]
    fn sync_config_defaults_and_with_interval() {
        let cfg = SyncConfig::new(Endpoint::embedded("mem://"), Endpoint::embedded("mem://"));
        assert_eq!(cfg.interval, Duration::from_secs(15));
        assert_eq!(cfg.min_backoff, Duration::from_millis(500));
        assert_eq!(cfg.max_backoff, Duration::from_secs(30));
        assert_eq!(cfg.lookback, Duration::from_secs(5));
        assert_eq!(cfg.gc_grace, Duration::from_secs(86_400));
        assert_eq!(cfg.gc_every, 240);
        let cfg = cfg.with_interval(Duration::from_secs(2));
        assert_eq!(cfg.interval, Duration::from_secs(2));
    }
}

#[cfg(test)]
mod fabric_tests {
    use super::*;
    use antumbra_store::repo::principal;

    /// Binding is a real sign-in against the engine, not a stored label: an
    /// unprovisioned principal is a configuration error and has to read as one,
    /// because the alternative is a collector that silently runs as whatever it
    /// was already signed in as.
    #[tokio::test]
    async fn binding_needs_a_principal_that_exists() -> Result<()> {
        let store = Store::connect_memory(EMBED_DIM).await?;
        let fabric = Fabric::new("ws:org", "user:lily");
        assert!(
            fabric.bind(&store).await.is_err(),
            "an unprovisioned user cannot be bound"
        );

        principal::provision(&store, &fabric.tenant, &fabric.user).await?;
        fabric.bind(&store).await?;
        Ok(())
    }

    /// The default is owner mode over the whole tenant, which is what every
    /// deployment runs today. Scoping is opt-in and stays that way until the
    /// per-table replication scope exists to make it lossless.
    #[test]
    fn a_collector_is_unscoped_unless_it_is_told_otherwise() {
        let cfg = SyncConfig::new(Endpoint::embedded("mem://"), Endpoint::embedded("mem://"));
        assert_eq!(cfg.fabric, None);
    }
}
