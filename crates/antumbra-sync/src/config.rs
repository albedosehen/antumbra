//! Collector configuration: the two endpoints (local embedded, remote
//! authoritative) and the reconcile cadence / reconnect backoff.

use std::time::Duration;

use surql::connection::ConnectionConfig;

use antumbra_core::{AntumbraError, Result};
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

/// The collector's full configuration.
#[derive(Debug, Clone)]
pub struct SyncConfig {
    pub local: Endpoint,
    pub remote: Endpoint,
    /// Delay between reconcile cycles once connected.
    pub interval: Duration,
    /// Reconnect backoff bounds after a lost connection.
    pub min_backoff: Duration,
    pub max_backoff: Duration,
}

impl SyncConfig {
    pub fn new(local: Endpoint, remote: Endpoint) -> Self {
        Self {
            local,
            remote,
            interval: Duration::from_secs(15),
            min_backoff: Duration::from_millis(500),
            max_backoff: Duration::from_secs(30),
        }
    }

    pub fn with_interval(mut self, interval: Duration) -> Self {
        self.interval = interval;
        self
    }
}
