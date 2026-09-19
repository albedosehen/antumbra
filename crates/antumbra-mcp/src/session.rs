//! Keeping a record session alive. The store's tenant access defines a session
//! duration ([`antumbra_store::TENANT_SESSION`]); a connection signed in as a
//! record and then reused for longer answers every query with "the session has
//! expired". The stdio server signs in once for the life of the process, and
//! the networked server keeps one signed-in connection per identity, so both
//! outlive that duration in ordinary use. A [`SessionKeeper`] re-signs the
//! connection in before it goes stale, checked at every tool call, so the
//! expiry is never observed.

use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::Mutex;

use antumbra_core::{TenantId, UserId};
use antumbra_store::{Store, TENANT_SESSION};

/// Re-sign in once a session is this old: a third of its duration, so a
/// refresh is never a close call and a clock that drifts a little changes
/// nothing.
pub const REFRESH_AFTER: Duration = Duration::from_secs(TENANT_SESSION.as_secs() / 3);

/// One record-signed connection and when it was last signed in.
pub struct SessionKeeper {
    store: Store,
    tenant: TenantId,
    user: UserId,
    signed_in_at: Mutex<Instant>,
}

impl SessionKeeper {
    /// For a `store` the caller has just signed in as `(tenant, user)`.
    pub fn new(store: Store, tenant: TenantId, user: UserId) -> Arc<Self> {
        Arc::new(Self {
            store,
            tenant,
            user,
            signed_in_at: Mutex::new(Instant::now()),
        })
    }

    /// The connection this keeper keeps signed in.
    pub fn store(&self) -> &Store {
        &self.store
    }

    /// Whether a session signed in at `signed_in_at` should be refreshed now.
    pub fn is_stale(signed_in_at: Instant, now: Instant) -> bool {
        now.saturating_duration_since(signed_in_at) >= REFRESH_AFTER
    }

    /// Note that someone else just signed the connection in (a JSON-RPC
    /// session's initialize does), so the clock restarts.
    pub async fn mark_signed_in(&self) {
        *self.signed_in_at.lock().await = Instant::now();
    }

    /// Re-sign in when the session is old enough; a no-op otherwise. Returns
    /// whether a refresh happened. Held under the keeper's lock so concurrent
    /// calls refresh once.
    pub async fn refresh_if_stale(&self) -> antumbra_core::Result<bool> {
        let mut signed_in_at = self.signed_in_at.lock().await;
        if !Self::is_stale(*signed_in_at, Instant::now()) {
            return Ok(false);
        }
        self.store.signin(&self.tenant, &self.user).await?;
        *signed_in_at = Instant::now();
        Ok(true)
    }

    /// Pretend the session was signed in `age` ago.
    #[cfg(test)]
    pub async fn backdate(&self, age: Duration) {
        *self.signed_in_at.lock().await = Instant::now() - age;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antumbra_store::repo::principal;
    use antumbra_store::EMBED_DIM;

    #[test]
    fn staleness_is_a_third_of_the_session() {
        assert_eq!(REFRESH_AFTER, Duration::from_secs(20 * 60));
        assert!(REFRESH_AFTER < TENANT_SESSION);
        let now = Instant::now();
        assert!(!SessionKeeper::is_stale(now, now));
        assert!(!SessionKeeper::is_stale(
            now - (REFRESH_AFTER - Duration::from_secs(1)),
            now
        ));
        assert!(SessionKeeper::is_stale(now - REFRESH_AFTER, now));
        assert!(SessionKeeper::is_stale(now - TENANT_SESSION, now));
    }

    #[tokio::test]
    async fn a_stale_session_is_signed_in_again_and_a_fresh_one_left_alone() {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        let tenant = TenantId::new("ws:t");
        let user = UserId::new("user:u");
        principal::provision(&store, &tenant, &user).await.unwrap();
        store.signin(&tenant, &user).await.unwrap();
        let keeper = SessionKeeper::new(store, tenant, user);
        assert!(!keeper.refresh_if_stale().await.unwrap(), "just signed in");
        keeper
            .backdate(REFRESH_AFTER + Duration::from_secs(1))
            .await;
        assert!(
            keeper.refresh_if_stale().await.unwrap(),
            "stale: re-signed in"
        );
        assert!(!keeper.refresh_if_stale().await.unwrap(), "fresh again");
        keeper.backdate(TENANT_SESSION).await;
        keeper.mark_signed_in().await;
        assert!(
            !keeper.refresh_if_stale().await.unwrap(),
            "an initialize restarted the clock"
        );
    }
}
