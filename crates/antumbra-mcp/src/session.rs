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
        self.refresh_if_stale_at(Instant::now()).await
    }

    /// `refresh_if_stale` as if it were `now`. Time is only ever moved
    /// forward from a real instant, never back: an `Instant` less than the
    /// host's uptime before now cannot be made, so a test that backdated a
    /// session by an hour panicked on a machine that had booted an hour ago.
    async fn refresh_if_stale_at(&self, now: Instant) -> antumbra_core::Result<bool> {
        let mut signed_in_at = self.signed_in_at.lock().await;
        if !Self::is_stale(*signed_in_at, now) {
            return Ok(false);
        }
        self.store.signin(&self.tenant, &self.user).await?;
        *signed_in_at = now;
        Ok(true)
    }

    /// `mark_signed_in` as if it were `now`.
    #[cfg(test)]
    async fn mark_signed_in_at(&self, now: Instant) {
        *self.signed_in_at.lock().await = now;
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
        // Forward from a real instant only: one less than the host's uptime
        // before now cannot be made.
        let then = Instant::now();
        assert!(!SessionKeeper::is_stale(then, then));
        assert!(!SessionKeeper::is_stale(
            then,
            then + (REFRESH_AFTER - Duration::from_secs(1))
        ));
        assert!(SessionKeeper::is_stale(then, then + REFRESH_AFTER));
        assert!(SessionKeeper::is_stale(then, then + TENANT_SESSION));
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
        let later = Instant::now() + REFRESH_AFTER + Duration::from_secs(1);
        assert!(
            keeper.refresh_if_stale_at(later).await.unwrap(),
            "stale: re-signed in"
        );
        assert!(
            !keeper.refresh_if_stale_at(later).await.unwrap(),
            "fresh again"
        );
        // Stale by the old clock, fresh once an initialize restarts it.
        let much_later = later + TENANT_SESSION;
        keeper.mark_signed_in_at(much_later).await;
        assert!(
            !keeper.refresh_if_stale_at(much_later).await.unwrap(),
            "an initialize restarted the clock"
        );
    }
}
