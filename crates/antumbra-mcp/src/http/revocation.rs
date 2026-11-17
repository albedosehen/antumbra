//! The token denylist the networked surface checks every request's token id
//! against. The store's `revoked_token` table is where a revocation lands
//! (`antumbra revoke-token`, through the owner connection); this module reads
//! it into the server's set at startup and every [`REFRESH`], in the owner
//! view under the auth lock like the other background passes, sweeping the
//! rows whose token has expired on its own. A request's check is a set
//! lookup, so the hot path pays nothing for the table.

use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;

use antumbra_store::repo::revocation;

use super::HttpState;

/// How long a revocation written to the store takes to reach every server at
/// most: the interval between two reads of the table.
pub(crate) const REFRESH: Duration = Duration::from_secs(30);

/// Read the list once now, then every [`REFRESH`] for as long as the server
/// runs. A read that fails is said on stderr and tried again next time; the
/// set keeps what it had.
pub(super) fn spawn(state: Arc<HttpState>) {
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(REFRESH).await;
            if let Err(e) = refresh(&state).await {
                eprintln!("antumbra-mcp: token denylist not refreshed: {e:#}");
            }
        }
    });
}

/// Replace the server's denylist with the table's live rows, sweeping the
/// expired ones on the way; how many ids are on the list after.
pub(super) async fn refresh(state: &HttpState) -> anyhow::Result<usize> {
    let now = Utc::now();
    let live = {
        let _guard = state.auth.lock().await;
        state.store.signin_root().await?;
        revocation::sweep_expired(&state.store, now).await?;
        revocation::live(&state.store, now).await?
    };
    // A poisoned lock holds a whole set: nothing writes it in two steps.
    let mut set = state
        .revoked
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *set = live.into_iter().collect();
    Ok(set.len())
}
