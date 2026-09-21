//! Genesis requests: the work one node in a user's fabric could not run, left
//! where the machine that can will find it (ADR-0017 A2).
//!
//! Keyed per (tenant, user, compartment), so a compartment that goes on being
//! reinforced on a memory node asks once. Asking again while the request is
//! open is a no-op that keeps the original `created_at`, which is what says how
//! long the work has been waiting; asking again after one has run re-opens it,
//! because the compartment has moved on since.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use surql::query::builder::Query;
use surql::query::crud::{query_records, upsert_record};
use surql::types::operators::{and_, eq};
use surql::types::RecordID;

use antumbra_core::{CompartmentId, GenesisRequest, GenesisStatus, Result, TenantId, UserId};

use crate::dto::parse_dt;
use crate::error::map;
use crate::store::Store;

const GENESIS_REQUEST: &str = "genesis_request";

#[derive(Serialize, Deserialize)]
struct RequestRow {
    key: String,
    tenant_id: String,
    user: String,
    compartment: String,
    from_host: String,
    to_host: String,
    // The wire string rather than the enum, for the reason the device row keeps
    // its role as one: a status a newer node writes must not fail the listing
    // out from under the ones this build understands.
    status: String,
    created_at: String,
    updated_at: String,
}

impl RequestRow {
    fn from_domain(r: &GenesisRequest) -> Self {
        RequestRow {
            key: r.id.clone(),
            tenant_id: r.tenant_id.as_str().to_string(),
            user: r.user.as_str().to_string(),
            compartment: r.compartment.as_str().to_string(),
            from_host: r.from_host.clone(),
            to_host: r.to_host.clone(),
            status: r.status.as_str().to_string(),
            created_at: r.created_at.to_rfc3339(),
            updated_at: r.updated_at.to_rfc3339(),
        }
    }

    /// `None` when the row is in a state this build cannot reason about.
    fn into_domain(self) -> Result<Option<GenesisRequest>> {
        let Ok(status) = self.status.parse::<GenesisStatus>() else {
            return Ok(None);
        };
        Ok(Some(GenesisRequest {
            id: self.key,
            tenant_id: TenantId::new(self.tenant_id),
            user: UserId::new(self.user),
            compartment: CompartmentId::new(self.compartment),
            from_host: self.from_host,
            to_host: self.to_host,
            status,
            created_at: parse_dt(&self.created_at)?,
            updated_at: parse_dt(&self.updated_at)?,
        }))
    }
}

fn record_key(request: &GenesisRequest) -> &str {
    request
        .id
        .split_once(':')
        .map(|(_, key)| key)
        .unwrap_or(request.id.as_str())
}

async fn put(store: &Store, request: &GenesisRequest) -> Result<()> {
    let rid = RecordID::<()>::new(GENESIS_REQUEST, record_key(request)).map_err(map)?;
    let data: Value = serde_json::to_value(RequestRow::from_domain(request))?;
    upsert_record(store.client(), &rid, data)
        .await
        .map_err(map)?;
    Ok(())
}

/// Ask for a compartment to be trained somewhere else. Returns the request as
/// it now stands: the one already waiting if there is one, otherwise a fresh
/// one. Idempotent while the request is open, so a reinforced compartment does
/// not ask once per write.
pub async fn ask(store: &Store, request: &GenesisRequest) -> Result<GenesisRequest> {
    if let Some(open) = get(
        store,
        &request.tenant_id,
        &request.user,
        &request.compartment,
    )
    .await?
    .filter(GenesisRequest::is_open)
    {
        return Ok(open);
    }
    put(store, request).await?;
    Ok(request.clone())
}

/// One compartment's request, whatever state it is in.
pub async fn get(
    store: &Store,
    tenant: &TenantId,
    user: &UserId,
    compartment: &CompartmentId,
) -> Result<Option<GenesisRequest>> {
    let key = GenesisRequest::id_for(tenant, user, compartment);
    Ok(list_for_user(store, tenant, user)
        .await?
        .into_iter()
        .find(|r| r.id == key))
}

/// Every request a user's fabric has raised, newest first.
pub async fn list_for_user(
    store: &Store,
    tenant: &TenantId,
    user: &UserId,
) -> Result<Vec<GenesisRequest>> {
    let query = Query::new()
        .select(None)
        .from_table(GENESIS_REQUEST)
        .map_err(map)?
        .where_(and_(
            eq("tenant_id", tenant.as_str()),
            eq("user", user.as_str()),
        ));
    let rows: Vec<RequestRow> = query_records(store.client(), &query).await.map_err(map)?;
    let mut found: Vec<GenesisRequest> = rows
        .into_iter()
        .map(RequestRow::into_domain)
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect();
    found.sort_by_key(|r| std::cmp::Reverse(r.created_at));
    Ok(found)
}

/// The work waiting for a trainer: asked or taken, oldest first, because the
/// run that has been waiting longest should be the next one done.
pub async fn list_open_for_user(
    store: &Store,
    tenant: &TenantId,
    user: &UserId,
) -> Result<Vec<GenesisRequest>> {
    let mut open: Vec<GenesisRequest> = list_for_user(store, tenant, user)
        .await?
        .into_iter()
        .filter(GenesisRequest::is_open)
        .collect();
    open.reverse();
    Ok(open)
}

/// Move a request on: claimed when a node takes it, done when it has run.
pub async fn set_status(
    store: &Store,
    request: &GenesisRequest,
    status: GenesisStatus,
    now: chrono::DateTime<chrono::Utc>,
) -> Result<()> {
    put(store, &request.clone().with_status(status, now)).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::EMBED_DIM;

    async fn store() -> Result<Store> {
        Store::connect_memory(EMBED_DIM).await
    }

    fn request(compartment: &str, at: chrono::DateTime<chrono::Utc>) -> GenesisRequest {
        GenesisRequest::new(
            TenantId::new("ws:t"),
            UserId::new("user:a"),
            CompartmentId::new(compartment),
            "laptop",
            "rig",
            at,
        )
    }

    #[tokio::test]
    async fn a_compartment_asked_about_twice_waits_once() -> Result<()> {
        let s = store().await?;
        let tenant = TenantId::new("ws:t");
        let user = UserId::new("user:a");
        let first = chrono::Utc::now();

        let asked = ask(&s, &request("comp:rust", first)).await?;
        // Reinforced again a minute later: the same request, still waiting from
        // when it was first raised.
        let again = ask(
            &s,
            &request("comp:rust", first + chrono::Duration::minutes(1)),
        )
        .await?;
        assert_eq!(again.id, asked.id);
        assert_eq!(again.created_at, first, "the wait is measured from the ask");
        assert_eq!(list_open_for_user(&s, &tenant, &user).await?.len(), 1);

        // Run it, and the compartment can ask again afterwards, because it has
        // moved on since.
        set_status(&s, &again, GenesisStatus::Done, first).await?;
        assert!(list_open_for_user(&s, &tenant, &user).await?.is_empty());
        let later = first + chrono::Duration::hours(2);
        let reopened = ask(&s, &request("comp:rust", later)).await?;
        assert_eq!(reopened.status, GenesisStatus::Pending);
        assert_eq!(reopened.created_at, later);
        Ok(())
    }

    #[tokio::test]
    async fn the_run_that_has_waited_longest_is_the_next_one() -> Result<()> {
        let s = store().await?;
        let tenant = TenantId::new("ws:t");
        let user = UserId::new("user:a");
        let now = chrono::Utc::now();

        ask(&s, &request("comp:surql", now - chrono::Duration::hours(3))).await?;
        ask(&s, &request("comp:rust", now)).await?;
        ask(&s, &request("comp:tui", now - chrono::Duration::hours(1))).await?;

        let queue = list_open_for_user(&s, &tenant, &user).await?;
        assert_eq!(
            queue
                .iter()
                .map(|r| r.compartment.as_str())
                .collect::<Vec<_>>(),
            vec!["comp:surql", "comp:tui", "comp:rust"]
        );

        // Taken is not finished: a claimed run stays in the queue until it has
        // actually run, so a trainer that dies does not lose the work.
        set_status(&s, &queue[0], GenesisStatus::Claimed, now).await?;
        assert_eq!(list_open_for_user(&s, &tenant, &user).await?.len(), 3);
        Ok(())
    }

    #[tokio::test]
    async fn a_status_this_build_cannot_reason_about_is_skipped() -> Result<()> {
        let s = store().await?;
        let tenant = TenantId::new("ws:t");
        let user = UserId::new("user:a");
        ask(&s, &request("comp:rust", chrono::Utc::now())).await?;
        crate::repo::sync::put_row(
            &s,
            &serde_json::json!({
                "id": "genesis_request:from_the_future",
                "key": "genesis_request:from_the_future",
                "tenant_id": "ws:t",
                "user": "user:a",
                "compartment": "comp:later",
                "from_host": "laptop",
                "to_host": "rig",
                "status": "deferred",
                "created_at": chrono::Utc::now().to_rfc3339(),
                "updated_at": chrono::Utc::now().to_rfc3339(),
            }),
        )
        .await?;

        assert_eq!(
            list_for_user(&s, &tenant, &user)
                .await?
                .iter()
                .map(|r| r.compartment.as_str())
                .collect::<Vec<_>>(),
            vec!["comp:rust"]
        );
        Ok(())
    }
}
