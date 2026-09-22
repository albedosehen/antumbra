//! The tenant hive's two gates and its offer ledger (ADR-0017 B).
//!
//! The interesting half of this module is what it does *not* offer. There is no
//! `accept` a member can call: `HIVE_OFFER_PERMS` denies update to every record
//! session, so acceptance is an owner-mode write and the engine enforces that
//! rather than this code remembering to. Likewise nothing here lets a session
//! open the tenant gate or opt another member in.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use surql::query::builder::Query;
use surql::query::crud::{query_records, upsert_record};
use surql::types::operators::{and_, eq};
use surql::types::RecordID;

use antumbra_core::{
    Hive, HiveMembership, HiveOffer, OfferStatus, OfferedKind, Result, TenantId, UserId,
};

use crate::dto::parse_dt;
use crate::error::map;
use crate::store::Store;

const HIVE: &str = "hive";
const MEMBERSHIP: &str = "hive_membership";
const OFFER: &str = "hive_offer";

fn sanitize(s: &str) -> String {
    s.replace([':', '/', '\\', '|', ' '], "_")
}

#[derive(Serialize, Deserialize)]
struct HiveRow {
    tenant_id: String,
    enabled: bool,
    updated_at: String,
}

#[derive(Serialize, Deserialize)]
struct MembershipRow {
    tenant_id: String,
    user: String,
    opted_in: bool,
    updated_at: String,
}

#[derive(Serialize, Deserialize)]
struct OfferRow {
    key: String,
    tenant_id: String,
    // Wire strings rather than enums, for the reason the device row keeps its
    // role as one: a kind or status a newer build writes must not fail the whole
    // listing out from under the ones this build understands.
    subject_kind: String,
    subject_id: String,
    offered_by: String,
    status: String,
    created_at: String,
    updated_at: String,
}

/// Open or close the tenant's gate. **Owner-mode only** -- `HIVE_PERMS` denies
/// every write to a record session, so a member calling this gets a refusal
/// from the engine rather than a decision.
pub async fn set_enabled(store: &Store, hive: &Hive) -> Result<()> {
    let rid = RecordID::<()>::new(HIVE, sanitize(hive.tenant_id.as_str()).as_str()).map_err(map)?;
    let data: Value = serde_json::to_value(HiveRow {
        tenant_id: hive.tenant_id.as_str().to_string(),
        enabled: hive.enabled,
        updated_at: hive.updated_at.to_rfc3339(),
    })?;
    upsert_record(store.client(), &rid, data)
        .await
        .map_err(map)?;
    Ok(())
}

/// The tenant's gate, or `None` if the owner has never ruled on it -- which
/// reads as closed.
pub async fn get(store: &Store, tenant: &TenantId) -> Result<Option<Hive>> {
    let query = Query::new()
        .select(None)
        .from_table(HIVE)
        .map_err(map)?
        .where_(eq("tenant_id", tenant.as_str()));
    let rows: Vec<HiveRow> = query_records(store.client(), &query).await.map_err(map)?;
    rows.into_iter()
        .next()
        .map(|r| {
            Ok(Hive {
                tenant_id: TenantId::new(r.tenant_id),
                enabled: r.enabled,
                updated_at: parse_dt(&r.updated_at)?,
            })
        })
        .transpose()
}

/// A member's own gate. Writable only by that member (`HIVE_MEMBERSHIP_PERMS`),
/// so an owner cannot opt anyone in on their behalf.
pub async fn set_membership(store: &Store, membership: &HiveMembership) -> Result<()> {
    let key = sanitize(&format!(
        "{}|{}",
        membership.tenant_id.as_str(),
        membership.user.as_str()
    ));
    let rid = RecordID::<()>::new(MEMBERSHIP, key.as_str()).map_err(map)?;
    let data: Value = serde_json::to_value(MembershipRow {
        tenant_id: membership.tenant_id.as_str().to_string(),
        user: membership.user.as_str().to_string(),
        opted_in: membership.opted_in,
        updated_at: membership.updated_at.to_rfc3339(),
    })?;
    upsert_record(store.client(), &rid, data)
        .await
        .map_err(map)?;
    Ok(())
}

/// One member's gate, or `None` if they have never ruled -- which reads as out.
pub async fn membership(
    store: &Store,
    tenant: &TenantId,
    user: &UserId,
) -> Result<Option<HiveMembership>> {
    let query = Query::new()
        .select(None)
        .from_table(MEMBERSHIP)
        .map_err(map)?
        .where_(and_(
            eq("tenant_id", tenant.as_str()),
            eq("user", user.as_str()),
        ));
    let rows: Vec<MembershipRow> = query_records(store.client(), &query).await.map_err(map)?;
    rows.into_iter()
        .next()
        .map(|r| {
            Ok(HiveMembership {
                tenant_id: TenantId::new(r.tenant_id),
                user: UserId::new(r.user),
                opted_in: r.opted_in,
                updated_at: parse_dt(&r.updated_at)?,
            })
        })
        .transpose()
}

/// Offer a subject into the hive. A member may only offer as themselves
/// (`HIVE_OFFER_PERMS` create), and the offer lands as `offered` -- never
/// `accepted`, because putting a subject in the active hive is the owner's act.
pub async fn offer(store: &Store, offer: &HiveOffer) -> Result<()> {
    put_offer(store, offer).await
}

/// Record the owner's ruling. **Owner-mode only**: update is `false` for every
/// record session, so a member cannot accept their own offer. That denial is
/// the whole curation boundary, and it lives in the engine rather than here.
pub async fn rule(
    store: &Store,
    offer: &HiveOffer,
    status: OfferStatus,
    now: chrono::DateTime<chrono::Utc>,
) -> Result<()> {
    put_offer(store, &offer.clone().with_status(status, now)).await
}

async fn put_offer(store: &Store, offer: &HiveOffer) -> Result<()> {
    let key = offer
        .id
        .split_once(':')
        .map(|(_, k)| k)
        .unwrap_or(offer.id.as_str());
    let rid = RecordID::<()>::new(OFFER, key).map_err(map)?;
    let data: Value = serde_json::to_value(OfferRow {
        key: offer.id.clone(),
        tenant_id: offer.tenant_id.as_str().to_string(),
        subject_kind: offer.subject_kind.as_str().to_string(),
        subject_id: offer.subject_id.clone(),
        offered_by: offer.offered_by.as_str().to_string(),
        status: offer.status.as_str().to_string(),
        created_at: offer.created_at.to_rfc3339(),
        updated_at: offer.updated_at.to_rfc3339(),
    })?;
    upsert_record(store.client(), &rid, data)
        .await
        .map_err(map)?;
    Ok(())
}

/// Every offer in the tenant, newest first. The whole offered set is readable by
/// every member, not just the owner: a hive whose contents were secret from the
/// people contributing to it could not be audited by them.
pub async fn offers(store: &Store, tenant: &TenantId) -> Result<Vec<HiveOffer>> {
    let query = Query::new()
        .select(None)
        .from_table(OFFER)
        .map_err(map)?
        .where_(eq("tenant_id", tenant.as_str()));
    let rows: Vec<OfferRow> = query_records(store.client(), &query).await.map_err(map)?;
    let mut found: Vec<HiveOffer> = rows
        .into_iter()
        .map(into_offer)
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect();
    found.sort_by_key(|o| std::cmp::Reverse(o.created_at));
    Ok(found)
}

/// The **active** hive: what the owner has accepted, and nothing else.
pub async fn accepted(store: &Store, tenant: &TenantId) -> Result<Vec<HiveOffer>> {
    Ok(offers(store, tenant)
        .await?
        .into_iter()
        .filter(|o| o.status.is_active())
        .collect())
}

/// `None` when the row names a kind or status this build cannot reason about.
fn into_offer(row: OfferRow) -> Result<Option<HiveOffer>> {
    let (Ok(subject_kind), Ok(status)) = (
        row.subject_kind.parse::<OfferedKind>(),
        row.status.parse::<OfferStatus>(),
    ) else {
        return Ok(None);
    };
    Ok(Some(HiveOffer {
        id: row.key,
        tenant_id: TenantId::new(row.tenant_id),
        subject_kind,
        subject_id: row.subject_id,
        offered_by: UserId::new(row.offered_by),
        status,
        created_at: parse_dt(&row.created_at)?,
        updated_at: parse_dt(&row.updated_at)?,
    }))
}

/// Withdraw an offer. A member may delete their own (`HIVE_OFFER_PERMS`), which
/// is the other half of contribution being theirs: having offered is not a
/// commitment they cannot take back.
pub async fn withdraw(store: &Store, offer: &HiveOffer) -> Result<()> {
    let condition = and_(
        eq("tenant_id", offer.tenant_id.as_str()),
        eq("key", offer.id.as_str()),
    );
    surql::query::crud::delete_records(store.client(), OFFER, Some(&condition))
        .await
        .map_err(map)?;
    Ok(())
}
