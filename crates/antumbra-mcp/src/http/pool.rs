//! A workspace's shared pool: the compartments its members have offered to
//! everyone in it, and its admins have accepted ([`antumbra_core::hive`]).
//!
//! A compartment is its owner's alone until it is shared. Sharing with one
//! member is a grant; sharing with the workspace is this. It takes three
//! consents, and the engine's read rule asks for all three: the workspace's
//! owner has opened the pool, the member has joined it, and an admin has
//! accepted what the member offered. A member reads an accepted compartment
//! through their own token from then on, by recall as by listing.
//!
//! `GET /pool` is any member's to read: a pool whose contents were secret
//! from the people contributing to it could not be audited by them. `POST
//! /pool/join` and `/pool/offer` and `/pool/withdraw` are a member's own;
//! `/pool/rule` is an admin's or an owner's; `/pool/open` is an owner's.
//! None is an MCP tool, so no agent offers or accepts anything.

use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::Request;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use chrono::Utc;
use serde::Deserialize;
use serde_json::json;

use antumbra_core::membership::Role;
use antumbra_core::{
    AntumbraError, CompartmentId, Hive, HiveMembership, HiveOffer, OfferStatus, OfferedKind,
    TenantId, UserId,
};
use antumbra_store::repo::{compartment, hive, memory, principal, profile};

use super::members::{as_owner, caller, failed, token_of};
use super::{bad_request, HttpState};

/// The largest body read here.
const MAX_BODY: usize = 4 * 1024;
/// How many of an offered compartment's memories an admin is shown before
/// deciding, and how much of each.
const PREVIEW: usize = 5;
const PREVIEW_CHARS: usize = 240;

pub(super) fn routes() -> Router<Arc<HttpState>> {
    Router::new()
        .route("/pool", get(read))
        .route("/pool/open", post(open))
        .route("/pool/join", post(join))
        .route("/pool/offer", post(offer))
        .route("/pool/withdraw", post(withdraw))
        .route("/pool/rule", post(rule))
}

async fn body<T: serde::de::DeserializeOwned>(req: Request<Body>) -> Result<T, Response> {
    let bytes = axum::body::to_bytes(req.into_body(), MAX_BODY)
        .await
        .map_err(|_| bad_request("request body unreadable or too large"))?;
    serde_json::from_slice(&bytes).map_err(|e| bad_request(&format!("invalid request: {e}")))
}

/// The caller's workspace, user and role there.
async fn member(
    state: &HttpState,
    token: Option<String>,
    route: &str,
) -> Result<(TenantId, UserId, Option<Role>), Response> {
    let access = caller(state, token, route).await?;
    let (tenant, user) = (
        TenantId::new(&access.identity.tenant),
        UserId::new(&access.identity.user),
    );
    let runs_the_server = access.operator;
    let (t, u) = (tenant.clone(), user.clone());
    let role = as_owner(state, |store| async move {
        principal::role_of(&store, &t, &u, runs_the_server).await
    })
    .await
    .map_err(|e| failed(route, &e))?;
    Ok((tenant, user, role))
}

async fn read(State(state): State<Arc<HttpState>>, req: Request<Body>) -> Response {
    let (tenant, user, role) = match member(&state, token_of(&req), "/pool").await {
        Ok(who) => who,
        Err(response) => return response,
    };
    let rules = role.is_some_and(Role::admits);
    let (t, u) = (tenant.clone(), user.clone());
    let read = as_owner(&state, |store| async move {
        let opened = hive::get(&store, &t).await?.is_some_and(|h| h.enabled);
        let joined = hive::membership(&store, &t, &u)
            .await?
            .is_some_and(|m| m.opted_in);
        let profiles = profile::list(&store, &t).await?;
        let mut offers = Vec::new();
        for offered in hive::offers(&store, &t).await? {
            if offered.subject_kind != OfferedKind::Compartment {
                continue;
            }
            let id = CompartmentId::new(offered.subject_id.clone());
            let held = compartment::get(&store, &t, &id).await?;
            let memories = memory::list_by_compartment(&store, &t, &id).await?;
            // What becomes everyone's is shown to whoever decides it, and to
            // whoever offered it; an accepted one is everyone's already.
            let shows = rules || offered.offered_by == u || offered.status == OfferStatus::Accepted;
            let preview: Vec<String> = memories
                .iter()
                .filter(|_| shows)
                .take(PREVIEW)
                .map(|m| m.content.chars().take(PREVIEW_CHARS).collect())
                .collect();
            let by = profiles
                .iter()
                .find(|p| p.user == offered.offered_by)
                .and_then(|p| p.display.clone());
            offers.push(json!({
                "key": offered.id,
                "compartment": offered.subject_id,
                "name": held.map(|c| c.name),
                "offered_by": offered.offered_by.as_str(),
                "offered_by_name": by,
                "yours": offered.offered_by == u,
                "status": offered.status.as_str(),
                "memories": memories.len(),
                "preview": preview,
                "offered_at": offered.created_at.to_rfc3339(),
            }));
        }
        Ok((opened, joined, offers))
    })
    .await;
    match read {
        Ok((opened, joined, offers)) => Json(json!({
            "open": opened,
            "joined": joined,
            "your_role": role.map(Role::as_str),
            "offers": offers,
        }))
        .into_response(),
        Err(e) => failed("/pool", &e),
    }
}

#[derive(Deserialize)]
struct Switch {
    on: bool,
}

/// Open or close the workspace's pool: its owner's alone. Closed, nothing in
/// it is read by anyone but its own owner, and the offers stay as they were.
async fn open(State(state): State<Arc<HttpState>>, req: Request<Body>) -> Response {
    let (tenant, user, role) = match member(&state, token_of(&req), "/pool/open").await {
        Ok(who) => who,
        Err(response) => return response,
    };
    if role != Some(Role::Owner) {
        return failed(
            "/pool/open",
            &AntumbraError::rejected(
                "only an owner of this workspace opens or closes its shared pool",
            ),
        );
    }
    let switch: Switch = match body(req).await {
        Ok(switch) => switch,
        Err(response) => return response,
    };
    eprintln!(
        "antumbra-mcp: shared pool of {} {} by {}",
        tenant.as_str(),
        if switch.on { "opened" } else { "closed" },
        user.as_str()
    );
    match as_owner(&state, |store| async move {
        hive::set_enabled(&store, &Hive::new(tenant, switch.on, Utc::now())).await
    })
    .await
    {
        Ok(()) => Json(json!({ "open": switch.on })).into_response(),
        Err(e) => failed("/pool/open", &e),
    }
}

/// Join or leave the pool, as oneself. Left, what one offered stops being
/// read by the others at once, accepted or not.
async fn join(State(state): State<Arc<HttpState>>, req: Request<Body>) -> Response {
    let (tenant, user, _) = match member(&state, token_of(&req), "/pool/join").await {
        Ok(who) => who,
        Err(response) => return response,
    };
    let switch: Switch = match body(req).await {
        Ok(switch) => switch,
        Err(response) => return response,
    };
    match as_owner(&state, |store| async move {
        hive::set_membership(
            &store,
            &HiveMembership::new(tenant, user, switch.on, Utc::now()),
        )
        .await
    })
    .await
    {
        Ok(()) => Json(json!({ "joined": switch.on })).into_response(),
        Err(e) => failed("/pool/join", &e),
    }
}

#[derive(Deserialize)]
struct Offered {
    /// A compartment of the caller's, by id.
    compartment: String,
}

/// Offer one of one's own compartments to the workspace. Offering is joining:
/// someone who offers has said they share. It waits for an admin.
async fn offer(State(state): State<Arc<HttpState>>, req: Request<Body>) -> Response {
    let (tenant, user, _) = match member(&state, token_of(&req), "/pool/offer").await {
        Ok(who) => who,
        Err(response) => return response,
    };
    let offered: Offered = match body(req).await {
        Ok(offered) => offered,
        Err(response) => return response,
    };
    let made = as_owner(&state, |store| async move {
        let id = CompartmentId::new(offered.compartment.trim());
        let owned = compartment::get(&store, &tenant, &id)
            .await?
            .is_some_and(|c| c.owner == user);
        if !owned {
            return Err(AntumbraError::rejected(
                "that is not a compartment of yours: only its owner offers a compartment",
            ));
        }
        let now = Utc::now();
        // One compartment is one offer: asked again, the one there is stands.
        let key = HiveOffer::id_for(&tenant, OfferedKind::Compartment, id.as_str());
        if let Some(held) = hive::offers(&store, &tenant)
            .await?
            .into_iter()
            .find(|o| o.id == key)
        {
            return Ok(held);
        }
        hive::set_membership(
            &store,
            &HiveMembership::new(tenant.clone(), user.clone(), true, now),
        )
        .await?;
        let new = HiveOffer::new(tenant, OfferedKind::Compartment, id.as_str(), user, now);
        hive::offer(&store, &new).await?;
        Ok(new)
    })
    .await;
    match made {
        Ok(made) => Json(json!({ "key": made.id, "status": made.status.as_str() })).into_response(),
        Err(e) => failed("/pool/offer", &e),
    }
}

#[derive(Deserialize)]
struct Named {
    key: String,
    #[serde(default)]
    accept: Option<bool>,
}

/// The workspace's offer `key`, or the refusal that says there is none.
async fn offer_of(
    store: &antumbra_store::Store,
    tenant: &TenantId,
    key: &str,
) -> antumbra_core::Result<HiveOffer> {
    hive::offers(store, tenant)
        .await?
        .into_iter()
        .find(|o| o.id == key.trim())
        .ok_or_else(|| AntumbraError::rejected("no such offer in this workspace"))
}

/// Take back an offer of one's own, accepted or not.
async fn withdraw(State(state): State<Arc<HttpState>>, req: Request<Body>) -> Response {
    let (tenant, user, _) = match member(&state, token_of(&req), "/pool/withdraw").await {
        Ok(who) => who,
        Err(response) => return response,
    };
    let named: Named = match body(req).await {
        Ok(named) => named,
        Err(response) => return response,
    };
    match as_owner(&state, |store| async move {
        let held = offer_of(&store, &tenant, &named.key).await?;
        if held.offered_by != user {
            return Err(AntumbraError::rejected(
                "that offer is not yours: only who offered a compartment takes it back",
            ));
        }
        hive::withdraw(&store, &held).await
    })
    .await
    {
        Ok(()) => Json(json!({ "withdrawn": true })).into_response(),
        Err(e) => failed("/pool/withdraw", &e),
    }
}

/// Accept or decline an offer: an admin's or an owner's. Accepted, every
/// member of the pool reads the compartment; declined, nobody new does.
async fn rule(State(state): State<Arc<HttpState>>, req: Request<Body>) -> Response {
    let (tenant, user, role) = match member(&state, token_of(&req), "/pool/rule").await {
        Ok(who) => who,
        Err(response) => return response,
    };
    if !role.is_some_and(Role::admits) {
        return failed(
            "/pool/rule",
            &AntumbraError::rejected(
                "only an admin or an owner of this workspace accepts what is offered to it",
            ),
        );
    }
    let named: Named = match body(req).await {
        Ok(named) => named,
        Err(response) => return response,
    };
    let Some(accept) = named.accept else {
        return bad_request("`accept` says whether the offer is accepted");
    };
    eprintln!(
        "antumbra-mcp: offer {} {} by {}/{}",
        named.key.trim(),
        if accept { "accepted" } else { "declined" },
        tenant.as_str(),
        user.as_str()
    );
    let status = if accept {
        OfferStatus::Accepted
    } else {
        OfferStatus::Declined
    };
    match as_owner(&state, |store| async move {
        let held = offer_of(&store, &tenant, &named.key).await?;
        hive::rule(&store, &held, status, Utc::now()).await
    })
    .await
    {
        Ok(()) => Json(json!({ "status": status.as_str() })).into_response(),
        Err(e) => failed("/pool/rule", &e),
    }
}
