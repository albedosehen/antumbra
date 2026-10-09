//! The active hive as a read layer, and the three conditions that have to
//! hold together for it.
//!
//! This is the hive's own validation criterion: "With the owner's tenant toggle
//! on and lily opted in, a compartment lily offers and the owner accepts is
//! readable by oslo's agent. One the owner has not accepted is not."
//!
//! Every other test of compartment privacy in this crate runs with no hive at
//! all, so none of them would notice this branch being too wide. That is what
//! this file is for: it is the only place the hive rule is switched on, so it is
//! the only place a mistake in it can show.

use chrono::Utc;

use antumbra_core::{
    Compartment, Hive, HiveMembership, HiveOffer, Memory, MemoryNetwork, OfferStatus, OfferedKind,
    Result, TenantId, UserId,
};
use antumbra_store::repo::{compartment, hive, memory, principal};
use antumbra_store::Store;

const DIM: usize = 4;

struct Org {
    store: Store,
    tenant: TenantId,
    lily: UserId,
    oslo: UserId,
}

/// Lily owns a compartment with one memory in it; oslo is another member who has
/// been granted nothing. Whatever oslo can see of `comp:brand` is the hive's
/// doing and nothing else's.
async fn org() -> Result<Org> {
    let store = Store::connect_memory(DIM).await?;
    let tenant = TenantId::new("ws:org");
    let lily = UserId::new("user:lily");
    let oslo = UserId::new("user:oslo");
    principal::provision(&store, &tenant, &lily).await?;
    principal::provision(&store, &tenant, &oslo).await?;
    let now = Utc::now();
    compartment::create(
        &store,
        &Compartment::new("comp:brand", tenant.clone(), lily.clone(), "brand", now),
    )
    .await?;
    compartment::create(
        &store,
        &Compartment::new("comp:private", tenant.clone(), lily.clone(), "private", now),
    )
    .await?;
    for (id, comp, content) in [
        ("memory:brand", "comp:brand", "the brand voice is dry"),
        ("memory:private", "comp:private", "lily's salary review"),
    ] {
        let mut m = Memory::new(id, tenant.clone(), MemoryNetwork::World, content, 0.9, now);
        m.compartment = Some(antumbra_core::CompartmentId::new(comp));
        m.embedding = Some(vec![1.0, 0.0, 0.0, 0.0]);
        memory::upsert(&store, &m).await?;
    }
    Ok(Org {
        store,
        tenant,
        lily,
        oslo,
    })
}

/// What oslo's own session can recall, which is what the engine decides.
async fn oslo_sees(o: &Org) -> Result<Vec<String>> {
    o.store.invalidate().await?;
    o.store.signin(&o.tenant, &o.oslo).await?;
    let mut seen: Vec<String> =
        memory::recall(&o.store, &o.tenant, &[1.0, 0.0, 0.0, 0.0], 10, None)
            .await?
            .into_iter()
            .map(|m| m.content)
            .collect();
    seen.sort();
    Ok(seen)
}

/// Open both gates and accept the offer, as owner.
async fn open_hive_and_accept(o: &Org, offer: &HiveOffer) -> Result<()> {
    let now = Utc::now();
    o.store.invalidate().await?;
    hive::set_enabled(&o.store, &Hive::new(o.tenant.clone(), true, now)).await?;
    hive::set_membership(
        &o.store,
        &HiveMembership::new(o.tenant.clone(), o.lily.clone(), true, now),
    )
    .await?;
    hive::offer(&o.store, offer).await?;
    hive::rule(&o.store, offer, OfferStatus::Accepted, now).await?;
    Ok(())
}

fn brand_offer(o: &Org) -> HiveOffer {
    HiveOffer::new(
        o.tenant.clone(),
        OfferedKind::Compartment,
        "comp:brand",
        o.lily.clone(),
        Utc::now(),
    )
}

#[tokio::test]
async fn an_accepted_compartment_is_readable_and_the_rest_of_lily_stays_private() -> Result<()> {
    let o = org().await?;
    assert!(
        oslo_sees(&o).await?.is_empty(),
        "before any hive, oslo sees none of lily's compartments"
    );

    open_hive_and_accept(&o, &brand_offer(&o)).await?;
    assert_eq!(
        oslo_sees(&o).await?,
        vec!["the brand voice is dry".to_string()],
        "the accepted compartment is the hive"
    );
    // The line that matters most: offering one compartment does not offer the
    // member. Her un-offered compartment is exactly as private as before.
    assert!(
        !oslo_sees(&o).await?.iter().any(|m| m.contains("salary")),
        "a member's own private, un-offered memory stays private"
    );
    Ok(())
}

/// Each gate alone, from the read side. All three conditions hold together or
/// the hive is shut -- which is the same statement `is_open` makes about the two
/// gates, now enforced by the engine on the read path.
#[tokio::test]
async fn every_condition_is_load_bearing() -> Result<()> {
    let offer_of = |o: &Org| brand_offer(o);

    // Offered but never accepted: a member cannot publish unilaterally.
    let o = org().await?;
    let now = Utc::now();
    hive::set_enabled(&o.store, &Hive::new(o.tenant.clone(), true, now)).await?;
    hive::set_membership(
        &o.store,
        &HiveMembership::new(o.tenant.clone(), o.lily.clone(), true, now),
    )
    .await?;
    hive::offer(&o.store, &offer_of(&o)).await?;
    assert!(
        oslo_sees(&o).await?.is_empty(),
        "offered is not accepted, and only accepted is the hive"
    );

    // Accepted, but the member never opted in: an owner cannot conscript.
    let o = org().await?;
    let offer = offer_of(&o);
    hive::set_enabled(&o.store, &Hive::new(o.tenant.clone(), true, now)).await?;
    hive::offer(&o.store, &offer).await?;
    hive::rule(&o.store, &offer, OfferStatus::Accepted, now).await?;
    assert!(
        oslo_sees(&o).await?.is_empty(),
        "accepting an offer from a member who has not joined shares nothing"
    );

    // Accepted and opted in, but the tenant gate is shut.
    let o = org().await?;
    let offer = offer_of(&o);
    hive::set_membership(
        &o.store,
        &HiveMembership::new(o.tenant.clone(), o.lily.clone(), true, now),
    )
    .await?;
    hive::offer(&o.store, &offer).await?;
    hive::rule(&o.store, &offer, OfferStatus::Accepted, now).await?;
    assert!(
        oslo_sees(&o).await?.is_empty(),
        "no hive was ever opened for this tenant"
    );
    Ok(())
}

/// Closing a gate closes the hive at once, because the rule is evaluated per
/// read rather than cached into a grant. A member who leaves takes their
/// contribution with them.
#[tokio::test]
async fn withdrawing_consent_takes_effect_immediately() -> Result<()> {
    let o = org().await?;
    let offer = brand_offer(&o);
    open_hive_and_accept(&o, &offer).await?;
    assert_eq!(oslo_sees(&o).await?.len(), 1);

    // Lily opts out, as herself.
    o.store.invalidate().await?;
    o.store.signin(&o.tenant, &o.lily).await?;
    hive::set_membership(
        &o.store,
        &HiveMembership::new(o.tenant.clone(), o.lily.clone(), false, Utc::now()),
    )
    .await?;
    assert!(
        oslo_sees(&o).await?.is_empty(),
        "leaving the hive withdraws what she contributed to it"
    );

    // She rejoins; the owner's acceptance still stands, so it returns.
    o.store.invalidate().await?;
    o.store.signin(&o.tenant, &o.lily).await?;
    hive::set_membership(
        &o.store,
        &HiveMembership::new(o.tenant.clone(), o.lily.clone(), true, Utc::now()),
    )
    .await?;
    assert_eq!(oslo_sees(&o).await?.len(), 1);

    // The owner closes the tenant gate: the whole hive shuts for everyone.
    o.store.invalidate().await?;
    hive::set_enabled(&o.store, &Hive::new(o.tenant.clone(), false, Utc::now())).await?;
    assert!(oslo_sees(&o).await?.is_empty());
    Ok(())
}

/// The hive is a read layer. Seeing another member's offered compartment does
/// not make it writable, so oslo cannot put anything into lily's brand voice.
#[tokio::test]
async fn the_hive_shares_reading_and_not_writing() -> Result<()> {
    let o = org().await?;
    open_hive_and_accept(&o, &brand_offer(&o)).await?;

    o.store.invalidate().await?;
    o.store.signin(&o.tenant, &o.oslo).await?;
    let mut planted = Memory::new(
        "memory:planted",
        o.tenant.clone(),
        MemoryNetwork::World,
        "oslo was here",
        0.9,
        Utc::now(),
    );
    planted.compartment = Some(antumbra_core::CompartmentId::new("comp:brand"));
    planted.embedding = Some(vec![1.0, 0.0, 0.0, 0.0]);
    // The engine refuses by persisting nothing, without an error.
    memory::upsert(&o.store, &planted).await.ok();

    assert_eq!(
        oslo_sees(&o).await?,
        vec!["the brand voice is dry".to_string()],
        "a shared read is not a shared write"
    );
    Ok(())
}
