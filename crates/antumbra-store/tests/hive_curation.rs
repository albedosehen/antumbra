//! Engine-enforced curation: contribution is the member's and
//! curation is the owner's, and neither is a convention the app remembers.
//!
//! The three denials that matter are all written as permission predicates, so
//! they hold whatever the app layer does: a member cannot open the tenant gate,
//! cannot opt another member in, and cannot accept their own offer. The last is
//! the load-bearing one -- acceptance is the only thing that puts a subject in
//! the active hive, so a member who could accept could publish into the org
//! unilaterally.

use chrono::Utc;

use antumbra_core::{
    is_open, Hive, HiveMembership, HiveOffer, OfferStatus, OfferedKind, Result, TenantId, UserId,
};
use antumbra_store::repo::{hive, principal};
use antumbra_store::Store;

const DIM: usize = 4;

struct Org {
    store: Store,
    tenant: TenantId,
    lily: UserId,
    oslo: UserId,
}

async fn org() -> Result<Org> {
    let store = Store::connect_memory(DIM).await?;
    let tenant = TenantId::new("ws:org");
    let lily = UserId::new("user:lily");
    let oslo = UserId::new("user:oslo");
    principal::provision(&store, &tenant, &lily).await?;
    principal::provision(&store, &tenant, &oslo).await?;
    Ok(Org {
        store,
        tenant,
        lily,
        oslo,
    })
}

fn offer_of(o: &Org, subject: &str, by: &UserId) -> HiveOffer {
    HiveOffer::new(
        o.tenant.clone(),
        OfferedKind::Compartment,
        subject,
        by.clone(),
        Utc::now(),
    )
}

/// The denial the whole design rests on. A member offers; only the owner rules.
#[tokio::test]
async fn a_member_cannot_accept_their_own_offer() -> Result<()> {
    let o = org().await?;
    let offered = offer_of(&o, "comp:brand", &o.lily);

    o.store.signin(&o.tenant, &o.lily).await?;
    hive::offer(&o.store, &offered).await?;
    // The engine refuses by persisting nothing, without an error.
    hive::rule(&o.store, &offered, OfferStatus::Accepted, Utc::now())
        .await
        .ok();

    o.store.invalidate().await?;
    let seen = hive::offers(&o.store, &o.tenant).await?;
    assert_eq!(seen.len(), 1, "the offer itself was hers to make");
    assert_eq!(
        seen[0].status,
        OfferStatus::Offered,
        "accepting is the owner's act, and the engine is what says so"
    );
    assert!(hive::accepted(&o.store, &o.tenant).await?.is_empty());

    // As owner, the same call rules.
    hive::rule(&o.store, &offered, OfferStatus::Accepted, Utc::now()).await?;
    assert_eq!(
        hive::accepted(&o.store, &o.tenant)
            .await?
            .iter()
            .map(|a| a.subject_id.as_str())
            .collect::<Vec<_>>(),
        vec!["comp:brand"]
    );
    Ok(())
}

/// A member offers as themselves and nobody else, so one member cannot volunteer
/// another's compartment into the org.
#[tokio::test]
async fn a_member_cannot_offer_on_someone_elses_behalf() -> Result<()> {
    let o = org().await?;
    o.store.signin(&o.tenant, &o.lily).await?;
    hive::offer(&o.store, &offer_of(&o, "comp:his", &o.oslo))
        .await
        .ok();

    o.store.invalidate().await?;
    assert!(
        hive::offers(&o.store, &o.tenant).await?.is_empty(),
        "lily cannot offer oslo's compartment under his name"
    );
    Ok(())
}

/// The two gates, from the engine's side. A member cannot open the tenant's, and
/// the owner cannot close a member's -- the second being the half that protects
/// the member from being conscripted.
#[tokio::test]
async fn neither_party_can_reach_across_and_set_the_others_gate() -> Result<()> {
    let o = org().await?;
    let now = Utc::now();

    // A member cannot open the tenant gate.
    o.store.signin(&o.tenant, &o.lily).await?;
    hive::set_enabled(&o.store, &Hive::new(o.tenant.clone(), true, now))
        .await
        .ok();
    o.store.invalidate().await?;
    assert!(
        hive::get(&o.store, &o.tenant).await?.is_none(),
        "opening the hive is the owner's decision"
    );

    // The owner opens it; a member joins for themselves.
    hive::set_enabled(&o.store, &Hive::new(o.tenant.clone(), true, now)).await?;
    o.store.signin(&o.tenant, &o.lily).await?;
    hive::set_membership(
        &o.store,
        &HiveMembership::new(o.tenant.clone(), o.lily.clone(), true, now),
    )
    .await?;
    // ...and cannot join anyone else.
    hive::set_membership(
        &o.store,
        &HiveMembership::new(o.tenant.clone(), o.oslo.clone(), true, now),
    )
    .await
    .ok();

    o.store.invalidate().await?;
    assert!(
        hive::membership(&o.store, &o.tenant, &o.oslo)
            .await?
            .is_none(),
        "one member cannot opt another in"
    );

    // Both gates, together, are what open it.
    let gate = hive::get(&o.store, &o.tenant).await?;
    assert!(is_open(
        gate.as_ref(),
        hive::membership(&o.store, &o.tenant, &o.lily)
            .await?
            .as_ref()
    ));
    assert!(
        !is_open(
            gate.as_ref(),
            hive::membership(&o.store, &o.tenant, &o.oslo)
                .await?
                .as_ref()
        ),
        "oslo never joined, so the hive is not open to him"
    );
    Ok(())
}

/// A member withdraws their own offer, which is the other half of contribution
/// being theirs: having offered is not a commitment they cannot take back.
#[tokio::test]
async fn a_member_can_withdraw_what_they_offered() -> Result<()> {
    let o = org().await?;
    let offered = offer_of(&o, "comp:brand", &o.lily);
    o.store.signin(&o.tenant, &o.lily).await?;
    hive::offer(&o.store, &offered).await?;
    assert_eq!(hive::offers(&o.store, &o.tenant).await?.len(), 1);

    hive::withdraw(&o.store, &offered).await?;
    assert!(
        hive::offers(&o.store, &o.tenant).await?.is_empty(),
        "what a member offered, a member can take back"
    );
    Ok(())
}
