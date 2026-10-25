//! Engine-enforced relic privacy: a relic in a private compartment is seen by
//! its owner, and by another member only once the compartment is granted; the
//! slug, title and versions are the owner's to change alone, even for a member
//! granted `link`; and no one in another tenant can take a slug that is held.

use chrono::Utc;

use antumbra_core::{
    Capability, Compartment, CompartmentId, Grant, Relic, RelicVersion, Result, TenantId, UserId,
    Visibility,
};
use antumbra_store::repo::{compartment, principal, relic};
use antumbra_store::Store;

const DIM: usize = 4;

struct Org {
    store: Store,
    tenant: TenantId,
    lily: UserId,
    oslo: UserId,
    lilys: CompartmentId,
}

/// Two members of one tenant; lily owns a private compartment.
async fn org() -> Result<Org> {
    let store = Store::connect_memory(DIM).await?;
    let tenant = TenantId::new("ws:org");
    let lily = UserId::new("user:lily");
    let oslo = UserId::new("user:oslo");
    let lilys = CompartmentId::new("comp:lily-private");
    principal::provision(&store, &tenant, &lily).await?;
    principal::provision(&store, &tenant, &oslo).await?;
    compartment::create(
        &store,
        &Compartment::new(
            lilys.clone(),
            tenant.clone(),
            lily.clone(),
            "lily private",
            Utc::now(),
        ),
    )
    .await?;
    Ok(Org {
        store,
        tenant,
        lily,
        oslo,
        lilys,
    })
}

fn page(o: &Org, owner: &UserId, title: &str) -> Relic {
    let now = Utc::now();
    Relic {
        slug: "ceiling".into(),
        tenant: o.tenant.clone(),
        owner: owner.clone(),
        compartment: o.lilys.clone(),
        title: title.into(),
        visibility: Visibility::Private,
        versions: vec![RelicVersion {
            number: 1,
            copal_file: "01aaa".into(),
            copal_digest: "d1".into(),
            created_at: now,
            author_host: None,
        }],
        source: None,
        created_at: now,
        updated_at: now,
    }
}

#[tokio::test]
async fn a_private_relic_is_seen_by_its_compartments_audience_only() -> Result<()> {
    let o = org().await?;
    o.store.signin(&o.tenant, &o.lily).await?;
    assert!(relic::put(&o.store, &page(&o, &o.lily, "The Ceiling")).await?);

    // Oslo, same tenant: nothing.
    o.store.invalidate().await?;
    o.store.signin(&o.tenant, &o.oslo).await?;
    assert!(relic::get(&o.store, &o.tenant, "ceiling").await?.is_none());
    assert!(relic::list(&o.store, &o.tenant).await?.is_empty());

    // Granted the compartment: seen at once.
    o.store.invalidate().await?;
    compartment::grant(
        &o.store,
        &Grant::new(
            o.tenant.clone(),
            o.lilys.clone(),
            o.oslo.clone(),
            Capability::Link,
            o.lily.clone(),
            Utc::now(),
        ),
    )
    .await?;
    o.store.signin(&o.tenant, &o.oslo).await?;
    assert_eq!(
        relic::get(&o.store, &o.tenant, "ceiling")
            .await?
            .map(|r| r.title),
        Some("The Ceiling".to_string())
    );

    // Even with `link`, the page is lily's: a rewrite by oslo does not land.
    let hijack = page(&o, &o.oslo, "Oslo's now");
    assert!(!relic::put(&o.store, &hijack).await?);
    let mut retitled = page(&o, &o.lily, "Retitled by oslo");
    retitled.visibility = Visibility::Public;
    assert!(!relic::put(&o.store, &retitled).await?);
    o.store.invalidate().await?;
    let stored = relic::get(&o.store, &o.tenant, "ceiling").await?.unwrap();
    assert_eq!(stored.title, "The Ceiling");
    assert_eq!(stored.owner, o.lily);
    assert_eq!(stored.visibility, Visibility::Private);

    // Revoked: hidden again.
    compartment::revoke(&o.store, &o.tenant, &o.lilys, &o.oslo, Utc::now()).await?;
    o.store.signin(&o.tenant, &o.oslo).await?;
    assert!(relic::get(&o.store, &o.tenant, "ceiling").await?.is_none());
    Ok(())
}

#[tokio::test]
async fn another_tenant_cannot_take_a_held_slug() -> Result<()> {
    let o = org().await?;
    o.store.signin(&o.tenant, &o.lily).await?;
    assert!(relic::put(&o.store, &page(&o, &o.lily, "The Ceiling")).await?);
    o.store.invalidate().await?;

    let elsewhere = TenantId::new("ws:elsewhere");
    let mallory = UserId::new("user:mallory");
    let theirs = CompartmentId::new("comp:mallory");
    principal::provision(&o.store, &elsewhere, &mallory).await?;
    compartment::create(
        &o.store,
        &Compartment::new(
            theirs.clone(),
            elsewhere.clone(),
            mallory.clone(),
            "m",
            Utc::now(),
        ),
    )
    .await?;
    o.store.signin(&elsewhere, &mallory).await?;
    let mut taken = page(&o, &mallory, "Mallory's ceiling");
    taken.tenant = elsewhere.clone();
    taken.compartment = theirs;
    assert!(!relic::put(&o.store, &taken).await?, "the slug is held");

    o.store.invalidate().await?;
    let stored = relic::get(&o.store, &o.tenant, "ceiling").await?.unwrap();
    assert_eq!(stored.owner, o.lily, "lily's relic is untouched");
    Ok(())
}
