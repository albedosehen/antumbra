//! Engine-enforced document privacy (ADR-0020): a document kept in a private
//! compartment is invisible to another member of the same tenant until the
//! compartment is granted, and hidden again on revoke, on reads that carry no
//! compartment filter of their own. Before this, `document_chunk` carried the
//! tenant-wide rule, so every document was readable by every member.

use chrono::Utc;

use antumbra_core::{
    Capability, Compartment, CompartmentId, DocumentChunk, DocumentChunkId, Grant, Result,
    TenantId, UserId,
};
use antumbra_store::repo::{compartment, document, principal};
use antumbra_store::Store;

const DIM: usize = 4;

fn chunk(id: &str, title: &str, content: &str, compartment: Option<&str>) -> DocumentChunk {
    let c = DocumentChunk {
        id: DocumentChunkId::new(id),
        tenant: TenantId::new("ws:org"),
        title: title.into(),
        source: None,
        ordinal: 0,
        content: content.into(),
        embedding: Some(vec![1.0, 0.0, 0.0, 0.0]),
        created_at: Utc::now(),
        copal_file: None,
        copal_digest: None,
        compartment: None,
    };
    match compartment {
        Some(comp) => c.in_compartment(comp),
        None => c,
    }
}

/// What a session can recall by vector and list by title: the two reads the MCP
/// surface makes, neither of which filters by compartment.
async fn visible(store: &Store, tenant: &TenantId) -> Result<(Vec<String>, Vec<String>)> {
    let mut recalled: Vec<String> = document::recall(store, tenant, &[1.0, 0.0, 0.0, 0.0], 10)
        .await?
        .into_iter()
        .map(|c| c.content)
        .collect();
    recalled.sort();
    Ok((recalled, document::list_titles(store, tenant).await?))
}

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

#[tokio::test]
async fn a_private_document_is_recallable_only_by_its_audience() -> Result<()> {
    let o = org().await?;
    document::insert_chunks(
        &o.store,
        &[
            chunk(
                "docchunk:private",
                "review",
                "lily's salary review",
                Some("comp:lily-private"),
            ),
            chunk("docchunk:pool", "handbook", "the team handbook", None),
        ],
    )
    .await?;

    // Oslo, same tenant: the shared pool, and nothing of lily's.
    o.store.signin(&o.tenant, &o.oslo).await?;
    assert_eq!(
        visible(&o.store, &o.tenant).await?,
        (
            vec!["the team handbook".to_string()],
            vec!["handbook".to_string()]
        ),
        "a member must not recall another member's private document"
    );

    // Lily grants him the compartment: visible at once.
    o.store.invalidate().await?;
    compartment::grant(
        &o.store,
        &Grant::new(
            o.tenant.clone(),
            o.lilys.clone(),
            o.oslo.clone(),
            Capability::Reference,
            o.lily.clone(),
            Utc::now(),
        ),
    )
    .await?;
    o.store.signin(&o.tenant, &o.oslo).await?;
    assert_eq!(
        visible(&o.store, &o.tenant).await?.0,
        vec![
            "lily's salary review".to_string(),
            "the team handbook".to_string()
        ]
    );

    // Revoked: hidden again.
    o.store.invalidate().await?;
    compartment::revoke(&o.store, &o.tenant, &o.lilys, &o.oslo, Utc::now()).await?;
    o.store.signin(&o.tenant, &o.oslo).await?;
    assert_eq!(
        visible(&o.store, &o.tenant).await?.0,
        vec!["the team handbook".to_string()]
    );

    // Lily always sees her own and the pool.
    o.store.invalidate().await?;
    o.store.signin(&o.tenant, &o.lily).await?;
    assert_eq!(
        visible(&o.store, &o.tenant).await?.0,
        vec![
            "lily's salary review".to_string(),
            "the team handbook".to_string()
        ]
    );
    Ok(())
}

/// The write side. A member can neither plant a document in another member's
/// compartment nor erase one from it by naming its title, which matters because
/// ingest deletes a title before it writes the next generation.
#[tokio::test]
async fn a_member_cannot_write_into_or_delete_from_anothers_compartment() -> Result<()> {
    let o = org().await?;
    document::insert_chunks(
        &o.store,
        &[chunk(
            "docchunk:private",
            "review",
            "lily's salary review",
            Some("comp:lily-private"),
        )],
    )
    .await?;

    o.store.signin(&o.tenant, &o.oslo).await?;
    // The engine refuses by persisting nothing, without an error.
    document::insert_chunks(
        &o.store,
        &[chunk(
            "docchunk:planted",
            "planted",
            "oslo was here",
            Some("comp:lily-private"),
        )],
    )
    .await
    .ok();
    document::delete_title(&o.store, &o.tenant, "review", Some(&o.lilys))
        .await
        .ok();
    assert!(
        !document::title_exists(&o.store, &o.tenant, "planted", Some(&o.lilys)).await?,
        "which is why ingest looks before it reports chunks as stored"
    );

    // As owner, read what is really there.
    o.store.invalidate().await?;
    assert_eq!(
        visible(&o.store, &o.tenant).await?,
        (
            vec!["lily's salary review".to_string()],
            vec!["review".to_string()]
        ),
        "lily's document survived and nothing was planted beside it"
    );

    // And the app-side mirror of the rule agrees with the engine.
    for (user, expected) in [(&o.lily, true), (&o.oslo, false)] {
        assert_eq!(
            compartment::can_write(&o.store, &o.tenant, user, &o.lilys).await?,
            expected
        );
    }
    assert!(
        !compartment::can_write(
            &o.store,
            &o.tenant,
            &o.lily,
            &CompartmentId::new("comp:no-such")
        )
        .await?
    );
    Ok(())
}

/// A `reference` grant reads; only `link` writes.
#[tokio::test]
async fn only_a_link_grant_lets_a_grantee_write() -> Result<()> {
    let o = org().await?;
    for (capability, expected) in [(Capability::Reference, false), (Capability::Link, true)] {
        compartment::grant(
            &o.store,
            &Grant::new(
                o.tenant.clone(),
                o.lilys.clone(),
                o.oslo.clone(),
                capability,
                o.lily.clone(),
                Utc::now(),
            ),
        )
        .await?;
        assert_eq!(
            compartment::can_write(&o.store, &o.tenant, &o.oslo, &o.lilys).await?,
            expected,
            "{capability:?}"
        );
    }
    Ok(())
}
