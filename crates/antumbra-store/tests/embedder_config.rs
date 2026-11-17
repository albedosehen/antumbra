//! Per-workspace embedder config: one row per tenant, upsert/get/delete, with
//! independent workspaces (a tenant with no row falls back to the server default).

use anyhow::Result;

use antumbra_core::TenantId;
use antumbra_store::repo::embedder_config::{self, EmbedderConfig};
use antumbra_store::testing;

#[tokio::test]
async fn embedder_config_roundtrips_per_tenant() -> Result<()> {
    let store = testing::mem_store().await?;
    let alpha = TenantId::new("ws:alpha");

    // Unset -> None (the caller falls back to the server default).
    assert!(embedder_config::get(&store, &alpha).await?.is_none());

    embedder_config::upsert(
        &store,
        &EmbedderConfig {
            tenant_id: "ws:alpha".into(),
            url: "http://a:11434/v1/embeddings".into(),
            model: "all-minilm".into(),
            api_key: Some("k".into()),
            source_dim: None,
        },
    )
    .await?;

    let Some(got) = embedder_config::get(&store, &alpha).await? else {
        anyhow::bail!("the config was not set");
    };
    assert_eq!(got.url, "http://a:11434/v1/embeddings");
    assert_eq!(got.model, "all-minilm");
    assert_eq!(got.api_key.as_deref(), Some("k"));
    assert_eq!(got.source_dim, None, "strict path persists as None");

    // A different workspace is independent.
    assert!(embedder_config::get(&store, &TenantId::new("ws:beta"))
        .await?
        .is_none());

    // Upsert replaces (idempotent set); clears the optional key and switches to
    // the Matryoshka path (a longer source dimension persists round-trip).
    embedder_config::upsert(
        &store,
        &EmbedderConfig {
            tenant_id: "ws:alpha".into(),
            url: "http://a2/v1/embeddings".into(),
            model: "bge-m3".into(),
            api_key: None,
            source_dim: Some(1024),
        },
    )
    .await?;
    let Some(got) = embedder_config::get(&store, &alpha).await? else {
        anyhow::bail!("the config is no longer set");
    };
    assert_eq!(got.url, "http://a2/v1/embeddings");
    assert_eq!(got.api_key, None);
    assert_eq!(got.source_dim, Some(1024), "Matryoshka source dim persists");

    // Delete reverts to the default.
    embedder_config::delete(&store, &alpha).await?;
    assert!(embedder_config::get(&store, &alpha).await?.is_none());
    Ok(())
}
