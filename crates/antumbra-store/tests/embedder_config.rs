//! Per-workspace embedder config: one row per tenant, upsert/get/delete, with
//! independent workspaces (a tenant with no row falls back to the server default).

use antumbra_core::TenantId;
use antumbra_store::repo::embedder_config::{self, EmbedderConfig};
use antumbra_store::{Store, EMBED_DIM};

#[tokio::test]
async fn embedder_config_roundtrips_per_tenant() {
    let store = Store::connect_memory(EMBED_DIM).await.unwrap();
    let alpha = TenantId::new("ws:alpha");

    // Unset -> None (the caller falls back to the server default).
    assert!(embedder_config::get(&store, &alpha).await.unwrap().is_none());

    embedder_config::upsert(
        &store,
        &EmbedderConfig {
            tenant_id: "ws:alpha".into(),
            url: "http://a:11434/v1/embeddings".into(),
            model: "all-minilm".into(),
            api_key: Some("k".into()),
        },
    )
    .await
    .unwrap();

    let got = embedder_config::get(&store, &alpha)
        .await
        .unwrap()
        .expect("set");
    assert_eq!(got.url, "http://a:11434/v1/embeddings");
    assert_eq!(got.model, "all-minilm");
    assert_eq!(got.api_key.as_deref(), Some("k"));

    // A different workspace is independent.
    assert!(embedder_config::get(&store, &TenantId::new("ws:beta"))
        .await
        .unwrap()
        .is_none());

    // Upsert replaces (idempotent set); clears the optional key.
    embedder_config::upsert(
        &store,
        &EmbedderConfig {
            tenant_id: "ws:alpha".into(),
            url: "http://a2/v1/embeddings".into(),
            model: "bge".into(),
            api_key: None,
        },
    )
    .await
    .unwrap();
    let got = embedder_config::get(&store, &alpha)
        .await
        .unwrap()
        .expect("still set");
    assert_eq!(got.url, "http://a2/v1/embeddings");
    assert_eq!(got.api_key, None);

    // Delete reverts to the default.
    embedder_config::delete(&store, &alpha).await.unwrap();
    assert!(embedder_config::get(&store, &alpha).await.unwrap().is_none());
}
