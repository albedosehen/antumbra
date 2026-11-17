//! How multi-tenant isolation works on an **embedded** SurrealDB engine: the
//! edge/IoT case where there is no separate database server.
//!
//! Two facts, each a regression:
//!   1. Embedded `surrealkv` is **single-writer**: a second connection to the
//!      same on-disk datastore is refused. So a server must multiplex tenants
//!      over ONE connection, not open one connection per identity.
//!   2. Over that one connection, signing in per request isolates tenants:
//!      the engine hides other tenants' rows even on an unfiltered query. So the
//!      security guarantee holds on embedded, no network required.

use anyhow::Result;
use chrono::Utc;

use antumbra_core::{Memory, MemoryNetwork, TenantId, UserId};
use antumbra_store::repo::{memory, principal};
use antumbra_store::{testing, ConnectionConfig, Store};

fn trace(id: &str, tenant: &str, embed: Vec<f32>) -> Memory {
    Memory::new(id, tenant, MemoryNetwork::World, "x", 0.8, Utc::now()).with_embedding(embed)
}

#[tokio::test]
async fn embedded_surrealkv_is_single_writer() -> Result<()> {
    let path = "./target/test-embedded-singlewriter";
    let _ = std::fs::remove_dir_all(path);
    let url = format!("surrealkv://{path}");
    let cfg = || {
        ConnectionConfig::builder()
            .url(&url)
            .namespace("antumbra")
            .database("main")
            .build()
    };

    // The first connection owns the datastore.
    let _s1 = Store::connect(cfg()?, 4).await?;
    // A second connection to the same path is refused; this is why a server
    // holds ONE connection and multiplexes tenants over it (see the next test),
    // rather than opening a connection per identity.
    let second = Store::connect(cfg()?, 4).await;
    assert!(
        second.is_err(),
        "embedded surrealkv must be single-writer; a 2nd connection should be refused"
    );

    drop(_s1);
    let _ = std::fs::remove_dir_all(path);
    Ok(())
}

#[tokio::test]
async fn one_connection_serially_isolates_tenants() -> Result<()> {
    // The model a server uses on embedded: a single connection, signed in per
    // request as the requesting identity. mem:// is one connection here.
    let store = testing::mem_store_of(4).await?;
    let alpha = TenantId::new("ws:alpha");
    let beta = TenantId::new("ws:beta");
    let user_a = UserId::new("user:a");
    let user_b = UserId::new("user:b");

    // Owner provisions both and seeds a row each.
    principal::provision(&store, &alpha, &user_a).await?;
    principal::provision(&store, &beta, &user_b).await?;
    memory::upsert(
        &store,
        &trace("memory:a", "ws:alpha", vec![1.0, 0.0, 0.0, 0.0]),
    )
    .await?;
    memory::upsert(
        &store,
        &trace("memory:b", "ws:beta", vec![0.0, 1.0, 0.0, 0.0]),
    )
    .await?;

    // Request from alpha: bind, serve, see only alpha.
    store.signin(&alpha, &user_a).await?;
    let seen = memory::all_unscoped(&store).await?;
    assert_eq!(seen.len(), 1, "alpha session must see only alpha");
    assert_eq!(seen[0].tenant, alpha);

    // The next request, beta, re-signs the SAME connection in: now only beta.
    store.signin(&beta, &user_b).await?;
    let seen = memory::all_unscoped(&store).await?;
    assert_eq!(seen.len(), 1, "beta session must see only beta");
    assert_eq!(seen[0].tenant, beta);

    // Owner view restored when auth is dropped.
    store.invalidate().await?;
    assert_eq!(memory::all_unscoped(&store).await?.len(), 2);
    Ok(())
}
