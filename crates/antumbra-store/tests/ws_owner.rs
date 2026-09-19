//! Live, over-`ws://` validation of the owner/root session path: `signin_root`'s
//! RootCredentials branch, which the embedded suite cannot reach. On `mem://`
//! anonymous *is* the owner, so `signin_root` takes the `invalidate` fallback;
//! on an authenticated remote `invalidate` would drop to a permission-less
//! anonymous, so returning to the owner view must re-sign-in as root. This drives
//! that branch (and the cross-tenant owner read it enables) against a real
//! SurrealDB v3.
//!
//! `#[ignore]`d, so the normal suite stays network-free; run it with the env set and
//! `cargo test -p antumbra-store --test ws_owner -- --ignored`. Keeps the suite
//! network-free. Reproduce with a fresh SurrealDB v3 container:
//!
//!   docker run --rm -d -p 8002:8000 surrealdb/surrealdb:v3.0.5 \
//!     start --user root --pass root --bind 0.0.0.0:8000 memory
//!   ANTUMBRA_STORE_WS=ws://127.0.0.1:8002/rpc \
//!     cargo test -p antumbra-store --test ws_owner -- --nocapture
//!
//! Optional: `ANTUMBRA_STORE_WS_USER` / `ANTUMBRA_STORE_WS_PASS` (default root/root).

use chrono::Utc;

use antumbra_core::{Memory, MemoryNetwork, TenantId, UserId};
use antumbra_store::repo::{memory, principal};
use antumbra_store::{ConnectionConfig, Store, EMBED_DIM};

fn mem(id: &str, tenant: &TenantId, content: &str) -> Memory {
    Memory::new(
        id,
        tenant.clone(),
        MemoryNetwork::World,
        content,
        0.8,
        Utc::now(),
    )
    .with_embedding(vec![0.1f32; EMBED_DIM])
}

#[tokio::test]
#[ignore = "live ws:// SurrealDB: set ANTUMBRA_STORE_WS=ws://127.0.0.1:8002/rpc and run with --ignored"]
async fn signin_root_restores_the_owner_view_over_ws() {
    let url = std::env::var("ANTUMBRA_STORE_WS").expect(
        "ANTUMBRA_STORE_WS must name a live SurrealDB (ws://127.0.0.1:8002/rpc) to run this ignored test",
    );
    let user = std::env::var("ANTUMBRA_STORE_WS_USER").unwrap_or_else(|_| "root".into());
    let pass = std::env::var("ANTUMBRA_STORE_WS_PASS").unwrap_or_else(|_| "root".into());

    // A root connection: the config carries credentials, so `signin_root` takes
    // the RootCredentials branch (the embedded `invalidate` fallback is what the
    // mem:// tests exercise instead).
    let config = ConnectionConfig::builder()
        .url(&url)
        .namespace("antumbra")
        .database("main")
        .username(&user)
        .password(&pass)
        .build()
        .unwrap();
    let store = Store::connect(config, EMBED_DIM).await.unwrap();

    let alpha = TenantId::new("ws:owner-alpha");
    let beta = TenantId::new("ws:owner-beta");
    principal::provision(&store, &alpha, &UserId::new("user:a"))
        .await
        .unwrap();
    principal::provision(&store, &beta, &UserId::new("user:b"))
        .await
        .unwrap();
    memory::upsert(
        &store,
        &mem("eeeeeeee-0000-0000-0000-00000000000a", &alpha, "alpha"),
    )
    .await
    .unwrap();
    memory::upsert(
        &store,
        &mem("eeeeeeee-0000-0000-0000-00000000000b", &beta, "beta"),
    )
    .await
    .unwrap();

    // Bind a tenant record session: the engine hides the other tenant's rows.
    store.signin(&alpha, &UserId::new("user:a")).await.unwrap();
    let scoped = memory::all_unscoped(&store).await.unwrap();
    assert!(
        !scoped.is_empty() && scoped.iter().all(|m| m.tenant == alpha),
        "a tenant session sees only its own rows over ws://"
    );

    // Return to the owner view via signin_root (the branch under test). Here
    // `invalidate` would drop to a permission-less anonymous; re-signing-in as
    // root restores the cross-tenant read.
    store.signin_root().await.unwrap();
    let owner = memory::all_unscoped(&store).await.unwrap();
    assert!(
        owner.iter().any(|m| m.tenant == alpha) && owner.iter().any(|m| m.tenant == beta),
        "signin_root restores the cross-tenant owner view"
    );

    eprintln!("RESULT: PASS - signin_root owner view over ws://");
}
