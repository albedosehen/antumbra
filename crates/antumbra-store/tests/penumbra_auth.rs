//! Engine-enforced tenant isolation (Stage 2): with a per-tenant record-access
//! session bound (`$auth.tenant`), the SurrealDB engine itself refuses another
//! tenant's rows — even on an unfiltered `SELECT` with no app-side WHERE. This
//! is the structural guarantee the app-side filter only approximates.

use chrono::Utc;

use antumbra_core::{Memory, MemoryNetwork, TenantId};
use antumbra_store::repo::{memory, principal};
use antumbra_store::Store;

fn trace(id: &str, tenant: &str, content: &str, embed: Vec<f32>) -> Memory {
    Memory::new(id, tenant, MemoryNetwork::World, content, 0.8, Utc::now()).with_embedding(embed)
}

#[tokio::test]
async fn engine_enforces_tenant_isolation_under_record_auth() {
    let store = Store::connect_memory(4).await.unwrap();
    let alpha = TenantId::new("ws:alpha");
    let beta = TenantId::new("ws:beta");

    // Owner/root provisions principals and seeds both tenants' memories.
    principal::provision(&store, &alpha).await.unwrap();
    principal::provision(&store, &beta).await.unwrap();
    memory::upsert(&store, &trace("memory:a", "ws:alpha", "alpha", vec![1.0, 0.0, 0.0, 0.0]))
        .await
        .unwrap();
    memory::upsert(&store, &trace("memory:b", "ws:beta", "beta", vec![0.0, 1.0, 0.0, 0.0]))
        .await
        .unwrap();

    // Owner view: the unfiltered query spans both tenants (cross-tenant read).
    assert_eq!(memory::all_unscoped(&store).await.unwrap().len(), 2);

    // Bind a tenant session: $auth.tenant = ws:alpha, engine PERMISSIONS active.
    store.signin_tenant(&alpha).await.unwrap();

    // The SAME unfiltered query now returns ONLY alpha's row — the engine hides
    // beta's, with no app-side WHERE involved. This is the structural guarantee.
    let seen = memory::all_unscoped(&store).await.unwrap();
    assert_eq!(seen.len(), 1, "engine must hide other tenants under record auth");
    assert_eq!(seen[0].tenant, alpha);

    // Even an explicit WHERE-filtered read for beta yields nothing this session.
    assert!(memory::list(&store, &beta).await.unwrap().is_empty());

    // Drop auth → owner view restored, both tenants visible again.
    store.invalidate().await.unwrap();
    assert_eq!(memory::all_unscoped(&store).await.unwrap().len(), 2);
}
