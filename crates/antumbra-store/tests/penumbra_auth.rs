//! Engine-enforced tenant isolation (Stage 2): with a per-tenant record-access
//! session bound (`$auth.tenant`), the SurrealDB engine itself refuses another
//! tenant's rows, even on an unfiltered `SELECT` with no app-side WHERE. This
//! is the structural guarantee the app-side filter only approximates.

use chrono::Utc;

use antumbra_core::router::{LearnedRouter, RouterExpert};
use antumbra_core::{Expert, ExpertId, Generation, Memory, MemoryNetwork, TenantId, UserId};
use antumbra_store::repo::{expert, memory, principal, router};
use antumbra_store::Store;

fn trace(id: &str, tenant: &str, content: &str, embed: Vec<f32>) -> Memory {
    Memory::new(id, tenant, MemoryNetwork::World, content, 0.8, Utc::now()).with_embedding(embed)
}

#[tokio::test]
async fn shared_population_is_readable_under_tenant_auth() {
    // The umbra (experts + learned router) is shared: a tenant-authenticated
    // session can READ it (to route), while the per-tenant memory stays private.
    let store = Store::connect_memory(4).await.unwrap();
    let alpha = TenantId::new("ws:alpha");
    principal::provision(&store, &alpha, &UserId::new("user:a"))
        .await
        .unwrap();

    let now = Utc::now();
    expert::insert(
        &store,
        &Expert {
            id: ExpertId::new("expert:adder"),
            name: "adder".into(),
            base_model: "base".into(),
            artifact_uri: "mem://a".into(),
            capability_card: serde_json::Value::Null,
            capability_vec: Some(vec![1.0, 0.0, 0.0, 0.0]),
            fitness: 1.0,
            frozen_at: Some(now),
            generation: Generation::ZERO,
            owner: None,
            compartment: None,
            placed_on: None,
            created_at: now,
        },
    )
    .await
    .unwrap();
    // A PRIVATE expert owned by a different user must not be visible to user:a.
    expert::insert(
        &store,
        &Expert {
            id: ExpertId::new("expert:private"),
            name: "private".into(),
            base_model: "base".into(),
            artifact_uri: "mem://p".into(),
            capability_card: serde_json::Value::Null,
            capability_vec: Some(vec![0.0, 1.0, 0.0, 0.0]),
            fitness: 1.0,
            frozen_at: Some(now),
            generation: Generation::ZERO,
            owner: Some(UserId::new("user:other")),
            compartment: Some(antumbra_core::CompartmentId::new("comp:other")),
            placed_on: None,
            created_at: now,
        },
    )
    .await
    .unwrap();
    router::save(
        &store,
        &LearnedRouter {
            weights: vec![1.0; 4],
            experts: vec![RouterExpert {
                id: ExpertId::new("expert:adder"),
                centroid: vec![1.0, 0.0, 0.0, 0.0],
            }],
            temperature: 0.1,
            floor: -1.0,
        },
    )
    .await
    .unwrap();

    // As user:a: the shared expert + router read through, but another user's
    // private expert is hidden by the engine (owner-scoped select).
    store.signin(&alpha, &UserId::new("user:a")).await.unwrap();
    let seen = expert::list(&store).await.unwrap();
    assert_eq!(
        seen.len(),
        1,
        "private expert of another user must be hidden"
    );
    assert_eq!(seen[0].id, ExpertId::new("expert:adder"));
    assert!(router::load(&store).await.unwrap().is_some());
}

#[tokio::test]
async fn engine_enforces_tenant_isolation_under_record_auth() {
    let store = Store::connect_memory(4).await.unwrap();
    let alpha = TenantId::new("ws:alpha");
    let beta = TenantId::new("ws:beta");

    // Owner/root provisions principals and seeds both tenants' memories.
    principal::provision(&store, &alpha, &UserId::new("user:a"))
        .await
        .unwrap();
    principal::provision(&store, &beta, &UserId::new("user:b"))
        .await
        .unwrap();
    memory::upsert(
        &store,
        &trace("memory:a", "ws:alpha", "alpha", vec![1.0, 0.0, 0.0, 0.0]),
    )
    .await
    .unwrap();
    memory::upsert(
        &store,
        &trace("memory:b", "ws:beta", "beta", vec![0.0, 1.0, 0.0, 0.0]),
    )
    .await
    .unwrap();

    // Owner view: the unfiltered query spans both tenants (cross-tenant read).
    assert_eq!(memory::all_unscoped(&store).await.unwrap().len(), 2);

    // Bind a tenant session: $auth.tenant = ws:alpha, engine PERMISSIONS active.
    store.signin(&alpha, &UserId::new("user:a")).await.unwrap();

    // The SAME unfiltered query now returns ONLY alpha's row; the engine hides
    // beta's, with no app-side WHERE involved. This is the structural guarantee.
    let seen = memory::all_unscoped(&store).await.unwrap();
    assert_eq!(
        seen.len(),
        1,
        "engine must hide other tenants under record auth"
    );
    assert_eq!(seen[0].tenant, alpha);

    // Even an explicit WHERE-filtered read for beta yields nothing this session.
    assert!(memory::list(&store, &beta).await.unwrap().is_empty());

    // Drop auth → owner view restored, both tenants visible again.
    store.invalidate().await.unwrap();
    assert_eq!(memory::all_unscoped(&store).await.unwrap().len(), 2);
}
