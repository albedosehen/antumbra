//! End-to-end with fakes: the loop grows a population, the gate routes a task
//! to a graduated expert, and the serve seam answers for it -- the whole v0
//! spine (train -> graduate -> route -> serve) wired together, no GPU.

use antumbra_core::ports::{ActRequest, Embedder, Serve};
use antumbra_core::testing::{EchoServe, FixedEmbedder, ScriptedTrainer};
use antumbra_core::{
    ClusterConfig, Compartment, CompartmentId, Memory, MemoryNetwork, RunId, TenantId, UserId,
};
use antumbra_gate::{route, GateConfig};
use antumbra_loop::{GenerationLoop, LoopConfig};
use antumbra_store::repo::{boundary, compartment, expert, memory, principal};
use antumbra_store::Store;
use chrono::Utc;

#[tokio::test]
async fn train_graduate_route_then_serve() {
    let store = Store::connect_memory(8).await.expect("connect");
    let exemplars = vec!["reverse a string".to_string(), "format text".to_string()];
    let trainer = ScriptedTrainer::graduating_with_exemplars(exemplars.clone());
    let embedder = FixedEmbedder::new(8);

    let run = RunId::new("run:e2e");
    GenerationLoop::new(&store, &trainer, &embedder, LoopConfig::default())
        .run_until(&run, 1)
        .await
        .expect("run");

    let experts = expert::list(&store).await.unwrap();
    assert_eq!(experts.len(), 1, "the loop graduated one expert");

    // Route a task near the expert's learned capability. Threshold off so the
    // pipeline (not the escalation heuristic, covered elsewhere) is what's under test.
    let task_vec = embedder.embed("reverse a string").await.unwrap();
    let boundaries = boundary::list(&store).await.unwrap();
    let cfg = GateConfig {
        coverage_threshold: -1.0,
        ..GateConfig::default()
    };
    let decision = route(&task_vec, &experts, &boundaries, 1, &cfg);
    assert!(!decision.escalate);
    let chosen = decision.chosen[0].clone();
    assert_eq!(chosen, experts[0].id);

    // Serve the chosen expert via the fake seam.
    let out = EchoServe
        .act(ActRequest {
            task_id: "t".into(),
            prompt: "reverse 'abc'".into(),
            adapters: vec![chosen],
            weights: Vec::new(),
        })
        .await
        .unwrap();
    assert_eq!(out.final_output, "reverse 'abc'");
}

/// The `propose-compartments` op's distinctive logic: the pool filter (inbox +
/// authored-uncompartmented, excluding filed compartments and other users) and
/// the apply round-trip (create proposed compartment, move members in). Mirrors
/// `ops::propose_compartments` against a real store (the op is private to the
/// bin). Clustering itself is unit-tested in `antumbra-core::penumbra`.
#[tokio::test]
async fn propose_compartments_pool_filter_and_apply() {
    let store = Store::connect_memory(8).await.expect("connect");
    let embedder = FixedEmbedder::new(8);
    let tenant = TenantId::new("ws:e2e");
    let user_a = UserId::new("user:a");
    let user_b = UserId::new("user:b");
    let inbox = CompartmentId::new("comp:ws:e2e:user:a:default");
    let now = Utc::now();
    principal::provision(&store, &tenant, &user_a)
        .await
        .unwrap();

    // Three inbox memories (user A) that should cluster + move.
    for (id, content) in [
        ("memory:1", "deno run typescript module"),
        ("memory:2", "deno test typescript suite"),
        ("memory:3", "deno bundle typescript output"),
    ] {
        let v = embedder.embed(content).await.unwrap();
        let m = Memory::new(id, tenant.clone(), MemoryNetwork::World, content, 0.8, now)
            .with_embedding(v)
            .by(user_a.clone(), "host")
            .in_compartment(inbox.clone());
        memory::upsert(&store, &m).await.unwrap();
    }
    // One already in a NAMED compartment (filed) -> excluded.
    {
        let v = embedder.embed("deno fmt typescript files").await.unwrap();
        let m = Memory::new(
            "memory:filed",
            tenant.clone(),
            MemoryNetwork::World,
            "filed",
            0.8,
            now,
        )
        .with_embedding(v)
        .by(user_a.clone(), "host")
        .in_compartment(CompartmentId::new("comp:named"));
        memory::upsert(&store, &m).await.unwrap();
    }
    // One uncompartmented but authored by ANOTHER user -> excluded.
    {
        let v = embedder.embed("deno lint typescript code").await.unwrap();
        let m = Memory::new(
            "memory:other",
            tenant.clone(),
            MemoryNetwork::World,
            "other",
            0.8,
            now,
        )
        .with_embedding(v)
        .by(user_b.clone(), "host");
        memory::upsert(&store, &m).await.unwrap();
    }

    // The op's pool filter.
    let pool: Vec<Memory> = memory::list(&store, &tenant)
        .await
        .unwrap()
        .into_iter()
        .filter(|m| {
            m.compartment.as_ref() == Some(&inbox)
                || (m.compartment.is_none() && m.author.as_ref() == Some(&user_a))
        })
        .collect();
    assert_eq!(pool.len(), 3, "only user A's inbox memories are candidates");

    // Threshold 0.0 groups the whole inbox into one cluster regardless of the
    // fake embedder's geometry, so the wiring is deterministic.
    let cfg = ClusterConfig {
        similarity_threshold: 0.0,
        min_size: 3,
        ..ClusterConfig::default()
    };
    let proposals = antumbra_core::propose_compartments(&pool, &cfg);
    assert_eq!(proposals.len(), 1);
    let p = &proposals[0];

    // Apply: create the proposed compartment and move its members in.
    let new_id = CompartmentId::new("comp:ws:e2e:user:a:proposed:region");
    let c = Compartment::new(
        new_id.clone(),
        tenant.clone(),
        user_a.clone(),
        &p.label,
        now,
    )
    .proposed();
    compartment::create(&store, &c).await.unwrap();
    for mid in &p.members {
        let mut m = memory::get(&store, &tenant, mid).await.unwrap().unwrap();
        m.compartment = Some(new_id.clone());
        memory::upsert(&store, &m).await.unwrap();
    }

    let moved = memory::list_by_compartment(&store, &tenant, &new_id)
        .await
        .unwrap();
    assert_eq!(moved.len(), 3, "the inbox region moved into the proposal");
    // The filed and other-user memories are untouched.
    let still_filed =
        memory::list_by_compartment(&store, &tenant, &CompartmentId::new("comp:named"))
            .await
            .unwrap();
    assert_eq!(still_filed.len(), 1);
}
