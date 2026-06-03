//! End-to-end with fakes: the loop grows a population, the gate routes a task
//! to a graduated expert, and the serve seam answers for it -- the whole v0
//! spine (train -> graduate -> route -> serve) wired together, no GPU.

use antumbra_core::ports::{ActRequest, Embedder, Serve};
use antumbra_core::testing::{EchoServe, FixedEmbedder, ScriptedTrainer};
use antumbra_core::RunId;
use antumbra_gate::{route, GateConfig};
use antumbra_loop::{GenerationLoop, LoopConfig};
use antumbra_store::repo::{boundary, expert};
use antumbra_store::Store;

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
        })
        .await
        .unwrap();
    assert_eq!(out.final_output, "reverse 'abc'");
}
