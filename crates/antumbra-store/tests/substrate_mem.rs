//! End-to-end substrate test against an embedded in-memory SurrealDB:
//! schema apply, expert insert/get/list, capability KNN recall, and the
//! durable generation-head checkpoint of the generational loop.

use anyhow::{Context, Result};

use antumbra_core::{Expert, ExpertId, Generation, GenerationHead, LoopState, RunId};
use antumbra_store::repo::{expert, generation};
use antumbra_store::testing;

fn expert_fixture(key: &str, name: &str, vec: Vec<f32>) -> Expert {
    Expert {
        id: ExpertId::new(key),
        name: name.to_string(),
        base_model: "code-base".into(),
        artifact_uri: format!("memory://adapter/{name}"),
        capability_card: serde_json::json!({"does": name}),
        capability_vec: Some(vec),
        fitness: 0.5,
        frozen_at: None,
        generation: Generation::ZERO,
        owner: None,
        compartment: None,
        placed_on: None,
        created_at: chrono::Utc::now(),
    }
}

#[tokio::test]
async fn substrate_roundtrip() -> Result<()> {
    let store = testing::mem_store_of(8).await.context("connect mem")?;

    let e1 = expert_fixture(
        "expert:deno",
        "deno-conv",
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    );
    let e2 = expert_fixture(
        "expert:brand",
        "brand-voice",
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    );
    expert::insert(&store, &e1).await.context("insert e1")?;
    expert::insert(&store, &e2).await.context("insert e2")?;

    // round-trip a single record
    let Some(got) = expert::get(&store, &e1.id).await.context("get")? else {
        anyhow::bail!("e1 is not present");
    };
    assert_eq!(got.name, "deno-conv");
    assert_eq!(got.base_model, "code-base");
    assert_eq!(got.capability_vec.as_ref().map(Vec::len), Some(8));
    assert!(!got.is_frozen());

    // list the whole population
    let all = expert::list(&store).await.context("list")?;
    assert_eq!(all.len(), 2);

    // routing-as-retrieval: nearest expert to a query close to e1
    let near = expert::knn_by_capability(&store, &[0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1)
        .await
        .context("knn")?;
    assert_eq!(near.len(), 1);
    assert_eq!(near[0].name, "deno-conv");

    // durable loop checkpoint survives a re-read
    let run = RunId::new("run:loop");
    let head = GenerationHead::new(run.clone(), chrono::Utc::now());
    generation::save_head(&store, &head)
        .await
        .context("save head")?;
    let Some(loaded) = generation::load_head(&store, &run).await.context("load")? else {
        anyhow::bail!("no head after the first save");
    };
    assert_eq!(loaded.state, LoopState::Grow);
    assert_eq!(loaded.generation, Generation::ZERO);

    // advance and re-checkpoint
    let mut head = loaded;
    head.advance_to(LoopState::Explore, chrono::Utc::now())?;
    generation::save_head(&store, &head)
        .await
        .context("save head 2")?;
    let Some(loaded) = generation::load_head(&store, &run)
        .await
        .context("load 2")?
    else {
        anyhow::bail!("no head after the second save");
    };
    assert_eq!(loaded.state, LoopState::Explore);
    Ok(())
}
