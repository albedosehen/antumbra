//! The console's reads gather what each page shows, and each operator action
//! changes what it names and reports an absent target instead of inventing one.

use super::*;
use crate::schema::EMBED_DIM;
use antumbra_core::{
    Expert, FailureBoundary, Generation, GenerationHead, Memory, MemoryNetwork, RunId, Shadow,
};

fn expert(id: &str) -> Expert {
    Expert {
        id: ExpertId::new(id),
        name: id.to_string(),
        base_model: "base".into(),
        artifact_uri: "adapters/x.safetensors".into(),
        capability_card: serde_json::json!({}),
        capability_vec: None,
        fitness: 0.5,
        frozen_at: None,
        generation: Generation::ZERO,
        owner: None,
        compartment: None,
        placed_on: None,
        created_at: Utc::now(),
    }
}

fn boundary(id: &str) -> FailureBoundary {
    FailureBoundary {
        id: BoundaryId::new(id),
        behavior: "run `npm install`".into(),
        fail_context: serde_json::json!({ "repo": "uses bun" }),
        near_ok_context: None,
        governing_features: Vec::new(),
        grain: None,
        context_vec: None,
        ok_context_vec: None,
        confidence: 0.8,
        generation: Generation::ZERO,
        created_at: Utc::now(),
    }
}

async fn seeded() -> Result<Store> {
    let s = Store::connect_memory(EMBED_DIM).await?;
    expert::insert(&s, &expert("expert:a")).await?;
    shadow::upsert(
        &s,
        &Shadow::spawn(
            ShadowId::new("shadow:a"),
            Generation::ZERO,
            None,
            Utc::now(),
        ),
    )
    .await?;
    boundary::upsert(&s, &boundary("boundary:a")).await?;
    generation::save_head(&s, &GenerationHead::new(RunId::new("run:a"), Utc::now())).await?;
    memory::upsert(
        &s,
        &Memory::new(
            "memory:a",
            "ws:a",
            MemoryNetwork::World,
            "a fact",
            0.6,
            Utc::now(),
        ),
    )
    .await?;
    Ok(s)
}

#[tokio::test]
async fn everything_reads_both_parts_and_marks_an_operator() -> Result<()> {
    let s = seeded().await?;
    let snapshot = everything(&s).await?;
    assert_eq!(snapshot.experts.len(), 1);
    assert_eq!(snapshot.memories.len(), 1);
    assert!(
        snapshot.memories[0].embedding.is_none(),
        "the console never shows vectors"
    );
    assert_eq!(snapshot.shadows.len(), 1);
    assert_eq!(snapshot.boundaries.len(), 1);
    assert_eq!(snapshot.loop_heads.len(), 1);
    assert!(!snapshot.loop_halt_pending);
    assert!(snapshot.operator);

    let workspace_only = workspace(&s).await?;
    assert!(!workspace_only.operator);
    assert!(workspace_only.shadows.is_empty() && workspace_only.boundaries.is_empty());
    Ok(())
}

#[tokio::test]
async fn each_action_changes_what_it_names() -> Result<()> {
    let s = seeded().await?;
    let shadow_id = ShadowId::new("shadow:a");
    let expert_id = ExpertId::new("expert:a");
    let run = RunId::new("run:a");

    assert!(
        apply(
            &s,
            &Action::GraduateShadow {
                shadow: shadow_id.clone()
            }
        )
        .await?
    );
    let graduated = shadow::get(&s, &shadow_id).await?.expect("kept");
    assert_eq!(graduated.status, ShadowStatus::Graduated);
    assert!(
        apply(
            &s,
            &Action::PruneShadow {
                shadow: shadow_id.clone()
            }
        )
        .await?
    );
    let pruned = shadow::get(&s, &shadow_id).await?.expect("kept");
    assert_eq!(pruned.status, ShadowStatus::Pruned);

    assert!(
        apply(
            &s,
            &Action::FreezeExpert {
                expert: expert_id.clone()
            }
        )
        .await?
    );
    assert!(expert::get(&s, &expert_id)
        .await?
        .expect("kept")
        .frozen_at
        .is_some());
    assert!(
        apply(
            &s,
            &Action::ThawExpert {
                expert: expert_id.clone()
            }
        )
        .await?
    );
    assert!(expert::get(&s, &expert_id)
        .await?
        .expect("kept")
        .frozen_at
        .is_none());

    assert!(apply(&s, &Action::HaltLoop { run: run.clone() }).await?);
    assert!(everything(&s).await?.loop_halt_pending);
    assert!(apply(&s, &Action::CancelHalt { run: run.clone() }).await?);
    assert!(!everything(&s).await?.loop_halt_pending);

    assert!(
        apply(
            &s,
            &Action::DeleteBoundary {
                boundary: BoundaryId::new("boundary:a")
            }
        )
        .await?
    );
    assert!(boundary::list(&s).await?.is_empty());
    Ok(())
}

#[tokio::test]
async fn an_absent_target_is_reported_and_left_alone() -> Result<()> {
    let s = seeded().await?;
    for action in [
        Action::PruneShadow {
            shadow: ShadowId::new("shadow:none"),
        },
        Action::FreezeExpert {
            expert: ExpertId::new("expert:none"),
        },
        Action::DeleteBoundary {
            boundary: BoundaryId::new("boundary:none"),
        },
        Action::HaltLoop {
            run: RunId::new("run:none"),
        },
    ] {
        assert!(!apply(&s, &action).await?, "{action:?}");
    }
    assert_eq!(
        loop_control::load(&s, &RunId::new("run:none")).await?,
        LoopCommand::Run,
        "no halt is written for a loop that does not exist"
    );
    Ok(())
}
