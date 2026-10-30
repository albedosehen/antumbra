//! The console's reads gather what each page shows, and each operator action
//! changes what it names and reports an absent target instead of inventing one.

use super::*;
use crate::schema::EMBED_DIM;
use antumbra_core::{
    EdgeType, Expert, FailureBoundary, Generation, GenerationHead, Memory, MemoryEdge, MemoryId,
    MemoryNetwork, MemoryPolicy, RunId, Shadow,
};

use crate::repo::memory_use::{self, Seen};

fn ws() -> TenantId {
    TenantId::new("ws:a")
}

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
        owner: None,
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
    let snapshot = everything(&s, &ws()).await?;
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

    let workspace_only = workspace(&s, &ws()).await?;
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
            &ws(),
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
            &ws(),
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
            &ws(),
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
            &ws(),
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

    assert!(apply(&s, &ws(), &Action::HaltLoop { run: run.clone() }).await?);
    assert!(everything(&s, &ws()).await?.loop_halt_pending);
    assert!(apply(&s, &ws(), &Action::CancelHalt { run: run.clone() }).await?);
    assert!(!everything(&s, &ws()).await?.loop_halt_pending);

    assert!(
        apply(
            &s,
            &ws(),
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
async fn the_workspace_part_carries_use_knobs_and_only_whole_edges() -> Result<()> {
    let s = seeded().await?;
    let old = Utc::now() - chrono::Duration::days(400);
    memory::upsert(
        &s,
        &Memory::new(
            "memory:b",
            "ws:a",
            MemoryNetwork::World,
            "an old fact",
            0.9,
            old,
        ),
    )
    .await?;
    let now = Utc::now();
    for (from, to) in [("memory:a", "memory:b"), ("memory:gone", "memory:a")] {
        edge::relate(
            &s,
            &MemoryEdge::new("ws:a", from, to, EdgeType::References, 1.0, now),
        )
        .await?;
    }
    memory_use::record(&s, &ws(), &[MemoryId::new("memory:a")], Seen::USED, now).await?;

    let snapshot = workspace(&s, &ws()).await?;
    assert_eq!(
        snapshot.edges.len(),
        1,
        "the edge from a missing memory is left out"
    );
    assert_eq!(snapshot.usage.len(), 1);
    assert_eq!(snapshot.policy, MemoryPolicy::default());
    // The fresh, used memory outranks the more confident one left unused for
    // more than two world half-lives.
    let order: Vec<&str> = snapshot.memories.iter().map(|m| m.id.as_str()).collect();
    assert_eq!(order, ["memory:a", "memory:b"]);
    Ok(())
}

#[tokio::test]
async fn an_operator_turns_the_workspaces_knobs() -> Result<()> {
    let s = seeded().await?;
    let turned = MemoryPolicy {
        world_half_life_days: 30.0,
        max_links: 99,
        ..MemoryPolicy::default()
    };
    assert!(apply(&s, &ws(), &Action::SetMemoryPolicy { policy: turned }).await?);
    let policy = workspace(&s, &ws()).await?.policy;
    assert_eq!(policy.world_half_life_days, 30.0);
    assert_eq!(policy.max_links, 10, "clamped");
    let other = workspace(&s, &TenantId::new("ws:b")).await?.policy;
    assert_eq!(other, MemoryPolicy::default());
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
        assert!(!apply(&s, &ws(), &action).await?, "{action:?}");
    }
    assert_eq!(
        loop_control::load(&s, &RunId::new("run:none")).await?,
        LoopCommand::Run,
        "no halt is written for a loop that does not exist"
    );
    Ok(())
}
