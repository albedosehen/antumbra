//! The console's demo (`antumbra tui --demo`): a seeded population held in
//! memory, so the console can be explored before anything is set up, and so a
//! headless snapshot has something to draw. Nothing in it persists.

use anyhow::Result;

use antumbra_core::ports::Embedder;
use antumbra_core::testing::FixedEmbedder;
use antumbra_store::{Store, EMBED_DIM};

use crate::DEMO_TENANT;

/// A store in memory holding a small population: experts, shadows, boundaries,
/// memories and their links, a loop run, its evaluations and a trained router.
pub(crate) async fn seed() -> Result<Store> {
    use antumbra_core::generational::{GenerationHead, LoopState};
    use antumbra_core::router::{LearnedRouter, RouterExpert};
    use antumbra_core::{
        BoundaryId, EdgeType, EvalStatus, EvaluationRun, Expert, ExpertId, FailureBoundary,
        Generation, Grain, Memory, MemoryEdge, MemoryNetwork, RunId, Shadow, ShadowId,
        ShadowStatus, SubjectKind,
    };
    use antumbra_store::repo::{
        boundary, edge, evaluation, expert, generation, memory, router, shadow,
    };
    use chrono::Utc;

    let store = Store::connect_memory(EMBED_DIM).await?;
    let demo = [
        ("arith-specialist", 0.92f32, true),
        ("string-specialist", 0.81, true),
        ("datetime-specialist", 0.74, true),
        ("json-shaper", 0.55, false),
        ("regex-smith", 0.63, true),
    ];
    let now = Utc::now();
    // Embed each expert's name for its capability vector and router centroid, so
    // the route-ask (which embeds the query the same way) routes coherently and
    // the drill-down's route preview shows a real distribution.
    let embedder = FixedEmbedder::new(EMBED_DIM);
    let mut probe = std::collections::HashMap::new();
    for (name, _, _) in &demo {
        probe.insert(*name, embedder.embed(name).await?);
    }
    for (name, fitness, frozen) in demo.iter().copied() {
        expert::insert(
            &store,
            &Expert {
                id: ExpertId::new(format!("expert:{name}")),
                name: name.into(),
                base_model: "Qwen/Qwen2.5-Coder-1.5B".into(),
                artifact_uri: format!("mem://{name}"),
                capability_card: serde_json::json!({
                    "description": format!("demo specialist for {name}"),
                    "exemplars": ["ex-1", "ex-2", "ex-3"],
                }),
                capability_vec: Some(probe[name].clone()),
                fitness,
                frozen_at: frozen.then_some(now),
                generation: Generation::ZERO,
                owner: None,
                compartment: None,
                // A `mem://` artifact is on nobody's disk: placed nowhere,
                // servable anywhere, which is what a demo wants.
                placed_on: None,
                created_at: now,
            },
        )
        .await?;
    }
    router::save(
        &store,
        &LearnedRouter {
            weights: vec![1.0; EMBED_DIM],
            experts: demo
                .iter()
                .map(|(n, _, _)| RouterExpert {
                    id: ExpertId::new(format!("expert:{n}")),
                    centroid: probe[*n].clone(),
                })
                .collect(),
            temperature: 0.2,
            floor: 0.1,
        },
    )
    .await?;
    // A couple of boundaries (the antumbra) so the inspector has content: one
    // actionable (a recovered C/C' contrast that gates routing) and one open
    // (recorded from a collapse, not yet scoped).
    let scopes = [
        (
            "run `npm install`",
            "runtime",
            serde_json::json!({ "runtime": "deno" }),
            Some(serde_json::json!({ "runtime": "node" })),
            true,
        ),
        (
            "approach of shadow:g3",
            "",
            serde_json::json!({ "generation": 3 }),
            None,
            false,
        ),
    ];
    for (i, (behavior, feature, fail, near_ok, actionable)) in scopes.into_iter().enumerate() {
        boundary::upsert(
            &store,
            &FailureBoundary {
                id: BoundaryId::new(format!("boundary:demo-{i}")),
                behavior: behavior.into(),
                fail_context: fail,
                near_ok_context: near_ok,
                governing_features: if feature.is_empty() {
                    Vec::new()
                } else {
                    vec![feature.into()]
                },
                grain: actionable.then_some(Grain::Project),
                context_vec: None,
                ok_context_vec: None,
                confidence: if actionable { 0.8 } else { 0.3 },
                generation: Generation::ZERO,
                owner: None,
                created_at: now,
            },
        )
        .await?;
    }
    // A few shadows (the penumbra) across the lifecycle, so the training view has
    // content: one still exploring, one graduated (rising reward), one collapsed.
    let shadows = [
        ("shadow:g4", ShadowStatus::Exploring, vec![0.10, 0.32, 0.55]),
        (
            "shadow:g3",
            ShadowStatus::Graduated,
            vec![0.20, 0.60, 0.85, 0.93],
        ),
        ("shadow:g2", ShadowStatus::Pruned, vec![0.05, 0.04, 0.06]),
    ];
    for (i, (id, status, reward_curve)) in shadows.into_iter().enumerate() {
        shadow::upsert(
            &store,
            &Shadow {
                id: ShadowId::new(id),
                parent_expert: None,
                adapter_uri: Some(format!("mem://{id}")),
                status,
                generation: Generation(4 - i as u32),
                reward_curve,
                created_at: now,
            },
        )
        .await?;
    }
    // And a standing expert's training, as the keeper records one on a node
    // that trains: its shadow, from the base model's held-out pass rate to the
    // trained expert's, and an open boundary, private to its user, for the
    // behavior it missed, with the task it was not followed on.
    let user = antumbra_core::UserId::new("user:demo");
    let standing = ExpertId::new(format!(
        "expert:{DEMO_TENANT}:{}:behavior:code",
        user.as_str()
    ));
    shadow::upsert(
        &store,
        &Shadow {
            id: antumbra_core::behavior::shadow_id(
                &antumbra_core::TenantId::new(DEMO_TENANT),
                &user,
                "code",
                "5c0ffee1d2e3f4a5",
            ),
            parent_expert: None,
            adapter_uri: Some("mem://standing-code".into()),
            status: ShadowStatus::Graduated,
            generation: Generation::ZERO,
            reward_curve: vec![0.42, 0.91],
            created_at: now,
        },
    )
    .await?;
    boundary::upsert(
        &store,
        &FailureBoundary {
            id: BoundaryId::new(format!("boundary:{}:memory:tests-first", standing.as_str())),
            behavior: "Run the test suite before calling a change done.".into(),
            fail_context: serde_json::json!({
                "scope": "code",
                "task": "Rename the config loader's environment variable and tell me when it is finished.",
                "answer": "Done: ANTUMBRA_CFG is now ANTUMBRA_CONFIG everywhere.",
            }),
            near_ok_context: None,
            governing_features: Vec::new(),
            grain: None,
            context_vec: None,
            ok_context_vec: None,
            confidence: 0.33,
            generation: Generation::ZERO,
            owner: Some(user),
            created_at: now,
        },
    )
    .await?;
    // Penumbra memory traces (the Memory page) across the three networks, with one
    // consolidated into an expert, plus edges including a contradiction and a
    // supersession; the native consolidation→retire signal.
    let tenant = DEMO_TENANT;
    let mems = [
        (
            "memory:deno",
            MemoryNetwork::World,
            "prefer deno over node for new scripts",
            0.86,
            4u32,
            false,
            false,
        ),
        (
            "memory:surql",
            MemoryNetwork::World,
            "use surql-rs builders, never raw SurrealQL",
            0.93,
            7,
            false,
            true,
        ),
        (
            "memory:node-incident",
            MemoryNetwork::Bank,
            "npm install failed under the deno runtime",
            0.70,
            2,
            true,
            false,
        ),
        (
            "memory:retry",
            MemoryNetwork::Bank,
            "retrying the flaky step fixed the test",
            0.58,
            1,
            true,
            false,
        ),
        (
            "memory:terse",
            MemoryNetwork::Opinion,
            "keep operator docs terse, no emojis",
            0.80,
            3,
            false,
            false,
        ),
        (
            "memory:canvas",
            MemoryNetwork::Opinion,
            "canvas-first rendering beats raster over SSH",
            0.75,
            2,
            false,
            false,
        ),
    ];
    for (id, network, content, confidence, reinforcement, volatile, consolidated) in mems {
        let mut m = Memory::new(id, tenant, network, content, confidence, now)
            .by("user:operator", "windows")
            .volatile(volatile)
            .with_evidence(vec!["session-log".into()]);
        m.reinforcement = reinforcement;
        if consolidated {
            m.mark_consolidated(ExpertId::new("expert:json-shaper"), now);
        }
        memory::upsert(&store, &m).await?;
    }
    let edges = [
        (
            "memory:node-incident",
            "memory:deno",
            EdgeType::Contradicts,
            0.8,
        ),
        ("memory:deno", "memory:node-incident", EdgeType::Caused, 0.6),
        (
            "memory:retry",
            "memory:node-incident",
            EdgeType::Follows,
            0.5,
        ),
        ("memory:surql", "memory:deno", EdgeType::References, 0.4),
        ("memory:terse", "memory:canvas", EdgeType::Supersedes, 0.5),
    ];
    for (from, to, edge_type, weight) in edges {
        edge::relate(
            &store,
            &MemoryEdge::new(tenant, from, to, edge_type, weight, now),
        )
        .await?;
    }
    // The generational loop head (mid-cycle) for the Loop page.
    let mut head = GenerationHead::new(RunId::new("run:demo"), now);
    head.generation = Generation(4);
    head.state = LoopState::Score;
    generation::save_head(&store, &head).await?;
    // A few evaluation runs across subjects/statuses for the Evals page (one
    // failure = the regression tripwire firing).
    let runs = [
        (
            "run:e1",
            SubjectKind::Expert,
            "expert:arith-specialist",
            EvalStatus::Success,
            Some("a1b2c3d4"),
        ),
        (
            "run:e2",
            SubjectKind::Expert,
            "expert:json-shaper",
            EvalStatus::Success,
            Some("9f8e7d6c"),
        ),
        (
            "run:e3",
            SubjectKind::Shadow,
            "shadow:g3",
            EvalStatus::Failure,
            None,
        ),
        (
            "run:e4",
            SubjectKind::Router,
            "router",
            EvalStatus::Running,
            None,
        ),
    ];
    for (id, kind, subject, status, fp) in runs {
        evaluation::insert(
            &store,
            &EvaluationRun {
                run_id: RunId::new(id),
                subject_kind: kind,
                subject_id: subject.into(),
                corpus_task_id: "task:arith".into(),
                status,
                metrics: None,
                regression_fingerprint: fp.map(str::to_string),
                created_at: now,
            },
        )
        .await?;
    }
    Ok(store)
}
