use super::*;

#[test]
fn every_page_has_a_number_key_and_no_key_reaches_past_the_last() {
    let keys: Vec<Option<usize>> = "0123456789x".chars().map(Page::index_for_key).collect();
    let reached: Vec<usize> = keys.iter().flatten().copied().collect();
    assert_eq!(reached, (0..Page::ALL.len()).collect::<Vec<_>>());
    assert_eq!(Page::index_for_key('0'), None);
    assert_eq!(Page::index_for_key('x'), None);
    // The tab the key selects is the tab the bar numbers it as.
    for (index, page) in Page::ALL.iter().enumerate() {
        assert_eq!(page.index(), index);
    }
    assert_eq!(
        Page::index_for_key('5').and_then(|i| Page::ALL.get(i)),
        Some(&Page::Sovereign)
    );
}

/// The whole path the page depends on: counters written to a store, read by
/// the snapshot the console loads, and driven by the keys every page shares.
#[tokio::test]
async fn the_sovereign_page_reads_counters_from_the_store_and_selects_among_them(
) -> anyhow::Result<()> {
    let store = Store::connect_memory(EMBED_DIM).await?;
    let mut app = App::load(&store).await?;
    app.set_page(Page::Sovereign);
    // Nothing to select: the keys must not move a selection that is not there.
    app.select_next();
    assert_eq!((app.sovereign.skills.len(), app.selected_skill), (0, 0));

    for skill in ["deploy", "review"] {
        let counter = Memory::new(
            format!("memory:{skill}"),
            "ws:t",
            MemoryNetwork::World,
            format!("[skill-use:{skill}] {skill}"),
            0.9,
            Utc::now(),
        );
        memory::upsert(&store, &counter).await?;
    }
    app.apply_snapshot(Snapshot::load(&store).await?);
    assert_eq!(app.sovereign.skills.len(), 2);
    app.select_next();
    assert_eq!(app.selected_skill, 1);
    app.select_next();
    assert_eq!(
        app.selected_skill, 0,
        "the selection wraps, as on every page"
    );

    // A reload that leaves fewer rows must not leave the selection past the end.
    app.select_next();
    memory::delete(
        &store,
        &TenantId::new("ws:t"),
        &MemoryId::new("memory:review"),
    )
    .await?;
    app.apply_snapshot(Snapshot::load(&store).await?);
    assert!(app.selected_skill < app.sovereign.skills.len().max(1));
    Ok(())
}
use antumbra_core::generational::LoopState;
use antumbra_core::{
    EdgeType, EvalStatus, Generation, MemoryId, MemoryNetwork, RunId, SubjectKind, TenantId,
};
use antumbra_store::EMBED_DIM;
use chrono::Utc;

/// An in-memory store with one exploring shadow and one boundary, loaded into
/// a fresh `App` (so reads go through the real `App::load`/`reload` path).
async fn seeded() -> (Store, App) {
    let store = Store::connect_memory(EMBED_DIM).await.unwrap();
    shadow::upsert(
        &store,
        &Shadow {
            id: ShadowId::new("shadow:s1"),
            parent_expert: None,
            adapter_uri: None,
            status: ShadowStatus::Exploring,
            generation: Generation(1),
            reward_curve: vec![],
            created_at: Utc::now(),
        },
    )
    .await
    .unwrap();
    boundary::upsert(
        &store,
        &FailureBoundary {
            id: BoundaryId::new("boundary:b1"),
            behavior: "do the thing".into(),
            fail_context: serde_json::json!({}),
            near_ok_context: None,
            governing_features: Vec::new(),
            grain: None,
            context_vec: None,
            ok_context_vec: None,
            confidence: 0.5,
            generation: Generation::ZERO,
            created_at: Utc::now(),
        },
    )
    .await
    .unwrap();
    let app = App::load(&store).await.unwrap();
    (store, app)
}

#[tokio::test]
async fn prune_writes_to_the_store_and_raises_an_event() {
    let (store, mut app) = seeded().await;
    app.focus = Focus::Shadows;
    app.request_action();
    assert!(matches!(app.pending, Some(Pending::PruneShadow(_))));
    app.apply_pending(&store).await.unwrap();

    let s = app
        .shadows
        .iter()
        .find(|s| s.id.as_str() == "shadow:s1")
        .unwrap();
    assert_eq!(s.status.as_str(), "pruned", "the store reflects the prune");
    assert!(
        app.events.iter().any(|e| e.text.contains("pruned")),
        "the prune surfaced as an event"
    );
    assert_eq!(app.mode, Mode::Normal);
    assert!(app.pending.is_none());
}

#[tokio::test]
async fn delete_removes_the_boundary_and_raises_an_event() {
    let (store, mut app) = seeded().await;
    app.focus = Focus::Boundaries;
    app.request_action();
    app.apply_pending(&store).await.unwrap();

    assert!(app.boundaries.is_empty(), "the boundary is gone");
    assert!(
        app.events
            .iter()
            .any(|e| e.text.contains("boundary removed")),
        "the deletion surfaced as an event"
    );
}

#[tokio::test]
async fn graduate_writes_to_the_store_and_raises_an_event() {
    let (store, mut app) = seeded().await;
    app.focus = Focus::Shadows;
    app.request_graduate();
    assert!(matches!(app.pending, Some(Pending::GraduateShadow(_))));
    app.apply_pending(&store).await.unwrap();

    let s = app
        .shadows
        .iter()
        .find(|s| s.id.as_str() == "shadow:s1")
        .unwrap();
    assert_eq!(
        s.status.as_str(),
        "graduated",
        "the store reflects the promotion"
    );
    assert!(
        app.events.iter().any(|e| e.text.contains("graduated")),
        "the graduation surfaced as an event"
    );
    assert_eq!(app.mode, Mode::Normal);
}

#[tokio::test]
async fn request_graduate_skips_an_already_graduated_shadow() {
    let (_store, mut app) = seeded().await;
    app.shadows[0].status = ShadowStatus::Graduated;
    app.focus = Focus::Shadows;
    app.request_graduate();
    assert!(app.pending.is_none());
}

#[tokio::test]
async fn freeze_toggles_the_expert_and_raises_events() {
    let (store, mut app) = seeded().await;
    let mut v = vec![0.0f32; EMBED_DIM];
    v[0] = 1.0;
    expert::insert(
        &store,
        &Expert {
            id: ExpertId::new("expert:e1"),
            name: "scout".into(),
            base_model: "base".into(),
            artifact_uri: "a.safetensors".into(),
            capability_card: serde_json::json!({}),
            capability_vec: Some(v),
            fitness: 1.0,
            frozen_at: None,
            generation: Generation::ZERO,
            owner: None,
            compartment: None,
            placed_on: None,
            created_at: Utc::now(),
        },
    )
    .await
    .unwrap();
    app.reload(&store).await.unwrap();
    app.focus = Focus::Experts;
    app.selected = app
        .experts
        .iter()
        .position(|e| e.id.as_str() == "expert:e1")
        .unwrap();

    // First toggle freezes.
    app.request_freeze();
    assert!(matches!(app.pending, Some(Pending::FreezeExpert(_, true))));
    app.apply_pending(&store).await.unwrap();
    let frozen = app
        .experts
        .iter()
        .find(|e| e.id.as_str() == "expert:e1")
        .unwrap();
    assert!(frozen.is_frozen(), "the store reflects the freeze");
    assert!(app.events.iter().any(|ev| ev.text.contains("frozen")));

    // Second toggle thaws (the command reads the current state).
    app.request_freeze();
    assert!(matches!(app.pending, Some(Pending::FreezeExpert(_, false))));
    app.apply_pending(&store).await.unwrap();
    let thawed = app
        .experts
        .iter()
        .find(|e| e.id.as_str() == "expert:e1")
        .unwrap();
    assert!(!thawed.is_frozen(), "the store reflects the thaw");
    assert!(app.events.iter().any(|ev| ev.text.contains("thawed")));
}

fn an_expert(id: &str, name: &str, fitness: f32, gen: u32) -> Expert {
    Expert {
        id: ExpertId::new(id),
        name: name.into(),
        base_model: "base".into(),
        artifact_uri: "a".into(),
        capability_card: serde_json::json!({}),
        capability_vec: None,
        fitness,
        frozen_at: None,
        generation: Generation(gen),
        owner: None,
        compartment: None,
        placed_on: None,
        created_at: Utc::now(),
    }
}

#[tokio::test]
async fn sort_experts_orders_and_preserves_selection() {
    let (_store, mut app) = seeded().await;
    app.experts = vec![
        an_expert("expert:b", "beta", 0.5, 2),
        an_expert("expert:a", "alpha", 0.9, 1),
        an_expert("expert:c", "gamma", 0.7, 3),
    ];
    app.selected = 0; // beta

    app.sort = SortKey::Fitness;
    app.sort_experts();
    let names: Vec<&str> = app.experts.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(
        names,
        ["alpha", "gamma", "beta"],
        "fitness, strongest first"
    );
    assert_eq!(
        app.selected_expert().unwrap().name,
        "beta",
        "selection tracks the expert across the re-sort"
    );

    app.cycle_sort(); // Fitness → Name
    assert_eq!(app.sort, SortKey::Name);
    let names: Vec<&str> = app.experts.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, ["alpha", "beta", "gamma"]);

    app.cycle_sort(); // Name → Generation
    assert_eq!(
        app.experts.first().unwrap().name,
        "gamma",
        "newest gen first"
    );
}

#[tokio::test]
async fn loop_and_evals_load_for_their_pages() {
    let (store, mut app) = seeded().await;
    assert!(app.loop_heads.is_empty());
    assert!(app.evals.is_empty());
    let now = Utc::now();

    let mut head = GenerationHead::new(RunId::new("run:x"), now);
    head.state = LoopState::Decide;
    generation::save_head(&store, &head).await.unwrap();
    evaluation::insert(
        &store,
        &EvaluationRun {
            run_id: RunId::new("run:e"),
            subject_kind: SubjectKind::Expert,
            subject_id: "expert:a".into(),
            corpus_task_id: "task:a".into(),
            status: EvalStatus::Failure,
            metrics: None,
            regression_fingerprint: None,
            created_at: now,
        },
    )
    .await
    .unwrap();
    app.reload(&store).await.unwrap();

    assert_eq!(app.loop_heads.len(), 1);
    assert_eq!(app.loop_heads[0].state, LoopState::Decide);
    assert_eq!(app.evals.len(), 1);
    assert_eq!(app.evals[0].status, EvalStatus::Failure);

    // The Evals page drives the run selection (and is inert for mutations).
    app.page = Page::Evals;
    app.select_next();
    assert_eq!(app.selected_eval, 0, "single run, selection holds");
    app.request_action();
    assert!(app.pending.is_none());
}

#[tokio::test]
async fn loop_page_selects_runs_and_drills_into_eval_history() {
    let (store, mut app) = seeded().await;
    let now = Utc::now();
    let mut a = GenerationHead::new(RunId::new("run:a"), now);
    a.state = LoopState::Score;
    let b = GenerationHead::new(RunId::new("run:b"), now);
    generation::save_head(&store, &a).await.unwrap();
    generation::save_head(&store, &b).await.unwrap();
    // One evaluation, tied to run:b only.
    evaluation::insert(
        &store,
        &EvaluationRun {
            run_id: RunId::new("run:b"),
            subject_kind: SubjectKind::Shadow,
            subject_id: "shadow:b0".into(),
            corpus_task_id: "task:b".into(),
            status: EvalStatus::Success,
            metrics: None,
            regression_fingerprint: None,
            created_at: now,
        },
    )
    .await
    .unwrap();
    app.reload(&store).await.unwrap();
    assert_eq!(app.loop_heads.len(), 2);

    // j/k drive selected_loop and wrap (the Loop page now owns a selection).
    app.page = Page::Loop;
    assert_eq!(app.selected_loop, 0);
    app.select_next();
    assert_eq!(app.selected_loop, 1);
    app.select_next();
    assert_eq!(app.selected_loop, 0, "wraps");
    app.select_prev();
    assert_eq!(app.selected_loop, 1, "wraps back");

    // Enter drills in — the Loop page now has a detail overlay.
    app.open_detail();
    assert_eq!(app.mode, Mode::Detail);

    // The drill-down's eval history is scoped to one run (the Loop→Evals
    // link), independent of how the store ordered the heads.
    let with_eval = app
        .loop_heads
        .iter()
        .find(|h| h.run_id.as_str() == "run:b")
        .unwrap()
        .clone();
    let without = app
        .loop_heads
        .iter()
        .find(|h| h.run_id.as_str() == "run:a")
        .unwrap()
        .clone();
    let hist = app.loop_eval_history(&with_eval);
    assert_eq!(hist.len(), 1);
    assert_eq!(hist[0].subject_id, "shadow:b0");
    assert!(app.loop_eval_history(&without).is_empty());

    // Reload clamps an out-of-range selection back into the heads.
    app.selected_loop = 9;
    app.reload(&store).await.unwrap();
    assert!(app.selected_loop < app.loop_heads.len());

    // With no heads, Enter is a no-op (no empty drill-down).
    app.loop_heads.clear();
    app.mode = Mode::Normal;
    app.open_detail();
    assert_eq!(app.mode, Mode::Normal);
}

#[tokio::test]
async fn loop_halt_request_writes_the_control_and_cancel_clears_it() {
    let (store, mut app) = seeded().await;
    let head = GenerationHead::new(RunId::new("run:z"), Utc::now());
    generation::save_head(&store, &head).await.unwrap();
    app.reload(&store).await.unwrap();
    assert_eq!(app.loop_heads.len(), 1);
    assert!(!app.loop_halt_pending);

    // `x` on the Loop page stages a halt confirm; applying writes the control.
    app.page = Page::Loop;
    app.request_action();
    assert!(matches!(app.pending, Some(Pending::HaltLoop(_))));
    app.apply_pending(&store).await.unwrap();
    assert!(app.loop_halt_pending, "halt is now pending");
    assert_eq!(
        loop_control::load(&store, &RunId::new("run:z"))
            .await
            .unwrap(),
        LoopCommand::Halt
    );

    // Re-requesting is inert while a halt is already pending.
    app.request_action();
    assert!(app.pending.is_none());

    // Canceling clears the control back to Run.
    app.cancel_halt(&store).await.unwrap();
    assert!(!app.loop_halt_pending);
    assert_eq!(
        loop_control::load(&store, &RunId::new("run:z"))
            .await
            .unwrap(),
        LoopCommand::Run
    );
}

#[tokio::test]
async fn eval_history_groups_by_subject_and_drill_down_opens() {
    let (_store, mut app) = seeded().await;
    let now = Utc::now();
    let mk = |id: &str, subj: &str, status| EvaluationRun {
        run_id: RunId::new(id),
        subject_kind: SubjectKind::Expert,
        subject_id: subj.into(),
        corpus_task_id: "task:a".into(),
        status,
        metrics: None,
        regression_fingerprint: None,
        created_at: now,
    };
    app.evals = vec![
        mk("run:x2", "expert:x", EvalStatus::Failure),
        mk("run:x1", "expert:x", EvalStatus::Success),
        mk("run:y", "expert:y", EvalStatus::Success),
    ];
    let first = app.evals[0].clone();
    assert_eq!(app.eval_history(&first).len(), 2, "two runs for expert:x");
    assert_eq!(app.eval_history(&app.evals[2].clone()).len(), 1);

    // Enter drills into a run on the Evals page (inert when there are none).
    app.page = Page::Evals;
    app.selected_eval = 0;
    app.open_detail();
    assert_eq!(app.mode, Mode::Detail);
    app.mode = Mode::Normal;
    app.evals.clear();
    app.open_detail();
    assert_eq!(app.mode, Mode::Normal, "no drill-down with no runs");
}

#[tokio::test]
async fn dashboard_metrics_summarize_the_substrate() {
    let (_store, mut app) = seeded().await;
    let m = app.dashboard_metrics();
    assert_eq!(
        m.map(|(label, _)| label),
        ["fitness", "frozen", "graduated", "gating", "memory"]
    );
    // No experts seeded → zero mean fitness; the one shadow is exploring.
    assert_eq!(m[0].1, 0.0, "no experts");
    assert_eq!(m[2].1, 0.0, "no graduated shadows");
    // Graduate the lone shadow → the graduation ratio becomes 1/1.
    app.shadows[0].status = ShadowStatus::Graduated;
    assert_eq!(app.dashboard_metrics()[2].1, 1.0);
}

#[tokio::test]
async fn memories_and_edges_load_and_drive_the_memory_page() {
    let (store, mut app) = seeded().await;
    assert!(app.memories.is_empty(), "no memory traces in the base seed");
    let now = Utc::now();
    memory::upsert(
        &store,
        &Memory::new("memory:a", "ws:1", MemoryNetwork::World, "alpha", 0.5, now),
    )
    .await
    .unwrap();
    memory::upsert(
        &store,
        &Memory::new("memory:b", "ws:1", MemoryNetwork::Bank, "beta", 0.9, now),
    )
    .await
    .unwrap();
    edge::relate(
        &store,
        &MemoryEdge::new(
            "ws:1",
            "memory:a",
            "memory:b",
            EdgeType::References,
            0.5,
            now,
        ),
    )
    .await
    .unwrap();
    app.reload(&store).await.unwrap();

    assert_eq!(app.memories.len(), 2);
    assert_eq!(app.edges.len(), 1);

    // On the Memory page, navigation drives the trace selection and wraps.
    app.page = Page::Memory;
    app.selected_memory = 0;
    let id = app.selected_memory().unwrap().id.as_str().to_string();
    assert_eq!(app.memory_edges(&id).len(), 1, "the selected trace's edge");
    app.select_next();
    assert_eq!(app.selected_memory, 1);
    app.select_next();
    assert_eq!(app.selected_memory, 0, "selection wraps");

    // Operator actions and the drill-down overlay are inert on the Memory page.
    app.request_action();
    assert!(app.pending.is_none());
    app.open_detail();
    assert_eq!(app.mode, Mode::Normal);
}

#[tokio::test]
async fn apply_pending_is_a_noop_with_nothing_staged() {
    let (store, mut app) = seeded().await;
    app.apply_pending(&store).await.unwrap();
    assert_eq!(app.shadows.len(), 1);
    assert_eq!(app.boundaries.len(), 1);
    assert_eq!(app.mode, Mode::Normal);
}

#[tokio::test]
async fn request_action_skips_a_pruned_shadow_and_experts() {
    let (_store, mut app) = seeded().await;
    // Pruned shadow: no action is staged.
    app.shadows[0].status = ShadowStatus::Pruned;
    app.focus = Focus::Shadows;
    app.request_action();
    assert!(
        app.pending.is_none(),
        "no prune on an already-pruned shadow"
    );
    assert_eq!(app.mode, Mode::Normal);
    // Experts focus has no operator action.
    app.focus = Focus::Experts;
    app.request_action();
    assert!(app.pending.is_none());
}
