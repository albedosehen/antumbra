//! The console's fixtures and its golden frames.
//!
//! A golden is a whole rendered frame compared byte for byte, so these are
//! what says the console still looks like itself. The behaviour half sits in
//! `behaviour`, a child of this module, so it reaches these fixtures the same
//! way it always did.

use super::*;
use crate::app::{Focus, LayoutMode, Mode, Palette};
use crate::command::Action;
use antumbra_core::generational::{GenerationHead, LoopState};
use antumbra_core::router::{LearnedRouter, RouterExpert};
use antumbra_core::{
    BoundaryId, EdgeType, EvalStatus, EvaluationRun, Expert, ExpertId, FailureBoundary, Generation,
    Grain, Memory, MemoryEdge, MemoryNetwork, RunId, Shadow, ShadowId, ShadowStatus, SubjectKind,
};
use chrono::Utc;
use ratatui::layout::{Constraint, Layout};

/// One loop head, mid-cycle (the Loop page).
fn demo_loop_heads(now: chrono::DateTime<Utc>) -> Vec<GenerationHead> {
    let mut head = GenerationHead::new(RunId::new("run:demo"), now);
    head.generation = Generation(4);
    head.state = LoopState::Score;
    vec![head]
}

/// A few evaluation runs across subjects and statuses (the Evals page).
fn demo_evals(now: chrono::DateTime<Utc>) -> Vec<EvaluationRun> {
    let mk = |id: &str, kind, subj: &str, status, fp: Option<&str>| EvaluationRun {
        run_id: RunId::new(id),
        subject_kind: kind,
        subject_id: subj.into(),
        corpus_task_id: "task:arith".into(),
        status,
        metrics: None,
        regression_fingerprint: fp.map(str::to_string),
        created_at: now,
    };
    vec![
        // Two runs for the same frozen expert: the newer one drifted (the
        // no-forgetting tripwire), so the drill-down has a real comparison.
        mk(
            "run:1b",
            SubjectKind::Expert,
            "expert:arith-specialist",
            EvalStatus::Failure,
            Some("ffffffff"),
        ),
        mk(
            "run:1",
            SubjectKind::Expert,
            "expert:arith-specialist",
            EvalStatus::Success,
            Some("a1b2c3d4"),
        ),
        mk(
            "run:2",
            SubjectKind::Shadow,
            "shadow:g3",
            EvalStatus::Failure,
            None,
        ),
        mk(
            "run:3",
            SubjectKind::Router,
            "router",
            EvalStatus::Running,
            None,
        ),
    ]
}

/// Three demo traces (one per network) and two edges, for the Memory page.
fn demo_memories(now: chrono::DateTime<Utc>) -> Vec<Memory> {
    let mut deno = Memory::new(
        "memory:deno",
        "ws:demo",
        MemoryNetwork::World,
        "prefer deno over node for new scripts",
        0.86,
        now,
    )
    .with_evidence(vec!["session-log".into()]);
    deno.reinforcement = 4;
    let mut incident = Memory::new(
        "memory:incident",
        "ws:demo",
        MemoryNetwork::Bank,
        "npm install failed under the deno runtime",
        0.70,
        now,
    );
    incident.reinforcement = 2;
    incident.volatile = true;
    let terse = Memory::new(
        "memory:terse",
        "ws:demo",
        MemoryNetwork::Opinion,
        "keep operator docs terse, no emojis",
        0.80,
        now,
    );
    vec![deno, incident, terse]
}

fn demo_edges(now: chrono::DateTime<Utc>) -> Vec<MemoryEdge> {
    vec![
        MemoryEdge::new(
            "ws:demo",
            "memory:incident",
            "memory:deno",
            EdgeType::Contradicts,
            0.8,
            now,
        ),
        MemoryEdge::new(
            "ws:demo",
            "memory:deno",
            "memory:incident",
            EdgeType::Caused,
            0.6,
            now,
        ),
    ]
}

fn demo_app() -> App {
    let now = Utc::now();
    let expert = Expert {
        id: ExpertId::new("expert:arith"),
        name: "arith-specialist".into(),
        base_model: "base".into(),
        artifact_uri: "mem://a".into(),
        capability_card: serde_json::json!({ "exemplars": ["a", "b"] }),
        capability_vec: Some(vec![0.0; 8]),
        fitness: 0.9,
        frozen_at: Some(now),
        generation: Generation::ZERO,
        owner: None,
        compartment: None,
        placed_on: None,
        created_at: now,
    };
    App {
        experts: vec![expert],
        boundaries: Vec::new(),
        shadows: Vec::new(),
        memories: demo_memories(now),
        edges: demo_edges(now),
        selected_memory: 0,
        sovereign: crate::sovereign::View::default(),
        selected_skill: 0,
        loop_heads: demo_loop_heads(now),
        selected_loop: 0,
        loop_halt_pending: false,
        evals: demo_evals(now),
        selected_eval: 0,
        router: None,
        selected: 0,
        selected_boundary: 0,
        selected_shadow: 0,
        page: crate::app::Page::Population,
        focus: Focus::Experts,
        layout: LayoutMode::Focused,
        sort: crate::app::SortKey::Fitness,
        mode: Mode::Normal,
        palette: Palette::default(),
        filter: String::new(),
        filter_selected: 0,
        events: Vec::new(),
        events_scroll: 0,
        detail_scroll: 0,
        pending: None,
        ask_query: String::new(),
        ask_result: None,
        real_embedder: false,
        ev_experts: std::collections::HashMap::new(),
        ev_shadows: std::collections::HashMap::new(),
        ev_boundaries: std::collections::HashMap::new(),
        ev_router: false,
        ev_baseline: false,
        theme_idx: 0,
        target_fps: 144,
        auto_fps: false,
        window: 0,
        fps: 0.0,
        clock_ms: 0.0,
        since_reload_ms: 0.0,
        since_input_ms: 0.0,
        power_save: false,
        idle: false,
        render_tier: crate::render::RenderTier::default(),
        demo: false,
        store_url: "surrealkv://./data/antumbra.skv".into(),
        should_quit: false,
    }
}

fn graduated_shadow() -> Shadow {
    Shadow {
        id: ShadowId::new("shadow:g3"),
        parent_expert: None,
        adapter_uri: Some("mem://g3".into()),
        status: ShadowStatus::Graduated,
        generation: Generation(3),
        reward_curve: vec![0.2, 0.6, 0.85, 0.93],
        created_at: Utc::now(),
    }
}

fn actionable_boundary() -> FailureBoundary {
    FailureBoundary {
        id: BoundaryId::new("boundary:demo-0"),
        behavior: "run `npm install`".into(),
        fail_context: serde_json::json!({ "runtime": "deno" }),
        near_ok_context: Some(serde_json::json!({ "runtime": "node" })),
        governing_features: vec!["runtime".into()],
        grain: Some(Grain::Project),
        context_vec: None,
        ok_context_vec: None,
        confidence: 0.8,
        generation: Generation::ZERO,
        created_at: Utc::now(),
    }
}

/// Compare `actual` against the stored golden `tests/golden/<name>.txt`.
/// Re-create/update goldens by running with `ANTUMBRA_BLESS=1`.
fn assert_golden(name: &str, actual: &str) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
        .join(format!("{name}.txt"));
    if std::env::var_os("ANTUMBRA_BLESS").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, actual).unwrap();
        return;
    }
    let expected = std::fs::read_to_string(&path).unwrap_or_else(|_| {
        panic!(
            "missing golden {}; run `ANTUMBRA_BLESS=1 cargo test -p antumbra-tui golden`",
            path.display()
        )
    });
    assert!(
        actual == expected,
        "golden mismatch for `{name}` (re-bless with ANTUMBRA_BLESS=1 if intentional)\
         \n--- expected ---\n{expected}\n--- actual ---\n{actual}"
    );
}

/// Render the active overlay's modal box and golden just that region, so the
/// animated graph behind it never churns the comparison.
fn golden_overlay(name: &str, app: &mut App, w: u16, h: u16) {
    let buf = render(app, w, h, 1600.0).unwrap();
    let rect = crate::ui::overlay_area(app, buf.area).expect("an overlay must be open");
    assert_golden(name, &to_text_in(&buf, rect));
}

// Golden the keybinding help modal exactly; adding/renaming a binding must be
// a deliberate re-bless, not a silent drift.
#[test]
fn golden_help_overlay() {
    let mut app = demo_app();
    app.toggle_help();
    golden_overlay("help", &mut app, 120, 36);
}

// Golden the command palette's full command list.
#[test]
fn golden_command_palette() {
    let mut app = demo_app();
    app.open_palette();
    golden_overlay("palette", &mut app, 120, 36);
}

// Golden the Memory page: the penumbra edge graph (memories clustered by
// network, type-coloured edges) beside the selected trace's detail. The graph
// is statically laid out (no animation clock), so the full frame is stable.
#[test]
fn golden_memory_page() {
    let mut app = demo_app();
    app.set_page(crate::app::Page::Memory);
    let buf = render(&mut app, 120, 36, 1600.0).unwrap();
    assert_golden("memory_page", &to_text(&buf));
}

// Golden the Sovereign page: the rules on the left, skill use on the right,
// stalest first. Fixed dates, so the frame does not depend on the clock (the
// clock only decides a colour, which the text golden does not see).
#[test]
fn golden_sovereign_page() -> anyhow::Result<()> {
    use chrono::TimeZone;
    let day = |d: u32| {
        Utc.with_ymd_and_hms(2026, 9, d, 12, 0, 0)
            .single()
            .ok_or_else(|| anyhow::anyhow!("2026-09-{d} is not a date"))
    };
    let keyed = [
        ("[claude-code:verified] The claude-code rules in this compartment were checked against Claude Code 2.1.278 on 2026-09-19.", 0.9, 0, 19),
        ("[claude-code:agents-md] AGENTS.md is no longer read as project instructions.", 0.9, 0, 19),
        ("[claude-code:agents-md] an earlier text of the same rule", 0.675, 0, 12),
        ("[claude-code:mcp-schemas] MCP tools the API rejects are no longer excluded.", 0.9, 0, 19),
        ("[claude-code:advisor] the advisor tool", 0.675, 0, 19),
        ("[skill-use:elegant-design] elegant-design", 0.9, 11, 18),
        ("[skill-use:code-review] code-review", 0.9, 2, 3),
        ("[skill-use:deploy] deploy", 0.9, 0, 9),
    ];
    let mut app = demo_app();
    for (n, (content, confidence, reinforcement, d)) in keyed.into_iter().enumerate() {
        let when = day(d)?;
        let mut m = Memory::new(
            format!("memory:keyed-{n}"),
            "ws:demo",
            MemoryNetwork::World,
            content,
            confidence,
            when,
        );
        m.reinforcement = reinforcement;
        m.updated_at = when;
        app.memories.push(m);
    }
    app.sovereign = crate::sovereign::View::from_memories(&app.memories);
    app.set_page(crate::app::Page::Sovereign);
    app.select_next();
    let buf = render(&mut app, 120, 36, 1600.0)?;
    assert_golden("sovereign_page", &to_text(&buf));
    Ok(())
}

// Golden the sortable population table (full-width data grid, no animated
// graph, so the full frame is deterministic). Default sort is fitness-desc.
#[test]
fn golden_population_table() {
    let now = Utc::now();
    let mk = |id: &str, name: &str, fitness: f32, gen: u32, frozen: bool| Expert {
        id: ExpertId::new(id),
        name: name.into(),
        base_model: "Qwen2.5-Coder-1.5B".into(),
        artifact_uri: "mem://x".into(),
        capability_card: serde_json::json!({}),
        capability_vec: None,
        fitness,
        frozen_at: frozen.then_some(now),
        generation: Generation(gen),
        owner: None,
        compartment: None,
        placed_on: None,
        created_at: now,
    };
    let mut app = demo_app();
    app.experts = vec![
        mk("expert:arith", "arith-specialist", 0.92, 0, true),
        mk("expert:string", "string-specialist", 0.81, 1, true),
        mk("expert:json", "json-shaper", 0.55, 2, false),
    ];
    app.sort_experts();
    app.set_layout(LayoutMode::Table);
    let buf = render(&mut app, 120, 36, 1600.0).unwrap();
    assert_golden("population_table", &to_text(&buf));
}

// Golden the reward-landscape heatmap (the bottom strip of the dashboard
// layout). Scoped to that strip so the animated population graph above it
// never churns the comparison; the heatmap itself is deterministic.
#[test]
fn golden_reward_heatmap() {
    let now = Utc::now();
    let mk = |id: &str, curve: Vec<f32>| Shadow {
        id: ShadowId::new(id),
        parent_expert: None,
        adapter_uri: None,
        status: ShadowStatus::Exploring,
        generation: Generation(4),
        reward_curve: curve,
        created_at: now,
    };
    let mut app = demo_app();
    app.shadows = vec![
        mk("shadow:rise", vec![0.1, 0.4, 0.7, 0.95]),
        mk("shadow:flat", vec![0.5, 0.5, 0.5, 0.5]),
        mk("shadow:fall", vec![0.6, 0.3, 0.1, 0.05]),
    ];
    app.set_layout(LayoutMode::Dashboard);
    let buf = render(&mut app, 120, 36, 1600.0).unwrap();
    // Recompute the heatmap strip: the body is the 4th chrome row, then the
    // dashboard puts the heatmap in the bottom Length(9) slice.
    let chrome = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Min(0),
        Constraint::Length(1),
    ])
    .split(buf.area);
    let strip = Layout::vertical([Constraint::Min(0), Constraint::Length(9)]).split(chrome[3])[1];
    assert_golden("reward_heatmap", &to_text_in(&buf, strip));
}

// Golden the Loop page: the generational pipeline with the current stage lit
// (deterministic: no animated graph, no wall-clock fields shown).
#[test]
fn golden_loop_page() {
    let mut app = demo_app();
    app.set_page(crate::app::Page::Loop);
    let buf = render(&mut app, 120, 36, 1600.0).unwrap();
    assert_golden("loop_page", &to_text(&buf));
}

// Golden the Loop page with a pending halt (the operator graceful-stop).
#[test]
fn golden_loop_halted() {
    let mut app = demo_app();
    app.set_page(crate::app::Page::Loop);
    app.loop_halt_pending = true;
    let buf = render(&mut app, 120, 36, 1600.0).unwrap();
    assert_golden("loop_halted", &to_text(&buf));
}

// Golden the Evals page: the evaluation-run table with a regression failure
// (deterministic: fixed runs, no timestamps shown).
#[test]
fn golden_evals_page() {
    let mut app = demo_app();
    app.set_page(crate::app::Page::Evals);
    let buf = render(&mut app, 120, 36, 1600.0).unwrap();
    assert_golden("evals_page", &to_text(&buf));
}

// Golden the evaluation drill-down: the regression comparison + run history
// for a drifted subject (scoped to the modal; no timestamps shown).
#[test]
fn golden_eval_detail() {
    let mut app = demo_app();
    app.set_page(crate::app::Page::Evals);
    app.selected_eval = 0; // the drifted arith run
    app.open_detail();
    golden_overlay("eval_detail", &mut app, 120, 36);
}

// Golden the loop drill-down: the done/current/pending lifecycle ladder and
// the run's evaluation history (the Loop → Evals link), scoped to the modal.
#[test]
fn golden_loop_detail() {
    let mut app = demo_app();
    app.set_page(crate::app::Page::Loop);
    // Tie a couple of eval runs to the displayed run (run:demo) so the
    // drill-down shows a non-empty evaluation history.
    let now = Utc::now();
    let ev = |subj: &str, kind, status| EvaluationRun {
        run_id: RunId::new("run:demo"),
        subject_kind: kind,
        subject_id: subj.into(),
        corpus_task_id: "task:arith".into(),
        status,
        metrics: None,
        regression_fingerprint: None,
        created_at: now,
    };
    app.evals = vec![
        ev(
            "expert:arith-specialist",
            SubjectKind::Expert,
            EvalStatus::Success,
        ),
        ev("shadow:g4", SubjectKind::Shadow, EvalStatus::Failure),
    ];
    app.open_detail();
    golden_overlay("loop_detail", &mut app, 120, 36);
}

// Golden the Loop page with several live runs: the selectable run list with
// the focused run highlighted, its pipeline below (no timestamps shown).
#[test]
fn golden_loop_multihead() {
    let mut app = demo_app();
    app.set_page(crate::app::Page::Loop);
    let now = Utc::now();
    let mk = |id: &str, gen: u32, state| {
        let mut h = GenerationHead::new(RunId::new(id), now);
        h.generation = Generation(gen);
        h.state = state;
        h
    };
    app.loop_heads = vec![
        mk("run:arith", 4, LoopState::Score),
        mk("run:strings", 2, LoopState::Explore),
        mk("run:lists", 7, LoopState::Consolidate),
    ];
    app.selected_loop = 1;
    let buf = render(&mut app, 120, 36, 1600.0).unwrap();
    assert_golden("loop_multihead", &to_text(&buf));
}

// Golden the gate (router) inspector: the weight profile + self-routing
// health (a healthy gate routes each centroid back to itself).
#[test]
fn golden_gate_inspector() {
    let onehot = |n: usize, i: usize| {
        let mut v = vec![0.0f32; n];
        v[i] = 1.0;
        v
    };
    let mut app = demo_app();
    app.router = Some(LearnedRouter {
        weights: vec![0.2, 0.6, 1.0, 1.6, 2.0, 1.4, 0.8, 0.3],
        experts: vec![
            RouterExpert {
                id: ExpertId::new("expert:arith-specialist"),
                centroid: onehot(8, 2),
            },
            RouterExpert {
                id: ExpertId::new("expert:string-specialist"),
                centroid: onehot(8, 4),
            },
            RouterExpert {
                id: ExpertId::new("expert:json-shaper"),
                centroid: onehot(8, 6),
            },
        ],
        temperature: 0.2,
        floor: 0.1,
    });
    app.open_gate();
    golden_overlay("gate_inspector", &mut app, 120, 36);
}

// Golden the operator-action confirm prompt.
#[test]
fn golden_confirm_delete() {
    let mut app = demo_app();
    app.boundaries = vec![actionable_boundary()];
    app.focus = Focus::Boundaries;
    app.request_action();
    golden_overlay("confirm_delete", &mut app, 120, 36);
}

// Golden the live event-stream overlay (events stamped at clock 0, frame at
// 1600ms → "1s ago", so it's deterministic).
#[test]
fn golden_event_stream() {
    let mut app = demo_app();
    app.operator_event("string-specialist emerged".into());
    app.operator_event("shadow:g4 graduated".into());
    app.open_events();
    golden_overlay("events", &mut app, 100, 12);
}

/// Golden the focused-layout right detail column (the panels, no animated
/// graph, no wall-clock timestamps, so it's stable).
fn golden_detail_column(name: &str, app: &mut App, w: u16, h: u16) {
    let buf = render(app, w, h, 1600.0).unwrap();
    let rect = crate::transition::detail_area(buf.area);
    assert_golden(name, &to_text_in(&buf, rect));
}

#[test]
fn golden_focused_experts_panel() {
    let mut app = demo_app();
    golden_detail_column("focused_experts", &mut app, 120, 36);
}

#[test]
fn golden_focused_shadows_panel() {
    let mut app = demo_app();
    app.shadows = vec![graduated_shadow()];
    app.focus = Focus::Shadows;
    golden_detail_column("focused_shadows", &mut app, 120, 36);
}

#[test]
fn golden_focused_boundaries_panel() {
    let mut app = demo_app();
    app.boundaries = vec![actionable_boundary()];
    app.focus = Focus::Boundaries;
    golden_detail_column("focused_boundaries", &mut app, 120, 36);
}

mod behaviour;
