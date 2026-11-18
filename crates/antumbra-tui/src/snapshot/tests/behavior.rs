//! What the console does, as opposed to how it looks: navigation, the
//! overlays, filtering, the frame-rate controls, and the event stream.
//!
//! A child of the snapshot tests, so `super::*` is their fixtures.

use super::*;

// e2e of the console without a terminal: render a frame and assert on the
// text grid. The whole layout is exercised (header, graph, detail, gate).
#[test]
fn headless_render_shows_the_console() -> anyhow::Result<()> {
    let mut app = demo_app();
    let buf = render(&mut app, 120, 36, 1600.0)?;
    let text = to_text(&buf);
    assert!(
        text.contains("Antumbra"),
        "header wordmark missing:\n{text}"
    );
    assert!(text.contains("Experts 1"), "the list's title is missing");
    assert!(
        text.contains("▸ arith-specialist"),
        "the expert is listed, and marked as selected"
    );
    assert!(text.contains("╭ Routing ─"), "routing panel title missing");
    assert!(text.contains("0.90"), "selected expert's fitness rendered");
    Ok(())
}

#[test]
fn empty_population_renders_the_hint() -> anyhow::Result<()> {
    let mut app = demo_app();
    app.experts.clear();
    let text = to_text(&render(&mut app, 100, 30, 1600.0)?);
    assert!(text.contains("No experts yet"), "empty-state hint shown");
    Ok(())
}

#[test]
fn fully_empty_store_shows_the_first_run_card() -> anyhow::Result<()> {
    let mut app = demo_app();
    app.experts.clear();
    app.boundaries.clear();
    app.shadows.clear();
    app.memories.clear();
    app.edges.clear();
    app.loop_heads.clear();
    app.evals.clear();
    assert!(app.is_empty(), "a store with nothing in it reads as empty");
    let text = to_text(&render(&mut app, 100, 30, 1600.0)?);
    assert!(
        text.contains("Nothing is here yet"),
        "the first-run card is shown:\n{text}"
    );
    // It points at what exists: the connect panel, the help and the demo.
    assert!(text.contains("how to connect an agent"), "{text}");
    assert!(
        text.contains("antumbra tui --demo"),
        "the demo, by a command that runs:\n{text}"
    );
    // And at nothing that does not: there is no `antumbra-tui` binary, and
    // `seed` and `loop` write to the CLI's own store, not the server's.
    for wrong in ["antumbra-tui", "antumbra seed", "antumbra loop"] {
        assert!(!text.contains(wrong), "the card names `{wrong}`:\n{text}");
    }
    // The card belongs to the live view only; an open overlay replaces it.
    app.mode = Mode::Help;
    let over = to_text(&render(&mut app, 100, 30, 1600.0)?);
    assert!(
        !over.contains("Nothing is here yet"),
        "a modal supersedes the card"
    );
    Ok(())
}

#[test]
fn connect_overlay_guides_agent_wiring() -> anyhow::Result<()> {
    let mut app = demo_app();
    app.open_connect();
    assert_eq!(app.mode, Mode::Connect, "the connect panel is open");
    let text = to_text(&render(&mut app, 100, 32, 1600.0)?);
    assert!(
        text.contains("antumbra setup local"),
        "shows the one-command setup"
    );
    assert!(text.contains("antumbra setup check"), "and how to check it");
    assert!(
        text.contains("http://127.0.0.1:8081"),
        "names the server this console reads"
    );
    // The read-only line names an operator's token too (it says how to mint
    // one), so the two roles are told apart by what only each one says.
    assert!(
        text.contains("an operator's: it may act") && !text.contains("read-only"),
        "an operator's console says it may act"
    );

    app.operator = false;
    let text = to_text(&render(&mut app, 100, 32, 1600.0)?);
    assert!(
        text.contains("read-only") && !text.contains("it may act"),
        "a plain token's console says it is read-only"
    );
    Ok(())
}

// The loop runs full-rate while interacting or animating, eases to 60 after
// an idle spell, and the connect panel, which says how the console is
// drawing, marks the eased state.
#[test]
fn idle_throttle_eases_the_rate_and_the_connect_panel_says_so() -> anyhow::Result<()> {
    let mut app = demo_app();
    app.target_fps = 244;
    app.power_save = true;
    // Fresh input: full rate.
    app.note_input();
    assert_eq!(app.frame_cap(false), 244);
    // A still spell eases to 60, unless something is animating.
    app.tick(10_000.0);
    assert_eq!(app.frame_cap(true), 244, "animation keeps full rate");
    assert_eq!(app.frame_cap(false), 60, "an idle, still view eases to 60");
    // With power-save off (the default), the still view stays at full rate.
    app.power_save = false;
    assert_eq!(app.frame_cap(false), 244, "default never throttles");
    // The connect panel reflects the eased state.
    app.idle = true;
    app.open_connect();
    let text = to_text(&render(&mut app, 130, 36, 1600.0)?);
    assert!(
        text.contains("fps, idle"),
        "the panel marks the idle throttle:\n{text}"
    );
    Ok(())
}

// Home/End/PageUp/PageDown drive the focused list, clamped to the ends.
#[test]
fn page_and_jump_navigation_clamps_to_the_list() -> anyhow::Result<()> {
    let mut app = demo_app();
    app.shadows = (0..25)
        .map(|i| Shadow {
            id: ShadowId::new(format!("shadow:s{i}")),
            parent_expert: None,
            adapter_uri: None,
            status: ShadowStatus::Exploring,
            generation: Generation(i),
            reward_curve: vec![],
            created_at: Utc::now(),
        })
        .collect();
    app.focus = Focus::Shadows;
    app.select_last();
    assert_eq!(app.selected_shadow, 24, "End jumps to the last item");
    app.select_page(-1);
    assert_eq!(app.selected_shadow, 14, "PageUp moves up a page of 10");
    app.select_first();
    assert_eq!(app.selected_shadow, 0, "Home jumps to the first item");
    app.select_page(-1);
    assert_eq!(app.selected_shadow, 0, "PageUp clamps at the top");
    app.select_page(99);
    assert_eq!(
        app.selected_shadow, 24,
        "a big page jump clamps at the bottom"
    );
    Ok(())
}

// The dashboard layout shows all three regions at once (umbra, penumbra,
// antumbra), unlike the focused layout that swaps a single detail panel.
#[test]
fn dashboard_layout_shows_every_region_at_once() -> anyhow::Result<()> {
    let mut app = demo_app();
    app.shadows = vec![graduated_shadow()];
    app.boundaries = vec![actionable_boundary()];
    app.layout = LayoutMode::Dashboard;
    let text = to_text(&render(&mut app, 130, 40, 1600.0)?);
    assert!(text.contains("Experts · 1"), "umbra panel shown:\n{text}");
    assert!(text.contains("Candidates · 1"), "candidates panel shown");
    assert!(text.contains("Boundaries · 1"), "antumbra panel shown");
    // All three regions' content is visible together.
    assert!(text.contains("arith-specialist"), "expert content shown");
    assert!(text.contains("shadow:g3"), "shadow content shown");
    assert!(text.contains("npm install"), "boundary content shown");
    Ok(())
}

// A list longer than its panel scrolls to keep the selection visible: a high
// selection shows the tail and hides the head.
#[test]
fn long_lists_scroll_to_the_selection() -> anyhow::Result<()> {
    let mut app = demo_app();
    app.shadows = (0..20)
        .map(|i| Shadow {
            id: ShadowId::new(format!("shadow:s{i}")),
            parent_expert: None,
            adapter_uri: None,
            status: ShadowStatus::Exploring,
            generation: Generation(i),
            reward_curve: vec![0.1, 0.2],
            created_at: Utc::now(),
        })
        .collect();
    app.focus = Focus::Shadows;
    app.selected_shadow = 19;
    // A short frame so the list panel is smaller than 20 rows.
    let text = to_text(&render(&mut app, 120, 18, 1600.0)?);
    assert!(
        text.contains("shadow:s19"),
        "the selected tail is visible:\n{text}"
    );
    assert!(
        !text.contains("shadow:s0\n") && !text.contains("shadow:s0 "),
        "the head has scrolled out of view:\n{text}"
    );
    Ok(())
}

// The graph layout drops the detail column for a full-width population view.
#[test]
fn graph_layout_is_full_width() -> anyhow::Result<()> {
    let mut app = demo_app();
    app.layout = LayoutMode::Graph;
    let text = to_text(&render(&mut app, 120, 36, 1600.0)?);
    assert!(text.contains("╭ Experts ─"), "graph panel still shown");
    // The expert detail column is gone in graph mode.
    assert!(!text.contains("learned from"), "no detail column:\n{text}");
    Ok(())
}

// The command palette filters its commands by the typed query and marks the
// selection, so a fuzzy query surfaces the matching action.
#[test]
fn command_palette_filters_to_the_query() -> anyhow::Result<()> {
    let mut app = demo_app();
    app.open_palette();
    for c in "theme".chars() {
        app.palette_input(c);
    }
    let text = to_text(&render(&mut app, 120, 36, 1600.0)?);
    assert!(text.contains("Commands"), "palette title shown:\n{text}");
    assert!(
        text.contains("› theme"),
        "the typed query is echoed:\n{text}"
    );
    assert!(
        text.contains("Theme: change to the next"),
        "the theme command surfaced for the query:\n{text}"
    );
    // The chosen action matches the filtered selection.
    assert_eq!(app.palette_action(), Some(Action::CycleTheme));
    Ok(())
}

// The `/` filter fuzzy-narrows the focused list and jumps the selection to
// the chosen match.
#[test]
fn filter_narrows_and_jumps_the_selection() -> anyhow::Result<()> {
    let mut app = demo_app();
    app.shadows = vec![
        graduated_shadow(),
        Shadow {
            id: ShadowId::new("shadow:g7"),
            parent_expert: None,
            adapter_uri: None,
            status: ShadowStatus::Exploring,
            generation: Generation(7),
            reward_curve: vec![],
            created_at: Utc::now(),
        },
    ];
    app.focus = Focus::Shadows;
    app.open_filter();
    for c in "g7".chars() {
        app.filter_input(c);
    }
    // Only g7 matches; it is shown in the filter modal.
    let text = to_text(&render(&mut app, 120, 36, 1600.0)?);
    assert!(
        text.contains("find in candidates"),
        "the modal says which list it searches:\n{text}"
    );
    assert!(text.contains("shadow:g7"), "the match is listed");
    assert_eq!(app.filtered().len(), 1, "g7 is the only match");
    // Applying jumps the focused selection to g7 (index 1) and closes.
    app.filter_apply();
    assert_eq!(app.selected_shadow, 1, "selection jumped to the match");
    assert_eq!(app.mode, Mode::Normal, "filter closed after applying");
    Ok(())
}

// The event stream diffs successive reloads: a baseline pass records state
// silently, then changes (spawn, graduate, boundary) emit events.
#[test]
fn event_stream_diffs_store_changes() -> anyhow::Result<()> {
    use crate::events::EventKind;
    let mut app = demo_app();
    app.shadows = vec![Shadow {
        id: ShadowId::new("shadow:x"),
        parent_expert: None,
        adapter_uri: None,
        status: ShadowStatus::Exploring,
        generation: Generation(1),
        reward_curve: vec![],
        created_at: Utc::now(),
    }];
    // Baseline: only the "connected" event, no per-entity flood.
    app.record_events();
    assert!(app.ev_baseline);
    assert_eq!(app.events.len(), 1, "baseline emits one connect event");
    assert_eq!(app.events[0].kind, EventKind::System);

    // The shadow graduates.
    app.shadows[0].status = ShadowStatus::Graduated;
    app.record_events();
    assert!(
        app.events
            .iter()
            .any(|e| e.kind == EventKind::Graduate && e.text.contains("graduated")),
        "graduation emitted: {:?}",
        app.events.iter().map(|e| &e.text).collect::<Vec<_>>()
    );

    // A new boundary is recorded.
    app.boundaries = vec![actionable_boundary()];
    app.record_events();
    assert!(
        app.events.iter().any(|e| e.kind == EventKind::Boundary),
        "boundary event emitted"
    );

    // The overlay renders the stream.
    app.open_events();
    let text = to_text(&render(&mut app, 100, 24, 1600.0)?);
    assert!(
        text.contains("History · "),
        "history overlay titled:\n{text}"
    );
    assert!(text.contains("graduated"), "an event is listed");
    Ok(())
}

// Enter drills into the selected boundary: a full-detail modal with the
// contrastive contexts the summary panel omits.
#[test]
fn drill_down_shows_full_boundary_detail() -> anyhow::Result<()> {
    let mut app = demo_app();
    app.boundaries = vec![actionable_boundary()];
    app.focus = Focus::Boundaries;
    app.open_detail();
    let text = to_text(&render(&mut app, 110, 36, 1600.0)?);
    assert!(text.contains("Boundary ·"), "detail modal titled:\n{text}");
    assert!(
        text.contains("Wrong in this case"),
        "the case it was wrong in is shown"
    );
    assert!(
        text.contains("Right in this case"),
        "the C' context section shown"
    );
    // The pretty-printed JSON of the contexts is present.
    assert!(text.contains("runtime"), "context json rendered");
    Ok(())
}

// The expert drill-down previews how the gate routes that expert's own
// specialty (probing the router with its capability vector, no embedder).
#[test]
fn drill_down_previews_gate_routing() -> anyhow::Result<()> {
    use antumbra_core::router::{LearnedRouter, RouterExpert};
    let mut app = demo_app();
    let vec = {
        let mut v = vec![0.0f32; 8];
        v[1] = 1.0;
        v
    };
    app.experts[0].capability_vec = Some(vec.clone());
    app.router = Some(LearnedRouter {
        weights: vec![1.0; 8],
        experts: vec![RouterExpert {
            id: ExpertId::new("expert:arith"),
            centroid: vec,
        }],
        temperature: 0.2,
        floor: 0.0,
    });
    app.focus = Focus::Experts;
    app.open_detail();
    let text = to_text(&render(&mut app, 110, 36, 1600.0)?);
    assert!(
        text.contains("Routing · a task of this kind goes to"),
        "route preview shown:\n{text}"
    );
    assert!(
        text.contains("100%"),
        "the sole expert wins its own specialty"
    );
    assert!(text.contains("arith-specialist"), "routed expert named");
    Ok(())
}

// The route-ask overlay echoes the query and shows the gate's routing
// distribution, resolving expert ids to names.
#[test]
fn ask_overlay_shows_routing_distribution() -> anyhow::Result<()> {
    let mut app = demo_app();
    app.open_ask();
    for c in "arith".chars() {
        app.ask_input(c);
    }
    app.set_ask_result(&crate::backend::Routed {
        routes: vec![(ExpertId::new("expert:arith"), 0.82)],
        reason: None,
    });
    let text = to_text(&render(&mut app, 80, 14, 1600.0)?);
    assert!(
        text.contains("Ask · which expert takes a task?"),
        "ask modal titled:\n{text}"
    );
    assert!(text.contains("task › arith"), "query echoed");
    assert!(
        text.contains("arith-specialist"),
        "routed expert named (id resolved)"
    );
    assert!(text.contains("82%"), "routing probability shown");
    Ok(())
}

// `x` on a focused boundary stages a delete behind a confirm prompt; on a
// shadow it stages a prune. Canceling clears it.
#[test]
fn operator_action_stages_a_confirm_prompt() -> anyhow::Result<()> {
    let mut app = demo_app();
    app.boundaries = vec![actionable_boundary()];
    app.focus = Focus::Boundaries;
    app.request_action();
    assert_eq!(app.mode, Mode::Confirm, "confirm prompt opened");
    let text = to_text(&render(&mut app, 90, 24, 1600.0)?);
    assert!(
        text.contains("╭ Confirm ─"),
        "confirm modal titled:\n{text}"
    );
    assert!(
        text.contains("Routing will no longer avoid it"),
        "the prompt says what going ahead does:\n{text}"
    );
    assert!(text.contains("Delete boundary"), "the delete prompt shown");
    assert!(text.contains("npm install"), "names the boundary");
    app.cancel_action();
    assert_eq!(app.mode, Mode::Normal, "cancel returns to the live view");
    assert!(app.pending.is_none(), "the staged action is cleared");

    // On a shadow, the action prunes.
    app.shadows = vec![graduated_shadow()];
    app.focus = Focus::Shadows;
    app.request_action();
    let Some(prompt) = app.pending_prompt() else {
        anyhow::bail!("no action is staged on the shadow");
    };
    assert!(prompt.contains("Prune"), "shadow action is a prune");
    Ok(())
}

// `K` opens the memory knobs; an operator turns one and applies it behind a
// confirm, and a console without an operator's token only reads them.
#[test]
fn the_knobs_panel_turns_a_knob_behind_a_confirm() -> anyhow::Result<()> {
    let mut app = demo_app();
    app.operator = true;
    app.open_knobs();
    assert_eq!(app.mode, Mode::Knobs);
    let text = to_text(&render(&mut app, 100, 26, 1600.0)?);
    assert!(text.contains("Memory knobs"), "knobs modal titled:\n{text}");
    assert!(text.contains("world_half_life_days"), "every knob listed");

    app.knob_move(1);
    app.knob_turn(-1);
    assert_eq!(app.knob_draft.world_half_life_days, 165.0);
    let text = to_text(&render(&mut app, 100, 26, 1600.0)?);
    assert!(
        text.contains("(was 180)"),
        "the turned knob shows what it was"
    );
    app.request_knobs();
    assert_eq!(app.mode, Mode::Confirm);
    assert_eq!(
        app.pending_prompt().as_deref(),
        Some("Apply world_half_life_days 180 → 165?")
    );
    app.cancel_action();

    // Without an operator's token nothing turns, and the panel says why.
    app.operator = false;
    app.open_knobs();
    app.knob_turn(1);
    assert_eq!(app.knob_draft, app.policy);
    let text = to_text(&render(&mut app, 100, 26, 1600.0)?);
    assert!(
        text.contains("Read-only"),
        "says who may turn them:\n{text}"
    );
    app.close_knobs();
    assert_eq!(app.mode, Mode::Normal);
    Ok(())
}

// Theme, layout, and focus each cycle through their states and wrap.
#[test]
fn theme_layout_focus_cycle_and_wrap() -> anyhow::Result<()> {
    let mut app = demo_app();
    assert_eq!(app.theme().name, "night");
    app.cycle_theme();
    assert_eq!(app.theme().name, "mist");
    app.cycle_theme();
    assert_eq!(app.theme().name, "mono");
    app.cycle_theme();
    assert_eq!(app.theme().name, "night", "theme wraps");

    assert_eq!(app.layout, LayoutMode::List);
    app.cycle_layout();
    assert_eq!(app.layout, LayoutMode::Dashboard);
    app.cycle_layout();
    assert_eq!(app.layout, LayoutMode::Graph);
    app.cycle_layout();
    assert_eq!(app.layout, LayoutMode::Table);
    app.cycle_layout();
    assert_eq!(app.layout, LayoutMode::List, "layout wraps");

    assert_eq!(app.focus, Focus::Experts);
    app.toggle_focus();
    assert_eq!(app.focus, Focus::Shadows);
    app.toggle_focus();
    assert_eq!(app.focus, Focus::Boundaries);
    app.toggle_focus();
    assert_eq!(app.focus, Focus::Experts, "focus wraps");
    Ok(())
}

// List selection wraps both ways and is safe on an empty list.
#[test]
fn selection_wraps_and_empty_is_safe() -> anyhow::Result<()> {
    let mut app = demo_app();
    app.shadows = vec![graduated_shadow(), graduated_shadow()];
    app.focus = Focus::Shadows;
    app.selected_shadow = 0;
    app.select_prev();
    assert_eq!(app.selected_shadow, 1, "prev wraps to the last");
    app.select_next();
    assert_eq!(app.selected_shadow, 0, "next wraps to the first");
    app.shadows.clear();
    app.select_next();
    app.select_prev();
    assert_eq!(
        app.selected_shadow, 0,
        "no panic / no move on an empty list"
    );
    Ok(())
}

// Every overlay opener sets its mode, and close/toggle return to Normal.
#[test]
fn overlays_open_to_their_modes_and_close() -> anyhow::Result<()> {
    let mut app = demo_app();
    app.open_palette();
    assert_eq!(app.mode, Mode::Palette);
    app.close_overlay();
    assert_eq!(app.mode, Mode::Normal);
    app.open_filter();
    assert_eq!(app.mode, Mode::Filter);
    app.open_events();
    assert_eq!(app.mode, Mode::Events);
    app.open_detail();
    assert_eq!(app.mode, Mode::Detail);
    app.open_ask();
    assert_eq!(app.mode, Mode::Ask);
    app.close_overlay();
    app.toggle_help();
    assert_eq!(app.mode, Mode::Help);
    app.toggle_help();
    assert_eq!(app.mode, Mode::Normal, "help toggles closed");
    Ok(())
}

// The event stream caps at MAX_EVENTS and keeps the newest first.
#[test]
fn event_stream_is_capped_and_newest_first() -> anyhow::Result<()> {
    let mut app = demo_app();
    let total = crate::events::MAX_EVENTS + 50;
    for i in 0..total {
        app.operator_event(format!("ev{i}"));
    }
    assert_eq!(app.events.len(), crate::events::MAX_EVENTS, "capped");
    assert_eq!(
        app.events[0].text,
        format!("ev{}", total - 1),
        "newest first"
    );
    Ok(())
}

// Typing in the ask invalidates the previous routing result.
#[test]
fn ask_input_invalidates_the_stale_result() -> anyhow::Result<()> {
    let mut app = demo_app();
    app.open_ask();
    app.set_ask_result(&crate::backend::Routed {
        routes: vec![(ExpertId::new("expert:arith"), 1.0)],
        reason: Some("a reason of some other task's".into()),
    });
    assert!(app.ask_result.is_some());
    app.ask_input('x');
    assert!(app.ask_reason.is_none(), "and the reason that went with it");
    assert!(
        app.ask_result.is_none(),
        "a new keystroke clears the result"
    );
    Ok(())
}

// Palette and filter selections wrap within their match lists.
#[test]
fn palette_and_filter_move_wrap() -> anyhow::Result<()> {
    let mut app = demo_app();
    app.open_palette();
    let n = app.palette_matches().len();
    assert!(n > 1);
    app.palette.selected = 0;
    app.palette_move(-1);
    assert_eq!(app.palette.selected, n - 1, "palette wraps up");
    app.palette_move(1);
    assert_eq!(app.palette.selected, 0, "palette wraps down");

    app.boundaries = vec![actionable_boundary(), actionable_boundary()];
    app.focus = Focus::Boundaries;
    app.open_filter();
    let m = app.filtered().len();
    assert_eq!(m, 2);
    app.filter_selected = 0;
    app.filter_move(-1);
    assert_eq!(app.filter_selected, m - 1, "filter wraps up");
    Ok(())
}

// Frame-rate and scroll controls clamp/pin correctly.
#[test]
fn fps_and_scroll_controls_clamp() -> anyhow::Result<()> {
    let mut app = demo_app();
    app.auto_fps = true;
    app.target_fps = 144;
    app.fps_up();
    assert!(!app.auto_fps, "fps_up pins manual");
    assert_eq!(app.target_fps, 165);
    app.fps_down();
    assert_eq!(app.target_fps, 144);
    app.pin_fps(99_999);
    assert_eq!(
        app.target_fps,
        crate::pacing::MAX_FPS,
        "pin clamps to the max"
    );

    app.detail_scroll = 0;
    app.detail_move(-5);
    assert_eq!(app.detail_scroll, 0, "detail scroll clamps at the top");
    app.detail_move(3);
    assert_eq!(app.detail_scroll, 3);

    app.events.clear();
    app.events_move(1);
    assert_eq!(app.events_scroll, 0, "empty event scroll stays at 0");
    app.operator_event("a".into());
    app.operator_event("b".into());
    app.events_scroll = 0;
    app.events_move(5);
    assert_eq!(
        app.events_scroll, 1,
        "event scroll clamps to the last index"
    );
    Ok(())
}

// The monitor-follow methods are safe to call (FFI on Windows, no-op elsewhere).
#[test]
fn monitor_methods_are_safe_to_call() -> anyhow::Result<()> {
    let mut app = demo_app();
    app.capture_window();
    app.follow_monitor();
    app.poll_monitor();
    assert!(app.auto_fps, "follow_monitor re-enables auto");
    Ok(())
}

// An out-of-distribution ask (empty routing) renders the escalate hint.
#[test]
fn escalating_ask_renders_the_hint() -> anyhow::Result<()> {
    let mut app = demo_app();
    app.open_ask();
    app.ask_result = Some(Vec::new());
    let text = to_text(&render(&mut app, 80, 12, 1600.0)?);
    assert!(
        text.contains("No expert is routed to it: it goes to your agent."),
        "it says where the task goes:\n{text}"
    );
    Ok(())
}

// Pressing `?` opens the help overlay over the live view, listing the keys.
#[test]
fn help_overlay_lists_the_keybindings() -> anyhow::Result<()> {
    let mut app = demo_app();
    app.mode = Mode::Help;
    let text = to_text(&render(&mut app, 120, 36, 1600.0)?);
    assert!(text.contains("help"), "help modal title shown:\n{text}");
    assert!(text.contains("change the theme"), "theme key documented");
    assert!(
        text.contains("0 follows the monitor"),
        "the key that follows the monitor is documented"
    );
    assert!(
        text.contains("move to the next list"),
        "the key that changes list is documented"
    );
    Ok(())
}

// Tab into the penumbra focus: the training view shows the selected shadow's
// status, generation, and reward curve.
#[test]
fn shadows_focus_shows_the_penumbra() -> anyhow::Result<()> {
    let mut app = demo_app();
    app.shadows = vec![graduated_shadow()];
    app.focus = Focus::Shadows;
    let text = to_text(&render(&mut app, 120, 36, 1600.0)?);
    assert!(
        text.contains("Candidates 1"),
        "candidates list title missing:\n{text}"
    );
    assert!(
        text.contains("╭ Candidate ─"),
        "candidate detail panel missing"
    );
    assert!(text.contains("shadow:g3"), "the shadow id is shown");
    assert!(text.contains("graduated"), "the shadow status is shown");
    assert!(text.contains("0.93"), "the final reward is shown");
    Ok(())
}

// The connect panel shows the live FPS readout and the cap, and the cap steps
// through common refresh rates (144 -> 165). What a key changed the cap to is
// said on the line above the footer, which the key-press tests read.
#[test]
fn the_connect_panel_shows_the_frame_rate_and_the_cap_steps() -> anyhow::Result<()> {
    let mut app = demo_app();
    app.open_connect();
    let text = to_text(&render(&mut app, 140, 36, 1600.0)?);
    assert!(
        text.contains("drawn at 144 fps (cap 144, pinned)"),
        "fps readout shown:\n{text}"
    );
    app.fps_up();
    assert_eq!(app.target_fps, 165, "cap stepped to 165");
    Ok(())
}

// Following the active monitor holds until a manual `+`/`-` pins the cap.
// (The flag is set directly to keep the test off real monitor detection.)
#[test]
fn a_manual_adjust_pins_the_cap() -> anyhow::Result<()> {
    let mut app = demo_app();
    app.auto_fps = true;
    app.fps_up();
    assert!(!app.auto_fps, "a manual adjust pins the cap");
    Ok(())
}

// A standing expert's training reads by the scope it trained, its pass rate
// from the base model's to the trained expert's, and not by its long id.
#[test]
fn a_standing_experts_training_reads_by_its_scope() -> anyhow::Result<()> {
    let mut app = demo_app();
    app.shadows = vec![kept_shadow()];
    app.focus = Focus::Shadows;
    let text = to_text(&render(&mut app, 120, 36, 1600.0)?);
    assert!(
        text.contains("▸ code") && text.contains("0.42 → 0.91 ● graduated"),
        "the list row names the scope and the change:\n{text}"
    );
    assert!(
        text.contains("a training run of its standing expert"),
        "titled as one:\n{text}"
    );
    assert!(
        text.contains("base model"),
        "the base model's rate is labeled"
    );
    assert!(text.contains("trained"), "the trained rate is labeled");
    assert!(
        text.contains("This expert has been trained once."),
        "two readings of one run are not drawn as a path:\n{text}"
    );
    assert!(
        !text.contains("shadow:ws:"),
        "the sixty-character id is not what the panel shows:\n{text}"
    );
    Ok(())
}

// A repository's scope is its path, which fills a narrow list's row: the
// lists name it by the repository, so a training keeps its reading and the
// behaviors one expert missed read apart, where each row had shown the same
// path and nothing after it.
#[test]
fn a_repository_scope_leaves_room_in_the_lists() -> anyhow::Result<()> {
    let scope = "github.com/acme/orders-service-backend";
    let mut app = demo_app();
    let mut s = kept_shadow();
    s.id = antumbra_core::behavior::shadow_id(
        &antumbra_core::TenantId::new("ws:default"),
        &antumbra_core::UserId::new("user:default"),
        scope,
        "5c0ffee1d2e3f4a5",
    );
    s.reward_curve = vec![0.68];
    app.shadows = vec![s];
    app.focus = Focus::Shadows;
    let text = to_text(&render(&mut app, 120, 36, 1600.0)?);
    assert!(
        text.lines()
            .any(|row| row.contains("▸ orders-service-backend") && row.contains("0.68")),
        "the repository's name and the reading fit:\n{text}"
    );

    let rules = [
        "Before handing back a branch, run the contract tests.",
        "Never log an order's payload, only its id.",
    ];
    app.boundaries = rules
        .iter()
        .map(|rule| {
            let mut b = missed_boundary();
            b.behavior = (*rule).into();
            b.fail_context["scope"] = scope.into();
            b
        })
        .collect();
    app.focus = Focus::Boundaries;
    let text = to_text(&render(&mut app, 120, 36, 1600.0)?);
    assert!(
        text.contains("▸ Before handing back a branch"),
        "one behavior reads:\n{text}"
    );
    assert!(
        text.contains("  Never log an order's"),
        "and the other, apart from it:\n{text}"
    );
    Ok(())
}

// An open boundary from a training shows what it was recorded from: the
// scope, the task the expert missed and its answer, and whose it is.
#[test]
fn an_open_boundary_shows_the_task_it_missed() -> anyhow::Result<()> {
    let mut app = demo_app();
    app.boundaries = vec![missed_boundary()];
    app.focus = Focus::Boundaries;
    let text = to_text(&render(&mut app, 120, 36, 1600.0)?);
    assert!(
        text.contains("▸ Run the test suite"),
        "the list row leads with the behavior:\n{text}"
    );
    assert!(
        text.contains("○ recorded · not yet in force"),
        "recorded, so not in force"
    );
    assert!(text.contains("private to"), "the owner is shown");
    assert!(text.contains("user:default"), "by name");
    assert!(
        text.contains("Rename the config"),
        "the missed task is shown"
    );
    assert!(text.contains("answer"), "and the answer given");
    assert!(
        !text.contains("decided by -"),
        "no empty feature row for a boundary that has none:\n{text}"
    );
    Ok(())
}

// The gate summary lists only boundaries that gate routing: an open one,
// listed there, would read as a gate.
#[test]
fn the_gate_summary_lists_no_open_boundary() -> anyhow::Result<()> {
    let mut app = demo_app();
    app.boundaries = vec![missed_boundary()];
    let text = to_text(&render(&mut app, 120, 36, 1600.0)?);
    assert!(
        text.contains("1 · 0 in force"),
        "counted, none in force:\n{text}"
    );
    assert!(
        !text.contains("Run the test suite"),
        "the open one is not listed:\n{text}"
    );
    Ok(())
}

// The console's prune and graduate steer the generational loop; on a standing
// expert's training, which its keeper settles, they decline and say why.
#[test]
fn a_standing_experts_training_is_left_to_its_keeper() -> anyhow::Result<()> {
    let mut app = demo_app();
    app.shadows = vec![kept_shadow()];
    app.focus = Focus::Shadows;
    app.request_action();
    assert!(app.pending.is_none(), "no prune staged");
    assert_eq!(app.mode, Mode::Normal);
    app.request_graduate();
    assert!(app.pending.is_none(), "no graduation staged");
    assert!(
        app.events
            .iter()
            .any(|e| e.text.contains("settled by the server")),
        "the reason is on the event line"
    );
    // The loop's own shadows still take both.
    app.shadows = vec![graduated_shadow()];
    app.shadows[0].status = ShadowStatus::Exploring;
    app.request_action();
    assert!(app.pending.is_some(), "a loop shadow is still pruned");
    Ok(())
}

// Tab into the boundaries focus: the inspector replaces the expert detail and
// shows the selected scope's behavior, actionable status, and governing feature.
#[test]
fn boundaries_focus_shows_the_inspector() -> anyhow::Result<()> {
    let mut app = demo_app();
    app.boundaries = vec![actionable_boundary()];
    app.focus = Focus::Boundaries;
    let text = to_text(&render(&mut app, 120, 36, 1600.0)?);
    assert!(
        text.contains("Boundaries 1"),
        "boundaries list title missing:\n{text}"
    );
    assert!(
        text.contains("╭ Boundary ─"),
        "boundary detail panel missing"
    );
    assert!(
        text.contains("npm install"),
        "the boundary's behavior is shown"
    );
    assert!(
        text.contains("● in force · routing avoids it"),
        "that it is in force is shown"
    );
    assert!(text.contains("runtime"), "the governing feature is shown");
    Ok(())
}
