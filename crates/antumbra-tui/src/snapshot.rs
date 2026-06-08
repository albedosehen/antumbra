//! Headless rendering of the operator console: draw a frame to an in-memory
//! buffer (no terminal) so the console can be e2e-tested and screenshotted.
//!
//! - [`to_text`] preserves every cell's glyph (braille graph, box-drawing,
//!   symbols), so it reads back exactly — for assertions and plain inspection.
//! - [`save_png`] adds the RGB colours for true visual fidelity, rasterizing the
//!   embedded Cascadia Mono (OFL; see `assets/CascadiaMono.LICENSE`).

use ab_glyph::{point, Font, FontRef, PxScale, ScaleFont};
use anyhow::{anyhow, Result};
use image::{Rgb, RgbImage};
use ratatui::backend::TestBackend;
use ratatui::buffer::Buffer;
#[cfg(test)]
use ratatui::layout::Rect;
use ratatui::style::Color;
use ratatui::Terminal;

use crate::app::App;
use crate::ui;

/// Cascadia Mono (OFL) — rasterized for the PNG snapshot so braille / box-drawing
/// / symbol glyphs render true.
const FONT: &[u8] = include_bytes!("../assets/CascadiaMono.ttf");

/// The dark-terminal background and default ink the console is designed against.
const DEFAULT_BG: [u8; 3] = [12, 14, 18];
const DEFAULT_FG: [u8; 3] = [180, 190, 200];

/// Render one frame of the console to an in-memory buffer at a fixed animation
/// clock, so the output is deterministic.
pub fn render(app: &mut App, width: u16, height: u16, at_ms: f64) -> Result<Buffer> {
    app.clock_ms = at_ms;
    let mut terminal = Terminal::new(TestBackend::new(width, height))?;
    terminal.draw(|f| ui::render(f, app))?;
    Ok(terminal.backend().buffer().clone())
}

/// The buffer as a Unicode text grid (rows joined by newlines, trailing spaces
/// trimmed). Every cell's symbol is preserved.
pub fn to_text(buf: &Buffer) -> String {
    let area = buf.area;
    let mut out = String::with_capacity((area.width as usize + 1) * area.height as usize);
    for y in 0..area.height {
        let mut row = String::with_capacity(area.width as usize);
        for x in 0..area.width {
            if let Some(cell) = buf.cell((x, y)) {
                row.push_str(cell.symbol());
            }
        }
        out.push_str(row.trim_end());
        out.push('\n');
    }
    out
}

/// The text of a sub-region of the buffer (rows trimmed) — for golden-testing a
/// modal without the animated background behind it.
#[cfg(test)]
pub fn to_text_in(buf: &Buffer, area: Rect) -> String {
    let mut out = String::new();
    for y in area.top()..area.bottom() {
        let mut row = String::new();
        for x in area.left()..area.right() {
            if let Some(cell) = buf.cell((x, y)) {
                row.push_str(cell.symbol());
            }
        }
        out.push_str(row.trim_end());
        out.push('\n');
    }
    out
}

/// Render the buffer to a PNG at `cell_w` x `cell_h` pixels per cell and save it.
pub fn save_png(buf: &Buffer, path: &str, cell_w: u32, cell_h: u32) -> Result<()> {
    let area = buf.area;
    let mut img = RgbImage::from_pixel(
        area.width as u32 * cell_w,
        area.height as u32 * cell_h,
        Rgb(DEFAULT_BG),
    );
    let font = FontRef::try_from_slice(FONT).map_err(|e| anyhow!("load font: {e}"))?;
    let scale = PxScale::from(cell_h as f32);
    let ascent = font.as_scaled(scale).ascent();

    for y in 0..area.height {
        for x in 0..area.width {
            let Some(cell) = buf.cell((x, y)) else {
                continue;
            };
            let bg = color_rgb(cell.bg, DEFAULT_BG);
            let fg = color_rgb(cell.fg, DEFAULT_FG);
            let (px0, py0) = (x as u32 * cell_w, y as u32 * cell_h);

            // Cell background.
            for yy in 0..cell_h {
                for xx in 0..cell_w {
                    img.put_pixel(px0 + xx, py0 + yy, Rgb(bg));
                }
            }

            // Glyph, blended over the background by per-pixel coverage.
            let ch = cell.symbol().chars().next().unwrap_or(' ');
            if ch == ' ' {
                continue;
            }
            let glyph = font
                .glyph_id(ch)
                .with_scale_and_position(scale, point(px0 as f32, py0 as f32 + ascent));
            if let Some(outline) = font.outline_glyph(glyph) {
                let bounds = outline.px_bounds();
                outline.draw(|gx, gy, c| {
                    let ix = bounds.min.x as i32 + gx as i32;
                    let iy = bounds.min.y as i32 + gy as i32;
                    if ix < 0 || iy < 0 || ix as u32 >= img.width() || iy as u32 >= img.height() {
                        return;
                    }
                    let px = img.get_pixel(ix as u32, iy as u32).0;
                    let blended = [
                        blend(px[0], fg[0], c),
                        blend(px[1], fg[1], c),
                        blend(px[2], fg[2], c),
                    ];
                    img.put_pixel(ix as u32, iy as u32, Rgb(blended));
                });
            }
        }
    }
    img.save(path)
        .map_err(|e| anyhow!("save png {path}: {e}"))?;
    Ok(())
}

/// `over` blended onto `under` by coverage `c` in `[0,1]`.
fn blend(under: u8, over: u8, c: f32) -> u8 {
    (under as f32 * (1.0 - c) + over as f32 * c)
        .round()
        .clamp(0.0, 255.0) as u8
}

/// Map a ratatui colour to RGB, falling back to `default` for `Reset`/unknown.
fn color_rgb(color: Color, default: [u8; 3]) -> [u8; 3] {
    match color {
        Color::Rgb(r, g, b) => [r, g, b],
        Color::Black => [0, 0, 0],
        Color::White => [235, 235, 235],
        Color::Red => [200, 70, 70],
        Color::Green => [70, 190, 110],
        Color::Yellow => [210, 200, 90],
        Color::Blue => [80, 130, 230],
        Color::Magenta => [200, 110, 220],
        Color::Cyan => [90, 200, 230],
        Color::Gray => [150, 160, 170],
        Color::DarkGray => [80, 90, 100],
        _ => default,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::{Focus, LayoutMode, Mode, Palette};
    use crate::command::Action;
    use antumbra_core::generational::{GenerationHead, LoopState};
    use antumbra_core::{
        BoundaryId, EdgeType, EvalStatus, EvaluationRun, Expert, ExpertId, FailureBoundary,
        Generation, Grain, Memory, MemoryEdge, MemoryNetwork, RunId, Shadow, ShadowId,
        ShadowStatus, SubjectKind,
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
            created_at: now,
        };
        App {
            experts: vec![expert],
            boundaries: Vec::new(),
            shadows: Vec::new(),
            memories: demo_memories(now),
            edges: demo_edges(now),
            selected_memory: 0,
            loop_heads: demo_loop_heads(now),
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

    // Golden the keybinding help modal exactly — adding/renaming a binding must be
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
        let strip =
            Layout::vertical([Constraint::Min(0), Constraint::Length(9)]).split(chrome[3])[1];
        assert_golden("reward_heatmap", &to_text_in(&buf, strip));
    }

    // Golden the Loop page: the generational pipeline with the current stage lit
    // (deterministic — no animated graph, no wall-clock fields shown).
    #[test]
    fn golden_loop_page() {
        let mut app = demo_app();
        app.set_page(crate::app::Page::Loop);
        let buf = render(&mut app, 120, 36, 1600.0).unwrap();
        assert_golden("loop_page", &to_text(&buf));
    }

    // Golden the Evals page: the evaluation-run table with a regression failure
    // (deterministic — fixed runs, no timestamps shown).
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

    /// Golden the focused-layout right detail column (the panels — no animated
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

    // e2e of the console without a terminal: render a frame and assert on the
    // text grid. The whole layout is exercised (header, graph, detail, gate).
    #[test]
    fn headless_render_shows_the_console() {
        let mut app = demo_app();
        let buf = render(&mut app, 120, 36, 1600.0).unwrap();
        let text = to_text(&buf);
        assert!(
            text.contains("ANTUMBRA"),
            "header wordmark missing:\n{text}"
        );
        assert!(text.contains("population"), "graph panel title missing");
        assert!(
            text.contains("arith-specialist"),
            "expert name not rendered"
        );
        assert!(text.contains("gate"), "gate panel title missing");
        assert!(text.contains("0.90"), "selected expert's fitness rendered");
    }

    #[test]
    fn empty_population_renders_the_hint() {
        let mut app = demo_app();
        app.experts.clear();
        let text = to_text(&render(&mut app, 100, 30, 1600.0).unwrap());
        assert!(text.contains("no experts yet"), "empty-state hint shown");
    }

    // The loop runs full-rate while interacting or animating, eases to 60 after
    // an idle spell, and the header marks the eased state.
    #[test]
    fn idle_throttle_eases_the_rate_and_shows_in_header() {
        let mut app = demo_app();
        app.target_fps = 244;
        app.power_save = true;
        // Fresh input: full rate.
        app.note_input();
        assert_eq!(app.frame_cap(false), 244);
        // A still spell eases to 60 — unless something is animating.
        app.tick(10_000.0);
        assert_eq!(app.frame_cap(true), 244, "animation keeps full rate");
        assert_eq!(app.frame_cap(false), 60, "an idle, still view eases to 60");
        // With power-save off (the default), the still view stays at full rate.
        app.power_save = false;
        assert_eq!(app.frame_cap(false), 244, "default never throttles");
        // The header reflects the eased state.
        app.idle = true;
        let text = to_text(&render(&mut app, 130, 12, 1600.0).unwrap());
        assert!(
            text.contains("idle"),
            "header marks the idle throttle:\n{text}"
        );
    }

    // Home/End/PageUp/PageDown drive the focused list, clamped to the ends.
    #[test]
    fn page_and_jump_navigation_clamps_to_the_list() {
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
    }

    // The dashboard layout shows all three regions at once (umbra, penumbra,
    // antumbra), unlike the focused layout that swaps a single detail panel.
    #[test]
    fn dashboard_layout_shows_every_region_at_once() {
        let mut app = demo_app();
        app.shadows = vec![graduated_shadow()];
        app.boundaries = vec![actionable_boundary()];
        app.layout = LayoutMode::Dashboard;
        let text = to_text(&render(&mut app, 130, 40, 1600.0).unwrap());
        assert!(
            text.contains("umbra · experts"),
            "umbra panel shown:\n{text}"
        );
        assert!(text.contains("penumbra · shadows"), "penumbra panel shown");
        assert!(
            text.contains("antumbra · boundaries"),
            "antumbra panel shown"
        );
        // All three regions' content is visible together.
        assert!(text.contains("arith-specialist"), "expert content shown");
        assert!(text.contains("shadow:g3"), "shadow content shown");
        assert!(text.contains("npm install"), "boundary content shown");
        // The header names the active layout.
        assert!(text.contains("dashboard"), "header names the layout");
    }

    // A list longer than its panel scrolls to keep the selection visible: a high
    // selection shows the tail and hides the head.
    #[test]
    fn long_lists_scroll_to_the_selection() {
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
        let text = to_text(&render(&mut app, 120, 18, 1600.0).unwrap());
        assert!(
            text.contains("shadow:s19"),
            "the selected tail is visible:\n{text}"
        );
        assert!(
            !text.contains("shadow:s0\n") && !text.contains("shadow:s0 "),
            "the head has scrolled out of view:\n{text}"
        );
    }

    // The graph layout drops the detail column for a full-width population view.
    #[test]
    fn graph_layout_is_full_width() {
        let mut app = demo_app();
        app.layout = LayoutMode::Graph;
        let text = to_text(&render(&mut app, 120, 36, 1600.0).unwrap());
        assert!(text.contains("population"), "graph panel still shown");
        // The expert detail column is gone in graph mode.
        assert!(
            !text.contains("learned exemplars"),
            "no detail column:\n{text}"
        );
    }

    // The command palette filters its commands by the typed query and marks the
    // selection, so a fuzzy query surfaces the matching action.
    #[test]
    fn command_palette_filters_to_the_query() {
        let mut app = demo_app();
        app.open_palette();
        for c in "theme".chars() {
            app.palette_input(c);
        }
        let text = to_text(&render(&mut app, 120, 36, 1600.0).unwrap());
        assert!(text.contains("command"), "palette title shown:\n{text}");
        assert!(
            text.contains("› theme"),
            "the typed query is echoed:\n{text}"
        );
        assert!(
            text.contains("cycle palette"),
            "the theme command surfaced for the query:\n{text}"
        );
        // The chosen action matches the filtered selection.
        assert_eq!(app.palette_action(), Some(Action::CycleTheme));
    }

    // The `/` filter fuzzy-narrows the focused list and jumps the selection to
    // the chosen match.
    #[test]
    fn filter_narrows_and_jumps_the_selection() {
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
        let text = to_text(&render(&mut app, 120, 36, 1600.0).unwrap());
        assert!(
            text.contains("filter penumbra"),
            "filter modal titled:\n{text}"
        );
        assert!(text.contains("shadow:g7"), "the match is listed");
        assert_eq!(app.filtered().len(), 1, "g7 is the only match");
        // Applying jumps the focused selection to g7 (index 1) and closes.
        app.filter_apply();
        assert_eq!(app.selected_shadow, 1, "selection jumped to the match");
        assert_eq!(app.mode, Mode::Normal, "filter closed after applying");
    }

    // The event stream diffs successive reloads: a baseline pass records state
    // silently, then changes (spawn, graduate, boundary) emit events.
    #[test]
    fn event_stream_diffs_store_changes() {
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
        let text = to_text(&render(&mut app, 100, 24, 1600.0).unwrap());
        assert!(text.contains("events"), "events overlay titled:\n{text}");
        assert!(text.contains("graduated"), "an event is listed");
    }

    // Enter drills into the selected boundary: a full-detail modal with the
    // contrastive contexts the summary panel omits.
    #[test]
    fn drill_down_shows_full_boundary_detail() {
        let mut app = demo_app();
        app.boundaries = vec![actionable_boundary()];
        app.focus = Focus::Boundaries;
        app.open_detail();
        let text = to_text(&render(&mut app, 110, 36, 1600.0).unwrap());
        assert!(text.contains("boundary ·"), "detail modal titled:\n{text}");
        assert!(text.contains("fail context"), "the C context section shown");
        assert!(
            text.contains("acceptable context"),
            "the C' context section shown"
        );
        // The pretty-printed JSON of the contexts is present.
        assert!(text.contains("runtime"), "context json rendered");
    }

    // The expert drill-down previews how the gate routes that expert's own
    // specialty (probing the router with its capability vector — no embedder).
    #[test]
    fn drill_down_previews_gate_routing() {
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
        let text = to_text(&render(&mut app, 110, 36, 1600.0).unwrap());
        assert!(
            text.contains("gate routing"),
            "route preview shown:\n{text}"
        );
        assert!(
            text.contains("100%"),
            "the sole expert wins its own specialty"
        );
        assert!(text.contains("arith-specialist"), "routed expert named");
    }

    // The route-ask overlay echoes the query and shows the gate's routing
    // distribution, resolving expert ids to names.
    #[test]
    fn ask_overlay_shows_routing_distribution() {
        let mut app = demo_app();
        app.open_ask();
        for c in "arith".chars() {
            app.ask_input(c);
        }
        app.set_ask_result(&[(ExpertId::new("expert:arith"), 0.82)]);
        let text = to_text(&render(&mut app, 80, 14, 1600.0).unwrap());
        assert!(text.contains("ask the gate"), "ask modal titled:\n{text}");
        assert!(text.contains("ask › arith"), "query echoed");
        assert!(
            text.contains("arith-specialist"),
            "routed expert named (id resolved)"
        );
        assert!(text.contains("82%"), "routing probability shown");
    }

    // `x` on a focused boundary stages a delete behind a confirm prompt; on a
    // shadow it stages a prune. Cancelling clears it.
    #[test]
    fn operator_action_stages_a_confirm_prompt() {
        let mut app = demo_app();
        app.boundaries = vec![actionable_boundary()];
        app.focus = Focus::Boundaries;
        app.request_action();
        assert_eq!(app.mode, Mode::Confirm, "confirm prompt opened");
        let text = to_text(&render(&mut app, 90, 24, 1600.0).unwrap());
        assert!(text.contains("confirm"), "confirm modal titled:\n{text}");
        assert!(text.contains("Delete boundary"), "the delete prompt shown");
        assert!(text.contains("npm install"), "names the boundary");
        app.cancel_action();
        assert_eq!(app.mode, Mode::Normal, "cancel returns to the live view");
        assert!(app.pending.is_none(), "the staged action is cleared");

        // On a shadow, the action prunes.
        app.shadows = vec![graduated_shadow()];
        app.focus = Focus::Shadows;
        app.request_action();
        assert!(
            app.pending_prompt().unwrap().contains("Prune"),
            "shadow action is a prune"
        );
    }

    // Theme, layout, and focus each cycle through their states and wrap.
    #[test]
    fn theme_layout_focus_cycle_and_wrap() {
        let mut app = demo_app();
        assert_eq!(app.theme().name, "shadow");
        app.cycle_theme();
        assert_eq!(app.theme().name, "ember");
        app.cycle_theme();
        assert_eq!(app.theme().name, "mono");
        app.cycle_theme();
        assert_eq!(app.theme().name, "shadow", "theme wraps");

        assert_eq!(app.layout, LayoutMode::Focused);
        app.cycle_layout();
        assert_eq!(app.layout, LayoutMode::Dashboard);
        app.cycle_layout();
        assert_eq!(app.layout, LayoutMode::Graph);
        app.cycle_layout();
        assert_eq!(app.layout, LayoutMode::Table);
        app.cycle_layout();
        assert_eq!(app.layout, LayoutMode::Focused, "layout wraps");

        assert_eq!(app.focus, Focus::Experts);
        app.toggle_focus();
        assert_eq!(app.focus, Focus::Shadows);
        app.toggle_focus();
        assert_eq!(app.focus, Focus::Boundaries);
        app.toggle_focus();
        assert_eq!(app.focus, Focus::Experts, "focus wraps");
    }

    // List selection wraps both ways and is safe on an empty list.
    #[test]
    fn selection_wraps_and_empty_is_safe() {
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
    }

    // Every overlay opener sets its mode, and close/toggle return to Normal.
    #[test]
    fn overlays_open_to_their_modes_and_close() {
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
    }

    // The event stream caps at MAX_EVENTS and keeps the newest first.
    #[test]
    fn event_stream_is_capped_and_newest_first() {
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
    }

    // Typing in the ask invalidates the previous routing result.
    #[test]
    fn ask_input_invalidates_the_stale_result() {
        let mut app = demo_app();
        app.open_ask();
        app.set_ask_result(&[(ExpertId::new("expert:arith"), 1.0)]);
        assert!(app.ask_result.is_some());
        app.ask_input('x');
        assert!(
            app.ask_result.is_none(),
            "a new keystroke clears the result"
        );
    }

    // Palette and filter selections wrap within their match lists.
    #[test]
    fn palette_and_filter_move_wrap() {
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
    }

    // Frame-rate and scroll controls clamp/pin correctly.
    #[test]
    fn fps_and_scroll_controls_clamp() {
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
    }

    // The monitor-follow methods are safe to call (FFI on Windows, no-op elsewhere).
    #[test]
    fn monitor_methods_are_safe_to_call() {
        let mut app = demo_app();
        app.capture_window();
        app.follow_monitor();
        app.poll_monitor();
        assert!(app.auto_fps, "follow_monitor re-enables auto");
    }

    // An out-of-distribution ask (empty routing) renders the escalate hint.
    #[test]
    fn escalating_ask_renders_the_hint() {
        let mut app = demo_app();
        app.open_ask();
        app.ask_result = Some(Vec::new());
        let text = to_text(&render(&mut app, 80, 12, 1600.0).unwrap());
        assert!(text.contains("escalates"), "escalate hint shown:\n{text}");
    }

    // Pressing `?` opens the help overlay over the live view, listing the keys.
    #[test]
    fn help_overlay_lists_the_keybindings() {
        let mut app = demo_app();
        app.mode = Mode::Help;
        let text = to_text(&render(&mut app, 120, 36, 1600.0).unwrap());
        assert!(text.contains("help"), "help modal title shown:\n{text}");
        assert!(text.contains("cycle theme"), "theme keybind documented");
        assert!(
            text.contains("follow the active monitor"),
            "fps follow keybind documented"
        );
        assert!(text.contains("switch focus"), "focus keybind documented");
    }

    // Tab into the penumbra focus: the training view shows the selected shadow's
    // status, generation, and reward curve.
    #[test]
    fn shadows_focus_shows_the_penumbra() {
        let mut app = demo_app();
        app.shadows = vec![graduated_shadow()];
        app.focus = Focus::Shadows;
        let text = to_text(&render(&mut app, 120, 36, 1600.0).unwrap());
        assert!(
            text.contains("shadows"),
            "shadows list title missing:\n{text}"
        );
        assert!(text.contains("training"), "training detail panel missing");
        assert!(text.contains("shadow:g3"), "the shadow id is shown");
        assert!(text.contains("graduated"), "the shadow status is shown");
        assert!(text.contains("0.93"), "the final reward is shown");
    }

    // The footer shows the active theme, and `t` cycles to the next palette,
    // which the footer reflects (shadow -> ember).
    #[test]
    fn cycling_theme_updates_the_footer() {
        let mut app = demo_app();
        let text = to_text(&render(&mut app, 120, 36, 1600.0).unwrap());
        assert!(
            text.contains("theme:shadow"),
            "default theme shown:\n{text}"
        );
        app.cycle_theme();
        let text = to_text(&render(&mut app, 120, 36, 1600.0).unwrap());
        assert!(text.contains("theme:ember"), "cycled theme shown:\n{text}");
    }

    // The header shows a live FPS readout and the footer the adjustable cap; the
    // cap steps through common refresh rates (144 -> 165).
    #[test]
    fn header_and_footer_show_the_frame_rate() {
        let mut app = demo_app();
        let text = to_text(&render(&mut app, 140, 36, 1600.0).unwrap());
        assert!(
            text.contains("144 fps"),
            "header fps readout shown:\n{text}"
        );
        assert!(text.contains("cap:144"), "footer fps cap shown:\n{text}");
        app.fps_up();
        let text = to_text(&render(&mut app, 140, 36, 1600.0).unwrap());
        assert!(text.contains("cap:165"), "cap stepped to 165:\n{text}");
    }

    // Following the active monitor tags the cap "auto"; a manual `+`/`-` pins it
    // and drops the tag. (The flag is set directly to keep the test off real
    // monitor detection.)
    #[test]
    fn auto_follow_tags_the_cap_until_pinned() {
        let mut app = demo_app();
        app.auto_fps = true;
        let text = to_text(&render(&mut app, 140, 36, 1600.0).unwrap());
        assert!(
            text.contains("auto"),
            "auto-follow tagged on the cap:\n{text}"
        );
        app.fps_up();
        assert!(!app.auto_fps, "a manual adjust pins the cap");
        let text = to_text(&render(&mut app, 140, 36, 1600.0).unwrap());
        assert!(
            !text.contains("cap:165 auto"),
            "a pinned cap shows no auto tag:\n{text}"
        );
    }

    // Tab into the boundaries focus: the inspector replaces the expert detail and
    // shows the selected scope's behavior, actionable status, and governing feature.
    #[test]
    fn boundaries_focus_shows_the_inspector() {
        let mut app = demo_app();
        app.boundaries = vec![actionable_boundary()];
        app.focus = Focus::Boundaries;
        let text = to_text(&render(&mut app, 120, 36, 1600.0).unwrap());
        assert!(
            text.contains("boundaries"),
            "boundaries list title missing:\n{text}"
        );
        assert!(text.contains("scope"), "scope detail panel missing");
        assert!(
            text.contains("npm install"),
            "the boundary's behavior is shown"
        );
        assert!(text.contains("actionable"), "actionable status is shown");
        assert!(text.contains("runtime"), "the governing feature is shown");
    }
}
