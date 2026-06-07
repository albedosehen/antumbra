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
    use antumbra_core::{
        BoundaryId, Expert, ExpertId, FailureBoundary, Generation, Grain, Shadow, ShadowId,
        ShadowStatus,
    };
    use chrono::Utc;

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
            router: None,
            selected: 0,
            selected_boundary: 0,
            selected_shadow: 0,
            focus: Focus::Experts,
            layout: LayoutMode::Focused,
            mode: Mode::Normal,
            palette: Palette::default(),
            filter: String::new(),
            filter_selected: 0,
            events: Vec::new(),
            events_scroll: 0,
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
