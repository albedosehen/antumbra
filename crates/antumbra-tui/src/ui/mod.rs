//! Rendering: the living population graph (umbra orbiting the tri-node core of
//! umbra/penumbra/antumbra, three linked minds), with detail and gate panels.
//! Orbit, pulse, and link-energy are hand-rolled from the animation clock so the
//! motion is fully under control. Every color tints from the active [`Theme`],
//! so cycling a theme (`t`) recolors the whole console.
//!
//! Split across submodules: [`graph`] (the canvas), [`chrome`] (header/footer),
//! [`panels`] (the region detail panels), and [`overlays`] (help / palette /
//! filter modals). The small shared widget helpers live here.

mod chrome;
mod detail;
mod evals;
mod graph;
mod heatmap;
mod knobs;
mod loops;
mod memory;
mod overlays;
mod panels;
mod sovereign;
mod table;

use std::f64::consts::TAU;

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType};
use ratatui::Frame;

use antumbra_core::behavior;
use antumbra_core::{FailureBoundary, Shadow};

use crate::app::{App, Focus, LayoutMode, Mode, Page};
use crate::overlay;
use crate::theme::Theme;

pub use overlays::overlay_area;

pub fn render(f: &mut Frame, app: &App) {
    // Guard the render loop against a degenerately-small terminal: the fixed
    // chrome layout would leave no space for the body and some widgets subtract
    // past zero. Rendering nothing is safe; a panic here would crash the console.
    if f.area().width < 8 || f.area().height < 3 {
        return;
    }
    let rows = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Min(0),
        Constraint::Length(1),
    ])
    .split(f.area());
    chrome::header(f, app, rows[0]);
    chrome::tabs(f, app, rows[1]);
    chrome::metrics(f, app, rows[2]);
    page_body(f, app, rows[3]);
    chrome::footer(f, app, rows[4]);
    // A fresh, empty store would otherwise render as inert panels; the first-run
    // hint explains how to populate it. Only in the live view (not over a modal).
    if app.mode == Mode::Normal && app.is_empty() {
        overlays::first_run_hint(f, app);
    }
    if app.mode != Mode::Normal {
        overlay::dim_backdrop(f, f.area());
        match app.mode {
            Mode::Help => overlays::help_overlay(f, app),
            Mode::Palette => overlays::palette_overlay(f, app),
            Mode::Filter => overlays::filter_overlay(f, app),
            Mode::Events => overlays::events_overlay(f, app),
            Mode::Detail => overlays::detail_overlay(f, app),
            Mode::Gate => overlays::gate_overlay(f, app),
            Mode::Connect => overlays::connect_overlay(f, app),
            Mode::Ask => overlays::ask_overlay(f, app),
            Mode::Confirm => overlays::confirm_overlay(f, app),
            Mode::Knobs => knobs::knobs_overlay(f, app),
            Mode::Normal => {}
        }
    }
}

/// Route the body to the active page. `Population` is the live view; the other
/// pages surface store subsystems and are placeholders until built out.
fn page_body(f: &mut Frame, app: &App, area: Rect) {
    match app.page {
        Page::Population => body(f, app, area),
        Page::Memory => memory::page(f, app, area),
        Page::Loop => loops::page(f, app, area),
        Page::Evals => evals::page(f, app, area),
        Page::Sovereign => sovereign::page(f, app, area),
    }
}

/// The body between header and footer, arranged per the active layout: graph
/// beside one focused detail, graph beside all three regions, or graph alone.
fn body(f: &mut Frame, app: &App, area: Rect) {
    match app.layout {
        LayoutMode::Table => table::experts_table(f, app, area),
        LayoutMode::Graph => graph::graph(f, app, area),
        LayoutMode::Focused => {
            let cols = Layout::horizontal([Constraint::Percentage(64), Constraint::Percentage(36)])
                .split(area);
            graph::graph(f, app, cols[0]);
            match app.focus {
                Focus::Experts => panels::detail(f, app, cols[1]),
                Focus::Shadows => panels::shadows(f, app, cols[1]),
                Focus::Boundaries => panels::boundaries(f, app, cols[1]),
            }
        }
        LayoutMode::Dashboard => {
            // Graph + the three regions up top, a full-width reward heatmap below.
            let stack = Layout::vertical([Constraint::Min(0), Constraint::Length(9)]).split(area);
            let cols = Layout::horizontal([Constraint::Percentage(58), Constraint::Percentage(42)])
                .split(stack[0]);
            graph::graph(f, app, cols[0]);
            panels::dashboard(f, app, cols[1]);
            heatmap::reward_heatmap(f, app, stack[1]);
        }
    }
}

/// `[0,1]` pulse from the animation clock.
fn pulse(clock_ms: f64, period_ms: f64, phase: f64) -> f64 {
    0.5 + 0.5 * ((clock_ms / period_ms * TAU) + phase).sin()
}

/// The visible window `[offset, offset + rows)` of a `len`-item list that keeps
/// `selected` on screen, so a `Table` builds only the rows it actually shows even
/// when the data is store-sized. Returns `(offset, in-window index of selected)`.
pub(super) fn visible_window(len: usize, selected: usize, rows: usize) -> (usize, usize) {
    if rows == 0 || len <= rows {
        return (0, selected.min(len.saturating_sub(1)));
    }
    let sel = selected.min(len - 1);
    let offset = sel.saturating_sub(rows / 2).min(len - rows);
    (offset, sel - offset)
}

/// A rounded, dim-bordered panel block with a titled, inked header.
fn panel<'a>(t: &Theme, title: impl Into<Line<'a>>) -> Block<'a> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(t.dim))
        .title(title.into())
}

/// A panel whose border and title brighten to the accent when it holds focus:
/// the dashboard's active-panel indicator.
fn panel_focused<'a>(t: &Theme, title: &str, focused: bool) -> Block<'a> {
    let (border, marker) = if focused {
        (t.accent, "◆ ")
    } else {
        (t.dim, "")
    };
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(border))
        .title(Line::from(format!(" {marker}{title} ")).style(Style::default().fg(border)))
}

/// A bold, accent-colored panel heading (selected item's name).
fn heading<'a>(t: &Theme, text: String) -> Line<'a> {
    Line::from(Span::styled(
        text,
        Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
    ))
}

fn kv<'a>(t: &Theme, k: &str, v: &str) -> Line<'a> {
    Line::from(vec![
        Span::styled(format!("{k:<11}"), Style::default().fg(t.dim)),
        Span::styled(v.to_string(), Style::default().fg(t.value)),
    ])
}

/// A horizontal gauge for a `[0,1]` value: `width` cells filled to eighth-of-a-
/// cell resolution in `color`, the remaining track dim.
fn gauge_spans<'a>(t: &Theme, value: f32, width: u16, color: Color) -> Vec<Span<'a>> {
    const EIGHTHS: [&str; 8] = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉"];
    let v = (value.clamp(0.0, 1.0) as f64) * width as f64;
    let mut full = v.floor() as usize;
    let mut rem = ((v - full as f64) * 8.0).round() as usize;
    if rem == 8 {
        full += 1;
        rem = 0;
    }
    let mut filled = "█".repeat(full);
    let cells = full + usize::from(rem > 0);
    if rem > 0 {
        filled.push_str(EIGHTHS[rem]);
    }
    let track = "─".repeat((width as usize).saturating_sub(cells));
    vec![
        Span::styled(filled, Style::default().fg(color)),
        Span::styled(track, Style::default().fg(t.dim)),
    ]
}

/// A `key  ▆▆▆▍──  0.62` row: a labeled gauge with its numeric value.
fn gauge_row<'a>(t: &Theme, k: &'a str, value: f32, color: Color) -> Line<'a> {
    let mut spans = vec![Span::styled(format!("{k:<11}"), Style::default().fg(t.dim))];
    spans.extend(gauge_spans(t, value, 12, color));
    spans.push(Span::styled(
        format!(" {value:.2}"),
        Style::default().fg(t.value),
    ));
    Line::from(spans)
}

/// A reward curve as block bars, each bar colored along the fitness gradient by
/// its own value, so a rising (or collapsing) trajectory reads at a glance.
fn sparkline_row<'a>(t: &Theme, k: &'a str, curve: &[f32]) -> Line<'a> {
    const BARS: [&str; 8] = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];
    let mut spans = vec![Span::styled(format!("{k:<11}"), Style::default().fg(t.dim))];
    for &v in curve {
        let level = (v.clamp(0.0, 1.0) * 7.0).round() as usize;
        spans.push(Span::styled(
            BARS[level.min(7)].to_string(),
            Style::default().fg(t.fitness(v, 1.0)),
        ));
    }
    Line::from(spans)
}

/// Status color: graduated = success, pruned = dim, in-flight = warning.
fn shadow_color(t: &Theme, status: &str) -> Color {
    match status {
        "graduated" => t.success,
        "pruned" => t.dim,
        _ => t.warning,
    }
}

/// Clip `s` to `max` columns, ending in an ellipsis when cut.
fn clip(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
    out.push('…');
    out
}

/// A shadow's name: a standing expert's training by the scope it trained,
/// any other (the generational loop's) by its id and generation. A standing
/// expert's trainings are all generation zero, and their ids run to some
/// sixty characters, the scope near the end where a list cuts it off.
fn shadow_name(s: &Shadow) -> String {
    match behavior::shadow_scope(&s.id) {
        Some(scope) => format!("{scope} · behaviors"),
        None => format!("{} · g{}", s.id.as_str(), s.generation.0),
    }
}

/// A reward curve from its first reading to its last, `0.42 → 0.91`, or the
/// one reading a curve of one has. A standing expert's training reads the
/// base model's held-out pass rate and then the trained expert's.
fn reward_span(curve: &[f32]) -> Option<String> {
    match curve {
        [] => None,
        [only] => Some(format!("{only:.2}")),
        [first, .., last] => Some(format!("{first:.2} → {last:.2}")),
    }
}

/// A shadow's list row: its status glyph (graduated ✦, pruned ×, in flight
/// ◌), its name and where its reward went. The glyph and the row's color
/// carry the status, which the training panel spells out; the list is
/// narrow, and the trained reading at the row's end is what it is read for.
fn shadow_row(s: &Shadow) -> String {
    let glyph = match s.status.as_str() {
        "graduated" => "✦",
        "pruned" => "×",
        _ => "◌",
    };
    let mut row = format!("{glyph} {}", shadow_name(s));
    if let Some(r) = reward_span(&s.reward_curve) {
        row.push_str(" · ");
        row.push_str(&r);
    }
    row
}

/// A boundary's list row: actionable ⛔ or open ○, then the scope its
/// context names, when it names one (a training's does), and the behavior.
fn boundary_row(b: &FailureBoundary) -> String {
    let glyph = if b.is_actionable() { "⛔" } else { "○" };
    match b.fail_context.get("scope").and_then(|s| s.as_str()) {
        Some(scope) => format!("{glyph} {scope} · {}", b.behavior),
        None => format!("{glyph} {}", b.behavior),
    }
}

/// A boundary context's fields as `(key, value)` rows, a string value as it
/// is and any other as JSON: the scope, the task and the answer first, as a
/// training records them, then the rest in order. A context that is not an
/// object is one row.
fn context_fields(context: &serde_json::Value) -> Vec<(String, String)> {
    const FIRST: [&str; 3] = ["scope", "task", "answer"];
    let Some(map) = context.as_object() else {
        return match context {
            serde_json::Value::Null => Vec::new(),
            other => vec![("context".into(), other.to_string())],
        };
    };
    let mut keys: Vec<&String> = map.keys().collect();
    keys.sort_by_key(|k| FIRST.iter().position(|f| f == k).unwrap_or(FIRST.len()));
    keys.into_iter()
        .map(|k| {
            let v = &map[k];
            (
                k.clone(),
                v.as_str().map_or_else(|| v.to_string(), str::to_string),
            )
        })
        .collect()
}
