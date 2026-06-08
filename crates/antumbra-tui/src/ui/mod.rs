//! Rendering: the living population graph (umbra orbiting the tri-node core —
//! umbra/penumbra/antumbra, three linked minds), with detail and gate panels.
//! Orbit, pulse, and link-energy are hand-rolled from the animation clock so the
//! motion is fully under control. Every colour tints from the active [`Theme`],
//! so cycling a theme (`t`) recolours the whole console.
//!
//! Split across submodules: [`graph`] (the canvas), [`chrome`] (header/footer),
//! [`panels`] (the region detail panels), and [`overlays`] (help / palette /
//! filter modals). The small shared widget helpers live here.

mod chrome;
mod graph;
mod heatmap;
mod memory;
mod overlays;
mod panels;
mod table;

use std::f64::consts::TAU;

use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Paragraph};
use ratatui::Frame;

use crate::app::{App, Focus, LayoutMode, Mode, Page};
use crate::overlay;
use crate::theme::Theme;

pub use overlays::overlay_area;

pub fn render(f: &mut Frame, app: &App) {
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
    if app.mode != Mode::Normal {
        overlay::dim_backdrop(f, f.area());
        match app.mode {
            Mode::Help => overlays::help_overlay(f, app),
            Mode::Palette => overlays::palette_overlay(f, app),
            Mode::Filter => overlays::filter_overlay(f, app),
            Mode::Events => overlays::events_overlay(f, app),
            Mode::Detail => overlays::detail_overlay(f, app),
            Mode::Ask => overlays::ask_overlay(f, app),
            Mode::Confirm => overlays::confirm_overlay(f, app),
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
        Page::Loop => placeholder(
            f,
            app,
            area,
            " generational loop ",
            "grow → explore → score → decide → consolidate  ·  generation timeline",
        ),
        Page::Evals => placeholder(
            f,
            app,
            area,
            " evaluations ",
            "regression tripwire  ·  pass / fail metrics per expert",
        ),
    }
}

/// A page not yet built out: a titled panel naming what it will surface.
fn placeholder(f: &mut Frame, app: &App, area: Rect, title: &str, blurb: &str) {
    let t = app.theme();
    let block = panel(
        &t,
        Span::styled(title.to_string(), Style::default().fg(t.ink)),
    );
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.height < 3 {
        return;
    }
    let lines = vec![
        Line::from(Span::styled(
            "— in progress —",
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(Span::styled(blurb, Style::default().fg(t.dim))),
    ];
    let mid = Rect {
        x: inner.x,
        y: inner.y + inner.height / 2 - 1,
        width: inner.width,
        height: 3,
    };
    f.render_widget(Paragraph::new(lines).alignment(Alignment::Center), mid);
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

/// A rounded, dim-bordered panel block with a titled, inked header.
fn panel<'a>(t: &Theme, title: impl Into<Line<'a>>) -> Block<'a> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(t.dim))
        .title(title.into())
}

/// A panel whose border and title brighten to the accent when it holds focus —
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

/// A bold, accent-coloured panel heading (selected item's name).
fn heading<'a>(t: &Theme, text: String) -> Line<'a> {
    Line::from(Span::styled(
        text,
        Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
    ))
}

fn kv<'a>(t: &Theme, k: &'a str, v: &str) -> Line<'a> {
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

/// A `key  ▆▆▆▍──  0.62` row: a labelled gauge with its numeric value.
fn gauge_row<'a>(t: &Theme, k: &'a str, value: f32, color: Color) -> Line<'a> {
    let mut spans = vec![Span::styled(format!("{k:<11}"), Style::default().fg(t.dim))];
    spans.extend(gauge_spans(t, value, 12, color));
    spans.push(Span::styled(
        format!(" {value:.2}"),
        Style::default().fg(t.value),
    ));
    Line::from(spans)
}

/// A reward curve as block bars, each bar coloured along the fitness gradient by
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

/// Status colour: graduated = success, pruned = dim, in-flight = warning.
fn shadow_color(t: &Theme, status: &str) -> Color {
    match status {
        "graduated" => t.success,
        "pruned" => t.dim,
        _ => t.warning,
    }
}
