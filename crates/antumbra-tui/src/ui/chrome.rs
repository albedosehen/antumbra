//! The console chrome: the header wordmark/status line and the footer keybar.

use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::app::{App, Focus};
use crate::theme::rgb;

use super::{panel, pulse};

pub(super) fn header(f: &mut Frame, app: &App, area: Rect) {
    let t = app.theme();
    // Three pulsing diamonds = the tri-node core, then the wordmark.
    let mut spans = Vec::new();
    for k in 0..3 {
        let g = 0.5 + 0.5 * pulse(app.clock_ms, 700.0, k as f64 * 1.3);
        spans.push(Span::styled("◆", Style::default().fg(rgb(t.core, g))));
    }
    spans.push(Span::styled(
        "  ANTUMBRA",
        Style::default().fg(t.text).add_modifier(Modifier::BOLD),
    ));
    let gen = app
        .experts
        .iter()
        .map(|e| e.generation.0)
        .max()
        .unwrap_or(0);
    spans.push(Span::styled(
        format!(
            "   self-improving substrate · {} umbra · gen {}",
            app.experts.len(),
            gen
        ),
        Style::default().fg(t.ink),
    ));
    // Live frame-rate readout (the cap is set in the footer with `±`; `idle`
    // marks the eased rate after a still spell) and the active layout (`l`).
    spans.push(Span::styled(
        format!(
            "  ·  {} fps{}  ·  {}",
            app.shown_fps(),
            if app.idle { " idle" } else { "" },
            app.layout.name()
        ),
        Style::default().fg(t.dim),
    ));
    // Live event-stream indicator (`e` opens the stream).
    if !app.events.is_empty() {
        spans.push(Span::styled(
            format!("  ·  {} events", app.events.len()),
            Style::default().fg(t.accent),
        ));
    }
    f.render_widget(Paragraph::new(Line::from(spans)).block(panel(&t, "")), area);
}

pub(super) fn footer(f: &mut Frame, app: &App, area: Rect) {
    let t = app.theme();
    let router = if app.router.is_some() {
        "learned"
    } else {
        "heuristic"
    };
    let focus = match app.focus {
        Focus::Experts => "umbra",
        Focus::Shadows => "penumbra",
        Focus::Boundaries => "antumbra",
    };
    let actionable = app.boundaries.iter().filter(|b| b.is_actionable()).count();
    let key = |s: &'static str| Span::styled(s, Style::default().fg(Color::Black).bg(t.ink));
    let lbl = |s: String| Span::styled(s, Style::default().fg(t.ink));
    let line = Line::from(vec![
        key(" q "),
        lbl(" quit  ".into()),
        key(" ↑↓ "),
        lbl(" select  ".into()),
        key(" tab "),
        lbl(format!(" focus:{focus}  ")),
        key(" t "),
        lbl(format!(" theme:{}  ", t.name)),
        key(" ± "),
        lbl(format!(
            " cap:{}{}  ",
            app.target_fps,
            if app.auto_fps { " auto" } else { "" }
        )),
        key(" r "),
        lbl(" reload  ".into()),
        key(" ? "),
        lbl(" help  ".into()),
        key(" : "),
        lbl(" palette  ".into()),
        Span::styled(
            format!(
                "   {} umbra · {} penumbra · {} antumbra ({actionable} actionable) · gate {router}",
                app.experts.len(),
                app.shadows.len(),
                app.boundaries.len()
            ),
            Style::default().fg(t.dim),
        ),
    ]);
    f.render_widget(Paragraph::new(line), area);
}
