//! The console chrome: the header wordmark/status line, the notice that says
//! what the console last did, and the footer's keys.

use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Tabs};
use ratatui::Frame;

use crate::app::{App, Page};
use crate::keys::{self, Binding};
use crate::theme::{Theme, Tone};

use super::{clip, gauge_spans, panel, pulse};

/// The KPI strip: a row of labeled mini-gauges for the substrate's quality
/// ratios (fitness / frozen / graduated / gating / memory), shown on every page.
pub(super) fn metrics(f: &mut Frame, app: &App, area: Rect) {
    let t = app.theme();
    // Every gauge fills in the one accent, as the dashboard's meters do: the
    // label says which is which.
    let mut spans = Vec::new();
    for (label, value) in app.dashboard_metrics() {
        cell(&mut spans, &t, label, value);
    }
    f.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// One `label ▆▆▆▍ 0.73` gauge cell appended to the strip.
fn cell(spans: &mut Vec<Span<'static>>, t: &Theme, label: &str, value: f32) {
    spans.push(Span::styled(
        format!("  {label} "),
        Style::default().fg(t.ink2),
    ));
    spans.extend(gauge_spans(t, value, 6, t.accent));
    spans.push(Span::styled(
        format!(" {value:.2}"),
        Style::default().fg(t.ink),
    ));
}

/// The top-level tab strip: one numbered tab per [`Page`], active one accented.
pub(super) fn tabs(f: &mut Frame, app: &App, area: Rect) {
    let t = app.theme();
    let titles: Vec<Line> = Page::ALL
        .iter()
        .enumerate()
        .map(|(i, p)| Line::from(format!(" {} {} ", i + 1, p.title())))
        .collect();
    let tabs = Tabs::new(titles)
        .select(app.page.index())
        .style(Style::default().fg(t.ink2))
        // The active tab as the dashboard marks a chosen segment: the soft
        // accent behind it, its name at full strength.
        .highlight_style(t.selected().fg(t.ink))
        .divider(Span::styled("·", Style::default().fg(t.line)));
    f.render_widget(tabs, area);
}

pub(super) fn header(f: &mut Frame, app: &App, area: Rect) {
    let t = app.theme();
    // Three pulsing diamonds = the tri-node core, then the wordmark.
    let mut spans = Vec::new();
    for k in 0..3 {
        let g = 0.5 + 0.5 * pulse(app.clock_ms, 700.0, k as f64 * 1.3);
        spans.push(Span::styled("◆", Style::default().fg(t.glow(g))));
    }
    spans.push(Span::styled(
        "  ANTUMBRA",
        Style::default().fg(t.ink).add_modifier(Modifier::BOLD),
    ));
    // A throwaway `--demo` store gets a badge so the operator knows nothing persists.
    if app.demo {
        spans.push(Span::styled("  DEMO ", t.chip_primary(false)));
    }
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
        Style::default().fg(t.ink2),
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
        Style::default().fg(t.ink3),
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

/// The line above the footer: what the console last did, declined to do, or
/// could not do, under the shape of how it went. Empty once that has aged out.
pub(super) fn notice(f: &mut Frame, app: &App, area: Rect) {
    let t = app.theme();
    let Some(notice) = app.current_notice() else {
        return;
    };
    // A refusal or a failure is said in the danger colour; anything else at
    // full strength.
    let ink = if notice.tone == Tone::Failed {
        t.danger
    } else {
        t.ink
    };
    let room = usize::from(area.width).saturating_sub(4);
    let line = Line::from(vec![
        Span::styled(
            format!(" {} ", notice.tone.glyph()),
            Style::default().fg(t.tone(notice.tone)),
        ),
        Span::styled(clip(&notice.text, room), Style::default().fg(ink)),
    ]);
    f.render_widget(Paragraph::new(line), area);
}

/// The columns a binding takes in the footer: its chip, its hint, and the gap
/// to the next.
fn footer_width(b: &Binding) -> usize {
    b.chip.chars().count() + b.hint.chars().count() + 5
}

/// The bindings the footer shows in `width` columns, in the order they are
/// listed: every one with a place in the footer when they fit, else the most
/// useful, with help and quit kept however narrow the row.
pub(super) fn footer_bindings(bindings: &[Binding], width: u16) -> Vec<&Binding> {
    let mut shown: Vec<&Binding> = bindings.iter().filter(|b| b.rank.is_some()).collect();
    while shown.iter().map(|b| footer_width(b)).sum::<usize>() > usize::from(width) {
        // The least useful goes first: the highest rank, the later of equals.
        let Some(drop) = shown
            .iter()
            .enumerate()
            .filter(|(_, b)| b.rank != Some(keys::KEPT))
            .max_by_key(|(i, b)| (b.rank, *i))
            .map(|(i, _)| i)
        else {
            break;
        };
        shown.remove(drop);
    }
    shown
}

/// The footer: the keys of the view in front, the page's or the overlay's,
/// each as a chip and a word. It is drawn from the same list the keys are
/// read from ([`keys::bindings`]), so it cannot offer a key that does nothing
/// here.
pub(super) fn footer(f: &mut Frame, app: &App, area: Rect) {
    let t = app.theme();
    let bindings = keys::bindings(app);
    let mut spans = Vec::new();
    for b in footer_bindings(&bindings, area.width) {
        spans.push(Span::styled(format!(" {} ", b.chip), t.chip()));
        spans.push(Span::styled(
            format!(" {}  ", b.hint),
            Style::default().fg(t.ink2),
        ));
    }
    f.render_widget(Paragraph::new(Line::from(spans)), area);
}
