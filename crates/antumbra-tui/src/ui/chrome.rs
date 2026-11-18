//! The console chrome: the top bar, the tabs, the sentence that says what the
//! page is, the notice that says what the console last did, and the footer's
//! keys.

use ratatui::layout::{Alignment, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Tabs};
use ratatui::Frame;

use crate::app::{App, Page};
use crate::keys::{self, Binding};
use crate::theme::Tone;
use crate::words;

use super::{clip, pulse};

/// The top bar, in one row as the dashboard's is: the name, then what the
/// console reads and what its token may do, and at the right whether reads
/// are succeeding. A person who opens the console sees, before anything else,
/// which server this is, whether they can act on it, and whether what is
/// shown is current.
pub(super) fn header(f: &mut Frame, app: &App, area: Rect) {
    let t = app.theme();
    let dim = Style::default().fg(t.ink2);

    // At the right: live, or not answering. The demo reads nothing.
    let status: Vec<Span> = if app.demo {
        Vec::new()
    } else if app.reads_failing {
        vec![Span::styled(
            format!("{} not answering ", Tone::Failed.glyph()),
            Style::default().fg(t.danger),
        )]
    } else {
        vec![
            Span::styled(
                format!("{} ", Tone::Settled.glyph()),
                Style::default().fg(t.accent),
            ),
            Span::styled("live ", dim),
        ]
    };
    let status_width: usize = status.iter().map(|s| s.content.chars().count()).sum();

    // Three pulsing diamonds, the mark, then the name.
    let mut spans = vec![Span::raw(" ")];
    for k in 0..3 {
        let g = 0.5 + 0.5 * pulse(app.clock_ms, 700.0, f64::from(k) * 1.3);
        spans.push(Span::styled("◆", Style::default().fg(t.glow(g))));
    }
    spans.push(Span::styled(
        " Antumbra   ",
        Style::default().fg(t.ink).add_modifier(Modifier::BOLD),
    ));
    if app.demo {
        // A throwaway store: say so, so nobody takes it for their own.
        spans.push(Span::styled(" demo ", t.chip()));
        spans.push(Span::styled(" sample data · nothing here is saved", dim));
    } else {
        let room = usize::from(area.width).saturating_sub(status_width + 32);
        spans.push(Span::styled(clip(&app.source, room.max(12)), dim));
        spans.push(Span::styled(" · ", Style::default().fg(t.line)));
        if app.operator {
            spans.push(Span::styled("operator", dim));
        } else {
            // At full strength: it is why the keys that act do nothing.
            spans.push(Span::styled("read-only", Style::default().fg(t.ink)));
        }
    }
    let left = Rect {
        width: area
            .width
            .saturating_sub(u16::try_from(status_width).unwrap_or(0)),
        ..area
    };
    f.render_widget(Paragraph::new(Line::from(spans)), left);
    f.render_widget(
        Paragraph::new(Line::from(status)).alignment(Alignment::Right),
        area,
    );
}

/// The top-level tab strip: one numbered tab per [`Page`], the number being
/// the key that goes to it, and the active one marked.
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

/// The sentence under the tabs: what the page, or the list in front, is
/// ([`words::caption`]). Its first word, the name of the thing, is at full
/// strength.
pub(super) fn caption(f: &mut Frame, app: &App, area: Rect) {
    let t = app.theme();
    let room = usize::from(area.width).saturating_sub(2);
    let text = clip(words::caption(app), room);
    let (name, rest) = match text.split_once(':') {
        Some((name, rest)) => (format!(" {name}:"), rest.to_string()),
        None => (String::from(" "), text),
    };
    let line = Line::from(vec![
        Span::styled(
            name,
            Style::default().fg(t.ink).add_modifier(Modifier::BOLD),
        ),
        Span::styled(rest, Style::default().fg(t.ink2)),
    ]);
    f.render_widget(Paragraph::new(line), area);
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
