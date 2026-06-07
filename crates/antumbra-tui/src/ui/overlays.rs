//! The modal overlays drawn over the dimmed live view: the keybinding help, the
//! command palette, and the focused-list filter.

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph};
use ratatui::Frame;

use crate::app::{App, Focus, Mode};
use crate::overlay;
use crate::scroll;

/// The modal box rectangle for the active overlay (the size source the open
/// animation also targets). `None` in the live view.
pub fn overlay_area(app: &App, frame: Rect) -> Option<Rect> {
    match app.mode {
        Mode::Help => Some(overlay::centered(frame, 52, 23)),
        Mode::Palette => {
            let listed = app.palette_matches().len().max(1) as u16;
            Some(overlay::centered(frame, 56, listed + 4))
        }
        Mode::Filter => {
            let listed = (app.filtered().len() as u16 + 4).min(18);
            Some(overlay::centered(frame, 50, listed))
        }
        Mode::Normal => None,
    }
}

/// The keybinding reference, a centred modal over the live view (`?` toggles).
pub(super) fn help_overlay(f: &mut Frame, app: &App) {
    let t = app.theme();
    let Some(area) = overlay_area(app, f.area()) else {
        return;
    };
    let inner = overlay::modal(f, &t, area, "help");

    let group = |label: &str| {
        Line::from(Span::styled(
            label.to_string(),
            Style::default().fg(t.dim).add_modifier(Modifier::BOLD),
        ))
    };
    let bind = |keys: &str, desc: &str| {
        Line::from(vec![
            Span::styled(
                format!("  {keys:<8}"),
                Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
            ),
            Span::styled(desc.to_string(), Style::default().fg(t.ink)),
        ])
    };
    let lines = vec![
        group("navigate"),
        bind("↑↓ jk", "select in the focused list"),
        bind("g G", "jump to first / last  (home / end)"),
        bind("pgup/dn", "move by a page"),
        bind("tab", "switch focus: umbra / penumbra / antumbra"),
        Line::from(""),
        group("view"),
        bind("l", "cycle layout: focused / dashboard / graph"),
        bind("t", "cycle theme: shadow / ember / mono"),
        Line::from(""),
        group("frame rate"),
        bind("+ -", "pin the cap to a refresh rate"),
        bind("a", "follow the active monitor"),
        Line::from(""),
        group("command"),
        bind("/", "filter the focused list, jump to a match"),
        bind(":", "open the command palette"),
        bind("r", "reload from the store"),
        bind("? esc", "close this help"),
        bind("q", "quit"),
    ];
    f.render_widget(Paragraph::new(lines), inner);
}

/// The command palette: a query line over a fuzzy-ranked command list (`:` opens).
pub(super) fn palette_overlay(f: &mut Frame, app: &App) {
    let t = app.theme();
    let matches = app.palette_matches();
    let Some(area) = overlay_area(app, f.area()) else {
        return;
    };
    let inner = overlay::modal(f, &t, area, "command");
    let rows = Layout::vertical([Constraint::Length(2), Constraint::Min(0)]).split(inner);

    // Query line with a block cursor.
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("› ", Style::default().fg(t.accent)),
            Span::styled(app.palette.query.clone(), Style::default().fg(t.text)),
            Span::styled("▏", Style::default().fg(t.accent)),
        ])),
        rows[0],
    );

    // Ranked matches as a list, the selection full-row highlighted.
    if matches.is_empty() {
        f.render_widget(
            Paragraph::new(Span::styled(
                "  (no matching command)",
                Style::default().fg(t.dim),
            )),
            rows[1],
        );
    } else {
        let lines: Vec<Line> = matches
            .iter()
            .map(|c| Line::from(Span::styled(c.label, Style::default().fg(t.ink))))
            .collect();
        scroll::list(
            f,
            &t,
            rows[1],
            Block::default(),
            lines,
            Some(app.palette.selected),
        );
    }
}

/// The focused-list filter (`/`): a query over the focused region's items,
/// fuzzy-ranked; picking one jumps the selection to it.
pub(super) fn filter_overlay(f: &mut Frame, app: &App) {
    let t = app.theme();
    let Some(area) = overlay_area(app, f.area()) else {
        return;
    };
    let region = match app.focus {
        Focus::Experts => "umbra",
        Focus::Shadows => "penumbra",
        Focus::Boundaries => "antumbra",
    };
    let inner = overlay::modal(f, &t, area, &format!("filter {region}"));
    let rows = Layout::vertical([Constraint::Length(2), Constraint::Min(0)]).split(inner);

    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("/ ", Style::default().fg(t.accent)),
            Span::styled(app.filter.clone(), Style::default().fg(t.text)),
            Span::styled("▏", Style::default().fg(t.accent)),
        ])),
        rows[0],
    );

    let matches = app.filtered();
    if matches.is_empty() {
        f.render_widget(
            Paragraph::new(Span::styled("  (no match)", Style::default().fg(t.dim))),
            rows[1],
        );
    } else {
        let lines: Vec<Line> = matches
            .iter()
            .map(|(_, label)| Line::from(Span::styled(label.clone(), Style::default().fg(t.ink))))
            .collect();
        scroll::list(
            f,
            &t,
            rows[1],
            Block::default(),
            lines,
            Some(app.filter_selected),
        );
    }
}
