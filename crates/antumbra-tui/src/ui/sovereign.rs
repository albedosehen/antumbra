//! The Sovereign page: what a coding agent loses with its feature
//! flags off, as the rules the CLI keeps, and which skills are used, stalest
//! first. Read-only: it shows [`crate::sovereign::View`] and changes nothing.

use chrono::Utc;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Cell, Paragraph, Row, Table, TableState, Wrap};
use ratatui::Frame;

use crate::app::App;
use crate::theme::{Theme, Tone};

use super::{panel, state_line};

/// A skill not used for this long is called out.
const STALE_DAYS: i64 = 30;

fn header<'a>(t: &Theme, names: &[&'a str]) -> Row<'a> {
    Row::new(names.iter().map(|name| Cell::from(*name)))
        .style(Style::default().fg(t.ink2).add_modifier(Modifier::BOLD))
}

fn rules(f: &mut Frame, app: &App, area: Rect) {
    let t = app.theme();
    let view = &app.sovereign;
    let title = format!(" Rules · {} ", view.rules.len());
    let block = panel(&t, Span::styled(title, Style::default().fg(t.ink2)));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let note = view
        .checked
        .as_deref()
        .unwrap_or("No rules kept yet: `antumbra claude remember` writes one.");
    let parts = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(0)])
        .split(inner);
    let (Some(note_area), Some(table_area)) = (parts.first(), parts.get(1)) else {
        return;
    };
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(note, Style::default().fg(t.ink2))))
            .wrap(Wrap { trim: true }),
        *note_area,
    );

    let rows = view.rules.iter().map(|rule| {
        // Current everywhere is in force, retired everywhere is ended, and a
        // rule current in some workspaces and retired in others is open.
        let (tone, standing) = match (rule.current_in, rule.retired_in) {
            (0, _) => (Tone::Ended, "retired".to_string()),
            (current, 0) => (Tone::Settled, format!("current in {current}")),
            (current, retired) => (
                Tone::Open,
                format!("current in {current}, retired in {retired}"),
            ),
        };
        Row::new(vec![
            Cell::from(Span::styled(rule.rule.clone(), Style::default().fg(t.ink))),
            Cell::from(state_line(&t, tone, standing)),
        ])
    });
    let table = Table::new(rows, [Constraint::Min(22), Constraint::Min(16)])
        .header(header(&t, &["RULE", "WORKSPACES"]));
    f.render_widget(table, *table_area);
}

fn skills(f: &mut Frame, app: &App, area: Rect) {
    let t = app.theme();
    let view = &app.sovereign;
    let title = format!(
        " Skills · {} · least recently used first ",
        view.skills.len()
    );
    let block = panel(&t, Span::styled(title, Style::default().fg(t.ink2)));

    if view.skills.is_empty() {
        let inner = block.inner(area);
        f.render_widget(block, area);
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "No skill use counted yet: the hooks count one with `antumbra claude skill-used`.",
                Style::default().fg(t.ink3),
            )))
            .wrap(Wrap { trim: true }),
            inner,
        );
        return;
    }

    // Build only on-screen rows, as the other tables do.
    let visible = (area.height as usize).saturating_sub(3); // borders + header
    let (offset, rel_sel) = super::visible_window(view.skills.len(), app.selected_skill, visible);
    let end = (offset + visible).min(view.skills.len());
    let now = Utc::now();
    let rows = view
        .skills
        .get(offset..end)
        .unwrap_or_default()
        .iter()
        .map(|row| {
            // A stale skill's date fades; the list is stalest first, so the
            // faded dates are the ones it opens with.
            let idle = (now - row.last_used).num_days();
            let color = if idle >= STALE_DAYS { t.ink3 } else { t.ink };
            Row::new(vec![
                Cell::from(Span::styled(row.skill.clone(), Style::default().fg(t.ink))),
                Cell::from(Span::styled(
                    row.workspace.clone(),
                    Style::default().fg(t.ink2),
                )),
                Cell::from(Span::styled(
                    row.uses.to_string(),
                    Style::default().fg(t.ink),
                )),
                Cell::from(Span::styled(
                    row.last_used.format("%Y-%m-%d").to_string(),
                    Style::default().fg(color),
                )),
            ])
        });
    // Fixed widths for what has one, so the date is never what gets cut.
    let widths = [
        Constraint::Min(14),
        Constraint::Length(16),
        Constraint::Length(5),
        Constraint::Length(10),
    ];
    let table = Table::new(rows, widths)
        .header(header(&t, &["SKILL", "WORKSPACE", "USES", "LAST USED"]))
        .block(block)
        .row_highlight_style(t.selected())
        .highlight_symbol("▸ ");
    let mut state = TableState::default();
    state.select(Some(rel_sel));
    f.render_stateful_widget(table, area, &mut state);
}

pub(super) fn page(f: &mut Frame, app: &App, area: Rect) {
    let halves = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
        .split(area);
    let (Some(left), Some(right)) = (halves.first(), halves.get(1)) else {
        return;
    };
    rules(f, app, *left);
    skills(f, app, *right);
}
