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
use crate::theme::Theme;

use super::panel;

/// A skill not used for this long is called out.
const STALE_DAYS: i64 = 30;

fn header<'a>(t: &Theme, names: &[&'a str]) -> Row<'a> {
    Row::new(names.iter().map(|name| Cell::from(*name)))
        .style(Style::default().fg(t.dim).add_modifier(Modifier::BOLD))
}

fn rules(f: &mut Frame, app: &App, area: Rect) {
    let t = app.theme();
    let view = &app.sovereign;
    let title = format!(" rules · {} ", view.rules.len());
    let block = panel(&t, Span::styled(title, Style::default().fg(t.ink)));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let note = view.checked.as_deref().unwrap_or(
        "no rules kept yet: `antumbra claude remember` writes them to a claude-code compartment",
    );
    let parts = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(0)])
        .split(inner);
    let (Some(note_area), Some(table_area)) = (parts.first(), parts.get(1)) else {
        return;
    };
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(note, Style::default().fg(t.dim))))
            .wrap(Wrap { trim: true }),
        *note_area,
    );

    let rows = view.rules.iter().map(|rule| {
        let (standing, color) = match (rule.current_in, rule.retired_in) {
            (0, _) => ("retired".to_string(), t.warning),
            (current, 0) => (format!("current in {current}"), t.success),
            (current, retired) => (
                format!("current in {current}, retired in {retired}"),
                t.text,
            ),
        };
        Row::new(vec![
            Cell::from(Span::styled(rule.rule.clone(), Style::default().fg(t.text))),
            Cell::from(Span::styled(standing, Style::default().fg(color))),
        ])
    });
    let table = Table::new(rows, [Constraint::Min(22), Constraint::Min(16)])
        .header(header(&t, &["rule", "workspaces"]));
    f.render_widget(table, *table_area);
}

fn skills(f: &mut Frame, app: &App, area: Rect) {
    let t = app.theme();
    let view = &app.sovereign;
    let title = format!(" skill use · {} · stalest first ", view.skills.len());
    let block = panel(&t, Span::styled(title, Style::default().fg(t.ink)));

    if view.skills.is_empty() {
        let inner = block.inner(area);
        f.render_widget(block, area);
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "no skill has been counted yet: two hooks run `antumbra claude skill-used`",
                Style::default().fg(t.dim),
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
            let idle = (now - row.last_used).num_days();
            let color = if idle >= STALE_DAYS {
                t.warning
            } else {
                t.text
            };
            Row::new(vec![
                Cell::from(Span::styled(row.skill.clone(), Style::default().fg(t.text))),
                Cell::from(Span::styled(
                    row.workspace.clone(),
                    Style::default().fg(t.dim),
                )),
                Cell::from(Span::styled(
                    row.uses.to_string(),
                    Style::default().fg(t.value),
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
        .header(header(&t, &["skill", "workspace", "uses", "last used"]))
        .block(block)
        .row_highlight_style(
            Style::default()
                .bg(t.sel)
                .fg(t.text)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("▌ ");
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
