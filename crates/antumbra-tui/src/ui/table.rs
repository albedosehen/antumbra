//! The population data-grid: a full-width sortable table of the umbra (experts),
//! with a gauge column for fitness. Sort the active column with `s`; the
//! selection tracks the expert, not the row, so re-sorting never loses your place.

use ratatui::layout::{Constraint, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Cell, Row, Table, TableState};
use ratatui::Frame;

use crate::app::App;

use super::{gauge_spans, panel};

pub(super) fn experts_table(f: &mut Frame, app: &App, area: Rect) {
    let t = app.theme();
    let title = format!(
        " population · {} experts · sort: {} (s) ",
        app.experts.len(),
        app.sort.name()
    );
    let block = panel(&t, Span::styled(title, Style::default().fg(t.ink)));

    let header = Row::new([
        Cell::from("name"),
        Cell::from("fitness"),
        Cell::from("gen"),
        Cell::from("state"),
        Cell::from("model"),
    ])
    .style(Style::default().fg(t.dim).add_modifier(Modifier::BOLD));

    // Build only the on-screen rows so the table scales to any population size.
    let visible = (area.height as usize).saturating_sub(3); // borders + header
    let (offset, rel_sel) = super::visible_window(app.experts.len(), app.selected, visible);
    let end = (offset + visible).min(app.experts.len());
    let rows = app.experts[offset..end].iter().map(|e| {
        let mut fit = gauge_spans(&t, e.fitness, 10, t.fitness(e.fitness, 1.0));
        fit.push(Span::styled(
            format!(" {:.2}", e.fitness),
            Style::default().fg(t.value),
        ));
        let state = if e.is_frozen() { "frozen" } else { "live" };
        let state_color = if e.is_frozen() { t.value } else { t.success };
        Row::new(vec![
            Cell::from(Span::styled(e.name.clone(), Style::default().fg(t.text))),
            Cell::from(Line::from(fit)),
            Cell::from(Span::styled(
                e.generation.0.to_string(),
                Style::default().fg(t.dim),
            )),
            Cell::from(Span::styled(state, Style::default().fg(state_color))),
            Cell::from(Span::styled(
                e.base_model.clone(),
                Style::default().fg(t.dim),
            )),
        ])
    });

    let widths = [
        Constraint::Min(16),
        Constraint::Length(16),
        Constraint::Length(5),
        Constraint::Length(8),
        Constraint::Min(12),
    ];
    let table = Table::new(rows, widths)
        .header(header)
        .block(block)
        .row_highlight_style(
            Style::default()
                .bg(t.sel)
                .fg(t.text)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("▌ ");

    let mut state = TableState::default();
    if !app.experts.is_empty() {
        state.select(Some(rel_sel));
    }
    f.render_stateful_widget(table, area, &mut state);
}
