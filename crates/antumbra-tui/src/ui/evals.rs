//! The Evals page: the evaluation runs as a table of subject, corpus
//! task, status, and the regression fingerprint. A `FAIL` on a frozen expert is
//! the no-forgetting tripwire firing, so failures are alert-coloured.

use ratatui::layout::{Constraint, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Cell, Paragraph, Row, Table, TableState};
use ratatui::Frame;

use antumbra_core::EvalStatus;

use crate::app::App;
use crate::theme::Theme;

use super::panel;

pub(super) fn status_color(t: &Theme, s: EvalStatus) -> Color {
    match s {
        EvalStatus::Success => t.success,
        EvalStatus::Failure | EvalStatus::Error => t.alert,
        EvalStatus::Running => t.warning,
        EvalStatus::Pending => t.dim,
    }
}

pub(super) fn status_label(s: EvalStatus) -> &'static str {
    match s {
        EvalStatus::Success => "pass",
        EvalStatus::Failure => "FAIL",
        EvalStatus::Error => "error",
        EvalStatus::Running => "running",
        EvalStatus::Pending => "pending",
    }
}

pub(super) fn page(f: &mut Frame, app: &App, area: Rect) {
    let t = app.theme();
    let failures = app
        .evals
        .iter()
        .filter(|e| matches!(e.status, EvalStatus::Failure | EvalStatus::Error))
        .count();
    let title = format!(
        " evaluations · {} runs · {} regressions ",
        app.evals.len(),
        failures
    );
    let block = panel(&t, Span::styled(title, Style::default().fg(t.ink)));

    if app.evals.is_empty() {
        let inner = block.inner(area);
        f.render_widget(block, area);
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "no evaluation runs yet; the tripwire is quiet",
                Style::default().fg(t.dim),
            ))),
            inner,
        );
        return;
    }

    let header = Row::new([
        Cell::from("subject"),
        Cell::from("kind"),
        Cell::from("task"),
        Cell::from("status"),
        Cell::from("fingerprint"),
    ])
    .style(Style::default().fg(t.dim).add_modifier(Modifier::BOLD));

    // Build only on-screen rows so the table scales to long eval histories.
    let visible = (area.height as usize).saturating_sub(3); // borders + header
    let (offset, rel_sel) = super::visible_window(app.evals.len(), app.selected_eval, visible);
    let end = (offset + visible).min(app.evals.len());
    let rows = app.evals[offset..end].iter().map(|e| {
        let fp = e
            .regression_fingerprint
            .as_deref()
            .map(|s| s.chars().take(8).collect::<String>())
            .unwrap_or_else(|| "n/a".to_string());
        Row::new(vec![
            Cell::from(Span::styled(
                e.subject_id.clone(),
                Style::default().fg(t.text),
            )),
            Cell::from(Span::styled(
                e.subject_kind.as_str(),
                Style::default().fg(t.dim),
            )),
            Cell::from(Span::styled(
                e.corpus_task_id.clone(),
                Style::default().fg(t.dim),
            )),
            Cell::from(Span::styled(
                status_label(e.status),
                Style::default()
                    .fg(status_color(&t, e.status))
                    .add_modifier(Modifier::BOLD),
            )),
            Cell::from(Span::styled(fp, Style::default().fg(t.value))),
        ])
    });

    let widths = [
        Constraint::Min(18),
        Constraint::Length(8),
        Constraint::Length(14),
        Constraint::Length(8),
        Constraint::Min(10),
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
    state.select(Some(rel_sel));
    f.render_stateful_widget(table, area, &mut state);
}
