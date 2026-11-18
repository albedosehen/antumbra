//! The Evals page: the evaluation runs as a table of subject, corpus
//! task, status, and the regression fingerprint. A `FAIL` on a frozen expert is
//! the no-forgetting tripwire firing, so a failure is the one status drawn in
//! the danger colour, under a shape of its own.

use ratatui::layout::{Constraint, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::Span;
use ratatui::widgets::{Cell, Paragraph, Row, Table, TableState, Wrap};
use ratatui::Frame;

use antumbra_core::EvalStatus;

use crate::app::App;
use crate::theme::Tone;

use super::{nothing_here, panel};

/// What a run's status is to the eye: a pass is settled, a failure or an
/// error failed, and one running or waiting still open.
pub(super) fn status_tone(s: EvalStatus) -> Tone {
    match s {
        EvalStatus::Success => Tone::Settled,
        EvalStatus::Failure | EvalStatus::Error => Tone::Failed,
        EvalStatus::Running | EvalStatus::Pending => Tone::Open,
    }
}

/// What a run checked, in the console's words: the stored kind `shadow` is a
/// candidate here, and the rest read as they are stored.
pub(super) fn kind_label(kind: &str) -> &str {
    match kind {
        "shadow" => "candidate",
        other => other,
    }
}

/// A run's status as its shape and its word: `● pass`, `▲ FAIL`.
pub(super) fn status_text(s: EvalStatus) -> String {
    format!("{} {}", status_tone(s).glyph(), status_label(s))
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
        " Evaluations · {} runs · {} failed ",
        app.evals.len(),
        failures
    );
    let block = panel(&t, Span::styled(title, Style::default().fg(t.ink2)));

    if app.evals.is_empty() {
        let inner = block.inner(area);
        f.render_widget(block, area);
        let why = "No evaluations yet: the training loop records one each time it checks an expert or a candidate.";
        f.render_widget(
            Paragraph::new(nothing_here(app, &t, why)).wrap(Wrap { trim: true }),
            inner,
        );
        return;
    }

    let header = Row::new([
        Cell::from("SUBJECT"),
        Cell::from("KIND"),
        Cell::from("TASK"),
        Cell::from("RESULT"),
        Cell::from("FINGERPRINT"),
    ])
    .style(Style::default().fg(t.ink2).add_modifier(Modifier::BOLD));

    // Build only on-screen rows so the table scales to long eval histories.
    let visible = (area.height as usize).saturating_sub(3); // borders + header
    let (offset, rel_sel) = super::visible_window(app.evals.len(), app.selected_eval, visible);
    let end = (offset + visible).min(app.evals.len());
    let rows = app.evals[offset..end].iter().map(|e| {
        let fp = e
            .regression_fingerprint
            .as_deref()
            .map(|s| s.chars().take(8).collect::<String>())
            .unwrap_or_else(|| "none".to_string());
        Row::new(vec![
            Cell::from(Span::styled(
                e.subject_id.clone(),
                Style::default().fg(t.ink),
            )),
            Cell::from(Span::styled(
                kind_label(e.subject_kind.as_str()),
                Style::default().fg(t.ink2),
            )),
            Cell::from(Span::styled(
                e.corpus_task_id.clone(),
                Style::default().fg(t.ink2),
            )),
            Cell::from(Span::styled(
                status_text(e.status),
                Style::default()
                    .fg(t.tone(status_tone(e.status)))
                    .add_modifier(Modifier::BOLD),
            )),
            Cell::from(Span::styled(fp, Style::default().fg(t.ink2))),
        ])
    });

    let widths = [
        Constraint::Min(18),
        Constraint::Length(10),
        Constraint::Length(14),
        Constraint::Length(10),
        Constraint::Min(10),
    ];
    let table = Table::new(rows, widths)
        .header(header)
        .block(block)
        .row_highlight_style(t.selected())
        .highlight_symbol("▸ ");

    let mut state = TableState::default();
    state.select(Some(rel_sel));
    f.render_stateful_widget(table, area, &mut state);
}
