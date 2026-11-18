//! The Evals page: the evaluation runs as a table of subject, corpus
//! task, status, and the regression fingerprint. A `FAIL` on a frozen expert is
//! the no-forgetting tripwire firing, so a failure is the one status drawn in
//! the danger colour, under a shape of its own.

use ratatui::layout::{Constraint, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::Span;
use ratatui::widgets::{Cell, Paragraph, Row, Table, TableState, Wrap};
use ratatui::Frame;

use antumbra_core::{EvalStatus, EvaluationRun, ExpertId, SubjectKind};

use crate::app::App;
use crate::theme::Tone;
use crate::words;

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

/// Who a run checked: an expert by the name the console shows it under, any
/// other subject by its id.
fn subject(app: &App, e: &EvaluationRun) -> String {
    if e.subject_kind == SubjectKind::Expert {
        app.expert_called(&ExpertId::new(e.subject_id.clone()))
    } else {
        e.subject_id.clone()
    }
}

/// What a run checked, in words: a standing expert's training or audit by
/// what it is and for which scope, any other by its task's id.
pub(super) fn task_words(task: &str) -> String {
    match task.split_once(':') {
        Some(("training", scope)) => format!("training · {}", words::scope_name(scope)),
        Some(("audit", scope)) => format!("audit · {}", words::scope_name(scope)),
        _ => task.to_string(),
    }
}

/// What a run read. A training: the base model's held-out pass rate, then
/// the expert's. An audit: how many later examples the expert passed, and
/// the base model beside it. Any other: the start of its fingerprint, which
/// is what tells a frozen expert's answer from the one before.
pub(super) fn readings(e: &EvaluationRun) -> String {
    let number = |key: &str| e.metrics.as_ref().and_then(|m| m.get(key)?.as_f64());
    let kind = e
        .metrics
        .as_ref()
        .and_then(|m| m.get("kind")?.as_str())
        .unwrap_or_default();
    match (kind, number("base"), number("expert")) {
        ("training", Some(base), Some(expert)) => format!("{base:.2} → {expert:.2}"),
        ("audit", Some(base), Some(expert)) => {
            let of = number("examples").unwrap_or(0.0);
            format!("{expert:.0} of {of:.0} · base {base:.0}")
        }
        _ => e
            .regression_fingerprint
            .as_deref()
            .map_or_else(|| "none".to_string(), |s| s.chars().take(8).collect()),
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
        let why = "No evaluations yet: each training of an expert is kept as one, and so is each check of a served expert on examples it had not seen.";
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
        Cell::from("READINGS"),
        Cell::from("WHEN"),
    ])
    .style(Style::default().fg(t.ink2).add_modifier(Modifier::BOLD));

    // Build only on-screen rows so the table scales to long eval histories.
    let visible = (area.height as usize).saturating_sub(3); // borders + header
    let (offset, rel_sel) = super::visible_window(app.evals.len(), app.selected_eval, visible);
    let end = (offset + visible).min(app.evals.len());
    let rows = app.evals[offset..end].iter().map(|e| {
        Row::new(vec![
            Cell::from(Span::styled(subject(app, e), Style::default().fg(t.ink))),
            Cell::from(Span::styled(
                kind_label(e.subject_kind.as_str()),
                Style::default().fg(t.ink2),
            )),
            Cell::from(Span::styled(
                task_words(&e.corpus_task_id),
                Style::default().fg(t.ink2),
            )),
            Cell::from(Span::styled(
                status_text(e.status),
                Style::default()
                    .fg(t.tone(status_tone(e.status)))
                    .add_modifier(Modifier::BOLD),
            )),
            Cell::from(Span::styled(readings(e), Style::default().fg(t.ink2))),
            Cell::from(Span::styled(
                super::since(e.created_at),
                Style::default().fg(t.ink2),
            )),
        ])
    });

    let widths = [
        Constraint::Min(18),
        Constraint::Length(10),
        Constraint::Min(16),
        Constraint::Length(10),
        Constraint::Min(18),
        Constraint::Length(8),
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
