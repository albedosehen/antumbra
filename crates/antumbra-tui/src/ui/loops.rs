//! The Loop page, the durable generational loop: the loop as a pipeline,
//! grow → explore → score → decide → consolidate → (grow), with the current
//! stage lit, the generation counter, and the run it belongs to. The loop state
//! is the durable checkpoint, so this is a live view of where the substrate is
//! in its self-improvement cycle. When more than one run is live, the runs are a
//! selectable list (j/k) and `enter` opens the run's drill-down.

use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};
use ratatui::Frame;

use antumbra_core::generational::LoopState;

use crate::app::App;

use super::{nothing_here, panel};

/// The forward cycle, in order (Paused is a rest state, shown separately).
pub(super) const STAGES: [LoopState; 5] = [
    LoopState::Grow,
    LoopState::Explore,
    LoopState::Score,
    LoopState::Decide,
    LoopState::Consolidate,
];

/// How many run rows the selectable list shows before it windows + summarizes.
const MAX_RUN_ROWS: usize = 8;

pub(super) fn stage_label(s: LoopState) -> &'static str {
    match s {
        LoopState::Grow => "grow",
        LoopState::Explore => "explore",
        LoopState::Score => "score",
        LoopState::Decide => "decide",
        LoopState::Consolidate => "consolidate",
        LoopState::Paused => "paused",
    }
}

pub(super) fn stage_blurb(s: LoopState) -> &'static str {
    match s {
        LoopState::Grow => "find where the experts are weak, and start candidates there",
        LoopState::Explore => "the candidates train",
        LoopState::Score => "each candidate's answers are checked",
        LoopState::Decide => {
            "the winners become experts; the rest are pruned, and where they failed is recorded"
        }
        LoopState::Consolidate => {
            "fitness is updated and the stores tidied, for the next generation"
        }
        LoopState::Paused => "at rest; it resumes where it stopped",
    }
}

pub(super) fn page(f: &mut Frame, app: &App, area: Rect) {
    let t = app.theme();
    let block = panel(
        &t,
        Span::styled(" Training loop ", Style::default().fg(t.ink2)),
    );
    let inner = block.inner(area);
    f.render_widget(block, area);

    let Some(head) = app.selected_loop_head() else {
        let why = "No training loop has run yet: `antumbra loop` or `antumbra train` runs one.";
        f.render_widget(
            Paragraph::new(nothing_here(app, &t, why)).wrap(Wrap { trim: true }),
            inner,
        );
        return;
    };

    let mut lines: Vec<Line> = vec![Line::from("")];

    // More than one run: a selectable list, the focused run highlighted, so the
    // pipeline + drill-down below always describe the run under the cursor.
    if app.loop_heads.len() > 1 {
        let (offset, _) =
            super::visible_window(app.loop_heads.len(), app.selected_loop, MAX_RUN_ROWS);
        let end = (offset + MAX_RUN_ROWS).min(app.loop_heads.len());
        for (i, h) in app.loop_heads[offset..end].iter().enumerate() {
            let idx = offset + i;
            let selected = idx == app.selected_loop;
            let (marker, style) = if selected {
                ("▸ ", t.selected().fg(t.ink))
            } else {
                ("  ", Style::default().fg(t.ink2))
            };
            // A run with a halt pending says so in the list, so the one that
            // was halted is found without stepping through them.
            let halting = if app.halt_pending(&h.run_id) {
                " · halt requested"
            } else {
                ""
            };
            lines.push(Line::from(vec![
                Span::styled(marker, Style::default().fg(t.accent)),
                Span::styled(
                    format!(
                        "{:<24}  gen {:<4}  {}{halting}",
                        h.run_id.as_str(),
                        h.generation.0,
                        stage_label(h.state)
                    ),
                    style,
                ),
            ]));
        }
        if end < app.loop_heads.len() {
            lines.push(Line::from(Span::styled(
                format!("  and {} more", app.loop_heads.len() - end),
                Style::default().fg(t.ink3),
            )));
        }
        lines.push(Line::from(""));
    }

    lines.push(Line::from(vec![
        Span::styled("  generation ", Style::default().fg(t.ink2)),
        Span::styled(
            head.generation.0.to_string(),
            Style::default().fg(t.ink).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("    ·    {}    ·    stage ", head.run_id.as_str()),
            Style::default().fg(t.ink2),
        ),
        Span::styled(
            stage_label(head.state),
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
        ),
    ]));
    lines.push(Line::from(""));

    // The pipeline: each stage a node, the current one lit; loops back at the end.
    let mut spans = vec![Span::raw("  ")];
    for (i, &s) in STAGES.iter().enumerate() {
        let current = s == head.state;
        let glyph = if current { "◉" } else { "○" };
        let color = if current { t.accent } else { t.ink3 };
        let mut style = Style::default().fg(color);
        if current {
            style = style.add_modifier(Modifier::BOLD);
        }
        spans.push(Span::styled(format!("{glyph} {}", stage_label(s)), style));
        if i < STAGES.len() - 1 {
            spans.push(Span::styled("  →  ", Style::default().fg(t.line)));
        }
    }
    spans.push(Span::styled("  ↺", Style::default().fg(t.line)));
    lines.push(Line::from(spans));

    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled("  → ", Style::default().fg(t.accent)),
        Span::styled(stage_blurb(head.state), Style::default().fg(t.ink2)),
    ]));

    if head.state == LoopState::Paused {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "  ‖ paused: it resumes where it stopped",
            Style::default().fg(t.ink2),
        )));
    }

    lines.push(Line::from(""));
    if app.halt_pending(&head.run_id) {
        // A halt asked for is in force until it is reached or canceled: the
        // accent, which marks what is in force, and not the danger colour,
        // which is kept for what failed.
        lines.push(Line::from(Span::styled(
            "  ● halt requested: this run stops at its next generation",
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(Span::styled(
            "    x · cancel the halt",
            Style::default().fg(t.ink3),
        )));
    } else if app.operator {
        lines.push(Line::from(Span::styled(
            "  x · halt this run at its next generation",
            Style::default().fg(t.ink3),
        )));
    }
    lines.push(Line::from(Span::styled(
        "  enter · everything about this run",
        Style::default().fg(t.ink3),
    )));

    f.render_widget(Paragraph::new(lines), inner);
}
