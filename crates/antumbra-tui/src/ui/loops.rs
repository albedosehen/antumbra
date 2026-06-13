//! The Loop page, the durable generational loop: the loop as a pipeline,
//! grow → explore → score → decide → consolidate → (grow), with the current
//! stage lit, the generation counter, and the run it belongs to. The loop state
//! is the durable checkpoint, so this is a live view of where the substrate is
//! in its self-improvement cycle. When more than one run is live, the runs are a
//! selectable list (j/k) and `enter` opens the run's drill-down.

use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use antumbra_core::generational::LoopState;

use crate::app::App;

use super::panel;

/// The forward cycle, in order (Paused is a rest state, shown separately).
pub(super) const STAGES: [LoopState; 5] = [
    LoopState::Grow,
    LoopState::Explore,
    LoopState::Score,
    LoopState::Decide,
    LoopState::Consolidate,
];

/// How many run rows the selectable list shows before it windows + summarises.
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
        LoopState::Grow => "find where the population is weak; spawn shadows there",
        LoopState::Explore => "shadows train (DIY candle QLoRA via the trainer)",
        LoopState::Score => "verifiers + critic produce reward signals",
        LoopState::Decide => "winners graduate to experts; losers prune + log a boundary",
        LoopState::Consolidate => "update fitness, merge/decay the stores, reindex, checkpoint",
        LoopState::Paused => "fully resumable rest state",
    }
}

pub(super) fn page(f: &mut Frame, app: &App, area: Rect) {
    let t = app.theme();
    let block = panel(
        &t,
        Span::styled(" generational loop ", Style::default().fg(t.ink)),
    );
    let inner = block.inner(area);
    f.render_widget(block, area);

    let Some(head) = app.selected_loop_head() else {
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "loop not started; no checkpoint yet",
                Style::default().fg(t.dim),
            ))),
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
                (
                    "▌ ",
                    Style::default()
                        .bg(t.sel)
                        .fg(t.text)
                        .add_modifier(Modifier::BOLD),
                )
            } else {
                ("  ", Style::default().fg(t.dim))
            };
            lines.push(Line::from(vec![
                Span::styled(marker, Style::default().fg(t.accent)),
                Span::styled(
                    format!(
                        "{:<24}  gen {:<4}  {}",
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
                format!("  + {} more run(s)", app.loop_heads.len() - end),
                Style::default().fg(t.dim),
            )));
        }
        lines.push(Line::from(""));
    }

    lines.push(Line::from(vec![
        Span::styled("  generation ", Style::default().fg(t.dim)),
        Span::styled(
            head.generation.0.to_string(),
            Style::default().fg(t.value).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("    ·    {}    ·    state ", head.run_id.as_str()),
            Style::default().fg(t.dim),
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
        let color = if current { t.accent } else { t.dim };
        let mut style = Style::default().fg(color);
        if current {
            style = style.add_modifier(Modifier::BOLD);
        }
        spans.push(Span::styled(format!("{glyph} {}", stage_label(s)), style));
        if i < STAGES.len() - 1 {
            spans.push(Span::styled("  →  ", Style::default().fg(t.dim)));
        }
    }
    spans.push(Span::styled("  ↺", Style::default().fg(t.dim)));
    lines.push(Line::from(spans));

    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled("  → ", Style::default().fg(t.accent)),
        Span::styled(stage_blurb(head.state), Style::default().fg(t.ink)),
    ]));

    if head.state == LoopState::Paused {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "  ‖ paused; resumes into grow",
            Style::default().fg(t.warning),
        )));
    }

    lines.push(Line::from(""));
    if app.loop_halt_pending {
        lines.push(Line::from(Span::styled(
            "  ⚠ halt requested; the loop will stop at the next generation",
            Style::default().fg(t.warning).add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(Span::styled(
            "    cancel via the palette (loop · cancel a pending halt)",
            Style::default().fg(t.dim),
        )));
    } else {
        lines.push(Line::from(Span::styled(
            "  x · request a graceful halt (stops at the next generation)",
            Style::default().fg(t.dim),
        )));
    }
    lines.push(Line::from(Span::styled(
        "  enter · inspect this run",
        Style::default().fg(t.dim),
    )));

    f.render_widget(Paragraph::new(lines), inner);
}
