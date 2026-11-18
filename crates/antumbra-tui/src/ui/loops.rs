//! The Loop page, the durable generational loop: the loop as a pipeline,
//! grow → explore → score → decide → consolidate → (grow), with the current
//! stage lit, the generation counter, and the run it belongs to. The loop state
//! is the durable checkpoint, so this is a live view of where the substrate is
//! in its self-improvement cycle. When more than one run is live, the runs are a
//! selectable list (j/k) and `enter` opens the run's drill-down.

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Cell, Paragraph, Row, Table, Wrap};
use ratatui::Frame;

use antumbra_core::generational::LoopState;

use crate::app::App;
use crate::standing::{Scope, Standing};
use crate::theme::Tone;
use crate::words;

use super::{nothing_here, panel, since};

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

/// The Training page: what the server teaches without being asked, over the
/// loop a person runs. A workspace with no behaviors has only the second.
pub(super) fn page(f: &mut Frame, app: &App, area: Rect) {
    if app.standing.is_empty() {
        training_loop(f, app, area);
        return;
    }
    // Its rows, a heading, a note and the border. The loop under it keeps
    // the rows its pipeline needs, or, with no run, the few its one sentence
    // does, so a short terminal still lists the experts.
    let kept = if app.loop_heads.is_empty() { 4 } else { 10 };
    let wanted = u16::try_from(app.standing.len())
        .unwrap_or(u16::MAX)
        .saturating_add(4);
    let parts = Layout::vertical([
        Constraint::Length(wanted.min(area.height.saturating_sub(kept))),
        Constraint::Min(0),
    ])
    .split(area);
    standing(f, app, parts[0]);
    training_loop(f, app, parts[1]);
}

/// A wait as its coarsest unit: `5h`, `40m`.
fn wait(left: chrono::Duration) -> String {
    if left.num_hours() > 0 {
        format!("{}h", left.num_hours())
    } else {
        format!("{}m", left.num_minutes().max(1))
    }
}

/// Where a scope's training stands, as a shape and a sentence.
fn stands(app: &App, s: &Scope) -> (Tone, String) {
    match s.state {
        Standing::UpToDate => (Tone::Settled, "up to date".into()),
        Standing::Waiting => (Tone::Open, "trained at the server's next pass".into()),
        Standing::Refused => (
            Tone::Failed,
            "not admitted: it waits for a behavior to change".into(),
        ),
        Standing::Dropping => (
            Tone::Ended,
            "nothing left to teach: dropped at the next pass".into(),
        ),
        Standing::Proposed => (
            Tone::Open,
            match s.accepts_in {
                Some(left) if left <= chrono::Duration::zero() => {
                    "accepted at the server's next pass".into()
                }
                Some(left) => format!("its first is accepted in {}", wait(left)),
                None if app.policy.accept_after_hours > 0.0 => {
                    "none can be taught yet: the dashboard says why".into()
                }
                None => "waits for you to accept it".into(),
            },
        ),
    }
}

/// The standing experts: each scope, how many of its behaviors are taught,
/// wait and are proposed, when it was last trained, and where it stands.
/// The server works through these by itself, so the panel says what it will
/// do next and not what to type.
fn standing(f: &mut Frame, app: &App, area: Rect) {
    let t = app.theme();
    let name = format!(
        " Standing experts {} · taught by the server ",
        app.standing.len()
    );
    let block = panel(&t, Span::styled(name, Style::default().fg(t.ink2)));
    let inner = block.inner(area);
    f.render_widget(block, area);
    let parts = Layout::vertical([Constraint::Min(0), Constraint::Length(1)]).split(inner);

    let count = |n: usize| {
        let ink = if n == 0 { t.ink3 } else { t.ink };
        Cell::from(Span::styled(n.to_string(), Style::default().fg(ink)))
    };
    let rows = app.standing.iter().map(|s| {
        let (tone, says) = stands(app, s);
        Row::new(vec![
            Cell::from(Span::styled(
                words::scope_name(&s.scope).to_string(),
                Style::default().fg(t.ink),
            )),
            count(s.trained),
            count(s.accepted),
            count(s.proposed),
            Cell::from(Span::styled(
                s.last_trained.map_or_else(|| "-".to_string(), since),
                Style::default().fg(t.ink2),
            )),
            Cell::from(Line::from(vec![
                Span::styled(
                    format!("{} ", tone.glyph()),
                    Style::default().fg(t.tone(tone)),
                ),
                Span::styled(says, Style::default().fg(t.tone(tone))),
            ])),
        ])
    });
    let header = Row::new(
        [
            "EXPERT",
            "TRAINED",
            "ACCEPTED",
            "PROPOSED",
            "LAST TRAINED",
            "STATE",
        ]
        .into_iter()
        .map(Cell::from),
    )
    .style(Style::default().fg(t.ink2).add_modifier(Modifier::BOLD));
    let widths = [
        Constraint::Min(14),
        Constraint::Length(7),
        Constraint::Length(8),
        Constraint::Length(8),
        Constraint::Length(12),
        Constraint::Min(34),
    ];
    f.render_widget(Table::new(rows, widths).header(header), parts[0]);

    let hours = app.policy.accept_after_hours;
    let note = if hours > 0.0 {
        format!(
            "A proposed behavior that can be taught is accepted after {hours:.0} hours. K changes the wait."
        )
    } else {
        "Proposed behaviors wait for you to accept them. K sets a wait after which the server does."
            .to_string()
    };
    f.render_widget(
        Paragraph::new(Span::styled(note, Style::default().fg(t.ink3))),
        parts[1],
    );
}

/// The generational loop: the run under the cursor as a pipeline.
fn training_loop(f: &mut Frame, app: &App, area: Rect) {
    let t = app.theme();
    let block = panel(
        &t,
        Span::styled(" Training loop ", Style::default().fg(t.ink2)),
    );
    let inner = block.inner(area);
    f.render_widget(block, area);

    let Some(head) = app.selected_loop_head() else {
        let why = "No training loop has run yet: `antumbra train` runs one over a corpus, by hand. A standing expert needs none: the server trains it when a behavior of yours is accepted, and lists it here.";
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
