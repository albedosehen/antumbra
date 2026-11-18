//! The population page's panels: an expert's detail and how tasks are routed,
//! the candidates in training, the boundaries, and all three at once.
//!
//! Each list and each detail is a function of its own, so a layout is only an
//! arrangement of them.

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Axis, Chart, Dataset, GraphType, Paragraph, Wrap};
use ratatui::Frame;

use crate::app::{App, Focus};
use crate::scroll;
use crate::theme::Tone;

use antumbra_core::behavior;

use super::{
    boundary_row, boundary_tone, clip, context_fields, expert_state, gauge_row, heading, kv,
    nothing_here, panel, panel_focused, shadow_row, shadow_tone, state_line, state_row, title,
};

/// The experts in front: the selected one's detail over how tasks are routed.
pub(super) fn detail(f: &mut Frame, app: &App, area: Rect) {
    let rows = Layout::vertical([Constraint::Min(0), Constraint::Length(7)]).split(area);
    expert_detail(f, app, rows[0]);
    routing(f, app, rows[1]);
}

/// The selected expert: what it is, how well it does and what it learned from.
fn expert_detail(f: &mut Frame, app: &App, area: Rect) {
    let t = app.theme();
    let mut lines: Vec<Line> = Vec::new();
    if let Some(e) = app.selected_expert() {
        lines.push(heading(&t, e.name.clone()));
        lines.push(gauge_row(&t, "fitness", e.fitness, t.accent));
        let (tone, state) = expert_state(e);
        lines.push(state_row(&t, "state", tone, state));
        lines.push(kv(&t, "generation", &e.generation.0.to_string()));
        lines.push(kv(&t, "base model", &e.base_model));
        if let Some(desc) = e
            .capability_card
            .get("description")
            .and_then(|v| v.as_str())
        {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                desc.to_string(),
                Style::default().fg(t.ink2),
            )));
        }
        if let Some(ex) = e
            .capability_card
            .get("exemplars")
            .and_then(|v| v.as_array())
        {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                format!("learned from {} examples", ex.len()),
                Style::default().fg(t.ink2),
            )));
        }
    } else {
        // Every token reads its workspace's experts, so an empty list is
        // empty.
        lines.push(Line::from(Span::styled(
            "No experts yet: `antumbra train` or `antumbra teach` trains one from verified work.",
            Style::default().fg(t.ink3),
        )));
    }
    f.render_widget(
        Paragraph::new(lines)
            .block(panel(&t, title(&t, "Expert")))
            .wrap(Wrap { trim: true }),
        area,
    );
}

/// How tasks are routed: whether the router is trained, when it hands a task
/// on, and the boundaries in force, which are the ones routing avoids.
fn routing(f: &mut Frame, app: &App, area: Rect) {
    let t = app.theme();
    let mut g: Vec<Line> = Vec::new();
    match &app.router {
        Some(r) => {
            g.push(kv(
                &t,
                "router",
                &format!("trained on {} experts", r.experts.len()),
            ));
            g.push(kv(
                &t,
                "hands on",
                &format!("tasks matching below {:.2}", r.floor),
            ));
        }
        None if app.operator || app.demo => g.push(kv(
            &t,
            "router",
            "not trained yet: a task goes to the nearest expert",
        )),
        // The router is the server's, which this token does not read.
        None => g.push(kv(&t, "router", "needs an operator's token")),
    }
    let in_force = app.boundaries.iter().filter(|b| b.is_actionable()).count();
    g.push(kv(
        &t,
        "boundaries",
        &format!("{} · {} in force", app.boundaries.len(), in_force),
    ));
    // Only the ones in force keep a task from an expert; one that is only
    // recorded, listed here, would read as doing so.
    for b in app.boundaries.iter().filter(|b| b.is_actionable()).take(3) {
        let feat = b.governing_features.first().cloned().unwrap_or_default();
        g.push(state_line(
            &t,
            Tone::Settled,
            format!("{} · {}", b.behavior, feat),
        ));
    }
    f.render_widget(
        Paragraph::new(g)
            .block(panel(&t, title(&t, "Routing")))
            .wrap(Wrap { trim: true }),
        area,
    );
}

/// Beside the boundaries' list: the selected one's detail, in the whole
/// column, so the case it was wrong in has room.
pub(super) fn boundary_side(f: &mut Frame, app: &App, area: Rect) {
    boundary_detail(f, app, area);
}

/// The boundaries, in force ● or recorded ○. `lit` marks the panel as the
/// one the keys drive, when several are drawn at once.
fn boundary_list(f: &mut Frame, app: &App, area: Rect, selecting: bool, lit: bool) {
    let t = app.theme();
    let name = format!("Boundaries · {}", app.boundaries.len());
    let block = if lit {
        panel_focused(&t, &name, true)
    } else {
        panel(&t, title(&t, &name))
    };
    if app.boundaries.is_empty() {
        let why =
            "No boundaries yet: one is recorded when an expert is found to be wrong somewhere.";
        f.render_widget(
            Paragraph::new(nothing_here(app, &t, why))
                .block(block)
                .wrap(Wrap { trim: true }),
            area,
        );
        return;
    }
    let lines: Vec<Line> = app.boundaries.iter().map(|b| boundary_row(&t, b)).collect();
    scroll::list(
        f,
        &t,
        area,
        block,
        lines,
        selecting.then_some(app.selected_boundary),
    );
}

/// The selected boundary: whether it is in force, what decides it, and the
/// case it was wrong in beside the nearby case it was right in.
fn boundary_detail(f: &mut Frame, app: &App, area: Rect) {
    let t = app.theme();
    let mut d: Vec<Line> = Vec::new();
    if let Some(b) = app.selected_boundary() {
        // Each row on one line, so none pushes the next out of the panel: the
        // room inside the border, and beside a row's key.
        let width = (area.width as usize).saturating_sub(2);
        let room = width.saturating_sub(11);
        d.push(heading(&t, clip(&b.behavior, width)));
        let status = if b.is_actionable() {
            "in force · routing avoids it"
        } else {
            "recorded · not yet in force"
        };
        d.push(state_row(&t, "status", boundary_tone(b), status));
        if !b.governing_features.is_empty() {
            d.push(kv(&t, "decided by", &b.governing_features.join(", ")));
        }
        if let Some(g) = b.grain {
            d.push(kv(&t, "applies to", &format!("{g:?}").to_lowercase()));
        }
        d.push(gauge_row(&t, "confidence", b.confidence, t.accent));
        if let Some(owner) = &b.owner {
            d.push(kv(&t, "private to", owner.as_str()));
        }
        match &b.near_ok_context {
            // The pair that puts it in force: wrong in one case, right in a
            // case beside it.
            Some(ok) => {
                d.push(kv(
                    &t,
                    "wrong when",
                    &clip(&b.fail_context.to_string(), room),
                ));
                d.push(kv(&t, "right when", &clip(&ok.to_string(), room)));
            }
            // One that is only recorded has the case it failed in. A
            // training's names the scope, the held-out task and the answer
            // given.
            None => {
                for (k, v) in context_fields(&b.fail_context) {
                    let flat = v.split_whitespace().collect::<Vec<_>>().join(" ");
                    d.push(kv(&t, &k, &clip(&flat, room)));
                }
            }
        }
    } else if !app.boundaries.is_empty() {
        d.push(Line::from(Span::styled(
            "Select a boundary with ↑↓.",
            Style::default().fg(t.ink3),
        )));
    }
    f.render_widget(
        Paragraph::new(d)
            .block(panel(&t, title(&t, "Boundary")))
            .wrap(Wrap { trim: true }),
        area,
    );
}

/// Beside the candidates' list: the selected one's detail over its reward as
/// a chart.
pub(super) fn candidate_side(f: &mut Frame, app: &App, area: Rect) {
    let rows = Layout::vertical([Constraint::Length(6), Constraint::Min(0)]).split(area);
    candidate_detail(f, app, rows[0]);
    let kept = app
        .selected_shadow()
        .is_some_and(|s| behavior::shadow_scope(&s.id).is_some());
    let curve = app
        .selected_shadow()
        .map(|s| s.reward_curve.as_slice())
        .unwrap_or(&[]);
    reward_chart(f, app, rows[1], curve, kept);
}

/// The candidates, newest first: graduated ●, under way ○, pruned ×.
fn candidate_list(f: &mut Frame, app: &App, area: Rect, selecting: bool, lit: bool) {
    let t = app.theme();
    let name = format!("Candidates · {}", app.shadows.len());
    let block = if lit {
        panel_focused(&t, &name, true)
    } else {
        panel(&t, title(&t, &name))
    };
    if app.shadows.is_empty() {
        let why = "No candidates yet: one appears while an expert is being trained.";
        f.render_widget(
            Paragraph::new(nothing_here(app, &t, why))
                .block(block)
                .wrap(Wrap { trim: true }),
            area,
        );
        return;
    }
    let lines: Vec<Line> = app.shadows.iter().map(|s| shadow_row(&t, s)).collect();
    scroll::list(
        f,
        &t,
        area,
        block,
        lines,
        selecting.then_some(app.selected_shadow),
    );
}

/// The selected candidate: its status, and where its reward got to.
fn candidate_detail(f: &mut Frame, app: &App, area: Rect) {
    let t = app.theme();
    let mut d: Vec<Line> = Vec::new();
    let scope = app
        .selected_shadow()
        .and_then(|s| behavior::shadow_scope(&s.id));
    if let Some(s) = app.selected_shadow() {
        let width = (area.width as usize).saturating_sub(2);
        let name = match scope {
            Some(scope) => format!("{scope} · a standing expert's training"),
            None => s.id.as_str().to_string(),
        };
        d.push(heading(&t, clip(&name, width)));
        d.push(state_row(
            &t,
            "status",
            shadow_tone(s.status.as_str()),
            s.status.as_str(),
        ));
        let last = s.reward_curve.last().copied().unwrap_or(0.0);
        if scope.is_some() {
            // The held-out pass rate before training and after: the base
            // model's, then the expert's. One recorded from an expert's card
            // has only the trained one, which the card keeps.
            let base = match s.reward_curve.as_slice() {
                [base, _, ..] => format!("{base:.2}"),
                _ => "not recorded".into(),
            };
            d.push(kv(&t, "base model", &base));
            d.push(gauge_row(&t, "trained", last, t.accent));
        } else {
            d.push(kv(&t, "generation", &s.generation.0.to_string()));
            d.push(gauge_row(&t, "reward", last, t.accent));
        }
    } else if !app.shadows.is_empty() {
        d.push(Line::from(Span::styled(
            "Select a candidate with ↑↓.",
            Style::default().fg(t.ink3),
        )));
    }
    f.render_widget(
        Paragraph::new(d)
            .block(panel(&t, title(&t, "Candidate")))
            .wrap(Wrap { trim: true }),
        area,
    );
}

/// The selected candidate's reward as a line chart over training steps: the
/// path that decides whether it graduates. A standing expert's training
/// (`kept`) reads two points, the base model's pass rate and the trained
/// expert's.
fn reward_chart(f: &mut Frame, app: &App, area: Rect, curve: &[f32], kept: bool) {
    let t = app.theme();
    let name = if kept {
        "Pass rate · base model → trained"
    } else {
        "Reward · step by step"
    };
    let block = panel(&t, title(&t, name));
    if curve.is_empty() {
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "No reward recorded yet.",
                Style::default().fg(t.ink3),
            )))
            .block(block),
            area,
        );
        return;
    }
    let data: Vec<(f64, f64)> = curve
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as f64, v as f64))
        .collect();
    let last = curve.len().saturating_sub(1).max(1) as f64;
    let datasets = vec![Dataset::default()
        .marker(app.canvas_marker())
        .graph_type(GraphType::Line)
        .style(Style::default().fg(t.accent))
        .data(&data)];
    let chart = Chart::new(datasets)
        .block(block)
        // A chart draws on a canvas, which resets its cells' background
        // unless told the page's.
        .style(Style::default().bg(t.ground))
        .x_axis(
            Axis::default()
                .style(Style::default().fg(t.line))
                .bounds([0.0, last]),
        )
        .y_axis(
            Axis::default()
                .style(Style::default().fg(t.ink3))
                .bounds([0.0, 1.0])
                .labels(["0.0", "1.0"]),
        );
    f.render_widget(chart, area);
}

/// All three lists at once, the one the keys drive given more room and an
/// accent border (the dashboard layout).
pub(super) fn dashboard(f: &mut Frame, app: &App, area: Rect) {
    let (experts, candidates, boundaries) = match app.focus {
        Focus::Experts => (46, 27, 27),
        Focus::Shadows => (27, 46, 27),
        Focus::Boundaries => (27, 27, 46),
    };
    let rows = Layout::vertical([
        Constraint::Percentage(experts),
        Constraint::Percentage(candidates),
        Constraint::Percentage(boundaries),
    ])
    .split(area);
    expert_summary(f, app, rows[0], app.focus == Focus::Experts);
    let lit = app.focus == Focus::Shadows;
    candidate_list(f, app, rows[1], lit, lit);
    let lit = app.focus == Focus::Boundaries;
    boundary_list(f, app, rows[2], lit, lit);
}

/// The selected expert in brief (the dashboard layout's first panel).
fn expert_summary(f: &mut Frame, app: &App, area: Rect, focused: bool) {
    let t = app.theme();
    let mut lines: Vec<Line> = Vec::new();
    if let Some(e) = app.selected_expert() {
        lines.push(heading(&t, e.name.clone()));
        lines.push(gauge_row(&t, "fitness", e.fitness, t.accent));
        let (tone, state) = expert_state(e);
        lines.push(state_row(&t, "state", tone, state));
        lines.push(kv(&t, "generation", &e.generation.0.to_string()));
    } else {
        lines.push(Line::from(Span::styled(
            "No experts yet.",
            Style::default().fg(t.ink3),
        )));
    }
    let name = format!("Experts · {}", app.experts.len());
    let block = if focused {
        panel_focused(&t, &name, true)
    } else {
        panel(&t, title(&t, &name))
    };
    f.render_widget(
        Paragraph::new(lines).block(block).wrap(Wrap { trim: true }),
        area,
    );
}
