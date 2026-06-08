//! The region detail panels: the expert detail + gate (umbra), the shadows
//! penumbra, the boundaries antumbra, and the all-at-once dashboard.

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Axis, Chart, Dataset, GraphType, Paragraph, Wrap};
use ratatui::Frame;

use crate::app::{App, Focus};
use crate::scroll;

use super::{gauge_row, heading, kv, panel, panel_focused, shadow_color};

/// The selected expert's detail plus the gate summary (the umbra focus).
pub(super) fn detail(f: &mut Frame, app: &App, area: Rect) {
    let t = app.theme();
    let rows = Layout::vertical([Constraint::Min(0), Constraint::Length(7)]).split(area);

    // Selected expert.
    let mut lines: Vec<Line> = Vec::new();
    if let Some(e) = app.selected_expert() {
        lines.push(heading(&t, e.name.clone()));
        lines.push(gauge_row(
            &t,
            "fitness",
            e.fitness,
            t.fitness(e.fitness, 1.0),
        ));
        lines.push(kv(&t, "frozen", if e.is_frozen() { "yes" } else { "no" }));
        lines.push(kv(&t, "generation", &e.generation.0.to_string()));
        lines.push(kv(&t, "base", &e.base_model));
        if let Some(desc) = e
            .capability_card
            .get("description")
            .and_then(|v| v.as_str())
        {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                desc.to_string(),
                Style::default().fg(t.ink),
            )));
        }
        if let Some(ex) = e
            .capability_card
            .get("exemplars")
            .and_then(|v| v.as_array())
        {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                format!("{} learned exemplars", ex.len()),
                Style::default().fg(t.dim),
            )));
        }
    } else {
        lines.push(Line::from(Span::styled(
            "(no experts yet — grow some with `antumbra train`/`teach`)",
            Style::default().fg(t.dim),
        )));
    }
    f.render_widget(
        Paragraph::new(lines)
            .block(panel(
                &t,
                Span::styled(" expert ", Style::default().fg(t.ink)),
            ))
            .wrap(Wrap { trim: true }),
        rows[0],
    );

    // Gate + boundaries.
    let mut g: Vec<Line> = Vec::new();
    match &app.router {
        Some(r) => g.push(kv(
            &t,
            "router",
            &format!(
                "learned · {} experts · OOD floor {:.2}",
                r.experts.len(),
                r.floor
            ),
        )),
        None => g.push(kv(&t, "router", "heuristic (untrained)")),
    }
    let actionable = app.boundaries.iter().filter(|b| b.is_actionable()).count();
    g.push(kv(
        &t,
        "boundaries",
        &format!("{} ({} actionable)", app.boundaries.len(), actionable),
    ));
    for b in app.boundaries.iter().take(3) {
        let feat = b.governing_features.first().cloned().unwrap_or_default();
        g.push(Line::from(Span::styled(
            format!("  ⛔ {} · {}", b.behavior, feat),
            Style::default().fg(t.alert),
        )));
    }
    f.render_widget(
        Paragraph::new(g)
            .block(panel(
                &t,
                Span::styled(" gate ", Style::default().fg(t.ink)),
            ))
            .wrap(Wrap { trim: true }),
        rows[1],
    );
}

/// The boundaries inspector (the antumbra, ADR-0004): the learned scopes, with
/// the selected one's detail — actionable vs open, governing feature, grain,
/// confidence, and the C -> C' contrast it was recovered from.
pub(super) fn boundaries(f: &mut Frame, app: &App, area: Rect) {
    let t = app.theme();
    let rows = Layout::vertical([Constraint::Min(0), Constraint::Length(9)]).split(area);

    let mut lines: Vec<Line> = Vec::new();
    if app.boundaries.is_empty() {
        lines.push(Line::from(Span::styled(
            "(no boundaries yet — the antumbra is empty)",
            Style::default().fg(t.dim),
        )));
    } else {
        for b in &app.boundaries {
            let actionable = b.is_actionable();
            lines.push(Line::from(Span::styled(
                format!("{} {}", if actionable { "⛔" } else { "○" }, b.behavior),
                Style::default().fg(if actionable { t.alert } else { t.dim }),
            )));
        }
    }
    scroll::list(
        f,
        &t,
        rows[0],
        panel(
            &t,
            Span::styled(
                format!(" boundaries · {} ", app.boundaries.len()),
                Style::default().fg(t.ink),
            ),
        ),
        lines,
        (!app.boundaries.is_empty()).then_some(app.selected_boundary),
    );

    let mut d: Vec<Line> = Vec::new();
    if let Some(b) = app.selected_boundary() {
        d.push(heading(&t, b.behavior.clone()));
        let (status, sc) = if b.is_actionable() {
            ("actionable · gates routing", t.alert)
        } else {
            ("open · recorded, inert", t.dim)
        };
        d.push(Line::from(vec![
            Span::styled(format!("{:<11}", "status"), Style::default().fg(t.dim)),
            Span::styled(status.to_string(), Style::default().fg(sc)),
        ]));
        let feat = if b.governing_features.is_empty() {
            "-".to_string()
        } else {
            b.governing_features.join(", ")
        };
        d.push(kv(&t, "feature", &feat));
        d.push(kv(
            &t,
            "grain",
            &b.grain
                .map(|g| format!("{g:?}"))
                .unwrap_or_else(|| "-".into()),
        ));
        d.push(gauge_row(&t, "confidence", b.confidence, t.accent));
        // The contrastive pair that makes it actionable: incorrect in C, fine in C'.
        if let Some(ok) = &b.near_ok_context {
            d.push(kv(&t, "incorrect", &b.fail_context.to_string()));
            d.push(kv(&t, "acceptable", &ok.to_string()));
        }
    } else {
        d.push(Line::from(Span::styled(
            "(select a boundary with ↑↓)",
            Style::default().fg(t.dim),
        )));
    }
    f.render_widget(
        Paragraph::new(d)
            .block(panel(
                &t,
                Span::styled(" scope ", Style::default().fg(t.ink)),
            ))
            .wrap(Wrap { trim: true }),
        rows[1],
    );
}

/// The penumbra: shadows in (or recently out of) training, newest first, with the
/// selected one's lineage — status, generation, final reward, and its reward curve
/// as a sparkline (the anti-collapse signal, ADR-0002/0003).
pub(super) fn shadows(f: &mut Frame, app: &App, area: Rect) {
    let t = app.theme();
    let rows = Layout::vertical([
        Constraint::Min(0),
        Constraint::Length(6),
        Constraint::Length(9),
    ])
    .split(area);

    let mut lines: Vec<Line> = Vec::new();
    if app.shadows.is_empty() {
        lines.push(Line::from(Span::styled(
            "(no shadows yet — the penumbra is quiet)",
            Style::default().fg(t.dim),
        )));
    } else {
        for s in &app.shadows {
            let glyph = match s.status.as_str() {
                "graduated" => "✦",
                "pruned" => "×",
                _ => "◌",
            };
            lines.push(Line::from(Span::styled(
                format!(
                    "{} {} · g{} · {}",
                    glyph,
                    s.id.as_str(),
                    s.generation.0,
                    s.status.as_str()
                ),
                Style::default().fg(shadow_color(&t, s.status.as_str())),
            )));
        }
    }
    scroll::list(
        f,
        &t,
        rows[0],
        panel(
            &t,
            Span::styled(
                format!(" shadows · {} ", app.shadows.len()),
                Style::default().fg(t.ink),
            ),
        ),
        lines,
        (!app.shadows.is_empty()).then_some(app.selected_shadow),
    );

    let mut d: Vec<Line> = Vec::new();
    if let Some(s) = app.selected_shadow() {
        d.push(heading(&t, s.id.as_str().to_string()));
        d.push(Line::from(vec![
            Span::styled(format!("{:<11}", "status"), Style::default().fg(t.dim)),
            Span::styled(
                s.status.as_str().to_string(),
                Style::default().fg(shadow_color(&t, s.status.as_str())),
            ),
        ]));
        d.push(kv(&t, "generation", &s.generation.0.to_string()));
        let final_reward = s.reward_curve.last().copied().unwrap_or(0.0);
        d.push(gauge_row(
            &t,
            "final reward",
            final_reward,
            t.fitness(final_reward, 1.0),
        ));
    } else {
        d.push(Line::from(Span::styled(
            "(select a shadow with ↑↓)",
            Style::default().fg(t.dim),
        )));
    }
    f.render_widget(
        Paragraph::new(d)
            .block(panel(
                &t,
                Span::styled(" training ", Style::default().fg(t.ink)),
            ))
            .wrap(Wrap { trim: true }),
        rows[1],
    );

    let curve = app
        .selected_shadow()
        .map(|s| s.reward_curve.as_slice())
        .unwrap_or(&[]);
    reward_chart(f, app, rows[2], curve);
}

/// The selected shadow's reward curve as a line chart over training steps — the
/// trajectory that decides graduation vs collapse.
fn reward_chart(f: &mut Frame, app: &App, area: Rect, curve: &[f32]) {
    let t = app.theme();
    let block = panel(
        &t,
        Span::styled(" reward · over steps ", Style::default().fg(t.ink)),
    );
    if curve.is_empty() {
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "(no reward signal yet)",
                Style::default().fg(t.dim),
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
        .x_axis(
            Axis::default()
                .style(Style::default().fg(t.dim))
                .bounds([0.0, last]),
        )
        .y_axis(
            Axis::default()
                .style(Style::default().fg(t.dim))
                .bounds([0.0, 1.0])
                .labels(["0.0", "1.0"]),
        );
    f.render_widget(chart, area);
}

/// All three regions stacked at once, the focused one given more room and an
/// accent border so the active panel reads at a glance (the dashboard layout).
pub(super) fn dashboard(f: &mut Frame, app: &App, area: Rect) {
    let (u, p, a) = match app.focus {
        Focus::Experts => (46, 27, 27),
        Focus::Shadows => (27, 46, 27),
        Focus::Boundaries => (27, 27, 46),
    };
    let rows = Layout::vertical([
        Constraint::Percentage(u),
        Constraint::Percentage(p),
        Constraint::Percentage(a),
    ])
    .split(area);
    umbra_panel(f, app, rows[0], app.focus == Focus::Experts);
    penumbra_panel(f, app, rows[1], app.focus == Focus::Shadows);
    antumbra_panel(f, app, rows[2], app.focus == Focus::Boundaries);
}

/// Compact selected-expert summary (the dashboard's umbra panel).
fn umbra_panel(f: &mut Frame, app: &App, area: Rect, focused: bool) {
    let t = app.theme();
    let mut lines: Vec<Line> = Vec::new();
    if let Some(e) = app.selected_expert() {
        lines.push(heading(&t, e.name.clone()));
        lines.push(gauge_row(
            &t,
            "fitness",
            e.fitness,
            t.fitness(e.fitness, 1.0),
        ));
        lines.push(kv(&t, "frozen", if e.is_frozen() { "yes" } else { "no" }));
        lines.push(kv(&t, "generation", &e.generation.0.to_string()));
    } else {
        lines.push(Line::from(Span::styled(
            "(no experts yet)",
            Style::default().fg(t.dim),
        )));
    }
    f.render_widget(
        Paragraph::new(lines)
            .block(panel_focused(&t, "umbra · experts", focused))
            .wrap(Wrap { trim: true }),
        area,
    );
}

/// Compact shadows list (the dashboard's penumbra panel).
fn penumbra_panel(f: &mut Frame, app: &App, area: Rect, focused: bool) {
    let t = app.theme();
    let lines: Vec<Line> = if app.shadows.is_empty() {
        vec![Line::from(Span::styled(
            "(penumbra quiet)",
            Style::default().fg(t.dim),
        ))]
    } else {
        app.shadows
            .iter()
            .map(|s| {
                let glyph = match s.status.as_str() {
                    "graduated" => "✦",
                    "pruned" => "×",
                    _ => "◌",
                };
                Line::from(Span::styled(
                    format!("{} {} · {}", glyph, s.id.as_str(), s.status.as_str()),
                    Style::default().fg(shadow_color(&t, s.status.as_str())),
                ))
            })
            .collect()
    };
    let selected = (focused && !app.shadows.is_empty()).then_some(app.selected_shadow);
    scroll::list(
        f,
        &t,
        area,
        panel_focused(&t, "penumbra · shadows", focused),
        lines,
        selected,
    );
}

/// Compact boundaries list (the dashboard's antumbra panel).
fn antumbra_panel(f: &mut Frame, app: &App, area: Rect, focused: bool) {
    let t = app.theme();
    let lines: Vec<Line> = if app.boundaries.is_empty() {
        vec![Line::from(Span::styled(
            "(antumbra empty)",
            Style::default().fg(t.dim),
        ))]
    } else {
        app.boundaries
            .iter()
            .map(|b| {
                let actionable = b.is_actionable();
                Line::from(Span::styled(
                    format!("{} {}", if actionable { "⛔" } else { "○" }, b.behavior),
                    Style::default().fg(if actionable { t.alert } else { t.dim }),
                ))
            })
            .collect()
    };
    let selected = (focused && !app.boundaries.is_empty()).then_some(app.selected_boundary);
    scroll::list(
        f,
        &t,
        area,
        panel_focused(&t, "antumbra · boundaries", focused),
        lines,
        selected,
    );
}
