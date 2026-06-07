//! Rendering: the living population graph (umbra orbiting the tri-node core —
//! umbra/penumbra/antumbra, three linked minds), with detail and gate panels.
//! Orbit, pulse, and link-energy are hand-rolled from the animation clock so the
//! motion is fully under control. Every colour tints from the active [`Theme`],
//! so cycling a theme (`t`) recolours the whole console.

use std::f64::consts::{FRAC_PI_2, TAU};

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span};
use ratatui::widgets::canvas::{Canvas, Line as CanvasLine};
use ratatui::widgets::{Block, BorderType, Paragraph, Wrap};
use ratatui::Frame;

use crate::app::{App, Focus, LayoutMode, Mode};
use crate::overlay;
use crate::scroll;
use crate::theme::{rgb, Theme};

pub fn render(f: &mut Frame, app: &App) {
    let rows = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(0),
        Constraint::Length(1),
    ])
    .split(f.area());
    header(f, app, rows[0]);
    body(f, app, rows[1]);
    footer(f, app, rows[2]);
    if app.mode != Mode::Normal {
        overlay::dim_backdrop(f, f.area());
        match app.mode {
            Mode::Help => help_overlay(f, app),
            Mode::Palette => palette_overlay(f, app),
            Mode::Normal => {}
        }
    }
}

/// The body between header and footer, arranged per the active layout: graph
/// beside one focused detail, graph beside all three regions, or graph alone.
fn body(f: &mut Frame, app: &App, area: Rect) {
    match app.layout {
        LayoutMode::Graph => graph(f, app, area),
        LayoutMode::Focused => {
            let cols = Layout::horizontal([Constraint::Percentage(64), Constraint::Percentage(36)])
                .split(area);
            graph(f, app, cols[0]);
            match app.focus {
                Focus::Experts => detail(f, app, cols[1]),
                Focus::Shadows => shadows(f, app, cols[1]),
                Focus::Boundaries => boundaries(f, app, cols[1]),
            }
        }
        LayoutMode::Dashboard => {
            let cols = Layout::horizontal([Constraint::Percentage(58), Constraint::Percentage(42)])
                .split(area);
            graph(f, app, cols[0]);
            dashboard(f, app, cols[1]);
        }
    }
}

/// All three regions stacked at once, the focused one given more room and an
/// accent border so the active panel reads at a glance (the dashboard layout).
fn dashboard(f: &mut Frame, app: &App, area: Rect) {
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

/// The modal box rectangle for the active overlay (the size source the open
/// animation also targets). `None` in the live view.
pub fn overlay_area(app: &App, frame: Rect) -> Option<Rect> {
    match app.mode {
        Mode::Help => Some(overlay::centered(frame, 52, 23)),
        Mode::Palette => {
            let listed = app.palette_matches().len().max(1) as u16;
            Some(overlay::centered(frame, 56, listed + 4))
        }
        Mode::Normal => None,
    }
}

/// The command palette: a query line over a fuzzy-ranked command list (`:` opens).
fn palette_overlay(f: &mut Frame, app: &App) {
    let t = app.theme();
    let matches = app.palette_matches();
    let Some(area) = overlay_area(app, f.area()) else {
        return;
    };
    let inner = overlay::modal(f, &t, area, "command");
    let rows = Layout::vertical([Constraint::Length(2), Constraint::Min(0)]).split(inner);

    // Query line with a block cursor.
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("› ", Style::default().fg(t.accent)),
            Span::styled(app.palette.query.clone(), Style::default().fg(t.text)),
            Span::styled("▏", Style::default().fg(t.accent)),
        ])),
        rows[0],
    );

    // Ranked matches as a list, the selection full-row highlighted.
    if matches.is_empty() {
        f.render_widget(
            Paragraph::new(Span::styled(
                "  (no matching command)",
                Style::default().fg(t.dim),
            )),
            rows[1],
        );
    } else {
        let lines: Vec<Line> = matches
            .iter()
            .map(|c| Line::from(Span::styled(c.label, Style::default().fg(t.ink))))
            .collect();
        let block = Block::default();
        scroll::list(f, &t, rows[1], block, lines, Some(app.palette.selected));
    }
}

/// The keybinding reference, a centred modal over the live view (`?` toggles).
fn help_overlay(f: &mut Frame, app: &App) {
    let t = app.theme();
    let Some(area) = overlay_area(app, f.area()) else {
        return;
    };
    let inner = overlay::modal(f, &t, area, "help");

    let group = |label: &str| {
        Line::from(Span::styled(
            label.to_string(),
            Style::default().fg(t.dim).add_modifier(Modifier::BOLD),
        ))
    };
    let bind = |keys: &str, desc: &str| {
        Line::from(vec![
            Span::styled(
                format!("  {keys:<8}"),
                Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
            ),
            Span::styled(desc.to_string(), Style::default().fg(t.ink)),
        ])
    };
    let lines = vec![
        group("navigate"),
        bind("↑↓ jk", "select in the focused list"),
        bind("g G", "jump to first / last  (home / end)"),
        bind("pgup/dn", "move by a page"),
        bind("tab", "switch focus: umbra / penumbra / antumbra"),
        Line::from(""),
        group("view"),
        bind("l", "cycle layout: focused / dashboard / graph"),
        bind("t", "cycle theme: shadow / ember / mono"),
        Line::from(""),
        group("frame rate"),
        bind("+ -", "pin the cap to a refresh rate"),
        bind("a", "follow the active monitor"),
        Line::from(""),
        group("command"),
        bind(":", "open the command palette"),
        bind("r", "reload from the store"),
        bind("? esc", "close this help"),
        bind("q", "quit"),
    ];
    f.render_widget(Paragraph::new(lines), inner);
}

/// `[0,1]` pulse from the animation clock.
fn pulse(clock_ms: f64, period_ms: f64, phase: f64) -> f64 {
    0.5 + 0.5 * ((clock_ms / period_ms * TAU) + phase).sin()
}

/// A rounded, dim-bordered panel block with a titled, inked header.
fn panel<'a>(t: &Theme, title: impl Into<Line<'a>>) -> Block<'a> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(t.dim))
        .title(title.into())
}

/// A panel whose border and title brighten to the accent when it holds focus —
/// the dashboard's active-panel indicator.
fn panel_focused<'a>(t: &Theme, title: &str, focused: bool) -> Block<'a> {
    let (border, marker) = if focused {
        (t.accent, "◆ ")
    } else {
        (t.dim, "")
    };
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(border))
        .title(Line::from(format!(" {marker}{title} ")).style(Style::default().fg(border)))
}

fn header(f: &mut Frame, app: &App, area: Rect) {
    let t = app.theme();
    // Three pulsing diamonds = the tri-node core, then the wordmark.
    let mut spans = Vec::new();
    for k in 0..3 {
        let g = 0.5 + 0.5 * pulse(app.clock_ms, 700.0, k as f64 * 1.3);
        spans.push(Span::styled("◆", Style::default().fg(rgb(t.core, g))));
    }
    spans.push(Span::styled(
        "  ANTUMBRA",
        Style::default().fg(t.text).add_modifier(Modifier::BOLD),
    ));
    let gen = app
        .experts
        .iter()
        .map(|e| e.generation.0)
        .max()
        .unwrap_or(0);
    spans.push(Span::styled(
        format!(
            "   self-improving substrate · {} umbra · gen {}",
            app.experts.len(),
            gen
        ),
        Style::default().fg(t.ink),
    ));
    // Live frame-rate readout (the cap is set in the footer with `±`; `idle`
    // marks the eased rate after a still spell) and the active layout (`l`).
    spans.push(Span::styled(
        format!(
            "  ·  {} fps{}  ·  {}",
            app.shown_fps(),
            if app.idle { " idle" } else { "" },
            app.layout.name()
        ),
        Style::default().fg(t.dim),
    ));
    f.render_widget(Paragraph::new(Line::from(spans)).block(panel(&t, "")), area);
}

fn graph(f: &mut Frame, app: &App, area: Rect) {
    let t = app.theme();
    let block = panel(&t, Span::styled(" population ", Style::default().fg(t.ink)));
    // Ease the whole graph in over the first 0.8s (intro reveal).
    let intro = (app.clock_ms / 800.0).min(1.0);
    let canvas = Canvas::default()
        .block(block)
        .marker(Marker::Braille)
        .x_bounds([-100.0, 100.0])
        .y_bounds([-100.0, 100.0])
        .paint(move |ctx| {
            let n = app.experts.len().max(1);
            // A tilted ring (squashed vertically) reads as a 3D orbit: depth is
            // sin(angle) — front nodes sit lower and glow brighter, back nodes
            // higher and dimmer.
            let (rx, ry) = (72.0, 34.0);
            let rot = app.clock_ms * 0.00006 * TAU;
            let angle = |i: usize| rot + (i as f64 / n as f64) * TAU;

            // Draw back-to-front so nearer nodes occlude farther ones.
            let mut order: Vec<usize> = (0..app.experts.len()).collect();
            order.sort_by(|&a, &b| {
                angle(a)
                    .sin()
                    .partial_cmp(&angle(b).sin())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            for &i in &order {
                let e = &app.experts[i];
                let ang = angle(i);
                let (ex, ey) = (rx * ang.cos(), ry * ang.sin());
                let front = 0.5 + 0.5 * ang.sin();

                // Link from the core (fainter to the back) and an energy mote
                // travelling out along it.
                ctx.draw(&CanvasLine {
                    x1: 0.0,
                    y1: 0.0,
                    x2: ex,
                    y2: ey,
                    color: rgb(t.core, 0.10 + 0.18 * front),
                });
                let tt = (app.clock_ms * 0.0006 + i as f64 * 0.37) % 1.0;
                ctx.print(
                    ex * tt,
                    ey * tt,
                    Span::styled("·", Style::default().fg(rgb(t.core, 0.4 + 0.5 * front))),
                );

                // A short fading trail behind the node along its orbit.
                for k in 1..=3 {
                    let a = ang - k as f64 * 0.05;
                    let fade = (0.34 - 0.09 * k as f64) * front * intro;
                    ctx.print(
                        rx * a.cos(),
                        ry * a.sin(),
                        Span::styled("·", Style::default().fg(t.fitness(e.fitness, fade))),
                    );
                }

                let glow = (0.4 + 0.6 * front)
                    * intro
                    * (0.75 + 0.25 * pulse(app.clock_ms, 1500.0, i as f64));
                let selected = i == app.selected;
                let glyph = if selected {
                    "◉"
                } else if e.is_frozen() {
                    "●"
                } else {
                    "○"
                };
                let col = if selected {
                    rgb(t.core, intro)
                } else {
                    t.fitness(e.fitness, glow)
                };
                ctx.print(ex, ey, Span::styled(glyph, Style::default().fg(col)));
                ctx.print(
                    ex + 4.0,
                    ey,
                    Span::styled(e.name.clone(), Style::default().fg(t.ink)),
                );
            }

            // The tri-node core: three linked minds, slowly counter-rotating.
            let core_r = 13.0;
            let core: Vec<(f64, f64)> = (0..3)
                .map(|k| {
                    let a = TAU * (k as f64) / 3.0 - FRAC_PI_2 - app.clock_ms * 0.00009 * TAU;
                    (core_r * a.cos(), core_r * a.sin())
                })
                .collect();
            for k in 0..3 {
                let (x1, y1) = core[k];
                let (x2, y2) = core[(k + 1) % 3];
                let g = pulse(app.clock_ms, 900.0, k as f64 * 2.0);
                ctx.draw(&CanvasLine {
                    x1,
                    y1,
                    x2,
                    y2,
                    color: rgb(t.core, 0.35 + 0.45 * g),
                });
            }
            for (k, (x, y)) in core.iter().enumerate() {
                let g = (0.55 + 0.45 * pulse(app.clock_ms, 700.0, k as f64 * 1.3)) * intro;
                ctx.print(
                    *x,
                    *y,
                    Span::styled("◆", Style::default().fg(rgb(t.core, g))),
                );
            }
        });
    f.render_widget(canvas, area);
}

fn detail(f: &mut Frame, app: &App, area: Rect) {
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
fn boundaries(f: &mut Frame, app: &App, area: Rect) {
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
fn shadows(f: &mut Frame, app: &App, area: Rect) {
    let t = app.theme();
    let rows = Layout::vertical([Constraint::Min(0), Constraint::Length(9)]).split(area);

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
        if !s.reward_curve.is_empty() {
            d.push(sparkline_row(&t, "reward", &s.reward_curve));
        }
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
}

/// Status colour: graduated = success, pruned = dim, in-flight = warning.
fn shadow_color(t: &Theme, status: &str) -> Color {
    match status {
        "graduated" => t.success,
        "pruned" => t.dim,
        _ => t.warning,
    }
}

fn footer(f: &mut Frame, app: &App, area: Rect) {
    let t = app.theme();
    let router = if app.router.is_some() {
        "learned"
    } else {
        "heuristic"
    };
    let focus = match app.focus {
        Focus::Experts => "umbra",
        Focus::Shadows => "penumbra",
        Focus::Boundaries => "antumbra",
    };
    let actionable = app.boundaries.iter().filter(|b| b.is_actionable()).count();
    let key = |s: &'static str| Span::styled(s, Style::default().fg(Color::Black).bg(t.ink));
    let lbl = |s: String| Span::styled(s, Style::default().fg(t.ink));
    let line = Line::from(vec![
        key(" q "),
        lbl(" quit  ".into()),
        key(" ↑↓ "),
        lbl(" select  ".into()),
        key(" tab "),
        lbl(format!(" focus:{focus}  ")),
        key(" t "),
        lbl(format!(" theme:{}  ", t.name)),
        key(" ± "),
        lbl(format!(
            " cap:{}{}  ",
            app.target_fps,
            if app.auto_fps { " auto" } else { "" }
        )),
        key(" r "),
        lbl(" reload  ".into()),
        key(" ? "),
        lbl(" help  ".into()),
        key(" : "),
        lbl(" palette  ".into()),
        Span::styled(
            format!(
                "   {} umbra · {} penumbra · {} antumbra ({actionable} actionable) · gate {router}",
                app.experts.len(),
                app.shadows.len(),
                app.boundaries.len()
            ),
            Style::default().fg(t.dim),
        ),
    ]);
    f.render_widget(Paragraph::new(line), area);
}

/// A bold, accent-coloured panel heading (selected item's name).
fn heading<'a>(t: &Theme, text: String) -> Line<'a> {
    Line::from(Span::styled(
        text,
        Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
    ))
}

fn kv<'a>(t: &Theme, k: &'a str, v: &str) -> Line<'a> {
    Line::from(vec![
        Span::styled(format!("{k:<11}"), Style::default().fg(t.dim)),
        Span::styled(v.to_string(), Style::default().fg(t.value)),
    ])
}

/// A horizontal gauge for a `[0,1]` value: `width` cells filled to eighth-of-a-
/// cell resolution in `color`, the remaining track dim.
fn gauge_spans<'a>(t: &Theme, value: f32, width: u16, color: Color) -> Vec<Span<'a>> {
    const EIGHTHS: [&str; 8] = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉"];
    let v = (value.clamp(0.0, 1.0) as f64) * width as f64;
    let mut full = v.floor() as usize;
    let mut rem = ((v - full as f64) * 8.0).round() as usize;
    if rem == 8 {
        full += 1;
        rem = 0;
    }
    let mut filled = "█".repeat(full);
    let cells = full + usize::from(rem > 0);
    if rem > 0 {
        filled.push_str(EIGHTHS[rem]);
    }
    let track = "─".repeat((width as usize).saturating_sub(cells));
    vec![
        Span::styled(filled, Style::default().fg(color)),
        Span::styled(track, Style::default().fg(t.dim)),
    ]
}

/// A `key  ▆▆▆▍──  0.62` row: a labelled gauge with its numeric value.
fn gauge_row<'a>(t: &Theme, k: &'a str, value: f32, color: Color) -> Line<'a> {
    let mut spans = vec![Span::styled(format!("{k:<11}"), Style::default().fg(t.dim))];
    spans.extend(gauge_spans(t, value, 12, color));
    spans.push(Span::styled(
        format!(" {value:.2}"),
        Style::default().fg(t.value),
    ));
    Line::from(spans)
}

/// A reward curve as block bars, each bar coloured along the fitness gradient by
/// its own value, so a rising (or collapsing) trajectory reads at a glance.
fn sparkline_row<'a>(t: &Theme, k: &'a str, curve: &[f32]) -> Line<'a> {
    const BARS: [&str; 8] = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];
    let mut spans = vec![Span::styled(format!("{k:<11}"), Style::default().fg(t.dim))];
    for &v in curve {
        let level = (v.clamp(0.0, 1.0) * 7.0).round() as usize;
        spans.push(Span::styled(
            BARS[level.min(7)].to_string(),
            Style::default().fg(t.fitness(v, 1.0)),
        ));
    }
    Line::from(spans)
}
