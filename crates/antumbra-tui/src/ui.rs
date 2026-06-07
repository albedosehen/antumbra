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
    match app.mode {
        Mode::Help => help_overlay(f, app),
        Mode::Palette => palette_overlay(f, app),
        Mode::Normal => {}
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
        lines.push(kv(&t, "fitness", &format!("{:.2}", e.fitness)));
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
            .enumerate()
            .map(|(i, s)| {
                let sel = focused && i == app.selected_shadow;
                let mut style = Style::default().fg(shadow_color(&t, s.status.as_str()));
                if sel {
                    style = style.add_modifier(Modifier::BOLD);
                }
                let glyph = match s.status.as_str() {
                    "graduated" => "✦",
                    "pruned" => "×",
                    _ => "◌",
                };
                Line::from(Span::styled(
                    format!(
                        "{}{} {} · {}",
                        if sel { "▸ " } else { "  " },
                        glyph,
                        s.id.as_str(),
                        s.status.as_str()
                    ),
                    style,
                ))
            })
            .collect()
    };
    let selected = if focused { app.selected_shadow } else { 0 };
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
            .enumerate()
            .map(|(i, b)| {
                let sel = focused && i == app.selected_boundary;
                let actionable = b.is_actionable();
                let mut style = Style::default().fg(if actionable { t.alert } else { t.dim });
                if sel {
                    style = style.add_modifier(Modifier::BOLD);
                }
                Line::from(Span::styled(
                    format!(
                        "{}{} {}",
                        if sel { "▸ " } else { "  " },
                        if actionable { "⛔" } else { "○" },
                        b.behavior
                    ),
                    style,
                ))
            })
            .collect()
    };
    let selected = if focused { app.selected_boundary } else { 0 };
    scroll::list(
        f,
        &t,
        area,
        panel_focused(&t, "antumbra · boundaries", focused),
        lines,
        selected,
    );
}

/// The command palette: a query line over a fuzzy-ranked command list (`:` opens).
fn palette_overlay(f: &mut Frame, app: &App) {
    let t = app.theme();
    let matches = app.palette_matches();
    let listed = matches.len().max(1) as u16;
    let area = overlay::centered(f.area(), 56, listed + 4);
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

    // Ranked matches, the selection marked.
    let selected = app.palette.selected.min(matches.len().saturating_sub(1));
    let lines: Vec<Line> = if matches.is_empty() {
        vec![Line::from(Span::styled(
            "  (no matching command)",
            Style::default().fg(t.dim),
        ))]
    } else {
        matches
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let sel = i == selected;
                let style = if sel {
                    Style::default().fg(t.accent).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(t.ink)
                };
                Line::from(Span::styled(
                    format!("{}{}", if sel { "▸ " } else { "  " }, c.label),
                    style,
                ))
            })
            .collect()
    };
    f.render_widget(Paragraph::new(lines), rows[1]);
}

/// The keybinding reference, a centred modal over the live view (`?` toggles).
fn help_overlay(f: &mut Frame, app: &App) {
    let t = app.theme();
    let area = overlay::centered(f.area(), 52, 21);
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
    // Live frame-rate readout (the cap is set in the footer with `±`) and the
    // active layout (cycled with `l`).
    spans.push(Span::styled(
        format!("  ·  {} fps  ·  {}", app.shown_fps(), app.layout.name()),
        Style::default().fg(t.dim),
    ));
    f.render_widget(Paragraph::new(Line::from(spans)).block(panel(&t, "")), area);
}

fn graph(f: &mut Frame, app: &App, area: Rect) {
    let t = app.theme();
    let block = panel(&t, Span::styled(" population ", Style::default().fg(t.ink)));
    // Ease the whole graph in over the first 0.8s (intro reveal).
    let intro = (app.clock_ms / 800.0).min(1.0);
    let link = rgb(t.core, 0.22);
    let mote = rgb(t.core, 0.85);
    let canvas = Canvas::default()
        .block(block)
        .marker(Marker::Braille)
        .x_bounds([-100.0, 100.0])
        .y_bounds([-100.0, 100.0])
        .paint(move |ctx| {
            let n = app.experts.len().max(1);
            let orbit_r = 64.0;
            let rot = app.clock_ms * 0.00006 * TAU;

            // Each expert: a link from the core with a travelling energy mote,
            // then the node (size by frozen/selected, colour by fitness/glow).
            for (i, e) in app.experts.iter().enumerate() {
                let ang = rot + (i as f64 / n as f64) * TAU;
                let (ex, ey) = (orbit_r * ang.cos(), orbit_r * ang.sin());
                ctx.draw(&CanvasLine {
                    x1: 0.0,
                    y1: 0.0,
                    x2: ex,
                    y2: ey,
                    color: link,
                });
                let tt = (app.clock_ms * 0.0006 + i as f64 * 0.37) % 1.0;
                ctx.print(
                    ex * tt,
                    ey * tt,
                    Span::styled("·", Style::default().fg(mote)),
                );

                let glow = (0.6 + 0.4 * pulse(app.clock_ms, 1500.0, i as f64)) * intro;
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
        lines.push(kv(&t, "fitness", &format!("{:.2}", e.fitness)));
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
        for (i, b) in app.boundaries.iter().enumerate() {
            let sel = i == app.selected_boundary;
            let actionable = b.is_actionable();
            let mut style = Style::default().fg(if actionable { t.alert } else { t.dim });
            if sel {
                style = style.add_modifier(Modifier::BOLD);
            }
            lines.push(Line::from(Span::styled(
                format!(
                    "{}{} {}",
                    if sel { "▸ " } else { "  " },
                    if actionable { "⛔" } else { "○" },
                    b.behavior
                ),
                style,
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
        app.selected_boundary,
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
        d.push(kv(&t, "confidence", &format!("{:.2}", b.confidence)));
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
        for (i, s) in app.shadows.iter().enumerate() {
            let sel = i == app.selected_shadow;
            let mut style = Style::default().fg(shadow_color(&t, s.status.as_str()));
            if sel {
                style = style.add_modifier(Modifier::BOLD);
            }
            let glyph = match s.status.as_str() {
                "graduated" => "✦",
                "pruned" => "×",
                _ => "◌",
            };
            lines.push(Line::from(Span::styled(
                format!(
                    "{}{} {} · g{} · {}",
                    if sel { "▸ " } else { "  " },
                    glyph,
                    s.id.as_str(),
                    s.generation.0,
                    s.status.as_str()
                ),
                style,
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
        app.selected_shadow,
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
        d.push(kv(&t, "final reward", &format!("{final_reward:.2}")));
        if !s.reward_curve.is_empty() {
            d.push(kv(&t, "reward", &sparkline(&s.reward_curve)));
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

/// A reward curve as block-character bars, each value in `[0,1]` mapped to one of
/// eight heights.
fn sparkline(curve: &[f32]) -> String {
    const BARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    curve
        .iter()
        .map(|&v| BARS[((v.clamp(0.0, 1.0) * 7.0).round() as usize).min(7)])
        .collect()
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
