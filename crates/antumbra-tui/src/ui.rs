//! Rendering: the living population graph (umbra orbiting the tri-node core —
//! umbra/penumbra/antumbra, three linked minds), with detail and gate panels.
//! Orbit, pulse, and link-energy are hand-rolled from the animation clock so
//! the motion is fully under control; tachyonfx layers the intro reveal.

use std::f64::consts::{FRAC_PI_2, TAU};

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span};
use ratatui::widgets::canvas::{Canvas, Line as CanvasLine};
use ratatui::widgets::{Block, BorderType, Paragraph, Wrap};
use ratatui::Frame;

use crate::app::{App, Focus};

const INK: Color = Color::Rgb(120, 140, 160);
const DIM: Color = Color::Rgb(70, 90, 110);
const ALERT: Color = Color::Rgb(200, 90, 90);

pub fn render(f: &mut Frame, app: &App) {
    let rows = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(0),
        Constraint::Length(1),
    ])
    .split(f.area());
    header(f, app, rows[0]);
    let body =
        Layout::horizontal([Constraint::Percentage(64), Constraint::Percentage(36)]).split(rows[1]);
    graph(f, app, body[0]);
    match app.focus {
        Focus::Experts => detail(f, app, body[1]),
        Focus::Shadows => shadows(f, app, body[1]),
        Focus::Boundaries => boundaries(f, app, body[1]),
    }
    footer(f, app, rows[2]);
}

/// `[0,1]` pulse from the animation clock.
fn pulse(clock_ms: f64, period_ms: f64, phase: f64) -> f64 {
    0.5 + 0.5 * ((clock_ms / period_ms * TAU) + phase).sin()
}

/// Green (low fitness) -> bright cyan (high), scaled by a glow factor.
fn fitness_color(fitness: f32, glow: f64) -> Color {
    let f = fitness.clamp(0.0, 1.0) as f64;
    let r = ((30.0 + 90.0 * f) * glow).clamp(0.0, 255.0);
    let g = ((150.0 + 90.0 * f) * glow).clamp(0.0, 255.0);
    let b = ((110.0 + 145.0 * f) * glow).clamp(0.0, 255.0);
    Color::Rgb(r as u8, g as u8, b as u8)
}

fn header(f: &mut Frame, app: &App, area: Rect) {
    // Three pulsing diamonds = the tri-node core, then the wordmark.
    let mut spans = Vec::new();
    for k in 0..3 {
        let g = 0.5 + 0.5 * pulse(app.clock_ms, 700.0, k as f64 * 1.3);
        let c = Color::Rgb((120.0 * g) as u8, (220.0 * g) as u8, 255);
        spans.push(Span::styled("◆", Style::default().fg(c)));
    }
    spans.push(Span::styled(
        "  ANTUMBRA",
        Style::default()
            .fg(Color::Rgb(210, 230, 245))
            .add_modifier(Modifier::BOLD),
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
        Style::default().fg(INK),
    ));
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(DIM));
    f.render_widget(Paragraph::new(Line::from(spans)).block(block), area);
}

fn graph(f: &mut Frame, app: &App, area: Rect) {
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(DIM))
        .title(Span::styled(" population ", Style::default().fg(INK)));
    // Ease the whole graph in over the first 0.8s (intro reveal).
    let intro = (app.clock_ms / 800.0).min(1.0);
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
                    color: Color::Rgb(28, 50, 66),
                });
                let t = (app.clock_ms * 0.0006 + i as f64 * 0.37) % 1.0;
                ctx.print(
                    ex * t,
                    ey * t,
                    Span::styled("·", Style::default().fg(Color::Rgb(80, 200, 255))),
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
                    Color::Rgb((150.0 + 80.0 * glow) as u8, (90.0 * glow) as u8, 255)
                } else {
                    fitness_color(e.fitness, glow)
                };
                ctx.print(ex, ey, Span::styled(glyph, Style::default().fg(col)));
                ctx.print(
                    ex + 4.0,
                    ey,
                    Span::styled(e.name.clone(), Style::default().fg(INK)),
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
                    color: Color::Rgb((50.0 + 110.0 * g) as u8, (120.0 + 100.0 * g) as u8, 230),
                });
            }
            for (k, (x, y)) in core.iter().enumerate() {
                let g = (0.55 + 0.45 * pulse(app.clock_ms, 700.0, k as f64 * 1.3)) * intro;
                ctx.print(
                    *x,
                    *y,
                    Span::styled(
                        "◆",
                        Style::default().fg(Color::Rgb((120.0 * g) as u8, (220.0 * g) as u8, 255)),
                    ),
                );
            }
        });
    f.render_widget(canvas, area);
}

fn detail(f: &mut Frame, app: &App, area: Rect) {
    let rows = Layout::vertical([Constraint::Min(0), Constraint::Length(7)]).split(area);

    // Selected expert.
    let mut lines: Vec<Line> = Vec::new();
    if let Some(e) = app.selected_expert() {
        lines.push(Line::from(Span::styled(
            e.name.clone(),
            Style::default()
                .fg(Color::Rgb(230, 200, 255))
                .add_modifier(Modifier::BOLD),
        )));
        lines.push(kv("fitness", &format!("{:.2}", e.fitness)));
        lines.push(kv("frozen", if e.is_frozen() { "yes" } else { "no" }));
        lines.push(kv("generation", &e.generation.0.to_string()));
        lines.push(kv("base", &e.base_model));
        if let Some(desc) = e
            .capability_card
            .get("description")
            .and_then(|v| v.as_str())
        {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                desc.to_string(),
                Style::default().fg(INK),
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
                Style::default().fg(DIM),
            )));
        }
    } else {
        lines.push(Line::from(Span::styled(
            "(no experts yet — grow some with `antumbra train`/`teach`)",
            Style::default().fg(DIM),
        )));
    }
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(DIM))
        .title(Span::styled(" expert ", Style::default().fg(INK)));
    f.render_widget(
        Paragraph::new(lines).block(block).wrap(Wrap { trim: true }),
        rows[0],
    );

    // Gate + boundaries.
    let mut g: Vec<Line> = Vec::new();
    match &app.router {
        Some(r) => g.push(kv(
            "router",
            &format!(
                "learned · {} experts · OOD floor {:.2}",
                r.experts.len(),
                r.floor
            ),
        )),
        None => g.push(kv("router", "heuristic (untrained)")),
    }
    let actionable = app.boundaries.iter().filter(|b| b.is_actionable()).count();
    g.push(kv(
        "boundaries",
        &format!("{} ({} actionable)", app.boundaries.len(), actionable),
    ));
    for b in app.boundaries.iter().take(3) {
        let feat = b.governing_features.first().cloned().unwrap_or_default();
        g.push(Line::from(Span::styled(
            format!("  ⛔ {} · {}", b.behavior, feat),
            Style::default().fg(Color::Rgb(200, 90, 90)),
        )));
    }
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(DIM))
        .title(Span::styled(" gate ", Style::default().fg(INK)));
    f.render_widget(
        Paragraph::new(g).block(block).wrap(Wrap { trim: true }),
        rows[1],
    );
}

/// The boundaries inspector (the antumbra, ADR-0004): the learned scopes, with
/// the selected one's detail — actionable vs open, governing feature, grain,
/// confidence, and the C -> C' contrast it was recovered from.
fn boundaries(f: &mut Frame, app: &App, area: Rect) {
    let rows = Layout::vertical([Constraint::Min(0), Constraint::Length(9)]).split(area);

    let mut lines: Vec<Line> = Vec::new();
    if app.boundaries.is_empty() {
        lines.push(Line::from(Span::styled(
            "(no boundaries yet — the antumbra is empty)",
            Style::default().fg(DIM),
        )));
    } else {
        for (i, b) in app.boundaries.iter().enumerate() {
            let sel = i == app.selected_boundary;
            let actionable = b.is_actionable();
            let mut style = Style::default().fg(if actionable { ALERT } else { DIM });
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
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(DIM))
        .title(Span::styled(
            format!(" boundaries · {} ", app.boundaries.len()),
            Style::default().fg(INK),
        ));
    f.render_widget(
        Paragraph::new(lines).block(block).wrap(Wrap { trim: true }),
        rows[0],
    );

    let mut d: Vec<Line> = Vec::new();
    if let Some(b) = app.selected_boundary() {
        d.push(Line::from(Span::styled(
            b.behavior.clone(),
            Style::default()
                .fg(Color::Rgb(230, 200, 255))
                .add_modifier(Modifier::BOLD),
        )));
        let (status, sc) = if b.is_actionable() {
            ("actionable · gates routing", ALERT)
        } else {
            ("open · recorded, inert", DIM)
        };
        d.push(Line::from(vec![
            Span::styled(format!("{:<11}", "status"), Style::default().fg(DIM)),
            Span::styled(status.to_string(), Style::default().fg(sc)),
        ]));
        let feat = if b.governing_features.is_empty() {
            "-".to_string()
        } else {
            b.governing_features.join(", ")
        };
        d.push(kv("feature", &feat));
        d.push(kv(
            "grain",
            &b.grain
                .map(|g| format!("{g:?}"))
                .unwrap_or_else(|| "-".into()),
        ));
        d.push(kv("confidence", &format!("{:.2}", b.confidence)));
        // The contrastive pair that makes it actionable: incorrect in C, fine in C'.
        if let Some(ok) = &b.near_ok_context {
            d.push(kv("incorrect", &b.fail_context.to_string()));
            d.push(kv("acceptable", &ok.to_string()));
        }
    } else {
        d.push(Line::from(Span::styled(
            "(select a boundary with ↑↓)",
            Style::default().fg(DIM),
        )));
    }
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(DIM))
        .title(Span::styled(" scope ", Style::default().fg(INK)));
    f.render_widget(
        Paragraph::new(d).block(block).wrap(Wrap { trim: true }),
        rows[1],
    );
}

/// The penumbra: shadows in (or recently out of) training, newest first, with the
/// selected one's lineage — status, generation, final reward, and its reward curve
/// as a sparkline (the anti-collapse signal, ADR-0002/0003).
fn shadows(f: &mut Frame, app: &App, area: Rect) {
    let rows = Layout::vertical([Constraint::Min(0), Constraint::Length(9)]).split(area);

    let mut lines: Vec<Line> = Vec::new();
    if app.shadows.is_empty() {
        lines.push(Line::from(Span::styled(
            "(no shadows yet — the penumbra is quiet)",
            Style::default().fg(DIM),
        )));
    } else {
        for (i, s) in app.shadows.iter().enumerate() {
            let sel = i == app.selected_shadow;
            let mut style = Style::default().fg(shadow_color(s.status.as_str()));
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
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(DIM))
        .title(Span::styled(
            format!(" shadows · {} ", app.shadows.len()),
            Style::default().fg(INK),
        ));
    f.render_widget(
        Paragraph::new(lines).block(block).wrap(Wrap { trim: true }),
        rows[0],
    );

    let mut d: Vec<Line> = Vec::new();
    if let Some(s) = app.selected_shadow() {
        d.push(Line::from(Span::styled(
            s.id.as_str().to_string(),
            Style::default()
                .fg(Color::Rgb(230, 200, 255))
                .add_modifier(Modifier::BOLD),
        )));
        d.push(Line::from(vec![
            Span::styled(format!("{:<11}", "status"), Style::default().fg(DIM)),
            Span::styled(
                s.status.as_str().to_string(),
                Style::default().fg(shadow_color(s.status.as_str())),
            ),
        ]));
        d.push(kv("generation", &s.generation.0.to_string()));
        let final_reward = s.reward_curve.last().copied().unwrap_or(0.0);
        d.push(kv("final reward", &format!("{final_reward:.2}")));
        if !s.reward_curve.is_empty() {
            d.push(kv("reward", &sparkline(&s.reward_curve)));
        }
    } else {
        d.push(Line::from(Span::styled(
            "(select a shadow with ↑↓)",
            Style::default().fg(DIM),
        )));
    }
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(DIM))
        .title(Span::styled(" training ", Style::default().fg(INK)));
    f.render_widget(
        Paragraph::new(d).block(block).wrap(Wrap { trim: true }),
        rows[1],
    );
}

/// Status colour: graduated = alive cyan-green, pruned = dim, in-flight = amber.
fn shadow_color(status: &str) -> Color {
    match status {
        "graduated" => Color::Rgb(90, 200, 150),
        "pruned" => DIM,
        _ => Color::Rgb(210, 190, 90),
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
    let line = Line::from(vec![
        Span::styled(" q ", Style::default().fg(Color::Black).bg(INK)),
        Span::styled(" quit  ", Style::default().fg(INK)),
        Span::styled(" ↑↓ ", Style::default().fg(Color::Black).bg(INK)),
        Span::styled(" select  ", Style::default().fg(INK)),
        Span::styled(" tab ", Style::default().fg(Color::Black).bg(INK)),
        Span::styled(format!(" focus:{focus}  "), Style::default().fg(INK)),
        Span::styled(" r ", Style::default().fg(Color::Black).bg(INK)),
        Span::styled(" reload  ", Style::default().fg(INK)),
        Span::styled(
            format!(
                "   {} umbra · {} penumbra · {} antumbra ({actionable} actionable) · gate {router}",
                app.experts.len(),
                app.shadows.len(),
                app.boundaries.len()
            ),
            Style::default().fg(DIM),
        ),
    ]);
    f.render_widget(Paragraph::new(line), area);
}

fn kv<'a>(k: &'a str, v: &str) -> Line<'a> {
    Line::from(vec![
        Span::styled(format!("{k:<11}"), Style::default().fg(DIM)),
        Span::styled(
            v.to_string(),
            Style::default().fg(Color::Rgb(180, 200, 215)),
        ),
    ])
}
