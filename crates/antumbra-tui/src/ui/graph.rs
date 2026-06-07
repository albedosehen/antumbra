//! The animated population canvas: experts orbiting the tri-node core on a
//! tilted ring (depth from `sin(angle)`), with motion trails and energy motes.

use std::f64::consts::{FRAC_PI_2, TAU};

use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::Span;
use ratatui::widgets::canvas::{Canvas, Line as CanvasLine};
use ratatui::Frame;

use crate::app::App;
use crate::theme::rgb;

use super::{panel, pulse};

pub(super) fn graph(f: &mut Frame, app: &App, area: Rect) {
    let t = app.theme();
    let block = panel(&t, Span::styled(" population ", Style::default().fg(t.ink)));
    // Ease the whole graph in over the first 0.8s (intro reveal).
    let intro = (app.clock_ms / 800.0).min(1.0);
    let canvas = Canvas::default()
        .block(block)
        .marker(app.canvas_marker())
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
