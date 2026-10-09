//! The reward landscape: a density heatmap of the whole penumbra's trajectories
//! at a glance. Each shadow is a row, training steps run left→right, and the
//! reward at each step maps to a shade glyph + the fitness color ramp. Drawn
//! with shade characters (not just background color) so it reads in a text
//! snapshot and in a monochrome terminal, then tints by the same fitness ramp as
//! the rest of the console, the universal path. (Raster fidelity, where a
//! terminal supports it, is the `raster`-feature enhancement; see `render.rs`.)

use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::app::App;

use super::panel;

/// Shade ramp from cold/empty to hot/full reward.
const RAMP: [char; 5] = ['·', '░', '▒', '▓', '█'];

/// The shade glyph for a `[0,1]` value (reused by the gate weight profile).
pub(super) fn heat_cell(v: f32) -> char {
    let idx = (v.clamp(0.0, 1.0) * (RAMP.len() - 1) as f32).round() as usize;
    RAMP[idx.min(RAMP.len() - 1)]
}

pub(super) fn reward_heatmap(f: &mut Frame, app: &App, area: Rect) {
    let t = app.theme();
    let block = panel(
        &t,
        Span::styled(
            " reward landscape · shadows × steps ",
            Style::default().fg(t.ink),
        ),
    );
    let inner = block.inner(area);
    f.render_widget(block, area);

    let w = inner.width as usize;
    if app.shadows.is_empty() || w == 0 {
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "(no shadows, the penumbra is quiet)",
                Style::default().fg(t.dim),
            ))),
            inner,
        );
        return;
    }

    let mut lines: Vec<Line> = Vec::new();
    for s in app.shadows.iter().take(inner.height as usize) {
        let curve = &s.reward_curve;
        if curve.is_empty() {
            lines.push(Line::from(Span::styled(
                "·".repeat(w),
                Style::default().fg(t.dim),
            )));
            continue;
        }
        let spans: Vec<Span> = (0..w)
            .map(|x| {
                // Stretch the curve across the row (nearest step per column).
                let pos = if w <= 1 {
                    0
                } else {
                    x * (curve.len() - 1) / (w - 1)
                };
                let v = curve[pos.min(curve.len() - 1)];
                Span::styled(
                    heat_cell(v).to_string(),
                    Style::default().fg(t.fitness(v, 1.0)),
                )
            })
            .collect();
        lines.push(Line::from(spans));
    }
    f.render_widget(Paragraph::new(lines), inner);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heat_cell_maps_reward_to_a_shade() {
        assert_eq!(heat_cell(0.0), '·');
        assert_eq!(heat_cell(0.5), '▒');
        assert_eq!(heat_cell(1.0), '█');
        // Clamps out-of-range without panicking.
        assert_eq!(heat_cell(2.0), '█');
        assert_eq!(heat_cell(-1.0), '·');
    }
}
