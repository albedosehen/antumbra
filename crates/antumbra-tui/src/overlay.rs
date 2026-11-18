//! Modal overlays drawn on top of the console: the centered, cleared box the
//! help screen and command palette render into. Hand-rolled on ratatui core
//! (`Clear` + a centered `Rect`) to keep the dependency tree small.

use ratatui::layout::{Constraint, Flex, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::Line;
use ratatui::widgets::{Block, BorderType, Clear};
use ratatui::Frame;

use crate::theme::{Rgb, Theme};

/// How much of a colour is left once it is dimmed; the rest is the ground.
const KEPT: f64 = 0.4;

/// Pull a color most of the way toward `ground`, so dimmed chrome recedes.
fn dimmed(c: Color, ground: Rgb) -> Color {
    match c {
        Color::Rgb(r, g, b) => {
            let mix =
                |v: u8, target: f64| (f64::from(v) * KEPT + target * (1.0 - KEPT)).round() as u8;
            Color::Rgb(mix(r, ground.0), mix(g, ground.1), mix(b, ground.2))
        }
        other => other,
    }
}

/// Dim everything already drawn in `area` toward the theme's ground, so an
/// overlay drawn on top reads clearly against a recessed backdrop. A cell on
/// the ground itself stays on it, and one left to the terminal stays left.
pub fn dim_backdrop(f: &mut Frame, t: &Theme, area: Rect) {
    let ground = t.ground_rgb();
    let buf = f.buffer_mut();
    for y in area.top()..area.bottom() {
        for x in area.left()..area.right() {
            if let Some(cell) = buf.cell_mut((x, y)) {
                let (fg, bg) = (dimmed(cell.fg, ground), dimmed(cell.bg, ground));
                cell.set_fg(fg).set_bg(bg);
            }
        }
    }
}

/// A centered rectangle `width` x `height`, clamped to `area`.
pub fn centered(area: Rect, width: u16, height: u16) -> Rect {
    let [row] = Layout::vertical([Constraint::Length(height.min(area.height))])
        .flex(Flex::Center)
        .areas(area);
    let [cell] = Layout::horizontal([Constraint::Length(width.min(area.width))])
        .flex(Flex::Center)
        .areas(row);
    cell
}

/// Clear `area` and draw a titled modal box into it, returning the inner
/// content rectangle the caller fills. The box is raised from the page by its
/// fill, the theme's surface, and marked as what the keys now drive by its
/// accent border.
pub fn modal(f: &mut Frame, t: &Theme, area: Rect, title: &str) -> Rect {
    f.render_widget(Clear, area);
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(t.accent))
        .style(Style::default().bg(t.surface).fg(t.ink))
        .title(Line::from(format!(" {title} ")).style(Style::default().fg(t.ink2)));
    let inner = block.inner(area);
    f.render_widget(block, area);
    inner
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theme::{MIST, MONO, NIGHT};
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    #[test]
    fn dimming_settles_toward_the_themes_ground() -> anyhow::Result<()> {
        // On a light ground a dimmed ink grows paler, on a dark one darker.
        for theme in [NIGHT, MIST] {
            let Color::Rgb(before, ..) = theme.ink else {
                anyhow::bail!("{} has no RGB ink", theme.name);
            };
            let Color::Rgb(after, ..) = dimmed(theme.ink, theme.ground_rgb()) else {
                anyhow::bail!("a dimmed RGB colour is RGB");
            };
            let ground = theme.ground_rgb().0;
            assert!(
                (f64::from(after) - ground).abs() < (f64::from(before) - ground).abs(),
                "{}: {before} -> {after} toward {ground}",
                theme.name
            );
        }
        // The ground dims to itself, and a colour left to the terminal is left.
        assert_eq!(dimmed(NIGHT.ground, NIGHT.ground_rgb()), NIGHT.ground);
        assert_eq!(dimmed(Color::Reset, MONO.ground_rgb()), Color::Reset);
        Ok(())
    }

    #[test]
    fn a_modal_is_filled_with_the_surface() -> anyhow::Result<()> {
        for theme in [NIGHT, MIST, MONO] {
            let mut terminal = Terminal::new(TestBackend::new(40, 10))?;
            let mut inner = Rect::default();
            terminal.draw(|f| {
                inner = modal(f, &theme, Rect::new(4, 2, 30, 6), "title");
            })?;
            let buf = terminal.backend().buffer();
            for y in 2..8 {
                for x in 4..34 {
                    assert_eq!(buf[(x, y)].bg, theme.surface, "{} at {x},{y}", theme.name);
                }
            }
            assert_eq!(inner, Rect::new(5, 3, 28, 4));
        }
        Ok(())
    }
}
