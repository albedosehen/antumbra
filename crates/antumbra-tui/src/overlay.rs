//! Modal overlays drawn on top of the console — the centred, cleared box the
//! help screen and command palette render into. Hand-rolled on ratatui core
//! (`Clear` + a centred `Rect`) to keep the dependency tree small.

use ratatui::layout::{Constraint, Flex, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::Line;
use ratatui::widgets::{Block, BorderType, Clear, Paragraph};
use ratatui::Frame;

use crate::theme::Theme;

/// The dark terminal background dimmed colours settle toward.
const BG: (f32, f32, f32) = (12.0, 14.0, 18.0);

/// Pull a colour most of the way toward the background, so dimmed chrome recedes.
fn dimmed(c: Color) -> Color {
    match c {
        Color::Rgb(r, g, b) => {
            let mix = |v: u8, target: f32| (v as f32 * 0.4 + target * 0.6) as u8;
            Color::Rgb(mix(r, BG.0), mix(g, BG.1), mix(b, BG.2))
        }
        other => other,
    }
}

/// Dim everything already drawn in `area` toward the background, so an overlay
/// drawn on top reads clearly against a recessed backdrop.
pub fn dim_backdrop(f: &mut Frame, area: Rect) {
    let buf = f.buffer_mut();
    for y in area.top()..area.bottom() {
        for x in area.left()..area.right() {
            if let Some(cell) = buf.cell_mut((x, y)) {
                let (fg, bg) = (dimmed(cell.fg), dimmed(cell.bg));
                cell.set_fg(fg).set_bg(bg);
            }
        }
    }
}

/// A centred rectangle `width` x `height`, clamped to `area`.
pub fn centered(area: Rect, width: u16, height: u16) -> Rect {
    let [row] = Layout::vertical([Constraint::Length(height.min(area.height))])
        .flex(Flex::Center)
        .areas(area);
    let [cell] = Layout::horizontal([Constraint::Length(width.min(area.width))])
        .flex(Flex::Center)
        .areas(row);
    cell
}

/// Clear `area` and draw a titled, accent-bordered modal box into it, returning
/// the inner content rectangle the caller fills.
pub fn modal(f: &mut Frame, t: &Theme, area: Rect, title: &str) -> Rect {
    f.render_widget(Clear, area);
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(t.accent))
        .title(Line::from(format!(" {title} ")).style(Style::default().fg(t.ink)));
    let inner = block.inner(area);
    f.render_widget(Paragraph::new("").block(block), area);
    inner
}
