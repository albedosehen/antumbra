//! Modal overlays drawn on top of the console — the centred, cleared box the
//! help screen and command palette render into. Hand-rolled on ratatui core
//! (`Clear` + a centred `Rect`) to keep the dependency tree small.

use ratatui::layout::{Constraint, Flex, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::Line;
use ratatui::widgets::{Block, BorderType, Clear, Paragraph};
use ratatui::Frame;

use crate::theme::Theme;

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
