//! Scrolling for the one-row-per-item lists (shadows, boundaries): a
//! selection-following viewport — the selected row stays centred once a list
//! outgrows its panel — plus a slim scrollbar for a sense of position.
//! Hand-rolled on ratatui core.

use ratatui::layout::{Margin, Rect};
use ratatui::style::Style;
use ratatui::text::Line;
use ratatui::widgets::{Block, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState};
use ratatui::Frame;

use crate::theme::Theme;

/// The `[start, end)` slice of `total` rows to show in `height` rows, keeping the
/// `selected` row visible (centred once the list overflows).
pub fn viewport(total: usize, height: usize, selected: usize) -> (usize, usize) {
    if height == 0 || total <= height {
        return (0, total);
    }
    let start = selected.saturating_sub(height / 2).min(total - height);
    (start, start + height)
}

/// Render `lines` (one row each) into `area` inside `block`, scrolled so
/// `selected` stays visible, with a scrollbar when the list overflows the panel.
pub fn list<'a>(
    f: &mut Frame,
    t: &Theme,
    area: Rect,
    block: Block<'a>,
    lines: Vec<Line<'a>>,
    selected: usize,
) {
    let height = block.inner(area).height as usize;
    let total = lines.len();
    let (start, end) = viewport(total, height, selected);
    f.render_widget(
        Paragraph::new(lines[start..end].to_vec()).block(block),
        area,
    );
    if total > height {
        let mut state = ScrollbarState::new(total).position(start);
        f.render_stateful_widget(
            Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .begin_symbol(None)
                .end_symbol(None)
                .thumb_style(Style::default().fg(t.accent))
                .track_style(Style::default().fg(t.dim)),
            area.inner(Margin::new(0, 1)),
            &mut state,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viewport_keeps_the_selection_visible() {
        // Everything fits: the whole list, no scrolling.
        assert_eq!(viewport(5, 10, 3), (0, 5));
        // Overflow: the selected row stays within the window as it moves.
        for selected in 0..20 {
            let (start, end) = viewport(20, 10, selected);
            assert_eq!(end - start, 10, "window is always the panel height");
            assert!(
                (start..end).contains(&selected),
                "selected {selected} visible in [{start},{end})"
            );
        }
        // The window clamps at the ends rather than running past them.
        assert_eq!(viewport(20, 10, 0), (0, 10));
        assert_eq!(viewport(20, 10, 19), (10, 20));
        // A zero-height panel never panics.
        assert_eq!(viewport(20, 0, 5), (0, 20));
    }
}
