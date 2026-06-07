//! Scrolling for the one-row-per-item lists (shadows, boundaries): a
//! selection-following viewport — the selected row stays centred once a list
//! outgrows its panel — plus a slim scrollbar for a sense of position.
//! Hand-rolled on ratatui core.

use ratatui::layout::{Margin, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::Line;
use ratatui::widgets::{
    Block, List, ListItem, ListState, Scrollbar, ScrollbarOrientation, ScrollbarState,
};
use ratatui::Frame;

use crate::theme::Theme;

/// The first visible row index for `total` rows in `height` rows, keeping the
/// `selected` row visible (centred once the list overflows).
pub fn first_visible(total: usize, height: usize, selected: usize) -> usize {
    if height == 0 || total <= height {
        return 0;
    }
    selected.saturating_sub(height / 2).min(total - height)
}

/// Render `lines` (one row each) into `area` inside `block` as a list: the
/// `selected` row (if any) gets a full-row highlight and a marker, the view
/// scrolls to keep it visible, and a scrollbar shows position on overflow.
pub fn list<'a>(
    f: &mut Frame,
    t: &Theme,
    area: Rect,
    block: Block<'a>,
    lines: Vec<Line<'a>>,
    selected: Option<usize>,
) {
    let height = block.inner(area).height as usize;
    let total = lines.len();
    let start = first_visible(total, height, selected.unwrap_or(0));
    let items: Vec<ListItem> = lines.into_iter().map(ListItem::new).collect();
    let mut state = ListState::default();
    state.select(selected.map(|s| s.min(total.saturating_sub(1))));
    *state.offset_mut() = start;
    f.render_stateful_widget(
        List::new(items)
            .block(block)
            .highlight_symbol("▸ ")
            .highlight_style(Style::default().bg(t.sel).add_modifier(Modifier::BOLD)),
        area,
        &mut state,
    );
    if total > height {
        let mut sb = ScrollbarState::new(total).position(start);
        f.render_stateful_widget(
            Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .begin_symbol(None)
                .end_symbol(None)
                .thumb_style(Style::default().fg(t.accent))
                .track_style(Style::default().fg(t.dim)),
            area.inner(Margin::new(0, 1)),
            &mut sb,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_visible_keeps_the_selection_in_view() {
        // Everything fits: start at the top, no scrolling.
        assert_eq!(first_visible(5, 10, 3), 0);
        // Overflow: the selected row stays within the [start, start+height) window.
        for selected in 0..20 {
            let start = first_visible(20, 10, selected);
            assert!(
                (start..start + 10).contains(&selected),
                "selected {selected} visible in [{start},{})",
                start + 10
            );
            assert!(start + 10 <= 20, "window never runs past the end");
        }
        // The window clamps at the ends rather than running past them.
        assert_eq!(first_visible(20, 10, 0), 0);
        assert_eq!(first_visible(20, 10, 19), 10);
        // A zero-height panel never panics.
        assert_eq!(first_visible(20, 0, 5), 0);
    }
}
