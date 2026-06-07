//! View-switch transitions (tachyonfx): a short effect played over the console
//! when the focused region changes (the detail panel re-assembles) or the theme
//! cycles (a colour wash settles into the new palette). Effects are processed
//! inside the draw pass against the live frame buffer, then idle to `None` when
//! done, so the loop only spends the extra frames while one is running.

use ratatui::layout::{Constraint, Layout, Rect};
use tachyonfx::{fx, Effect, EffectTimer, Interpolation};

use crate::theme::Theme;

/// The dark terminal background the console is designed against (the colour a
/// wash settles out of).
const BG: ratatui::style::Color = ratatui::style::Color::Rgb(12, 14, 18);

/// The detail column (right body panel) a focus-switch effect plays over — the
/// same split [`crate::ui::render`] uses, so the effect lands on the panel that
/// actually changed.
pub fn detail_area(frame: Rect) -> Rect {
    let rows = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(0),
        Constraint::Length(1),
    ])
    .split(frame);
    Layout::horizontal([Constraint::Percentage(64), Constraint::Percentage(36)]).split(rows[1])[1]
}

/// Focus switched (Tab): the detail panel's cells re-assemble from scattered.
pub fn focus_switch() -> Effect {
    fx::coalesce(EffectTimer::from((260u32, Interpolation::QuadOut)))
}

/// Theme cycled (`t`): a brief wash of the new accent settles into the palette.
pub fn theme_wash(theme: &Theme) -> Effect {
    fx::fade_from(
        theme.accent,
        BG,
        EffectTimer::from((220u32, Interpolation::SineOut)),
    )
}
