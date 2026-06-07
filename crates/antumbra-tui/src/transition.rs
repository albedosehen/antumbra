//! View-switch transitions (tachyonfx): a short effect played over the console
//! when the focused region changes (the detail panel re-assembles), the theme
//! cycles (a colour wash settles into the new palette), or the layout switches
//! (the whole body re-assembles). Each effect carries the [`Scope`] it plays
//! over; effects are processed inside the draw pass against the live frame
//! buffer, then idle to `None` when done, so the loop only spends the extra
//! frames while one is running.

use ratatui::layout::{Constraint, Layout, Rect};
use tachyonfx::{fx, Effect, EffectTimer, Interpolation};

use crate::theme::Theme;

/// The dark terminal background the console is designed against (the colour a
/// wash settles out of).
const BG: ratatui::style::Color = ratatui::style::Color::Rgb(12, 14, 18);

/// Which region of the frame an effect plays over.
pub enum Scope {
    /// The right detail column (a focus or theme change).
    Detail,
    /// The whole body between header and footer (a layout change).
    Body,
    /// The active overlay's modal box (resolved by the caller via the app's
    /// current mode, since the palette's size depends on its matches).
    Overlay,
}

/// A queued effect and the part of the frame it animates.
pub type Pending = (Effect, Scope);

/// The body row (between the 3-row header and 1-row footer).
fn body(frame: Rect) -> Rect {
    Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(0),
        Constraint::Length(1),
    ])
    .split(frame)[1]
}

/// The detail column (right body panel) a focus-switch effect plays over — the
/// same split [`crate::ui::render`] uses, so the effect lands on the panel that
/// actually changed.
pub fn detail_area(frame: Rect) -> Rect {
    Layout::horizontal([Constraint::Percentage(64), Constraint::Percentage(36)]).split(body(frame))
        [1]
}

/// Resolve a fixed [`Scope`] to the rectangle it animates within `frame`.
/// [`Scope::Overlay`] is resolved by the caller (it needs the app's mode) and
/// falls back to the body here.
pub fn scope_area(scope: &Scope, frame: Rect) -> Rect {
    match scope {
        Scope::Detail => detail_area(frame),
        Scope::Body | Scope::Overlay => body(frame),
    }
}

/// Focus switched (Tab): the detail panel's cells re-assemble from scattered.
pub fn focus_switch() -> Pending {
    (
        fx::coalesce(EffectTimer::from((260u32, Interpolation::QuadOut))),
        Scope::Detail,
    )
}

/// Theme cycled (`t`): a brief wash of the new accent settles into the palette.
pub fn theme_wash(theme: &Theme) -> Pending {
    (
        fx::fade_from(
            theme.accent,
            BG,
            EffectTimer::from((220u32, Interpolation::SineOut)),
        ),
        Scope::Detail,
    )
}

/// Layout switched (`L`): the whole body re-assembles into the new arrangement.
pub fn layout_switch() -> Pending {
    (
        fx::coalesce(EffectTimer::from((320u32, Interpolation::QuadOut))),
        Scope::Body,
    )
}

/// An overlay opened (`?` / `:`): its modal box coalesces into view.
pub fn overlay_open() -> Pending {
    (
        fx::coalesce(EffectTimer::from((200u32, Interpolation::QuadOut))),
        Scope::Overlay,
    )
}
