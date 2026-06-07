//! Render-capability tiers (ADR-0009): how visuals realize themselves. The
//! default is a universal hand-rolled Braille/Canvas drawing that renders
//! identically over SSH, inside multiplexers, and in CI; richer terminals can
//! opt into a raster image path (sixel/kitty/iTerm2) for genuinely raster-y
//! content (dense heatmaps, photographic previews, real PNG exports).
//!
//! The tier is resolved ONCE at startup and never re-probed, so the frame loop
//! just reads it. Vector visuals (the orbit graph, line charts, gauges) stay on
//! the Canvas path at every tier — Braille's sub-cell resolution looks identical
//! at terminal scale and is strictly more robust than rasterizing them. Raster
//! is a progressive enhancement behind the `raster` cargo feature; the default
//! build carries no graphics dependency and ships on the Canvas tier.

use std::io::IsTerminal;

use ratatui::symbols::Marker;

/// The operator's requested strategy (`--render`, `ANTUMBRA_RENDER`).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, clap::ValueEnum)]
pub enum RenderMode {
    /// Probe the terminal once at startup and pick the richest tier it supports.
    #[default]
    Auto,
    /// Force the raster image path (needs the `raster` build + a capable terminal).
    Raster,
    /// Force the universal Braille/Canvas vector path.
    Canvas,
    /// Force the lowest-fidelity path (coarse dot markers) for dumb terminals.
    Ascii,
}

/// The resolved capability, fixed for the run.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum RenderTier {
    /// A terminal graphics protocol is available; raster-worthy visuals may emit images.
    // Constructed by the raster probe (`feature = "raster"`); see `probe_tier`.
    #[allow(dead_code)]
    Raster,
    /// Truecolor cells but no graphics protocol; raster-worthy visuals fall to half-blocks.
    // Constructed by the raster probe (`feature = "raster"`); see `probe_tier`.
    #[allow(dead_code)]
    HalfBlock,
    /// The universal floor: Braille/Octant vector drawing. Works everywhere.
    #[default]
    Canvas,
    /// Dumb terminal / pipe / CI: coarse dot markers, no graphics, minimal Unicode.
    Ascii,
}

impl RenderTier {
    /// The Canvas marker for VECTOR drawings (the orbit graph, line charts,
    /// vector heatmaps). Braille at every tier that can render it; the coarse Dot
    /// only where Unicode density can't be assumed.
    pub fn marker(self) -> Marker {
        match self {
            RenderTier::Ascii => Marker::Dot,
            _ => Marker::Braille,
        }
    }

    /// Whether a raster-worthy visual (dense heatmap, photographic content, a
    /// real PNG preview) may emit an actual image at this tier. Consumed by
    /// raster-worthy visuals once the graphics backend is wired (`feature = "raster"`).
    #[allow(dead_code)]
    pub fn allows_raster(self) -> bool {
        matches!(self, RenderTier::Raster)
    }
}

/// Resolve the capability tier once, at startup. Short-circuits to a safe vector
/// tier when there is no TTY (pipes/CI) or a hostile multiplexer — BEFORE any
/// (potentially multi-second) graphics probe — so the frame loop is never touched.
pub fn resolve_tier(mode: RenderMode) -> RenderTier {
    match mode {
        RenderMode::Canvas => return RenderTier::Canvas,
        RenderMode::Ascii => return RenderTier::Ascii,
        RenderMode::Auto | RenderMode::Raster => {}
    }
    // No interactive terminal: a graphics probe would hang or garble the stream.
    if !std::io::stdout().is_terminal() {
        return RenderTier::Canvas;
    }
    // A multiplexer that filters graphics passthrough breaks raster; demote.
    if hostile_multiplexer() {
        return RenderTier::Canvas;
    }
    probe_tier(mode)
}

/// Whether we're inside a multiplexer that breaks graphics-protocol passthrough
/// (tmux without `allow-passthrough`, or zellij). Conservative: any tmux/screen/
/// zellij marker demotes out of the raster tier.
fn hostile_multiplexer() -> bool {
    if std::env::var_os("ZELLIJ").is_some() {
        return true;
    }
    match std::env::var("TERM") {
        Ok(term) => term.starts_with("tmux") || term.starts_with("screen"),
        Err(_) => false,
    }
}

#[cfg(not(feature = "raster"))]
fn probe_tier(_mode: RenderMode) -> RenderTier {
    // No raster backend compiled in: the richest universally-safe tier is Canvas.
    RenderTier::Canvas
}

#[cfg(feature = "raster")]
fn probe_tier(_mode: RenderMode) -> RenderTier {
    // Extension point: the `ratatui-image` `Picker::from_query_stdio()` probe
    // (kitty/sixel/iTerm2 + cell pixel size) slots in here, gated so the default
    // build carries no image dependency. Resolved once; falls to Canvas on a
    // probe timeout/error (the common Windows-Terminal outcome) so `auto` never
    // fails visibly. Wired with the first raster-worthy visual.
    RenderTier::Canvas
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_modes_resolve_without_probing() {
        assert_eq!(resolve_tier(RenderMode::Canvas), RenderTier::Canvas);
        assert_eq!(resolve_tier(RenderMode::Ascii), RenderTier::Ascii);
    }

    #[test]
    fn auto_without_a_tty_is_canvas() {
        // The test harness captures stdout (not a TTY), so `auto` short-circuits
        // to the universal vector path rather than probing.
        assert_eq!(resolve_tier(RenderMode::Auto), RenderTier::Canvas);
        assert_eq!(resolve_tier(RenderMode::Raster), RenderTier::Canvas);
    }

    #[test]
    fn markers_degrade_with_the_tier() {
        assert_eq!(RenderTier::Canvas.marker(), Marker::Braille);
        assert_eq!(RenderTier::HalfBlock.marker(), Marker::Braille);
        assert_eq!(RenderTier::Raster.marker(), Marker::Braille);
        assert_eq!(RenderTier::Ascii.marker(), Marker::Dot);
    }

    #[test]
    fn only_the_raster_tier_allows_images() {
        assert!(RenderTier::Raster.allows_raster());
        assert!(!RenderTier::HalfBlock.allows_raster());
        assert!(!RenderTier::Canvas.allows_raster());
        assert!(!RenderTier::Ascii.allows_raster());
    }
}
