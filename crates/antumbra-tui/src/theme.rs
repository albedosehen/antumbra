//! The console's switchable semantic palette. Every panel (chrome, status, and
//! the animated population graph) tints from a [`Theme`], so cycling a theme
//! recolours the whole console while the umbra/penumbra/antumbra identity holds.
//! Hand-rolled (no theme crate) to keep the bespoke look and a tiny dep tree.

use ratatui::style::Color;

/// An RGB triple kept as `f64` so the animation can scale it by a glow/pulse
/// factor before quantizing to a [`Color`].
pub type Rgb = (f64, f64, f64);

/// Quantize a glow-scaled [`Rgb`] to a terminal colour.
pub fn rgb(c: Rgb, glow: f64) -> Color {
    Color::Rgb(
        (c.0 * glow).clamp(0.0, 255.0) as u8,
        (c.1 * glow).clamp(0.0, 255.0) as u8,
        (c.2 * glow).clamp(0.0, 255.0) as u8,
    )
}

/// A complete palette. Semantic fields drive the chrome and status; the `*_rgb`
/// fields are bases the animation scales (so a theme recolours the live graph).
#[derive(Clone, Copy)]
pub struct Theme {
    pub name: &'static str,
    /// Primary text (wordmark, headings).
    pub text: Color,
    /// Secondary text / panel titles.
    pub ink: Color,
    /// Muted: borders, inactive, hints.
    pub dim: Color,
    /// `key`-value values.
    pub value: Color,
    /// Selection / the tri-node core / headings highlight.
    pub accent: Color,
    /// The background fill behind the selected list row.
    pub sel: Color,
    /// Actionable boundary, pruned shadow, error.
    pub alert: Color,
    /// Graduated shadow, success.
    pub success: Color,
    /// In-flight shadow, mid state.
    pub warning: Color,
    /// Fitness gradient: low-fitness end.
    pub fit_lo: Rgb,
    /// Fitness gradient: high-fitness end.
    pub fit_hi: Rgb,
    /// The tri-node core base (pulsed by the animation).
    pub core: Rgb,
}

/// The default: the cast-shadow blues and cyans the console was designed around.
pub const SHADOW: Theme = Theme {
    name: "shadow",
    text: Color::Rgb(210, 230, 245),
    ink: Color::Rgb(120, 140, 160),
    dim: Color::Rgb(70, 90, 110),
    value: Color::Rgb(180, 200, 215),
    accent: Color::Rgb(120, 220, 255),
    sel: Color::Rgb(22, 44, 60),
    alert: Color::Rgb(200, 90, 90),
    success: Color::Rgb(90, 200, 150),
    warning: Color::Rgb(210, 190, 90),
    fit_lo: (30.0, 150.0, 110.0),
    fit_hi: (120.0, 240.0, 255.0),
    core: (120.0, 220.0, 255.0),
};

/// Warm: ambers and oranges, the population glowing like embers.
pub const EMBER: Theme = Theme {
    name: "ember",
    text: Color::Rgb(245, 225, 200),
    ink: Color::Rgb(170, 140, 110),
    dim: Color::Rgb(105, 80, 60),
    value: Color::Rgb(225, 205, 180),
    accent: Color::Rgb(255, 180, 90),
    sel: Color::Rgb(58, 36, 18),
    alert: Color::Rgb(235, 95, 70),
    success: Color::Rgb(205, 200, 95),
    warning: Color::Rgb(235, 160, 70),
    fit_lo: (150.0, 90.0, 40.0),
    fit_hi: (255.0, 200.0, 110.0),
    core: (255.0, 180.0, 90.0),
};

/// High-contrast greyscale with a single cool accent (low-colour terminals).
pub const MONO: Theme = Theme {
    name: "mono",
    text: Color::Rgb(235, 235, 240),
    ink: Color::Rgb(160, 160, 168),
    dim: Color::Rgb(95, 95, 102),
    value: Color::Rgb(200, 200, 208),
    accent: Color::Rgb(150, 200, 255),
    sel: Color::Rgb(44, 44, 52),
    alert: Color::Rgb(220, 120, 120),
    success: Color::Rgb(195, 210, 195),
    warning: Color::Rgb(220, 210, 160),
    fit_lo: (110.0, 110.0, 118.0),
    fit_hi: (235.0, 235.0, 240.0),
    core: (170.0, 200.0, 240.0),
};

/// All themes, in cycle order.
pub const ALL: [Theme; 3] = [SHADOW, EMBER, MONO];

impl Theme {
    /// Interpolate the fitness gradient (`fit_lo` -> `fit_hi`) at `fitness` in
    /// `[0,1]`, scaled by `glow`.
    pub fn fitness(&self, fitness: f32, glow: f64) -> Color {
        let f = fitness.clamp(0.0, 1.0) as f64;
        let lerp = |a: f64, b: f64| a + (b - a) * f;
        rgb(
            (
                lerp(self.fit_lo.0, self.fit_hi.0),
                lerp(self.fit_lo.1, self.fit_hi.1),
                lerp(self.fit_lo.2, self.fit_hi.2),
            ),
            glow,
        )
    }
}
