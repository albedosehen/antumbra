//! The console's palette: the dashboard's tokens, so the two read as one
//! product. Night and Pearl on dark, Mist and Ink on light, Periwinkle the only
//! accent; the values are the dashboard stylesheet's
//! (`antumbra-mcp/src/http/dashboard/app.css`), and a change there is a change
//! here. Every panel tints from a [`Theme`], so cycling one (`t`) recolors the
//! whole console.
//!
//! The roles are few on purpose. Text has three strengths, there is one accent,
//! and the only other colour is [`Theme::danger`], kept for failure. A state is
//! told apart by its shape and its word ([`Tone`]), never by a hue of its own.

use ratatui::style::{Color, Modifier, Style};

/// An RGB triple kept as `f64`, so the animation can mix it before it is
/// quantized to a [`Color`].
pub type Rgb = (f64, f64, f64);

/// An opaque colour, as the stylesheet writes one.
type Hex = (u8, u8, u8);

const fn solid(c: Hex) -> Color {
    Color::Rgb(c.0, c.1, c.2)
}

const fn float(c: Hex) -> Rgb {
    (c.0 as f64, c.1 as f64, c.2 as f64)
}

/// `top` at `percent` opacity over `under`: the stylesheet's
/// `rgb(r g b / 0.74)`, flattened, since a terminal cell has no alpha.
const fn over(top: Hex, under: Hex, percent: u32) -> Color {
    const fn channel(top: u8, under: u8, percent: u32) -> u8 {
        ((top as u32 * percent + under as u32 * (100 - percent) + 50) / 100) as u8
    }
    Color::Rgb(
        channel(top.0, under.0, percent),
        channel(top.1, under.1, percent),
        channel(top.2, under.2, percent),
    )
}

/// How far down the accent a low value is drawn: a gauge at zero is faint, not
/// gone.
const FAINT: f64 = 0.42;

/// A complete palette.
#[derive(Clone, Copy)]
pub struct Theme {
    pub name: &'static str,
    /// The page everything is drawn on. `Reset` leaves the terminal's own.
    pub ground: Color,
    /// A modal's fill: one step up from the ground.
    pub surface: Color,
    /// A key chip's fill: one step up again.
    pub raised: Color,
    /// Primary text: names, values, headings.
    pub ink: Color,
    /// Secondary text: labels, panel titles, what a key does.
    pub ink2: Color,
    /// Tertiary text: hints, placeholders, what is inactive.
    pub ink3: Color,
    /// Borders and the track of a gauge.
    pub line: Color,
    /// The one accent: the selection, the focus, what is in force.
    pub accent: Color,
    /// The fill behind the selected row.
    pub accent_soft: Color,
    /// Text on an accent fill.
    pub accent_ink: Color,
    /// Failure and refusal, and nothing else.
    pub danger: Color,
    /// The accent and the ground as triples, which the animation mixes
    /// between. For a theme that leaves the ground to the terminal, the ground
    /// here is the dark one it is designed against.
    accent_rgb: Rgb,
    ground_rgb: Rgb,
}

const NIGHT_GROUND: Hex = (0x12, 0x12, 0x1b);
const PEARL: Hex = (0xef, 0xe7, 0xd8);
const NIGHT_ACCENT: Hex = (0x8a, 0x84, 0xcc);

/// The default: the dashboard's dark palette.
pub const NIGHT: Theme = Theme {
    name: "night",
    ground: solid(NIGHT_GROUND),
    surface: solid((0x1a, 0x1a, 0x26)),
    raised: solid((0x22, 0x22, 0x31)),
    ink: solid(PEARL),
    ink2: over(PEARL, NIGHT_GROUND, 74),
    ink3: over(PEARL, NIGHT_GROUND, 60),
    line: over(PEARL, NIGHT_GROUND, 18),
    accent: solid(NIGHT_ACCENT),
    accent_soft: over(NIGHT_ACCENT, NIGHT_GROUND, 16),
    accent_ink: solid(NIGHT_GROUND),
    danger: solid((0xd9, 0x91, 0x7f)),
    accent_rgb: float(NIGHT_ACCENT),
    ground_rgb: float(NIGHT_GROUND),
};

const MIST_GROUND: Hex = (0xee, 0xec, 0xf5);
const INK: Hex = (0x1b, 0x1a, 0x2e);
/// The accent as small text, a step darker than the brand shade so it clears
/// 4.5:1 on the light ground; a terminal draws the accent as text.
const MIST_ACCENT: Hex = (0x5c, 0x55, 0xa8);

/// The dashboard's light palette, for a light terminal.
pub const MIST: Theme = Theme {
    name: "mist",
    ground: solid(MIST_GROUND),
    surface: solid((0xf8, 0xf7, 0xfb)),
    raised: solid((0xff, 0xff, 0xff)),
    ink: solid(INK),
    ink2: over(INK, MIST_GROUND, 74),
    ink3: over(INK, MIST_GROUND, 65),
    line: over(INK, MIST_GROUND, 20),
    accent: solid(MIST_ACCENT),
    accent_soft: over((0x6a, 0x63, 0xb8), MIST_GROUND, 12),
    accent_ink: solid((0xff, 0xff, 0xff)),
    danger: solid((0x9c, 0x46, 0x36)),
    accent_rgb: float(MIST_ACCENT),
    ground_rgb: float(MIST_GROUND),
};

/// Greys on the terminal's own background, with one cool accent: for a
/// terminal with few colours, or one whose background should show through.
pub const MONO: Theme = Theme {
    name: "mono",
    ground: Color::Reset,
    surface: Color::Reset,
    raised: Color::Rgb(58, 58, 66),
    ink: Color::Rgb(235, 235, 240),
    ink2: Color::Rgb(172, 172, 180),
    ink3: Color::Rgb(128, 128, 136),
    line: Color::Rgb(95, 95, 102),
    accent: Color::Rgb(150, 200, 255),
    accent_soft: Color::Rgb(44, 44, 52),
    accent_ink: Color::Rgb(12, 14, 18),
    danger: Color::Rgb(226, 132, 132),
    accent_rgb: (150.0, 200.0, 255.0),
    ground_rgb: (12.0, 14.0, 18.0),
};

/// All themes, in cycle order.
pub const ALL: [Theme; 3] = [NIGHT, MIST, MONO];

/// What a state is, as far as the eye needs: each has a shape and a strength
/// of ink, and the word beside it says the rest. The dashboard marks an
/// expert's status the same way.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tone {
    /// Settled, or in force: a filled dot in the accent.
    Settled,
    /// Open, or under way: a ring in the secondary ink.
    Open,
    /// Ended, or set aside: a cross in the faint ink.
    Ended,
    /// Failed: a triangle in the danger colour.
    Failed,
}

impl Tone {
    /// The shape, from the geometric block every monospace font carries.
    pub fn glyph(self) -> &'static str {
        match self {
            Tone::Settled => "●",
            Tone::Open => "○",
            Tone::Ended => "×",
            Tone::Failed => "▲",
        }
    }
}

impl Theme {
    /// The colour a state's shape and word are drawn in.
    pub fn tone(&self, tone: Tone) -> Color {
        match tone {
            Tone::Settled => self.accent,
            Tone::Open => self.ink2,
            Tone::Ended => self.ink3,
            Tone::Failed => self.danger,
        }
    }

    /// The style of text on the page: the ink on the ground.
    pub fn page(&self) -> Style {
        Style::default().fg(self.ink).bg(self.ground)
    }

    /// A key chip: what to press, set a step above the page it sits on.
    pub fn chip(&self) -> Style {
        Style::default().fg(self.ink).bg(self.raised)
    }

    /// The chip of the key that goes ahead: the accent, or the danger colour
    /// when going ahead removes something.
    pub fn chip_primary(&self, removes: bool) -> Style {
        Style::default()
            .fg(self.accent_ink)
            .bg(if removes { self.danger } else { self.accent })
            .add_modifier(Modifier::BOLD)
    }

    /// The selected row: the soft accent behind it, its text bold. It sets no
    /// ink, so a state in the row keeps the colour that says what it is.
    pub fn selected(&self) -> Style {
        Style::default()
            .bg(self.accent_soft)
            .add_modifier(Modifier::BOLD)
    }

    /// Where a name is listed: its place in [`ALL`], or the default's.
    pub fn index_of(name: &str) -> usize {
        ALL.iter().position(|t| t.name == name).unwrap_or(0)
    }

    /// The accent at `strength` in `[0,1]` over the ground: what a pulse, a
    /// trail or a far node is drawn in. Zero is the ground itself, so a fading
    /// mark sinks into the page on a light theme as on a dark one.
    pub fn glow(&self, strength: f64) -> Color {
        let s = strength.clamp(0.0, 1.0);
        let mix = |ground: f64, accent: f64| (ground + (accent - ground) * s).round() as u8;
        Color::Rgb(
            mix(self.ground_rgb.0, self.accent_rgb.0),
            mix(self.ground_rgb.1, self.accent_rgb.1),
            mix(self.ground_rgb.2, self.accent_rgb.2),
        )
    }

    /// A `[0,1]` measure as a strength of the accent, faint when low and full
    /// when high, itself at `glow` of the way up from the ground.
    pub fn fitness(&self, fitness: f32, glow: f64) -> Color {
        let f = f64::from(fitness.clamp(0.0, 1.0));
        self.glow((FAINT + (1.0 - FAINT) * f) * glow)
    }

    /// The ground as a triple, which a dimmed backdrop and a transition settle
    /// toward.
    pub fn ground_rgb(&self) -> Rgb {
        self.ground_rgb
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The dashboard stylesheet, which these values are taken from.
    const STYLESHEET: &str = include_str!("../../antumbra-mcp/src/http/dashboard/app.css");

    fn hex(c: Color) -> anyhow::Result<String> {
        match c {
            Color::Rgb(r, g, b) => Ok(format!("#{r:02x}{g:02x}{b:02x}")),
            other => anyhow::bail!("{other:?} is not an RGB colour"),
        }
    }

    /// The value the stylesheet gives `token`, in its light block (the first)
    /// or its dark one (the second).
    fn token(name: &str, dark: bool) -> anyhow::Result<&'static str> {
        let needle = format!("--{name}:");
        let mut found = STYLESHEET
            .lines()
            .filter_map(|line| line.trim().strip_prefix(needle.as_str()))
            .map(|value| value.trim().trim_end_matches(';'));
        let value = if dark { found.nth(1) } else { found.next() };
        value.ok_or_else(|| anyhow::anyhow!("the stylesheet has no --{name}"))
    }

    #[test]
    fn night_and_mist_are_the_dashboards_palettes() -> anyhow::Result<()> {
        for (theme, dark) in [(NIGHT, true), (MIST, false)] {
            assert_eq!(hex(theme.ground)?, token("ground", dark)?, "{}", theme.name);
            assert_eq!(
                hex(theme.surface)?,
                token("surface", dark)?,
                "{}",
                theme.name
            );
            assert_eq!(hex(theme.raised)?, token("raised", dark)?, "{}", theme.name);
            assert_eq!(hex(theme.ink)?, token("ink", dark)?, "{}", theme.name);
            assert_eq!(hex(theme.danger)?, token("danger", dark)?, "{}", theme.name);
            assert_eq!(
                hex(theme.accent_ink)?,
                token("accent-ink", dark)?,
                "{}",
                theme.name
            );
            // The accent as text: the stylesheet's `--accent-text`, which on
            // dark is the brand shade and on light a step darker.
            assert_eq!(
                hex(theme.accent)?,
                token("accent-text", dark)?,
                "{}",
                theme.name
            );
        }
        Ok(())
    }

    #[test]
    fn a_strength_of_ink_is_the_ink_over_the_ground() -> anyhow::Result<()> {
        // `rgb(239 231 216 / 0.74)` over `#12121b`, by hand.
        assert_eq!(NIGHT.ink2, Color::Rgb(182, 176, 167));
        assert_eq!(NIGHT.ink3, Color::Rgb(151, 146, 140));
        // Each strength sits between the ground and the ink, in order.
        for theme in [NIGHT, MIST] {
            let lum = |c: Color| match c {
                Color::Rgb(r, g, b) => Ok(u32::from(r) + u32::from(g) + u32::from(b)),
                other => Err(anyhow::anyhow!("{other:?} is not an RGB colour")),
            };
            let (ground, ink) = (lum(theme.ground)?, lum(theme.ink)?);
            let away = |c: Color| -> anyhow::Result<u32> { Ok(lum(c)?.abs_diff(ground)) };
            assert!(away(theme.ink2)? < ink.abs_diff(ground), "{}", theme.name);
            assert!(away(theme.ink3)? < away(theme.ink2)?, "{}", theme.name);
            assert!(away(theme.line)? < away(theme.ink3)?, "{}", theme.name);
        }
        Ok(())
    }

    #[test]
    fn a_glow_runs_from_the_ground_to_the_accent() -> anyhow::Result<()> {
        for theme in [NIGHT, MIST] {
            assert_eq!(theme.glow(0.0), theme.ground, "{}", theme.name);
            assert_eq!(theme.glow(1.0), theme.accent, "{}", theme.name);
            assert_eq!(theme.glow(7.0), theme.accent, "{}", theme.name);
            // A full measure at full glow is the accent; a low one is fainter
            // and still drawn.
            assert_eq!(theme.fitness(1.0, 1.0), theme.accent, "{}", theme.name);
            assert_ne!(theme.fitness(0.0, 1.0), theme.ground, "{}", theme.name);
        }
        Ok(())
    }

    #[test]
    fn mono_leaves_the_ground_to_the_terminal() -> anyhow::Result<()> {
        assert_eq!(MONO.ground, Color::Reset);
        assert_eq!(MONO.surface, Color::Reset);
        assert_eq!(MONO.page().bg, Some(Color::Reset));
        Ok(())
    }

    #[test]
    fn every_state_has_a_shape_of_its_own() -> anyhow::Result<()> {
        let tones = [Tone::Settled, Tone::Open, Tone::Ended, Tone::Failed];
        let shapes: std::collections::HashSet<&str> = tones.iter().map(|t| t.glyph()).collect();
        assert_eq!(shapes.len(), tones.len());
        // Only failure is drawn in the danger colour.
        for tone in tones {
            assert_eq!(
                NIGHT.tone(tone) == NIGHT.danger,
                tone == Tone::Failed,
                "{tone:?}"
            );
        }
        Ok(())
    }
}
