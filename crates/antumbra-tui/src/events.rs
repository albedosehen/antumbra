//! The live event stream: store changes detected by diffing successive reads
//! of the server, and what the console itself did or was refused, kept as a
//! capped ring the events overlay shows. The console reads again every few
//! seconds, so the stream fills as the population is trained from another
//! process.

/// What kind of change an [`Event`] records; drives its glyph and color.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EventKind {
    /// A new expert or shadow appeared.
    Spawn,
    /// A shadow graduated (joined the population).
    Graduate,
    /// A shadow was pruned (collapsed).
    Prune,
    /// An expert froze.
    Freeze,
    /// A boundary was recorded or became actionable.
    Boundary,
    /// Connect / router / housekeeping.
    System,
    /// Something was refused or went wrong: a read that failed, an action
    /// that was not done.
    Problem,
}

/// A single change on the store's timeline.
pub struct Event {
    /// The animation clock (ms) when it was noticed, for relative timestamps.
    pub at_ms: f64,
    pub kind: EventKind,
    pub text: String,
}

impl EventKind {
    /// The glyph shown beside the event: a shape from the blocks every
    /// monospace font carries, the same shapes a state is drawn with.
    pub fn glyph(self) -> &'static str {
        match self {
            EventKind::Spawn => "+",
            EventKind::Graduate => "●",
            EventKind::Prune => "×",
            EventKind::Freeze => "■",
            EventKind::Boundary => "◆",
            EventKind::System => "·",
            EventKind::Problem => "▲",
        }
    }
}

/// The most events the stream retains.
pub const MAX_EVENTS: usize = 200;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_kind_has_a_glyph() -> anyhow::Result<()> {
        for kind in [
            EventKind::Spawn,
            EventKind::Graduate,
            EventKind::Prune,
            EventKind::Freeze,
            EventKind::Boundary,
            EventKind::System,
            EventKind::Problem,
        ] {
            assert!(!kind.glyph().is_empty(), "{kind:?} has a glyph");
        }
        Ok(())
    }

    #[test]
    fn a_glyph_is_one_column_in_any_terminal_font() -> anyhow::Result<()> {
        // ASCII, Latin-1, or the geometric shapes: no emoji, which is two
        // columns in some terminals and a missing-glyph box in others.
        for kind in [
            EventKind::Spawn,
            EventKind::Graduate,
            EventKind::Prune,
            EventKind::Freeze,
            EventKind::Boundary,
            EventKind::System,
            EventKind::Problem,
        ] {
            let glyph = kind.glyph();
            let mut chars = glyph.chars();
            let (Some(c), None) = (chars.next(), chars.next()) else {
                anyhow::bail!("{kind:?}: `{glyph}` is not one character");
            };
            let code = u32::from(c);
            assert!(
                code < 0x100 || (0x25A0..=0x25FF).contains(&code),
                "{kind:?}: `{glyph}` (U+{code:04X}) is outside the safe blocks"
            );
        }
        Ok(())
    }
}
