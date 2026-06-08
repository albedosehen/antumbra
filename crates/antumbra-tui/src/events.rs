//! The live event stream: store changes detected by diffing successive reloads,
//! kept as a capped ring the events overlay shows. A reload runs on a live-query
//! notification (see [`crate::live`]) so an external write surfaces within a
//! frame, with a periodic reload every couple of seconds as the fallback, so the
//! stream fills as the population is trained from another process.

/// What kind of change an [`Event`] records; drives its glyph and colour.
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
}

/// A single change on the store's timeline.
pub struct Event {
    /// The animation clock (ms) when it was noticed, for relative timestamps.
    pub at_ms: f64,
    pub kind: EventKind,
    pub text: String,
}

impl EventKind {
    /// The glyph shown beside the event.
    pub fn glyph(self) -> &'static str {
        match self {
            EventKind::Spawn => "✷",
            EventKind::Graduate => "✦",
            EventKind::Prune => "×",
            EventKind::Freeze => "❄",
            EventKind::Boundary => "⛔",
            EventKind::System => "·",
        }
    }
}

/// The most events the stream retains.
pub const MAX_EVENTS: usize = 200;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_kind_has_a_glyph() {
        for kind in [
            EventKind::Spawn,
            EventKind::Graduate,
            EventKind::Prune,
            EventKind::Freeze,
            EventKind::Boundary,
            EventKind::System,
        ] {
            assert!(!kind.glyph().is_empty(), "{kind:?} has a glyph");
        }
    }
}
