//! The audit slice across generations: whether the loop's own score is
//! carrying any real competence with it.
//!
//! ADR-0022 states the signal exactly: "search fitness rising while audit
//! fitness stays flat is the definition of measured overtuning". This is the
//! only instrument that can catch a loop tuning itself against its own
//! estimate, because it is the only number no decision in the loop can reach.

use serde::{Deserialize, Serialize};

/// One generation's two numbers: what the loop scored itself, and what the
/// slice it cannot touch said.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub generation: u32,
    /// The score selection optimizes. Search fitness, graduation fitness,
    /// whatever the loop is climbing.
    pub search: f32,
    /// The audit slice's pass rate. `None` when the slice was not evaluated
    /// this generation, which is expected: it runs every k generations, not
    /// every one.
    pub audit: Option<f32>,
}

/// What the window says.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Trend {
    /// The audit slice moved with the search score. The gain is real, as far as
    /// this instrument can tell.
    Carrying,
    /// The search score climbed and the audit slice did not follow. The loop
    /// has measured its own overtuning. This is S-1's kill criterion.
    Overtuning,
    /// The search score did not climb, so there is no gain to ask about.
    Flat,
    /// Not enough measured generations, or not enough audited ones, to say
    /// anything. Deliberately distinct from `Carrying`: an unasked question has
    /// not been answered in the affirmative.
    Inconclusive,
}

impl Trend {
    pub fn as_str(self) -> &'static str {
        match self {
            Trend::Carrying => "carrying",
            Trend::Overtuning => "overtuning",
            Trend::Flat => "flat",
            Trend::Inconclusive => "inconclusive",
        }
    }
}

/// How the window is read. The generation count is ADR-0022's ("ten
/// generations"); `carry` is an operator's dial with a stated default rather
/// than a measured constant, and it is named here so nobody mistakes it for one.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Watch {
    /// How many generations must be in hand before the question is asked.
    pub window: usize,
    /// How many of those must have an audit measurement. The slice runs every
    /// k generations, so a window will not be fully audited; below this, the
    /// verdict is `Inconclusive` rather than a guess from two points.
    pub min_audited: usize,
    /// The share of the search gain the audit slice must show for the gain to
    /// count as carried. A dial: lower it and more runs read as honest, raise
    /// it and the instrument becomes strict enough to fire on noise.
    pub carry: f32,
    /// A search gain at or below this is no climb, so there is nothing for the
    /// audit slice to have failed to follow. Also in the score's own units.
    pub noise: f32,
}

impl Default for Watch {
    fn default() -> Self {
        Watch {
            window: 10,
            min_audited: 4,
            carry: 0.25,
            noise: 0.01,
        }
    }
}

impl Watch {
    /// Read the most recent `window` generations.
    ///
    /// Gain is the median of the last third minus the median of the first
    /// third, not the difference of the endpoints. Generational fitness is
    /// noisy, and an endpoint difference is two samples deciding ten
    /// generations' verdict. Medians rather than means, because a third is
    /// three or four points: one spiking generation moves a mean of three by a
    /// third of the spike, which is enough to read a flat run as a climb, and
    /// moves a median of three not at all.
    pub fn read(&self, points: &[Point]) -> Trend {
        let window = match points.len().checked_sub(self.window) {
            Some(from) => &points[from..],
            None => return Trend::Inconclusive,
        };
        let audited: Vec<Point> = window
            .iter()
            .copied()
            .filter(|p| p.audit.is_some())
            .collect();
        if audited.len() < self.min_audited {
            return Trend::Inconclusive;
        }
        let Some(search_gain) = gain(window, |p| Some(p.search)) else {
            return Trend::Inconclusive;
        };
        if search_gain <= self.noise {
            return Trend::Flat;
        }
        let Some(audit_gain) = gain(&audited, |p| p.audit) else {
            return Trend::Inconclusive;
        };
        if audit_gain >= search_gain * self.carry {
            Trend::Carrying
        } else {
            Trend::Overtuning
        }
    }
}

/// Median of the last third minus median of the first third. `None` when
/// either third is empty or a point is missing the value asked for.
fn gain(points: &[Point], of: impl Fn(&Point) -> Option<f32> + Copy) -> Option<f32> {
    if points.len() < 2 {
        return None;
    }
    let third = (points.len() / 3).max(1);
    let median = |slice: &[Point]| -> Option<f32> {
        let mut values: Vec<f32> = slice.iter().map(of).collect::<Option<Vec<f32>>>()?;
        if values.is_empty() {
            return None;
        }
        values.sort_by(f32::total_cmp);
        Some(values[values.len() / 2])
    };
    Some(median(&points[points.len() - third..])? - median(&points[..third])?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ten generations: `search` climbs by `climb` in total, `audit` by
    /// `carried`, measured every `every` generations.
    fn run(climb: f32, carried: f32, every: u32) -> Vec<Point> {
        (0..10)
            .map(|g| {
                let t = g as f32 / 9.0;
                Point {
                    generation: g,
                    search: 0.40 + climb * t,
                    audit: (g % every == 0).then_some(0.40 + carried * t),
                }
            })
            .collect()
    }

    #[test]
    fn a_climb_the_audit_slice_does_not_follow_is_the_kill_criterion() {
        // Search up 20 points, audit up half a point, audited every other
        // generation: the loop has tuned itself against its own estimate.
        assert_eq!(
            Watch::default().read(&run(0.20, 0.005, 2)),
            Trend::Overtuning
        );
    }

    #[test]
    fn a_climb_the_audit_slice_follows_is_a_real_gain() {
        assert_eq!(Watch::default().read(&run(0.20, 0.15, 2)), Trend::Carrying);
        // It does not have to follow all the way: the dial asks for a share.
        assert_eq!(Watch::default().read(&run(0.20, 0.06, 2)), Trend::Carrying);
    }

    #[test]
    fn a_run_that_is_not_climbing_has_no_gain_to_have_lost() {
        assert_eq!(Watch::default().read(&run(0.0, 0.0, 2)), Trend::Flat);
        // Nor does a run drifting down read as overtuning: it is a different
        // problem, and calling it this one would send the operator to the
        // wrong place.
        assert_eq!(Watch::default().read(&run(-0.10, 0.0, 2)), Trend::Flat);
    }

    #[test]
    fn too_little_measurement_is_not_a_clean_bill() {
        let watch = Watch::default();
        // Fewer generations than the window.
        assert_eq!(watch.read(&run(0.20, 0.0, 2)[..6]), Trend::Inconclusive);
        // Ten generations, but the audit slice ran twice in them.
        assert_eq!(watch.read(&run(0.20, 0.0, 7)), Trend::Inconclusive);
        assert_eq!(watch.read(&[]), Trend::Inconclusive);
    }

    #[test]
    fn only_the_most_recent_window_is_read() {
        // Twenty generations: overtuned for the first ten, honest for the last
        // ten. The verdict is about where the loop is now.
        let mut points: Vec<Point> = run(0.20, 0.0, 2);
        points.extend(run(0.20, 0.18, 2).into_iter().map(|mut p| {
            p.generation += 10;
            p.search += 0.20;
            p.audit = p.audit.map(|a| a + 0.02);
            p
        }));
        assert_eq!(points.len(), 20);
        assert_eq!(Watch::default().read(&points), Trend::Carrying);
    }

    #[test]
    fn the_gain_is_a_trend_and_not_two_samples() {
        // A single spiking generation at the end does not make a climb: the
        // thirds average it away, where an endpoint difference would not.
        let mut points = run(0.0, 0.0, 2);
        if let Some(last) = points.last_mut() {
            last.search = 0.90;
        }
        assert_eq!(Watch::default().read(&points), Trend::Flat);
    }
}
