//! The partition every instrument is built on: which tasks the loop may see,
//! which are frozen away from it, which exist only to be logged, and which
//! cannot be passed at all.
//!
//! ADR-0022's anchor invariant is about where reward comes from. These slices
//! are how that invariant becomes checkable: an instrument that reads a slice
//! selection can reach is measuring the loop's own estimate, not the loop.
//!
//! The partition is a pure function of the task id and a seed. It has to be,
//! because a held-out slice that drifts is not held out: a task that moves into
//! the visible set between generations quietly hands selection something it was
//! supposed to be measured against. Nothing here stores a list, so nothing can
//! fall out of step with one.
//!
//! It lives in the domain core rather than beside the instruments that read it
//! because a partition is only real where training happens. Labelling outcomes
//! after a run trained on every task measures nothing: the held-out tasks were
//! learned from and the audit tasks were scored for graduation. The trainer
//! enforces it through [`Holdout`], and the instruments read it back.

use serde::{Deserialize, Serialize};

/// Which side of the anchor a task sits on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Slice {
    /// Training, search, curriculum and graduation may all see these. Every
    /// number the loop optimizes is computed here.
    Visible,
    /// A frozen suite the loop never trains, searches or graduates against. The
    /// other half of the visible-minus-held-out gap.
    HeldOut,
    /// Touched by no decision at all, evaluated every k generations and only
    /// logged. Distinct from `HeldOut`: the held-out suite is the one S-1 may
    /// come to graduate against once its search score and graduation score are
    /// separated, and a candidate selected repeatedly against a suite is
    /// selected against it in the end. The audit slice is read by nothing that
    /// chooses.
    Audit,
    /// Tasks whose specification cannot be satisfied. The target is zero passes
    /// forever, and a pass is proof of a shortcut rather than a near miss.
    Impossible,
}

impl Slice {
    pub fn as_str(self) -> &'static str {
        match self {
            Slice::Visible => "visible",
            Slice::HeldOut => "held_out",
            Slice::Audit => "audit",
            Slice::Impossible => "impossible",
        }
    }

    /// Whether any decision the loop makes may read this slice. The one
    /// predicate the rest of the crate asks, and the shape the anchor invariant
    /// takes in code.
    pub fn selection_may_read(self) -> bool {
        matches!(self, Slice::Visible)
    }
}

impl std::str::FromStr for Slice {
    type Err = crate::AntumbraError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "visible" => Ok(Slice::Visible),
            "held_out" | "heldout" => Ok(Slice::HeldOut),
            "audit" => Ok(Slice::Audit),
            "impossible" => Ok(Slice::Impossible),
            other => Err(crate::AntumbraError::other(format!(
                "unknown slice `{other}` (use visible | held_out | audit | impossible)"
            ))),
        }
    }
}

/// How much of a corpus is withheld. The remainder, after held-out and audit,
/// is visible.
///
/// Impossible tasks are not drawn from the corpus: they are authored, because a
/// task that cannot be satisfied cannot be produced by taking a satisfiable one
/// away from the training set. [`Partition::of`] therefore never returns
/// `Impossible`; a corpus marks those itself.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Partition {
    /// Share of the corpus frozen away from training and search, but still
    /// gating graduation.
    pub held_out: f32,
    /// Share touched by no decision at all.
    pub audit: f32,
    /// Fixes the split. Changing it repartitions the corpus, which invalidates
    /// every gap measured before the change, so it is a value, not a constant:
    /// the loop records which seed a generation was measured under.
    pub seed: u64,
}

impl Default for Partition {
    /// A fifth held out and a tenth audited, leaving seven tenths visible.
    ///
    /// The audit slice is the smaller of the two on purpose. It is read every k
    /// generations and never by anything that chooses, so it buys trend rather
    /// than resolution, while the held-out suite gates every graduation and
    /// needs enough tasks that a candidate cannot clear it on noise.
    fn default() -> Self {
        Partition {
            held_out: 0.20,
            audit: 0.10,
            seed: 0,
        }
    }
}

impl Partition {
    /// `Err` when the shares do not leave a visible remainder. A corpus with
    /// nothing visible trains on nothing, and one whose shares exceed a whole
    /// is a typo that would otherwise silently starve training.
    pub fn new(held_out: f32, audit: f32, seed: u64) -> crate::Result<Self> {
        let sane = |x: f32| x.is_finite() && (0.0..1.0).contains(&x);
        if !sane(held_out) || !sane(audit) || held_out + audit >= 1.0 {
            return Err(crate::AntumbraError::other(format!(
                "a partition must leave something visible (held_out {held_out}, audit {audit})"
            )));
        }
        Ok(Partition {
            held_out,
            audit,
            seed,
        })
    }

    /// Which slice a task falls in. Deterministic in (task id, seed) and in
    /// nothing else: not in the order tasks arrive, not in how many there are,
    /// and not in when it is asked. A corpus that grows does not reshuffle the
    /// tasks already in it.
    pub fn of(&self, task_id: &str) -> Slice {
        let point = unit_hash(task_id, self.seed);
        if point < self.held_out {
            Slice::HeldOut
        } else if point < self.held_out + self.audit {
            Slice::Audit
        } else {
            Slice::Visible
        }
    }
}

/// What a training run is asked to withhold: the partition it must learn and
/// score inside, and whether it also measures the audit slice this time.
///
/// A trainer given one learns only from visible tasks, computes the fitness
/// that decides graduation only from visible tasks, and measures the held-out
/// slice (and the audit slice when `audit` is set) without learning from it.
/// It echoes what it enforced in [`crate::ports::TrainOutcome::holdout`], and
/// the loop measures a generation only when the echo matches what was asked:
/// a trainer that ignored the request would otherwise produce a gap that looks
/// like a measurement and is not.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Holdout {
    pub partition: Partition,
    /// Measure the audit slice in this run. ADR-0022 reads it every k
    /// generations rather than every one; the loop keeps the schedule.
    pub audit: bool,
}

impl Holdout {
    /// Whether a task in this slice may be learned from and scored for
    /// graduation. The same answer [`Slice::selection_may_read`] gives, asked
    /// where the learning happens.
    pub fn learns_from(&self, task_id: &str) -> bool {
        self.partition.of(task_id).selection_may_read()
    }

    /// Whether a task this run does not learn from is still measured in it:
    /// the held-out slice always, the audit slice only when it is due.
    pub fn measures(&self, task_id: &str) -> bool {
        match self.partition.of(task_id) {
            Slice::Visible | Slice::HeldOut | Slice::Impossible => true,
            Slice::Audit => self.audit,
        }
    }
}

/// A stable point in `[0, 1)` for a task id under a seed.
///
/// Written out rather than taken from `DefaultHasher`: this value decides which
/// measurements mean anything, and it has to be the same number next release.
/// `DefaultHasher` makes no such promise, and a silent repartition on a
/// toolchain bump would move tasks across the anchor with nothing failing.
///
/// FNV-1a accumulates, and then a finalizer mixes. The finalizer is not
/// optional here. FNV-1a's avalanche is poor in its high bits for short,
/// similar keys, which is exactly what task ids are: without the mix, `task:0`
/// through `task:9` all land in one slice and the shares come out half again
/// over what was asked for. The mix is MurmurHash3's `fmix64`, chosen because
/// it is a fixed, published constant set rather than something tuned here.
fn unit_hash(task_id: &str, seed: u64) -> f32 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h = OFFSET ^ seed;
    for byte in task_id.as_bytes() {
        h ^= u64::from(*byte);
        h = h.wrapping_mul(PRIME);
    }
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    h ^= h >> 33;
    // The top 24 bits, which is more resolution than any corpus needs and keeps
    // the division exact in f32.
    ((h >> 40) as f32) / ((1u32 << 24) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("task:{i}")).collect()
    }

    #[test]
    fn a_task_lands_in_the_same_slice_every_time_it_is_asked() {
        let p = Partition::default();
        for id in ids(200) {
            assert_eq!(p.of(&id), p.of(&id));
        }
        // And the answer does not depend on what else is in the corpus: a task
        // added today must not move the tasks measured yesterday.
        let before: Vec<Slice> = ids(50).iter().map(|id| p.of(id)).collect();
        let after: Vec<Slice> = ids(5000).iter().take(50).map(|id| p.of(id)).collect();
        assert_eq!(before, after);
    }

    /// The number that decides which measurements mean anything is pinned, so a
    /// toolchain or release that quietly changed it would fail here rather than
    /// move tasks across the anchor in silence.
    #[test]
    fn the_split_is_the_same_number_next_release() {
        let p = Partition::default();
        assert_eq!(p.of("task:0"), Slice::Visible);
        assert_eq!(p.of("task:1"), Slice::HeldOut);
        assert_eq!(p.of("task:19"), Slice::Audit);
        // A different seed is a different corpus, which is why the seed is
        // recorded with the generation that was measured under it.
        let other = Partition {
            seed: 7,
            ..Partition::default()
        };
        let moved = ids(400)
            .iter()
            .filter(|id| p.of(id) != other.of(id))
            .count();
        assert!(moved > 40, "a reseed repartitions, it does not nudge");
    }

    #[test]
    fn the_shares_come_out_roughly_where_they_were_asked_for() {
        let p = Partition::default();
        let corpus = ids(4000);
        let share = |s: Slice| {
            corpus.iter().filter(|id| p.of(id) == s).count() as f32 / corpus.len() as f32
        };
        assert!((share(Slice::HeldOut) - 0.20).abs() < 0.02, "held out");
        assert!((share(Slice::Audit) - 0.10).abs() < 0.02, "audited");
        assert!((share(Slice::Visible) - 0.70).abs() < 0.02, "visible");
        // Never drawn from the corpus: an unsatisfiable task is authored.
        assert_eq!(
            corpus
                .iter()
                .filter(|id| p.of(id) == Slice::Impossible)
                .count(),
            0
        );
    }

    #[test]
    fn only_the_visible_slice_is_something_a_decision_may_read() {
        assert!(Slice::Visible.selection_may_read());
        for closed in [Slice::HeldOut, Slice::Audit, Slice::Impossible] {
            assert!(
                !closed.selection_may_read(),
                "{} must not reach selection",
                closed.as_str()
            );
        }
    }

    #[test]
    fn a_holdout_learns_only_from_what_selection_may_read() {
        let due = Holdout {
            partition: Partition::default(),
            audit: true,
        };
        let not_due = Holdout {
            audit: false,
            ..due
        };
        // The ids eclipse's pinning test names: visible, held out, audited.
        assert!(not_due.learns_from("task:0"));
        assert!(!not_due.learns_from("task:1"), "held out");
        assert!(!not_due.learns_from("task:19"), "audited");
        assert!(not_due.measures("task:1"), "held out is measured every run");
        assert!(!not_due.measures("task:19"), "audit only when it is due");
        assert!(due.measures("task:19"));
        assert!(
            !due.learns_from("task:19"),
            "being measured is not being learned from"
        );
    }

    #[test]
    fn a_partition_that_leaves_nothing_to_train_on_is_refused() -> crate::Result<()> {
        assert!(Partition::new(0.9, 0.2, 0).is_err(), "over a whole");
        assert!(Partition::new(0.7, 0.3, 0).is_err(), "exactly a whole");
        assert!(Partition::new(-0.1, 0.1, 0).is_err(), "negative");
        assert!(Partition::new(f32::NAN, 0.1, 0).is_err(), "not a number");
        let ok = Partition::new(0.25, 0.05, 3)?;
        assert_eq!(ok.seed, 3);
        Ok(())
    }

    #[test]
    fn a_slice_survives_the_round_trip_it_takes_through_the_store() -> crate::Result<()> {
        for slice in [
            Slice::Visible,
            Slice::HeldOut,
            Slice::Audit,
            Slice::Impossible,
        ] {
            assert_eq!(slice.as_str().parse::<Slice>()?, slice);
        }
        assert!("training".parse::<Slice>().is_err());
        Ok(())
    }
}
