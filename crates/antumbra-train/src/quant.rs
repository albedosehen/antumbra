//! How far a quantized base moves an expert from the dense one.
//!
//! An expert is trained over the dense base and may be served over a quantized
//! copy of it. The adapter is the same; the base under it is not, so its
//! answers can shift. This measures the shift the way it shows: teacher-force
//! the same text through the expert on each base, and compare, position by
//! position, the log-probability each gives the token that is there and the
//! token each ranks first. A shift in log-probability that never changes the
//! first-ranked token changes no greedy answer; a changed first-ranked token is
//! a decision that went another way.

/// One position of a teacher-forced pass: the log-probability the model gave
/// the token that is there, and the token it ranked first.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Reading {
    pub logprob: f32,
    pub top1: u32,
}

/// The shift between two bases over every position compared so far.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Drift {
    positions: usize,
    shift_sum: f64,
    largest: f64,
    agree: usize,
}

impl Drift {
    /// Add one text's readings: `reference` from the dense base, `other` from
    /// the base being judged, position for position. Readings past the shorter
    /// of the two are not compared.
    pub fn add(&mut self, reference: &[Reading], other: &[Reading]) {
        for (r, o) in reference.iter().zip(other) {
            let shift = f64::from((r.logprob - o.logprob).abs());
            self.positions += 1;
            self.shift_sum += shift;
            self.largest = self.largest.max(shift);
            self.agree += usize::from(r.top1 == o.top1);
        }
    }

    /// How many positions were compared.
    pub fn positions(&self) -> usize {
        self.positions
    }

    /// The mean absolute shift in log-probability, in nats. Zero over nothing.
    pub fn mean_shift(&self) -> f64 {
        if self.positions == 0 {
            return 0.0;
        }
        self.shift_sum / self.positions as f64
    }

    /// The largest shift at any one position.
    pub fn largest_shift(&self) -> f64 {
        self.largest
    }

    /// The share of positions where both bases rank the same token first. One
    /// over nothing: no decision was seen to change.
    pub fn top1_agreement(&self) -> f64 {
        if self.positions == 0 {
            return 1.0;
        }
        self.agree as f64 / self.positions as f64
    }

    /// How many positions rank a different token first.
    pub fn flips(&self) -> usize {
        self.positions - self.agree
    }

    /// As an expert's card holds it: how many positions were compared, the
    /// mean and the largest shift in nats, and how many rank another token
    /// first.
    pub fn to_card(&self) -> serde_json::Value {
        serde_json::json!({
            "positions": self.positions(),
            "mean_shift": self.mean_shift(),
            "largest_shift": self.largest_shift(),
            "flips": self.flips(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn readings(rows: &[(f32, u32)]) -> Vec<Reading> {
        rows.iter()
            .map(|&(logprob, top1)| Reading { logprob, top1 })
            .collect()
    }

    #[test]
    fn a_base_that_reads_the_same_has_not_drifted() {
        let dense = readings(&[(-0.1, 7), (-2.0, 9)]);
        let mut drift = Drift::default();
        drift.add(&dense, &dense);
        assert_eq!(drift.positions(), 2);
        assert_eq!(drift.mean_shift(), 0.0);
        assert_eq!(drift.top1_agreement(), 1.0);
        assert_eq!(drift.flips(), 0);
    }

    #[test]
    fn shift_and_flips_are_counted_over_every_text() {
        let mut drift = Drift::default();
        drift.add(
            &readings(&[(-0.5, 1), (-1.0, 2)]),
            &readings(&[(-0.75, 1), (-1.5, 3)]),
        );
        drift.add(&readings(&[(-0.25, 4)]), &readings(&[(-0.25, 4)]));
        assert_eq!(drift.positions(), 3);
        assert!((drift.mean_shift() - 0.25).abs() < 1e-6);
        assert!((drift.largest_shift() - 0.5).abs() < 1e-6);
        assert_eq!(drift.flips(), 1);
        assert!((drift.top1_agreement() - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn nothing_compared_is_no_shift_and_no_flip() {
        let drift = Drift::default();
        assert_eq!(drift.mean_shift(), 0.0);
        assert_eq!(drift.top1_agreement(), 1.0);
        // Only the positions both passes hold are compared.
        let mut uneven = Drift::default();
        uneven.add(&readings(&[(-1.0, 1), (-1.0, 2)]), &readings(&[(-1.0, 1)]));
        assert_eq!(uneven.positions(), 1);
    }
}
