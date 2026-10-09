//! The typed decision head (ADR-0024): two layers over a frozen encoder,
//! trained against a strictly proper scoring rule on verifier outcomes.
//!
//! This is the half of a [`TypedDecider`](antumbra_core::ports::TypedDecider)
//! that Antumbra owns. The encoder is a general sentence model and comes from
//! elsewhere; what makes the answer *calibrated* is this head and the objective
//! it is trained against, and neither of those is something to rent.
//!
//! Deliberately here rather than in `antumbra-serve`: a head only exists to be
//! trained, `antumbra-train` already writes losses against candle tensors, and
//! candle is an ungated dependency here — so the head and its training step are
//! testable on CPU with no feature flag and no model download. The encoder that
//! eventually feeds it is the part that needs weights.
//!
//! The head is small on purpose. Two linear layers over a pooled vector is what
//! the reference design uses, and the expensive, slow, hard-to-audit part of a
//! decision model is the encoder underneath it rather than this.

use candle_core::{Result, Tensor};
use candle_nn::ops::softmax;
use candle_nn::{linear, Linear, Module, VarBuilder};

use crate::objective::brier_loss;

/// Two linear layers with a non-linearity between them, mapping a pooled encoder
/// vector to a distribution over a question's answer space.
///
/// One layer would be a linear probe and could not represent "these two options
/// are both plausible for different reasons", which is the distinction
/// [`Question::Choice`](antumbra_core::ports::Question) exists to express. More
/// than two buys little over a frozen encoder and costs the thing that makes
/// this affordable in a serving path.
pub struct DecisionHead {
    hidden: Linear,
    out: Linear,
}

impl DecisionHead {
    /// Build a head mapping `encoder_dim` to `answers` logits through a hidden
    /// layer of `hidden_dim`.
    pub fn new(
        encoder_dim: usize,
        hidden_dim: usize,
        answers: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            hidden: linear(encoder_dim, hidden_dim, vb.pp("hidden"))?,
            out: linear(hidden_dim, answers, vb.pp("out"))?,
        })
    }

    /// Unnormalized scores over the answer space, `(batch, answers)`.
    pub fn logits(&self, pooled: &Tensor) -> Result<Tensor> {
        self.out.forward(&self.hidden.forward(pooled)?.gelu()?)
    }

    /// The distribution the caller actually reads.
    ///
    /// Probabilities rather than logits at the boundary, because every consumer
    /// of this head thresholds the result, and a threshold on an unnormalized
    /// score is the defect ADR-0024 exists to remove.
    pub fn probs(&self, pooled: &Tensor) -> Result<Tensor> {
        softmax(&self.logits(pooled)?, 1)
    }

    /// The loss to minimize: [`brier_loss`] between this head's distribution and
    /// what a verifier said happened.
    ///
    /// `targets` is `(batch, answers)`, one-hot from a verifier's verdict or soft
    /// where the verifier reports a rate. Using a strictly proper rule here is
    /// what makes the head's output a probability rather than a confidence: it
    /// has no way to score better by overstating certainty.
    pub fn loss(&self, pooled: &Tensor, targets: &Tensor) -> Result<Tensor> {
        brier_loss(&self.probs(pooled)?, targets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};

    const ENCODER_DIM: usize = 8;
    const HIDDEN: usize = 16;

    fn head(answers: usize, dev: &Device) -> (DecisionHead, VarMap) {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, dev);
        let h = DecisionHead::new(ENCODER_DIM, HIDDEN, answers, vb).unwrap();
        (h, varmap)
    }

    /// Whatever the weights, the output is a distribution. A caller reads this as
    /// a probability and thresholds it, so this is the property everything else
    /// depends on.
    #[test]
    fn the_output_is_a_distribution() {
        let dev = Device::Cpu;
        let (h, _vm) = head(3, &dev);
        let pooled = Tensor::randn(0f32, 1.0, (4, ENCODER_DIM), &dev).unwrap();
        let probs = h.probs(&pooled).unwrap();
        assert_eq!(probs.dims2().unwrap(), (4, 3));
        let sums = probs.sum(1).unwrap().to_vec1::<f32>().unwrap();
        for s in sums {
            assert!((s - 1.0).abs() < 1e-5, "each row must sum to 1, got {s}");
        }
        let flat = probs.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            flat.iter().all(|p| (0.0..=1.0).contains(p)),
            "every probability is in [0, 1]"
        );
    }

    /// The head learns. A toy problem the encoder dimension can separate, trained
    /// against the Brier objective, must drive the loss down and end up answering
    /// correctly -- otherwise the wiring is decorative.
    #[test]
    fn it_learns_a_separable_decision_from_verifier_labels() {
        let dev = Device::Cpu;
        let (h, varmap) = head(2, &dev);
        // Two clusters: first feature positive means answer 0, negative means 1.
        // Stands in for "this memory answers the query" / "it does not".
        let xs = Tensor::from_vec(
            vec![
                1.0f32, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // -> 0
                0.9, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // -> 0
                -1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // -> 1
                -0.8, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // -> 1
            ],
            (4, ENCODER_DIM),
            &dev,
        )
        .unwrap();
        let ys = Tensor::from_vec(
            vec![1.0f32, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            (4, 2),
            &dev,
        )
        .unwrap();

        let before = h.loss(&xs, &ys).unwrap().to_scalar::<f32>().unwrap();
        let mut opt = AdamW::new(
            varmap.all_vars(),
            ParamsAdamW {
                lr: 0.05,
                ..Default::default()
            },
        )
        .unwrap();
        for _ in 0..300 {
            let loss = h.loss(&xs, &ys).unwrap();
            opt.backward_step(&loss).unwrap();
        }
        let after = h.loss(&xs, &ys).unwrap().to_scalar::<f32>().unwrap();
        assert!(
            after < before,
            "training must reduce the loss: {before} -> {after}"
        );
        assert!(
            after < 0.1,
            "a separable problem should be learned: {after}"
        );

        // And the answers are right, not merely low-loss.
        let probs = h.probs(&xs).unwrap().to_vec2::<f32>().unwrap();
        assert!(probs[0][0] > 0.5 && probs[1][0] > 0.5, "first cluster -> 0");
        assert!(
            probs[2][1] > 0.5 && probs[3][1] > 0.5,
            "second cluster -> 1"
        );
    }

    /// Trained on a soft label, the head reports the RATE rather than collapsing
    /// to the majority answer. This is the behavior the proper scoring rule is
    /// chosen for, and the reason the head can be thresholded at all: a verifier
    /// that says "right about 70% of the time" should produce ~0.7, not 1.0.
    #[test]
    fn a_soft_label_is_learned_as_a_rate_and_not_rounded_up() {
        let dev = Device::Cpu;
        let (h, varmap) = head(2, &dev);
        // A unit vector, as a pooled embedding is. A vector of ones is 20 times
        // longer, and from about 3 random starts in 100 the head overshot to
        // p = 1.0 on it (seen in CI on 2026-10-04, then 9 runs in 300 locally).
        let unit = 1.0 / (ENCODER_DIM as f32).sqrt();
        let xs = Tensor::from_vec(vec![unit; ENCODER_DIM], (1, ENCODER_DIM), &dev).unwrap();
        let ys = Tensor::from_vec(vec![0.7f32, 0.3], (1, 2), &dev).unwrap();

        let mut opt = AdamW::new(
            varmap.all_vars(),
            ParamsAdamW {
                lr: 0.05,
                ..Default::default()
            },
        )
        .unwrap();
        for _ in 0..400 {
            let loss = h.loss(&xs, &ys).unwrap();
            opt.backward_step(&loss).unwrap();
        }
        let p = h.probs(&xs).unwrap().to_vec2::<f32>().unwrap()[0][0];
        assert!(
            (p - 0.7).abs() < 0.05,
            "the head must report the rate it was shown, got {p} for a label of 0.7"
        );
    }

    /// A `Noul` is a two-answer question, so the same head serves it: the
    /// probability the caller floors on is simply the first column.
    #[test]
    fn a_noul_is_a_two_answer_head() {
        let dev = Device::Cpu;
        let (h, _vm) = head(2, &dev);
        let pooled = Tensor::randn(0f32, 1.0, (1, ENCODER_DIM), &dev).unwrap();
        let p = h.probs(&pooled).unwrap().to_vec2::<f32>().unwrap()[0].clone();
        assert_eq!(p.len(), 2);
        assert!((p[0] + p[1] - 1.0).abs() < 1e-5);
    }
}
