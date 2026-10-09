//! The SFT training objective: causal-LM cross-entropy over the
//! **completion** tokens only. The model predicts token `t+1` from position
//! `t`; we mask the prompt so the adapter is trained to *produce* the verified
//! winner, not to reconstruct the prompt. This is what `CausalLm::sft_step`
//! minimizes; it is model-agnostic and tested on CPU.

use candle_core::{Result, Tensor};
use candle_nn::ops::log_softmax;

/// Causal-LM masked cross-entropy.
///
/// - `logits`: `(batch, seq, vocab)`, where position `t` predicts token `t+1`.
/// - `input_ids`: `(batch, seq)` u32, the full prompt+completion token ids.
/// - `completion_mask`: `(batch, seq)` f32, `1.0` on completion tokens, `0.0`
///   on prompt (and padding) tokens.
///
/// Gradients flow into whatever produced `logits` (the LoRA adapter); the mask
/// and ids are constants.
pub fn causal_lm_loss(
    logits: &Tensor,
    input_ids: &Tensor,
    completion_mask: &Tensor,
) -> Result<Tensor> {
    let (_b, seq, _v) = logits.dims3()?;

    // Shift: predictions for positions 0..seq-1 supervise tokens 1..seq.
    let shift_logits = logits.narrow(1, 0, seq - 1)?; // (b, seq-1, vocab)
    let shift_labels = input_ids.narrow(1, 1, seq - 1)?.contiguous()?; // (b, seq-1)
    let shift_mask = completion_mask.narrow(1, 1, seq - 1)?; // (b, seq-1)

    let log_probs = log_softmax(&shift_logits, 2)?; // (b, seq-1, vocab)
                                                    // Negative log-likelihood of the realized next token at each position.
    let picked = log_probs
        .gather(&shift_labels.unsqueeze(2)?, 2)? // (b, seq-1, 1)
        .squeeze(2)?; // (b, seq-1)
    let nll = picked.neg()?;

    let masked = nll.mul(&shift_mask)?;
    // Average over the supervised (completion) tokens; the denominator is a
    // constant so dividing by it preserves the gradient through `masked`.
    let denom = shift_mask.sum_all()?.to_scalar::<f32>()?.max(1.0) as f64;
    masked.sum_all()?.affine(1.0 / denom, 0.0)
}

/// The Brier score: the training objective for a typed decision head, and a
/// **strictly proper scoring rule**.
///
/// That property is the whole reason this objective exists rather than any
/// other. Under a strictly proper rule the expected loss is minimized *only* by
/// reporting the true probability, so a head trained against it has no way to
/// score better by being confident than by being right. An accuracy objective,
/// or a cross-entropy over a hard label, buys a confident answer at the same
/// price as a calibrated one; this does not, and typed decisions turn on the
/// difference, since every one of them is thresholded and a miscalibrated
/// probability is worse than no probability at all.
///
/// Brier rather than the log score for one practical reason: it is BOUNDED.
/// A log score is unbounded below and hands a single confidently-wrong sample an
/// arbitrarily large gradient, which is how a rare mislabeled outcome comes to
/// dominate a batch. Both are strictly proper, so the bounded one is the safer
/// default under a verifier that is right nearly always rather than always.
///
/// - `probs`: `(batch, classes)`, each row a distribution that already sums to 1.
/// - `targets`: `(batch, classes)`, one-hot, or a soft label where the answer is
///   itself a distribution.
///
/// Returns the mean over the batch of the squared distance between them.
pub fn brier_loss(probs: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let (batch, _classes) = probs.dims2()?;
    let diff = (probs - targets)?;
    // Sum the squared error across classes, then average over the batch. Gradients
    // flow into whatever produced `probs`; `targets` are constants.
    (diff.sqr()?.sum_all()?).affine(1.0 / batch.max(1) as f64, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    // logits for (batch=1, seq=3, vocab=4); only positions 0,1 are scored.
    fn logits(values: Vec<f32>, dev: &Device) -> Tensor {
        Tensor::from_vec(values, (1, 3, 4), dev).unwrap()
    }

    #[test]
    fn rewards_predicting_the_completion_tokens() {
        let dev = Device::Cpu;
        // input ids: prompt=[1], completion=[2,3]
        let input_ids = Tensor::from_vec(vec![1u32, 2, 3], (1, 3), &dev).unwrap();
        let mask = Tensor::from_vec(vec![0f32, 1.0, 1.0], (1, 3), &dev).unwrap();

        // GOOD: pos0 favors token 2, pos1 favors token 3 (the realized next tokens).
        let good = logits(
            vec![
                0., 0., 10., 0., // pos0 -> predicts token 2
                0., 0., 0., 10., // pos1 -> predicts token 3
                0., 0., 0., 0., // pos2 (unused)
            ],
            &dev,
        );
        // BAD: predicts the wrong tokens.
        let bad = logits(
            vec![
                10., 0., 0., 0., // pos0 -> predicts token 0 (wrong)
                10., 0., 0., 0., // pos1 -> predicts token 0 (wrong)
                0., 0., 0., 0.,
            ],
            &dev,
        );

        let good_loss = causal_lm_loss(&good, &input_ids, &mask)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let bad_loss = causal_lm_loss(&bad, &input_ids, &mask)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();

        assert!(
            good_loss < 0.01,
            "confident-correct loss should be tiny: {good_loss}"
        );
        assert!(
            bad_loss > good_loss,
            "wrong predictions cost more: {bad_loss} vs {good_loss}"
        );
    }

    #[test]
    fn prompt_tokens_are_not_supervised() {
        let dev = Device::Cpu;
        let input_ids = Tensor::from_vec(vec![1u32, 2, 3], (1, 3), &dev).unwrap();

        // mask only the LAST token as completion; pos0's prediction (of token 2)
        // is now prompt and must not contribute.
        let mask_tail = Tensor::from_vec(vec![0f32, 0.0, 1.0], (1, 3), &dev).unwrap();
        // pos1 favors token 3 (good for the scored position); pos0 is wrong but unscored.
        let lg = logits(
            vec![
                10., 0., 0., 0., // pos0 wrong, but masked out
                0., 0., 0., 10., // pos1 -> token 3 (the only supervised target)
                0., 0., 0., 0.,
            ],
            &dev,
        );
        let loss = causal_lm_loss(&lg, &input_ids, &mask_tail)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            loss < 0.01,
            "only the completion token should be scored: {loss}"
        );
    }
}

/// The calibration objective for typed decisions, and the property that makes
/// it the right one. These are the tests the design rests on: it claims a head
/// can be trained to report true probabilities, and that claim is only as good
/// as the objective being strictly proper.
#[cfg(test)]
mod proper_scoring {
    use super::*;
    use candle_core::{Device, Tensor};

    fn row(values: &[f32], dev: &Device) -> Tensor {
        Tensor::from_vec(values.to_vec(), (1, values.len()), dev).unwrap()
    }

    /// The expected Brier loss of *reporting* `q` when the world is `p`, which
    /// is what a truthful-reporting argument is actually about. A decision head
    /// does not see one outcome; it sees many draws from `p` and is scored on
    /// all of them, so the question is which `q` minimizes the average.
    fn expected_loss(q: f32, p: f32, dev: &Device) -> f32 {
        let reported = row(&[q, 1.0 - q], dev);
        // outcome A, with probability p
        let a = brier_loss(&reported, &row(&[1.0, 0.0], dev))
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        // outcome B, with probability 1 - p
        let b = brier_loss(&reported, &row(&[0.0, 1.0], dev))
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        p * a + (1.0 - p) * b
    }

    /// **Strict properness**: the true report is the unique minimum. Sweep every report
    /// from 0 to 1 against a world that says 0.7, and the best report is 0.7.
    ///
    /// This is the property typed decisions need and the reason the objective is
    /// not simply accuracy: there is no report that scores better than the truth, so
    /// a head cannot buy a lower loss with confidence it has not earned.
    #[test]
    fn the_true_report_is_the_unique_minimum() {
        let dev = Device::Cpu;
        let truth = 0.7f32;
        let (mut best_q, mut best) = (0.0f32, f32::MAX);
        for step in 0..=100 {
            let q = step as f32 / 100.0;
            let loss = expected_loss(q, truth, &dev);
            if loss < best {
                best = loss;
                best_q = q;
            }
        }
        assert!(
            (best_q - truth).abs() < 0.011,
            "the best report must be the true probability, got {best_q} for a world of {truth}"
        );
    }

    /// Overconfidence is punished, and so is underconfidence. A rule that only
    /// punished one would be a bias dressed as a loss.
    #[test]
    fn confidence_beyond_the_evidence_costs_more_than_the_truth() {
        let dev = Device::Cpu;
        let truth = 0.7f32;
        let truthful = expected_loss(truth, truth, &dev);
        assert!(
            expected_loss(0.95, truth, &dev) > truthful,
            "claiming 0.95 when the world is 0.70 must cost more than saying 0.70"
        );
        assert!(
            expected_loss(0.50, truth, &dev) > truthful,
            "hedging to 0.50 when the world is 0.70 must also cost more"
        );
    }

    /// A perfect call is free and a confident miss is expensive, which is what
    /// makes the number usable as a reward rather than only as a loss.
    #[test]
    fn a_right_answer_costs_nothing_and_a_confident_miss_costs_most() {
        let dev = Device::Cpu;
        let certain_right = brier_loss(&row(&[1.0, 0.0], &dev), &row(&[1.0, 0.0], &dev))
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let certain_wrong = brier_loss(&row(&[0.0, 1.0], &dev), &row(&[1.0, 0.0], &dev))
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(certain_right.abs() < 1e-6, "a correct certainty is free");
        // Bounded: two classes, each off by one, so the worst case is exactly 2.
        assert!(
            (certain_wrong - 2.0).abs() < 1e-6,
            "a confident miss is bounded at 2.0, which is why this is safer than a log score"
        );
    }

    /// Averaged over a batch, so one sample cannot set the scale.
    #[test]
    fn the_loss_is_a_mean_over_the_batch() {
        let dev = Device::Cpu;
        let probs = Tensor::from_vec(vec![1.0f32, 0.0, 1.0, 0.0], (2, 2), &dev).unwrap();
        let targets = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &dev).unwrap();
        let loss = brier_loss(&probs, &targets)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        // One perfect row (0.0) and one confident miss (2.0), averaged.
        assert!(
            (loss - 1.0).abs() < 1e-6,
            "expected the mean 1.0, got {loss}"
        );
    }

    /// A soft target is a distribution, not a mistake: a verifier that says "this
    /// is right 70% of the time" is exactly what a calibrated head should learn,
    /// and the objective has to accept it without special-casing.
    #[test]
    fn a_soft_target_is_scored_like_any_other_distribution() {
        let dev = Device::Cpu;
        let matched = brier_loss(&row(&[0.7, 0.3], &dev), &row(&[0.7, 0.3], &dev))
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            matched.abs() < 1e-6,
            "matching a soft target exactly is free"
        );
    }
}
