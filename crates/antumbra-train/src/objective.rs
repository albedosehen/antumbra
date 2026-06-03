//! The SFT training objective (ADR-0010): causal-LM cross-entropy over the
//! **completion** tokens only. The model predicts token `t+1` from position
//! `t`; we mask the prompt so the adapter is trained to *produce* the verified
//! winner, not to reconstruct the prompt. This is what `CausalLm::sft_step`
//! minimizes; it is model-agnostic and tested on CPU.

use candle_core::{Result, Tensor};
use candle_nn::ops::log_softmax;

/// Causal-LM masked cross-entropy.
///
/// - `logits`: `(batch, seq, vocab)` — position `t` predicts token `t+1`.
/// - `input_ids`: `(batch, seq)` u32 — the full prompt+completion token ids.
/// - `completion_mask`: `(batch, seq)` f32 — `1.0` on completion tokens, `0.0`
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
