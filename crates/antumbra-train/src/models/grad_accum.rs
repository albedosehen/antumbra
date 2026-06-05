//! Mini-batch gradient accumulation over a candle [`GradStore`].
//!
//! Batch-of-1 SGD (one optimizer step per example) is noisy: each step chases a
//! single example's gradient, so the LoRA lurches toward whatever it saw last and
//! a usable learning rate has to be small. Accumulating the whole batch's
//! gradients and applying their *mean* in one step is true mini-batch descent --
//! a far less noisy update, stable at a higher learning rate ("Beware of the
//! Batch Size"). These are the two building blocks: sum per-parameter gradients
//! across examples, then scale the sum to a mean before the step.

use candle_core::backprop::GradStore;
use candle_core::Result as CResult;

/// Add every per-tensor gradient in `src` into `acc`, keyed by tensor id, so the
/// same parameter's gradients from different examples sum into one entry.
pub fn accumulate_into(acc: &mut GradStore, src: &GradStore) -> CResult<()> {
    let ids: Vec<_> = src.get_ids().copied().collect();
    for id in ids {
        if let Some(g) = src.get_id(id) {
            let summed = match acc.get_id(id) {
                Some(prev) => (prev + g)?,
                None => g.clone(),
            };
            acc.insert_id(id, summed);
        }
    }
    Ok(())
}

/// Scale every gradient in `acc` in place -- used to turn an accumulated *sum*
/// into the *mean* (scale = `1 / batch`) before the optimizer step.
pub fn scale_grads(acc: &mut GradStore, scale: f64) -> CResult<()> {
    let ids: Vec<_> = acc.get_ids().copied().collect();
    for id in ids {
        if let Some(g) = acc.get_id(id) {
            let scaled = g.affine(scale, 0.0)?;
            acc.insert_id(id, scaled);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var};

    // Accumulating two examples' gradients and averaging must equal the analytic
    // mean of the per-example gradients -- the core correctness property of
    // mini-batch accumulation.
    #[test]
    fn accumulate_then_average_is_the_mean_gradient() {
        let dev = Device::Cpu;
        let w = Var::new(&[2.0f32, 3.0], &dev).unwrap();
        let wt = w.as_tensor();

        // Example A: loss = sum(w^2) -> grad = 2w = [4, 6].
        let loss_a = wt.sqr().unwrap().sum_all().unwrap();
        let g_a = loss_a.backward().unwrap();
        // Example B: loss = sum(3w) -> grad = 3 = [3, 3].
        let loss_b = wt.affine(3.0, 0.0).unwrap().sum_all().unwrap();
        let g_b = loss_b.backward().unwrap();

        let mut acc = g_a;
        accumulate_into(&mut acc, &g_b).unwrap(); // sum = [7, 9]
        scale_grads(&mut acc, 0.5).unwrap(); // mean = [3.5, 4.5]

        let got: Vec<f32> = acc.get(wt).unwrap().to_vec1().unwrap();
        assert!(
            (got[0] - 3.5).abs() < 1e-5 && (got[1] - 4.5).abs() < 1e-5,
            "expected mean gradient [3.5, 4.5], got {got:?}"
        );
    }

    // A single example accumulated alone, then "averaged" by 1.0, is unchanged --
    // the batch-of-one degenerate case must match plain backward.
    #[test]
    fn single_example_is_unchanged() {
        let dev = Device::Cpu;
        let w = Var::new(&[5.0f32, -1.0], &dev).unwrap();
        let wt = w.as_tensor();
        let loss = wt.sqr().unwrap().sum_all().unwrap(); // grad = 2w = [10, -2]
        let mut acc = loss.backward().unwrap();
        scale_grads(&mut acc, 1.0).unwrap();
        let got: Vec<f32> = acc.get(wt).unwrap().to_vec1().unwrap();
        assert!(
            (got[0] - 10.0).abs() < 1e-5 && (got[1] + 2.0).abs() < 1e-5,
            "expected [10, -2], got {got:?}"
        );
    }
}
