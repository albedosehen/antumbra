//! Gradient accumulation: one optimizer step on the mean gradient of several
//! losses, each backpropagated as soon as it is computed.
//!
//! Stepping on the mean of the losses gives the same gradient, but it keeps
//! every loss's forward graph alive until the one backward, so memory grows
//! with the batch: two examples of workbench code already run a 24 GB card out.
//! Here each forward graph is dropped once its backward has run, and a batch of
//! any size costs one example's graph plus the summed adapter gradients.
//!
//! The sum is kept in the first loss's gradient store, keyed by the trained
//! variables themselves: the optimizer finds a gradient by its variable's
//! identity, so a store keyed by anything else would step nothing.

use candle_core::backprop::GradStore;
use candle_core::{Result, Tensor, Var};
use candle_nn::Optimizer;

/// The gradients of several losses, summed per variable.
#[derive(Default)]
pub struct GradSum {
    store: Option<GradStore>,
    count: usize,
}

impl GradSum {
    pub fn new() -> Self {
        Self::default()
    }

    /// Backpropagate `loss` and add its gradients for `vars` to the sum. The
    /// loss's forward graph is free to go as soon as this returns.
    pub fn add(&mut self, loss: &Tensor, vars: &[Var]) -> Result<()> {
        let grads = loss.backward()?;
        match &mut self.store {
            None => self.store = Some(grads),
            Some(sum) => {
                for var in vars {
                    if let Some(g) = grads.get(var) {
                        let total = match sum.get(var) {
                            Some(s) => (s + g)?,
                            None => g.clone(),
                        };
                        sum.insert(var, total);
                    }
                }
            }
        }
        self.count += 1;
        Ok(())
    }

    /// How many losses have been added.
    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Step `opt` once on the mean of the summed gradients. Returns whether it
    /// stepped: an empty sum leaves everything as it was.
    pub fn step<O: Optimizer>(self, opt: &mut O, vars: &[Var]) -> Result<bool> {
        let Some(mut sum) = self.store else {
            return Ok(false);
        };
        if self.count > 1 {
            let scale = 1.0 / self.count as f64;
            for var in vars {
                if let Some(g) = sum.remove(var) {
                    sum.insert(var, g.affine(scale, 0.0)?);
                }
            }
        }
        opt.step(&sum)?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::{AdamW, ParamsAdamW, SGD};

    fn var(values: &[f32]) -> Var {
        Var::from_slice(values, values.len(), &Device::Cpu).unwrap()
    }

    fn values(v: &Var) -> Vec<f32> {
        v.as_tensor().to_vec1().unwrap()
    }

    /// `sum((w * x - y)^2)`.
    fn loss(w: &Var, x: &[f32], y: &[f32]) -> Tensor {
        let x = Tensor::new(x, &Device::Cpu).unwrap();
        let y = Tensor::new(y, &Device::Cpu).unwrap();
        ((w.as_tensor() * x).unwrap() - y)
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
    }

    const EXAMPLES: [([f32; 3], [f32; 3]); 3] = [
        ([1.0, 2.0, 3.0], [0.5, 0.0, 1.0]),
        ([-1.0, 0.5, 2.0], [1.0, 1.0, -1.0]),
        ([0.0, 1.0, -2.0], [2.0, -1.0, 0.5]),
    ];

    fn close(a: &[f32], b: &[f32]) -> bool {
        a.iter().zip(b).all(|(p, q)| (p - q).abs() < 1e-5)
    }

    /// The accumulated step is the step on the mean loss.
    #[test]
    fn the_step_is_the_mean_losss_step() {
        let start = [0.3f32, -0.2, 0.7];

        let a = var(&start);
        let mut opt = SGD::new(vec![a.clone()], 0.1).unwrap();
        let mean = EXAMPLES
            .iter()
            .map(|(x, y)| loss(&a, x, y))
            .reduce(|p, q| (p + q).unwrap())
            .unwrap()
            .affine(1.0 / EXAMPLES.len() as f64, 0.0)
            .unwrap();
        opt.backward_step(&mean).unwrap();

        let b = var(&start);
        let vars = vec![b.clone()];
        let mut opt = SGD::new(vars.clone(), 0.1).unwrap();
        let mut sum = GradSum::new();
        for (x, y) in &EXAMPLES {
            sum.add(&loss(&b, x, y), &vars).unwrap();
        }
        assert_eq!(sum.len(), 3);
        assert!(sum.step(&mut opt, &vars).unwrap());

        assert!(!close(&values(&a), &start), "the reference moved");
        assert!(
            close(&values(&a), &values(&b)),
            "{:?} vs {:?}",
            values(&a),
            values(&b)
        );
    }

    /// A variable one loss does not reach still gets the others' gradients,
    /// averaged over every loss added.
    #[test]
    fn a_variable_missing_from_the_first_loss_is_still_summed() {
        let (a, b) = (var(&[1.0]), var(&[1.0]));
        let vars = vec![a.clone(), b.clone()];
        let mut opt = SGD::new(vars.clone(), 1.0).unwrap();
        let mut sum = GradSum::new();
        // d/da (2a) = 2 in the first, d/db (4b) = 4 in the second.
        sum.add(
            &a.as_tensor().affine(2.0, 0.0).unwrap().sum_all().unwrap(),
            &vars,
        )
        .unwrap();
        sum.add(
            &b.as_tensor().affine(4.0, 0.0).unwrap().sum_all().unwrap(),
            &vars,
        )
        .unwrap();
        sum.step(&mut opt, &vars).unwrap();
        assert!(close(&values(&a), &[0.0]), "{:?}", values(&a));
        assert!(close(&values(&b), &[-1.0]), "{:?}", values(&b));
    }

    /// Under the trainer's optimizer too, the step reaches the variables.
    #[test]
    fn an_adamw_step_moves_the_variables() {
        let start = [0.3f32, -0.2, 0.7];
        let w = var(&start);
        let vars = vec![w.clone()];
        let mut opt = AdamW::new(
            vars.clone(),
            ParamsAdamW {
                lr: 0.01,
                ..Default::default()
            },
        )
        .unwrap();
        let mut sum = GradSum::new();
        for (x, y) in &EXAMPLES[..2] {
            sum.add(&loss(&w, x, y), &vars).unwrap();
        }
        sum.step(&mut opt, &vars).unwrap();
        assert!(values(&w)
            .iter()
            .zip(start)
            .all(|(v, s)| (v - s).abs() > 1e-4));
    }

    #[test]
    fn an_empty_sum_steps_nothing() {
        let w = var(&[0.5]);
        let vars = vec![w.clone()];
        let mut opt = SGD::new(vars.clone(), 1.0).unwrap();
        let sum = GradSum::new();
        assert!(sum.is_empty());
        assert!(!sum.step(&mut opt, &vars).unwrap());
        assert_eq!(values(&w), [0.5]);
    }
}
