//! The model behind the recipe search: a Gaussian process over where a recipe
//! sits in the search space and the generation it ran in, predicting fitness.
//!
//! Population Based Bandits (Parker-Holder et al., NeurIPS 2020) fits exactly
//! this and picks the next recipes by an upper confidence bound on it. The time
//! term is theirs, taken from time-varying GP-UCB (Bogunovic et al., 2016): the
//! covariance between two runs decays as `(1 - decay)^(|t - t'| / 2)`, so an old
//! generation's fitness still counts but counts for less, because the corpus and
//! the population a recipe is judged against both move under it.
//!
//! The observations are few (a cohort of four to eight a generation), so the
//! linear algebra is a dense Cholesky solve written out here rather than a
//! dependency.

/// One observed run: the recipe as a point in the unit cube, the generation it
/// ran in, and the fitness it scored.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Sample {
    pub x: Vec<f64>,
    pub t: f64,
    pub y: f64,
}

/// A squared-exponential kernel over the recipe, times the time decay.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Kernel {
    /// How far apart, in the unit cube, two recipes can sit and still inform
    /// each other.
    pub lengthscale: f64,
    /// Observation noise, as a share of the standardized fitness variance.
    pub noise: f64,
    /// Per-generation decay of the covariance between runs.
    pub decay: f64,
}

impl Kernel {
    fn cov(&self, a: &Sample, b_x: &[f64], b_t: f64) -> f64 {
        let d2: f64 = a.x.iter().zip(b_x).map(|(p, q)| (p - q).powi(2)).sum();
        let space = (-d2 / (2.0 * self.lengthscale.powi(2))).exp();
        let time = (1.0 - self.decay).powf((a.t - b_t).abs() / 2.0);
        space * time
    }
}

/// The fitted process: what it predicts for a recipe in a generation.
#[derive(Debug, Clone)]
pub(crate) struct Posterior {
    kernel: Kernel,
    samples: Vec<Sample>,
    /// Lower Cholesky factor of the noisy covariance of the samples.
    chol: Vec<Vec<f64>>,
    /// The covariance's inverse applied to the standardized fitnesses.
    alpha: Vec<f64>,
    y_mean: f64,
    y_scale: f64,
}

/// Fit the process to `samples`. `None` with nothing to fit, or when the
/// covariance cannot be factored even with jitter, which only a degenerate
/// input produces.
pub(crate) fn fit(kernel: Kernel, samples: &[Sample]) -> Option<Posterior> {
    if samples.is_empty() {
        return None;
    }
    let n = samples.len() as f64;
    let y_mean = samples.iter().map(|s| s.y).sum::<f64>() / n;
    let var = samples.iter().map(|s| (s.y - y_mean).powi(2)).sum::<f64>() / n;
    // A constant set of fitnesses has no scale to standardize by; any positive
    // one leaves the prediction at that constant.
    let y_scale = if var > 1e-12 { var.sqrt() } else { 1.0 };
    let ys: Vec<f64> = samples.iter().map(|s| (s.y - y_mean) / y_scale).collect();

    let mut jitter = 0.0;
    for _ in 0..6 {
        let k: Vec<Vec<f64>> = samples
            .iter()
            .enumerate()
            .map(|(i, a)| {
                samples
                    .iter()
                    .enumerate()
                    .map(|(j, b)| {
                        let c = kernel.cov(a, &b.x, b.t);
                        if i == j {
                            c + kernel.noise + jitter
                        } else {
                            c
                        }
                    })
                    .collect()
            })
            .collect();
        if let Some(chol) = cholesky(&k) {
            let alpha = solve_upper(&chol, &solve_lower(&chol, &ys));
            return Some(Posterior {
                kernel,
                samples: samples.to_vec(),
                chol,
                alpha,
                y_mean,
                y_scale,
            });
        }
        jitter = if jitter == 0.0 { 1e-8 } else { jitter * 100.0 };
    }
    None
}

impl Posterior {
    /// The predicted fitness of a recipe at `x` run in generation `t`, and its
    /// standard deviation, in fitness units.
    pub(crate) fn predict(&self, x: &[f64], t: f64) -> (f64, f64) {
        let ks: Vec<f64> = self
            .samples
            .iter()
            .map(|s| self.kernel.cov(s, x, t))
            .collect();
        let mean: f64 = ks.iter().zip(&self.alpha).map(|(k, a)| k * a).sum();
        let v = solve_lower(&self.chol, &ks);
        let var = (1.0 - v.iter().map(|x| x * x).sum::<f64>()).max(0.0);
        (self.y_mean + self.y_scale * mean, self.y_scale * var.sqrt())
    }
}

/// Lower-triangular `L` with `L Lᵀ = a`, or `None` when `a` is not positive
/// definite.
fn cholesky(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let sum: f64 = (0..j).map(|k| l[i][k] * l[j][k]).sum();
            if i == j {
                let d = a[i][i] - sum;
                if d <= 0.0 || !d.is_finite() {
                    return None;
                }
                l[i][j] = d.sqrt();
            } else {
                l[i][j] = (a[i][j] - sum) / l[j][j];
            }
        }
    }
    Some(l)
}

/// Solve `L z = b` for lower-triangular `L`.
fn solve_lower(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let mut z = vec![0.0; b.len()];
    for i in 0..b.len() {
        let sum: f64 = (0..i).map(|k| l[i][k] * z[k]).sum();
        z[i] = (b[i] - sum) / l[i][i];
    }
    z
}

/// Solve `Lᵀ x = z` for lower-triangular `L`.
fn solve_upper(l: &[Vec<f64>], z: &[f64]) -> Vec<f64> {
    let n = z.len();
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let sum: f64 = (i + 1..n).map(|k| l[k][i] * x[k]).sum();
        x[i] = (z[i] - sum) / l[i][i];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    const KERNEL: Kernel = Kernel {
        lengthscale: 0.3,
        noise: 1e-4,
        decay: 0.1,
    };

    fn at(x: &[f64], t: f64, y: f64) -> Sample {
        Sample {
            x: x.to_vec(),
            t,
            y,
        }
    }

    #[test]
    fn it_passes_close_to_what_it_saw_and_is_sure_of_it() {
        let seen = [at(&[0.2, 0.2], 0.0, 0.3), at(&[0.8, 0.7], 0.0, 0.9)];
        let gp = fit(KERNEL, &seen).expect("fits");
        for s in &seen {
            let (mean, sd) = gp.predict(&s.x, s.t);
            assert!((mean - s.y).abs() < 0.01, "{mean} vs {}", s.y);
            assert!(sd < 0.02, "{sd}");
        }
        let (_, far) = gp.predict(&[0.5, 0.95], 0.0);
        assert!(far > 0.1, "unsure away from the data: {far}");
    }

    /// The same run informs a prediction less the further away in generations
    /// it is asked about, so an old fitness fades rather than lasting forever.
    #[test]
    fn an_old_generation_counts_for_less() {
        let seen = [at(&[0.5, 0.5], 0.0, 0.9), at(&[0.1, 0.1], 0.0, 0.1)];
        let gp = fit(KERNEL, &seen).expect("fits");
        let (_, near) = gp.predict(&[0.5, 0.5], 1.0);
        let (_, far) = gp.predict(&[0.5, 0.5], 20.0);
        assert!(far > near, "{far} should exceed {near}");
    }

    #[test]
    fn equal_fitnesses_predict_that_fitness() {
        let seen = [at(&[0.1], 0.0, 0.5), at(&[0.9], 0.0, 0.5)];
        let gp = fit(KERNEL, &seen).expect("fits");
        let (mean, _) = gp.predict(&[0.5], 0.0);
        assert!((mean - 0.5).abs() < 1e-9);
    }

    #[test]
    fn nothing_to_fit_is_no_model() {
        assert!(fit(KERNEL, &[]).is_none());
    }

    #[test]
    fn duplicate_points_still_factor() {
        let seen = [at(&[0.4], 0.0, 0.2), at(&[0.4], 0.0, 0.4)];
        let gp = fit(KERNEL, &seen).expect("jitter makes it factor");
        let (mean, _) = gp.predict(&[0.4], 0.0);
        assert!((mean - 0.3).abs() < 0.05, "{mean}");
    }

    #[test]
    fn the_triangular_solves_invert_the_factor() {
        let a = vec![vec![4.0, 2.0], vec![2.0, 3.0]];
        let l = cholesky(&a).expect("positive definite");
        let x = solve_upper(&l, &solve_lower(&l, &[2.0, 5.0]));
        // a · x = b
        let b0 = a[0][0] * x[0] + a[0][1] * x[1];
        let b1 = a[1][0] * x[0] + a[1][1] * x[1];
        assert!((b0 - 2.0).abs() < 1e-12 && (b1 - 5.0).abs() < 1e-12);
        assert!(cholesky(&[vec![1.0, 2.0], vec![2.0, 1.0]]).is_none());
    }
}
