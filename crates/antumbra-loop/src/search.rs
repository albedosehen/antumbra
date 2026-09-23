//! The recipe search (ADR-0022 S-1): which recipes the next generation's cohort
//! trains under, chosen from what earlier generations' recipes scored.
//!
//! It follows Population Based Bandits (Parker-Holder et al., NeurIPS 2020),
//! which the record picks over Population Based Training because PBT is
//! reported to lose to random search at the four-to-eight members a single
//! 24 GB card supports. A Gaussian process ([`gp`]) models fitness over the
//! recipe and the generation, and each proposal maximizes an upper confidence
//! bound on it. A cohort is proposed one member at a time, each pick added to
//! the model as if it had scored its predicted mean, so the members spread out
//! instead of piling onto one point.
//!
//! Ranking follows the record's fourth constraint. The argmax of noisy
//! estimates is enriched for favourable noise (the optimizer's curse, Smith and
//! Winkler 2006), so a recipe's score is shrunk toward its generation's mean
//! in proportion to how few evaluations it rests on, and one lucky evaluation
//! cannot carry a recipe past a well-measured one.
//!
//! Everything here is a pure function of the history it is given and a seed,
//! so a resumed run proposes what a continuous one would have.

mod gp;

use antumbra_core::{RecipeRecord, TrainingRecipe};

use gp::{Kernel, Sample};

/// The bounds the search stays inside. An axis whose bounds are equal is fixed
/// at that value and not searched.
#[derive(Debug, Clone, PartialEq)]
pub struct RecipeSpace {
    /// Learning-rate bounds, searched on a log scale: the record's
    /// "log-uniform learning rate".
    pub learning_rate: (f64, f64),
    /// The batch sizes to choose among, smallest first. The largest should be
    /// one the card can hold: 4 beside the 1.5B base on 24 GB.
    pub batch_sizes: Vec<u32>,
    /// KL-weight bounds. RAFT has no use for the weight, so the default fixes
    /// it at the trainer's value rather than spending a dimension on it.
    pub kl_beta: (f64, f64),
}

impl Default for RecipeSpace {
    fn default() -> Self {
        Self {
            learning_rate: (1e-5, 1e-3),
            batch_sizes: vec![1, 2, 4],
            kl_beta: (0.04, 0.04),
        }
    }
}

impl RecipeSpace {
    fn searches_lr(&self) -> bool {
        self.learning_rate.1 > self.learning_rate.0
    }

    fn searches_batch(&self) -> bool {
        self.batch_sizes.len() > 1
    }

    fn searches_kl(&self) -> bool {
        self.kl_beta.1 > self.kl_beta.0
    }

    /// How many axes are searched.
    pub fn dims(&self) -> usize {
        [
            self.searches_lr(),
            self.searches_batch(),
            self.searches_kl(),
        ]
        .into_iter()
        .filter(|searched| *searched)
        .count()
    }

    /// Where `recipe` sits in the unit cube, one coordinate per searched axis.
    pub fn to_unit(&self, recipe: &TrainingRecipe) -> Vec<f64> {
        let mut u = Vec::with_capacity(self.dims());
        if self.searches_lr() {
            let (lo, hi) = (self.learning_rate.0.ln(), self.learning_rate.1.ln());
            u.push(((recipe.learning_rate.ln() - lo) / (hi - lo)).clamp(0.0, 1.0));
        }
        if self.searches_batch() {
            let last = (self.batch_sizes.len() - 1) as f64;
            u.push(self.nearest_batch(recipe.batch_size) as f64 / last);
        }
        if self.searches_kl() {
            let (lo, hi) = self.kl_beta;
            u.push(((recipe.kl_beta - lo) / (hi - lo)).clamp(0.0, 1.0));
        }
        u
    }

    /// The recipe at a point of the unit cube. Batch sizes snap to the nearest
    /// allowed one; fixed axes take their value.
    pub fn from_unit(&self, u: &[f64]) -> TrainingRecipe {
        let mut coords = u.iter().map(|c| c.clamp(0.0, 1.0));
        let learning_rate = if self.searches_lr() {
            let (lo, hi) = (self.learning_rate.0.ln(), self.learning_rate.1.ln());
            (lo + coords.next().unwrap_or(0.0) * (hi - lo)).exp()
        } else {
            self.learning_rate.0
        };
        let batch_size = if self.searches_batch() {
            let last = (self.batch_sizes.len() - 1) as f64;
            let i = (coords.next().unwrap_or(0.0) * last).round() as usize;
            self.batch_sizes[i.min(self.batch_sizes.len() - 1)]
        } else {
            self.batch_sizes.first().copied().unwrap_or(1)
        };
        let kl_beta = if self.searches_kl() {
            let (lo, hi) = self.kl_beta;
            lo + coords.next().unwrap_or(0.0) * (hi - lo)
        } else {
            self.kl_beta.0
        };
        TrainingRecipe {
            learning_rate,
            batch_size,
            kl_beta,
        }
    }

    fn nearest_batch(&self, batch: u32) -> usize {
        self.batch_sizes
            .iter()
            .enumerate()
            .min_by_key(|(_, b)| b.abs_diff(batch))
            .map_or(0, |(i, _)| i)
    }
}

/// The unsearched policy that governs the search.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchPolicy {
    pub space: RecipeSpace,
    /// Shadows per generation. Four to eight is the range PB2 is designed for
    /// and the range one 24 GB card can train in a generation.
    pub cohort: usize,
    /// Seeds every proposal, so a resumed run proposes what a continuous one
    /// would have.
    pub seed: u64,
    /// Weight on the model's uncertainty when choosing: higher explores more.
    pub exploration: f64,
    /// How many evaluations' worth of weight the generational mean carries in
    /// a shrunk score. At 1, a recipe measured once scores halfway between its
    /// own fitness and its generation's mean.
    pub prior_weight: f64,
}

impl Default for SearchPolicy {
    fn default() -> Self {
        Self {
            space: RecipeSpace::default(),
            cohort: 4,
            seed: 0,
            exploration: 2.0,
            prior_weight: 1.0,
        }
    }
}

/// The fitness a recipe is ranked by: its mean shrunk toward its generation's
/// mean, by `prior_weight` evaluations' worth.
pub fn shrunk_fitness(record: &RecipeRecord, generation_mean: f32, prior_weight: f64) -> f64 {
    let n = f64::from(record.evaluations);
    (n * f64::from(record.fitness_mean) + prior_weight * f64::from(generation_mean))
        / (n + prior_weight).max(f64::EPSILON)
}

/// The best recipe in `history` by shrunk fitness, each measured against the
/// mean of its own generation. Ties go to the later generation, which was
/// measured against the population as it now is.
pub fn incumbent(history: &[RecipeRecord], prior_weight: f64) -> Option<&RecipeRecord> {
    history
        .iter()
        .map(|r| {
            (
                r,
                shrunk_fitness(r, generation_mean(history, r), prior_weight),
            )
        })
        .max_by(|(a, sa), (b, sb)| {
            sa.total_cmp(sb)
                .then_with(|| a.generation.0.cmp(&b.generation.0))
        })
        .map(|(r, _)| r)
}

fn generation_mean(history: &[RecipeRecord], of: &RecipeRecord) -> f32 {
    let peers: Vec<f32> = history
        .iter()
        .filter(|r| r.generation == of.generation)
        .map(|r| r.fitness_mean)
        .collect();
    peers.iter().sum::<f32>() / peers.len().max(1) as f32
}

/// The recipes the cohort of `generation` trains under, `policy.cohort` of
/// them and all different.
///
/// The first is the incumbent, or `anchor` (the recipe the run starts from)
/// before there is one: the recipe behind the best shadow propagates. The rest
/// maximize the upper confidence bound of the model fitted to `history`, or,
/// with no history, are spread through the space. `history` must be rows
/// measured under one partition, since fitness read under another split, or
/// under none, is not comparable.
pub fn propose(
    policy: &SearchPolicy,
    history: &[RecipeRecord],
    generation: u32,
    anchor: TrainingRecipe,
) -> Vec<TrainingRecipe> {
    if policy.cohort == 0 {
        return Vec::new();
    }
    let space = &policy.space;
    let mut rng = SplitMix(policy.seed ^ u64::from(generation).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let first = incumbent(history, policy.prior_weight).map_or(anchor, |r| r.recipe);
    let mut cohort = vec![first];
    if space.dims() == 0 {
        // Nothing to search: every member would train under the same recipe.
        return cohort;
    }
    let mut samples: Vec<Sample> = history
        .iter()
        .map(|r| Sample {
            x: space.to_unit(&r.recipe),
            t: f64::from(r.generation.0),
            y: f64::from(r.fitness_mean),
        })
        .collect();
    let t = f64::from(generation);
    let kernel = Kernel {
        lengthscale: 0.3,
        noise: 0.05,
        decay: 0.1,
    };
    let mut attempts = 0;
    while cohort.len() < policy.cohort && attempts < 64 {
        attempts += 1;
        // Candidates for the maximization: points drawn across the whole space,
        // and points near the incumbent, where the best recipes are likeliest.
        let mut candidates: Vec<Vec<f64>> = (0..256)
            .map(|_| (0..space.dims()).map(|_| rng.unit()).collect())
            .collect();
        let centre = space.to_unit(&cohort[0]);
        for _ in 0..64 {
            candidates.push(
                centre
                    .iter()
                    .map(|c| (c + 0.1 * rng.normal()).clamp(0.0, 1.0))
                    .collect(),
            );
        }
        let pick = match gp::fit(kernel, &samples) {
            Some(model) => candidates
                .into_iter()
                .map(|x| {
                    let (mean, sd) = model.predict(&x, t);
                    (mean + policy.exploration * sd, mean, x)
                })
                .max_by(|a, b| a.0.total_cmp(&b.0))
                .map(|(_, mean, x)| (x, mean)),
            // No history: spread the cohort, taking candidates as drawn.
            None => candidates.into_iter().next().map(|x| (x, 0.0)),
        };
        let Some((x, predicted)) = pick else { break };
        let recipe = space.from_unit(&x);
        if cohort.contains(&recipe) {
            continue;
        }
        // Pretend the pick scored what the model expects, so the next pick
        // looks elsewhere: the "kriging believer" batch heuristic.
        samples.push(Sample {
            x: space.to_unit(&recipe),
            t,
            y: predicted,
        });
        cohort.push(recipe);
    }
    cohort
}

/// A small seeded generator (splitmix64), enough to draw candidates
/// reproducibly without a dependency.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform on [0, 1).
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal, by Box-Muller.
    fn normal(&mut self) -> f64 {
        let u = self.unit().max(f64::MIN_POSITIVE);
        let v = self.unit();
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

#[cfg(test)]
mod tests;
