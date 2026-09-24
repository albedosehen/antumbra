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
//! cannot carry a recipe past a well-measured one. A recipe run in several
//! generations is ranked on all its runs together.
//!
//! Greed is blunted by two frequencies, as the record asks. Each cohort member
//! is a slot that keeps its recipe until its ready interval has passed. The
//! fast members' interval starts at one generation, so they are proposed afresh
//! every generation, and lengthens over the run ([`SearchPolicy::anneal`]). The
//! slow members keep theirs for [`SearchPolicy::slow_interval`] generations,
//! and nothing the fast members score can displace them sooner. Every member
//! still trains from the base, so holding a recipe means measuring it again:
//! the slow members are the recipes the search knows well, and pooled ranking
//! is what lets that knowledge count against a newcomer's lucky run.
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
    /// The batch sizes to choose among, smallest first. The trainer
    /// accumulates a batch's gradients one example at a time, so the size
    /// costs steps, not memory.
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
    /// Members in the slow cohort: the last `slow` of the cohort. The first
    /// member always carries the incumbent, so at least one is fast.
    pub slow: usize,
    /// Generations a slow member keeps its recipe before it is proposed a new
    /// one.
    pub slow_interval: u32,
    /// Generations over which the fast members' ready interval lengthens from
    /// one toward the slow interval, stopping one short of it so the two
    /// frequencies stay two. 0 holds it at one: the fast members are proposed
    /// afresh every generation.
    pub anneal: u32,
}

impl Default for SearchPolicy {
    fn default() -> Self {
        Self {
            space: RecipeSpace::default(),
            cohort: 4,
            seed: 0,
            exploration: 2.0,
            prior_weight: 1.0,
            slow: 0,
            slow_interval: 3,
            anneal: 0,
        }
    }
}

impl SearchPolicy {
    /// Whether the member at `index` of a cohort is in the slow cohort.
    pub fn is_slow(&self, index: usize) -> bool {
        self.slow > 0 && index > 0 && index >= self.cohort.saturating_sub(self.slow)
    }

    /// The fast members' ready interval at `generation`: one, lengthening
    /// linearly over [`Self::anneal`] generations toward the slow interval and
    /// stopping one short of it.
    pub fn fast_interval(&self, generation: u32) -> u32 {
        if self.anneal == 0 {
            return 1;
        }
        let span = u64::from(self.slow_interval.saturating_sub(1));
        let grown = span * u64::from(generation) / u64::from(self.anneal);
        (1 + grown).min(span.max(1)) as u32
    }

    /// Generations the member at `index` keeps a recipe at `generation`.
    fn ready_interval(&self, index: usize, generation: u32) -> u32 {
        if self.is_slow(index) {
            self.slow_interval.max(1)
        } else {
            self.fast_interval(generation)
        }
    }
}

/// What one cohort member ran before the generation being proposed: its
/// recipe in the previous generation, and for how many consecutive
/// generations, ending there, it had run that recipe.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Slot {
    pub recipe: TrainingRecipe,
    pub held: u32,
}

/// A mean over `evaluations` shrunk toward `generation_mean` by
/// `prior_weight` evaluations' worth.
pub fn shrink(mean: f32, evaluations: u32, generation_mean: f32, prior_weight: f64) -> f64 {
    let n = f64::from(evaluations);
    (n * f64::from(mean) + prior_weight * f64::from(generation_mean))
        / (n + prior_weight).max(f64::EPSILON)
}

/// The fitness a recipe is ranked by: its mean shrunk toward its generation's
/// mean, by `prior_weight` evaluations' worth.
pub fn shrunk_fitness(record: &RecipeRecord, generation_mean: f32, prior_weight: f64) -> f64 {
    shrink(
        record.fitness_mean,
        record.evaluations,
        generation_mean,
        prior_weight,
    )
}

/// The best recipe in `history` by pooled fitness ([`pooled_fitness`]), as
/// its latest row. Ties go to the later generation, which was measured against
/// the population as it now is.
pub fn incumbent(history: &[RecipeRecord], prior_weight: f64) -> Option<&RecipeRecord> {
    history
        .iter()
        .filter(|r| {
            !history
                .iter()
                .any(|o| o.recipe == r.recipe && o.generation.0 > r.generation.0)
        })
        .map(|r| (r, pooled_fitness(history, &r.recipe, prior_weight)))
        .max_by(|(a, sa), (b, sb)| {
            sa.total_cmp(sb)
                .then_with(|| a.generation.0.cmp(&b.generation.0))
        })
        .map(|(r, _)| r)
}

/// Every run of `recipe` in `history`, pooled: their fitness averaged by
/// evaluation count and shrunk, by `prior_weight` evaluations' worth, toward
/// the means of the generations they ran in. A recipe run once scores exactly
/// its [`shrunk_fitness`]; one run three times rests on three evaluations and
/// is shrunk a third as far.
pub fn pooled_fitness(history: &[RecipeRecord], recipe: &TrainingRecipe, prior_weight: f64) -> f64 {
    let (mut n, mut sum, mut target) = (0.0, 0.0, 0.0);
    for r in history.iter().filter(|r| r.recipe == *recipe) {
        // A row stands for at least the one run that wrote it.
        let w = f64::from(r.evaluations.max(1));
        n += w;
        sum += w * f64::from(r.fitness_mean);
        target += w * f64::from(generation_mean(history, r));
    }
    if n == 0.0 {
        return f64::NEG_INFINITY;
    }
    (sum + prior_weight * target / n) / (n + prior_weight).max(f64::EPSILON)
}

fn generation_mean(history: &[RecipeRecord], of: &RecipeRecord) -> f32 {
    let peers: Vec<f32> = history
        .iter()
        .filter(|r| r.generation == of.generation)
        .map(|r| r.fitness_mean)
        .collect();
    peers.iter().sum::<f32>() / peers.len().max(1) as f32
}

/// The recipe each member of `generation`'s cohort trains under, by slot: up
/// to `policy.cohort` of them, all different. A slot left `None` found no
/// recipe different from the others and sits the generation out; its index
/// still names it, so a slow member keeps its place.
///
/// `slots` is what each member ran before (see [`Slot`]), by index. A member
/// other than the first keeps its recipe while it has held it for less than
/// its ready interval. The first carries the incumbent, or `anchor` (the
/// recipe the run starts from) before there is one, so the recipe behind the
/// best shadow propagates, unless a kept member already runs it. The rest
/// maximize the upper confidence bound of the model fitted to `history`, or,
/// with no history, are spread through the space. `history` must be rows
/// measured under one partition, since fitness read under another split, or
/// under none, is not comparable.
pub fn propose(
    policy: &SearchPolicy,
    history: &[RecipeRecord],
    generation: u32,
    anchor: TrainingRecipe,
    slots: &[Option<Slot>],
) -> Vec<Option<TrainingRecipe>> {
    if policy.cohort == 0 {
        return Vec::new();
    }
    let space = &policy.space;
    let mut rng = SplitMix(policy.seed ^ u64::from(generation).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let lead = incumbent(history, policy.prior_weight).map_or(anchor, |r| r.recipe);
    if space.dims() == 0 {
        // Nothing to search: every member would train under the same recipe.
        return vec![Some(lead)];
    }
    let mut cohort: Vec<Option<TrainingRecipe>> = (0..policy.cohort)
        .map(|i| {
            let slot = slots.get(i).copied().flatten()?;
            (i > 0 && slot.held < policy.ready_interval(i, generation)).then_some(slot.recipe)
        })
        .collect();
    if !cohort.contains(&Some(lead)) {
        cohort[0] = Some(lead);
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
    // Kept members run this generation too. The model is told they score what
    // it expects of them, so the proposals look elsewhere.
    if let Some(model) = gp::fit(kernel, &samples) {
        let kept: Vec<Sample> = cohort
            .iter()
            .enumerate()
            .filter(|(i, r)| *i > 0 && **r != Some(lead))
            .filter_map(|(_, r)| *r)
            .map(|r| {
                let x = space.to_unit(&r);
                let (mean, _) = model.predict(&x, t);
                Sample { x, t, y: mean }
            })
            .collect();
        samples.extend(kept);
    }
    let mut attempts = 0;
    while let Some(open) = cohort.iter().position(Option::is_none) {
        if attempts >= 64 {
            break;
        }
        attempts += 1;
        // Candidates for the maximization: points drawn across the whole space,
        // and points near the incumbent, where the best recipes are likeliest.
        let mut candidates: Vec<Vec<f64>> = (0..256)
            .map(|_| (0..space.dims()).map(|_| rng.unit()).collect())
            .collect();
        let centre = space.to_unit(&lead);
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
        if cohort.contains(&Some(recipe)) {
            continue;
        }
        // Pretend the pick scored what the model expects, so the next pick
        // looks elsewhere: the "kriging believer" batch heuristic.
        samples.push(Sample {
            x: space.to_unit(&recipe),
            t,
            y: predicted,
        });
        cohort[open] = Some(recipe);
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
