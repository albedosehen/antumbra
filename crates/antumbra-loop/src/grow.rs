//! The grow step learns where to probe (ADR-0022 S-3): which region of the
//! corpus the next generation learns from, decided as gate, then score, then
//! regularize. Credit is recorded for the policy to learn from.
//!
//! - **Gate.** A region is admitted only where the population is non-trivially
//!   above zero, as judged by the authored verifiers the grow step cannot
//!   influence. That is the minimal criterion that keeps impossible and
//!   unreachable regions out.
//! - **Score.** Measured learnability, `p(1-p)` on the region's acceptability:
//!   highest where the population succeeds about half the time. A statistic of
//!   the census, not a learned gap score, which is why it is hard to game.
//! - **Regularize.** A region's score is discounted by how alike it is to the
//!   regions chosen recently, so the curriculum does not settle on one band.
//!   A share of every generation is sampled unfiltered from the whole visible
//!   slice, the cheapest defence against a curriculum quietly reweighting what
//!   the population is good at.
//! - **Credit.** Each decision records what the previous one realized: its
//!   region's acceptability now, less what it was when chosen.
//!
//! The census comes from the contribution measurement: the routed population
//! scored on the live tasks, the base model standing in where the gate
//! escalates.

use std::collections::BTreeMap;

use chrono::Utc;
use sha2::{Digest, Sha256};

use antumbra_core::ports::TaskPrompt;
use antumbra_core::slice::Holdout;
use antumbra_core::{
    cosine_similarity, this_host, Expert, ExpertId, Generation, GrowRecord, RegionCandidate,
    RegionCensus, Result, RunId,
};
use antumbra_store::repo::{boundary, grow, lifecycle};

use crate::contribution::route_top1;
use crate::GenerationLoop;

/// How the grow step chooses among the regions that pass the gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Choosing {
    /// The highest expected credit: the record's objective. Learnability is
    /// the prior, and the credit a region's past choices realized updates it
    /// (see [`by_credit`]).
    Credit,
    /// The most learnable, discounted for redundancy, with no regard to
    /// credit.
    Learnability,
    /// One at random, seeded by the run and generation: the uniform-sampling
    /// baseline the record measures the grow step against. Same gate, same
    /// focus, same unfiltered share; only the choice differs.
    Uniform,
}

/// How the grow step decides.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrowPolicy {
    pub choosing: Choosing,
    /// Start the region's shadow from the expert that serves the region now,
    /// so it refines what is there rather than relearning it from the base.
    /// Off, every shadow starts from fresh factors.
    pub warm_start: bool,
    /// Acceptability a region must exceed to pass the gate.
    pub gate: f32,
    /// Tasks a region's census must rest on to be weighed.
    pub min_tasks: u32,
    /// Share of the learned tasks sampled unfiltered from the whole visible
    /// slice.
    pub unfiltered: f32,
    /// How much of a region's learnability its likeness to a recent choice
    /// costs, at full likeness.
    pub redundancy: f32,
    /// Recent choices the redundancy is measured against.
    pub window: usize,
    /// The credit a perfectly learnable region is expected to realize before
    /// any evidence: the prior's scale.
    pub credit_scale: f32,
    /// How many realized credits' worth of weight the learnability prior
    /// carries.
    pub credit_prior: f32,
}

impl Default for GrowPolicy {
    fn default() -> Self {
        Self {
            choosing: Choosing::Credit,
            warm_start: true,
            gate: 0.05,
            min_tasks: 2,
            unfiltered: 0.25,
            redundancy: 0.5,
            window: 4,
            credit_scale: 0.1,
            credit_prior: 1.0,
        }
    }
}

/// One region of the census, as measured: `(tasks, acceptability, centroid)`.
pub(crate) type RegionReading = (u32, f32, Vec<f32>);

/// The census of a measurement: for each region, how many of its live tasks
/// the population was scored on, its mean score there, and the centroid of
/// its tasks' embeddings. `score` is the population's score on a task routed
/// to `to`, `None` where it was not scored.
pub(crate) fn census(
    tasks: &[TaskPrompt],
    routed: &[(String, Option<ExpertId>)],
    vectors: &BTreeMap<String, Vec<f32>>,
    score: impl Fn(&Option<ExpertId>, &str) -> Option<f32>,
) -> BTreeMap<String, RegionReading> {
    let region_of: BTreeMap<&str, &str> = tasks
        .iter()
        .map(|t| (t.id.as_str(), t.region.as_str()))
        .collect();
    let mut sums: BTreeMap<String, (u32, f32, Vec<f32>)> = BTreeMap::new();
    for (task, to) in routed {
        let (Some(region), Some(s)) = (region_of.get(task.as_str()), score(to, task)) else {
            continue;
        };
        let entry = sums
            .entry((*region).to_string())
            .or_insert_with(|| (0, 0.0, Vec::new()));
        entry.0 += 1;
        entry.1 += s;
        if let Some(v) = vectors.get(task) {
            if entry.2.is_empty() {
                entry.2 = vec![0.0; v.len()];
            }
            if entry.2.len() == v.len() {
                for (acc, x) in entry.2.iter_mut().zip(v) {
                    *acc += x;
                }
            }
        }
    }
    sums.into_iter()
        .map(|(region, (n, sum, centroid))| {
            let centroid = centroid.iter().map(|x| x / n as f32).collect();
            (region, (n, sum / n as f32, centroid))
        })
        .collect()
}

/// Weigh every region of the census, and choose the best admitted one, if
/// any. `recent` holds the centroids of the regions chosen lately.
pub fn choose(
    census: &[RegionCensus],
    recent: &[Vec<f32>],
    policy: &GrowPolicy,
) -> (Vec<RegionCandidate>, Option<String>) {
    let candidates: Vec<RegionCandidate> = census
        .iter()
        .map(|c| {
            let p = c.acceptability;
            let admitted = p > policy.gate && c.tasks >= policy.min_tasks;
            let learnability = p * (1.0 - p);
            let likeness = recent
                .iter()
                .map(|r| cosine_similarity(&c.centroid, r))
                .filter(|s| s.is_finite())
                .fold(0.0f32, f32::max)
                .max(0.0);
            let penalty = policy.redundancy * likeness;
            RegionCandidate {
                region: c.region.clone(),
                acceptability: p,
                admitted,
                learnability,
                penalty,
                score: learnability * (1.0 - penalty),
            }
        })
        .collect();
    let chosen = candidates
        .iter()
        .filter(|c| c.admitted)
        .max_by(|a, b| {
            a.score
                .total_cmp(&b.score)
                .then_with(|| b.region.cmp(&a.region))
        })
        .map(|c| c.region.clone());
    (candidates, chosen)
}

/// The diversity instruments over a run's decisions (ADR-0022 S-3's
/// validation): the entropy of the regions chosen, normalized to 1 when every
/// region is chosen equally; the share of the census's regions ever chosen;
/// and the regions once gated out that later passed the gate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Diversity {
    pub entropy: f32,
    pub coverage: f32,
    pub revived: u32,
}

pub fn diversity(history: &[GrowRecord], regions: usize) -> Diversity {
    let mut counts: BTreeMap<&str, u32> = BTreeMap::new();
    for r in history {
        if let Some(region) = &r.chosen {
            *counts.entry(region.as_str()).or_default() += 1;
        }
    }
    let total: u32 = counts.values().sum();
    let entropy = if total == 0 || regions < 2 {
        0.0
    } else {
        let h: f32 = counts
            .values()
            .map(|&n| {
                let p = n as f32 / total as f32;
                -p * p.ln()
            })
            .sum();
        // One region chosen every time sums to a negative zero.
        (h / (regions as f32).ln()).abs()
    };
    let coverage = counts.len() as f32 / regions.max(1) as f32;
    let mut gated_out: Vec<&str> = Vec::new();
    let mut revived = 0;
    for r in history {
        for c in &r.candidates {
            if !c.admitted && !gated_out.contains(&c.region.as_str()) {
                gated_out.push(c.region.as_str());
            }
        }
    }
    if let Some(last) = history.last() {
        revived = last
            .candidates
            .iter()
            .filter(|c| c.admitted && gated_out.contains(&c.region.as_str()))
            .count() as u32;
    }
    Diversity {
        entropy,
        coverage,
        revived,
    }
}

/// What the grow step decided for a generation, and the tasks it chose.
#[derive(Debug, Clone, PartialEq)]
pub struct Growth {
    pub record: GrowRecord,
    pub focus: Vec<String>,
    pub diversity: Diversity,
    /// The adapter the shadow starts from, when it is warm-started.
    pub parent_adapter: Option<String>,
}

/// The expert most of a region's tasks are routed to, when one is: `counts`
/// holds how many went to each expert, `None` for those the gate escalates.
/// Escalation can win too, and then no expert serves the region. Ties go to
/// the lowest id, so the answer is the same on every run.
pub fn plurality(counts: &BTreeMap<Option<String>, u32>) -> Option<String> {
    counts
        .iter()
        .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)))
        .and_then(|(who, _)| who.clone())
}

/// Credit as the objective (ADR-0022 S-3): the admitted region with the
/// highest expected credit, and that expectation.
///
/// A region's expected credit starts from its learnability, scaled so a
/// perfectly learnable one is expected to realize `credit_scale`, and is
/// pulled toward the mean credit its past choices realized, the prior
/// weighing `credit_prior` realized credits' worth. Redundancy is discounted
/// in the same units. Learnability shapes where the policy looks first; what
/// the population actually gained decides.
pub fn by_credit(
    candidates: &[RegionCandidate],
    realized: &BTreeMap<String, Vec<f32>>,
    policy: &GrowPolicy,
) -> Option<(String, f32)> {
    candidates
        .iter()
        .filter(|c| c.admitted)
        .map(|c| {
            let prior = policy.credit_scale * c.learnability / 0.25;
            let seen = realized.get(&c.region).map_or(&[][..], Vec::as_slice);
            let k = policy.credit_prior.max(0.0);
            let expected =
                (seen.iter().sum::<f32>() + k * prior) / (seen.len() as f32 + k).max(f32::EPSILON);
            (c.region.clone(), expected - policy.credit_scale * c.penalty)
        })
        .max_by(|a, b| a.1.total_cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
}

/// The credit each region's past choices realized, from a run's decisions in
/// order: the credit a decision records belongs to the choice before it.
fn realized(decisions: &[&GrowRecord], latest: Option<f32>) -> BTreeMap<String, Vec<f32>> {
    let mut by_region: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    let credits = decisions
        .iter()
        .skip(1)
        .map(|r| r.credit)
        .chain(std::iter::once(latest));
    for (choice, credit) in decisions.iter().zip(credits) {
        if let (Some(region), Some(c)) = (&choice.chosen, credit) {
            by_region.entry(region.clone()).or_default().push(c);
        }
    }
    by_region
}

/// Among the admitted candidates, the one a uniform draw seeded by the run
/// and generation picks.
fn uniformly(
    candidates: &[RegionCandidate],
    run_id: &RunId,
    generation: Generation,
) -> Option<String> {
    let admitted: Vec<&RegionCandidate> = candidates.iter().filter(|c| c.admitted).collect();
    if admitted.is_empty() {
        return None;
    }
    let digest = Sha256::digest(format!("grow-uniform:{run_id}:{}", generation.0));
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    let i = (u64::from_le_bytes(bytes) % admitted.len() as u64) as usize;
    Some(admitted[i].region.clone())
}

/// A stable draw for the unfiltered share: the same tasks for the same run
/// and generation.
fn draw(run_id: &RunId, generation: Generation, task: &str) -> [u8; 32] {
    Sha256::digest(format!("grow:{run_id}:{}:{task}", generation.0)).into()
}

impl GenerationLoop<'_> {
    /// Decide what this generation learns from. `None` when no policy is set:
    /// every visible task, as always.
    pub(crate) async fn plan_growth(
        &self,
        run_id: &RunId,
        generation: Generation,
        holdout: Option<Holdout>,
    ) -> Result<Option<Growth>> {
        let Some(policy) = self.cfg.grow else {
            return Ok(None);
        };
        let census = grow::census_before(self.store, run_id, generation).await?;
        let history = grow::history(self.store, run_id).await?;
        let earlier: Vec<&GrowRecord> = history
            .iter()
            .filter(|r| r.generation.0 < generation.0)
            .collect();
        let recent: Vec<Vec<f32>> = earlier
            .iter()
            .rev()
            .take(policy.window)
            .filter_map(|r| r.chosen.as_deref())
            .filter_map(|region| census.iter().find(|c| c.region == region))
            .map(|c| c.centroid.clone())
            .collect();
        let (candidates, best) = choose(&census, &recent, &policy);
        // What the last choice realized, now that there is a census after it.
        let credit = earlier.last().and_then(|last| {
            let region = last.chosen.as_deref()?;
            let then = last
                .candidates
                .iter()
                .find(|c| c.region == region)?
                .acceptability;
            let now = census.iter().find(|c| c.region == region)?.acceptability;
            (census.first()?.generation.0 >= last.generation.0).then_some(now - then)
        });
        let chosen = match policy.choosing {
            Choosing::Credit => {
                by_credit(&candidates, &realized(&earlier, credit), &policy).map(|(r, _)| r)
            }
            Choosing::Learnability => best,
            Choosing::Uniform => uniformly(&candidates, run_id, generation),
        };
        let mut focus = Vec::new();
        let mut unfiltered = 0u32;
        let mut warm = None;
        if let Some(region) = &chosen {
            let live = self.trainer.live_tasks(holdout).await?;
            let (mine, rest): (Vec<&TaskPrompt>, Vec<&TaskPrompt>) =
                live.iter().partition(|t| &t.region == region);
            if policy.warm_start {
                warm = self.serving(&mine).await?;
            }
            let share = policy.unfiltered.clamp(0.0, 0.9);
            let extra = ((mine.len() as f32) * share / (1.0 - share)).ceil() as usize;
            let mut others: Vec<&TaskPrompt> = rest;
            others.sort_by_key(|t| draw(run_id, generation, &t.id));
            focus.extend(mine.iter().map(|t| t.id.clone()));
            for t in others.into_iter().take(extra) {
                focus.push(t.id.clone());
                unfiltered += 1;
            }
        }
        let record = GrowRecord {
            run_id: run_id.clone(),
            generation,
            census_generation: census.first().map(|c| c.generation),
            chosen,
            candidates,
            focus: u32::try_from(focus.len()).unwrap_or(u32::MAX),
            unfiltered,
            credit,
            warm_from: warm.as_ref().map(|e| e.id.clone()),
            at: Utc::now(),
        };
        grow::upsert(self.store, &record).await?;
        let mut decided: Vec<GrowRecord> = earlier.into_iter().cloned().collect();
        decided.push(record.clone());
        let diversity = diversity(&decided, census.len());
        Ok(Some(Growth {
            record,
            focus,
            diversity,
            parent_adapter: warm.map(|e| e.artifact_uri),
        }))
    }

    /// The expert that serves these tasks now: the one the population routes
    /// most of them to, as a contribution measurement would route them.
    async fn serving(&self, tasks: &[&TaskPrompt]) -> Result<Option<Expert>> {
        let host = this_host();
        let mut experts: Vec<Expert> = lifecycle::routable(self.store)
            .await?
            .into_iter()
            .filter(|e| e.owner.is_none() && e.is_placed_on(&host))
            .collect();
        if experts.is_empty() || tasks.is_empty() {
            return Ok(None);
        }
        experts.sort_by(|a, b| a.id.as_str().cmp(b.id.as_str()));
        let router = lifecycle::load_router(self.store).await?;
        let boundaries = boundary::list(self.store).await?;
        let mut counts: BTreeMap<Option<String>, u32> = BTreeMap::new();
        for task in tasks {
            let v = self.embedder.embed(&task.prompt).await?;
            let routed = route_top1(&v, router.as_ref(), &experts, &boundaries, None);
            *counts
                .entry(routed.map(|id| id.as_str().to_string()))
                .or_default() += 1;
        }
        Ok(plurality(&counts).and_then(|id| experts.into_iter().find(|e| e.id.as_str() == id)))
    }

    /// Record the census a contribution measurement took.
    pub(crate) async fn record_census(
        &self,
        run_id: &RunId,
        generation: Generation,
        readings: BTreeMap<String, RegionReading>,
    ) -> Result<Vec<RegionCensus>> {
        let mut kept = Vec::with_capacity(readings.len());
        for (region, (tasks, acceptability, centroid)) in readings {
            let census = RegionCensus {
                run_id: run_id.clone(),
                generation,
                region,
                tasks,
                acceptability,
                centroid,
                at: Utc::now(),
            };
            grow::upsert_census(self.store, &census).await?;
            kept.push(census);
        }
        Ok(kept)
    }
}

#[cfg(test)]
mod tests;
