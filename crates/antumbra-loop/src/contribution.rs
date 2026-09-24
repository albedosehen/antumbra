//! The population's leave-one-out contribution (ADR-0022 S-5): for each shared
//! expert, mask it, route the live tasks again, score both ways, and record
//! the difference. It is the telemetry retirement is decided on: routing share
//! alone cannot tell an expert that is unused from one that is useless.
//!
//! Masking one expert moves only the tasks that were routed to it, so each
//! expert is scored on its own tasks, and on each of them the expert the gate
//! falls back to (or the base model, where none covers the task) is scored
//! too. Both sides draw from the same seeds, which makes every task a paired
//! comparison and keeps the difference from drowning in sampling noise. Every
//! adapter is evaluated once, over all the tasks either side needs it for.
//!
//! Routing is the one the CLI's `ask` uses: the learned router, masked to the
//! experts the gate may route to, when one is trained, and the heuristic gate
//! otherwise. The live tasks are the visible slice; the held-out and audit
//! slices are never touched, because demotion is a selection.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use chrono::Utc;
use sha2::{Digest, Sha256};

use antumbra_core::ports::{EvaluateRequest, TaskPrompt, TaskScores};
use antumbra_core::slice::Holdout;
use antumbra_core::{
    this_host, BaselineRecord, ContributionRecord, Expert, ExpertId, FailureBoundary, Generation,
    LearnedRouter, Result, RunId,
};
use antumbra_gate::{route as gate_route, GateConfig};
use antumbra_store::repo::{boundary, contribution, lifecycle};

use crate::baseline::compare;
use crate::grow;
use crate::GenerationLoop;

/// Inhibition above which a boundary escalates a task the learned router
/// covers, as `ask` and the MCP surface escalate it.
const INHIBIT: f32 = 0.5;

/// When and how the loop measures contribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContributionPolicy {
    /// Measured in every `every`-th generation, from generation 0.
    pub every: u32,
    /// Evaluations per task on each side, each under its own seed.
    pub seeds: u32,
    /// At most this many live tasks, chosen by a stable hash of their ids, so
    /// successive generations measure the same tasks.
    pub max_tasks: usize,
    /// Also compare the routed population against its best single expert
    /// (ADR-0022 S-5). It scores every expert on every live task, one more
    /// evaluation of them per expert.
    pub baseline: bool,
}

impl Default for ContributionPolicy {
    fn default() -> Self {
        Self {
            every: 2,
            seeds: 2,
            max_tasks: 32,
            baseline: true,
        }
    }
}

/// The expert a task goes to, or `None` when it escalates, with `masked` taken
/// out of the population.
pub(crate) fn route_top1(
    task: &[f32],
    router: Option<&LearnedRouter>,
    experts: &[Expert],
    boundaries: &[FailureBoundary],
    masked: Option<&ExpertId>,
) -> Option<ExpertId> {
    let kept = |id: &ExpertId| Some(id) != masked && experts.iter().any(|e| &e.id == id);
    match router {
        Some(learned) => {
            let learned = learned.clone().masked(kept);
            let cfg = GateConfig::default();
            let inhibition = boundaries
                .iter()
                .map(|b| b.inhibition_for(task, cfg.inhibition_radius))
                .fold(0.0f32, f32::max);
            if !learned.covers(task) || inhibition > INHIBIT {
                return None;
            }
            learned.route(task).into_iter().next().map(|(id, _)| id)
        }
        None => {
            let pool: Vec<Expert> = experts.iter().filter(|e| kept(&e.id)).cloned().collect();
            gate_route(task, &pool, boundaries, 1, &GateConfig::default())
                .chosen
                .into_iter()
                .next()
        }
    }
}

/// The mean cosine similarity of the tasks routed to `expert` to its
/// capability vector: how close what it is asked to do sits to what it was
/// minted for. `None` when nothing was routed to it or it has no vector.
fn affinity(expert: &Expert, routed: &[&str], vectors: &BTreeMap<String, Vec<f32>>) -> Option<f32> {
    let sims: Vec<f32> = routed
        .iter()
        .filter_map(|t| expert.capability_similarity(vectors.get(*t)?))
        .filter(|s| s.is_finite())
        .collect();
    (!sims.is_empty()).then(|| sims.iter().sum::<f32>() / sims.len() as f32)
}

/// The seeds a measurement named `purpose` draws from in a generation: from
/// the run and the generation, never from training's stream, and different
/// for each purpose.
pub(crate) fn seeds(purpose: &str, run_id: &RunId, generation: Generation, n: u32) -> Vec<u64> {
    (0..n)
        .map(|k| {
            let digest = Sha256::digest(format!("{purpose}:{run_id}:{}:{k}", generation.0));
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(&digest[..8]);
            u64::from_le_bytes(bytes)
        })
        .collect()
}

/// A stable order for choosing which live tasks to measure.
pub(crate) fn rank(id: &str) -> [u8; 32] {
    Sha256::digest(id.as_bytes()).into()
}

/// What a contribution measurement recorded.
#[derive(Debug, Clone, Default)]
pub(crate) struct Measured {
    pub contribution: Vec<ContributionRecord>,
    pub baseline: Option<BaselineRecord>,
    /// The region census, taken with the baseline for the grow step.
    pub census: Vec<antumbra_core::RegionCensus>,
}

/// Where each task routes with the whole population, and, for a task routed
/// to an expert, where it routes with that expert masked.
struct Routes {
    full: Vec<(String, Option<ExpertId>)>,
    fallback: BTreeMap<String, Option<ExpertId>>,
    /// Each routed task's embedding, for how close it sits to its expert.
    vectors: BTreeMap<String, Vec<f32>>,
}

impl GenerationLoop<'_> {
    /// At most `max` live tasks, the same ones every generation: chosen by a
    /// stable hash of their ids.
    pub(crate) async fn live_sample(
        &self,
        holdout: Option<Holdout>,
        max: usize,
    ) -> Result<Vec<TaskPrompt>> {
        let mut tasks = self.trainer.live_tasks(holdout).await?;
        tasks.sort_by_key(|t| rank(&t.id));
        tasks.truncate(max);
        Ok(tasks)
    }

    /// Measure and record the contribution of every shared expert this node
    /// serves, when the policy says this generation is due. Returns what it
    /// recorded, nothing when it was not due or there is nothing to measure.
    pub(crate) async fn measure_contribution(
        &self,
        run_id: &RunId,
        generation: Generation,
        holdout: Option<Holdout>,
    ) -> Result<Measured> {
        let Some(policy) = self.cfg.contribution else {
            return Ok(Measured::default());
        };
        if !generation.0.is_multiple_of(policy.every.max(1)) {
            return Ok(Measured::default());
        }
        let host = this_host();
        let experts: Vec<Expert> = lifecycle::routable(self.store)
            .await?
            .into_iter()
            .filter(|e| e.owner.is_none() && e.is_placed_on(&host))
            .collect();
        // With no expert there is nothing to credit, but the grow step still
        // needs the census: the base model alone, on every live task.
        if experts.is_empty() && self.cfg.grow.is_none() {
            return Ok(Measured::default());
        }
        let tasks = self.live_sample(holdout, policy.max_tasks).await?;
        let routes = self.route_live(&tasks, &experts).await?;
        let draws = seeds("contribution", run_id, generation, policy.seeds);
        let scores = self
            .score_routes(
                run_id,
                generation,
                &routes,
                &experts,
                &draws,
                policy.baseline,
            )
            .await?;
        let score = |who: &Option<ExpertId>, task: &str| -> Option<f32> {
            scores.get(who).and_then(|s| s.scores.get(task)).copied()
        };

        let mut recorded = Vec::with_capacity(experts.len());
        for e in &experts {
            let mine: Vec<&str> = routes
                .full
                .iter()
                .filter(|(_, to)| to.as_ref() == Some(&e.id))
                .map(|(id, _)| id.as_str())
                .collect();
            // Pairs scored on both sides; a task either side could not score
            // is left out of both, so the means stay paired.
            let pairs: Vec<(f32, f32)> = mine
                .iter()
                .filter_map(|&t| {
                    let without = routes.fallback.get(t)?;
                    Some((score(&Some(e.id.clone()), t)?, score(without, t)?))
                })
                .collect();
            let mean = |side: fn(&(f32, f32)) -> f32| {
                (!pairs.is_empty())
                    .then(|| pairs.iter().map(side).sum::<f32>() / pairs.len() as f32)
            };
            let record = ContributionRecord {
                expert: e.id.clone(),
                run_id: run_id.clone(),
                generation,
                routed: u32::try_from(mine.len()).unwrap_or(u32::MAX),
                tasks: u32::try_from(tasks.len()).unwrap_or(u32::MAX),
                with: mean(|p| p.0),
                without: mean(|p| p.1),
                seeds: policy.seeds,
                affinity: affinity(e, &mine, &routes.vectors),
                at: Utc::now(),
            };
            contribution::upsert(self.store, &record).await?;
            recorded.push(record);
        }
        let ids: Vec<ExpertId> = experts.iter().map(|e| e.id.clone()).collect();
        let baseline = match compare(&routes.full, &ids, score).filter(|_| policy.baseline) {
            Some(c) => {
                let record = BaselineRecord {
                    run_id: run_id.clone(),
                    generation,
                    tasks: c.tasks,
                    population: c.population,
                    best_alone: c.best.as_ref().map(|b| b.1),
                    best: c.best.map(|b| b.0),
                    seeds: policy.seeds,
                    at: Utc::now(),
                };
                contribution::upsert_baseline(self.store, &record).await?;
                Some(record)
            }
            None => None,
        };
        let census = if policy.baseline {
            let readings = grow::census(&tasks, &routes.full, &routes.vectors, score);
            self.record_census(run_id, generation, readings).await?
        } else {
            Vec::new()
        };
        Ok(Measured {
            contribution: recorded,
            baseline,
            census,
        })
    }

    async fn route_live(&self, tasks: &[TaskPrompt], experts: &[Expert]) -> Result<Routes> {
        let router = lifecycle::load_router(self.store).await?;
        let boundaries = boundary::list(self.store).await?;
        let mut full = Vec::with_capacity(tasks.len());
        let mut fallback = BTreeMap::new();
        let mut vectors = BTreeMap::new();
        for t in tasks {
            let v = self.embedder.embed(&t.prompt).await?;
            let to = route_top1(&v, router.as_ref(), experts, &boundaries, None);
            if let Some(expert) = &to {
                let without = route_top1(&v, router.as_ref(), experts, &boundaries, Some(expert));
                fallback.insert(t.id.clone(), without);
            }
            vectors.insert(t.id.clone(), v);
            full.push((t.id.clone(), to));
        }
        Ok(Routes {
            full,
            fallback,
            vectors,
        })
    }

    /// Evaluate every adapter the comparison needs once, over the union of
    /// the tasks it is needed for. `None` is the base model alone. With
    /// `baseline`, every expert is also scored on every live task, and the
    /// base on the tasks the population escalates.
    async fn score_routes(
        &self,
        run_id: &RunId,
        generation: Generation,
        routes: &Routes,
        experts: &[Expert],
        draws: &[u64],
        baseline: bool,
    ) -> Result<HashMap<Option<ExpertId>, TaskScores>> {
        let mut needed: HashMap<Option<ExpertId>, BTreeSet<String>> = HashMap::new();
        if baseline {
            for (task, to) in &routes.full {
                needed.entry(to.clone()).or_default().insert(task.clone());
                for e in experts {
                    needed
                        .entry(Some(e.id.clone()))
                        .or_default()
                        .insert(task.clone());
                }
            }
        }
        for (task, to) in &routes.full {
            let Some(expert) = to else { continue };
            needed
                .entry(Some(expert.clone()))
                .or_default()
                .insert(task.clone());
            if let Some(without) = routes.fallback.get(task) {
                needed
                    .entry(without.clone())
                    .or_default()
                    .insert(task.clone());
            }
        }
        let known: HashSet<&ExpertId> = experts.iter().map(|e| &e.id).collect();
        let mut scores = HashMap::new();
        for (who, task_ids) in needed {
            let adapter_uri = match &who {
                Some(id) if known.contains(id) => experts
                    .iter()
                    .find(|e| &e.id == id)
                    .map(|e| e.artifact_uri.clone()),
                Some(_) => continue,
                None => None,
            };
            let scored = self
                .trainer
                .evaluate(EvaluateRequest {
                    label: format!("contribution:{run_id}:g{}", generation.0),
                    base_model: self.cfg.base_model.clone(),
                    adapter_uri,
                    task_ids: task_ids.into_iter().collect(),
                    seeds: draws.to_vec(),
                })
                .await?;
            scores.insert(who, scored);
        }
        Ok(scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antumbra_core::RouterExpert;

    fn expert(id: &str, vec: Vec<f32>) -> Expert {
        Expert {
            id: ExpertId::new(id),
            name: id.into(),
            base_model: "base".into(),
            artifact_uri: format!("adapters/{id}"),
            capability_card: serde_json::json!({}),
            capability_vec: Some(vec),
            fitness: 1.0,
            frozen_at: Some(Utc::now()),
            generation: Generation::ZERO,
            owner: None,
            compartment: None,
            placed_on: None,
            created_at: Utc::now(),
        }
    }

    #[test]
    fn masking_the_routed_expert_falls_back_to_the_next_or_escalates() {
        let experts = [
            expert("a", vec![1.0, 0.0, 0.0]),
            expert("b", vec![0.7, 0.7, 0.0]),
        ];
        let router = LearnedRouter {
            weights: vec![1.0; 3],
            experts: experts
                .iter()
                .map(|e| RouterExpert {
                    id: e.id.clone(),
                    centroid: e.capability_vec.clone().unwrap_or_default(),
                })
                .collect(),
            temperature: 0.1,
            floor: 0.6,
        };
        let task = [1.0, 0.1, 0.0];
        let a = ExpertId::new("a");
        assert_eq!(
            route_top1(&task, Some(&router), &experts, &[], None),
            Some(a.clone())
        );
        assert_eq!(
            route_top1(&task, Some(&router), &experts, &[], Some(&a)),
            Some(ExpertId::new("b"))
        );
        // Far from b, the task escalates once a is masked.
        let only_a = [1.0, -0.9, 0.0];
        assert_eq!(
            route_top1(&only_a, Some(&router), &experts, &[], Some(&a)),
            None
        );
        // A router that knows an expert no longer in the population never
        // routes to it.
        let rest = [expert("b", vec![0.7, 0.7, 0.0])];
        assert_ne!(route_top1(&task, Some(&router), &rest, &[], None), Some(a));
    }

    #[test]
    fn without_a_router_the_heuristic_gate_routes() {
        let experts = [
            expert("a", vec![1.0, 0.0, 0.0]),
            expert("b", vec![0.0, 1.0, 0.0]),
        ];
        let a = ExpertId::new("a");
        let to = route_top1(&[1.0, 0.0, 0.0], None, &experts, &[], None);
        assert_eq!(to, Some(a.clone()));
        let masked = route_top1(&[1.0, 0.0, 0.0], None, &experts, &[], Some(&a));
        assert_ne!(masked, Some(a));
    }

    #[test]
    fn seeds_are_the_runs_and_differ_by_generation() {
        let run = RunId::new("run");
        let at = |g| seeds("contribution", &run, Generation(g), 2);
        assert_eq!(at(1), at(1));
        assert_ne!(at(1), at(2));
        assert_ne!(at(1), seeds("admission", &run, Generation(1), 2));
        assert_eq!(seeds("contribution", &run, Generation(1), 3).len(), 3);
    }
}
