//! Leave-one-in admission (ADR-0022 S-5, found on the GPU in S-3): a graduate
//! joins only if the population does better on the live tasks with it than
//! without it.
//!
//! The twin check catches a candidate that duplicates an expert. Two GPU runs
//! of the grow step found what it misses:
//! - a region specialist trained from the base on forty tasks can do worse on
//!   its own region than the generalist trained on all of them;
//! - one that does better there can still cost the population: with two
//!   experts 0.82 alike, the heuristic gate's top-two margin shrank, and a
//!   fifth of the live tasks escalated to the base model instead of the
//!   generalist.
//!
//! A check on the candidate's own tasks sees the first and not the second. So
//! every live task is routed twice, as the population routes it now and as it
//! would with the candidate in it. Under the heuristic gate the candidate
//! joins the pool; under a learned router it gets a centroid, projected into
//! the router's metric as a retrained router would place it. Only the tasks
//! whose routing the candidate changes can differ, and each of those is scored
//! both ways under the same seeds: whatever serves it now, expert or base
//! model, and whatever would with the candidate in. The candidate joins only
//! if the population does better on them by more than the margin; one the gate
//! would route nothing to adds nothing, and is not admitted either.

use std::collections::{BTreeSet, HashMap};

use antumbra_core::ports::{EvaluateRequest, TaskScores};
use antumbra_core::slice::Holdout;
use antumbra_core::{
    this_host, Expert, ExpertId, FailureBoundary, Generation, LearnedRouter, Result, RouterExpert,
    RunId,
};
use antumbra_store::repo::{boundary, lifecycle};

use crate::admission::{Admission, AdmissionPolicy};
use crate::contribution::{route_top1, seeds};
use crate::GenerationLoop;

/// Where a task goes with `candidate` in the population: the candidate joins
/// the heuristic gate's pool, or the learned router gains its centroid.
fn route_with(
    task: &[f32],
    router: Option<&LearnedRouter>,
    experts: &[Expert],
    boundaries: &[FailureBoundary],
    candidate: &Expert,
) -> Option<ExpertId> {
    let mut pool = experts.to_vec();
    pool.push(candidate.clone());
    let extended = router.map(|learned| {
        let mut learned = learned.clone();
        if let Some(cap) = candidate.capability_vec.as_deref() {
            let centroid = learned.project(cap);
            if !centroid.is_empty() {
                learned.experts.push(RouterExpert {
                    id: candidate.id.clone(),
                    centroid,
                });
            }
        }
        learned
    });
    route_top1(task, extended.as_ref(), &pool, boundaries, None)
}

/// The population's mean with the candidate and without it, over the tasks
/// whose routing it changes and that were scored both ways.
fn with_and_without(
    changed: &[(String, Option<ExpertId>, Option<ExpertId>)],
    scores: &HashMap<Option<ExpertId>, TaskScores>,
) -> Option<(u32, f32, f32)> {
    let score = |who: &Option<ExpertId>, task: &str| scores.get(who)?.scores.get(task).copied();
    let pairs: Vec<(f32, f32)> = changed
        .iter()
        .filter_map(|(task, with, without)| Some((score(with, task)?, score(without, task)?)))
        .collect();
    let n = pairs.len() as f32;
    (!pairs.is_empty()).then(|| {
        (
            u32::try_from(pairs.len()).unwrap_or(u32::MAX),
            pairs.iter().map(|p| p.0).sum::<f32>() / n,
            pairs.iter().map(|p| p.1).sum::<f32>() / n,
        )
    })
}

impl GenerationLoop<'_> {
    /// Whether the population does no better with `candidate` in it. `None`
    /// when it does better; the rejection otherwise.
    pub(crate) async fn outserved(
        &self,
        run_id: &RunId,
        generation: Generation,
        candidate: &Expert,
        holdout: Option<Holdout>,
        policy: &AdmissionPolicy,
    ) -> Result<Option<Admission>> {
        let host = this_host();
        let mut experts: Vec<Expert> = lifecycle::routable(self.store)
            .await?
            .into_iter()
            .filter(|e| e.owner.is_none() && e.is_placed_on(&host) && e.id != candidate.id)
            .collect();
        experts.sort_by(|a, b| a.id.as_str().cmp(b.id.as_str()));
        let tasks = self.live_sample(holdout, policy.max_tasks).await?;
        let router = lifecycle::load_router(self.store).await?;
        let boundaries = boundary::list(self.store).await?;
        let mut changed = Vec::new();
        for t in &tasks {
            let v = self.embedder.embed(&t.prompt).await?;
            let without = route_top1(&v, router.as_ref(), &experts, &boundaries, None);
            let with = route_with(&v, router.as_ref(), &experts, &boundaries, candidate);
            if with != without {
                changed.push((t.id.clone(), with, without));
            }
        }
        if changed.is_empty() {
            // The gate would route nothing to it: it adds nothing.
            return Ok(Some(Admission::Outserved {
                tasks: 0,
                candidate: 0.0,
                serving: 0.0,
            }));
        }
        let mut needed: HashMap<Option<ExpertId>, BTreeSet<String>> = HashMap::new();
        for (task, with, without) in &changed {
            for who in [with, without] {
                needed.entry(who.clone()).or_default().insert(task.clone());
            }
        }
        let draws = seeds("serving", run_id, generation, policy.seeds);
        let mut scores = HashMap::new();
        for (who, task_ids) in needed {
            let adapter_uri = match &who {
                Some(id) if id == &candidate.id => Some(candidate.artifact_uri.clone()),
                Some(id) => match experts.iter().find(|e| &e.id == id) {
                    Some(e) => Some(e.artifact_uri.clone()),
                    None => continue,
                },
                None => None,
            };
            let scored = self
                .trainer
                .evaluate(EvaluateRequest {
                    label: format!("serving:{run_id}:g{}", generation.0),
                    base_model: self.cfg.base_model.clone(),
                    adapter_uri,
                    task_ids: task_ids.into_iter().collect(),
                    seeds: draws.clone(),
                })
                .await?;
            scores.insert(who, scored);
        }
        Ok(match with_and_without(&changed, &scores) {
            Some((n, with, without)) if with <= without + policy.margin => {
                Some(Admission::Outserved {
                    tasks: n,
                    candidate: with,
                    serving: without,
                })
            }
            _ => None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scores(pairs: &[(&str, f32)]) -> TaskScores {
        TaskScores {
            scores: pairs.iter().map(|(t, s)| (t.to_string(), *s)).collect(),
        }
    }

    /// Each task the candidate reroutes is scored both ways, the base model
    /// wherever the gate would escalate, over the tasks scored both ways.
    #[test]
    fn the_population_is_compared_with_and_without_the_candidate() {
        let (c, g) = (ExpertId::new("c"), ExpertId::new("g"));
        let changed = vec![
            ("mine".to_string(), Some(c.clone()), Some(g.clone())),
            ("escalated".to_string(), None, Some(g.clone())),
            ("unscored".to_string(), Some(c.clone()), None),
        ];
        let by: HashMap<Option<ExpertId>, TaskScores> = [
            (Some(c), scores(&[("mine", 1.0)])),
            (Some(g), scores(&[("mine", 0.5), ("escalated", 0.75)])),
            (None, scores(&[("escalated", 0.25)])),
        ]
        .into_iter()
        .collect();
        assert_eq!(with_and_without(&changed, &by), Some((2, 0.625, 0.625)));
        assert_eq!(with_and_without(&[], &by), None);
    }
}
