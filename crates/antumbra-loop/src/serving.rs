//! Admission against what already serves (ADR-0022 S-5, found on the GPU in
//! S-3): a graduate joins only if it does better than what the population
//! routes the tasks it was trained for to.
//!
//! The twin check catches a candidate that duplicates an expert. It does not
//! catch one that is merely worse than what serves its region. The grow step's
//! first run on the GPU grew two such specialists, each trained from the base
//! on one region's forty tasks. Both did worse there than the generalist
//! trained on everything; both were admitted, since neither was a twin; and
//! the population fell below its best single expert as they drew tasks away.
//!
//! So each of those tasks (the generation's focus, or the live sample when it
//! had none) is routed as the population would route it today. The candidate
//! and whatever serves each task, expert or base model, are scored under the
//! same seeds, and the candidate must do better than the population by more
//! than the margin.

use std::collections::{BTreeSet, HashMap};

use antumbra_core::ports::{EvaluateRequest, TaskPrompt, TaskScores};
use antumbra_core::slice::Holdout;
use antumbra_core::{this_host, Expert, ExpertId, Generation, Result, RunId};
use antumbra_store::repo::{boundary, lifecycle};

use crate::admission::{Admission, AdmissionPolicy};
use crate::contribution::{rank, route_top1, seeds};
use crate::GenerationLoop;

/// The candidate's mean and the serving population's over the tasks both
/// were scored on, and how many there were.
fn against(
    candidate: &TaskScores,
    serving: &[(String, Option<ExpertId>)],
    scores: &HashMap<Option<ExpertId>, TaskScores>,
) -> Option<(u32, f32, f32)> {
    let pairs: Vec<(f32, f32)> = serving
        .iter()
        .filter_map(|(task, who)| {
            let mine = candidate.scores.get(task)?;
            let theirs = scores.get(who)?.scores.get(task)?;
            Some((*mine, *theirs))
        })
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
    /// Whether `candidate` is outserved on the tasks it was trained for.
    /// `None` when it does better than what serves them, or when there is
    /// nothing to judge it on; the rejection otherwise.
    pub(crate) async fn outserved(
        &self,
        run_id: &RunId,
        generation: Generation,
        candidate: &Expert,
        holdout: Option<Holdout>,
        focus: &[String],
        policy: &AdmissionPolicy,
    ) -> Result<Option<Admission>> {
        let host = this_host();
        let mut experts: Vec<Expert> = lifecycle::routable(self.store)
            .await?
            .into_iter()
            .filter(|e| e.owner.is_none() && e.is_placed_on(&host) && e.id != candidate.id)
            .collect();
        experts.sort_by(|a, b| a.id.as_str().cmp(b.id.as_str()));
        let mut tasks: Vec<TaskPrompt> = self
            .trainer
            .live_tasks(holdout)
            .await?
            .into_iter()
            .filter(|t| focus.is_empty() || focus.contains(&t.id))
            .collect();
        tasks.sort_by_key(|t| rank(&t.id));
        tasks.truncate(policy.max_tasks);
        if tasks.is_empty() {
            return Ok(None);
        }
        let router = lifecycle::load_router(self.store).await?;
        let boundaries = boundary::list(self.store).await?;
        let mut serving = Vec::with_capacity(tasks.len());
        for t in &tasks {
            let v = self.embedder.embed(&t.prompt).await?;
            serving.push((
                t.id.clone(),
                route_top1(&v, router.as_ref(), &experts, &boundaries, None),
            ));
        }
        let draws = seeds("serving", run_id, generation, policy.seeds);
        let request = |adapter_uri: Option<String>, task_ids: Vec<String>| EvaluateRequest {
            label: format!("serving:{run_id}:g{}", generation.0),
            base_model: self.cfg.base_model.clone(),
            adapter_uri,
            task_ids,
            seeds: draws.clone(),
        };
        let mine = self
            .trainer
            .evaluate(request(
                Some(candidate.artifact_uri.clone()),
                tasks.iter().map(|t| t.id.clone()).collect(),
            ))
            .await?;
        let mut needed: HashMap<Option<ExpertId>, BTreeSet<String>> = HashMap::new();
        for (task, who) in &serving {
            needed.entry(who.clone()).or_default().insert(task.clone());
        }
        let mut scores = HashMap::new();
        for (who, task_ids) in needed {
            let adapter_uri = match &who {
                Some(id) => match experts.iter().find(|e| &e.id == id) {
                    Some(e) => Some(e.artifact_uri.clone()),
                    None => continue,
                },
                None => None,
            };
            let scored = self
                .trainer
                .evaluate(request(adapter_uri, task_ids.into_iter().collect()))
                .await?;
            scores.insert(who, scored);
        }
        Ok(match against(&mine, &serving, &scores) {
            Some((n, candidate_mean, serving_mean))
                if candidate_mean <= serving_mean + policy.margin =>
            {
                Some(Admission::Outserved {
                    tasks: n,
                    candidate: candidate_mean,
                    serving: serving_mean,
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

    /// Each task is compared with whatever serves it, the base model where
    /// the population escalates, over the tasks both sides scored.
    #[test]
    fn the_candidate_is_compared_task_by_task_with_what_serves_it() {
        let g = ExpertId::new("g");
        let serving = vec![
            ("a".to_string(), Some(g.clone())),
            ("b".to_string(), None),
            ("c".to_string(), Some(g.clone())),
        ];
        let by: HashMap<Option<ExpertId>, TaskScores> = [
            (Some(g), scores(&[("a", 1.0), ("c", 0.5)])),
            (None, scores(&[("b", 0.0)])),
        ]
        .into_iter()
        .collect();
        let mine = scores(&[("a", 0.5), ("b", 1.0)]);
        assert_eq!(against(&mine, &serving, &by), Some((2, 0.75, 0.5)));
        assert_eq!(against(&scores(&[]), &serving, &by), None);
    }
}
