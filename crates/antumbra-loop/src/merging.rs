//! Conservative, reversible merging (ADR-0022 S-5): two sibling experts become
//! one, only when the merge costs nothing measurable.
//!
//! At most one merge a generation, of the most similar pair of active shared
//! experts, and only if:
//! - their capability vectors are at least `similar_above` alike: they are
//!   asked for the same things;
//! - their adapters are siblings, sharing their subspace. The merge averages
//!   the two deltas and keeps them at the population's rank, and the share of
//!   the average's energy that rank keeps is the overlap, required at least
//!   `retained_above`;
//! - the merged adapter, scored on the live tasks under the same seeds as the
//!   two it would replace, does at least as well as the better of them (less
//!   `margin`). That is the cost measured before and after rather than
//!   assumed.
//!
//! Then the merged expert enters the population and both originals are
//! archived as redundant with it: out of routing and serving, weights kept,
//! revivable, so the merge is undone by reviving them. Otherwise nothing
//! changes and the merged file is removed.
//!
//! The record also asks for low cumulative training, because the most-trained
//! experts merge worst. Under recipe-only propagation every expert trains from
//! the base on the same budget, so that condition holds for every pair here,
//! and the measurement guards the rest.

use chrono::Utc;

use antumbra_core::ports::{EvaluateRequest, MergeRequest, TaskScores};
use antumbra_core::slice::Holdout;
use antumbra_core::{this_host, Expert, ExpertId, Generation, Result, RunId};
use antumbra_store::repo::lifecycle;

use crate::contribution::seeds;
use crate::GenerationLoop;

/// When two experts are merged.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MergePolicy {
    /// Capability similarity at or above which a pair is a candidate.
    pub similar_above: f32,
    /// Share of the averaged delta's energy the population's rank must keep.
    pub retained_above: f32,
    /// Evaluations per task for each side, each under its own seed.
    pub seeds: u32,
    /// At most this many live tasks in the comparison.
    pub max_tasks: usize,
    /// How far below the better original the merged adapter may score.
    pub margin: f32,
}

impl Default for MergePolicy {
    fn default() -> Self {
        Self {
            similar_above: 0.85,
            retained_above: 0.9,
            seeds: 2,
            max_tasks: 32,
            margin: 0.0,
        }
    }
}

/// What merging did in a generation.
#[derive(Debug, Clone, PartialEq)]
pub enum Merge {
    /// The pair became `into`; both were archived as redundant with it.
    Merged {
        into: ExpertId,
        pair: (ExpertId, ExpertId),
        similarity: f32,
        retained: f32,
        merged: f32,
        better: f32,
    },
    /// The pair's adapters do not share enough of their subspace.
    NotSiblings {
        pair: (ExpertId, ExpertId),
        similarity: f32,
        retained: f32,
    },
    /// The merged adapter scored below the better of the two.
    Costly {
        pair: (ExpertId, ExpertId),
        similarity: f32,
        retained: f32,
        merged: f32,
        better: f32,
    },
}

/// Means over the tasks all three were scored on: (merged, left, right).
fn triple_means(
    merged: &TaskScores,
    left: &TaskScores,
    right: &TaskScores,
) -> Option<(f32, f32, f32)> {
    let rows: Vec<(f32, f32, f32)> = merged
        .scores
        .iter()
        .filter_map(|(t, &m)| Some((m, *left.scores.get(t)?, *right.scores.get(t)?)))
        .collect();
    let n = rows.len() as f32;
    (!rows.is_empty()).then(|| {
        (
            rows.iter().map(|r| r.0).sum::<f32>() / n,
            rows.iter().map(|r| r.1).sum::<f32>() / n,
            rows.iter().map(|r| r.2).sum::<f32>() / n,
        )
    })
}

/// Where a merged adapter is written: beside the first of the two.
fn merged_path(left: &str, run_id: &RunId, generation: Generation) -> String {
    let name =
        format!("merged-{run_id}-g{}.safetensors", generation.0).replace([':', '/', '\\'], "_");
    match std::path::Path::new(left).parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir.join(name).to_string_lossy().into_owned(),
        _ => name,
    }
}

/// The union of two cards' exemplars, the merged expert's card.
fn exemplars(a: &Expert, b: &Expert) -> Vec<serde_json::Value> {
    let mut all: Vec<serde_json::Value> = Vec::new();
    for e in [a, b] {
        for x in e
            .capability_card
            .get("exemplars")
            .and_then(|v| v.as_array())
            .into_iter()
            .flatten()
        {
            if !all.contains(x) {
                all.push(x.clone());
            }
        }
    }
    all
}

/// The mean of two capability vectors, normalized.
fn mean_vector(a: &[f32], b: &[f32]) -> Option<Vec<f32>> {
    if a.len() != b.len() {
        return None;
    }
    let sum: Vec<f32> = a.iter().zip(b).map(|(x, y)| x + y).collect();
    let norm = sum.iter().map(|v| v * v).sum::<f32>().sqrt();
    (norm > 0.0).then(|| sum.iter().map(|v| v / norm).collect())
}

impl GenerationLoop<'_> {
    /// Consider merging the most similar pair of active shared experts this
    /// node holds. `None` when no policy is set or no pair is similar enough.
    pub(crate) async fn consider_merge(
        &self,
        run_id: &RunId,
        generation: Generation,
        holdout: Option<Holdout>,
    ) -> Result<Option<Merge>> {
        let Some(policy) = self.cfg.merge else {
            return Ok(None);
        };
        let host = this_host();
        let experts: Vec<Expert> = lifecycle::routable(self.store)
            .await?
            .into_iter()
            .filter(|e| e.owner.is_none() && e.is_placed_on(&host))
            .collect();
        let mut best: Option<(usize, usize, f32)> = None;
        for i in 0..experts.len() {
            for j in i + 1..experts.len() {
                let Some(v) = experts[j].capability_vec.as_deref() else {
                    continue;
                };
                let Some(s) = experts[i]
                    .capability_similarity(v)
                    .filter(|s| s.is_finite())
                else {
                    continue;
                };
                if s >= policy.similar_above && best.is_none_or(|b| s > b.2) {
                    best = Some((i, j, s));
                }
            }
        }
        let Some((i, j, similarity)) = best else {
            return Ok(None);
        };
        let (a, b) = (&experts[i], &experts[j]);
        let pair = (a.id.clone(), b.id.clone());
        let overlap = self
            .trainer
            .merge(MergeRequest {
                left: a.artifact_uri.clone(),
                right: b.artifact_uri.clone(),
                out: None,
            })
            .await?;
        if overlap.retained < policy.retained_above {
            return Ok(Some(Merge::NotSiblings {
                pair,
                similarity,
                retained: overlap.retained,
            }));
        }
        let out = merged_path(&a.artifact_uri, run_id, generation);
        let written = self
            .trainer
            .merge(MergeRequest {
                left: a.artifact_uri.clone(),
                right: b.artifact_uri.clone(),
                out: Some(out.clone()),
            })
            .await?;
        let tasks = self.live_sample(holdout, policy.max_tasks).await?;
        let task_ids: Vec<String> = tasks.into_iter().map(|t| t.id).collect();
        let draws = seeds("merge", run_id, generation, policy.seeds);
        let score = |adapter_uri: &str| EvaluateRequest {
            label: format!("merge:{run_id}:g{}", generation.0),
            base_model: self.cfg.base_model.clone(),
            adapter_uri: Some(adapter_uri.to_string()),
            task_ids: task_ids.clone(),
            seeds: draws.clone(),
        };
        let merged_scores = self.trainer.evaluate(score(&out)).await?;
        let left_scores = self.trainer.evaluate(score(&a.artifact_uri)).await?;
        let right_scores = self.trainer.evaluate(score(&b.artifact_uri)).await?;
        let means = triple_means(&merged_scores, &left_scores, &right_scores);
        let Some((merged, left, right)) = means.filter(|(m, l, r)| m + policy.margin >= l.max(*r))
        else {
            let _ = std::fs::remove_file(&out);
            let (merged, better) = means.map_or((0.0, 0.0), |(m, l, r)| (m, l.max(r)));
            return Ok(Some(Merge::Costly {
                pair,
                similarity,
                retained: written.retained,
                merged,
                better,
            }));
        };
        let now = Utc::now();
        let into = Expert {
            id: ExpertId::new(format!("expert:{run_id}:g{}:merged", generation.0)),
            name: format!("{run_id}-g{}-merged", generation.0),
            base_model: self.cfg.base_model.clone(),
            artifact_uri: out,
            capability_card: serde_json::json!({
                "generation": generation.0,
                "exemplars": exemplars(a, b),
                "merged_from": [a.id.as_str(), b.id.as_str()],
            }),
            capability_vec: a
                .capability_vec
                .as_deref()
                .zip(b.capability_vec.as_deref())
                .and_then(|(x, y)| mean_vector(x, y)),
            fitness: merged,
            frozen_at: Some(now),
            generation,
            owner: None,
            compartment: None,
            placed_on: Some(host),
            created_at: now,
        };
        self.enter(run_id, generation, &into, merged).await?;
        self.archive_as_redundant(&a.id, &into.id, generation)
            .await?;
        self.archive_as_redundant(&b.id, &into.id, generation)
            .await?;
        Ok(Some(Merge::Merged {
            into: into.id,
            pair,
            similarity,
            retained: written.retained,
            merged,
            better: left.max(right),
        }))
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

    #[test]
    fn only_tasks_all_three_scored_are_compared() {
        let m = scores(&[("a", 1.0), ("b", 0.5), ("c", 1.0)]);
        let l = scores(&[("a", 0.5), ("b", 0.5)]);
        let r = scores(&[("a", 1.0), ("b", 0.0), ("c", 0.0)]);
        assert_eq!(triple_means(&m, &l, &r), Some((0.75, 0.5, 0.5)));
        assert_eq!(triple_means(&m, &scores(&[]), &r), None);
    }

    #[test]
    fn a_merged_adapter_lands_beside_the_first_and_is_named_for_its_generation() {
        let run = RunId::new("search:abc");
        let path = merged_path("/reports/adapters/x.safetensors", &run, Generation(3));
        assert!(path.ends_with("merged-search_abc-g3.safetensors"), "{path}");
        assert!(path.starts_with("/reports/adapters"), "{path}");
        assert_eq!(
            merged_path("x.safetensors", &run, Generation(0)),
            "merged-search_abc-g0.safetensors"
        );
    }

    #[test]
    fn the_merged_card_and_vector_cover_both() {
        let v = mean_vector(&[1.0, 0.0], &[0.0, 1.0]).expect("a vector");
        assert!((v[0] - v[1]).abs() < 1e-6 && (v[0] * v[0] + v[1] * v[1] - 1.0).abs() < 1e-6);
        assert!(mean_vector(&[1.0], &[1.0, 0.0]).is_none());
        assert!(mean_vector(&[1.0, 0.0], &[-1.0, 0.0]).is_none());
    }
}
