//! Admission gating (ADR-0022 S-5): before a shadow graduates, it is checked
//! against the population it would join, so the population does not fill with
//! twins.
//!
//! A twin costs more than disk. The heuristic gate routes on the margin
//! between its two best experts, and two experts minted from the same corpus
//! have capability vectors so close that the margin vanishes: on the GPU, the
//! second such expert left both unused, every live task escalated, where either
//! alone took them all. The learned router does not escalate, but splits their
//! traffic by noise.
//!
//! So a candidate whose capability vector is within `duplicate_above` of an
//! active shared expert is measured against it head to head: both adapters
//! on the same live tasks under the same seeds. It is admitted only if it does
//! better by more than `margin`, and then the expert it duplicates is
//! archived as redundant with it: out of routing and serving, weights kept,
//! revivable, never deleted. Otherwise it is not admitted, and the population
//! does not grow. Merging the two instead is the conservative alternative the
//! record also names; this is the one that needs no merge.

use antumbra_core::ports::EvaluateRequest;
use antumbra_core::slice::Holdout;
use antumbra_core::{this_host, Expert, ExpertId, Generation, Result, RunId};
use antumbra_store::repo::lifecycle;

use crate::contribution::seeds;
use crate::GenerationLoop;

/// When a candidate counts as a duplicate, and how it is measured against it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdmissionPolicy {
    /// Cosine similarity of capability vectors at or above which a candidate
    /// duplicates an expert.
    pub duplicate_above: f32,
    /// Evaluations per task for each side, each under its own seed.
    pub seeds: u32,
    /// At most this many live tasks in the head-to-head.
    pub max_tasks: usize,
    /// How much better than the duplicate a candidate must score to replace it.
    pub margin: f32,
}

impl Default for AdmissionPolicy {
    fn default() -> Self {
        Self {
            duplicate_above: 0.95,
            seeds: 2,
            max_tasks: 32,
            margin: 0.0,
        }
    }
}

/// What admission decided for a candidate that cleared graduation.
#[derive(Debug, Clone, PartialEq)]
pub enum Admission {
    /// No active shared expert was close enough to duplicate it. `nearest` is
    /// the closest one and its similarity, when there is any.
    Admitted { nearest: Option<(ExpertId, f32)> },
    /// It duplicated `archived` and scored better head to head, so it took its
    /// place: the other was archived as redundant with it.
    Superseded {
        archived: ExpertId,
        similarity: f32,
        candidate: f32,
        incumbent: f32,
    },
    /// It duplicated `duplicate_of` and did not score better, or could not be
    /// measured against it, so it was not admitted.
    Rejected {
        duplicate_of: ExpertId,
        similarity: f32,
        candidate: Option<f32>,
        incumbent: Option<f32>,
    },
}

impl Admission {
    pub fn admits(&self) -> bool {
        !matches!(self, Admission::Rejected { .. })
    }
}

/// The means of the pairs both sides scored, so they stay paired.
fn paired_means(
    candidate: &antumbra_core::ports::TaskScores,
    incumbent: &antumbra_core::ports::TaskScores,
) -> Option<(f32, f32)> {
    let pairs: Vec<(f32, f32)> = candidate
        .scores
        .iter()
        .filter_map(|(task, &c)| incumbent.scores.get(task).map(|&i| (c, i)))
        .collect();
    let n = pairs.len() as f32;
    (!pairs.is_empty()).then(|| {
        (
            pairs.iter().map(|p| p.0).sum::<f32>() / n,
            pairs.iter().map(|p| p.1).sum::<f32>() / n,
        )
    })
}

impl GenerationLoop<'_> {
    /// Decide whether `candidate`, which cleared graduation, joins the
    /// population. `None` when no admission policy is set: it joins, as every
    /// graduate did before.
    pub(crate) async fn admission(
        &self,
        run_id: &RunId,
        generation: Generation,
        candidate: &Expert,
        holdout: Option<Holdout>,
    ) -> Result<Option<Admission>> {
        let Some(policy) = self.cfg.admission else {
            return Ok(None);
        };
        let Some(cap) = candidate.capability_vec.as_deref() else {
            return Ok(Some(Admission::Admitted { nearest: None }));
        };
        // A generation run again may find its own first graduate; that is not
        // a twin.
        let nearest = lifecycle::routable(self.store)
            .await?
            .into_iter()
            .filter(|e| e.owner.is_none() && e.id != candidate.id)
            .filter_map(|e| e.capability_similarity(cap).map(|s| (e, s)))
            .filter(|(_, s)| s.is_finite())
            .max_by(|a, b| a.1.total_cmp(&b.1));
        let (twin, similarity) = match nearest {
            Some((e, s)) if s >= policy.duplicate_above => (e, s),
            other => {
                return Ok(Some(Admission::Admitted {
                    nearest: other.map(|(e, s)| (e.id, s)),
                }))
            }
        };
        let rejected = |candidate, incumbent| Admission::Rejected {
            duplicate_of: twin.id.clone(),
            similarity,
            candidate,
            incumbent,
        };
        // A duplicate held on another node cannot be measured here, and a
        // candidate that cannot show it is better does not replace it.
        if !twin.is_placed_on(&this_host()) {
            return Ok(Some(rejected(None, None)));
        }
        let tasks = self.live_sample(holdout, policy.max_tasks).await?;
        let task_ids: Vec<String> = tasks.into_iter().map(|t| t.id).collect();
        let draws = seeds("admission", run_id, generation, policy.seeds);
        let score = |adapter_uri: &str| EvaluateRequest {
            label: format!("admission:{run_id}:g{}", generation.0),
            base_model: self.cfg.base_model.clone(),
            adapter_uri: Some(adapter_uri.to_string()),
            task_ids: task_ids.clone(),
            seeds: draws.clone(),
        };
        let theirs = self.trainer.evaluate(score(&twin.artifact_uri)).await?;
        let ours = self
            .trainer
            .evaluate(score(&candidate.artifact_uri))
            .await?;
        let Some((mine, incumbent)) = paired_means(&ours, &theirs) else {
            return Ok(Some(rejected(None, None)));
        };
        Ok(Some(if mine > incumbent + policy.margin {
            Admission::Superseded {
                archived: twin.id,
                similarity,
                candidate: mine,
                incumbent,
            }
        } else {
            rejected(Some(mine), Some(incumbent))
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antumbra_core::ports::TaskScores;

    fn scores(pairs: &[(&str, f32)]) -> TaskScores {
        TaskScores {
            scores: pairs.iter().map(|(t, s)| (t.to_string(), *s)).collect(),
        }
    }

    #[test]
    fn only_tasks_both_sides_scored_are_compared() {
        let ours = scores(&[("a", 1.0), ("b", 0.5), ("c", 1.0)]);
        let theirs = scores(&[("a", 0.5), ("b", 0.5)]);
        assert_eq!(paired_means(&ours, &theirs), Some((0.75, 0.5)));
        assert_eq!(paired_means(&ours, &scores(&[])), None);
    }

    #[test]
    fn only_a_rejection_keeps_a_candidate_out() {
        let id = ExpertId::new("expert:x");
        assert!(Admission::Admitted { nearest: None }.admits());
        assert!(Admission::Superseded {
            archived: id.clone(),
            similarity: 0.99,
            candidate: 0.8,
            incumbent: 0.7,
        }
        .admits());
        assert!(!Admission::Rejected {
            duplicate_of: id,
            similarity: 0.99,
            candidate: None,
            incumbent: None,
        }
        .admits());
    }
}
