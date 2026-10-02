//! The loop's recheck of the verifiers that judged its training (ADR-0022
//! S-4): trust re-measured on the loop's own schedule, on the answers of the
//! policy a verifier is rewarding.
//!
//! A trusted synthesized verifier was promoted on cases, and a check that was
//! sound against one population is not thereby sound against the next. So in
//! every generation, each trusted synthesized verifier that judged answers on
//! the tasks training learned from is measured again on those answers:
//! - **The label:** each answer is judged by the task's anchor, a trusted
//!   authored verifier in the same domain, as many times as the protocol runs
//!   a case. An answer the anchor disagrees with itself on is left out, as the
//!   cases builder leaves it out.
//! - **The two rates:** the verifier's verdicts and the anchor's on the same
//!   answers are the visible and held-out pass rates the record names.
//!   Training saw the first and never the second.
//! - **The measurement:** it goes through the trust protocol and is recorded
//!   like any other. A verifier passing the policy's wrong answers past its
//!   bound is quarantined, taking every expert it taught with it. A sound one
//!   has its trust renewed.
//!
//! A shadow that trained under a verifier that no longer grants reward, by
//! this recheck or anyone's move, does not graduate.

use std::collections::BTreeSet;

use chrono::Utc;

use antumbra_core::ports::{TrainOutcome, Verifier, VerifyRequest};
use antumbra_core::{
    Anchor, Case, ExpertId, JudgedSample, Label, Result, RunId, Tally, TrustMeasurement,
    TrustPolicy, TrustState, VerifierId, VerifierOrigin, VerifierRecord,
};
use antumbra_store::repo::verifier;

use crate::GenerationLoop;

/// The most answers a verifier is rechecked on in one generation, spread
/// evenly over what it judged. Each costs the anchor's runs and the
/// verifier's.
pub const MAX_RECHECKED: usize = 128;

/// What the recheck found for one verifier.
#[derive(Debug, Clone, PartialEq)]
pub struct Recheck {
    pub verifier: VerifierId,
    /// Answers the anchor labeled, which the verifier was measured on.
    pub anchored: u32,
    /// Answers with no anchor, or whose anchor disagreed with itself.
    pub unanchored: u32,
    /// Rewarded answers the anchor failed: what the verifier taught that is
    /// not true.
    pub rewarded_wrong: u32,
    /// The measurement, when any answer was anchored.
    pub measurement: Option<TrustMeasurement>,
    /// Where the measurement moved the verifier, when it moved it.
    pub moved: Option<TrustState>,
    /// The experts that went out of use with it.
    pub archived: Vec<ExpertId>,
    /// How many of this run's generations the measurement pools: each
    /// recheck adds its answers to the ones this run already rechecked the
    /// verifier on, so a verifier on a task the policy rarely fails is judged
    /// on more than one generation's handful.
    pub generations: u32,
}

/// What a run has rechecked one verifier on so far: the generations, the
/// pooled counts, and the tasks whose reference answer is already in them.
#[derive(Debug, Default)]
pub(crate) struct Pooled {
    generations: u32,
    tally: Tally,
    referenced: BTreeSet<String>,
}

/// At most `max` of `items`, evenly spaced, in order.
pub(crate) fn spread<T: Copy>(items: &[T], max: usize) -> Vec<T> {
    if items.len() <= max {
        return items.to_vec();
    }
    (0..max).map(|i| items[i * items.len() / max]).collect()
}

/// The anchor for `task` when `checked` is rechecked: a trusted authored
/// verifier in its domain that applies to the task, one written for the task
/// preferred over one for the whole domain.
pub(crate) fn anchor_for<'r>(
    anchors: &'r [VerifierRecord],
    checked: &VerifierRecord,
    task: &str,
) -> Option<&'r VerifierRecord> {
    anchors
        .iter()
        .filter(|a| a.domain == checked.domain && a.applies_to(task))
        .max_by_key(|a| a.task.is_some())
}

/// The anchor's verdict on `completion` for `task`, or `None` when its runs
/// disagree.
async fn label(
    judge: &dyn Verifier,
    anchor: &VerifierRecord,
    task: &str,
    completion: &str,
    repeats: u32,
) -> Result<Option<bool>> {
    let run_id = RunId::new(format!("recheck:{}", anchor.id));
    let mut runs = Vec::new();
    for step_idx in 0..repeats.max(1) {
        let req = VerifyRequest {
            run_id: run_id.clone(),
            step_idx,
            dimension: "anchor".into(),
            artifact: serde_json::json!({
                "task": task,
                "completion": completion,
                "verify": anchor.spec,
            }),
        };
        runs.push(judge.verify(&req).await?.passed);
    }
    Ok(runs.iter().all(|&r| r == runs[0]).then_some(runs[0]))
}

impl GenerationLoop<'_> {
    /// Recheck every trusted synthesized verifier that judged `outcome`'s
    /// training. Nothing when the loop has no verifier to recheck with.
    pub(crate) async fn recheck(&self, outcome: &TrainOutcome) -> Result<Vec<Recheck>> {
        let Some(judge) = self.rechecker else {
            return Ok(Vec::new());
        };
        let judged: BTreeSet<&str> = outcome.judged.iter().map(|j| j.verifier.as_str()).collect();
        if judged.is_empty() {
            return Ok(Vec::new());
        }
        let records = verifier::list(self.store).await?;
        let mut checked = Vec::new();
        for record in records
            .iter()
            .filter(|r| r.origin == VerifierOrigin::Synthesized && judged.contains(r.id.as_str()))
        {
            if verifier::state_of(self.store, record).await? == TrustState::Trusted {
                checked.push(record);
            }
        }
        let domains: BTreeSet<&str> = checked.iter().map(|r| r.domain.as_str()).collect();
        let mut anchors = Vec::new();
        for record in records
            .iter()
            .filter(|r| r.origin == VerifierOrigin::Authored && domains.contains(r.domain.as_str()))
        {
            if verifier::state_of(self.store, record).await? == TrustState::Trusted {
                anchors.push(record.clone());
            }
        }
        let policy = TrustPolicy::default();
        let mut rechecks = Vec::new();
        for record in checked {
            let mine: Vec<&JudgedSample> = outcome
                .judged
                .iter()
                .filter(|j| j.verifier == record.id)
                .collect();
            let samples = spread(&mine, MAX_RECHECKED);
            rechecks.push(
                self.recheck_one(judge, record, &anchors, &samples, &policy)
                    .await?,
            );
        }
        Ok(rechecks)
    }

    async fn recheck_one(
        &self,
        judge: &dyn Verifier,
        record: &VerifierRecord,
        anchors: &[VerifierRecord],
        samples: &[&JudgedSample],
        policy: &TrustPolicy,
    ) -> Result<Recheck> {
        let mut cases = Vec::new();
        let (mut unanchored, mut rewarded_wrong) = (0u32, 0u32);
        for (i, sample) in samples.iter().enumerate() {
            let Some(anchor) = anchor_for(anchors, record, &sample.task) else {
                unanchored += 1;
                continue;
            };
            let Some(right) = label(
                judge,
                anchor,
                &sample.task,
                &sample.completion,
                policy.repeats,
            )
            .await?
            else {
                unanchored += 1;
                continue;
            };
            rewarded_wrong += u32::from(sample.rewarded && !right);
            cases.push(Case {
                id: format!("{}#{i}", sample.task),
                task: sample.task.clone(),
                completion: sample.completion.clone(),
                label: if right { Label::Good } else { Label::Bad },
                anchor: Anchor::Verifier {
                    id: anchor.id.clone(),
                },
            });
        }
        cases.extend(
            self.seeded_cases(judge, record, anchors, samples, policy)
                .await?,
        );
        let mut recheck = Recheck {
            verifier: record.id.clone(),
            anchored: u32::try_from(cases.len()).unwrap_or(u32::MAX),
            unanchored,
            rewarded_wrong,
            measurement: None,
            moved: None,
            archived: Vec::new(),
            generations: 0,
        };
        if cases.is_empty() {
            return Ok(recheck);
        }
        let mut pooled = antumbra_critic::trust::tally(judge, record, &cases, policy).await?;
        let generations = {
            let mut earlier = self.rechecked.lock().unwrap_or_else(|e| e.into_inner());
            let entry = earlier.entry(record.id.as_str().to_string()).or_default();
            pooled.merge(&entry.tally);
            entry.tally = pooled.clone();
            entry.generations += 1;
            entry.generations
        };
        recheck.generations = generations;
        let measurement = pooled.judge(&record.id, Utc::now(), policy);
        if let Some(moved) = verifier::record_measurement(self.store, record, &measurement).await? {
            recheck.moved = Some(moved.transition.to);
            recheck.archived = moved.archived;
        }
        recheck.measurement = Some(measurement);
        Ok(recheck)
    }

    /// The answers the run brings besides the policy's, for each task
    /// `samples` cover that this run has not yet seeded for `record`: the
    /// task's reference answer, and the answers built to be wrong
    /// ([`GenerationLoop::rechecking_against`]). Each is labeled by the
    /// task's anchor like the policy's, and counted once a run, so pooling
    /// never weighs the same answer twice.
    async fn seeded_cases(
        &self,
        judge: &dyn Verifier,
        record: &VerifierRecord,
        anchors: &[VerifierRecord],
        samples: &[&JudgedSample],
        policy: &TrustPolicy,
    ) -> Result<Vec<Case>> {
        let tasks: BTreeSet<&str> = samples.iter().map(|s| s.task.as_str()).collect();
        let mut cases = Vec::new();
        for task in tasks {
            let fresh = {
                let mut earlier = self.rechecked.lock().unwrap_or_else(|e| e.into_inner());
                let entry = earlier.entry(record.id.as_str().to_string()).or_default();
                entry.referenced.insert(task.to_string())
            };
            if !fresh {
                continue;
            }
            let Some(anchor) = anchor_for(anchors, record, task) else {
                continue;
            };
            let anchored = |id: String, completion: String, label: Label| Case {
                id,
                task: task.to_string(),
                completion,
                label,
                anchor: Anchor::Verifier {
                    id: anchor.id.clone(),
                },
            };
            if let Some(reference) = self.trainer.reference(task).await {
                if let Some(right) = label(judge, anchor, task, &reference, policy.repeats).await? {
                    let as_labeled = if right { Label::Good } else { Label::Bad };
                    cases.push(anchored(format!("{task}#reference"), reference, as_labeled));
                }
            }
            for (i, wrong) in self.deliberate.get(task).into_iter().flatten().enumerate() {
                if let Some(right) = label(judge, anchor, task, wrong, policy.repeats).await? {
                    let as_labeled = if right {
                        Label::Good
                    } else {
                        Label::Adversarial
                    };
                    cases.push(anchored(
                        format!("{task}#deliberate{i}"),
                        wrong.clone(),
                        as_labeled,
                    ));
                }
            }
        }
        Ok(cases)
    }

    /// The named verifiers `outcome` trained under that no longer grant
    /// reward: quarantined or revoked, by the recheck or by anyone else.
    pub(crate) async fn withdrawn(&self, outcome: &TrainOutcome) -> Result<Vec<VerifierId>> {
        let mut out = Vec::new();
        for grant in &outcome.granted_by {
            let Some(record) = verifier::get(self.store, &grant.verifier).await? else {
                continue;
            };
            let state = verifier::state_of(self.store, &record).await?;
            if matches!(state, TrustState::Quarantined | TrustState::Revoked) {
                out.push(grant.verifier.clone());
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antumbra_core::VerifierTier;

    fn record(domain: &str, task: Option<&str>, spec: &str) -> VerifierRecord {
        VerifierRecord::new(
            domain,
            task.map(str::to_string),
            VerifierTier::Reducible,
            VerifierOrigin::Authored,
            serde_json::json!({ "spec": spec }),
            Utc::now(),
        )
    }

    #[test]
    fn the_spread_keeps_every_answer_up_to_the_cap_then_spaces_them() {
        let xs: Vec<u32> = (0..10).collect();
        assert_eq!(spread(&xs, 20), xs);
        assert_eq!(spread(&xs, 5), vec![0, 2, 4, 6, 8]);
        assert!(spread(&xs, 0).is_empty());
    }

    #[test]
    fn a_task_anchor_is_preferred_and_the_domain_must_match() {
        let checked = record("strings", Some("swap"), "checked");
        let domain = record("strings", None, "domain");
        let task = record("strings", Some("swap"), "task");
        let other = record("grids", Some("swap"), "other");
        let anchors = vec![domain.clone(), task.clone(), other];
        assert_eq!(anchor_for(&anchors, &checked, "swap"), Some(&task));
        assert_eq!(anchor_for(&anchors, &checked, "caesar"), Some(&domain));
        assert_eq!(anchor_for(&anchors[2..], &checked, "swap"), None);
    }
}
