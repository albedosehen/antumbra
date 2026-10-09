//! Retirement as the loop's job: two detectors in series,
//! never one.
//!
//! The first is an early warning, label-free and advisory only. It reads what
//! the contribution stream says about the inputs routed to each expert and
//! its share of them:
//! - its share fell to under half of what it had been;
//! - it went unused, after tasks had been routed to it;
//! - what it is asked to do drifted away from what it was minted for.
//!
//! It is reported and changes nothing, because drift detectors are sensitive
//! and a routing shift alone is not evidence of uselessness.
//!
//! The second is the only thing permitted to change state: confirmation on the
//! leave-one-out stream. An expert is demoted, to dormant and never further,
//! only when its contribution on the tasks routed to it was at or below the
//! floor in each of its last `persist` measurements, each resting on at least
//! `min_routed` tasks. Every such measurement is kept with the move as its
//! evidence. Retirement's validation holds by construction: no expert whose
//! latest measurement shows it contributing is demoted. An unused expert is
//! never demoted either: unused is not useless, and the cure for a twin is
//! admission, not retirement.
//!
//! Only measurements since the expert last became active count, so a
//! person's revive starts its stream afresh.

use antumbra_core::{
    ContributionRecord, ExpertId, ExpertStatus, ExpertTransition, Generation, Result,
    TransitionCause,
};
use antumbra_store::repo::{contribution, lifecycle};

use crate::GenerationLoop;

/// When confirmation demotes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RetirementPolicy {
    /// A contribution at or below this adds nothing over what the gate falls
    /// back to.
    pub floor: f32,
    /// Consecutive measurements at or below the floor that confirm staleness.
    pub persist: u32,
    /// Tasks a measurement must rest on to count as evidence.
    pub min_routed: u32,
}

impl Default for RetirementPolicy {
    fn default() -> Self {
        Self {
            floor: 0.0,
            persist: 3,
            min_routed: 2,
        }
    }
}

/// An early warning: advisory, never a move.
#[derive(Debug, Clone, PartialEq)]
pub enum Warning {
    /// Its share of the live tasks fell to under half its earlier mean.
    ShareFell { share: f32, was: f32 },
    /// Nothing was routed to it, where something had been.
    WentUnused,
    /// The inputs routed to it sit further from its capability vector than
    /// they did, by more than a tenth of cosine similarity.
    Drifted { affinity: f32, was: f32 },
}

/// How far a mean affinity may fall before it warns.
const DRIFT: f32 = 0.1;

fn mean(values: impl Iterator<Item = f32>) -> Option<f32> {
    let (n, sum) = values.fold((0u32, 0.0f32), |(n, s), v| (n + 1, s + v));
    (n > 0).then(|| sum / n as f32)
}

/// The early warnings in a contribution stream, oldest first, about its last
/// measurement against those before it.
pub fn warnings(stream: &[ContributionRecord]) -> Vec<Warning> {
    let Some((now, before)) = stream.split_last() else {
        return Vec::new();
    };
    let mut found = Vec::new();
    if now.is_unused() {
        if before.last().is_some_and(|r| !r.is_unused()) {
            found.push(Warning::WentUnused);
        }
    } else if let Some(was) = mean(before.iter().map(ContributionRecord::share)) {
        if was > 0.0 && now.share() < was / 2.0 {
            found.push(Warning::ShareFell {
                share: now.share(),
                was,
            });
        }
    }
    if let (Some(affinity), Some(was)) =
        (now.affinity, mean(before.iter().filter_map(|r| r.affinity)))
    {
        if affinity < was - DRIFT {
            found.push(Warning::Drifted { affinity, was });
        }
    }
    found
}

/// Whether the stream confirms staleness: its last `persist` measurements
/// each rest on at least `min_routed` tasks and show a contribution at or
/// below the floor. Returns them as the evidence.
pub fn confirms(
    stream: &[ContributionRecord],
    policy: &RetirementPolicy,
) -> Option<Vec<(Generation, f32)>> {
    let persist = usize::try_from(policy.persist.max(1)).unwrap_or(usize::MAX);
    let window = stream.get(stream.len().checked_sub(persist)?..)?;
    window
        .iter()
        .map(|r| {
            let delta = r.delta()?;
            (r.routed >= policy.min_routed && delta <= policy.floor)
                .then_some((r.generation, delta))
        })
        .collect()
}

/// What the detectors found in a generation.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Detection {
    /// Advisory: reported, never acted on.
    pub warnings: Vec<(ExpertId, Warning)>,
    /// The demotions confirmation made, with their evidence.
    pub demoted: Vec<ExpertTransition>,
}

impl GenerationLoop<'_> {
    /// Run both detectors over every expert this generation measured.
    pub(crate) async fn detect(
        &self,
        generation: Generation,
        measured: &[ContributionRecord],
    ) -> Result<Detection> {
        let mut found = Detection::default();
        let Some(policy) = self.cfg.retirement else {
            return Ok(found);
        };
        for record in measured {
            let stream = self.stream_since_active(&record.expert).await?;
            for w in warnings(&stream) {
                found.warnings.push((record.expert.clone(), w));
            }
            if let Some(evidence) = confirms(&stream, &policy) {
                let (generations, contributions) = evidence.into_iter().unzip();
                let moved = lifecycle::transition(
                    self.store,
                    &record.expert,
                    ExpertStatus::Dormant,
                    TransitionCause::Stale {
                        generations,
                        contributions,
                    },
                    Some(generation),
                )
                .await?;
                found.demoted.push(moved);
            }
        }
        Ok(found)
    }

    /// An expert's contribution stream, oldest first, counting only
    /// measurements taken since it last became active.
    async fn stream_since_active(&self, expert: &ExpertId) -> Result<Vec<ContributionRecord>> {
        let since = lifecycle::history(self.store, expert)
            .await?
            .into_iter()
            .filter(|t| t.to == ExpertStatus::Active)
            .map(|t| t.at)
            .max();
        let mut stream: Vec<ContributionRecord> = contribution::history(self.store, expert)
            .await?
            .into_iter()
            .filter(|r| since.is_none_or(|at| r.at > at))
            .collect();
        stream.sort_by_key(|r| r.at);
        Ok(stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antumbra_core::RunId;
    use chrono::{Duration, Utc};

    fn measured(
        n: u32,
        routed: u32,
        delta: Option<f32>,
        affinity: Option<f32>,
    ) -> ContributionRecord {
        ContributionRecord {
            expert: ExpertId::new("expert:e"),
            run_id: RunId::new("run"),
            generation: Generation(n),
            routed,
            tasks: 10,
            with: delta.map(|d| 0.5 + d),
            without: delta.map(|_| 0.5),
            seeds: 2,
            affinity,
            at: Utc::now() + Duration::seconds(i64::from(n)),
        }
    }

    fn policy() -> RetirementPolicy {
        RetirementPolicy::default()
    }

    #[test]
    fn persistent_uselessness_confirms_with_its_evidence() {
        let stream = [
            measured(0, 4, Some(0.2), None),
            measured(2, 4, Some(0.0), None),
            measured(4, 3, Some(-0.25), None),
            measured(6, 5, Some(0.0), None),
        ];
        let evidence = confirms(&stream, &policy()).expect("confirmed");
        assert_eq!(
            evidence,
            [
                (Generation(2), 0.0),
                (Generation(4), -0.25),
                (Generation(6), 0.0)
            ]
        );
    }

    #[test]
    fn a_contributing_measurement_in_the_window_is_never_demoted() {
        let stream = [
            measured(0, 4, Some(0.0), None),
            measured(2, 4, Some(0.0), None),
            measured(4, 4, Some(0.05), None),
        ];
        assert!(
            confirms(&stream, &policy()).is_none(),
            "the latest shows it contributing"
        );
        let short = [
            measured(0, 4, Some(0.0), None),
            measured(2, 4, Some(0.0), None),
        ];
        assert!(confirms(&short, &policy()).is_none(), "not yet persistent");
    }

    #[test]
    fn unused_or_thin_measurements_are_not_evidence() {
        let unused = [
            measured(0, 4, Some(0.0), None),
            measured(2, 0, None, None),
            measured(4, 4, Some(0.0), None),
        ];
        assert!(
            confirms(&unused, &policy()).is_none(),
            "unused is not useless"
        );
        let thin = [
            measured(0, 4, Some(0.0), None),
            measured(2, 1, Some(-0.5), None),
            measured(4, 4, Some(0.0), None),
        ];
        assert!(
            confirms(&thin, &policy()).is_none(),
            "one task is not evidence"
        );
    }

    #[test]
    fn warnings_read_share_use_and_drift() {
        let fell = [
            measured(0, 6, Some(0.1), Some(0.9)),
            measured(2, 6, Some(0.1), Some(0.9)),
            measured(4, 2, Some(0.1), Some(0.7)),
        ];
        let found = warnings(&fell);
        assert!(
            found.contains(&Warning::ShareFell {
                share: 0.2,
                was: 0.6
            }),
            "{found:?}"
        );
        assert!(
            found.iter().any(|w| matches!(w, Warning::Drifted { .. })),
            "{found:?}"
        );
        let unused = [measured(0, 6, Some(0.1), None), measured(2, 0, None, None)];
        assert_eq!(warnings(&unused), [Warning::WentUnused]);
        let steady = [
            measured(0, 5, Some(0.1), Some(0.9)),
            measured(2, 5, Some(0.1), Some(0.88)),
        ];
        assert!(warnings(&steady).is_empty());
        assert!(warnings(&[]).is_empty());
    }
}
