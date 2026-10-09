//! How the loop reads its own generations through the standing instruments:
//! what a generation is asked to withhold, what its results say, and what the
//! audit slice says across the generations before it.
//!
//! Nothing here decides anything. The loop records these readings beside the
//! fitness they qualify and says so when the trend reads as overtuning. One
//! rule reaches a decision, and it is fixed by design rather than chosen by
//! this module: a generation that passed an impossible task fails whole, so it
//! does not graduate. No number measured here reaches training or reward.

use antumbra_core::ports::{Remeasurement, TaskOutcome, TrainOutcome};
use antumbra_core::slice::{Holdout, Slice};
use antumbra_core::{EvaluationRun, Generation, Result, RunId, SubjectKind};
use antumbra_eclipse::instrument::GenerationReport as InstrumentReport;
use antumbra_eclipse::{Outcome, Point, Trend};
use antumbra_store::repo::evaluation;

use crate::GenerationLoop;

/// What the instruments made of one generation.
#[derive(Debug, Clone, Default)]
pub(crate) struct Measurement {
    pub(crate) instruments: Option<InstrumentReport>,
    pub(crate) trend: Option<Trend>,
    /// What graduation was judged on, when the loop re-measured the shadow.
    pub(crate) remeasured: Option<Remeasurement>,
}

impl GenerationLoop<'_> {
    /// What this generation's trainer is asked to withhold. `None` with no
    /// partition configured. The audit slice is measured only on its schedule,
    /// counting from generation 0, which keeps it a trend instrument rather
    /// than one more number read every time.
    pub(crate) fn holdout_for(&self, generation: Generation) -> Option<Holdout> {
        self.cfg.partition.map(|partition| Holdout {
            partition,
            audit: generation.0.is_multiple_of(self.cfg.audit_every.max(1)),
        })
    }

    /// Read this generation through the standing instruments.
    ///
    /// `None` unless three things hold, and each `None` is the true answer
    /// rather than a degraded one. Something was held out: with no partition
    /// every task was learned from, and a gap between two sets of learned tasks
    /// measures nothing. The trainer confirmed it withheld exactly that: one
    /// that ignored the request learned from the held-out tasks, and slicing
    /// its results afterwards would label them without making them held out.
    /// And there are per-task results: the gap is undefined over a single
    /// aggregate number.
    ///
    /// The slice comes from the task id alone, through the partition the
    /// trainer enforced, so it cannot drift between generations and nothing the
    /// loop decides can move a task across the anchor.
    pub(crate) fn instruments(
        &self,
        asked: Option<&Holdout>,
        outcome: &TrainOutcome,
    ) -> Option<InstrumentReport> {
        let asked = asked?;
        if outcome.holdout.as_ref() != Some(asked) {
            eprintln!(
                "instruments: the trainer did not confirm it withheld partition seed {} \
                 (it reported {:?}); this generation is not measured",
                asked.partition.seed,
                outcome.holdout.map(|h| h.partition.seed)
            );
            return None;
        }
        if outcome.per_task.is_empty() {
            return None;
        }
        let outcomes: Vec<Outcome> = outcome
            .per_task
            .iter()
            .map(|t: &TaskOutcome| {
                // The partition never assigns the impossible slice: such a
                // task is authored, so the result says what it is.
                let slice = if t.impossible {
                    Slice::Impossible
                } else {
                    asked.partition.of(&t.task_id)
                };
                Outcome::new(t.task_id.clone(), slice, t.passed, t.size)
            })
            .collect();
        Some(InstrumentReport::of(&outcomes))
    }

    /// Measure a generation and read the trend it extends. The trend is asked
    /// only of a measured generation: an unmeasured one has no audit reading
    /// to add, and its search score alone would read as a climb nobody checked.
    pub(crate) async fn measure(
        &self,
        run_id: &RunId,
        generation: Generation,
        asked: Option<&Holdout>,
        outcome: &TrainOutcome,
    ) -> Result<Measurement> {
        let instruments = self.instruments(asked, outcome);
        let (Some(report), Some(asked)) = (instruments.as_ref(), asked) else {
            return Ok(Measurement {
                instruments,
                trend: None,
                remeasured: None,
            });
        };
        let current = Point {
            generation: generation.0,
            search: outcome.final_fitness,
            audit: report.audit.rate(),
        };
        let trend = self
            .read_trend(run_id, asked.partition.seed, current)
            .await?;
        if trend == Trend::Overtuning {
            eprintln!(
                "instruments: over the last {} measured generations the search score climbed \
                 and the audit slice did not follow -- measured overtuning, the recipe search's \
                 kill criterion",
                self.cfg.watch.window
            );
        }
        Ok(Measurement {
            instruments,
            trend: Some(trend),
            remeasured: None,
        })
    }

    /// The audit-slice trend over this run's measured generations, ending at
    /// `current`. Earlier generations come from the evaluation rows the loop
    /// already persists, so a resumed run reads the same history a continuous
    /// one would have.
    async fn read_trend(&self, run_id: &RunId, seed: u64, current: Point) -> Result<Trend> {
        let mut by_generation = std::collections::BTreeMap::new();
        for run in evaluation::list_for_run(self.store, run_id, SubjectKind::Shadow).await? {
            if let Some(point) = history_point(&run, seed) {
                // Oldest first, so a generation recorded twice keeps its latest.
                by_generation.insert(point.generation, point);
            }
        }
        by_generation.insert(current.generation, current);
        let points: Vec<Point> = by_generation.into_values().collect();
        Ok(self.cfg.watch.read(&points))
    }
}

/// One earlier generation as the trend reads it, or `None` when it cannot be
/// read that way: it carried no instruments (nothing was held out, or its
/// trainer did not confirm the holdout), or it was measured under another
/// partition seed. A reseed repartitions the corpus, so a trend across two
/// seeds would compare two different audit slices.
fn history_point(run: &EvaluationRun, seed: u64) -> Option<Point> {
    let metrics = run.metrics.as_ref()?;
    if metrics.get("partition_seed")?.as_u64()? != seed {
        return None;
    }
    let report: InstrumentReport =
        serde_json::from_value(metrics.get("instruments")?.clone()).ok()?;
    Some(Point {
        generation: run.corpus_task_id.strip_prefix("gen:")?.parse().ok()?,
        search: metrics.get("fitness")?.as_f64()? as f32,
        audit: report.audit.rate(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LoopConfig;
    use antumbra_core::EvalStatus;

    /// The two dials that meet in the trend: the audit slice runs every
    /// `audit_every` generations, and the watch wants `min_audited` of them in
    /// a window. Any `window` consecutive generations contain at least
    /// `window / audit_every` (rounded down) audited ones, so the defaults
    /// must satisfy this or the trend could never be anything but
    /// inconclusive for some windows.
    #[test]
    fn the_default_schedule_leaves_the_default_watch_able_to_answer() {
        let cfg = LoopConfig::default();
        let guaranteed = cfg.watch.window / cfg.audit_every as usize;
        assert!(
            guaranteed >= cfg.watch.min_audited,
            "every {} generations leaves {guaranteed} audited in a window of {}, \
             under the {} the watch asks for",
            cfg.audit_every,
            cfg.watch.window,
            cfg.watch.min_audited
        );
    }

    fn row(task: &str, metrics: serde_json::Value) -> EvaluationRun {
        EvaluationRun {
            run_id: RunId::new("run:h"),
            subject_kind: SubjectKind::Shadow,
            subject_id: "run:h:g0".into(),
            corpus_task_id: task.into(),
            status: EvalStatus::Success,
            metrics: Some(metrics),
            regression_fingerprint: None,
            created_at: chrono::Utc::now(),
        }
    }

    fn measured(audit_passed: u32, audit_measured: u32) -> serde_json::Value {
        serde_json::json!({
            "gaps": [],
            "impossible_passed": [],
            "impossible_measured": 0,
            "audit": { "passed": audit_passed, "measured": audit_measured },
        })
    }

    #[test]
    fn a_persisted_generation_reads_back_as_the_point_it_was() {
        let point = history_point(
            &row(
                "gen:3",
                serde_json::json!({
                    "fitness": 0.5,
                    "partition_seed": 0,
                    "instruments": measured(3, 4),
                }),
            ),
            0,
        );
        assert_eq!(
            point,
            Some(Point {
                generation: 3,
                search: 0.5,
                audit: Some(0.75),
            })
        );
        // An audit slice not due that generation is not a zero pass rate.
        let unaudited = history_point(
            &row(
                "gen:4",
                serde_json::json!({
                    "fitness": 0.5,
                    "partition_seed": 0,
                    "instruments": measured(0, 0),
                }),
            ),
            0,
        );
        assert_eq!(unaudited.and_then(|p| p.audit), None);
    }

    #[test]
    fn a_generation_the_trend_cannot_read_is_left_out_rather_than_guessed() {
        let unmeasured = serde_json::json!({
            "fitness": 0.9,
            "partition_seed": null,
            "instruments": null,
        });
        assert_eq!(history_point(&row("gen:0", unmeasured), 0), None);
        let reseeded = serde_json::json!({
            "fitness": 0.9,
            "partition_seed": 7,
            "instruments": measured(1, 1),
        });
        assert_eq!(history_point(&row("gen:1", reseeded), 0), None);
        // A freeze baseline shares the table and is not a generation.
        let freeze = serde_json::json!({ "fitness": 0.9, "event": "freeze" });
        assert_eq!(history_point(&row("freeze:g1", freeze), 0), None);
    }
}
