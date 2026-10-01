//! Which expert clearly won each live task (ADR-0024 D-1). A contribution
//! measurement with the baseline on scores every expert on every live task,
//! so it knows where each task should have gone, not only where the gate sent
//! it. Those wins are kept, and the learned router trains on them beside the
//! capability exemplars: two grow runs ended with the population 0.07 under
//! what its own experts would score routed as well as they could be, while
//! the router learned only from the text of what each expert had solved.
//!
//! Only a clear win counts: the winner beat every other expert scored on the
//! task, and the base model where it was scored, by at least [`MARGIN`]. A
//! task its experts tie on, or the base model wins, teaches the router
//! nothing it should act on.
//!
//! The live tasks now shape routing, so the population's score on them reads
//! high. Routing is judged on the withheld tasks the gate sweep reads.

use chrono::Utc;

use antumbra_core::ports::TaskPrompt;
use antumbra_core::{ExpertId, Generation, Result, RoutingOutcome, RunId};
use antumbra_store::repo::contribution;
use antumbra_store::Store;

/// How far a winner must beat the rest: a quarter of the score range, two of
/// eight samples at the usual two seeds of four.
pub const MARGIN: f32 = 0.25;

/// What recording a measurement's winners did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Recorded {
    /// Tasks with a clear winner.
    pub won: usize,
    /// Whether the winners the gate learns from changed.
    pub changed: bool,
}

/// Find and record the clear winner of each of `tasks` among `experts`,
/// scored by `score` (`None` is the base model). A win replaces what an
/// earlier measurement recorded for its task. A task with no clear winner now
/// loses the one recorded before, since the population that won it has
/// changed since.
pub async fn record_winners(
    store: &Store,
    tasks: &[TaskPrompt],
    experts: &[ExpertId],
    score: impl Fn(&Option<ExpertId>, &str) -> Option<f32>,
    run_id: &RunId,
    generation: Generation,
) -> Result<Recorded> {
    let wins = clear_winners(tasks, experts, score, MARGIN);
    let before = contribution::outcomes(store).await?;
    let now = Utc::now();
    let mut changed = false;
    for task in tasks {
        let was = before.iter().find(|o| o.task == task.id).map(|o| &o.winner);
        let won = wins.iter().find(|w| w.task.id == task.id);
        changed |= was != won.map(|w| &w.winner);
        match won {
            Some(won) => {
                contribution::upsert_outcome(
                    store,
                    &RoutingOutcome {
                        task: task.id.clone(),
                        prompt: task.prompt.clone(),
                        winner: won.winner.clone(),
                        score: won.score,
                        runner_up: won.runner_up,
                        run_id: run_id.clone(),
                        generation,
                        at: now,
                    },
                )
                .await?
            }
            None if was.is_some() => contribution::clear_outcome(store, &task.id).await?,
            None => {}
        }
    }
    Ok(Recorded {
        won: wins.len(),
        changed,
    })
}

/// A task's clear winner.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Win<'a> {
    pub task: &'a TaskPrompt,
    pub winner: ExpertId,
    pub score: f32,
    pub runner_up: f32,
}

/// The clear winner of each of `tasks` among `experts`, scored by `score`
/// (`None` is the base model). A task with fewer than two experts scored on
/// it is no routing choice and has none.
pub(crate) fn clear_winners<'a>(
    tasks: &'a [TaskPrompt],
    experts: &[ExpertId],
    score: impl Fn(&Option<ExpertId>, &str) -> Option<f32>,
    margin: f32,
) -> Vec<Win<'a>> {
    let mut wins = Vec::new();
    for task in tasks {
        let mut scored: Vec<(&ExpertId, f32)> = experts
            .iter()
            .filter_map(|e| Some((e, score(&Some(e.clone()), &task.id)?)))
            .collect();
        if scored.len() < 2 {
            continue;
        }
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        let (winner, best) = scored[0];
        let rest = scored[1..].iter().map(|s| s.1).fold(f32::MIN, f32::max);
        let base = score(&None, &task.id).unwrap_or(f32::MIN);
        let runner_up = rest.max(base);
        if best - runner_up >= margin {
            wins.push(Win {
                task,
                winner: winner.clone(),
                score: best,
                runner_up,
            });
        }
    }
    wins
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn task(id: &str) -> TaskPrompt {
        TaskPrompt {
            id: id.into(),
            prompt: format!("do {id}"),
            region: "r".into(),
        }
    }

    #[test]
    fn only_a_clear_win_over_every_other_expert_and_the_base_counts() {
        let (a, b) = (ExpertId::new("a"), ExpertId::new("b"));
        let tasks = vec![
            task("a-wins"),
            task("tie"),
            task("base-wins"),
            task("one-scored"),
        ];
        let table: HashMap<(Option<ExpertId>, &str), f32> = [
            ((Some(a.clone()), "a-wins"), 1.0),
            ((Some(b.clone()), "a-wins"), 0.5),
            ((Some(a.clone()), "tie"), 0.75),
            ((Some(b.clone()), "tie"), 0.625),
            ((Some(a.clone()), "base-wins"), 0.75),
            ((Some(b.clone()), "base-wins"), 0.25),
            ((None, "base-wins"), 0.875),
            ((Some(b.clone()), "one-scored"), 1.0),
        ]
        .into_iter()
        .collect();
        let score = |who: &Option<ExpertId>, t: &str| table.get(&(who.clone(), t)).copied();
        let wins = clear_winners(&tasks, &[a.clone(), b], score, MARGIN);
        assert_eq!(wins.len(), 1);
        assert_eq!(wins[0].task.id, "a-wins");
        assert_eq!(wins[0].winner, a);
        assert_eq!((wins[0].score, wins[0].runner_up), (1.0, 0.5));
    }
}
