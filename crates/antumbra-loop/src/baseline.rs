//! The population against its single best expert (ADR-0022 S-5): the rolling
//! comparison the record asks to be reported whether or not it flatters the
//! architecture. Routing across many experts has to beat sending everything to
//! the best one; if it stops doing so, the right answer is fewer and broader
//! experts.
//!
//! When it does not, the reason is one of two, and the scores already taken
//! tell them apart. Every expert is scored on every live task, so each task's
//! best is known: the gate's own choice or any expert alone. Their mean is
//! what these experts would score routed as well as they could be. Well above
//! the population, the gate is choosing badly. Close to it, the experts are
//! too alike for any routing to help.
//!
//! Taken as the highest of each task's noisy scores, that oracle reads high
//! even when the experts are identical: on eight samples a score, by 0.05 to
//! 0.08 (ADR-0024 D-1). So where the scores come in two halves of the seeds,
//! the oracle is cross-fitted ([`cross_fitted_oracle`]): each task's best is
//! chosen on one half and scored on the other.

use antumbra_core::ExpertId;

/// The comparison over the tasks every side was scored on.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Comparison {
    /// How many tasks every side was scored on.
    pub tasks: u32,
    /// The routed population's mean, the base model standing in where the
    /// gate escalates.
    pub population: f32,
    /// The best single expert and its mean alone.
    pub best: Option<(ExpertId, f32)>,
    /// The mean of each task's best score: the gate's own choice or any
    /// expert alone.
    pub oracle: f32,
}

/// Compare the routed population with each expert alone, over the tasks every
/// side was scored on. `None` when there is no such task.
pub(crate) fn compare(
    routed: &[(String, Option<ExpertId>)],
    experts: &[ExpertId],
    score: impl Fn(&Option<ExpertId>, &str) -> Option<f32>,
) -> Option<Comparison> {
    // A task counts only where the population and every expert alone were
    // scored, so every mean is over the same tasks.
    let rows: Vec<(f32, Vec<f32>)> = routed
        .iter()
        .filter_map(|(task, to)| {
            let population = score(to, task)?;
            let alone: Option<Vec<f32>> = experts
                .iter()
                .map(|e| score(&Some(e.clone()), task))
                .collect();
            Some((population, alone?))
        })
        .collect();
    if rows.is_empty() {
        return None;
    }
    let n = rows.len() as f32;
    let population = rows.iter().map(|r| r.0).sum::<f32>() / n;
    let oracle = rows
        .iter()
        .map(|r| r.1.iter().copied().fold(r.0, f32::max))
        .sum::<f32>()
        / n;
    let best = experts
        .iter()
        .enumerate()
        .map(|(i, e)| (e.clone(), rows.iter().map(|r| r.1[i]).sum::<f32>() / n))
        .max_by(|a, b| a.1.total_cmp(&b.1));
    Some(Comparison {
        tasks: u32::try_from(rows.len()).unwrap_or(u32::MAX),
        population,
        best,
        oracle,
    })
}

/// The oracle without its own noise. Each task's candidates are the gate's
/// own choice and every expert alone, each scored on two halves of the seeds
/// (`halves`). The best is chosen on one half and scored on the other, both
/// ways round, and the two are averaged. A tie keeps the gate's own choice.
/// The highest of noisy scores sits above the best expert's true score even
/// when the experts are identical; a choice scored on draws it was not chosen
/// on does not. `None` when no task has both halves for every candidate.
pub(crate) fn cross_fitted_oracle(
    routed: &[(String, Option<ExpertId>)],
    experts: &[ExpertId],
    halves: impl Fn(&Option<ExpertId>, &str) -> Option<(f32, f32)>,
) -> Option<f32> {
    let mut total = 0.0;
    let mut tasks = 0u32;
    for (task, to) in routed {
        let candidates: Option<Vec<(f32, f32)>> = std::iter::once(halves(to, task))
            .chain(experts.iter().map(|e| halves(&Some(e.clone()), task)))
            .collect();
        let Some(candidates) = candidates else {
            continue;
        };
        // The candidate best on `choose`, scored on `score`.
        let pick = |choose: fn(&(f32, f32)) -> f32, score: fn(&(f32, f32)) -> f32| {
            let mut best = candidates[0];
            for c in &candidates[1..] {
                if choose(c) > choose(&best) {
                    best = *c;
                }
            }
            score(&best)
        };
        total += 0.5 * (pick(|c| c.0, |c| c.1) + pick(|c| c.1, |c| c.0));
        tasks += 1;
    }
    (tasks > 0).then(|| total / tasks as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn id(s: &str) -> ExpertId {
        ExpertId::new(s)
    }

    /// Two specialists, each perfect on its own tasks and poor elsewhere: the
    /// routed population beats either alone.
    #[test]
    fn routing_to_specialists_beats_the_best_one_alone() {
        let (a, b) = (id("a"), id("b"));
        let routed = vec![
            ("a1".to_string(), Some(a.clone())),
            ("a2".to_string(), Some(a.clone())),
            ("b1".to_string(), Some(b.clone())),
            ("b2".to_string(), Some(b.clone())),
        ];
        let table: HashMap<(Option<ExpertId>, &str), f32> = [
            ((Some(a.clone()), "a1"), 1.0),
            ((Some(a.clone()), "a2"), 1.0),
            ((Some(a.clone()), "b1"), 0.0),
            ((Some(a.clone()), "b2"), 0.5),
            ((Some(b.clone()), "a1"), 0.0),
            ((Some(b.clone()), "a2"), 0.0),
            ((Some(b.clone()), "b1"), 1.0),
            ((Some(b.clone()), "b2"), 1.0),
        ]
        .into_iter()
        .collect();
        let score = |who: &Option<ExpertId>, task: &str| table.get(&(who.clone(), task)).copied();
        let c = compare(&routed, &[a.clone(), b], score).expect("scored");
        assert_eq!((c.tasks, c.population), (4, 1.0));
        assert_eq!(c.best, Some((a, 0.625)));
        // Routed perfectly already: nothing left for better routing.
        assert_eq!(c.oracle, 1.0);
    }

    /// A gate that sends each task to the wrong one of two specialists
    /// leaves the population below either alone, and the oracle shows what
    /// routing them well would reach.
    #[test]
    fn the_oracle_takes_each_tasks_best() {
        let (a, b) = (id("a"), id("b"));
        let routed = vec![
            ("a1".to_string(), Some(b.clone())),
            ("b1".to_string(), Some(a.clone())),
        ];
        let score = |who: &Option<ExpertId>, task: &str| match (who.as_ref(), task) {
            (Some(e), "a1") if e.as_str() == "a" => Some(1.0),
            (Some(e), "b1") if e.as_str() == "b" => Some(0.5),
            _ => Some(0.0),
        };
        let c = compare(&routed, &[a, b], score).expect("scored");
        assert_eq!(c.population, 0.0);
        assert_eq!(c.oracle, 0.75);
    }

    /// An escalated task is scored on the base model, and a task some side
    /// could not score is left out of every mean.
    #[test]
    fn escalations_use_the_base_and_unscored_tasks_drop_out() {
        let a = id("a");
        let routed = vec![
            ("t1".to_string(), Some(a.clone())),
            ("t2".to_string(), None),
            ("t3".to_string(), Some(a.clone())),
        ];
        let score = |who: &Option<ExpertId>, task: &str| match (who, task) {
            (None, "t2") => Some(0.25),
            (Some(_), "t1") => Some(1.0),
            (Some(_), "t2") => Some(0.75),
            _ => None,
        };
        let c = compare(&routed, std::slice::from_ref(&a), score).expect("scored");
        assert_eq!(c.tasks, 2, "t3 was not scored");
        assert_eq!(c.population, 0.625);
        assert_eq!(c.best, Some((a, 0.875)));
        // t2 goes to the expert rather than the base it escalated to.
        assert_eq!(c.oracle, 0.875);
        assert!(compare(&[], &[id("a")], |_, _| Some(1.0)).is_none());
    }

    /// Two identical experts whose halves disagree by noise: the highest of
    /// their scores reads above either, the cross-fitted oracle does not.
    #[test]
    fn noise_alone_is_not_headroom_once_cross_fitted() {
        let (a, b) = (id("a"), id("b"));
        let routed = vec![("t".to_string(), Some(a.clone()))];
        let halves = |who: &Option<ExpertId>, _: &str| match who.as_ref().map(ExpertId::as_str) {
            Some("a") => Some((1.0, 0.0)),
            Some("b") => Some((0.0, 1.0)),
            _ => None,
        };
        // Chosen on the first half, a scores 0 on the second; chosen on the
        // second, b scores 0 on the first.
        assert_eq!(cross_fitted_oracle(&routed, &[a, b], halves), Some(0.0));
    }

    /// An expert better on both halves keeps its headroom.
    #[test]
    fn a_consistently_better_expert_keeps_its_headroom() {
        let (a, b) = (id("a"), id("b"));
        let routed = vec![("t".to_string(), Some(a.clone()))];
        let halves = |who: &Option<ExpertId>, _: &str| match who.as_ref().map(ExpertId::as_str) {
            Some("a") => Some((0.5, 0.5)),
            Some("b") => Some((1.0, 1.0)),
            _ => None,
        };
        assert_eq!(
            cross_fitted_oracle(&routed, &[a.clone(), b], halves),
            Some(1.0)
        );
        assert_eq!(cross_fitted_oracle(&routed, &[a], |_, _| None), None);
    }
}
