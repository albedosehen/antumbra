//! `antumbra gate-sweep` (ADR-0024 D-1, Validation 1): the bar a typed gate
//! must clear is the margin gate at its best threshold, not at the one it
//! ships with.
//!
//! Every task the default partition withholds is routed with the threshold
//! out of the way, so each has the expert the margin gate would pick and the
//! margin it would pick it by. That expert and the base model both answer it
//! under the same seeds. The threshold is then swept offline: at each one the
//! gate routes the tasks whose margin clears it and escalates the rest to the
//! base model, which stands in for the agent above it. That gives a
//! risk-coverage curve, and the point on it with the best accuracy is the bar.

#[cfg(any(feature = "models", test))]
/// One held-out task, as the margin gate sees it and as it came out.
#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    pub task: String,
    /// The expert the gate would route it to with no threshold.
    pub expert: String,
    /// The gate's relative coverage: the top expert's margin over the next.
    pub coverage: f32,
    /// That expert's pass rate on it.
    pub expert_score: f32,
    /// The base model's pass rate on it.
    pub base_score: f32,
}

#[cfg(any(feature = "models", test))]
/// The gate at one threshold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub threshold: f32,
    /// The share of tasks routed to an expert rather than escalated.
    pub routed: f32,
    /// One minus the routed experts' mean pass rate; zero when none routed.
    pub risk: f32,
    /// The mean pass rate over every task, escalated ones at the base's.
    pub accuracy: f32,
}

#[cfg(any(feature = "models", test))]
/// The gate at `threshold`: tasks whose coverage clears it are routed.
pub fn at(outcomes: &[Outcome], threshold: f32) -> Point {
    let n = outcomes.len().max(1) as f32;
    let routed: Vec<&Outcome> = outcomes
        .iter()
        .filter(|o| o.coverage >= threshold)
        .collect();
    let risk = if routed.is_empty() {
        0.0
    } else {
        1.0 - routed.iter().map(|o| o.expert_score).sum::<f32>() / routed.len() as f32
    };
    let accuracy = outcomes
        .iter()
        .map(|o| {
            if o.coverage >= threshold {
                o.expert_score
            } else {
                o.base_score
            }
        })
        .sum::<f32>()
        / n;
    Point {
        threshold,
        routed: routed.len() as f32 / n,
        risk,
        accuracy,
    }
}

#[cfg(any(feature = "models", test))]
/// The whole curve: the gate at every distinct coverage, and past the largest
/// (routing nothing), in threshold order.
pub fn sweep(outcomes: &[Outcome]) -> Vec<Point> {
    let mut thresholds: Vec<f32> = outcomes.iter().map(|o| o.coverage).collect();
    thresholds.sort_by(f32::total_cmp);
    thresholds.dedup();
    thresholds.push(f32::INFINITY);
    thresholds.into_iter().map(|t| at(outcomes, t)).collect()
}

#[cfg(any(feature = "models", test))]
/// The best point: the highest accuracy, and of equals the one that routes
/// the most, since a task an owned expert serves is one the agent need not.
pub fn best(points: &[Point]) -> Option<Point> {
    points.iter().copied().reduce(|a, b| {
        if b.accuracy > a.accuracy || (b.accuracy == a.accuracy && b.routed > a.routed) {
            b
        } else {
            a
        }
    })
}

/// What `gate-sweep` is given.
#[derive(clap::Args, Debug)]
pub struct SweepArgs {
    #[arg(long)]
    pub corpus: String,
    /// Every satisfiable task, not only the ones the default partition
    /// withholds.
    #[arg(long)]
    pub all_tasks: bool,
    /// Seeds each side is scored under.
    #[arg(long, default_value_t = 2)]
    pub seeds: u64,
    /// Completions per task per seed.
    #[arg(long, default_value_t = 4)]
    pub samples: usize,
    /// Write every task's outcome here.
    #[arg(long)]
    pub out: Option<String>,
    /// Sweep the stored learned router instead of the margin gate: each task
    /// goes to the router's top expert, its coverage is the router's
    /// nearest-centroid similarity, and the router ships at its own floor.
    #[arg(long)]
    pub learned: bool,
}

pub async fn run(url: &str, args: SweepArgs) -> anyhow::Result<()> {
    #[cfg(feature = "models")]
    {
        use std::collections::BTreeMap;

        use anyhow::Context;

        use antumbra_core::ports::{EvaluateRequest, Trainer};
        use antumbra_gate::{route as gate_route, GateConfig};
        use antumbra_store::repo::{boundary, lifecycle};
        use antumbra_train::{CandleModelLoader, JsonCorpus, RaftConfig, RaftTrainer};

        let store = crate::connect(url).await?;
        let experts: Vec<antumbra_core::Expert> = lifecycle::routable(&store)
            .await?
            .into_iter()
            .filter(|e| e.owner.is_none())
            .collect();
        if experts.is_empty() {
            anyhow::bail!("the store holds no routable expert to sweep the gate over");
        }
        let boundaries = boundary::list(&store).await?;
        // Masked to the experts swept, so a pick is always one of them.
        let learned = if args.learned {
            let router = antumbra_store::repo::router::load(&store)
                .await?
                .context("no learned router is stored; run gate-train first")?;
            Some(router.masked(|id| experts.iter().any(|e| &e.id == id)))
        } else {
            None
        };
        let tasks: Vec<serde_json::Value> = serde_json::from_str(
            &std::fs::read_to_string(&args.corpus)
                .with_context(|| format!("reading {}", args.corpus))?,
        )?;
        let partition = antumbra_core::slice::Partition::default();
        let embedder = crate::make_embedder()?;
        let open = GateConfig {
            coverage_threshold: f32::MIN,
            ..GateConfig::default()
        };
        let mut routed: Vec<(String, String, f32)> = Vec::new();
        for task in &tasks {
            let (Some(id), Some(prompt)) = (task["id"].as_str(), task["prompt"].as_str()) else {
                continue;
            };
            let withheld = !partition.of(id).selection_may_read();
            if task["impossible"].as_bool().unwrap_or(false) || !(args.all_tasks || withheld) {
                continue;
            }
            let v = embedder.embed(prompt).await?;
            let pick = match &learned {
                Some(router) => router
                    .route(&v)
                    .first()
                    .map(|(expert, _)| (expert.to_string(), router.top_similarity(&v))),
                None => {
                    let decision = gate_route(&v, &experts, &boundaries, 1, &open);
                    decision
                        .chosen
                        .first()
                        .map(|expert| (expert.to_string(), decision.coverage))
                }
            };
            if let Some((expert, coverage)) = pick {
                routed.push((id.to_string(), expert, coverage));
            }
        }
        println!("{} task(s) over {} expert(s)", routed.len(), experts.len());

        let cfg = RaftConfig {
            samples_per_task: args.samples,
            ..RaftConfig::default()
        };
        let base_model = cfg.base_model.clone();
        let loader = CandleModelLoader::new(cfg.clone());
        let corpus = JsonCorpus::from_file(&args.corpus)?;
        let trainer = RaftTrainer::new(
            cfg,
            loader,
            corpus,
            std::sync::Arc::new(antumbra_critic::CommandVerifier),
        );
        let seeds: Vec<u64> = (1..=args.seeds.max(1)).collect();
        let score = |adapter_uri: Option<String>, task_ids: Vec<String>| EvaluateRequest {
            label: "gate-sweep".into(),
            base_model: base_model.clone(),
            adapter_uri,
            task_ids,
            seeds: seeds.clone(),
        };
        let base = trainer
            .evaluate(score(None, routed.iter().map(|r| r.0.clone()).collect()))
            .await?;
        let mut by_expert: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for (task, expert, _) in &routed {
            by_expert
                .entry(expert.clone())
                .or_default()
                .push(task.clone());
        }
        let mut expert_scores = BTreeMap::new();
        for (expert, task_ids) in by_expert {
            let uri = experts
                .iter()
                .find(|e| e.id.as_str() == expert)
                .map(|e| e.artifact_uri.clone());
            let scored = trainer.evaluate(score(uri, task_ids)).await?;
            expert_scores.extend(scored.scores);
        }
        let outcomes: Vec<Outcome> = routed
            .into_iter()
            .filter_map(|(task, expert, coverage)| {
                Some(Outcome {
                    expert_score: *expert_scores.get(&task)?,
                    base_score: *base.scores.get(&task)?,
                    task,
                    expert,
                    coverage,
                })
            })
            .collect();
        let points = sweep(&outcomes);
        let shipped_at = learned
            .as_ref()
            .map_or(GateConfig::default().coverage_threshold, |r| r.floor);
        let shipped = at(&outcomes, shipped_at);
        let show = |label: &str, p: &Point| {
            println!(
                "{label:<10} threshold {:>8.4}  routed {:.2}  risk {:.3}  accuracy {:.3}",
                p.threshold, p.routed, p.risk, p.accuracy
            );
        };
        show("shipped", &shipped);
        if let Some(b) = best(&points) {
            show("best", &b);
        }
        show("none", &at(&outcomes, f32::INFINITY));
        show("all", &at(&outcomes, f32::MIN));
        if let Some(path) = &args.out {
            let rows: Vec<serde_json::Value> = outcomes
                .iter()
                .map(|o| {
                    serde_json::json!({
                        "task": o.task, "expert": o.expert, "coverage": o.coverage,
                        "expert_score": o.expert_score, "base_score": o.base_score,
                    })
                })
                .collect();
            std::fs::write(path, serde_json::to_vec_pretty(&rows)?)?;
            println!("{} outcome(s) -> {path}", rows.len());
        }
        Ok(())
    }
    #[cfg(not(feature = "models"))]
    {
        let SweepArgs {
            corpus,
            all_tasks,
            seeds,
            samples,
            out,
            learned,
        } = args;
        let _ = (url, corpus, all_tasks, seeds, samples, out, learned);
        anyhow::bail!("`gate-sweep` requires building with --features models (candle + a GPU)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn o(coverage: f32, expert_score: f32, base_score: f32) -> Outcome {
        Outcome {
            task: format!("t{coverage}"),
            expert: "e".into(),
            coverage,
            expert_score,
            base_score,
        }
    }

    #[test]
    fn the_sweep_trades_coverage_for_risk() {
        // Confident routes are good, marginal ones worse than the base.
        let outcomes = vec![
            o(0.30, 1.0, 0.2),
            o(0.20, 0.8, 0.3),
            o(0.05, 0.1, 0.6),
            o(0.01, 0.0, 0.5),
        ];
        let points = sweep(&outcomes);
        assert_eq!(points.len(), 5);
        let all = points[0];
        assert_eq!(all.routed, 1.0);
        assert!((all.accuracy - 0.475).abs() < 1e-6);
        let none = *points.last().unwrap();
        assert_eq!((none.routed, none.risk), (0.0, 0.0));
        assert!((none.accuracy - 0.4).abs() < 1e-6);
        // The best routes the two confident tasks and escalates the rest.
        let b = best(&points).unwrap();
        assert_eq!(b.threshold, 0.20);
        assert_eq!(b.routed, 0.5);
        assert!((b.risk - 0.1).abs() < 1e-6);
        assert!((b.accuracy - 0.725).abs() < 1e-6);
    }

    #[test]
    fn of_equally_accurate_points_the_best_routes_more() {
        let outcomes = vec![o(0.5, 0.5, 0.5), o(0.1, 0.5, 0.5)];
        assert_eq!(best(&sweep(&outcomes)).unwrap().routed, 1.0);
        assert!(best(&[]).is_none());
    }
}
