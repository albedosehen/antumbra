//! `antumbra gate-outcomes`: the routing outcomes a population
//! holds without a loop run to record them. Every routable expert answers the
//! live tasks the contribution measurement samples, under the same seeds, and
//! each task's clear winner is recorded as the measurement records it, so
//! `gate-train` learns from them. The base model is not scored, as the
//! measurement scores it only on what the population escalates.

/// What `gate-outcomes` is given.
#[derive(clap::Args, Debug)]
pub struct OutcomeArgs {
    #[arg(long)]
    pub corpus: String,
    /// Seeds each expert is scored under.
    #[arg(long, default_value_t = 2)]
    pub seeds: u64,
    /// Completions per task per seed.
    #[arg(long, default_value_t = 4)]
    pub samples: usize,
    /// At most this many live tasks, sampled as the contribution measurement
    /// samples them: grow runs measure 64.
    #[arg(long, default_value_t = 64)]
    pub max_tasks: usize,
}

pub async fn run(url: &str, args: OutcomeArgs) -> anyhow::Result<()> {
    #[cfg(feature = "models")]
    {
        use std::collections::{BTreeMap, HashMap};

        use antumbra_core::ports::{EvaluateRequest, Trainer};
        use antumbra_core::{ExpertId, Generation, RunId};
        use antumbra_store::repo::lifecycle;
        use antumbra_train::{CandleModelLoader, JsonCorpus, RaftConfig, RaftTrainer};

        let store = crate::connect(url).await?;
        let experts: Vec<antumbra_core::Expert> = lifecycle::routable(&store)
            .await?
            .into_iter()
            .filter(|e| e.owner.is_none())
            .collect();
        if experts.len() < 2 {
            anyhow::bail!("a routing outcome needs two routable experts to choose between");
        }
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
        let tasks = antumbra_loop::sample_live(trainer.live_tasks(None).await?, args.max_tasks);
        let task_ids: Vec<String> = tasks.iter().map(|t| t.id.clone()).collect();
        let seeds: Vec<u64> = (1..=args.seeds.max(1)).collect();
        let mut scores: HashMap<ExpertId, HashMap<String, f32>> = HashMap::new();
        for e in &experts {
            let scored = trainer
                .evaluate(EvaluateRequest {
                    label: "gate-outcomes".into(),
                    base_model: base_model.clone(),
                    adapter_uri: Some(e.artifact_uri.clone()),
                    task_ids: task_ids.clone(),
                    seeds: seeds.clone(),
                })
                .await?;
            let mean = scored.scores.values().sum::<f32>() / scored.scores.len().max(1) as f32;
            println!("{} {mean:.3} over {} task(s)", e.id, scored.scores.len());
            scores.insert(e.id.clone(), scored.scores.into_iter().collect());
        }
        let ids: Vec<ExpertId> = experts.iter().map(|e| e.id.clone()).collect();
        let score = |who: &Option<ExpertId>, task: &str| -> Option<f32> {
            scores.get(who.as_ref()?)?.get(task).copied()
        };
        let recorded = antumbra_loop::record_winners(
            &store,
            &tasks,
            &ids,
            score,
            &RunId::new("run:gate-outcomes"),
            Generation::ZERO,
        )
        .await?;
        let mut by_winner: BTreeMap<String, usize> = BTreeMap::new();
        for o in antumbra_store::repo::contribution::outcomes(&store).await? {
            *by_winner.entry(o.winner.to_string()).or_default() += 1;
        }
        println!(
            "{} of {} live task(s) with a clear winner{}",
            recorded.won,
            tasks.len(),
            if recorded.changed {
                "; the winners changed, run gate-train --with-outcomes to learn them"
            } else {
                ""
            }
        );
        for (winner, n) in by_winner {
            println!("  {winner} won {n}");
        }
        Ok(())
    }
    #[cfg(not(feature = "models"))]
    {
        let OutcomeArgs {
            corpus,
            seeds,
            samples,
            max_tasks,
        } = args;
        let _ = (url, corpus, seeds, samples, max_tasks);
        anyhow::bail!("`gate-outcomes` requires building with --features models (candle + a GPU)")
    }
}
