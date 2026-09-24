//! `antumbra critic` (ADR-0022 S-2): train a critic on the verifier's verdicts
//! and read it against them, sliced.

use clap::Subcommand;

#[derive(Subcommand)]
pub enum CriticAction {
    /// Train a critic adapter on the verifier's verdicts over completions
    /// (`eval --completions` writes them), holding out a slice of the tasks,
    /// and read it on the held-out ones. Needs --features models and a GPU.
    Train {
        /// The corpus the completions answer: their prompts and skills.
        #[arg(long)]
        corpus: String,
        /// A JSON array of `{task, completion, passed}`. May be given more than
        /// once.
        #[arg(long)]
        completions: Vec<String>,
        /// Where to save the critic's adapter.
        #[arg(long)]
        out: String,
        /// Passes over the balanced examples.
        #[arg(long, default_value_t = 2)]
        rounds: usize,
    },
    /// Read a critic adapter against the verifier's verdicts, per skill.
    Measure {
        #[arg(long)]
        adapter: String,
        #[arg(long)]
        corpus: String,
        #[arg(long)]
        completions: Vec<String>,
        /// A twin critic, trained on another seed or slice: report how far
        /// the two agree, by rank, on the same completions.
        #[arg(long)]
        twin: Option<String>,
    },
}

#[cfg(any(feature = "models", test))]
/// One draw with its verdict and its task's prompt and skill.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub task: String,
    pub prompt: String,
    pub completion: String,
    pub passed: bool,
    pub skill: String,
}

#[cfg(any(feature = "models", test))]
/// The labeled completions: each draw of `{task, completion, passed}` with
/// its task's prompt and skill from the corpus. Draws of tasks the corpus
/// does not hold are dropped.
pub fn labeled(corpus: &[serde_json::Value], draws: &[serde_json::Value]) -> Vec<Row> {
    let mut out = Vec::new();
    for d in draws {
        let (Some(task), Some(completion), Some(passed)) = (
            d["task"].as_str(),
            d["completion"].as_str(),
            d["passed"].as_bool(),
        ) else {
            continue;
        };
        let Some(t) = corpus.iter().find(|t| t["id"].as_str() == Some(task)) else {
            continue;
        };
        let (Some(prompt), skill) = (t["prompt"].as_str(), t["skill"].as_str().unwrap_or("all"))
        else {
            continue;
        };
        out.push(Row {
            task: task.to_string(),
            prompt: prompt.to_string(),
            completion: completion.to_string(),
            passed,
            skill: skill.to_string(),
        });
    }
    out
}

/// Which half of the held-out tasks fits the recalibration; the other half
/// is read through it. A second partition, so the halves are stable.
#[cfg(feature = "models")]
fn fits_calibration(task: &str) -> bool {
    antumbra_core::slice::Partition::new(0.5, 0.0, 1)
        .map(|p| p.of(task).selection_may_read())
        .unwrap_or(true)
}

#[cfg(any(feature = "models", test))]
/// Whether a task is held out from the critic's training: the default
/// partition's withheld slices, so the split is the one every run uses.
pub fn held_out(task: &str) -> bool {
    !antumbra_core::slice::Partition::default()
        .of(task)
        .selection_may_read()
}

#[cfg(feature = "models")]
fn read_all(paths: &[String]) -> anyhow::Result<Vec<serde_json::Value>> {
    let mut out = Vec::new();
    for path in paths {
        let text =
            std::fs::read_to_string(path).map_err(|e| anyhow::anyhow!("reading {path}: {e}"))?;
        let items: Vec<serde_json::Value> = serde_json::from_str(&text)?;
        out.extend(items);
    }
    Ok(out)
}

pub async fn run(action: CriticAction) -> anyhow::Result<()> {
    #[cfg(feature = "models")]
    {
        use antumbra_train::critic::{judge, measure_critic, recalibrate, train_critic, Labeled};
        use antumbra_train::{CandleModelLoader, CausalLm, ModelLoader, RaftConfig};

        let to_labeled = |rows: Vec<Row>| {
            rows.into_iter()
                .map(|r| Labeled {
                    prompt: r.prompt,
                    completion: r.completion,
                    passed: r.passed,
                    slice: r.skill,
                })
                .collect::<Vec<_>>()
        };
        let report = |reading: &antumbra_train::critic::CriticReading| {
            let corr = reading
                .correlation
                .map_or("none".to_string(), |c| format!("{c:.3}"));
            println!("critic on {} completion(s): correlation {corr}", reading.n);
            for s in reading.overall.iter().chain(reading.slices.iter()) {
                println!(
                    "  {:<12} n {:>5}  ece {:.3}  agreement {:.3}",
                    s.slice, s.n, s.ece, s.agreement
                );
            }
        };
        match action {
            CriticAction::Train {
                corpus,
                completions,
                out,
                rounds,
            } => {
                let tasks = read_all(std::slice::from_ref(&corpus))?;
                let rows = labeled(&tasks, &read_all(&completions)?);
                let (held_rows, learn): (Vec<_>, Vec<_>) =
                    rows.into_iter().partition(|r| held_out(&r.task));
                let (held, learn) = (to_labeled(held_rows.clone()), to_labeled(learn));
                println!(
                    "critic: learning from {} completion(s), holding out {}",
                    learn.len(),
                    held.len()
                );
                let cfg = RaftConfig::default();
                let loader = CandleModelLoader::new(cfg.clone());
                let mut model = ModelLoader::load(&loader, &cfg.base_model, None).await?;
                let losses = train_critic(&mut model, &learn, rounds).await?;
                println!("losses: {losses:?}");
                model.save_adapter(&out)?;
                println!("critic adapter -> {out}");
                println!("held out, raw:");
                report(&measure_critic(&mut model, &held).await?);
                // Recalibrate on half the held-out tasks, read the other half
                // through it: the per-generation step, once.
                let (fit, read): (Vec<_>, Vec<_>) = held_rows
                    .into_iter()
                    .partition(|r| fits_calibration(&r.task));
                let fitted = measure_critic(&mut model, &to_labeled(fit)).await?;
                let evaluated = measure_critic(&mut model, &to_labeled(read)).await?;
                println!("the evaluation half, raw:");
                report(&evaluated);
                if let Some(fixed) = recalibrate(&fitted, &evaluated) {
                    println!("the evaluation half, recalibrated on the other:");
                    report(&fixed);
                }
            }
            CriticAction::Measure {
                adapter,
                corpus,
                completions,
                twin,
            } => {
                let tasks = read_all(std::slice::from_ref(&corpus))?;
                let rows = to_labeled(labeled(&tasks, &read_all(&completions)?));
                let cfg = RaftConfig::default();
                let loader = CandleModelLoader::new(cfg.clone());
                let mut model = ModelLoader::load(&loader, &cfg.base_model, Some(&adapter)).await?;
                let reading = measure_critic(&mut model, &rows).await?;
                report(&reading);
                if let Some(twin) = twin {
                    drop(model);
                    let mut other =
                        ModelLoader::load(&loader, &cfg.base_model, Some(&twin)).await?;
                    let mut theirs = Vec::with_capacity(rows.len());
                    for r in &rows {
                        theirs.push(judge(&mut other, &r.prompt, &r.completion).await?);
                    }
                    let ours: Vec<f32> = reading.scored.iter().map(|s| s.predicted).collect();
                    let agree = antumbra_core::critic::agreement(&ours, &theirs)
                        .map_or("none".to_string(), |a| format!("{a:.3}"));
                    println!("agreement with the twin {twin}: {agree}");
                }
            }
        }
        Ok(())
    }
    #[cfg(not(feature = "models"))]
    {
        let _ = action;
        anyhow::bail!("`critic` requires building with --features models (candle + a GPU)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draws_are_labeled_with_their_tasks_prompt_and_skill() {
        let corpus: Vec<serde_json::Value> = serde_json::from_value(serde_json::json!([
            { "id": "s/a", "skill": "strings", "prompt": "do a" },
            { "id": "n/b", "prompt": "do b" }
        ]))
        .unwrap();
        let draws: Vec<serde_json::Value> = serde_json::from_value(serde_json::json!([
            { "task": "s/a", "completion": "x", "passed": true },
            { "task": "n/b", "completion": "y", "passed": false },
            { "task": "gone", "completion": "z", "passed": true },
            { "task": "s/a", "completion": "w" }
        ]))
        .unwrap();
        let rows = labeled(&corpus, &draws);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].skill, "strings");
        assert!(rows[0].passed);
        assert_eq!(rows[0].prompt, "do a");
        assert_eq!(rows[1].skill, "all");
        assert_eq!(rows[1].completion, "y");
    }

    #[test]
    fn the_held_out_split_is_the_default_partitions() {
        let tasks: Vec<String> = (0..200).map(|i| format!("task:{i}")).collect();
        let held = tasks.iter().filter(|t| held_out(t)).count();
        assert!(held > 20 && held < 100, "{held}");
    }
}
