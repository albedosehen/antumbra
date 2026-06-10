//! The two heaviest growth handlers kept out of `main.rs`: `populate` (grow a
//! population until it covers a corpus) and `memory_import` (bootstrap
//! from an existing memory export). Both reuse `main.rs`'s
//! `connect` / `make_embedder` / `refresh_router` helpers via `crate::`, so no
//! infrastructure is duplicated (the same pattern `ops.rs` uses).

#[cfg(feature = "models")]
use chrono::Utc;

#[cfg(feature = "models")]
use antumbra_core::{Expert, ExpertId, Generation, RunId};
#[cfg(feature = "models")]
use antumbra_gate::{route as gate_route, GateConfig};
#[cfg(feature = "models")]
use antumbra_loop::{GenerationLoop, LoopConfig};
#[cfg(feature = "models")]
use antumbra_store::repo::{boundary, expert};
#[cfg(feature = "models")]
use antumbra_store::EMBED_DIM;
#[cfg(feature = "models")]
use antumbra_train::{CandleModelLoader, CaptureTrainer, JsonCorpus, RaftConfig};

#[cfg(feature = "models")]
use crate::refresh_router;

/// Parameters for [`populate`]; mirrors the clap variant so `main.rs`'s arm
/// stays a one-line dispatch.
#[cfg_attr(not(feature = "models"), allow(dead_code))]
pub struct PopulateArgs {
    pub corpus: String,
    pub run: String,
    pub target_coverage: f32,
    pub max_experts: usize,
    pub samples: usize,
    pub rounds: usize,
    pub max_new_tokens: usize,
}

/// Grow a population until it covers `corpus` (or the expert budget is spent):
/// route each task, treat escalations and served-and-failed tasks as gaps,
/// cluster the gaps by skill, and grow one frozen specialist per gap skill.
pub async fn populate(url: &str, args: PopulateArgs) -> anyhow::Result<()> {
    let PopulateArgs {
        corpus,
        run,
        target_coverage,
        max_experts,
        samples,
        rounds,
        max_new_tokens,
    } = args;
    #[cfg(feature = "models")]
    {
        use antumbra_train::{eval_pass_rate, raft_train, Corpus, CorpusTask, ModelLoader};
        let store = crate::connect(url).await?;
        let embedder = crate::make_embedder()?;
        let cfg = RaftConfig {
            samples_per_task: samples,
            rounds,
            max_new_tokens,
            ..RaftConfig::default()
        };
        let loader = CandleModelLoader::new(cfg.clone());
        let verifier = antumbra_critic::CommandVerifier;
        let tasks = JsonCorpus::from_file(&corpus)?.tasks(&[]);
        let radius = GateConfig::default().inhibition_radius;
        let mut grown = 0usize;

        // One extra round to confirm coverage after the last grow.
        for round in 0..(max_experts + 1) {
            let experts = expert::list(&store).await?;
            let boundaries = boundary::list(&store).await?;
            let router = antumbra_store::repo::router::load(&store).await?;

            // Route each task to an expert (or none, if the gate
            // escalates). Coverage is *serving* coverage: an expert only
            // covers a task if it actually serves a verified-correct
            // answer -- a route-claim it cannot fulfil is still a gap.
            let mut routed: Vec<(CorpusTask, Option<ExpertId>)> = Vec::new();
            for task in &tasks {
                let v = embedder.embed(&task.prompt).await?;
                let inhib = boundaries
                    .iter()
                    .map(|b| b.inhibition_for(&v, radius))
                    .fold(0.0f32, f32::max);
                let pick = match &router {
                    Some(r) if r.covers(&v) && inhib <= 0.5 => {
                        r.route(&v).first().map(|(id, _)| id.clone())
                    }
                    Some(_) => None,
                    None => {
                        let gc = GateConfig {
                            coverage_threshold: 0.08,
                            ..GateConfig::default()
                        };
                        let d = gate_route(&v, &experts, &boundaries, 1, &gc);
                        if d.escalate {
                            None
                        } else {
                            d.chosen.first().cloned()
                        }
                    }
                };
                routed.push((task.clone(), pick));
            }
            // Escalated tasks are gaps; routed tasks are gaps only if the
            // routed expert fails to serve them.
            let mut gaps: Vec<CorpusTask> = routed
                .iter()
                .filter(|(_, id)| id.is_none())
                .map(|(t, _)| t.clone())
                .collect();
            for e in &experts {
                let etasks: Vec<CorpusTask> = routed
                    .iter()
                    .filter(|(_, id)| id.as_ref() == Some(&e.id))
                    .map(|(t, _)| t.clone())
                    .collect();
                if etasks.is_empty() {
                    continue;
                }
                let mut m =
                    ModelLoader::load(&loader, &e.base_model, Some(&e.artifact_uri)).await?;
                let ev = eval_pass_rate(
                    &mut m,
                    &verifier,
                    &etasks,
                    &RunId::new("populate:cov"),
                    samples,
                )
                .await?;
                for (t, r) in etasks.iter().zip(&ev.per_task) {
                    if r.rate() < 0.5 {
                        gaps.push(t.clone());
                    }
                }
            }
            let coverage = 1.0 - gaps.len() as f32 / tasks.len().max(1) as f32;
            println!(
                "round {round}: coverage {coverage:.2} ({} experts, {} gaps served-and-failed/uncovered)",
                experts.len(),
                gaps.len()
            );
            if coverage >= target_coverage || gaps.is_empty() || grown >= max_experts {
                let why = if grown >= max_experts {
                    "expert budget reached"
                } else {
                    "covers the corpus"
                };
                println!("population {why} (coverage {coverage:.2})");
                break;
            }

            // Cluster the gap tasks by *skill* and grow a dedicated
            // specialist for each: a narrow frozen expert per skill, not
            // one generalist over all gaps (the umbra ideal, a population of small frozen experts).
            let mut groups: Vec<(String, Vec<CorpusTask>)> = Vec::new();
            for t in &gaps {
                let skill = t.skill();
                match groups.iter_mut().find(|(s, _)| *s == skill) {
                    Some((_, v)) => v.push(t.clone()),
                    None => groups.push((skill, vec![t.clone()])),
                }
            }
            for (skill, gtasks) in groups {
                if grown >= max_experts {
                    break;
                }
                let name = format!("{run}-{skill}");
                let mut model = ModelLoader::load(&loader, &cfg.base_model, None).await?;
                let out = raft_train(
                    &mut model,
                    &verifier,
                    &gtasks,
                    &RunId::new(name.clone()),
                    &cfg,
                )
                .await?;
                let solved = if out.capability_exemplars.is_empty() {
                    gtasks.iter().map(|t| t.prompt.clone()).collect()
                } else {
                    out.capability_exemplars.clone()
                };
                let mut acc = vec![0.0f32; EMBED_DIM];
                for text in &solved {
                    for (a, b) in acc.iter_mut().zip(embedder.embed(text).await?) {
                        *a += b;
                    }
                }
                let nproto = solved.len().max(1) as f32;
                let now = Utc::now();
                let expert = Expert {
                    id: ExpertId::new(format!("expert:{name}")),
                    name: name.clone(),
                    base_model: cfg.base_model.clone(),
                    artifact_uri: out.adapter_uri,
                    capability_card: serde_json::json!({ "exemplars": solved }),
                    capability_vec: Some(acc.iter().map(|x| x / nproto).collect()),
                    fitness: out.final_fitness,
                    frozen_at: Some(now),
                    generation: Generation::ZERO,
                    owner: None,
                    compartment: None,
                    created_at: now,
                };
                expert::delete(&store, &expert.id).await?; // supersede on re-run
                expert::insert(&store, &expert).await?;
                grown += 1;
                println!(
                    "  grew specialist {name} for skill '{skill}' on {} task(s) (fitness {:.2})",
                    gtasks.len(),
                    out.final_fitness
                );
            }
            if let Ok(Some(r)) = refresh_router(&store, embedder.as_ref(), 400).await {
                println!("  router refreshed over {} experts", r.experts.len());
            }
            let _ = round;
        }
        println!("population: {} experts", expert::list(&store).await?.len());
        Ok(())
    }
    #[cfg(not(feature = "models"))]
    {
        let _ = (
            url,
            &corpus,
            &run,
            target_coverage,
            max_experts,
            samples,
            rounds,
            max_new_tokens,
        );
        anyhow::bail!("`populate` requires building with --features models (candle + a GPU)")
    }
}

/// Parameters for [`memory_import`]; mirrors the clap variant.
#[cfg_attr(not(feature = "models"), allow(dead_code))]
pub struct MemoryImportArgs {
    pub source: String,
    pub out: String,
    pub capture_threshold: f32,
    pub train: bool,
    pub run: String,
    pub rounds: usize,
    pub samples: usize,
    pub max_new_tokens: usize,
    pub lr: f64,
}

/// Bootstrap from an existing memory export: adapt normalized
/// memories into capture tasks (reinforced ones trusted on import, weak ones
/// kept as RAFT seeds), write the converted corpus, and with `train` internalize
/// the captures through the capture loop.
pub async fn memory_import(url: &str, args: MemoryImportArgs) -> anyhow::Result<()> {
    let MemoryImportArgs {
        source,
        out,
        capture_threshold,
        train,
        run,
        rounds,
        samples,
        max_new_tokens,
        lr,
    } = args;
    #[cfg(feature = "models")]
    {
        use antumbra_train::memory::{
            import as import_memories, parse_export, ImportPolicy, Intake,
        };
        use antumbra_train::CorpusTask;

        let bytes = std::fs::read(&source)?;
        let records = parse_export(&bytes)?;
        let policy = ImportPolicy { capture_threshold };
        let imported = import_memories(&records, &policy);

        // Tier and cluster: captures are trusted on import, seeds wait for
        // RAFT to confirm them by experience.
        let captures: Vec<CorpusTask> = imported
            .iter()
            .filter(|i| i.intake == Intake::Capture)
            .map(|i| i.task.clone())
            .collect();
        let seeds = imported.len() - captures.len();
        let mut skills: Vec<(String, usize)> = Vec::new();
        for i in &imported {
            let s = i.task.skill();
            match skills.iter_mut().find(|(k, _)| *k == s) {
                Some((_, n)) => *n += 1,
                None => skills.push((s, 1)),
            }
        }
        println!(
            "imported {} memories: {} capture(s), {} seed(s) across {} skill(s)",
            imported.len(),
            captures.len(),
            seeds,
            skills.len()
        );
        for (skill, n) in &skills {
            println!("  skill '{skill}': {n} task(s)");
        }

        // Persist the whole conversion (captures carry a completion, seeds
        // do not) so `teach`/`populate`/`evolve` can consume it.
        let arr: Vec<serde_json::Value> = imported
            .iter()
            .map(|i| {
                let t = &i.task;
                let mut o = serde_json::Map::new();
                o.insert("id".into(), serde_json::json!(t.id));
                o.insert("prompt".into(), serde_json::json!(t.prompt));
                o.insert("verify".into(), t.verify.clone());
                if let Some(c) = &t.completion {
                    o.insert("completion".into(), serde_json::json!(c));
                }
                o.insert("skill".into(), serde_json::json!(t.skill()));
                serde_json::Value::Object(o)
            })
            .collect();
        std::fs::write(&out, serde_json::to_vec_pretty(&arr)?)?;
        println!("wrote capture corpus -> {out}");

        if !train {
            println!("run `antumbra teach --corpus {out}` to internalize the captures");
        } else if captures.is_empty() {
            println!("nothing to train: no memory met the capture threshold {capture_threshold}");
        } else {
            let store = crate::connect(url).await?;
            let cfg = RaftConfig {
                samples_per_task: samples,
                rounds,
                max_new_tokens,
                learning_rate: lr,
                ..RaftConfig::default()
            };
            let corpus = JsonCorpus::from_tasks(captures.clone());
            let verifier = std::sync::Arc::new(antumbra_critic::CommandVerifier);
            let loader = CandleModelLoader::new(cfg.clone());
            let trainer = CaptureTrainer::new(cfg, loader, corpus, verifier);
            let embedder = crate::make_embedder()?;
            let loop_cfg = LoopConfig {
                graduate_threshold: 0.3,
                base_model: "Qwen/Qwen2.5-Coder-1.5B".into(),
            };
            let lp = GenerationLoop::new(&store, &trainer, embedder.as_ref(), loop_cfg);
            let reports = lp.run_until(&RunId::new(run), 1).await?;
            for r in &reports {
                println!(
                    "gen {:<3} expert {:<16} internalized={:.2} graduated={}",
                    r.generation.0, r.shadow, r.fitness, r.graduated
                );
            }
            let experts = expert::list(&store).await?;
            println!("population: {} experts", experts.len());
            if let Ok(Some(r)) = refresh_router(&store, embedder.as_ref(), 400).await {
                println!("router refreshed over {} experts", r.experts.len());
            }
        }
        Ok(())
    }
    #[cfg(not(feature = "models"))]
    {
        let _ = (
            url,
            &source,
            &out,
            capture_threshold,
            train,
            &run,
            rounds,
            samples,
            max_new_tokens,
            lr,
        );
        anyhow::bail!(
            "`memory-import` requires building with --features models (it adapts a memory \
             store into the capture corpus and, with --train, internalizes it)"
        )
    }
}

/// Parameters for [`evolve`]; mirrors the clap variant.
#[cfg_attr(not(feature = "models"), allow(dead_code))]
pub struct EvolveArgs {
    pub corpus: String,
    pub run: String,
    pub target: f32,
    pub max_gens: usize,
    pub samples: usize,
    pub rounds: usize,
    pub max_new_tokens: usize,
}

/// Self-improvement loop: generation over generation, eval the current
/// capability, train only the failing tasks warm-started from the prior
/// adapter, and once converged persist the result as a routable expert.
pub async fn evolve(url: &str, args: EvolveArgs) -> anyhow::Result<()> {
    let EvolveArgs {
        corpus,
        run,
        target,
        max_gens,
        samples,
        rounds,
        max_new_tokens,
    } = args;
    #[cfg(feature = "models")]
    {
        use antumbra_train::{eval_pass_rate, raft_train, Corpus, ModelLoader};
        let store = crate::connect(url).await?;
        let embedder = crate::make_embedder()?;
        let cfg = RaftConfig {
            samples_per_task: samples,
            rounds,
            max_new_tokens,
            ..RaftConfig::default()
        };
        let loader = CandleModelLoader::new(cfg.clone());
        let verifier = antumbra_critic::CommandVerifier;
        let tasks = JsonCorpus::from_file(&corpus)?.tasks(&[]);
        let mut parent: Option<String> = None;
        let mut solved: Vec<String> = Vec::new();
        let mut final_rate = 0.0f32;
        for gen in 0..max_gens {
            // Load the current capability (warm-started from the prior
            // generation's adapter, or the bare base on gen 0).
            let mut model = ModelLoader::load(&loader, &cfg.base_model, parent.as_deref()).await?;
            let ev = eval_pass_rate(
                &mut model,
                &verifier,
                &tasks,
                &RunId::new("evolve:eval"),
                samples,
            )
            .await?;
            final_rate = ev.pass_rate;
            println!(
                "gen {gen}: pass-rate {:.2} ({}/{}) [{}]",
                ev.pass_rate,
                ev.passed,
                ev.total,
                parent.as_deref().unwrap_or("base")
            );
            if ev.pass_rate >= target {
                println!("converged at gen {gen} (>= target {target:.2})");
                break;
            }
            // Below the bar: train only the tasks currently failing
            // (the gaps the serving check just found), warm-started.
            let failing: Vec<_> = tasks
                .iter()
                .zip(&ev.per_task)
                .filter(|(_, r)| r.rate() < target)
                .map(|(t, _)| t.clone())
                .collect();
            let gaps = if failing.is_empty() {
                tasks.clone()
            } else {
                failing
            };
            let run_id = RunId::new(format!("{run}-g{gen}"));
            println!("  training {} gap task(s)", gaps.len());
            let out = raft_train(&mut model, &verifier, &gaps, &run_id, &cfg).await?;
            println!(
                "  trained gen {gen}: round-final {:.2} -> {}",
                out.final_fitness, out.adapter_uri
            );
            parent = Some(out.adapter_uri);
            solved = out.capability_exemplars;
            final_rate = out.final_fitness;
        }

        // Self-improvement feeds the population: persist the converged
        // capability as a routable expert (behavior-derived capability
        // vector), then refresh the gate so it can route to it.
        if let Some(uri) = parent {
            let protos: Vec<String> = if solved.is_empty() {
                tasks.iter().map(|t| t.prompt.clone()).collect()
            } else {
                solved.clone()
            };
            let mut acc = vec![0.0f32; EMBED_DIM];
            for text in &protos {
                for (a, b) in acc.iter_mut().zip(embedder.embed(text).await?) {
                    *a += b;
                }
            }
            let n = protos.len().max(1) as f32;
            let centroid: Vec<f32> = acc.iter().map(|x| x / n).collect();
            let now = Utc::now();
            let expert = Expert {
                id: ExpertId::new(format!("expert:{run}")),
                name: run.clone(),
                base_model: cfg.base_model.clone(),
                artifact_uri: uri,
                capability_card: serde_json::json!({ "exemplars": solved }),
                capability_vec: Some(centroid),
                fitness: final_rate,
                frozen_at: Some(now),
                generation: Generation::ZERO,
                owner: None,
                compartment: None,
                created_at: now,
            };
            expert::delete(&store, &expert.id).await?; // supersede on re-run
            expert::insert(&store, &expert).await?;
            println!("persisted expert {run} into the population (fitness {final_rate:.2})");
            if let Ok(Some(r)) = refresh_router(&store, embedder.as_ref(), 400).await {
                println!("router refreshed over {} experts", r.experts.len());
            }
        }
        Ok(())
    }
    #[cfg(not(feature = "models"))]
    {
        let _ = (
            url,
            &corpus,
            &run,
            target,
            max_gens,
            samples,
            rounds,
            max_new_tokens,
        );
        anyhow::bail!("`evolve` requires building with --features models (candle + a GPU)")
    }
}

/// Parameters for [`serve`].
#[cfg_attr(not(feature = "models"), allow(dead_code))]
pub struct ServeArgs {
    pub task: Option<String>,
    pub max_new_tokens: usize,
    pub threshold: f32,
    pub temperature: f64,
}

/// Resident multi-adapter server with hardware-adaptive serving: load the shared base once and
/// hot-swap each routed expert's adapter per prompt via [`MultiAdapterServe`].
/// Answers a single `--task` or a stream of prompts from stdin, routing each
/// through the learned router (boundary-conditioned gate as fallback). A stream
/// pays the base load only on the first prompt; repeated routes to the same
/// expert reuse the resident factors.
pub async fn serve(url: &str, args: ServeArgs) -> anyhow::Result<()> {
    let ServeArgs {
        task,
        max_new_tokens,
        threshold,
        temperature,
    } = args;
    #[cfg(feature = "models")]
    {
        use antumbra_serve::MultiAdapterServe;

        let store = crate::connect(url).await?;
        let embedder = crate::make_embedder()?;
        let experts = expert::list(&store).await?;
        if experts.is_empty() {
            anyhow::bail!(
                "no experts in the population; grow some with `populate` / `evolve` first"
            );
        }
        let boundaries = boundary::list(&store).await?;
        let router = antumbra_store::repo::router::load(&store).await?;

        // The resident engine: one shared base, every expert's adapter registered
        // so a route hot-swaps to it without reloading the base.
        let base_model = experts[0].base_model.clone();
        let cfg = RaftConfig::for_serving(max_new_tokens, temperature);
        let mut engine = MultiAdapterServe::new(base_model, cfg);
        for e in &experts {
            engine.register(e.id.clone(), e.artifact_uri.clone());
        }
        eprintln!(
            "resident server: {} adapter(s) registered; base loads on the first prompt",
            engine.len()
        );

        match task {
            Some(prompt) => {
                serve_prompt(
                    &engine,
                    embedder.as_ref(),
                    &router,
                    &boundaries,
                    &experts,
                    threshold,
                    &prompt,
                )
                .await?;
            }
            None => {
                use std::io::BufRead;
                let stdin = std::io::stdin();
                for line in stdin.lock().lines() {
                    let prompt = line?;
                    let prompt = prompt.trim();
                    if prompt.is_empty() {
                        continue;
                    }
                    serve_prompt(
                        &engine,
                        embedder.as_ref(),
                        &router,
                        &boundaries,
                        &experts,
                        threshold,
                        prompt,
                    )
                    .await?;
                }
            }
        }
        Ok(())
    }
    #[cfg(not(feature = "models"))]
    {
        let _ = (url, &task, max_new_tokens, threshold, temperature);
        anyhow::bail!("`serve` requires building with --features models (candle + a GPU)")
    }
}

/// Route one prompt and serve its answer from the resident engine. Shared by the
/// one-shot and stdin paths so residency (base loaded once, adapters swapped) is
/// the only difference between them.
#[cfg(feature = "models")]
#[allow(clippy::too_many_arguments)]
async fn serve_prompt(
    engine: &antumbra_serve::MultiAdapterServe,
    embedder: &dyn antumbra_core::ports::Embedder,
    router: &Option<antumbra_core::LearnedRouter>,
    boundaries: &[antumbra_core::FailureBoundary],
    experts: &[Expert],
    threshold: f32,
    prompt: &str,
) -> anyhow::Result<()> {
    use antumbra_core::ports::{ActRequest, Serve};

    let v = embedder.embed(prompt).await?;
    let radius = GateConfig::default().inhibition_radius;
    let inhib = boundaries
        .iter()
        .map(|b| b.inhibition_for(&v, radius))
        .fold(0.0f32, f32::max);
    let chosen: Option<ExpertId> = match router {
        Some(r) if r.covers(&v) && inhib <= 0.5 => r.route(&v).first().map(|(id, _)| id.clone()),
        Some(_) => None,
        None => {
            let gc = GateConfig {
                coverage_threshold: threshold,
                ..GateConfig::default()
            };
            let d = gate_route(&v, experts, boundaries, 1, &gc);
            if d.escalate {
                None
            } else {
                d.chosen.first().cloned()
            }
        }
    };
    match chosen {
        None => println!("[escalate] no in-scope expert for: {prompt}"),
        Some(id) => {
            let out = engine
                .act(ActRequest {
                    task_id: "serve".into(),
                    prompt: prompt.into(),
                    adapters: vec![id.clone()],
                })
                .await?;
            let name = experts
                .iter()
                .find(|e| e.id == id)
                .map(|e| e.name.as_str())
                .unwrap_or_else(|| id.as_str());
            println!("[{name}] {}", out.final_output);
        }
    }
    Ok(())
}

/// Parameters for [`metabolize`].
#[cfg_attr(not(feature = "models"), allow(dead_code))]
pub struct MetabolizeArgs {
    /// A normalized trace file (any harness exports to this shape).
    pub source: String,
    pub out: String,
    pub min_recurrence: u32,
    /// Drop the per-step decomposition (learn only the collapsed outcome).
    pub no_steps: bool,
    /// Re-pull and metabolize on a cadence.
    pub watch: bool,
    pub interval_secs: u64,
    pub train: bool,
    pub run: String,
    pub rounds: usize,
    pub samples: usize,
    pub max_new_tokens: usize,
    pub lr: f64,
}

/// Metabolize a harness's successful orchestration traces into the frozen-expert population: adapt a
/// normalized harness-trace export (loop runs, behavior-graph evaluations, task
/// executions) into capture tasks the population internalizes, so the brain
/// learns to do in one shot what the harness did in many steps. Writes the
/// converted corpus; with `--train`, internalizes it through the capture loop.
pub async fn metabolize(url: &str, args: MetabolizeArgs) -> anyhow::Result<()> {
    let MetabolizeArgs {
        source,
        out,
        min_recurrence,
        no_steps,
        watch,
        interval_secs,
        train,
        run,
        rounds,
        samples,
        max_new_tokens,
        lr,
    } = args;
    #[cfg(feature = "models")]
    {
        use antumbra_train::{
            metabolize as metabolize_traces, parse_harness_traces, HarnessTrace, MetabolizePolicy,
        };

        let policy = MetabolizePolicy {
            min_recurrence,
            include_steps: !no_steps,
        };

        // Read and normalize the trace export (re-read each cycle under --watch).
        let get_traces = || -> anyhow::Result<Vec<HarnessTrace>> {
            Ok(parse_harness_traces(&std::fs::read(&source)?)?)
        };

        // The store + embedder are reused across watch cycles when training.
        let store = if train {
            Some(crate::connect(url).await?)
        } else {
            None
        };
        let embedder = if train {
            Some(crate::make_embedder()?)
        } else {
            None
        };

        loop {
            let traces = get_traces()?;
            let tasks = metabolize_traces(&traces, &policy);

            // Cluster by kind (loop / graph / task) for report.
            let mut kinds: Vec<(String, usize)> = Vec::new();
            for t in &tasks {
                let k = t.skill();
                match kinds.iter_mut().find(|(s, _)| *s == k) {
                    Some((_, n)) => *n += 1,
                    None => kinds.push((k, 1)),
                }
            }
            println!(
                "metabolized {} task(s) from {} trace(s) across {} kind(s)",
                tasks.len(),
                traces.len(),
                kinds.len()
            );
            for (k, n) in &kinds {
                println!("  kind '{k}': {n} task(s)");
            }

            // Persist the converted capture corpus ({id,prompt,verify,completion,skill}).
            let arr: Vec<serde_json::Value> = tasks
                .iter()
                .map(|t| {
                    let mut o = serde_json::Map::new();
                    o.insert("id".into(), serde_json::json!(t.id));
                    o.insert("prompt".into(), serde_json::json!(t.prompt));
                    o.insert("verify".into(), t.verify.clone());
                    if let Some(c) = &t.completion {
                        o.insert("completion".into(), serde_json::json!(c));
                    }
                    o.insert("skill".into(), serde_json::json!(t.skill()));
                    serde_json::Value::Object(o)
                })
                .collect();
            std::fs::write(&out, serde_json::to_vec_pretty(&arr)?)?;
            println!("wrote capture corpus -> {out}");

            if !train {
                println!(
                    "run `antumbra teach --corpus {out}` to internalize the metabolized traces"
                );
            } else if tasks.is_empty() {
                println!("nothing to train: no trace cleared the metabolization gate");
            } else {
                let store = store.as_ref().expect("store built when train");
                let embedder = embedder
                    .as_ref()
                    .expect("embedder built when train")
                    .as_ref();
                let cfg = RaftConfig {
                    samples_per_task: samples,
                    rounds,
                    max_new_tokens,
                    learning_rate: lr,
                    ..RaftConfig::default()
                };
                let corpus = JsonCorpus::from_tasks(tasks.clone());
                let verifier = std::sync::Arc::new(antumbra_critic::CommandVerifier);
                let loader = CandleModelLoader::new(cfg.clone());
                let trainer = CaptureTrainer::new(cfg, loader, corpus, verifier);
                let loop_cfg = LoopConfig {
                    graduate_threshold: 0.3,
                    base_model: "Qwen/Qwen2.5-Coder-1.5B".into(),
                };
                let lp = GenerationLoop::new(store, &trainer, embedder, loop_cfg);
                let reports = lp.run_until(&RunId::new(run.clone()), 1).await?;
                for r in &reports {
                    println!(
                        "gen {:<3} expert {:<16} internalized={:.2} graduated={}",
                        r.generation.0, r.shadow, r.fitness, r.graduated
                    );
                }
                if let Ok(Some(r)) = refresh_router(store, embedder, 400).await {
                    println!("router refreshed over {} experts", r.experts.len());
                }
            }

            if !watch {
                break;
            }
            println!("metabolize: sleeping {interval_secs}s until the next cycle (ctrl-c to stop)");
            tokio::select! {
                _ = tokio::time::sleep(std::time::Duration::from_secs(interval_secs)) => {}
                _ = tokio::signal::ctrl_c() => { println!("metabolize: stopping"); break; }
            }
        }
        Ok(())
    }
    #[cfg(not(feature = "models"))]
    {
        let _ = (
            url,
            &source,
            &out,
            min_recurrence,
            no_steps,
            watch,
            interval_secs,
            train,
            &run,
            rounds,
            samples,
            max_new_tokens,
            lr,
        );
        anyhow::bail!(
            "`metabolize` requires building with --features models (it adapts harness traces \
             into the capture corpus and, with --train, internalizes them)"
        )
    }
}
