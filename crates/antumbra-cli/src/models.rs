//! The handlers that need a model: `ask`, `scope`, `train`, and `teach`.
//!
//! They sit here for the reason `ops.rs` and `commands.rs` give for holding
//! theirs: `main.rs` was already large, and these four were the largest arms
//! in it. Each reuses `main.rs`'s `connect` / `make_embedder` /
//! `refresh_router` through `crate::`, so no infrastructure is duplicated, and
//! each is gated inside its body exactly as it was in the match.

#[cfg(feature = "models")]
use antumbra_boundary::{discover_boundary, find_scope_over_contexts, finding_to_boundary};
#[cfg(feature = "models")]
use antumbra_core::ports::{ActRequest, Serve, Trainer};
#[cfg(feature = "models")]
use antumbra_core::{BoundaryId, ExpertId, Generation, Grain, RunId};
#[cfg(feature = "models")]
use antumbra_gate::{route as gate_route, GateConfig};
#[cfg(feature = "models")]
use antumbra_loop::{GenerationLoop, LoopConfig};
#[cfg(feature = "models")]
use antumbra_serve::{CandleServe, GenerateVerifyProbe};
#[cfg(feature = "models")]
use antumbra_store::repo::{boundary, expert, lifecycle};
#[cfg(feature = "models")]
use antumbra_train::{
    CandleModelLoader, CaptureTrainer, GrpoTrainer, JsonCorpus, RaftConfig, RaftTrainer,
};
#[cfg(feature = "models")]
use chrono::Utc;

#[cfg(feature = "models")]
use crate::{connect, make_embedder, refresh_router, RouterRefresh};

pub use crate::eval_args::EvalArgs;

/// `antumbra eval`: an adapter's pass rate on a corpus, with no training.
pub async fn eval(args: EvalArgs) -> anyhow::Result<()> {
    #[cfg(feature = "models")]
    {
        use antumbra_train::{eval_pass_rate, CausalLm, Corpus, ModelLoader};
        let EvalArgs {
            corpus,
            adapter,
            base_model,
            samples,
            max_new_tokens,
            report,
            completions,
            seed,
            temperature,
            top_p,
            dtype,
        } = args;
        let defaults = RaftConfig::default();
        let cfg = RaftConfig {
            samples_per_task: samples,
            max_new_tokens,
            temperature: temperature.unwrap_or(defaults.temperature),
            top_p: top_p.unwrap_or(defaults.top_p),
            dtype: match dtype.as_deref() {
                Some(name) => name.parse()?,
                None => defaults.dtype,
            },
            ..defaults
        };
        let corpus_doc = JsonCorpus::from_file(&corpus)?;
        let tasks = corpus_doc.tasks(&[]);
        let base_model = base_model.unwrap_or_else(|| cfg.base_model.clone());
        let (cfg_temperature, cfg_top_p, cfg_dtype) = (cfg.temperature, cfg.top_p, cfg.dtype);
        let loader = CandleModelLoader::new(cfg);
        let mut model = ModelLoader::load(&loader, &base_model, adapter.as_deref()).await?;
        if let Some(seed) = seed {
            model.seed_draws(seed)?;
        }
        let verifier = antumbra_critic::CommandVerifier;
        let out =
            eval_pass_rate(&mut model, &verifier, &tasks, &RunId::new("eval"), samples).await?;
        println!(
            "pass-rate {:.2} ({}/{}): {} on {} ({} tasks)",
            out.pass_rate,
            out.passed,
            out.total,
            adapter.as_deref().unwrap_or("(base only)"),
            corpus,
            tasks.len()
        );
        for (i, ex) in out.examples.iter().enumerate() {
            println!("  sample[{i}]: {}", ex.replace('\n', " ").trim());
        }
        if let Some(path) = &report {
            let mut record = antumbra_train::eval::report(
                &corpus,
                &base_model,
                adapter.as_deref(),
                samples,
                max_new_tokens,
                &out,
            );
            record["temperature"] = serde_json::json!(cfg_temperature);
            record["top_p"] = serde_json::json!(cfg_top_p);
            record["dtype"] = serde_json::json!(format!("{cfg_dtype:?}"));
            record["seed"] = serde_json::json!(seed);
            std::fs::write(path, serde_json::to_vec_pretty(&record)?)?;
            println!("per-task results -> {path}");
        }
        if let Some(path) = &completions {
            let draws: Vec<serde_json::Value> = out
                .draws
                .iter()
                .map(|d| {
                    serde_json::json!({ "task": d.task, "completion": d.completion, "passed": d.passed })
                })
                .collect();
            std::fs::write(path, serde_json::to_vec_pretty(&draws)?)?;
            println!("{} completion(s) -> {path}", draws.len());
        }
        Ok(())
    }
    #[cfg(not(feature = "models"))]
    {
        let EvalArgs {
            corpus,
            adapter,
            base_model,
            samples,
            max_new_tokens,
            report,
            completions,
            seed,
            temperature,
            top_p,
            dtype,
        } = args;
        let _ = (
            corpus,
            adapter,
            base_model,
            samples,
            max_new_tokens,
            report,
            completions,
            seed,
            temperature,
            top_p,
            dtype,
        );
        anyhow::bail!("`eval` requires building with --features models (candle + a GPU)")
    }
}

/// What `antumbra ask` was given.
pub struct AskArgs {
    pub task: String,
    pub k: usize,
    pub max_new_tokens: usize,
    pub threshold: f32,
    pub with: Option<String>,
    pub self_weight: f32,
    pub temperature: f64,
}

pub async fn ask(url: &str, args: AskArgs) -> anyhow::Result<()> {
    let task = args.task;
    let k = args.k;
    let max_new_tokens = args.max_new_tokens;
    let threshold = args.threshold;
    let with = args.with;
    let self_weight = args.self_weight;
    let temperature = args.temperature;
    #[cfg(feature = "models")]
    {
        let store = connect(url).await?;
        let embedder = make_embedder()?;
        let task_vec = embedder.embed(&task).await?;
        // Routing sees the experts the gate may route to; a standing expert
        // named with --with is served whether or not it is routed to.
        let experts = lifecycle::routable(&store).await?;
        let named = lifecycle::servable(&store).await?;
        // Pick the expert via the learned router when trained, else the
        // heuristic boundary-conditioned gate.
        let chosen_id: Option<ExpertId> =
            if let Some(router) = lifecycle::load_router(&store).await? {
                let ranked = router.route(&task_vec);
                let sim = router.top_similarity(&task_vec);
                // Escalate when out of distribution, or when a boundary
                // inhibits this context (a known failure region) -- and
                // never index an empty ranking (a width-mismatched router).
                let inhib = boundary::list(&store)
                    .await?
                    .iter()
                    .map(|b| b.inhibition_for(&task_vec, GateConfig::default().inhibition_radius))
                    .fold(0.0f32, f32::max);
                match ranked.first() {
                    Some((top, p)) if router.covers(&task_vec) && inhib <= 0.5 => {
                        println!("learned router: top {top} (p={p:.3}, sim={sim:.3})");
                        Some(top.clone())
                    }
                    _ => None,
                }
            } else {
                let boundaries = boundary::list(&store).await?;
                let cfg = GateConfig {
                    coverage_threshold: threshold,
                    ..GateConfig::default()
                };
                let decision = gate_route(&task_vec, &experts, &boundaries, k, &cfg);
                decision.chosen.first().cloned()
            };
        if let Some(chosen) = chosen_id.as_ref() {
            let expert = experts
                .iter()
                .find(|e| &e.id == chosen)
                .ok_or_else(|| anyhow::anyhow!("routed expert {chosen} not found"))?;
            println!(
                "routing to {} (adapter {})",
                expert.name, expert.artifact_uri
            );
            // Compose the routed (contextual) expert with the named
            // standing experts (your conventions, always applied); else
            // serve the routed expert alone.
            let serve = if let Some(with_spec) = &with {
                let mut specs: Vec<(String, f32)> =
                    vec![(expert.artifact_uri.clone(), self_weight)];
                for part in with_spec.split(',') {
                    let (name, w) = part
                        .split_once(':')
                        .ok_or_else(|| anyhow::anyhow!("bad --with `{part}` (want name:weight)"))?;
                    let s = named
                        .iter()
                        .find(|e| e.name == name.trim())
                        .ok_or_else(|| {
                            anyhow::anyhow!("standing expert `{}` not found", name.trim())
                        })?;
                    specs.push((s.artifact_uri.clone(), w.trim().parse()?));
                }
                let merged = "adapters/_composed.safetensors";
                let rank = antumbra_train::compose_adapters(&specs, merged)?;
                let base_scale = RaftConfig::default().lora_scale();
                println!(
                    "composing with {} standing expert(s) -> rank {rank}",
                    specs.len() - 1
                );
                let cfg = RaftConfig {
                    lora_rank: rank,
                    lora_alpha: base_scale * rank as f64,
                    ..RaftConfig::for_serving(max_new_tokens, temperature)
                };
                CandleServe::new(expert.base_model.clone(), Some(merged.to_string()), cfg)
            } else {
                let cfg = RaftConfig::for_serving(max_new_tokens, temperature);
                CandleServe::new(
                    expert.base_model.clone(),
                    Some(expert.artifact_uri.clone()),
                    cfg,
                )
            };
            let out = serve
                .act(ActRequest {
                    task_id: "ask".into(),
                    prompt: task.clone(),
                    adapters: vec![expert.id.clone()],
                })
                .await?;
            println!("---");
            println!("{}", out.final_output);
        } else {
            println!("decision: ESCALATE to flagship; no in-scope expert");
        }
        Ok(())
    }
    #[cfg(not(feature = "models"))]
    {
        let _ = (
            url,
            &task,
            k,
            max_new_tokens,
            threshold,
            &with,
            self_weight,
            temperature,
        );
        anyhow::bail!("`ask` requires building with --features models (candle + a GPU)");
    }
}

/// What `antumbra scope` was given.
pub struct ScopeArgs {
    pub spec: String,
    pub expert: Option<String>,
    pub discover: bool,
    pub max_new_tokens: usize,
    pub samples: usize,
    pub temperature: f64,
    pub confidence: f32,
}

pub async fn scope(url: &str, args: ScopeArgs) -> anyhow::Result<()> {
    let spec = args.spec;
    let expert = args.expert;
    let discover = args.discover;
    let max_new_tokens = args.max_new_tokens;
    let samples = args.samples;
    let temperature = args.temperature;
    let confidence = args.confidence;
    #[cfg(feature = "models")]
    {
        let store = connect(url).await?;
        let raw = std::fs::read_to_string(&spec)?;
        let doc: serde_json::Value = serde_json::from_str(&raw)?;
        let behavior = doc["behavior"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("spec.behavior must be a string"))?;
        let governing_feature = doc["governing_feature"].as_str().unwrap_or("context");
        let fail_context = doc["fail_context"].clone();
        let candidates: Vec<serde_json::Value> = doc["candidates"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("spec.candidates must be an array of contexts"))?
            .clone();

        // Probe with an expert's adapter (maps that expert's boundary) or
        // the bare base. The expert acts the behavior; the verifier judges.
        let (base_model, adapter) = match &expert {
            Some(name) => {
                let experts = lifecycle::servable(&store).await?;
                let e = experts
                    .iter()
                    .find(|e| &e.name == name)
                    .ok_or_else(|| anyhow::anyhow!("expert `{name}` not found"))?;
                println!(
                    "probing with expert {} (adapter {})",
                    e.name, e.artifact_uri
                );
                (e.base_model.clone(), Some(e.artifact_uri.clone()))
            }
            None => (
                doc["base_model"]
                    .as_str()
                    .map_or_else(|| RaftConfig::default().base_model, str::to_string),
                None,
            ),
        };

        let cfg = RaftConfig {
            max_new_tokens,
            temperature,
            ..RaftConfig::default()
        };
        let serve = CandleServe::new(base_model, adapter, cfg);
        let probe =
            GenerateVerifyProbe::new(serve, antumbra_critic::CommandVerifier).with_samples(samples);

        let finding_opt = if discover {
            // Treat fail_context + candidates as one pool and infer the
            // governing feature from which contexts the expert passes.
            let pool: Vec<serde_json::Value> = std::iter::once(fail_context.clone())
                .chain(candidates.iter().cloned())
                .collect();
            discover_boundary(behavior, &pool, &probe).await?
        } else {
            find_scope_over_contexts(
                behavior,
                governing_feature,
                &fail_context,
                &candidates,
                &probe,
            )
            .await?
        };

        match finding_opt {
            Some(finding) => {
                if discover {
                    println!("(governing feature inferred from pass/fail, not supplied)");
                }
                println!("recovered governing feature: {}", finding.governing_feature);
                println!("C' (acceptable context): {}", finding.near_ok_context);
                let embedder = make_embedder()?;
                // Embed the *semantic* fail context only -- strip the
                // verify spec, whose code/JSON pollutes the vector and
                // weakens the boundary's in-scope inhibition.
                let mut ctx_only = fail_context.clone();
                if let Some(obj) = ctx_only.as_object_mut() {
                    obj.remove("verify");
                }
                let context_vec = embedder.embed(&format!("{behavior} {ctx_only}")).await?;
                // Embed C' too, so inhibition is relative (closer to the
                // failure than to the acceptable context) -- the only
                // way to scope contexts that differ slightly (the counterfactual boundary of competence).
                let mut ok_only = finding.near_ok_context.clone();
                if let Some(obj) = ok_only.as_object_mut() {
                    obj.remove("verify");
                }
                let ok_vec = embedder.embed(&format!("{behavior} {ok_only}")).await?;
                let id = BoundaryId::new(format!("boundary:scope:{}", finding.governing_feature));
                let bound = finding_to_boundary(
                    id,
                    &finding,
                    Grain::Project,
                    confidence,
                    Some(context_vec),
                    Some(ok_vec),
                    Generation::ZERO,
                    Utc::now(),
                );
                boundary::upsert(&store, &bound).await?;
                println!(
                    "stored actionable boundary (governing={}, confidence={confidence:.2})",
                    finding.governing_feature
                );
            }
            None => {
                println!("no candidate context was acceptable; boundary stays open");
            }
        }
        Ok(())
    }
    #[cfg(not(feature = "models"))]
    {
        let _ = (
            url,
            &spec,
            &expert,
            discover,
            max_new_tokens,
            samples,
            temperature,
            confidence,
        );
        anyhow::bail!("`scope` requires building with --features models (candle + GPU + python)");
    }
}

pub use crate::train_args::TrainArgs;

pub async fn train(url: &str, args: TrainArgs) -> anyhow::Result<()> {
    let corpus = args.corpus;
    let generations = args.generations;
    let run = args.run;
    let samples = args.samples;
    let rounds = args.rounds;
    let max_new_tokens = args.max_new_tokens;
    let algo = args.algo;
    let critic = args.critic;
    #[cfg(feature = "models")]
    let critic_weight = args.critic_weight;
    if critic.is_some() && algo != "grpo" {
        anyhow::bail!("--critic shapes GRPO's advantages; use it with --algo grpo");
    }
    let critic_twin = args.critic_twin;
    if critic_twin.is_some() && critic.is_none() {
        anyhow::bail!("--critic-twin is read against a critic; give --critic too");
    }
    let quantize_base = args.quantize_base;
    let parent = args.parent;
    let holdout = args.holdout;
    let search = args.search.then(|| {
        let cohort = args.cohort.max(1);
        let mut policy = antumbra_loop::search::SearchPolicy {
            cohort,
            slow: args.slow.unwrap_or(cohort / 3),
            slow_interval: args.slow_interval.max(1),
            anneal: args.anneal.unwrap_or(generations),
            ..Default::default()
        };
        // GRPO weighs its KL penalty; RAFT has none to weigh.
        if algo == "grpo" {
            policy.space.kl_beta = (0.0, 0.2);
        }
        policy
    });
    let remeasure = match args.remeasure.unwrap_or(if args.search { 3 } else { 0 }) {
        0 => None,
        repeats => Some(antumbra_loop::Remeasure { repeats }),
    };
    // The grow step reads the census the contribution measurement takes, so
    // it needs one every generation, and over enough tasks to cover every
    // region.
    let max_tasks = args.contribution_tasks.unwrap_or(if args.grow {
        64
    } else {
        antumbra_loop::ContributionPolicy::default().max_tasks
    });
    let contribution = match (args.contribution_every, args.grow) {
        (0, false) => None,
        (0, true) => Some(antumbra_loop::ContributionPolicy {
            every: 1,
            max_tasks,
            ..Default::default()
        }),
        (every, _) => Some(antumbra_loop::ContributionPolicy {
            every,
            max_tasks,
            ..Default::default()
        }),
    };
    let choosing = match args.grow_by.as_str() {
        "credit" => antumbra_loop::Choosing::Credit,
        "learnability" => antumbra_loop::Choosing::Learnability,
        "uniform" => antumbra_loop::Choosing::Uniform,
        other => anyhow::bail!("--grow-by {other}: use credit, learnability or uniform"),
    };
    let warm_start = match args.grow_from.as_str() {
        "incumbent" => true,
        "base" => false,
        other => anyhow::bail!("--grow-from {other}: use incumbent or base"),
    };
    let grow = args.grow.then(|| antumbra_loop::GrowPolicy {
        choosing,
        warm_start,
        ..Default::default()
    });
    let admission = (args.duplicate_above <= 1.0).then(|| antumbra_loop::AdmissionPolicy {
        duplicate_above: args.duplicate_above,
        ..Default::default()
    });
    let retirement = (args.retire_after > 0).then(|| antumbra_loop::RetirementPolicy {
        persist: args.retire_after,
        ..Default::default()
    });
    let merge = args.merge.then(|| antumbra_loop::MergePolicy {
        retained_above: args.merge_retained,
        ..Default::default()
    });
    #[cfg(feature = "models")]
    {
        let store = connect(url).await?;
        let cfg = RaftConfig {
            samples_per_task: samples,
            rounds,
            max_new_tokens,
            quantize_base,
            parent_adapter: parent.clone(),
            ..RaftConfig::default()
        };
        let corpus = JsonCorpus::from_file(&corpus)?;
        // A task may name a verifier in the namespace instead of carrying its
        // spec; the gate runs it only while it may grant reward (ADR-0022 S-4).
        let verifier = std::sync::Arc::new(antumbra_critic::Governed::new(
            antumbra_critic::CommandVerifier,
            std::sync::Arc::new(antumbra_store::repo::verifier::Registry::new(store.clone())),
        ));
        // The loop hands this name to the trainer, which loads it, so it must be
        // the configured base rather than a literal: a literal here once kept
        // every `train` run on the raw completion model after the default moved
        // to the Instruct one the recipe was validated on.
        let base_model = cfg.base_model.clone();
        let start = cfg.recipe();
        let loader = CandleModelLoader::new(cfg.clone());
        let trainer: Box<dyn Trainer> = match algo.as_str() {
            "grpo" => {
                let grpo = GrpoTrainer::new(cfg.clone(), loader, corpus, verifier);
                match &critic {
                    Some(adapter) => {
                        use antumbra_train::ModelLoader;
                        let judge = CandleModelLoader::new(cfg.clone());
                        let model =
                            ModelLoader::load(&judge, &cfg.base_model, Some(adapter)).await?;
                        println!("critic: {adapter} at weight {critic_weight}");
                        let critic = antumbra_train::critic::ModelCritic::new(Box::new(model));
                        let shaped = grpo.with_critic(std::sync::Arc::new(critic), critic_weight);
                        match &critic_twin {
                            Some(twin) => {
                                let model =
                                    ModelLoader::load(&judge, &cfg.base_model, Some(twin)).await?;
                                println!("critic twin: {twin}, watched and shaping nothing");
                                let twin =
                                    antumbra_train::critic::ModelCritic::new(Box::new(model));
                                Box::new(shaped.with_critic_twin(std::sync::Arc::new(twin)))
                            }
                            None => Box::new(shaped),
                        }
                    }
                    None => Box::new(grpo),
                }
            }
            "raft" => Box::new(RaftTrainer::new(cfg, loader, corpus, verifier)),
            other => anyhow::bail!("unknown --algo `{other}` (use raft or grpo)"),
        };
        println!("algorithm: {algo}");
        let embedder = make_embedder()?;
        let loop_cfg = LoopConfig {
            graduate_threshold: 0.3,
            base_model,
            partition: holdout.then(antumbra_core::slice::Partition::default),
            // A searched run starts from the recipe the trainer is configured with.
            recipe: search.is_some().then_some(start),
            search,
            remeasure,
            contribution,
            admission,
            retirement,
            merge,
            grow,
            ..LoopConfig::default()
        };
        // Every generation, the synthesized verifiers that judged its training
        // are measured again against the tasks' authored anchors (ADR-0022 S-4).
        let rechecker = antumbra_critic::CommandVerifier;
        let lp = GenerationLoop::new(&store, trainer.as_ref(), embedder.as_ref(), loop_cfg)
            .rechecking(&rechecker);
        let reports = lp.run_until(&RunId::new(run), generations).await?;
        for r in &reports {
            let curve: Vec<String> = r.reward_curve.iter().map(|p| format!("{p:.2}")).collect();
            println!(
                "gen {:<3} shadow {:<16} pass-rate/round=[{}] final={:.2} graduated={}",
                r.generation.0,
                r.shadow,
                curve.join(", "),
                r.fitness,
                r.graduated
            );
            if let Some(m) = &r.instruments {
                let gap = m
                    .widest_gap()
                    .map(|(band, w)| format!("{w:+.2} ({} tasks)", band.as_str()))
                    .unwrap_or_else(|| "not measured".into());
                let audit = m
                    .audit
                    .rate()
                    .map(|a| format!("{a:.2} over {}", m.audit.measured))
                    .unwrap_or_else(|| "not due".into());
                let trend = r.trend.map_or("not read", |t| t.as_str());
                println!("        widest held-out gap {gap}, audit {audit}, trend {trend}");
            }
            if let Some(recipe) = &r.recipe {
                println!(
                    "        recipe lr {:.1e}, batch {}, kl {}",
                    recipe.learning_rate, recipe.batch_size, recipe.kl_beta
                );
            }
            if r.cohort.len() > 1 {
                for m in &r.cohort {
                    let recipe = m.recipe.map_or("unreported".to_string(), |x| {
                        format!(
                            "lr {:.1e} batch {} kl {}",
                            x.learning_rate, x.batch_size, x.kl_beta
                        )
                    });
                    println!(
                        "        member {} {recipe} fitness {:.2}{}",
                        m.shadow,
                        m.fitness,
                        if m.slow { " (slow)" } else { "" }
                    );
                }
                if r.remeasured.is_none() {
                    println!(
                        "        graduation score {:.2} (the best, shrunk toward the cohort's mean)",
                        r.graduation_score
                    );
                }
            }
            if let Some(m) = &r.remeasured {
                let rates: Vec<String> = m.pass_rates.iter().map(|p| format!("{p:.2}")).collect();
                println!(
                    "        re-measured on {} {} task(s) under {} seed(s): [{}], graduation score {:.2}",
                    m.tasks,
                    if m.held_out { "held-out" } else { "trained" },
                    m.pass_rates.len(),
                    rates.join(", "),
                    r.graduation_score
                );
            }
            match &r.admission {
                Some(antumbra_loop::Admission::Superseded {
                    archived,
                    similarity,
                    candidate,
                    incumbent,
                }) => println!(
                    "        admitted in place of {archived} (similarity {similarity:.3}): \
                     {candidate:.2} against its {incumbent:.2}; {archived} archived"
                ),
                Some(antumbra_loop::Admission::Rejected {
                    duplicate_of,
                    similarity,
                    candidate,
                    incumbent,
                }) => {
                    let head_to_head = match (candidate, incumbent) {
                        (Some(c), Some(i)) => format!("{c:.2} against its {i:.2}"),
                        _ => "not measurable here".to_string(),
                    };
                    println!(
                        "        not admitted: duplicates {duplicate_of} (similarity {similarity:.3}), {head_to_head}"
                    );
                }
                Some(antumbra_loop::Admission::Outserved { tasks, .. }) if *tasks == 0 => println!(
                    "        not admitted: the gate would route none of the live tasks to it"
                ),
                Some(antumbra_loop::Admission::Outserved {
                    tasks,
                    escalated,
                    candidate,
                    serving,
                }) => println!(
                    "        not admitted: the population scores {candidate:.2} with it on the {tasks} live task(s) it would reroute ({escalated} of them escalated to the base model), against {serving:.2} without it"
                ),
                Some(antumbra_loop::Admission::Admitted {
                    nearest: Some((id, s)),
                }) => println!("        admitted: nearest expert {id} at similarity {s:.3}"),
                _ => {}
            }
            if let Some(w) = &r.critic {
                let read = |v: Option<f32>| v.map_or("-".to_string(), |v| format!("{v:.2}"));
                println!(
                    "        critic over {} answer(s): correlation {}, calibration error {} ({} recalibrated), twin agreement {}",
                    w.n,
                    read(w.correlation),
                    read(w.ece),
                    read(w.recalibrated_ece),
                    read(w.twin_agreement)
                );
            }
            for c in &r.rechecks {
                let verdict = c
                    .measurement
                    .as_ref()
                    .map_or("not measured".to_string(), |m| format!("{:?}", m.verdict));
                let moved = c.moved.map_or(String::new(), |to| {
                    format!("; now {to:?}, {} expert(s) archived", c.archived.len())
                });
                println!(
                    "        recheck {}: {} answer(s) anchored, {} not; {} rewarded answer(s) the anchor failed; {verdict}{moved}",
                    c.verifier, c.anchored, c.unanchored, c.rewarded_wrong
                );
            }
            if !r.withdrawn.is_empty() {
                let ids: Vec<&str> = r.withdrawn.iter().map(|v| v.as_str()).collect();
                println!(
                    "        not graduated: trained under {}, which no longer grant reward",
                    ids.join(", ")
                );
            }
            for (expert, warning) in &r.detection.warnings {
                println!("        warning (advisory) {expert}: {warning:?}");
            }
            for moved in &r.detection.demoted {
                println!(
                    "        demoted {} to dormant on {:?}",
                    moved.expert, moved.cause
                );
            }
            if let Some(g) = &r.growth {
                match &g.record.chosen {
                    Some(region) => {
                        let learnability = g
                            .record
                            .candidates
                            .iter()
                            .find(|c| &c.region == region)
                            .map_or(0.0, |c| c.learnability);
                        let credit = g
                            .record
                            .credit
                            .map_or("none yet".to_string(), |c| format!("{c:+.2}"));
                        let from = g
                            .record
                            .warm_from
                            .as_ref()
                            .map_or("fresh factors".to_string(), |e| format!("warm from {e}"));
                        println!(
                            "        grow: learned from {region} (learnability {learnability:.3}; {} task(s), {} unfiltered; {from}); last choice's credit {credit}; entropy {:.2}, coverage {:.2}, revived {}",
                            g.record.focus, g.record.unfiltered, g.diversity.entropy, g.diversity.coverage, g.diversity.revived
                        );
                    }
                    None => println!(
                        "        grow: no region chosen (no census yet, or none passed the gate); learned from every visible task"
                    ),
                }
            }
            match &r.merge {
                Some(antumbra_loop::Merge::Merged {
                    into,
                    pair,
                    similarity,
                    retained,
                    merged,
                    better,
                }) => println!(
                    "        merged {} and {} into {into} (similarity {similarity:.3}, overlap {retained:.3}): {merged:.2} against the better's {better:.2}; both archived",
                    pair.0, pair.1
                ),
                Some(antumbra_loop::Merge::NotSiblings {
                    pair,
                    similarity,
                    retained,
                }) => println!(
                    "        not merged: {} and {} (similarity {similarity:.3}) share {retained:.3} of their subspace",
                    pair.0, pair.1
                ),
                Some(antumbra_loop::Merge::Costly {
                    pair,
                    retained,
                    merged,
                    better,
                    ..
                }) => println!(
                    "        not merged: {} and {} (overlap {retained:.3}) merge to {merged:.2} against the better's {better:.2}",
                    pair.0, pair.1
                ),
                None => {}
            }
            if let Some(b) = &r.baseline {
                let best = match (&b.best, b.best_alone, b.delta()) {
                    (Some(id), Some(alone), Some(d)) => {
                        format!("its best single expert {id} {alone:.2} (routing adds {d:+.2})")
                    }
                    _ => "no single expert to compare".to_string(),
                };
                let headroom = b.headroom().map_or(String::new(), |h| {
                    format!(
                        "; routed as well as it could be, {:.2} ({h:+.2})",
                        b.population + h
                    )
                });
                println!(
                    "        population {:.2} over {} live task(s) against {best}{headroom}",
                    b.population, b.tasks
                );
            }
            for c in &r.contribution {
                let delta = match (c.with, c.without, c.delta()) {
                    (Some(with), Some(without), Some(d)) => {
                        format!("with {with:.2} without {without:.2} contribution {d:+.2}")
                    }
                    _ => "unused".to_string(),
                };
                println!(
                    "        expert {} routed {}/{} {delta}",
                    c.expert, c.routed, c.tasks
                );
            }
            let scored = r.contribution_scores;
            if scored.asked > 0 {
                println!(
                    "        contribution scores: {} task score(s) asked, {} already known",
                    scored.asked, scored.reused
                );
            }
        }
        println!("population: {} experts", expert::list(&store).await?.len());
        // Self-maintaining gate: keep the learned router current with the
        // population so routing never needs a manual `gate-train`.
        if let Ok(RouterRefresh::Trained(r)) = refresh_router(&store, embedder.as_ref(), 400).await
        {
            println!("router refreshed over {} experts", r.experts.len());
        }
        Ok(())
    }
    #[cfg(not(feature = "models"))]
    {
        let _ = (
            url,
            &corpus,
            generations,
            &run,
            samples,
            rounds,
            max_new_tokens,
            &algo,
            quantize_base,
            &parent,
            holdout,
            &search,
            &remeasure,
            &contribution,
            &admission,
            &retirement,
            &merge,
            &grow,
        );
        anyhow::bail!("`train` requires building with --features models (candle + a GPU)");
    }
}

/// What `antumbra teach` was given.
pub struct TeachArgs {
    pub corpus: String,
    pub generations: u32,
    pub run: String,
    pub rounds: usize,
    pub samples: usize,
    pub max_new_tokens: usize,
    pub lr: f64,
    pub parent: Option<String>,
}

pub async fn teach(url: &str, args: TeachArgs) -> anyhow::Result<()> {
    let corpus = args.corpus;
    let generations = args.generations;
    let run = args.run;
    let rounds = args.rounds;
    let samples = args.samples;
    let max_new_tokens = args.max_new_tokens;
    let lr = args.lr;
    let parent = args.parent;
    #[cfg(feature = "models")]
    {
        let store = connect(url).await?;
        let cfg = RaftConfig {
            samples_per_task: samples,
            rounds,
            max_new_tokens,
            learning_rate: lr,
            parent_adapter: parent.clone(),
            ..RaftConfig::default()
        };
        let corpus = JsonCorpus::from_file(&corpus)?;
        let verifier = std::sync::Arc::new(antumbra_critic::CommandVerifier);
        let base_model = cfg.base_model.clone();
        let loader = CandleModelLoader::new(cfg.clone());
        let trainer = CaptureTrainer::new(cfg, loader, corpus, verifier);
        let embedder = make_embedder()?;
        let loop_cfg = LoopConfig {
            graduate_threshold: 0.3,
            base_model,
            ..LoopConfig::default()
        };
        let lp = GenerationLoop::new(&store, &trainer, embedder.as_ref(), loop_cfg);
        let reports = lp.run_until(&RunId::new(run), generations).await?;
        for r in &reports {
            println!(
                "gen {:<3} expert {:<16} internalized={:.2} graduated={}",
                r.generation.0, r.shadow, r.fitness, r.graduated
            );
        }
        // Retire boundaries the new correction resolves: a captured
        // expert whose competence lands in a failure region supersedes
        // the boundary that flagged it (the boundary lifecycle). The gate
        // then routes the region to the fix instead of escalating.
        // Only an expert the gate routes to covers a region.
        let experts = lifecycle::routable(&store).await?;
        for b in boundary::list(&store).await? {
            let covered = experts.iter().any(|e| {
                e.capability_vec
                    .as_deref()
                    .is_some_and(|v| b.is_covered_by(v))
            });
            if covered {
                boundary::delete(&store, &b.id).await?;
                println!("retired boundary {} (resolved by a captured expert)", b.id);
            }
        }
        println!("population: {} experts", experts.len());
        if let Ok(RouterRefresh::Trained(r)) = refresh_router(&store, embedder.as_ref(), 400).await
        {
            println!("router refreshed over {} experts", r.experts.len());
        }
        Ok(())
    }
    #[cfg(not(feature = "models"))]
    {
        let _ = (
            url,
            &corpus,
            generations,
            &run,
            rounds,
            samples,
            max_new_tokens,
            lr,
            &parent,
        );
        anyhow::bail!("`teach` requires building with --features models (candle + a GPU)");
    }
}
