//! Antumbra operator CLI (`antumbra`). v0 surface: apply the schema, print the
//! generated DDL, inspect the population, and drive the generational loop.
//!
//! Until the real candle/llama engines land (antumbra-train / antumbra-serve),
//! `loop` runs with the demo trainer + embedder so the ADR-0008 machine is
//! exercisable end-to-end.

use antumbra_core::ports::Embedder;
use antumbra_core::testing::{FixedEmbedder, ScriptedTrainer};
use antumbra_core::{Expert, ExpertId, Generation, RunId, ShadowStatus};
use antumbra_gate::{route as gate_route, GateConfig};
use antumbra_loop::{GenerationLoop, LoopConfig};
use antumbra_store::repo::{boundary, expert, shadow};
use antumbra_store::{schema, ConnectionConfig, Store, EMBED_DIM};
use chrono::Utc;
use clap::Parser;

mod ops;
mod cli;
use cli::{Cli, Command};

#[cfg(feature = "models")]
use antumbra_boundary::{discover_boundary, find_scope_over_contexts, finding_to_boundary};
#[cfg(feature = "models")]
use antumbra_core::ports::{ActRequest, Serve, Trainer};
#[cfg(feature = "models")]
use antumbra_core::{BoundaryId, Grain};
#[cfg(feature = "models")]
use antumbra_serve::{CandleServe, GenerateVerifyProbe};
#[cfg(feature = "models")]
use antumbra_train::{
    CandleModelLoader, CaptureTrainer, GrpoTrainer, JsonCorpus, RaftConfig, RaftTrainer,
};


async fn connect(url: &str) -> anyhow::Result<Store> {
    let config = ConnectionConfig::builder()
        .url(url)
        .namespace("antumbra")
        .database("main")
        .build()?;
    Ok(Store::connect(config, EMBED_DIM).await?)
}

/// The active embedder: real candle BERT under `--features models`, else the
/// byte-histogram fake. Both produce `EMBED_DIM`-wide vectors so the gate and
/// store stay dimension-consistent.
#[cfg(feature = "models")]
fn make_embedder() -> anyhow::Result<Box<dyn Embedder>> {
    Ok(Box::new(antumbra_serve::BertEmbedder::load()?))
}

#[cfg(not(feature = "models"))]
fn make_embedder() -> anyhow::Result<Box<dyn Embedder>> {
    Ok(Box::new(FixedEmbedder::new(EMBED_DIM)))
}

/// Train (or retrain) the learned router over the whole population's exemplars
/// and persist it (ADR-0009). Returns the expert count it covers, or `None` when
/// the population is too small to need a router (<2 experts/exemplars). This is
/// the self-maintaining gate: `train`/`teach` call it so routing stays current
/// without a manual `gate-train`.
#[cfg(feature = "models")]
async fn refresh_router(
    store: &Store,
    embedder: &dyn Embedder,
    epochs: usize,
) -> anyhow::Result<Option<antumbra_core::LearnedRouter>> {
    let experts = expert::list(store).await?;
    if experts.len() < 2 {
        return Ok(None);
    }
    let mut exemplars: Vec<(ExpertId, Vec<f32>)> = Vec::new();
    for e in &experts {
        let cards = e
            .capability_card
            .get("exemplars")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        for ex in cards {
            if let Some(s) = ex.as_str() {
                exemplars.push((e.id.clone(), embedder.embed(s).await?));
            }
        }
    }
    if exemplars.len() < 2 {
        return Ok(None);
    }
    let router = antumbra_train::train_learned_router(&exemplars, epochs)?;
    antumbra_store::repo::router::save(store, &router).await?;
    Ok(Some(router))
}

/// The demo specialists `seed` registers, as (name, capability description).
const DEMO_SPECIALISTS: [(&str, &str); 3] = [
    (
        "arith-specialist",
        "adds subtracts multiplies and divides integers and numbers arithmetic math",
    ),
    (
        "string-specialist",
        "reverses concatenates slices and formats text strings",
    ),
    (
        "datetime-specialist",
        "parses formats and computes differences between dates times and calendars",
    ),
];

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Migrate => {
            connect(&cli.url).await?;
            println!("schema applied at {}", cli.url);
        }
        Command::Schema => {
            for statement in schema::schema_statements(EMBED_DIM as u32)? {
                println!("{statement}");
            }
        }
        Command::Experts => {
            let store = connect(&cli.url).await?;
            let experts = expert::list(&store).await?;
            if experts.is_empty() {
                println!("(no experts yet)");
            }
            for e in experts {
                println!(
                    "{:<20} fitness={:.2}  base={}  frozen={}",
                    e.name,
                    e.fitness,
                    e.base_model,
                    e.is_frozen()
                );
            }
        }
        Command::Status => {
            let store = connect(&cli.url).await?;
            let experts = expert::list(&store).await?.len();
            let graduated = shadow::list_by_status(&store, ShadowStatus::Graduated)
                .await?
                .len();
            let pruned = shadow::list_by_status(&store, ShadowStatus::Pruned)
                .await?
                .len();
            let boundaries = boundary::list(&store).await?;
            let actionable = boundaries.iter().filter(|b| b.is_actionable()).count();
            println!("experts (umbra):        {experts}");
            println!("shadows graduated:      {graduated}");
            println!("shadows pruned:         {pruned}");
            println!(
                "boundaries (antumbra):  {} ({actionable} actionable, {} open)",
                boundaries.len(),
                boundaries.len() - actionable
            );
        }
        Command::Loop { generations, run } => {
            let store = connect(&cli.url).await?;
            let trainer = ScriptedTrainer::graduating();
            let embedder = FixedEmbedder::new(EMBED_DIM);
            let lp = GenerationLoop::new(&store, &trainer, &embedder, LoopConfig::default());
            let reports = lp.run_until(&RunId::new(run), generations).await?;
            for r in &reports {
                println!(
                    "gen {:<3} shadow {:<14} fitness={:.2} graduated={}",
                    r.generation.0, r.shadow, r.fitness, r.graduated
                );
            }
            let population = expert::list(&store).await?;
            println!("population: {} experts", population.len());
        }
        Command::Seed => {
            let store = connect(&cli.url).await?;
            let embedder = make_embedder()?;
            for (name, desc) in DEMO_SPECIALISTS {
                let now = Utc::now();
                let e = Expert {
                    id: ExpertId::new(format!("expert:{name}")),
                    name: name.to_string(),
                    base_model: "Qwen/Qwen2.5-Coder-1.5B".to_string(),
                    artifact_uri: format!("mem://{name}"),
                    capability_card: serde_json::json!({ "description": desc }),
                    capability_vec: Some(embedder.embed(desc).await?),
                    fitness: 1.0,
                    frozen_at: Some(now),
                    generation: Generation::ZERO,
                    owner: None,
                    compartment: None,
                    created_at: now,
                };
                expert::insert(&store, &e).await?;
                println!("seeded {name}");
            }
            println!("population: {} experts", expert::list(&store).await?.len());
        }
        Command::Route { task, k, threshold } => {
            let store = connect(&cli.url).await?;
            let embedder = make_embedder()?;
            let task_vec = embedder.embed(&task).await?;
            // Prefer the learned router (ADR-0009) once trained; it separates
            // specialists from generalists where raw-cosine coverage cannot.
            if let Some(router) = antumbra_store::repo::router::load(&store).await? {
                let ranked = router.route(&task_vec);
                let (top, p) = ranked[0].clone();
                let sim = router.top_similarity(&task_vec);
                // The learned router routes; boundaries still inhibit (a known
                // failure region escalates even if an expert covers it).
                let inhib = boundary::list(&store)
                    .await?
                    .iter()
                    .map(|b| b.inhibition_for(&task_vec, GateConfig::default().inhibition_radius))
                    .fold(0.0f32, f32::max);
                if !router.covers(&task_vec) {
                    println!(
                        "decision: ESCALATE (out of distribution: similarity {sim:.3} < floor {:.3})",
                        router.floor
                    );
                } else if inhib > 0.5 {
                    println!("decision: ESCALATE (boundary inhibits this context: {inhib:.3})");
                } else {
                    println!("decision: route to [{top}] (learned, p={p:.3}, sim={sim:.3})");
                }
                for (id, pr) in ranked.iter().take(k.max(3)) {
                    println!("  {:<22} p={pr:.3}", id.to_string());
                }
            } else {
                let experts = expert::list(&store).await?;
                let boundaries = boundary::list(&store).await?;
                let cfg = GateConfig {
                    coverage_threshold: threshold,
                    ..GateConfig::default()
                };
                let decision = gate_route(&task_vec, &experts, &boundaries, k, &cfg);
                if decision.escalate {
                    println!(
                        "decision: ESCALATE to flagship (coverage {:.3} < threshold {threshold:.3})",
                        decision.coverage
                    );
                } else {
                    let names: Vec<String> =
                        decision.chosen.iter().map(ToString::to_string).collect();
                    println!(
                        "decision: route to [{}] (coverage {:.3})",
                        names.join(", "),
                        decision.coverage
                    );
                }
                for scored in decision.ranked.iter().take(k.max(3)) {
                    println!("  {:<18} score={:.3}", scored.id.to_string(), scored.score);
                }
            }
        }
        Command::Ask {
            task,
            k,
            max_new_tokens,
            threshold,
            with,
            self_weight,
        } => {
            #[cfg(feature = "models")]
            {
                let store = connect(&cli.url).await?;
                let embedder = make_embedder()?;
                let task_vec = embedder.embed(&task).await?;
                let experts = expert::list(&store).await?;
                // Pick the expert via the learned router when trained, else the
                // heuristic boundary-conditioned gate.
                let chosen_id: Option<ExpertId> =
                    if let Some(router) = antumbra_store::repo::router::load(&store).await? {
                        let ranked = router.route(&task_vec);
                        let (top, p) = ranked[0].clone();
                        let sim = router.top_similarity(&task_vec);
                        println!("learned router: top {top} (p={p:.3}, sim={sim:.3})");
                        // Escalate when out of distribution, or when a boundary
                        // inhibits this context (a known failure region).
                        let inhib = boundary::list(&store)
                            .await?
                            .iter()
                            .map(|b| {
                                b.inhibition_for(&task_vec, GateConfig::default().inhibition_radius)
                            })
                            .fold(0.0f32, f32::max);
                        (router.covers(&task_vec) && inhib <= 0.5).then_some(top)
                    } else {
                        let boundaries = boundary::list(&store).await?;
                        let cfg = GateConfig {
                            coverage_threshold: threshold,
                            ..GateConfig::default()
                        };
                        let decision = gate_route(&task_vec, &experts, &boundaries, k, &cfg);
                        decision.chosen.first().cloned()
                    };
                if chosen_id.is_none() {
                    println!("decision: ESCALATE to flagship; no in-scope expert");
                } else {
                    let chosen = chosen_id.as_ref().unwrap();
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
                            let (name, w) = part.split_once(':').ok_or_else(|| {
                                anyhow::anyhow!("bad --with `{part}` (want name:weight)")
                            })?;
                            let s = experts
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
                        println!("composing with {} standing expert(s) -> rank {rank}", specs.len() - 1);
                        let cfg = RaftConfig {
                            lora_rank: rank,
                            lora_alpha: base_scale * rank as f64,
                            max_new_tokens,
                            ..RaftConfig::default()
                        };
                        CandleServe::new(expert.base_model.clone(), Some(merged.to_string()), cfg)
                    } else {
                        let cfg = RaftConfig {
                            max_new_tokens,
                            ..RaftConfig::default()
                        };
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
                }
            }
            #[cfg(not(feature = "models"))]
            {
                let _ = (&task, k, max_new_tokens, threshold, &with, self_weight);
                anyhow::bail!("`ask` requires building with --features models (candle + a GPU)");
            }
        }
        Command::Scope {
            spec,
            expert,
            discover,
            max_new_tokens,
            samples,
            temperature,
            confidence,
        } => {
            #[cfg(feature = "models")]
            {
                let store = connect(&cli.url).await?;
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
                        let experts = expert::list(&store).await?;
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
                            .unwrap_or("Qwen/Qwen2.5-Coder-1.5B")
                            .to_string(),
                        None,
                    ),
                };

                let cfg = RaftConfig {
                    max_new_tokens,
                    temperature,
                    ..RaftConfig::default()
                };
                let serve = CandleServe::new(base_model, adapter, cfg);
                let probe = GenerateVerifyProbe::new(serve, antumbra_critic::CommandVerifier)
                    .with_samples(samples);

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
                        // way to scope contexts that differ slightly (ADR-0004).
                        let mut ok_only = finding.near_ok_context.clone();
                        if let Some(obj) = ok_only.as_object_mut() {
                            obj.remove("verify");
                        }
                        let ok_vec = embedder.embed(&format!("{behavior} {ok_only}")).await?;
                        let id = BoundaryId::new(format!(
                            "boundary:scope:{}",
                            finding.governing_feature
                        ));
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
            }
            #[cfg(not(feature = "models"))]
            {
                let _ = (
                    &spec,
                    &expert,
                    discover,
                    max_new_tokens,
                    samples,
                    temperature,
                    confidence,
                );
                anyhow::bail!(
                    "`scope` requires building with --features models (candle + GPU + python)"
                );
            }
        }
        Command::Train {
            corpus,
            generations,
            run,
            samples,
            rounds,
            max_new_tokens,
            algo,
            quantize_base,
            parent,
        } => {
            #[cfg(feature = "models")]
            {
                let store = connect(&cli.url).await?;
                let cfg = RaftConfig {
                    samples_per_task: samples,
                    rounds,
                    max_new_tokens,
                    quantize_base,
                    parent_adapter: parent.clone(),
                    ..RaftConfig::default()
                };
                let corpus = JsonCorpus::from_file(&corpus)?;
                let verifier = std::sync::Arc::new(antumbra_critic::CommandVerifier);
                let loader = CandleModelLoader::new(cfg.clone());
                let trainer: Box<dyn Trainer> = match algo.as_str() {
                    "grpo" => Box::new(GrpoTrainer::new(cfg, loader, corpus, verifier)),
                    "raft" => Box::new(RaftTrainer::new(cfg, loader, corpus, verifier)),
                    other => anyhow::bail!("unknown --algo `{other}` (use raft or grpo)"),
                };
                println!("algorithm: {algo}");
                let embedder = make_embedder()?;
                let loop_cfg = LoopConfig {
                    graduate_threshold: 0.3,
                    base_model: "Qwen/Qwen2.5-Coder-1.5B".into(),
                    max_steps: 8,
                };
                let lp = GenerationLoop::new(&store, trainer.as_ref(), embedder.as_ref(), loop_cfg);
                let reports = lp.run_until(&RunId::new(run), generations).await?;
                for r in &reports {
                    let curve: Vec<String> =
                        r.reward_curve.iter().map(|p| format!("{p:.2}")).collect();
                    println!(
                        "gen {:<3} shadow {:<16} pass-rate/round=[{}] final={:.2} graduated={}",
                        r.generation.0,
                        r.shadow,
                        curve.join(", "),
                        r.fitness,
                        r.graduated
                    );
                }
                println!("population: {} experts", expert::list(&store).await?.len());
                // Self-maintaining gate: keep the learned router current with the
                // population so routing never needs a manual `gate-train`.
                if let Ok(Some(r)) = refresh_router(&store, embedder.as_ref(), 400).await {
                    println!("router refreshed over {} experts", r.experts.len());
                }
            }
            #[cfg(not(feature = "models"))]
            {
                let _ = (
                    &corpus,
                    generations,
                    &run,
                    samples,
                    rounds,
                    max_new_tokens,
                    &algo,
                    quantize_base,
                    &parent,
                );
                anyhow::bail!("`train` requires building with --features models (candle + a GPU)");
            }
        }
        Command::Eval {
            corpus,
            adapter,
            base_model,
            samples,
            max_new_tokens,
        } => {
            #[cfg(feature = "models")]
            {
                use antumbra_train::{eval_pass_rate, Corpus, ModelLoader};
                let cfg = RaftConfig {
                    samples_per_task: samples,
                    max_new_tokens,
                    ..RaftConfig::default()
                };
                let corpus_doc = JsonCorpus::from_file(&corpus)?;
                let tasks = corpus_doc.tasks(&[]);
                let loader = CandleModelLoader::new(cfg);
                let mut model =
                    ModelLoader::load(&loader, &base_model, adapter.as_deref()).await?;
                let verifier = antumbra_critic::CommandVerifier;
                let out = eval_pass_rate(
                    &mut model,
                    &verifier,
                    &tasks,
                    &RunId::new("eval"),
                    samples,
                )
                .await?;
                println!(
                    "pass-rate {:.2} ({}/{}) — {} on {} ({} tasks)",
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
            }
            #[cfg(not(feature = "models"))]
            {
                let _ = (&corpus, &adapter, &base_model, samples, max_new_tokens);
                anyhow::bail!("`eval` requires building with --features models (candle + a GPU)");
            }
        }
        Command::Teach {
            corpus,
            generations,
            run,
            rounds,
            samples,
            max_new_tokens,
            lr,
            parent,
        } => {
            #[cfg(feature = "models")]
            {
                let store = connect(&cli.url).await?;
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
                let loader = CandleModelLoader::new(cfg.clone());
                let trainer = CaptureTrainer::new(cfg, loader, corpus, verifier);
                let embedder = make_embedder()?;
                let loop_cfg = LoopConfig {
                    graduate_threshold: 0.3,
                    base_model: "Qwen/Qwen2.5-Coder-1.5B".into(),
                    max_steps: 8,
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
                // the boundary that flagged it (ADR-0004 lifecycle). The gate
                // then routes the region to the fix instead of escalating.
                let experts = expert::list(&store).await?;
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
                if let Ok(Some(r)) = refresh_router(&store, embedder.as_ref(), 400).await {
                    println!("router refreshed over {} experts", r.experts.len());
                }
            }
            #[cfg(not(feature = "models"))]
            {
                let _ = (
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
        Command::GateTrain { epochs } => {
            #[cfg(feature = "models")]
            {
                let store = connect(&cli.url).await?;
                let embedder = make_embedder()?;
                match refresh_router(&store, embedder.as_ref(), epochs).await? {
                    Some(r) => println!(
                        "trained learned router: {} experts, {} epochs, OOD floor={:.3}",
                        r.experts.len(),
                        epochs,
                        r.floor
                    ),
                    None => anyhow::bail!("need >=2 experts (with exemplars) to train a router"),
                }
            }
            #[cfg(not(feature = "models"))]
            {
                let _ = epochs;
                anyhow::bail!("`gate-train` requires building with --features models (real embedder)");
            }
        }
        Command::Compose {
            task,
            experts,
            max_new_tokens,
        } => {
            #[cfg(feature = "models")]
            {
                let store = connect(&cli.url).await?;
                let population = expert::list(&store).await?;
                // Resolve "name:weight,..." to adapter paths + weights.
                let mut specs: Vec<(String, f32)> = Vec::new();
                let mut base_model = String::new();
                for part in experts.split(',') {
                    let (name, w) = part
                        .split_once(':')
                        .ok_or_else(|| anyhow::anyhow!("bad spec `{part}` (want name:weight)"))?;
                    let e = population
                        .iter()
                        .find(|e| e.name == name.trim())
                        .ok_or_else(|| anyhow::anyhow!("expert `{}` not found", name.trim()))?;
                    base_model = e.base_model.clone();
                    specs.push((e.artifact_uri.clone(), w.trim().parse()?));
                }
                let merged = "adapters/_composed.safetensors";
                let rank = antumbra_train::compose_adapters(&specs, merged)?;
                println!(
                    "composed {} experts -> rank {} blended adapter",
                    specs.len(),
                    rank
                );
                // Keep the effective LoRA scale at the experts' value (alpha/rank)
                // even though the merged rank is larger.
                let base_scale = RaftConfig::default().lora_scale();
                let cfg = RaftConfig {
                    lora_rank: rank,
                    lora_alpha: base_scale * rank as f64,
                    max_new_tokens,
                    ..RaftConfig::default()
                };
                let serve = CandleServe::new(base_model, Some(merged.to_string()), cfg);
                let out = serve
                    .act(ActRequest {
                        task_id: "compose".into(),
                        prompt: task.clone(),
                        adapters: vec![],
                    })
                    .await?;
                println!("---");
                println!("{}", out.final_output);
            }
            #[cfg(not(feature = "models"))]
            {
                let _ = (&task, &experts, max_new_tokens);
                anyhow::bail!("`compose` requires building with --features models (candle + a GPU)");
            }
        }
        Command::Evolve {
            corpus,
            run,
            target,
            max_gens,
            samples,
            rounds,
            max_new_tokens,
        } => {
            #[cfg(feature = "models")]
            {
                use antumbra_train::{eval_pass_rate, raft_train, Corpus, ModelLoader};
                let store = connect(&cli.url).await?;
                let embedder = make_embedder()?;
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
                    let mut model =
                        ModelLoader::load(&loader, &cfg.base_model, parent.as_deref()).await?;
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
                    let gaps = if failing.is_empty() { tasks.clone() } else { failing };
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
            }
            #[cfg(not(feature = "models"))]
            {
                let _ = (&corpus, &run, target, max_gens, samples, rounds, max_new_tokens);
                anyhow::bail!("`evolve` requires building with --features models (candle + a GPU)");
            }
        }
        Command::Populate {
            corpus,
            run,
            target_coverage,
            max_experts,
            samples,
            rounds,
            max_new_tokens,
        } => {
            #[cfg(feature = "models")]
            {
                use antumbra_train::{eval_pass_rate, raft_train, Corpus, CorpusTask, ModelLoader};
                let store = connect(&cli.url).await?;
                let embedder = make_embedder()?;
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
                    // specialist for each — a narrow frozen expert per skill, not
                    // one generalist over all gaps (the umbra ideal, ADR-0001).
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
                        let out = raft_train(&mut model, &verifier, &gtasks, &RunId::new(name.clone()), &cfg)
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
            }
            #[cfg(not(feature = "models"))]
            {
                let _ = (
                    &corpus,
                    &run,
                    target_coverage,
                    max_experts,
                    samples,
                    rounds,
                    max_new_tokens,
                );
                anyhow::bail!("`populate` requires building with --features models (candle + a GPU)");
            }
        }
        Command::MemoryImport {
            source,
            out,
            capture_threshold,
            train,
            run,
            rounds,
            samples,
            max_new_tokens,
            lr,
        } => {
            #[cfg(feature = "models")]
            {
                use antumbra_train::memory::{import as import_memories, parse_export, ImportPolicy, Intake};
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
                    let store = connect(&cli.url).await?;
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
                    let embedder = make_embedder()?;
                    let loop_cfg = LoopConfig {
                        graduate_threshold: 0.3,
                        base_model: "Qwen/Qwen2.5-Coder-1.5B".into(),
                        max_steps: 8,
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
            }
            #[cfg(not(feature = "models"))]
            {
                let _ = (
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
                );
            }
        }
        Command::Consolidate {
            source,
            log,
            min_recurrence,
            min_confidence,
            train,
            replay_ratio,
            run,
            rounds,
            samples,
            max_new_tokens,
            lr,
        } => {
            ops::consolidate(
                &cli.url,
                ops::ConsolidateArgs {
                    source,
                    log,
                    min_recurrence,
                    min_confidence,
                    train,
                    replay_ratio,
                    run,
                    rounds,
                    samples,
                    max_new_tokens,
                    lr,
                },
            )
            .await?;
        }
        Command::ConsolidateCompartment {
            tenant,
            user,
            compartment,
            min_recurrence,
            min_confidence,
            rounds,
            samples,
            max_new_tokens,
            lr,
            replay_ratio,
        } => {
            ops::consolidate_compartment(
                &cli.url,
                ops::ConsolidateCompartmentArgs {
                    tenant,
                    user,
                    compartment,
                    min_recurrence,
                    min_confidence,
                    rounds,
                    samples,
                    max_new_tokens,
                    lr,
                    replay_ratio,
                },
            )
            .await?;
        }
        Command::Retire { expert } => {
            ops::retire(&cli.url, &expert).await?;
        }
    }
    Ok(())
}
