//! Antumbra operator CLI (`antumbra`). v0 surface: apply the schema, print the
//! generated DDL, inspect the population, and drive the generational loop.
//!
//! `loop --demo` drives the durable generational loop with the scripted demo
//! trainer + embedder so persistence and resume are exercisable without a model;
//! real training is `train` under `--features models`.

use antumbra_copal::CopalArchive;
use antumbra_core::ports::Embedder;
use antumbra_core::testing::{FixedEmbedder, ScriptedTrainer};
use antumbra_core::{Expert, ExpertId, Generation, RunId, ShadowStatus};
use antumbra_embed::HttpEmbedder;
use antumbra_gate::{route as gate_route, GateConfig};
use antumbra_loop::{GenerationLoop, LoopConfig};
use antumbra_store::repo::{boundary, expert, shadow};
use antumbra_store::{schema, ConnectionConfig, Store, EMBED_DIM};
use chrono::Utc;
use clap::Parser;

mod cli;
mod commands;
mod gitctx;
mod gitfacts;
mod ops;
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

/// Root DB credentials for an authenticated remote, captured once from the global
/// `--db-user`/`--db-pass` so `connect` (called from many handlers that only carry
/// the url) can apply them without threading them through every call site.
static DB_CREDS: std::sync::OnceLock<(Option<String>, Option<String>)> = std::sync::OnceLock::new();

async fn connect(url: &str) -> anyhow::Result<Store> {
    let mut builder = ConnectionConfig::builder()
        .url(url)
        .namespace("antumbra")
        .database("main");
    // Root login for an authenticated remote (`ws://`); embedded stores need none.
    if let Some((Some(user), Some(pass))) = DB_CREDS.get() {
        builder = builder.username(user.as_str()).password(pass.as_str());
    }
    let config = builder.build()?;
    Ok(Store::connect(config, EMBED_DIM).await?)
}

/// The embedder choice from the global `--embedder-url` / `--fake-embedder`
/// flags, captured once (like `DB_CREDS`) so the many handlers that embed can
/// call `make_embedder` without threading the flags through every call site.
static EMBEDDER: std::sync::OnceLock<EmbedderChoice> = std::sync::OnceLock::new();

/// The copal document-of-record archive from the global `--copal-*` flags,
/// resolved once like the embedder: `None` when no address is configured.
static COPAL: std::sync::OnceLock<Option<std::sync::Arc<CopalArchive>>> =
    std::sync::OnceLock::new();

fn copal_archive() -> Option<std::sync::Arc<CopalArchive>> {
    COPAL.get().cloned().flatten()
}

struct EmbedderChoice {
    url: Option<String>,
    model: String,
    key: Option<String>,
    fake: bool,
}

/// The active embedder. A configured `--embedder-url` always wins (the same
/// endpoint the population was built with). Otherwise the byte-histogram
/// stand-in only when `--fake-embedder` asks for it explicitly, the real candle
/// BERT under `--features models`, and a refusal without `models`: a command
/// whose recall silently matched character statistics instead of meaning would
/// be worse than one that stops and says so. Every path produces `EMBED_DIM`-wide
/// vectors so the gate and store stay dimension-consistent.
fn make_embedder() -> anyhow::Result<Box<dyn Embedder>> {
    let choice = EMBEDDER
        .get()
        .ok_or_else(|| anyhow::anyhow!("embedder flags were not captured before a handler ran"))?;
    if let Some(url) = &choice.url {
        return Ok(Box::new(HttpEmbedder::new(
            url.clone(),
            choice.model.clone(),
            choice.key.clone(),
        )));
    }
    if choice.fake {
        return Ok(Box::new(FixedEmbedder::new(EMBED_DIM)));
    }
    builtin_embedder()
}

#[cfg(feature = "models")]
fn builtin_embedder() -> anyhow::Result<Box<dyn Embedder>> {
    Ok(Box::new(antumbra_serve::BertEmbedder::load()?))
}

#[cfg(not(feature = "models"))]
fn builtin_embedder() -> anyhow::Result<Box<dyn Embedder>> {
    anyhow::bail!(
        "no embedder configured: pass --embedder-url <OpenAI-compatible /embeddings endpoint \
         returning {EMBED_DIM}-d vectors> (for example Ollama serving all-minilm), or \
         --fake-embedder to accept the non-semantic byte-histogram stand-in (demos only)"
    )
}

/// Train (or retrain) the learned router over the whole population's exemplars
/// and persist it (the learned gate). Returns the expert count it covers, or `None` when
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

fn main() -> anyhow::Result<()> {
    // SurrealDB query evaluation (the engine-enforced ACL subqueries for tenant
    // isolation and memory compartments) recurses deep; the OS default main-thread stack (1 MB on Windows)
    // overflows. Host the runtime on a thread with a large stack. (Tests pass
    // because they run on tokio worker threads, which already have room.)
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(|| {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?
                .block_on(run())
        })?
        .join()
        .map_err(|_| anyhow::anyhow!("antumbra worker thread panicked"))?
}

async fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let _ = DB_CREDS.set((cli.db_user.clone(), cli.db_pass.clone()));
    let _ = EMBEDDER.set(EmbedderChoice {
        url: cli.embedder_url.clone(),
        model: cli.embedder_model.clone(),
        key: cli.embedder_key.clone(),
        fake: cli.fake_embedder,
    });
    let _ = COPAL.set(CopalArchive::from_flags(
        cli.copal_addr.as_deref(),
        cli.copal_tenant.clone(),
        cli.copal_key.clone(),
        cli.copal_keys.as_deref(),
    )?);
    match cli.command {
        Command::Migrate => {
            connect(&cli.url).await?;
            println!("schema applied at {}", cli.url);
        }
        Command::Sync {
            remote,
            remote_user,
            remote_pass,
            interval,
            once,
        } => {
            let local = antumbra_sync::Endpoint::embedded(&cli.url);
            let remote_ep = match (remote_user, remote_pass) {
                (Some(user), Some(pass)) => {
                    antumbra_sync::Endpoint::authoritative(&remote, user, pass)
                }
                _ => antumbra_sync::Endpoint::embedded(&remote),
            };
            let cfg = antumbra_sync::SyncConfig::new(local, remote_ep)
                .with_interval(std::time::Duration::from_secs(interval));
            if once {
                let stats = antumbra_sync::worker::run_once(&cfg).await?;
                println!("sync: {} pushed, {} pulled", stats.pushed, stats.pulled);
            } else {
                println!(
                    "sync: reconciling {} <-> {} every {interval}s (ctrl-c to stop)",
                    cli.url, remote
                );
                let (tx, rx) = tokio::sync::watch::channel(false);
                tokio::spawn(async move {
                    let _ = tokio::signal::ctrl_c().await;
                    let _ = tx.send(true);
                });
                antumbra_sync::worker::run(cfg, rx).await?;
                println!("sync: stopped");
            }
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
        Command::Loop {
            generations,
            run,
            demo,
        } => {
            if !demo {
                anyhow::bail!(
                    "`loop` drives the durable loop with a scripted demo trainer that always \
                     graduates; no model is trained. Pass --demo to run it as such, or use \
                     `train` under --features models for real training."
                );
            }
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
            // Prefer the learned router (the learned gate) once trained; it separates
            // specialists from generalists where raw-cosine coverage cannot.
            if let Some(router) = antumbra_store::repo::router::load(&store).await? {
                let ranked = router.route(&task_vec);
                let sim = router.top_similarity(&task_vec);
                // The learned router routes; boundaries still inhibit (a known
                // failure region escalates even if an expert covers it).
                let inhib = boundary::list(&store)
                    .await?
                    .iter()
                    .map(|b| b.inhibition_for(&task_vec, GateConfig::default().inhibition_radius))
                    .fold(0.0f32, f32::max);
                // The learned router abstains by its own
                // OOD floor; --threshold governs only the heuristic fallback below.
                println!(
                    "note: --threshold ({threshold:.3}) applies to the heuristic gate; the learned router abstains by its OOD floor {:.3}",
                    router.floor
                );
                if !router.covers(&task_vec) {
                    println!(
                        "decision: ESCALATE (out of distribution: similarity {sim:.3} < floor {:.3})",
                        router.floor
                    );
                } else if inhib > 0.5 {
                    println!("decision: ESCALATE (boundary inhibits this context: {inhib:.3})");
                } else if let Some((top, p)) = ranked.first() {
                    println!("decision: route to [{top}] (learned, p={p:.3}, sim={sim:.3})");
                } else {
                    // covers() is false when route() is empty, so this is belt-and-
                    // suspenders against a width-mismatched router; never panic.
                    println!("decision: ESCALATE (learned router surfaced no in-scope expert)");
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
            temperature,
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
                        let sim = router.top_similarity(&task_vec);
                        // Escalate when out of distribution, or when a boundary
                        // inhibits this context (a known failure region) -- and
                        // never index an empty ranking (a width-mismatched router).
                        let inhib = boundary::list(&store)
                            .await?
                            .iter()
                            .map(|b| {
                                b.inhibition_for(&task_vec, GateConfig::default().inhibition_radius)
                            })
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
                            let (name, w) = part.split_once(':').ok_or_else(|| {
                                anyhow::anyhow!("bad --with `{part}` (want name:weight)")
                            })?;
                            let s = experts.iter().find(|e| e.name == name.trim()).ok_or_else(
                                || anyhow::anyhow!("standing expert `{}` not found", name.trim()),
                            )?;
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
            }
            #[cfg(not(feature = "models"))]
            {
                let _ = (
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
                        // way to scope contexts that differ slightly (the counterfactual boundary of competence).
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
                let mut model = ModelLoader::load(&loader, &base_model, adapter.as_deref()).await?;
                let verifier = antumbra_critic::CommandVerifier;
                let out =
                    eval_pass_rate(&mut model, &verifier, &tasks, &RunId::new("eval"), samples)
                        .await?;
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
                anyhow::bail!(
                    "`gate-train` requires building with --features models (real embedder)"
                );
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
                anyhow::bail!(
                    "`compose` requires building with --features models (candle + a GPU)"
                );
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
            commands::evolve(
                &cli.url,
                commands::EvolveArgs {
                    corpus,
                    run,
                    target,
                    max_gens,
                    samples,
                    rounds,
                    max_new_tokens,
                },
            )
            .await?;
        }
        Command::Serve {
            task,
            max_new_tokens,
            threshold,
            temperature,
        } => {
            commands::serve(
                &cli.url,
                commands::ServeArgs {
                    task,
                    max_new_tokens,
                    threshold,
                    temperature,
                },
            )
            .await?;
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
            commands::populate(
                &cli.url,
                commands::PopulateArgs {
                    corpus,
                    run,
                    target_coverage,
                    max_experts,
                    samples,
                    rounds,
                    max_new_tokens,
                },
            )
            .await?;
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
            commands::memory_import(
                &cli.url,
                commands::MemoryImportArgs {
                    source,
                    out,
                    capture_threshold,
                    train,
                    run,
                    rounds,
                    samples,
                    max_new_tokens,
                    lr,
                },
            )
            .await?;
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
            grad_accumulation,
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
                    grad_accumulation,
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
        Command::Remember {
            tenant,
            user,
            compartment,
            content,
            network,
            confidence,
        } => {
            ops::remember(
                &cli.url,
                ops::RememberArgs {
                    tenant,
                    user,
                    compartment,
                    content,
                    network,
                    confidence,
                },
            )
            .await?;
        }
        Command::Ingest {
            tenant,
            user,
            title,
            file,
            source,
            path,
            no_git,
            run,
        } => {
            let (content, default_source) = match (file, run.is_empty()) {
                (Some(f), true) => (
                    std::fs::read_to_string(&f)
                        .map_err(|e| anyhow::anyhow!("read {}: {e}", f.display()))?,
                    f.display().to_string(),
                ),
                (None, false) => (ops::run_for_output(&run)?, format!("$ {}", run.join(" "))),
                _ => anyhow::bail!("give exactly one of --file <path> or `-- <command ...>`"),
            };
            // The anchor is what makes the document answerable later: a recalled
            // chunk names the commit it describes, so a session can tell whether
            // it is still on HEAD.
            let provenance = (!no_git)
                .then(gitctx::detect)
                .flatten()
                .map(|p| match path {
                    Some(path) => p.at_path(path),
                    None => p,
                });
            ops::ingest(
                &cli.url,
                copal_archive().as_deref(),
                ops::IngestArgs {
                    tenant,
                    user,
                    title,
                    source: Some(source.unwrap_or(default_source)),
                    content,
                    provenance,
                },
            )
            .await?;
        }
        Command::GitFacts {
            tenant,
            user,
            compartment,
            days,
            top,
            dry_run,
        } => {
            let log = gitfacts::read_log(days).ok_or_else(|| {
                anyhow::anyhow!("not inside a git repository (or git is not on the path)")
            })?;
            let facts = gitfacts::derive(&gitfacts::parse_log(&log), top);
            let sentences = gitfacts::render(&facts, days);
            let anchor = gitctx::detect();
            let evidence = gitfacts::evidence(anchor.as_ref(), &facts, days);
            for s in &sentences {
                println!("{s}");
            }
            if dry_run {
                println!("(dry run: {} fact(s) not stored)", sentences.len());
            } else {
                let stored = ops::store_facts(
                    &cli.url,
                    ops::FactsArgs {
                        tenant,
                        user,
                        compartment,
                        facts: sentences,
                        evidence,
                    },
                )
                .await?;
                println!("stored {stored} fact(s) with evidence {:?}", facts.range);
            }
        }
        Command::Metabolize {
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
        } => {
            commands::metabolize(
                &cli.url,
                commands::MetabolizeArgs {
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
                },
            )
            .await?;
        }
        Command::ProposeCompartments {
            tenant,
            user,
            inbox,
            similarity_threshold,
            min_size,
            apply,
        } => {
            ops::propose_compartments(
                &cli.url,
                ops::ProposeCompartmentsArgs {
                    tenant,
                    user,
                    inbox,
                    similarity_threshold,
                    min_size,
                    apply,
                },
            )
            .await?;
        }
        Command::SetEmbedder {
            tenant,
            endpoint,
            model,
            key,
            source_dim,
        } => {
            ops::set_embedder(&cli.url, &tenant, &endpoint, &model, key, source_dim).await?;
        }
        Command::GetEmbedder { tenant } => {
            ops::get_embedder(&cli.url, &tenant).await?;
        }
        Command::Reembed {
            tenant,
            dry_run,
            yes,
        } => {
            ops::reembed(&cli.url, &tenant, dry_run, yes).await?;
        }
    }
    Ok(())
}
