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
use clap::{Parser, Subcommand};

#[cfg(feature = "models")]
use antumbra_boundary::{discover_boundary, find_scope_over_contexts, finding_to_boundary};
#[cfg(feature = "models")]
use antumbra_core::ports::{ActRequest, Serve, Trainer};
#[cfg(feature = "models")]
use antumbra_core::{BoundaryId, Grain};
#[cfg(feature = "models")]
use antumbra_serve::{CandleServe, GenerateVerifyProbe};
#[cfg(feature = "models")]
use antumbra_train::{CandleModelLoader, GrpoTrainer, JsonCorpus, RaftConfig, RaftTrainer};

#[derive(Parser)]
#[command(name = "antumbra", about = "Antumbra operator CLI", version)]
struct Cli {
    /// SurrealDB url: `mem://` (ephemeral), `surrealkv://./data/antumbra.skv`
    /// (persistent), or `ws://host:8000/rpc`.
    #[arg(long, global = true, default_value = "mem://")]
    url: String,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Apply the schema (idempotent).
    Migrate,
    /// Print the generated schema DDL (surql-rs builder output).
    Schema,
    /// List the frozen expert population.
    Experts,
    /// Summarize the population, shadows, and inhibitory store.
    Status,
    /// Run the generational loop for N generations (demo trainer).
    Loop {
        #[arg(long, default_value_t = 1)]
        generations: u32,
        #[arg(long, default_value = "run:cli")]
        run: String,
    },
    /// Route a task through the boundary-conditioned gate. Uses the real BERT
    /// embedder under `--features models`, else the byte-histogram fake.
    Route {
        /// The task description to embed and route.
        task: String,
        #[arg(long, default_value_t = 2)]
        k: usize,
        /// Abstention threshold on relative coverage (top-1-minus-top-2
        /// capability margin); below it the gate escalates. Calibratable
        /// risk-coverage knob, not an absolute cosine floor.
        #[arg(long, default_value_t = 0.08)]
        threshold: f32,
    },
    /// Seed the population with described demo specialists, embedding each
    /// capability card with the active embedder (real BERT under --features
    /// models). Pair with a persistent --url to then `route` against them.
    Seed,
    /// Route a task to an expert and serve a real answer from its adapter
    /// (needs --features models + a GPU; the expert must have a trained adapter).
    Ask {
        /// The task to route and answer.
        task: String,
        #[arg(long, default_value_t = 1)]
        k: usize,
        /// Max tokens to generate for the answer.
        #[arg(long, default_value_t = 128)]
        max_new_tokens: usize,
    },
    /// Recover a failure boundary's scope by generate-then-verify (ADR-0004):
    /// hold a behavior fixed, vary the context, and find the governing feature
    /// and C' by actually serving and checking. Stores an actionable boundary.
    /// Needs --features models + a GPU + python.
    Scope {
        /// Path to a scope spec JSON: { behavior, base_model?, governing_feature?,
        /// fail_context, candidates: [<full context objects, each with verify>] }.
        #[arg(long)]
        spec: String,
        /// Probe with this expert's adapter (by name) instead of the bare base,
        /// so the search maps that expert's own competence boundary.
        #[arg(long)]
        expert: Option<String>,
        /// Infer the governing feature from which contexts pass vs fail, instead
        /// of trusting the spec's label (treats fail_context + candidates as a pool).
        #[arg(long)]
        discover: bool,
        #[arg(long, default_value_t = 96)]
        max_new_tokens: usize,
        /// Best-of-K samples per context check (generation is stochastic).
        #[arg(long, default_value_t = 8)]
        samples: usize,
        /// Sampling temperature; higher diversifies the best-of-K draws.
        #[arg(long, default_value_t = 0.8)]
        temperature: f64,
        /// Confidence assigned to the recovered boundary.
        #[arg(long, default_value_t = 0.7)]
        confidence: f32,
    },
    /// Train shadows with the real candle trainer (needs --features models + GPU).
    Train {
        /// Path to the JSON corpus of verifiable tasks ({id,prompt,verify}).
        #[arg(long)]
        corpus: String,
        #[arg(long, default_value_t = 1)]
        generations: u32,
        #[arg(long, default_value = "run:train")]
        run: String,
        /// Completions sampled per task per round (RAFT's K).
        #[arg(long, default_value_t = 8)]
        samples: usize,
        /// Rounds per shadow.
        #[arg(long, default_value_t = 4)]
        rounds: usize,
        /// Max tokens generated per completion.
        #[arg(long, default_value_t = 256)]
        max_new_tokens: usize,
        /// Algorithm: `raft` (reward-ranked SFT) or `grpo` (ADR-0011).
        #[arg(long, default_value = "raft")]
        algo: String,
        /// Quantize the frozen base to 4-bit Q4_K (QLoRA-proper, ADR-0011).
        #[arg(long)]
        quantize_base: bool,
        /// Warm-start the LoRA from this saved adapter (continual fine-tune)
        /// instead of fresh factors. EXP-010's monolithic arm (ADR-0011).
        #[arg(long)]
        parent: Option<String>,
    },
    /// Score a saved adapter's pass-rate on a corpus, with no training (the
    /// EXP-010 forgetting probe). Needs --features models + a GPU + python.
    Eval {
        /// Path to the JSON corpus of verifiable tasks ({id,prompt,verify}).
        #[arg(long)]
        corpus: String,
        /// Saved adapter to load over the base before scoring.
        #[arg(long)]
        adapter: String,
        #[arg(long, default_value = "Qwen/Qwen2.5-Coder-1.5B")]
        base_model: String,
        /// Completions sampled per task (the pass-rate denominator is tasks x K).
        #[arg(long, default_value_t = 8)]
        samples: usize,
        #[arg(long, default_value_t = 64)]
        max_new_tokens: usize,
    },
}

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
                let names: Vec<String> = decision.chosen.iter().map(ToString::to_string).collect();
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
        Command::Ask {
            task,
            k,
            max_new_tokens,
        } => {
            #[cfg(feature = "models")]
            {
                let store = connect(&cli.url).await?;
                let embedder = make_embedder()?;
                let task_vec = embedder.embed(&task).await?;
                let experts = expert::list(&store).await?;
                let boundaries = boundary::list(&store).await?;
                let decision =
                    gate_route(&task_vec, &experts, &boundaries, k, &GateConfig::default());
                if decision.escalate {
                    println!(
                        "decision: ESCALATE to flagship (coverage {:.3}); no in-scope expert",
                        decision.coverage
                    );
                } else {
                    let chosen = &decision.chosen[0];
                    let expert = experts
                        .iter()
                        .find(|e| &e.id == chosen)
                        .ok_or_else(|| anyhow::anyhow!("routed expert {chosen} not found"))?;
                    println!(
                        "routing to {} (adapter {})",
                        expert.name, expert.artifact_uri
                    );
                    let cfg = RaftConfig {
                        max_new_tokens,
                        ..RaftConfig::default()
                    };
                    let serve = CandleServe::new(
                        expert.base_model.clone(),
                        Some(expert.artifact_uri.clone()),
                        cfg,
                    );
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
                let _ = (&task, k, max_new_tokens);
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
                        let context_text = format!("{behavior} {fail_context}");
                        let context_vec = embedder.embed(&context_text).await?;
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
                    ModelLoader::load(&loader, &base_model, Some(adapter.as_str())).await?;
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
                    "pass-rate {:.2} ({}/{}) — adapter {} on {} ({} tasks)",
                    out.pass_rate,
                    out.passed,
                    out.total,
                    adapter,
                    corpus,
                    tasks.len()
                );
            }
            #[cfg(not(feature = "models"))]
            {
                let _ = (&corpus, &adapter, &base_model, samples, max_new_tokens);
                anyhow::bail!("`eval` requires building with --features models (candle + a GPU)");
            }
        }
    }
    Ok(())
}
