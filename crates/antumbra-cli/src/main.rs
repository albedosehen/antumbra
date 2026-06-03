//! Antumbra operator CLI (`antumbra`). v0 surface: apply the schema, print the
//! generated DDL, inspect the population, and drive the generational loop.
//!
//! Until the real candle/llama engines land (antumbra-train / antumbra-serve),
//! `loop` runs with the demo trainer + embedder so the ADR-0008 machine is
//! exercisable end-to-end.

use antumbra_core::ports::Embedder;
use antumbra_core::testing::{FixedEmbedder, ScriptedTrainer};
use antumbra_core::{RunId, ShadowStatus};
use antumbra_gate::{route as gate_route, GateConfig};
use antumbra_loop::{GenerationLoop, LoopConfig};
use antumbra_store::repo::{boundary, expert, shadow};
use antumbra_store::{schema, ConnectionConfig, Store, EMBED_DIM};
use clap::{Parser, Subcommand};

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
    /// Route a task through the boundary-conditioned gate (demo embedder).
    Route {
        /// The task description to embed and route.
        task: String,
        #[arg(long, default_value_t = 2)]
        k: usize,
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
        Command::Route { task, k } => {
            let store = connect(&cli.url).await?;
            let task_vec = FixedEmbedder::new(EMBED_DIM).embed(&task).await?;
            let experts = expert::list(&store).await?;
            let boundaries = boundary::list(&store).await?;
            let decision = gate_route(&task_vec, &experts, &boundaries, k, &GateConfig::default());
            if decision.escalate {
                println!("decision: ESCALATE to flagship (no in-scope expert)");
            } else {
                let names: Vec<String> = decision.chosen.iter().map(ToString::to_string).collect();
                println!("decision: route to [{}]", names.join(", "));
            }
            for scored in decision.ranked.iter().take(k.max(3)) {
                println!("  {:<18} score={:.3}", scored.id.to_string(), scored.score);
            }
        }
        Command::Train {
            corpus,
            generations,
            run,
        } => {
            #[cfg(feature = "models")]
            {
                use antumbra_train::{CandleModelLoader, JsonCorpus, RaftConfig, RaftTrainer};

                let store = connect(&cli.url).await?;
                let raft_cfg = RaftConfig::default();
                let loader = CandleModelLoader::new(raft_cfg.clone());
                let corpus = JsonCorpus::from_file(&corpus)?;
                let verifier = std::sync::Arc::new(antumbra_critic::CommandVerifier);
                let trainer = RaftTrainer::new(raft_cfg, loader, corpus, verifier);
                let embedder = FixedEmbedder::new(EMBED_DIM);
                let loop_cfg = LoopConfig {
                    graduate_threshold: 0.3,
                    base_model: "Qwen/Qwen2.5-Coder-1.5B".into(),
                    max_steps: 8,
                };
                let lp = GenerationLoop::new(&store, &trainer, &embedder, loop_cfg);
                let reports = lp.run_until(&RunId::new(run), generations).await?;
                for r in &reports {
                    println!(
                        "gen {:<3} shadow {:<16} fitness={:.2} graduated={}",
                        r.generation.0, r.shadow, r.fitness, r.graduated
                    );
                }
                println!("population: {} experts", expert::list(&store).await?.len());
            }
            #[cfg(not(feature = "models"))]
            {
                let _ = (&corpus, generations, &run);
                anyhow::bail!("`train` requires building with --features models (candle + a GPU)");
            }
        }
    }
    Ok(())
}
