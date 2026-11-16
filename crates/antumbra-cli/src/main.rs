// Test code may unwrap until its sweep lands (#295 to #302); production code may not.
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]
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
use antumbra_gate::{route as gate_route, GateConfig};
use antumbra_loop::{GenerationLoop, LoopConfig};
use antumbra_store::repo::{boundary, expert, lifecycle, shadow};
use antumbra_store::{schema, Store, EMBED_DIM};
use chrono::Utc;
use clap::Parser;

mod behave;
mod capture_args;
mod claude;
mod cli;
mod commands;
mod critic_cmd;
mod eval_args;
mod gate_outcomes;
mod gate_sweep;
#[cfg(feature = "models")]
mod generation_report;
mod gitctx;
mod gitfacts;
#[cfg(feature = "models")]
mod learned_gate;
mod machine;
mod models;
mod ops;
mod setup;
mod train_args;
mod verifier_args;
mod verifier_name;
mod verifier_synth;
mod verifiers;

use cli::{Cli, Command};
#[cfg(feature = "models")]
pub(crate) use learned_gate::{refresh_router, train_router, RouterRefresh};

#[cfg(feature = "models")]
use antumbra_core::ports::{ActRequest, Serve};
#[cfg(feature = "models")]
use antumbra_serve::CandleServe;
#[cfg(feature = "models")]
use antumbra_train::RaftConfig;

/// The DB credentials, captured once from the global flags so `connect` (called
/// from many handlers that only carry the url) can apply them without threading
/// them through every call site.
static DB_CREDS: std::sync::OnceLock<antumbra_store::args::DbAuthArgs> = std::sync::OnceLock::new();

async fn connect(url: &str) -> anyhow::Result<Store> {
    let db = DB_CREDS.get().cloned().unwrap_or_default();
    Ok(db.connect(url, EMBED_DIM).await?)
}

/// The embedder flags, captured once (like `DB_CREDS`) so the many handlers
/// that embed can call `make_embedder` without threading them through every
/// call site.
static EMBEDDER: std::sync::OnceLock<antumbra_embed::args::EmbedderArgs> =
    std::sync::OnceLock::new();

/// The copal document-of-record archive from the global `--copal-*` flags,
/// resolved once like the embedder: `None` when no address is configured.
static COPAL: std::sync::OnceLock<Option<std::sync::Arc<CopalArchive>>> =
    std::sync::OnceLock::new();

fn copal_archive() -> Option<std::sync::Arc<CopalArchive>> {
    COPAL.get().cloned().flatten()
}

/// The active embedder: what the flags name outright (a configured
/// `--embedder-url` always wins, the stand-in only when `--fake-embedder` asks
/// for it), else the real candle BERT under `--features models`, and a refusal
/// without `models`. Every path produces `EMBED_DIM`-wide vectors so the gate
/// and store stay dimension-consistent.
fn make_embedder() -> anyhow::Result<Box<dyn Embedder>> {
    let flags = EMBEDDER
        .get()
        .ok_or_else(|| anyhow::anyhow!("embedder flags were not captured before a handler ran"))?;
    match flags.configured() {
        Some(embedder) => Ok(embedder),
        None => builtin_embedder(),
    }
}

#[cfg(feature = "models")]
fn builtin_embedder() -> anyhow::Result<Box<dyn Embedder>> {
    Ok(Box::new(antumbra_serve::BertEmbedder::load()?))
}

#[cfg(not(feature = "models"))]
fn builtin_embedder() -> anyhow::Result<Box<dyn Embedder>> {
    Err(antumbra_embed::args::NoEmbedder.into())
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

/// The stack of each runtime worker thread: room for the embedded SurrealDB 4
/// engine, which overflows the 2 MiB default (SurrealDB recommends 10 MiB).
const WORKER_STACK: usize = 16 * 1024 * 1024;

fn main() -> anyhow::Result<()> {
    // SurrealDB query evaluation (the engine-enforced ACL subqueries for tenant
    // isolation and memory compartments) recurses deep; the OS default main-thread stack (1 MB on Windows)
    // overflows. Host the runtime on a thread with a large stack. The embedded
    // SurrealDB 4 engine also runs queries on the runtime's worker threads,
    // whose 2 MiB default overflows on it; SurrealDB asks for 10 MiB there.
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(|| {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .thread_stack_size(WORKER_STACK)
                .build()?
                .block_on(run())
        })?
        .join()
        .map_err(|_| anyhow::anyhow!("antumbra worker thread panicked"))?
}

/// `antumbra --tui ...` is `antumbra tui ...`. Only a leading `--tui` is read
/// so; anywhere else it is an argument like any other.
fn tui_alias(args: impl Iterator<Item = std::ffi::OsString>) -> Vec<std::ffi::OsString> {
    let mut args: Vec<_> = args.collect();
    if args.get(1).is_some_and(|a| a == "--tui") {
        args[1] = "tui".into();
    }
    args
}

async fn run() -> anyhow::Result<()> {
    let cli = Cli::parse_from(tui_alias(std::env::args_os()));
    let _ = DB_CREDS.set(cli.db.clone());
    let _ = EMBEDDER.set(cli.embedder.clone());
    // The CLI ingests documents; it publishes no relics, so it needs no
    // render origin.
    let _ = COPAL.set(cli.copal.archive(None)?);
    match cli.command {
        Command::Setup { mode } => {
            // Files, Docker, HTTP and prompts: all blocking, so off the runtime.
            tokio::task::spawn_blocking(move || setup::run(mode)).await??;
        }
        Command::Claude { action } => {
            // Blocking work (files, git, an HTTP surface): keep it off the runtime.
            tokio::task::spawn_blocking(move || claude::run::run(action)).await??;
        }
        Command::Tui(args) => antumbra_tui::run(args).await?,
        Command::Verifier { action } => verifiers::run(&cli.url, action).await?,
        Command::Critic { action } => critic_cmd::run(action).await?,
        Command::GateSweep(args) => gate_sweep::run(&cli.url, args).await?,
        Command::GateOutcomes(args) => gate_outcomes::run(&cli.url, args).await?,
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
            let experts = lifecycle::population(&store).await?;
            if experts.is_empty() {
                println!("(no experts yet)");
            }
            for (e, status) in experts {
                println!(
                    "{:<20} {:<8} fitness={:.2}  base={}  frozen={}",
                    e.name,
                    status.as_str(),
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
                    // A `mem://` artifact is on nobody's disk, so it is placed
                    // nowhere and servable anywhere, which is what a demo wants.
                    placed_on: None,
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
            if let Some(router) = lifecycle::load_router(&store).await? {
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
                let experts = lifecycle::routable(&store).await?;
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
        Command::Ask(args) => models::ask(&cli.url, args).await?,
        Command::Scope(args) => models::scope(&cli.url, args).await?,
        Command::Train(args) => {
            models::train(&cli.url, args).await?;
        }
        Command::Eval(args) => models::eval(args).await?,
        Command::Teach(args) => models::teach(&cli.url, args).await?,

        Command::GateTrain {
            epochs,
            with_outcomes,
        } => {
            #[cfg(feature = "models")]
            {
                let store = connect(&cli.url).await?;
                let embedder = make_embedder()?;
                match train_router(&store, embedder.as_ref(), epochs, with_outcomes).await? {
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
                let _ = (epochs, with_outcomes);
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
                // Named experts are served whether or not the gate routes to
                // them: a dormant one still composes, an archived one does not.
                let population = lifecycle::servable(&store).await?;
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
                        weights: Vec::new(),
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
        Command::Evolve(args) => commands::evolve(&cli.url, args).await?,
        Command::Serve(args) => commands::serve(&cli.url, args).await?,
        Command::Populate(args) => commands::populate(&cli.url, args).await?,
        Command::MemoryImport(args) => commands::memory_import(&cli.url, args).await?,
        Command::Consolidate(args) => ops::consolidate(&cli.url, args).await?,
        Command::Behave(args) => behave::run(&cli.url, args).await?,
        Command::Retire {
            expert,
            archive,
            note,
        } => {
            let to = if archive {
                antumbra_core::ExpertStatus::Archived
            } else {
                antumbra_core::ExpertStatus::Dormant
            };
            ops::move_expert(&cli.url, &expert, to, note).await?;
        }
        Command::Revive { expert, note } => {
            ops::move_expert(&cli.url, &expert, antumbra_core::ExpertStatus::Active, note).await?;
        }
        Command::Remember(args) => ops::remember(&cli.url, args).await?,
        Command::Ingest(args) => {
            ops::ingest(&cli.url, copal_archive().as_deref(), args).await?;
        }
        Command::GitFacts(args) => ops::git_facts(&cli.url, args).await?,
        Command::Metabolize(args) => commands::metabolize(&cli.url, args).await?,
        Command::ProposeCompartments(args) => ops::propose_compartments(&cli.url, args).await?,
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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn parsed(argv: &[&str]) -> Command {
        let argv = argv.iter().map(std::ffi::OsString::from);
        Cli::try_parse_from(tui_alias(argv)).unwrap().command
    }

    // The store flags are the shared ones (#303): the password's environment
    // value is hidden in --help, the url reads its variable, and both reach
    // every subcommand, as in every binary.
    #[test]
    fn the_store_flags_read_as_in_every_binary() {
        use clap::CommandFactory;
        let cmd = Cli::command();
        cmd.clone().debug_assert();
        let flag = |id: &str| {
            cmd.get_arguments().find(|a| a.get_id() == id).map(|a| {
                (
                    a.get_env().map(|e| e.to_string_lossy().into_owned()),
                    a.is_hide_env_values_set(),
                    a.is_global_set(),
                )
            })
        };
        assert_eq!(
            flag("db_pass"),
            Some((Some("ANTUMBRA_DB_PASS".into()), true, true))
        );
        assert_eq!(
            flag("url"),
            Some((Some("ANTUMBRA_DB_URL".into()), false, true))
        );
        assert_eq!(
            flag("db_auth").map(|f| f.0),
            Some(Some("ANTUMBRA_DB_AUTH".into()))
        );
    }

    /// What clap says against `argv`, or nothing when it parses.
    fn refusal(argv: &[&str]) -> String {
        let argv = argv.iter().map(std::ffi::OsString::from);
        Cli::try_parse_from(tui_alias(argv)).map_or_else(|e| e.to_string(), |_| String::new())
    }

    // A day count is bounded at the parser, so the date arithmetic behind it
    // cannot overflow on a typo: a usage error naming the range, not a panic.
    #[test]
    fn a_day_count_past_a_century_is_a_usage_error() {
        let huge = "999999999999999";
        for argv in [
            ["antumbra", "claude", "skills", "--days", huge],
            ["antumbra", "claude", "reanchor", "--days", huge],
            ["antumbra", "verifier", "measure", "--ttl-days", huge],
        ] {
            assert!(
                refusal(&argv).contains("0..=36500"),
                "{argv:?}: {}",
                refusal(&argv)
            );
        }
        // clap reads `-1` as a flag before any range applies: refused all the same.
        assert!(!refusal(&["antumbra", "claude", "reanchor", "--days", "-1"]).is_empty());
        assert!(refusal(&["antumbra", "claude", "skills", "--days", "45"]).is_empty());
        assert!(refusal(&["antumbra", "claude", "reanchor", "--days", "3"]).is_empty());
    }

    #[test]
    fn the_console_is_tui_or_a_leading_tui_flag() {
        assert!(matches!(parsed(&["antumbra", "tui"]), Command::Tui(_)));
        assert!(matches!(
            parsed(&["antumbra", "--tui", "--demo"]),
            Command::Tui(_)
        ));
        let elsewhere = tui_alias(
            ["antumbra", "setup", "--tui"]
                .iter()
                .map(std::ffi::OsString::from),
        );
        assert_eq!(elsewhere[2], "--tui", "only a leading --tui is the console");
    }

    // The embedder and copal flags are the shared ones (#304): the model reads
    // its variable, the keys hide their environment values in --help, every
    // flag keeps its name and default and reaches every subcommand.
    #[test]
    fn the_embedder_and_copal_flags_read_as_in_every_binary() {
        use clap::CommandFactory;
        let cmd = Cli::command();
        cmd.clone().debug_assert();
        let flag = |id: &str| {
            cmd.get_arguments().find(|a| a.get_id() == id).map(|a| {
                (
                    a.get_long().map(str::to_owned),
                    a.get_env().map(|e| e.to_string_lossy().into_owned()),
                    a.is_hide_env_values_set(),
                    a.is_global_set(),
                    a.get_default_values()
                        .iter()
                        .map(|v| v.to_string_lossy().into_owned())
                        .collect::<Vec<_>>(),
                )
            })
        };
        assert_eq!(
            flag("embedder_model"),
            Some((
                Some("embedder-model".into()),
                Some("ANTUMBRA_EMBEDDER_MODEL".into()),
                false,
                true,
                vec!["all-MiniLM-L6-v2".into()],
            ))
        );
        for (id, long, env) in [
            ("embedder_url", "embedder-url", "ANTUMBRA_EMBEDDER_URL"),
            ("embedder_key", "embedder-key", "ANTUMBRA_EMBEDDER_KEY"),
            ("fake_embedder", "fake-embedder", "ANTUMBRA_FAKE_EMBEDDER"),
            ("copal_addr", "copal-addr", "ANTUMBRA_COPAL_ADDR"),
            ("copal_tenant", "copal-tenant", "ANTUMBRA_COPAL_TENANT"),
            ("copal_key", "copal-key", "ANTUMBRA_COPAL_KEY"),
            ("copal_keys", "copal-keys", "ANTUMBRA_COPAL_KEYS"),
        ] {
            let got = flag(id).unwrap_or_else(|| panic!("{id} is a flag"));
            assert_eq!(got.0.as_deref(), Some(long), "{id}");
            assert_eq!(got.1.as_deref(), Some(env), "{id}");
            assert!(got.3, "{id} is global");
            assert_eq!(
                got.2,
                id.ends_with("_key"),
                "{id} hides its value iff a key"
            );
        }
    }

    /// What a command that captures parsed: its own `--run` and the shared
    /// capture-SFT block.
    fn captured(argv: &[&str]) -> anyhow::Result<(String, capture_args::CaptureSftArgs)> {
        match parsed(argv) {
            Command::Teach(a) => Ok((a.run, a.capture)),
            Command::MemoryImport(a) => Ok((a.run, a.capture)),
            Command::Consolidate(a) => Ok((a.run, a.capture)),
            Command::Metabolize(a) => Ok((a.run, a.capture)),
            _ => anyhow::bail!("{argv:?} is not a command that captures"),
        }
    }

    // The capture-SFT block is the shared one (#308): the four commands that
    // capture parse the same defaults for it, each keeps its own `--run`, and
    // a flag given reaches the block.
    #[test]
    fn the_capture_sft_block_reads_the_same_in_every_command_that_captures() -> anyhow::Result<()> {
        use clap::CommandFactory;
        Cli::command().debug_assert();
        for (argv, run) in [
            (["antumbra", "teach", "--corpus", "c.json"], "run:teach"),
            (
                ["antumbra", "memory-import", "--source", "m.json"],
                "run:memory",
            ),
            (
                ["antumbra", "consolidate", "--source", "m.json"],
                "expert:consolidated",
            ),
            (
                ["antumbra", "metabolize", "--source", "t.json"],
                "run:metabolize",
            ),
        ] {
            let (own_run, capture) = captured(&argv)?;
            assert_eq!(own_run, run, "{argv:?}");
            assert_eq!(capture.rounds, 40, "{argv:?}");
            assert_eq!(capture.samples, 8, "{argv:?}");
            assert_eq!(capture.max_new_tokens, 32, "{argv:?}");
            assert_eq!(capture.lr, 3e-4, "{argv:?}");
        }
        let (_, capture) = captured(&[
            "antumbra",
            "consolidate",
            "--source",
            "m.json",
            "--rounds",
            "3",
            "--lr",
            "0.01",
        ])?;
        assert_eq!((capture.rounds, capture.lr), (3, 0.01));
        Ok(())
    }

    /// The `claude` commands that call the surface; `--help` says the same of
    /// `--surface` and `--token` in each.
    const SURFACE_CALLERS: [&str; 6] = [
        "dependencies",
        "remember",
        "auto-mode-env",
        "skill-used",
        "skills",
        "reanchor",
    ];
    const SURFACE_HELP: &str = "The Antumbra MCP surface, as the hooks know it";
    const TOKEN_HELP: &str =
        "The bearer token for that surface. Prefer the environment variable: a \
                              flag ends up in the shell's history";

    /// The surface pair as a `claude` command parsed it.
    fn surface_of(argv: &[&str]) -> anyhow::Result<claude::surface_args::SurfaceArgs> {
        use cli::ClaudeAction;
        let Command::Claude { action } = parsed(argv) else {
            anyhow::bail!("{argv:?} is not a claude command");
        };
        match action {
            ClaudeAction::Dependencies { surface, .. }
            | ClaudeAction::Remember { surface, .. }
            | ClaudeAction::AutoModeEnv { surface, .. }
            | ClaudeAction::SkillUsed { surface, .. }
            | ClaudeAction::Skills { surface, .. }
            | ClaudeAction::Reanchor { surface, .. } => Ok(surface),
            _ => anyhow::bail!("{argv:?} does not call the surface"),
        }
    }

    // The surface pair is the shared one (#308): the six `claude` commands
    // that call the surface read `--surface` and `--token` the same way and
    // say the same of them in --help.
    #[test]
    fn the_surface_pair_reads_the_same_in_every_claude_command_that_calls_it() -> anyhow::Result<()>
    {
        use clap::CommandFactory;
        // `--surface` reads ANTUMBRA_URL and `--token` ANTUMBRA_TOKEN, so a
        // shell that sets them parses its own values.
        let surface =
            std::env::var("ANTUMBRA_URL").unwrap_or_else(|_| "http://127.0.0.1:8081".to_owned());
        let token = std::env::var("ANTUMBRA_TOKEN").ok();
        for sub in SURFACE_CALLERS {
            let got = surface_of(&["antumbra", "claude", sub])?;
            assert_eq!(got.surface, surface, "{sub}");
            assert_eq!(got.token, token, "{sub}");
            let given = [
                "antumbra",
                "claude",
                sub,
                "--surface",
                "http://surface.test",
                "--token",
                "t",
            ];
            let got = surface_of(&given)?;
            assert_eq!(got.surface, "http://surface.test", "{sub}");
            assert_eq!(got.token.as_deref(), Some("t"), "{sub}");
        }
        let cli = Cli::command();
        let claude = cli
            .find_subcommand("claude")
            .ok_or_else(|| anyhow::anyhow!("no claude command"))?;
        for sub in SURFACE_CALLERS {
            let cmd = claude
                .find_subcommand(sub)
                .ok_or_else(|| anyhow::anyhow!("no claude {sub}"))?;
            let flag = |id: &str| {
                cmd.get_arguments()
                    .find(|a| a.get_id() == id)
                    .ok_or_else(|| anyhow::anyhow!("claude {sub} has no --{id}"))
            };
            let env = |a: &clap::Arg| a.get_env().map(|e| e.to_string_lossy().into_owned());
            let help = |a: &clap::Arg| a.get_help().map(ToString::to_string);
            let surface = flag("surface")?;
            assert_eq!(
                surface
                    .get_default_values()
                    .iter()
                    .map(|v| v.to_string_lossy().into_owned())
                    .collect::<Vec<_>>(),
                vec!["http://127.0.0.1:8081".to_owned()],
                "{sub}"
            );
            assert_eq!(env(surface), Some("ANTUMBRA_URL".to_owned()), "{sub}");
            assert_eq!(help(surface), Some(SURFACE_HELP.to_owned()), "{sub}");
            let token = flag("token")?;
            assert_eq!(env(token), Some("ANTUMBRA_TOKEN".to_owned()), "{sub}");
            assert!(token.is_hide_env_values_set(), "{sub}");
            assert_eq!(help(token), Some(TOKEN_HELP.to_owned()), "{sub}");
        }
        Ok(())
    }

    // Each subcommand the parser now hands over whole (#308) parses with only
    // its required flags, and keeps its defaults.
    #[test]
    fn every_subcommand_handed_over_whole_parses_with_its_required_flags_alone(
    ) -> anyhow::Result<()> {
        let Command::Ask(a) = parsed(&["antumbra", "ask", "add two numbers"]) else {
            anyhow::bail!("ask");
        };
        assert_eq!(
            (a.task.as_str(), a.k, a.max_new_tokens, a.with),
            ("add two numbers", 1, 128, None)
        );
        assert_eq!(
            (a.threshold, a.self_weight, a.temperature),
            (0.08, 0.4, 0.0)
        );
        let Command::Serve(a) = parsed(&["antumbra", "serve"]) else {
            anyhow::bail!("serve");
        };
        assert_eq!(
            (a.task, a.max_new_tokens, a.threshold, a.temperature),
            (None, 128, 0.08, 0.0)
        );
        let Command::Scope(a) = parsed(&["antumbra", "scope", "--spec", "s.json"]) else {
            anyhow::bail!("scope");
        };
        assert_eq!(
            (a.spec.as_str(), a.expert, a.discover),
            ("s.json", None, false)
        );
        assert_eq!(
            (a.max_new_tokens, a.samples, a.temperature, a.confidence),
            (96, 8, 0.8, 0.7)
        );
        let Command::Teach(a) = parsed(&["antumbra", "teach", "--corpus", "c.json"]) else {
            anyhow::bail!("teach");
        };
        assert_eq!(
            (a.corpus.as_str(), a.generations, a.parent),
            ("c.json", 1, None)
        );
        let Command::Evolve(a) = parsed(&["antumbra", "evolve", "--corpus", "c.json"]) else {
            anyhow::bail!("evolve");
        };
        assert_eq!(
            (a.run.as_str(), a.target, a.max_gens),
            ("run:evolve", 0.9, 3)
        );
        assert_eq!((a.samples, a.rounds, a.max_new_tokens), (6, 2, 32));
        let Command::Populate(a) = parsed(&["antumbra", "populate", "--corpus", "c.json"]) else {
            anyhow::bail!("populate");
        };
        assert_eq!(
            (a.run.as_str(), a.target_coverage, a.max_experts),
            ("grown", 0.9, 3)
        );
        assert_eq!((a.samples, a.rounds, a.max_new_tokens), (8, 3, 32));
        let Command::MemoryImport(a) = parsed(&["antumbra", "memory-import", "--source", "m.json"])
        else {
            anyhow::bail!("memory-import");
        };
        assert_eq!(
            (a.out.as_str(), a.capture_threshold, a.train),
            ("corpora/_imported.json", 0.5, false)
        );
        let Command::Consolidate(a) = parsed(&["antumbra", "consolidate", "--source", "m.json"])
        else {
            anyhow::bail!("consolidate");
        };
        assert_eq!(
            (a.log.as_str(), a.min_recurrence, a.min_confidence),
            ("corpora/_consolidated.json", 2, 0.5)
        );
        assert_eq!((a.train, a.replay_ratio, a.batch_size), (false, 0.5, 1));
        let remember = [
            "antumbra",
            "remember",
            "--tenant",
            "t",
            "--user",
            "u",
            "--compartment",
            "c",
            "--content",
            "x",
        ];
        let Command::Remember(a) = parsed(&remember) else {
            anyhow::bail!("remember");
        };
        assert_eq!((a.network.as_str(), a.confidence), ("world", 1.0));
        let ingest = [
            "antumbra", "ingest", "--tenant", "t", "--user", "u", "--title", "routes",
        ];
        let Command::Ingest(a) = parsed(&ingest) else {
            anyhow::bail!("ingest");
        };
        assert_eq!(
            (a.file, a.source, a.path, a.no_git, a.compartment),
            (None, None, None, false, None)
        );
        assert!(a.run.is_empty());
        let with_command = [&ingest[..], &["--", "deno", "task", "routes"]].concat();
        let Command::Ingest(a) = parsed(&with_command) else {
            anyhow::bail!("ingest with a command");
        };
        assert_eq!(a.run, ["deno", "task", "routes"]);
        let both = [&ingest[..], &["--file", "f", "--", "ls"]].concat();
        assert!(!refusal(&both).is_empty(), "--file and a command conflict");
        let git_facts = [
            "antumbra",
            "git-facts",
            "--tenant",
            "t",
            "--user",
            "u",
            "--compartment",
            "c",
        ];
        let Command::GitFacts(a) = parsed(&git_facts) else {
            anyhow::bail!("git-facts");
        };
        assert_eq!((a.days, a.top, a.dry_run), (90, 10, false));
        let Command::Metabolize(a) = parsed(&["antumbra", "metabolize", "--source", "t.json"])
        else {
            anyhow::bail!("metabolize");
        };
        assert_eq!(
            (a.out.as_str(), a.min_recurrence, a.no_steps, a.watch),
            ("corpora/_metabolized.json", 1, false, false)
        );
        assert_eq!((a.interval_secs, a.train), (300, false));
        let Command::ProposeCompartments(a) = parsed(&[
            "antumbra",
            "propose-compartments",
            "--tenant",
            "t",
            "--user",
            "u",
        ]) else {
            anyhow::bail!("propose-compartments");
        };
        assert_eq!(
            (a.inbox, a.similarity_threshold, a.min_size, a.apply),
            (None, 0.6, 3, false)
        );
        Ok(())
    }
}
