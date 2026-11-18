//! Antumbra CLI argument surface: the clap `Cli` / `Command` definitions,
//! kept out of `main.rs`. Handlers live in `main.rs` + `ops.rs`.

use clap::{Parser, Subcommand};

use crate::claude::surface_args::SurfaceArgs;

#[derive(Parser)]
#[command(name = "antumbra", about = "Antumbra operator CLI", version)]
pub struct Cli {
    /// SurrealDB url. Defaults to a persistent on-disk store, so state carries
    /// across commands. Use `mem://` for an ephemeral run, or
    /// `ws://host:8000/rpc` for a remote server. (`antumbra tui` reads the
    /// Antumbra server instead.)
    #[arg(
        long,
        global = true,
        env = "ANTUMBRA_DB_URL",
        default_value = "surrealkv://./data/antumbra.skv"
    )]
    pub url: String,
    #[command(flatten)]
    pub db: antumbra_store::args::DbAuthArgs,
    #[command(flatten)]
    pub embedder: antumbra_embed::args::EmbedderArgs,
    #[command(flatten)]
    pub copal: antumbra_copal::args::CopalArgs,
    #[command(subcommand)]
    pub command: Command,
}

/// What `antumbra claude` can do.
#[derive(Subcommand)]
pub enum ClaudeAction {
    /// Say whether this Claude Code session is in sovereign mode (its telemetry
    /// is off, which also turns off its feature flags), list what that costs,
    /// and check the settings that bring some of it back. Exits non-zero when a
    /// required setting is missing. Also says whether each Antumbra hook the
    /// settings run is the one this build ships. Reads settings; never writes
    /// them.
    Doctor {
        /// The project to judge (its settings files, and whether its AGENTS.md
        /// is being read). Defaults to the current directory.
        #[arg(long)]
        dir: Option<std::path::PathBuf>,
    },
    /// Get a project's AGENTS.md read again in sovereign mode. Beside each
    /// AGENTS.md the agent would have read, writes a CLAUDE.local.md that imports
    /// it, and lists that file in the clone's own `.git/info/exclude`: nothing
    /// the repository tracks is changed, so it is safe in a checkout you do not
    /// own. Leaves alone any directory that has instructions of its own, where
    /// the agent was never going to read AGENTS.md.
    Bridge {
        /// The project to bridge. Defaults to the current directory.
        #[arg(long)]
        dir: Option<std::path::PathBuf>,
        /// Say what would be done and write nothing.
        #[arg(long, default_value_t = false)]
        dry_run: bool,
        /// Take the bridges back out. Only a CLAUDE.local.md that is a bridge
        /// and nothing else is deleted.
        #[arg(long, default_value_t = false)]
        remove: bool,
    },
    /// Print what an agent should know about the session it is starting in: a
    /// few lines, for a session-start hook to pass on. Prints nothing outside
    /// sovereign mode. Reads settings; writes nothing; never starts the agent.
    Brief {
        /// The project the session starts in. Defaults to the current directory.
        #[arg(long)]
        dir: Option<std::path::PathBuf>,
    },
    /// Record which of your repositories depend on which, from their manifests
    /// (package.json, Cargo.toml, go.mod, pyproject.toml, requirements*.txt),
    /// as `declared` edges in the workspace's dependency graph, for
    /// `blast_radius`. A repository depends on another when one of its
    /// manifests names a package the other publishes; only the repositories
    /// read together are matched. Recording an edge again reinforces it.
    Dependencies {
        /// A repository to read. Defaults to the current directory.
        #[arg(long)]
        dir: Option<std::path::PathBuf>,
        /// A directory whose child directories are your repositories, read
        /// alongside `dir`.
        #[arg(long)]
        repos: Option<std::path::PathBuf>,
        #[command(flatten)]
        surface: SurfaceArgs,
        /// Say which edges were found and record nothing.
        #[arg(long, default_value_t = false)]
        dry_run: bool,
    },
    /// Keep the rules as memories too, in a `claude-code` compartment of your
    /// own, so an agent can recall why a feature is missing. Written through the
    /// same surface an agent writes through. Volatile, so they never train an
    /// expert. Safe to run again: a rule already there is kept, a changed one is
    /// stored and the old text penalized, and nothing is deleted.
    Remember {
        #[command(flatten)]
        surface: SurfaceArgs,
        /// Say what would be done and write nothing.
        #[arg(long, default_value_t = false)]
        dry_run: bool,
    },
    /// Draft `autoMode.environment`, the prose that tells the agent's classifier
    /// what is inside your boundary, so that pushing to your own repository is
    /// not taken for exfiltration. The agent's own `/auto-mode-setup` is gone in
    /// sovereign mode, and drafts from your session transcripts; this drafts
    /// from your working trees' remotes and from what Antumbra remembers, and
    /// reads no transcript. An owner is proposed only when you push there over
    /// ssh and it is plainly yours; the rest are listed with the reason they
    /// were left out. Prints the block. Never writes it.
    AutoModeEnv {
        /// The project. Defaults to the current directory.
        #[arg(long)]
        dir: Option<std::path::PathBuf>,
        /// A directory whose child directories are your repositories.
        #[arg(long)]
        repos: Option<std::path::PathBuf>,
        #[command(flatten)]
        surface: SurfaceArgs,
        /// How many memories to offer for each slot.
        #[arg(long, default_value_t = 4)]
        each: u32,
    },
    /// Write the `env` lines the doctor asks for into your own settings file,
    /// and nothing else. Only names the matrix knows, only ones the file does
    /// not already set, and never anything under `permissions` (not even
    /// `defaultMode`, which the doctor asks for and this still leaves to you).
    /// Backs the file up first, then re-reads it and restores the backup unless
    /// the result is exactly what was there plus those names. Your key order and
    /// formatting are left alone: the edit is textual, not a reserialization.
    Apply {
        /// The project whose settings are read to decide what is missing.
        /// Defaults to the current directory.
        #[arg(long)]
        dir: Option<std::path::PathBuf>,
        /// The file to write. Defaults to your own `~/.claude/settings.json`,
        /// which is the one the agent reads these from.
        #[arg(long)]
        file: Option<std::path::PathBuf>,
        /// Say what would be written, check it, and write nothing.
        #[arg(long, default_value_t = false)]
        dry_run: bool,
    },
    /// Count one use of a skill. Meant for two hooks, because a skill is used
    /// two ways and each hook sees only one: `PostToolUse` matching `Skill`
    /// (the agent called it) and `UserPromptExpansion` (you typed `/name`).
    /// With no --name it reads the hook's input on standard input, and then it
    /// never fails and says nothing: a counter must not be able to stop a
    /// session. The count is one volatile memory per skill in your
    /// `claude-code` compartment, reinforced on each use.
    SkillUsed {
        /// The skill, when not run as a hook.
        #[arg(long)]
        name: Option<String>,
        #[command(flatten)]
        surface: SurfaceArgs,
    },
    /// Which of your skills are used, how often, and when last: the ones never
    /// used come first. Reads the skills installed for you and for the project,
    /// and the counters `skill-used` keeps. The agent's own `/skill-doctor` is
    /// gone in sovereign mode.
    Skills {
        /// The project. Defaults to the current directory.
        #[arg(long)]
        dir: Option<std::path::PathBuf>,
        /// Call out a skill not used in this many days.
        #[arg(long, default_value_t = 30, value_parser = clap::value_parser!(i64).range(0..=36_500))]
        days: i64,
        #[command(flatten)]
        surface: SurfaceArgs,
    },
    /// Move the memories of merged branches onto the branch each merged into,
    /// so recall from there counts them in scope. Reads the repository's merged
    /// pull requests with the GitHub CLI (`gh`, signed in) and reports each to
    /// the surface, which is what the GitHub webhook does for a server GitHub
    /// can reach. Safe to run again: memories already moved are left alone.
    Reanchor {
        /// The repository, by its working tree. Defaults to the current directory.
        #[arg(long)]
        dir: Option<std::path::PathBuf>,
        /// How many of the most recent merged pull requests to report.
        #[arg(long, default_value_t = 30)]
        limit: u32,
        /// Only the merges of the last this many days. The session-start hook
        /// passes a few, so each session reports only what is recent.
        #[arg(long, value_parser = clap::value_parser!(i64).range(0..=36_500))]
        days: Option<i64>,
        /// Say which merges would be reported and write nothing.
        #[arg(long, default_value_t = false)]
        dry_run: bool,
        #[command(flatten)]
        surface: SurfaceArgs,
    },
    /// Check an MCP server's tools for input schemas the API refuses. In
    /// sovereign mode the agent no longer leaves such a tool out, so one of them
    /// fails every request with a 400 that names it only by position. Run this
    /// from outside the agent: it is the way back in. Prints the deny rules
    /// that fix it, and exits non-zero when any tool would break requests.
    /// Also names tools the agent drops without a word (a combinator at the
    /// schema's root). Give a saved `tools/list` answer with --from, or the
    /// server's own command after `--`.
    McpLint {
        /// The server's name as the agent knows it, for the deny rules.
        #[arg(long)]
        server: String,
        /// A saved `tools/list` answer (`-` for standard input).
        #[arg(long, conflicts_with = "command")]
        from: Option<std::path::PathBuf>,
        /// How long to wait for each answer from a server started here.
        #[arg(long, default_value_t = 30)]
        timeout_secs: u64,
        /// The stdio server's command, as the agent's configuration has it.
        #[arg(last = true)]
        command: Vec<String>,
    },
}

#[derive(Subcommand)]
pub enum McpAction {
    /// Carry an agent's MCP connection to the server set up on this
    /// machine, over standard input and output. The token is read from its
    /// file for each request, so it is in no agent's config; a plain
    /// http:// address works where an agent would refuse one; and a server
    /// that restarted is given a new session without the agent noticing.
    Bridge(crate::agents::bridge::BridgeArgs),
}

#[derive(Subcommand)]
pub enum Command {
    /// Set Antumbra up for your coding agents in one command: on this machine
    /// (`setup local`), against a hosted workspace (`setup hosted <url>`),
    /// or check what is there (`setup check`). With no mode, it asks.
    Setup {
        #[command(subcommand)]
        mode: Option<crate::setup::SetupMode>,
    },
    /// Claude Code with its telemetry off: what that silently costs,
    /// and what is done about it.
    Claude {
        #[command(subcommand)]
        action: ClaudeAction,
    },
    /// Answer one hook of a coding agent other than Claude Code, in that
    /// agent's own shape, with the hook scripts Claude Code runs: what a
    /// session opens with, what is recalled for a prompt, and what is kept
    /// when a turn ends. `antumbra setup` writes the hooks that run it. It
    /// never fails the agent: whatever goes wrong is said on standard error,
    /// and nothing is printed.
    Hook(crate::agents::hook::HookArgs),
    /// The server's MCP endpoint for an agent that runs a command: Gemini
    /// CLI, Codex, OpenCode, goose and Copilot CLI are wired to it by
    /// `antumbra setup`.
    Mcp {
        #[command(subcommand)]
        action: McpAction,
    },
    /// The operator console: a live view of the population, the memory and
    /// the loop, read from the Antumbra server with the hooks' address and
    /// token. `antumbra --tui` is the same. `--demo` shows a seeded
    /// population with no server.
    Tui(antumbra_tui::Args),
    /// The verifier namespace: propose a check, measure it
    /// against ground truth the loop did not produce, and move it. Only a
    /// sound measurement lets a synthesized verifier grant reward.
    Verifier {
        #[command(subcommand)]
        action: crate::verifier_args::VerifierAction,
    },
    /// The margin gate's risk-coverage curve over a store's population: every
    /// held-out task routed with the threshold out of the way, its expert and
    /// the base both scored, and the threshold swept offline. The best point
    /// is the bar a typed gate must clear. Needs --features models and a GPU.
    GateSweep(crate::gate_sweep::SweepArgs),
    /// Record each live task's clear winner over a store's population, for
    /// `gate-train` to learn where tasks should go: every routable expert
    /// scored on every live task. Needs --features models and a GPU.
    GateOutcomes(crate::gate_outcomes::OutcomeArgs),
    /// A critic trained on the verifier's verdicts: train one
    /// and read it against them, per skill.
    Critic {
        #[command(subcommand)]
        action: crate::critic_cmd::CriticAction,
    },
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
        /// Acknowledge that this drives the loop with the scripted DEMO trainer
        /// (it always graduates; no model is trained). Required, so the demo is
        /// never mistaken for training: that is `train` under `--features models`.
        #[arg(long, default_value_t = false)]
        demo: bool,
    },
    /// Route a task through the boundary-conditioned gate. Embeds the task with
    /// `--embedder-url` when set, else the built-in embedder (`--features models`).
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
    Ask(crate::models::AskArgs),
    /// Resident multi-adapter server with hardware-adaptive serving: load the shared base ONCE and
    /// hot-swap each routed expert's adapter per prompt, instead of cold-loading
    /// a model per call. Reads prompts from stdin (one per line) or a single
    /// --task, routes each via the learned router, and serves the answer from the
    /// resident engine, so a stream of prompts pays the base load only once.
    /// Needs --features models. A build with no GPU backend serves on the CPU,
    /// from a Q8_0 base.
    Serve(crate::commands::ServeArgs),
    /// Collector/sync: bidirectionally reconcile this local store (the
    /// global `--url`, an embedded penumbra) with a remote authoritative
    /// SurrealDB, last-write-wins by each row's version timestamp. Runs on a
    /// cadence until ctrl-c, or `--once` for a single pass. Replicates the
    /// penumbra tables (memory, edges, compartments, grants, devices, genesis
    /// requests) and each expert's row and status across the fleet. An
    /// expert's adapter is a file and is not carried: a server pulls it for
    /// itself, from the server that trained the expert or from the archive.
    /// A store a server holds open cannot be opened here: such a server runs
    /// this collector itself, with `--sync-remote`.
    Sync {
        /// Remote authoritative SurrealDB url, e.g. `ws://host:8000/rpc`.
        #[arg(long)]
        remote: String,
        /// Root username for the remote (omit for an unauthenticated remote).
        #[arg(long)]
        remote_user: Option<String>,
        /// Root password for the remote.
        #[arg(long)]
        remote_pass: Option<String>,
        /// Seconds between reconcile cycles.
        #[arg(long, default_value_t = 15)]
        interval: u64,
        /// Reconcile once and exit, instead of running continuously.
        #[arg(long, default_value_t = false)]
        once: bool,
    },
    /// Recover a failure boundary's scope by generate-then-verify (the counterfactual boundary of competence):
    /// hold a behavior fixed, vary the context, and find the governing feature
    /// and C' by actually serving and checking. Stores an actionable boundary.
    /// Needs --features models + a GPU + python.
    Scope(crate::models::ScopeArgs),
    /// Train shadows with the real candle trainer (needs --features models + GPU).
    Train(crate::train_args::TrainArgs),
    /// Score a saved adapter's pass-rate on a corpus, with no training.
    /// Needs --features models + a GPU + python.
    Eval(crate::eval_args::EvalArgs),
    /// Capture a supplied, verifier-checked correction into a frozen expert (the
    /// other intake path beside `train`, the capture intake into the umbra). The corpus carries a
    /// `completion` per task. Needs --features models + a GPU + python.
    Teach(crate::models::TeachArgs),
    /// Train the learned router over the population's exemplars (the learned gate): a
    /// per-dimension metric that separates specialists from generalists where
    /// raw-cosine routing cannot. Retrain after the population changes. Needs
    /// --features models (the real embedder).
    GateTrain {
        #[arg(long, default_value_t = 400)]
        epochs: usize,
        /// Train on the live tasks each expert clearly won as well as the
        /// capability exemplars (`gate-outcomes` records them).
        /// Off by default: the outcome-trained router scored no higher on the
        /// withheld tasks.
        #[arg(long)]
        with_outcomes: bool,
    },
    /// Compose several experts into one blended adapter and serve a task
    /// through it (the heterogeneous composed model): the population as a capability multiplier. Needs
    /// --features models + a GPU.
    Compose {
        /// The task to answer with the blended experts.
        task: String,
        /// Expert blend as `name:weight,...` (e.g. `bunexpert-g0:0.6,generaldeps-g0:0.4`).
        #[arg(long)]
        experts: String,
        #[arg(long, default_value_t = 24)]
        max_new_tokens: usize,
    },
    /// Autonomous self-improvement: train -> evaluate -> if below target, keep
    /// training (warm-started from the prior adapter) until it passes or the
    /// generation budget runs out. The system improves itself to a quality bar,
    /// eval-gated, with no manual per-step driving. Needs --features models + GPU.
    Evolve(crate::commands::EvolveArgs),
    /// Autonomous population growth: route each task across the population, and
    /// grow a NEW specialist for the cluster the gate cannot cover, until
    /// coverage meets target or the expert budget runs out. Additive: existing
    /// experts are kept. Needs --features models + a GPU.
    Populate(crate::commands::PopulateArgs),
    /// Bootstrap from an existing memory corpus (qdrant, surrealdb, a json
    /// file): adapt a normalized memory export into capture tasks and learn
    /// from them, instead of discovering every skill cold via RAFT. Reinforced
    /// memories are captured (trusted on import); weak ones become RAFT seeds.
    /// Writes the converted corpus; with --train, internalizes the captures.
    /// Needs --features models (+ a GPU + python only when --train).
    MemoryImport(crate::commands::MemoryImportArgs),
    /// Consolidation: score a normalized memory export against the
    /// graduation gate (recurrence x verifiability x stability), graduate the
    /// survivors into per-skill specialists rehearsing already-consolidated
    /// skills (replay, the catastrophic-interference fix), and append them to
    /// the consolidated log. The store keeps everything that does not graduate
    /// (volatile / episodic / under-reinforced). Needs --features models.
    Consolidate(crate::ops::ConsolidateArgs),
    /// Retire an expert: population-level forgetting that demotes rather than
    /// deletes. The expert goes dormant, masked from the gate
    /// but kept, and still served when named; `--archive` takes it out of
    /// serving too. Its adapter stays on disk and `revive` brings it back.
    /// Wire a store's contradiction report against a consolidated memory to
    /// this to undo a graduation (a frozen LoRA cannot be edited per-fact).
    /// The router masks it at once; with --features models it is retrained
    /// over the experts that remain.
    Retire {
        /// The expert, by name or id.
        #[arg(long)]
        expert: String,
        /// Archive it: out of serving as well as routing.
        #[arg(long)]
        archive: bool,
        /// Why, recorded with the move.
        #[arg(long)]
        note: Option<String>,
    },
    /// Bring a dormant or archived expert back into the population the gate
    /// routes over. A deleted expert cannot be revived.
    Revive {
        /// The expert, by name or id.
        #[arg(long)]
        expert: String,
        /// Why, recorded with the move.
        #[arg(long)]
        note: Option<String>,
    },
    /// End a token before it expires: the one saved on this machine, a token
    /// file, or an id. Every networked server refuses it within half a minute.
    /// For a token on a lost or shared machine; rotating the signing key ends
    /// every token at once instead.
    RevokeToken(crate::ops::RevokeTokenArgs),
    /// Seed a memory into a user's compartment from the CLI (the owner/admin
    /// path; agents write via the MCP `store_memory` tool). No embedding is
    /// attached; consolidation gathers a compartment by membership. Pair with
    /// `consolidate-compartment` to mint the compartment into a private expert.
    Remember(crate::ops::RememberArgs),
    /// Ingest a knowledge document for `recall_documents`: chunk, embed, store,
    /// from a file or from what a command prints. The command form is the
    /// parser-free answer to an inventory question: run the framework's own
    /// lister (`deno task routes`, an OpenAPI export, `cargo metadata`) and keep
    /// its output, stamped with the repository, commit, and branch it ran at, so
    /// a recalled chunk says which commit it describes.
    Ingest(crate::ops::IngestArgs),
    /// Derive facts the repository's history already states, from `git log`
    /// with no parser: ownership per top-level area, change hotspots, and files
    /// that change together, over a window. Each fact is stored as a `world`
    /// memory whose evidence names the commit range it came from, so a later
    /// session sees how old it is instead of trusting a refreshed-or-not index.
    GitFacts(crate::ops::FactsArgs),
    /// Train a user's accepted behaviors into their private standing expert.
    /// Each behavior's worked examples are the tasks, a quarter held out. The
    /// base model's own answers to everyday prompts are replayed beside them,
    /// so the expert keeps what the base does elsewhere. The
    /// expert is minted only if every behavior clearly rose on its held-out
    /// tasks and the controls held. Needs --features models + a GPU.
    Behave(crate::behave::BehaveArgs),
    /// Metabolize orchestration traces into the frozen-expert population: adapt a
    /// normalized trace export (loop runs, behavior-graph evaluations, task
    /// executions) into capture tasks the population internalizes, so the brain
    /// learns to do in one shot what it took many steps to do. Only successful,
    /// recurrent traces metabolize. Writes the converted corpus; with --train,
    /// internalizes it. Needs --features models.
    Metabolize(crate::commands::MetabolizeArgs),
    /// Have the antumbra propose memory compartments by clustering a user's
    /// unorganized memory (their inbox compartment + anything they authored
    /// uncompartmented) into competence-coherent regions. Needs no model: it
    /// clusters the embeddings already stored on each memory. With --apply it
    /// creates each proposal as an `Origin::Proposed` compartment and moves its
    /// members in (reversible by deleting the compartment).
    ProposeCompartments(crate::ops::ProposeCompartmentsArgs),
    /// Set a workspace's embedder endpoint (hosted multi-tenant): the
    /// OpenAI-compatible `/embeddings` URL + model it embeds with. Without
    /// `--source-dim` the endpoint must produce the index dimension; with it,
    /// the endpoint is a Matryoshka model returning longer vectors whose
    /// renormalized 384-d prefix is stored. Applies to sessions built after;
    /// reconnect to apply to an active one, and `reembed` if the model changed.
    SetEmbedder {
        #[arg(long)]
        tenant: String,
        /// The OpenAI-compatible `/embeddings` endpoint URL for this workspace.
        #[arg(long)]
        endpoint: String,
        #[arg(long)]
        model: String,
        /// Optional bearer key for the endpoint.
        #[arg(long)]
        key: Option<String>,
        /// Matryoshka source dimension: the width the endpoint returns (e.g.
        /// `1024` for BGE-M3 / multilingual-e5). When set, the renormalized
        /// leading 384-d prefix is stored into the fixed HNSW index. Omit for a
        /// strict embedder that already returns the index dimension.
        #[arg(long)]
        source_dim: Option<u32>,
    },
    /// Show a workspace's configured embedder (or that it uses the server default).
    GetEmbedder {
        #[arg(long)]
        tenant: String,
    },
    /// Re-embed all of a workspace's memories with its configured embedder. Run
    /// after changing the workspace's embedder model so recall stays coherent.
    Reembed {
        #[arg(long)]
        tenant: String,
        /// Report what would be re-embedded and exit without writing anything.
        #[arg(long)]
        dry_run: bool,
        /// Skip the confirmation prompt (for non-interactive / scripted use).
        #[arg(long)]
        yes: bool,
    },
}
