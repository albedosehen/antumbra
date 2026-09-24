//! Antumbra CLI argument surface: the clap `Cli` / `Command` definitions,
//! kept out of `main.rs`. Handlers live in `main.rs` + `ops.rs`.

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "antumbra", about = "Antumbra operator CLI", version)]
pub struct Cli {
    /// SurrealDB url. Defaults to a persistent on-disk store shared with the
    /// operator console (`antumbra-tui`), so state carries across commands. Use
    /// `mem://` for an ephemeral run, or `ws://host:8000/rpc` for a remote server.
    #[arg(long, global = true, default_value = "surrealkv://./data/antumbra.skv")]
    pub url: String,
    /// Root username for an authenticated remote SurrealDB (`ws://`). Omit for an
    /// embedded store or an unauthenticated server.
    #[arg(long, global = true, env = "ANTUMBRA_DB_USER")]
    pub db_user: Option<String>,
    /// Root password for the remote SurrealDB.
    #[arg(long, global = true, env = "ANTUMBRA_DB_PASS")]
    pub db_pass: Option<String>,
    /// Embed via an OpenAI-compatible `/embeddings` endpoint (for example Ollama
    /// serving `all-minilm`) instead of the built-in embedder. It must return
    /// vectors of the store's dimension. Use the SAME endpoint the population and
    /// memories were built with, or recall and routing lose coherence.
    #[arg(long, global = true, env = "ANTUMBRA_EMBEDDER_URL")]
    pub embedder_url: Option<String>,
    /// Model name sent to `--embedder-url`.
    #[arg(
        long,
        global = true,
        env = "ANTUMBRA_EMBEDDER_MODEL",
        default_value = "all-MiniLM-L6-v2"
    )]
    pub embedder_model: String,
    /// Optional bearer key for `--embedder-url`.
    #[arg(long, global = true, env = "ANTUMBRA_EMBEDDER_KEY")]
    pub embedder_key: Option<String>,
    /// Use the deterministic byte-histogram stand-in (NOT semantic: it matches
    /// character statistics), for demos and smoke tests only. Without
    /// `--features models`, a command that embeds refuses to run with neither
    /// this nor `--embedder-url` rather than silently degrading.
    #[arg(
        long,
        global = true,
        env = "ANTUMBRA_FAKE_EMBEDDER",
        default_value_t = false
    )]
    pub fake_embedder: bool,
    /// Archive each ingested document's ORIGINAL content to a copal file
    /// service (the document of record), exactly as the MCP server does: bare
    /// `host:port` or a full URL base. The upload happens before any chunk is
    /// stored (a dead copal fails the ingest), and every chunk carries the
    /// copal file id + digest. Off when unset: ingest keeps only the chunks.
    #[arg(long, global = true, env = "ANTUMBRA_COPAL_ADDR")]
    pub copal_addr: Option<String>,
    /// Land every workspace's documents under this ONE copal tenant (copal's
    /// header auth mode). Omit all three tenancy args for per-workspace
    /// tenancy, the default. At most one of --copal-tenant / --copal-key /
    /// --copal-keys.
    #[arg(long, global = true, env = "ANTUMBRA_COPAL_TENANT")]
    pub copal_tenant: Option<String>,
    /// One `ck1` copal API key for every workspace (keys auth mode, shared
    /// tenancy: the key's tenant is THE tenant).
    #[arg(long, global = true, env = "ANTUMBRA_COPAL_KEY")]
    pub copal_key: Option<String>,
    /// Path to a JSON file mapping workspace tenant -> `ck1` copal API key
    /// (keys auth mode, per-workspace tenancy); a workspace absent from the
    /// map fails its ingest rather than landing in another tenant.
    #[arg(long, global = true, env = "ANTUMBRA_COPAL_KEYS")]
    pub copal_keys: Option<std::path::PathBuf>,
    #[command(subcommand)]
    pub command: Command,
}

/// What `antumbra claude` can do.
#[derive(Subcommand)]
pub enum ClaudeAction {
    /// Say whether this Claude Code session is in sovereign mode (its telemetry
    /// is off, which also turns off its feature flags), list what that costs,
    /// and check the settings that bring some of it back. Exits non-zero when a
    /// required setting is missing. Reads settings; never writes them.
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
    /// Keep the rules as memories too, in a `claude-code` compartment of your
    /// own, so an agent can recall why a feature is missing. Written through the
    /// same surface an agent writes through. Volatile, so they never train an
    /// expert. Safe to run again: a rule already there is kept, a changed one is
    /// stored and the old text penalized, and nothing is deleted.
    Remember {
        /// The Antumbra MCP surface, as the hooks know it.
        #[arg(
            long = "surface",
            env = "ANTUMBRA_URL",
            default_value = "http://127.0.0.1:8081"
        )]
        surface: String,
        /// The bearer token for that surface. Prefer the environment variable:
        /// a flag ends up in the shell's history.
        #[arg(long, env = "ANTUMBRA_TOKEN", hide_env_values = true)]
        token: Option<String>,
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
        /// The Antumbra MCP surface to ask for memories, as the hooks know it.
        #[arg(
            long = "surface",
            env = "ANTUMBRA_URL",
            default_value = "http://127.0.0.1:8081"
        )]
        surface: String,
        /// The bearer token for that surface. Prefer the environment variable.
        #[arg(long, env = "ANTUMBRA_TOKEN", hide_env_values = true)]
        token: Option<String>,
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
        /// The Antumbra MCP surface, as the hooks know it.
        #[arg(
            long = "surface",
            env = "ANTUMBRA_URL",
            default_value = "http://127.0.0.1:8081"
        )]
        surface: String,
        /// The bearer token for that surface. Prefer the environment variable.
        #[arg(long, env = "ANTUMBRA_TOKEN", hide_env_values = true)]
        token: Option<String>,
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
        #[arg(long, default_value_t = 30)]
        days: i64,
        /// The Antumbra MCP surface, as the hooks know it.
        #[arg(
            long = "surface",
            env = "ANTUMBRA_URL",
            default_value = "http://127.0.0.1:8081"
        )]
        surface: String,
        /// The bearer token for that surface. Prefer the environment variable.
        #[arg(long, env = "ANTUMBRA_TOKEN", hide_env_values = true)]
        token: Option<String>,
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
        #[arg(long)]
        days: Option<i64>,
        /// Say which merges would be reported and write nothing.
        #[arg(long, default_value_t = false)]
        dry_run: bool,
        /// The Antumbra MCP surface, as the hooks know it.
        #[arg(
            long = "surface",
            env = "ANTUMBRA_URL",
            default_value = "http://127.0.0.1:8081"
        )]
        surface: String,
        /// The bearer token for that surface. Prefer the environment variable.
        #[arg(long, env = "ANTUMBRA_TOKEN", hide_env_values = true)]
        token: Option<String>,
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
pub enum Command {
    /// Claude Code with its telemetry off (ADR-0021): what that silently costs,
    /// and what is done about it.
    Claude {
        #[command(subcommand)]
        action: ClaudeAction,
    },
    /// The verifier namespace (ADR-0022 S-4): propose a check, measure it
    /// against ground truth the loop did not produce, and move it. Only a
    /// sound measurement lets a synthesized verifier grant reward.
    Verifier {
        #[command(subcommand)]
        action: crate::verifier_args::VerifierAction,
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
    Ask {
        /// The task to route and answer.
        task: String,
        #[arg(long, default_value_t = 1)]
        k: usize,
        /// Max tokens to generate for the answer.
        #[arg(long, default_value_t = 128)]
        max_new_tokens: usize,
        /// Abstention threshold on relative coverage. Adjacent experts compress
        /// the margin; lower it to serve rather than escalate.
        #[arg(long, default_value_t = 0.08)]
        threshold: f32,
        /// Standing experts always composed onto the routed one (your conventions),
        /// `name:weight,...`, the static rule layer, internalized into the learned gate.
        #[arg(long)]
        with: Option<String>,
        /// Blend weight for the task-routed (contextual) expert when composing
        /// with standing experts.
        #[arg(long, default_value_t = 0.4)]
        self_weight: f32,
        /// Sampling temperature. 0 = greedy (deterministic, the learned mode,
        /// the right default for serving); raise for diverse draws.
        #[arg(long, default_value_t = 0.0)]
        temperature: f64,
    },
    /// Resident multi-adapter server with hardware-adaptive serving: load the shared base ONCE and
    /// hot-swap each routed expert's adapter per prompt, instead of cold-loading
    /// a model per call. Reads prompts from stdin (one per line) or a single
    /// --task, routes each via the learned router, and serves the answer from the
    /// resident engine, so a stream of prompts pays the base load only once.
    /// Needs --features models + a GPU.
    Serve {
        /// Answer this one prompt and exit; omit to stream prompts from stdin.
        #[arg(long)]
        task: Option<String>,
        /// Max tokens to generate per answer.
        #[arg(long, default_value_t = 128)]
        max_new_tokens: usize,
        /// Abstention threshold on relative coverage; below it the gate escalates.
        #[arg(long, default_value_t = 0.08)]
        threshold: f32,
        /// Sampling temperature. 0 = greedy (deterministic, the learned mode).
        #[arg(long, default_value_t = 0.0)]
        temperature: f64,
    },
    /// Collector/sync (R-1): bidirectionally reconcile this local store (the
    /// global `--url`, an embedded penumbra) with a remote authoritative
    /// SurrealDB, last-write-wins by each row's version timestamp. Runs on a
    /// cadence until ctrl-c, or `--once` for a single pass. Replicates the
    /// penumbra tables (memory, edges, compartments, grants) across the fleet;
    /// experts/adapters (on-disk safetensors) are out of scope.
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
    Train(crate::train_args::TrainArgs),
    /// Score a saved adapter's pass-rate on a corpus, with no training.
    /// Needs --features models + a GPU + python.
    Eval {
        /// Path to the JSON corpus of verifiable tasks ({id,prompt,verify}).
        #[arg(long)]
        corpus: String,
        /// Saved adapter to load over the base before scoring. Omit to score
        /// the bare base: the prior floor.
        #[arg(long)]
        adapter: Option<String>,
        /// Base model to score. Defaults to the one training loads
        /// (Qwen2.5-Coder-1.5B-Instruct), so an eval measures the model a run
        /// would start from.
        #[arg(long)]
        base_model: Option<String>,
        /// Completions sampled per task (the pass-rate denominator is tasks x K).
        #[arg(long, default_value_t = 8)]
        samples: usize,
        /// Tokens generated per completion. The training default, so a function
        /// long enough to pass training is not cut short here.
        #[arg(long, default_value_t = 256)]
        max_new_tokens: usize,
        /// Write every task's result (passes out of samples) to this JSON file.
        #[arg(long)]
        report: Option<String>,
        /// Sampling temperature. Defaults to training's, so the eval sees the
        /// draws a run would; 0 is greedy.
        #[arg(long)]
        temperature: Option<f64>,
        /// Nucleus cutoff. Defaults to training's (1.0, off).
        #[arg(long)]
        top_p: Option<f64>,
        /// Compute precision on the GPU: f32, bf16 or f16. Defaults to
        /// training's (bf16). Comparing f32 with bf16 on the same draws is how
        /// to tell a numerics problem from a model that cannot do the task.
        #[arg(long)]
        dtype: Option<String>,
    },
    /// Capture a supplied, verifier-checked correction into a frozen expert (the
    /// other intake path beside `train`, the capture intake into the umbra). The corpus carries a
    /// `completion` per task. Needs --features models + a GPU + python.
    Teach {
        /// JSON corpus of {id, prompt, completion, verify} corrections.
        #[arg(long)]
        corpus: String,
        #[arg(long, default_value_t = 1)]
        generations: u32,
        #[arg(long, default_value = "run:teach")]
        run: String,
        /// SFT epochs over the verified corrections. Overriding a strong base
        /// prior from one example needs many (try 40-80).
        #[arg(long, default_value_t = 40)]
        rounds: usize,
        /// Completions sampled to measure whether the correction internalized.
        #[arg(long, default_value_t = 8)]
        samples: usize,
        #[arg(long, default_value_t = 32)]
        max_new_tokens: usize,
        /// LoRA learning rate. Capturing a correction wants it higher than RAFT.
        #[arg(long, default_value_t = 3e-4)]
        lr: f64,
        /// Warm-start from this adapter (accumulate onto an existing expert).
        #[arg(long)]
        parent: Option<String>,
    },
    /// Train the learned router over the population's exemplars (the learned gate): a
    /// per-dimension metric that separates specialists from generalists where
    /// raw-cosine routing cannot. Retrain after the population changes. Needs
    /// --features models (the real embedder).
    GateTrain {
        #[arg(long, default_value_t = 400)]
        epochs: usize,
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
    Evolve {
        /// JSON corpus of verifiable tasks ({id,prompt,verify}).
        #[arg(long)]
        corpus: String,
        #[arg(long, default_value = "run:evolve")]
        run: String,
        /// Stop once the held pass-rate reaches this.
        #[arg(long, default_value_t = 0.9)]
        target: f32,
        /// Generation budget (each is one warm-started RAFT pass).
        #[arg(long, default_value_t = 3)]
        max_gens: usize,
        #[arg(long, default_value_t = 6)]
        samples: usize,
        #[arg(long, default_value_t = 2)]
        rounds: usize,
        #[arg(long, default_value_t = 32)]
        max_new_tokens: usize,
    },
    /// Autonomous population growth: route each task across the population, and
    /// grow a NEW specialist for the cluster the gate cannot cover, until
    /// coverage meets target or the expert budget runs out. Additive: existing
    /// experts are kept. Needs --features models + a GPU.
    Populate {
        /// JSON corpus of verifiable tasks ({id,prompt,verify}).
        #[arg(long)]
        corpus: String,
        #[arg(long, default_value = "grown")]
        run: String,
        /// Stop once this fraction of tasks is covered by the population.
        #[arg(long, default_value_t = 0.9)]
        target_coverage: f32,
        /// How many new specialists to grow at most.
        #[arg(long, default_value_t = 3)]
        max_experts: usize,
        #[arg(long, default_value_t = 8)]
        samples: usize,
        #[arg(long, default_value_t = 3)]
        rounds: usize,
        #[arg(long, default_value_t = 32)]
        max_new_tokens: usize,
    },
    /// Bootstrap from an existing memory corpus (qdrant, surrealdb, a json
    /// file): adapt a normalized memory export into capture tasks and learn
    /// from them, instead of discovering every skill cold via RAFT. Reinforced
    /// memories are captured (trusted on import); weak ones become RAFT seeds.
    /// Writes the converted corpus; with --train, internalizes the captures.
    /// Needs --features models (+ a GPU + python only when --train).
    MemoryImport {
        /// JSON array of normalized memories ({content, scope, marker, forbid,
        /// confidence, ...}). See `memory::MemoryRecord`.
        #[arg(long)]
        source: String,
        /// Where to write the converted capture corpus for inspection / `teach`.
        #[arg(long, default_value = "corpora/_imported.json")]
        out: String,
        /// Confidence at/above which a memory is trusted on import (else seed).
        #[arg(long, default_value_t = 0.5)]
        capture_threshold: f32,
        /// Also internalize the captured memories now (the capture loop).
        #[arg(long, default_value_t = false)]
        train: bool,
        #[arg(long, default_value = "run:memory")]
        run: String,
        #[arg(long, default_value_t = 40)]
        rounds: usize,
        #[arg(long, default_value_t = 8)]
        samples: usize,
        #[arg(long, default_value_t = 32)]
        max_new_tokens: usize,
        #[arg(long, default_value_t = 3e-4)]
        lr: f64,
    },
    /// Consolidation: score a normalized memory export against the
    /// graduation gate (recurrence x verifiability x stability), graduate the
    /// survivors into per-skill specialists rehearsing already-consolidated
    /// skills (replay, the catastrophic-interference fix), and append them to
    /// the consolidated log. The store keeps everything that does not graduate
    /// (volatile / episodic / under-reinforced). Needs --features models.
    Consolidate {
        /// JSON array of normalized memories (see `memory::MemoryRecord`).
        #[arg(long)]
        source: String,
        /// Accumulating consolidated corpus: the replay source and demotion log.
        #[arg(long, default_value = "corpora/_consolidated.json")]
        log: String,
        /// Reinforcement count at/above which recurrence is satisfied.
        #[arg(long, default_value_t = 2)]
        min_recurrence: u32,
        /// Confidence floor a graduating memory must clear.
        #[arg(long, default_value_t = 0.5)]
        min_confidence: f32,
        /// Internalize the graduates now (else dry-run scoring + log only).
        #[arg(long, default_value_t = false)]
        train: bool,
        /// Rehearsal examples per winner interleaved into capture SFT.
        #[arg(long, default_value_t = 0.5)]
        replay_ratio: f64,
        #[arg(long, default_value = "expert:consolidated")]
        run: String,
        #[arg(long, default_value_t = 40)]
        rounds: usize,
        #[arg(long, default_value_t = 8)]
        samples: usize,
        #[arg(long, default_value_t = 32)]
        max_new_tokens: usize,
        #[arg(long, default_value_t = 3e-4)]
        lr: f64,
        /// Verified winners per optimizer step. 1 steps on each example in
        /// turn; more steps on the mean gradient of that many at a time (true
        /// mini-batch descent), which is less noisy, so a higher `--lr` stays
        /// stable and no example dominates by being trained last. Examples are
        /// backpropagated one at a time, so memory does not grow with it.
        #[arg(long, default_value_t = 1)]
        batch_size: usize,
    },
    /// Retire an expert: population-level forgetting that demotes rather than
    /// deletes (ADR-0022 S-5). The expert goes dormant, masked from the gate
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
    /// Seed a memory into a user's compartment from the CLI (the owner/admin
    /// path; agents write via the MCP `store_memory` tool). No embedding is
    /// attached; consolidation gathers a compartment by membership. Pair with
    /// `consolidate-compartment` to mint the compartment into a private expert.
    Remember {
        #[arg(long)]
        tenant: String,
        #[arg(long)]
        user: String,
        #[arg(long)]
        compartment: String,
        /// The memory content: the behavior/fact to internalize on consolidation.
        #[arg(long)]
        content: String,
        #[arg(long, default_value = "world")]
        network: String,
        #[arg(long, default_value_t = 1.0)]
        confidence: f32,
    },
    /// Ingest a knowledge document for `recall_documents`: chunk, embed, store,
    /// from a file or from what a command prints. The command form is the
    /// parser-free answer to an inventory question: run the framework's own
    /// lister (`deno task routes`, an OpenAPI export, `cargo metadata`) and keep
    /// its output, stamped with the repository, commit, and branch it ran at, so
    /// a recalled chunk says which commit it describes.
    Ingest {
        #[arg(long)]
        tenant: String,
        #[arg(long)]
        user: String,
        /// The document's title (groups and names its chunks; re-ingesting a
        /// title replaces its chunks in place).
        #[arg(long)]
        title: String,
        /// Read the document from this file.
        #[arg(long, conflicts_with = "run")]
        file: Option<std::path::PathBuf>,
        /// Where the document came from, for the record (defaults to the file
        /// path or the command line).
        #[arg(long)]
        source: Option<String>,
        /// The file the document is about, added to the git anchor.
        #[arg(long)]
        path: Option<String>,
        /// Do not stamp the current repository / commit / branch on the chunks.
        #[arg(long, default_value_t = false)]
        no_git: bool,
        /// Keep the document in this compartment, so only its owner and the people
        /// it is shared with can recall it. Omit it for the workspace's shared
        /// pool, which every member can recall.
        #[arg(long)]
        compartment: Option<String>,
        /// The command whose stdout is the document, after `--`
        /// (`antumbra ingest --title routes -- deno task routes`).
        #[arg(last = true)]
        run: Vec<String>,
    },
    /// Derive facts the repository's history already states, from `git log`
    /// with no parser: ownership per top-level area, change hotspots, and files
    /// that change together, over a window. Each fact is stored as a `world`
    /// memory whose evidence names the commit range it came from, so a later
    /// session sees how old it is instead of trusting a refreshed-or-not index.
    GitFacts {
        #[arg(long)]
        tenant: String,
        #[arg(long)]
        user: String,
        #[arg(long)]
        compartment: String,
        /// Look back this many days.
        #[arg(long, default_value_t = 90)]
        days: u32,
        /// How many hotspots and co-change pairs to keep.
        #[arg(long, default_value_t = 10)]
        top: usize,
        /// Print the facts without storing them.
        #[arg(long, default_value_t = false)]
        dry_run: bool,
    },
    /// Consolidate a private compartment into a **private expert** (a memory compartment):
    /// gather the compartment's memories, score them through the consolidation
    /// gate, capture the graduates, and mint an expert owned by the user (not in
    /// the shared router; routed for its owner by centroid). Needs --features
    /// models + a GPU.
    ConsolidateCompartment {
        #[arg(long)]
        tenant: String,
        #[arg(long)]
        user: String,
        #[arg(long)]
        compartment: String,
        /// Reinforcement floor for a memory to graduate (0 = any in-compartment).
        #[arg(long, default_value_t = 0)]
        min_recurrence: u32,
        #[arg(long, default_value_t = 0.5)]
        min_confidence: f32,
        #[arg(long, default_value_t = 40)]
        rounds: usize,
        #[arg(long, default_value_t = 8)]
        samples: usize,
        #[arg(long, default_value_t = 32)]
        max_new_tokens: usize,
        #[arg(long, default_value_t = 3e-4)]
        lr: f64,
        #[arg(long, default_value_t = 0.5)]
        replay_ratio: f64,
    },
    /// Metabolize orchestration traces into the frozen-expert population: adapt a
    /// normalized trace export (loop runs, behavior-graph evaluations, task
    /// executions) into capture tasks the population internalizes, so the brain
    /// learns to do in one shot what it took many steps to do. Only successful,
    /// recurrent traces metabolize. Writes the converted corpus; with --train,
    /// internalizes it. Needs --features models.
    Metabolize {
        /// JSON file of normalized harness traces ({goal, solution, kind, steps,
        /// success, recurrence, ...}). See `antumbra_train::HarnessTrace`. Any
        /// harness exports to this shape (see scripts/ for an example adapter).
        #[arg(long)]
        source: String,
        /// Where to write the converted capture corpus for inspection / `teach`.
        #[arg(long, default_value = "corpora/_metabolized.json")]
        out: String,
        /// Recurrence floor: only patterns seen at least this often metabolize.
        #[arg(long, default_value_t = 1)]
        min_recurrence: u32,
        /// Do NOT metabolize each trace's decomposition (its steps); learn only
        /// the collapsed one-shot outcome, not the process. Steps are on by default.
        #[arg(long, default_value_t = false)]
        no_steps: bool,
        /// Re-pull and metabolize on a cadence (a continuous learning daemon).
        #[arg(long, default_value_t = false)]
        watch: bool,
        /// Seconds between cycles when --watch is set.
        #[arg(long, default_value_t = 300)]
        interval_secs: u64,
        /// Also internalize the metabolized traces now (the capture loop).
        #[arg(long, default_value_t = false)]
        train: bool,
        #[arg(long, default_value = "run:metabolize")]
        run: String,
        #[arg(long, default_value_t = 40)]
        rounds: usize,
        #[arg(long, default_value_t = 8)]
        samples: usize,
        #[arg(long, default_value_t = 32)]
        max_new_tokens: usize,
        #[arg(long, default_value_t = 3e-4)]
        lr: f64,
    },
    /// Have the antumbra propose memory compartments by clustering a user's
    /// unorganized memory (their inbox compartment + anything they authored
    /// uncompartmented) into competence-coherent regions. Needs no model: it
    /// clusters the embeddings already stored on each memory. With --apply it
    /// creates each proposal as an `Origin::Proposed` compartment and moves its
    /// members in (reversible by deleting the compartment).
    ProposeCompartments {
        #[arg(long)]
        tenant: String,
        #[arg(long)]
        user: String,
        /// The inbox compartment to cluster; defaults to comp:{tenant}:{user}:default.
        #[arg(long)]
        inbox: Option<String>,
        /// Cosine at/above which two memories cluster together.
        #[arg(long, default_value_t = 0.6)]
        similarity_threshold: f32,
        /// Smallest cluster worth proposing.
        #[arg(long, default_value_t = 3)]
        min_size: usize,
        /// Persist the proposals (else print only).
        #[arg(long, default_value_t = false)]
        apply: bool,
    },
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
