//! Antumbra CLI argument surface: the clap `Cli` / `Command` definitions,
//! kept out of `main.rs`. Handlers live in `main.rs` + `ops.rs`.

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "antumbra", about = "Antumbra operator CLI", version)]
pub struct Cli {
    /// SurrealDB url: `mem://` (ephemeral), `surrealkv://./data/antumbra.skv`
    /// (persistent), or `ws://host:8000/rpc`.
    #[arg(long, global = true, default_value = "mem://")]
    pub url: String,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
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
        /// Abstention threshold on relative coverage. Adjacent experts compress
        /// the margin; lower it to serve rather than escalate (EXP-004/011).
        #[arg(long, default_value_t = 0.08)]
        threshold: f32,
        /// Standing experts always composed onto the routed one (your conventions),
        /// `name:weight,...` — the Kushtaka rule layer, internalized (ADR-0009).
        #[arg(long)]
        with: Option<String>,
        /// Blend weight for the task-routed (contextual) expert when composing
        /// with standing experts.
        #[arg(long, default_value_t = 0.4)]
        self_weight: f32,
        /// Sampling temperature. 0 = greedy (deterministic, the learned mode —
        /// the right default for serving); raise for diverse draws.
        #[arg(long, default_value_t = 0.0)]
        temperature: f64,
    },
    /// Resident multi-adapter server (ADR-0006): load the shared base ONCE and
    /// hot-swap each routed expert's adapter per prompt, instead of cold-loading
    /// a model per call. Reads prompts from stdin (one per line) or a single
    /// --task, routes each via the learned router, and serves the answer from the
    /// resident engine — so a stream of prompts pays the base load only once.
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
        /// Saved adapter to load over the base before scoring. Omit to score
        /// the bare base — the prior floor (EXP-011's load-bearing check).
        #[arg(long)]
        adapter: Option<String>,
        #[arg(long, default_value = "Qwen/Qwen2.5-Coder-1.5B")]
        base_model: String,
        /// Completions sampled per task (the pass-rate denominator is tasks x K).
        #[arg(long, default_value_t = 8)]
        samples: usize,
        #[arg(long, default_value_t = 64)]
        max_new_tokens: usize,
    },
    /// Capture a supplied, verifier-checked correction into a frozen expert (the
    /// other intake path beside `train`, ADR-0004/0009). The corpus carries a
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
    /// Train the learned router over the population's exemplars (ADR-0009): a
    /// per-dimension metric that separates specialists from generalists where
    /// raw-cosine routing cannot. Retrain after the population changes. Needs
    /// --features models (the real embedder).
    GateTrain {
        #[arg(long, default_value_t = 400)]
        epochs: usize,
    },
    /// Compose several experts into one blended adapter and serve a task
    /// through it (ADR-0009): the population as a capability multiplier. Needs
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
    /// coverage meets target or the expert budget runs out. Additive — existing
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
    /// Bootstrap from an existing memory corpus (Kushtaka, qdrant, surrealdb, a
    /// json file): adapt a normalized memory export into capture tasks and learn
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
    /// Consolidation (EXP-021): score a normalized memory export against the
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
        /// Accumulate the batch gradient and take one optimizer step per round
        /// (true mini-batch descent) instead of one step per example. Less noisy,
        /// so a higher `--lr` stays stable and no example dominates by being
        /// trained last.
        #[arg(long, default_value_t = false)]
        grad_accumulation: bool,
    },
    /// Retire an expert by name and refresh the router: population-level
    /// forgetting. Wire a store's contradiction report against a consolidated
    /// memory to this to undo a graduation (a frozen LoRA cannot be edited
    /// per-fact; retiring the whole expert is how you forget). Needs --features
    /// models (the real embedder for the router refresh).
    Retire {
        /// The expert name to supersede.
        #[arg(long)]
        expert: String,
    },
    /// Seed a memory into a user's compartment from the CLI (the owner/admin
    /// path; agents write via the MCP `store_memory` tool). No embedding is
    /// attached — consolidation gathers a compartment by membership. Pair with
    /// `consolidate-compartment` to mint the compartment into a private expert.
    Remember {
        #[arg(long)]
        tenant: String,
        #[arg(long)]
        user: String,
        #[arg(long)]
        compartment: String,
        /// The memory content — the behavior/fact to internalize on consolidation.
        #[arg(long)]
        content: String,
        #[arg(long, default_value = "world")]
        network: String,
        #[arg(long, default_value_t = 1.0)]
        confidence: f32,
    },
    /// Consolidate a private compartment into a **private expert** (ADR-0014):
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
    /// Metabolize a harness (ADR-0001): adapt a harness's successful
    /// orchestration traces (loop runs, behavior-graph evaluations, task
    /// executions) into capture tasks the population internalizes — so the brain
    /// learns to do in one shot what the harness did in many steps. Only
    /// successful, recurrent traces metabolize. Writes the converted corpus; with
    /// --train, internalizes it. Needs --features models.
    Metabolize {
        /// JSON file of normalized harness traces ({goal, solution, kind, steps,
        /// success, recurrence, ...}). See `harness::HarnessTrace`. Provide this
        /// OR --from-harness.
        #[arg(long)]
        source: Option<String>,
        /// Pull traces LIVE from a running Kushtaka harness instead of a file:
        /// the MCP engine base URL (e.g. http://10.0.0.110:8081). Needs --api-key
        /// (or ANTUMBRA_KUSHTAKA_KEY). Normalizes any trace-returning tool.
        #[arg(long)]
        from_harness: Option<String>,
        /// The Kushtaka trace tool to call (any trace-shaped response works).
        #[arg(long, default_value = "list_tasks")]
        harness_tool: String,
        /// Extra JSON args for the harness tool (e.g. '{"graph_id":"g","limit":50}').
        #[arg(long)]
        harness_args: Option<String>,
        /// Kushtaka API key for --from-harness (else env ANTUMBRA_KUSHTAKA_KEY).
        #[arg(long)]
        api_key: Option<String>,
        /// Kushtaka workspace/scope passed on each call.
        #[arg(long)]
        scope: Option<String>,
        /// Where to write the converted capture corpus for inspection / `teach`.
        #[arg(long, default_value = "corpora/_metabolized.json")]
        out: String,
        /// Recurrence floor: only patterns seen at least this often metabolize.
        #[arg(long, default_value_t = 1)]
        min_recurrence: u32,
        /// Do NOT metabolize each trace's decomposition (its steps) — learn only
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
    /// Have the antumbra propose compartments (ADR-0014) by clustering a user's
    /// unorganized memory (their inbox compartment + anything they authored
    /// uncompartmented) into competence-coherent regions. Needs no model — it
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
}
