# Antumbra

> A private, self-improving AI that plugs into your coding agent and grows a team of
> small specialists on your own hardware — turning the work you repeat into
> permanent, private skills, and learning the **scope** of what each is good at so it
> knows when to answer locally and when to escalate.

You keep the agent you already use (Claude Code, Cursor, any MCP client); Antumbra
becomes its **persistent brain** — memory, identity, multi-tenant boundaries, and a
growing population of experts. Unlike a memory layer that only makes a frozen model
*remember*, Antumbra makes it **get better**: verified outcomes are metabolized into
the weights, so the orchestration scaffolding shrinks as competence accrues. All-Rust,
single-process, your data stays on your hardware. Runs **fully offline** or as a
**hosted-but-private** service. This README describes the system as built today; see
**[Using Antumbra](docs/integration.md)** for how to wire it into your agent.

---

## What it is

Most "AI assistants" are one large model in someone else's data center: you rent
it, you send it your data, and it is exactly as good tomorrow as today — it never
learns *your* work.

Antumbra is the opposite. It maintains a **population of small, frozen specialists**
— LoRA adapters over one shared, code-capable base model — each good at a narrow,
recurring task. When a result is **verified** (a test passes, a command works, a
schema matches, you accept a draft), that competence is trained into an adapter and
**frozen** into the population. A **learned router** sends each new task to the
specialist most likely to handle it, and a **competence boundary** decides whether
to answer locally or escalate. Over time it gets measurably better at the work you
do most, on your own machine, with your data never leaving the building.

Two design commitments make this more than a model zoo:

- **Frozen experts.** A graduated specialist is immutable. Immutability is the only
  hard guarantee that a learned skill is never silently forgotten when the system
  trains something new (ADR-0001).
- **Scope, not just skill.** Most systems accumulate what *works*. Antumbra's bet is
  that the neglected, more valuable half is the **boundary** of a rule — learning
  that a behavior is right in one context and wrong in a neighbouring one, and
  *which contextual feature governs the switch*. Constraints are scoped, not
  absolute: "use `deno install`, not `npm install`" is true **in this repo**, not
  everywhere. That boundary is what the route-locally-vs-escalate decision rests on
  (ADR-0004).

---

## How you use it

Antumbra is not an app you open — it is the brain your coding agent plugs into,
through MCP plus three lifecycle hooks. The full guide is **[Using
Antumbra](docs/integration.md)**; the shape:

1. **Bootstrap on session start.** A hook pulls your standing conventions and the
   memory relevant to this project from Antumbra into the agent's opening context —
   no cold start, it already knows "this repo uses `deno`."
2. **Route or answer.** The agent calls Antumbra's `answer`/`route` tools: a task
   goes to the frozen expert most likely to cover it, or escalates when out of scope.
   A served task costs nothing; only genuine novelty hits the expensive model.
3. **Capture on stop.** A hook nudges the agent to write verified observations back.
   Those recurrent, checked traces are what `antumbra metabolize` later turns into a
   new permanent expert — so next session the agent is measurably better, and more of
   your work is served locally for free.

Ready-to-adapt hook templates (PowerShell + bash, Windows/macOS/Linux) live in
**[`scripts/hooks/`](scripts/hooks/)**.

**Why this beats a memory layer.** Retrieval-memory tools (Mem0, Letta, Zep, Cognee)
make a frozen model *remember*; every run re-pays the prompt/lookup cost against the
same base. Antumbra makes the model *get better* — capability compounds into weights
and the scaffold shrinks. It is also a memory store **and** a skill engine in one
private, ACL-governed process, where those tools stop at retrieval.

### Three ways to run it

Same engine; only transport, identity, and who runs the box differ.

- **Offline / private** — embedded store, stdio or loopback MCP, one identity.
  Nothing leaves the machine. The default and the privacy floor (a solo dev, an
  air-gapped or regulated box).
- **Hosted, still private** — networked HTTP/SSE, JWT per request, engine-enforced
  multi-tenancy, device sync, live propagation. A team or fleet shares one brain
  without running the infrastructure; each consumer's data is isolated in the engine.
- **Bespoke** — for businesses that want this implemented and operated for them,
  integrated into their repos, verifiers, and compliance boundary.

See **[Product surface](docs/product.md)** for the control plane (dashboard,
knowledge documents, onboarding) and how Antumbra supersedes a separate agent-memory
engine.

---

## How it learns (not distillation)

Antumbra learns from **verifiable outcomes in your environment**, not by imitating a
teacher's text:

- **The environment is the truth.** For coding over your repos: the test that
  passes, the command that runs, the build that goes green. That is the reward
  (RAFT — reward-ranked fine-tuning over verified completions, ADR-0010).
- **A critic turns a failure into a diagnostic signal** — "`npm install` failed
  because this is a Deno project; use `deno install`" — and names the *governing
  feature* of the boundary. That is counterfactual scope extraction (ADR-0004).
- **Training is on the verified outcome, not the critic's words.** That keeps
  learning grounded in your data, and clear of "trained on a provider's outputs." A
  frontier model, if used at all, is an optional cold-start accelerator — your own
  repos are a more authoritative teacher about *your* conventions than any model.

This makes **coding over your own repos the ideal first domain**: maximally
verifiable (you can *run* it), maximally context-scoped (per-repo conventions are
textbook boundaries), and the data is yours.

### How it pays for itself

You don't replace an expensive model overnight — you harvest it, then wean off it:

1. **Route through Antumbra**, still using the expensive model, logging every
   task + context + result.
2. **Verify and capture.** A result that passes a check becomes a training example
   you already paid for once.
3. **Graduate a local specialist** that reproduces that behavior on that one narrow
   task.
4. **Route local, escalate only out of scope.** Once the local expert passes the
   same verification reliably *within its scope*, that task is served for $0; the
   expensive model is called only when the boundary says "out of scope," and that
   call becomes the next training example.

You amortize the bill toward zero on the work you repeat, keeping the frontier model
on retainer for genuinely novel tasks. *Honest caveats:* there is an investment
period (you pay while teaching); you trade API bills for a GPU plus your time (it
pays off at volume, not light use); and some hard, novel tasks never graduate.

---

## Memory, multi-tenancy, and sharing

Antumbra includes its own first-class memory and runtime, rather than running
alongside a separate agent engine.

- **Memory store** (ADR-0012) — a tenant-scoped store with three networks
  (`world` facts, `bank` experiences, `opinion` judgments), HNSW vector recall,
  reinforcement counts, and per-memory provenance (who wrote it, on which device).
  It is both the **bootstrap** (existing memories seed the population with no cold
  start) and the **consolidation source**: an offline pass scores memories
  (recurrence × verifiability × stability), graduates the trusted ones into experts
  with an interleaved **replay** buffer that resists catastrophic forgetting, and
  retires an expert when a consolidated memory is later contradicted. Forgetting is
  a **soft-delete tombstone**, so a deletion is retained long enough to propagate
  and is not silently resurrected by another replica. The embedder is **yours** — a
  local model behind the `Embedder` port — so the embedding step that touches your
  content stays on your side (per-workspace bring-your-own-embedder is roadmap P-1).

- **Engine-enforced isolation** (ADR-0013) — multi-tenancy lives in the SurrealDB
  engine, not in handler code. Record-access binds `(tenant, user)` to `$auth`, and
  table permissions (`WHERE tenant_id = $auth.tenant`, plus compartment ownership
  and grant subqueries) filter every row at the engine — so a forgotten app-side
  filter cannot leak. Validated live against a real `ws://` server, including the
  fix (R-6) that makes requests run on a scoped, non-root connection so the engine
  ACL is genuinely enforced over the network.

- **Compartments** (ADR-0014) — named, ownable spaces of memory: the unit of
  organization, deletion, and sharing. A user grants another user `reference` or
  `link` capability on a compartment (engine-enforced); a private compartment
  consolidates into a *private* expert. Antumbra can also *propose* compartments by
  clustering unorganized memory. **Revocation is a tombstone** that fails closed
  immediately at the engine and propagates across devices — a revoked grantee
  cannot be kept in by a stale copy elsewhere.

## The runtime surface

- **MCP server** (`antumbra-mcp`, ADR-0015) — a Rust Model Context Protocol server
  exposing memory, graph, compartment, routing, and `answer` tools. Two transports:
  - **stdio** for a single local identity, and
  - a **networked, multi-tenant HTTP** surface where each request carries a signed
    JWT whose `(tenant, user)` claims become the engine's `$auth`. Runs in
    streamable-HTTP **stateful (SSE)** mode so the server can push notifications.

- **Live propagation** — when a shared compartment changes, a SurrealDB `LIVE`
  subscription resolves the change's audience (owner + grantees) and pushes a
  notification to each recipient's open SSE stream, so an agent learns of new or
  forgotten memories without polling. Validated end-to-end over the wire.

- **Collector / sync** (`antumbra-sync`) — keeps an edge device's local embedded
  store and a remote authoritative store in agreement by periodic **bidirectional
  last-write-wins** reconciliation (compared by each row's version timestamp, in
  Rust, so the flow converges and self-terminates). This is the multi-device story:
  a user's memories — and the grants and revocations that govern them — become
  visible across a fleet through one source of truth. CLI: `antumbra sync`.

---

## Status (honest)

A research project, not a product: every milestone is a falsifiable experiment with
a kill criterion. All-Rust, single process; data in SurrealDB via `surql-rs`
(builder-only — no hand-written SurrealQL); training and serving via `candle`.

**v0 substrate:** one frozen, code-capable base (Qwen2.5-Coder-1.5B-Instruct) on a
single 24 GB GPU, a growing library of frozen LoRA experts, and a
boundary-conditioned coverage gate. The real candle trainer/server is behind a
`models` feature; the default build runs a CPU demo trainer so the orchestration is
exercisable without a GPU.

**Validated (toward 2026-06):**

- **Training works on a real GPU.** RAFT lifts pass-rate to 1.0 under both a
  convention reward and a verifier that *executes* generated code. The
  generation-quality recipe is dialed in (instruct base + chat template, a decode
  policy of repetition penalty + no-repeat-n-gram + nucleus, shuffled SFT, tuned
  learning rate, and optional gradient accumulation): a small corpus trains an
  expert that emits correct, *generalizing* output — including held-out inputs.
- **Consolidation closes the loop:** memories score through the gate and graduate
  into a specialist that internalizes the skill; a private compartment consolidates
  into a private, owner-scoped expert.
- **Routing + boundary:** a real embedder drives a gate that routes to the right
  specialist and escalates out-of-scope queries by *relative coverage*, not an
  absolute similarity floor; the counterfactual boundary composes end-to-end.
- **Networked multi-tenancy is engine-enforced over `ws://`,** validated live
  against a real SurrealDB v3 server: cross-tenant isolation and intra-tenant
  compartment privacy both hold; grant makes a shared memory visible and revoke
  fails closed — over the wire.
- **Multi-device:** bidirectional LWW sync converges; deletes and grant
  revocations propagate as tombstones with no resurrection; live SSE notifications
  reach grantees end-to-end.

**Not yet:** large corpora and many experts (the real generalization-and-forgetting
test at scale); the learned latent-mixing gate (the north star beyond the coverage
gate); 4-bit quantized training (GRPO is built — RAFT and GRPO both ship);
heterogeneous composition (genuinely
separate experts wired by learned cross-attention bridges, ADR-0009) — for which the
v0 gate, boundary engine, loop, and substrate all carry over. On the **product**
side, the web control plane — dashboard, knowledge documents, the expert mixer,
hosted onboarding — is specified ([ADR-0016](docs/adr/0016-control-plane-and-product-surface.md),
roadmap P-1…P-5) but not built; the CLI/TUI and the [`scripts/hooks/`](scripts/hooks/)
templates are today's interface.

---

## Build & run

```bash
cargo test                      # whole workspace, green

# No-GPU demo: drive the durable loop with the CPU trainer, then inspect.
cargo run -p antumbra-cli -- schema                                  # print generated DDL
cargo run -p antumbra-cli -- --url surrealkv://./data/a.skv loop --generations 3
cargo run -p antumbra-cli -- --url surrealkv://./data/a.skv status

# Real training + serving (CUDA GPU + the Qwen weights; python for exec verifiers).
cargo run -p antumbra-cli --features models,cuda -- --url surrealkv://./data/a.skv \
  train --corpus corpora/arith.json --run arith --generations 1
cargo run -p antumbra-cli --features models,cuda -- --url surrealkv://./data/a.skv \
  ask "Write a Python function add(a, b) that returns their sum."   # route -> load adapter -> generate

# Bidirectional sync between this store and a remote authoritative SurrealDB.
cargo run -p antumbra-cli -- --url surrealkv://./data/a.skv \
  sync --remote ws://host:8000/rpc --remote-user root --remote-pass <pw>
```

The networked MCP server (`antumbra-mcp --http 0.0.0.0:8081 --url ws://... --db-user
root --db-pass <pw>`, with a JWT secret) and its live multi-tenant validation are
reproducible with the probes under `docs/` (run against a SurrealDB v3 server, e.g.
`docker run -p 8000:8000 surrealdb/surrealdb:v3.0.5 start --user root --pass root
memory`). See **[Running the trainer](docs/running-the-trainer.md)** for the CUDA
GPU recipe and the validated generation-quality settings.

---

## Documentation

- **[Antumbra, explained for anyone](docs/antumbra-explained.md)** — a plain-English,
  no-jargon tour with diagrams and analogies (no ML background needed). Share this
  with a non-technical friend.
- **[Using Antumbra](docs/integration.md)** — wire it into your coding agent (the
  bootstrap/capture lifecycle hooks), offline vs hosted, and why it beats a memory
  layer. Start here.
- **[Product surface](docs/product.md)** — who it's for, the three tiers, and the
  control-plane gap (dashboard, knowledge documents, onboarding) Antumbra is closing.
- **[Architecture](docs/architecture.md)** — system, substrate, decision chain,
  training and data flow, schema.
- **[Technical Reference](docs/technical-reference.md)** — crate map, domain model,
  port seams, algorithms as coded, build matrix, validation results.
- **[Roadmap](docs/roadmap.md)** — what is built and what is queued, with status
  per item (sync, live propagation, networked validation, the security fixes).
- **[Running the trainer](docs/running-the-trainer.md)** — the CUDA GPU recipe and
  the validated generation-quality recipe.
- **[Architecture Decision Records](docs/adr/README.md)** — every load-bearing
  decision, ADR-0001 … ADR-0016.
- **[Experiment Ledger](experiments/README.md)** — each falsifiable validation:
  claim, method, result, kill criterion, reproduce command.

### Crates

`antumbra-core` (domain types, ports) · `antumbra-store` (SurrealDB persistence via
surql-rs) · `antumbra-embed` (HTTP `/embeddings` client behind the `Embedder` port) ·
`antumbra-gate` (router/coverage gate) · `antumbra-boundary`
(counterfactual scope) · `antumbra-critic` (verifiers + credit assignment) ·
`antumbra-train` (candle Qwen + LoRA trainer) · `antumbra-serve` (resident
multi-adapter serving) · `antumbra-loop` (generational loop) · `antumbra-sync`
(collector/sync + live propagation) · `antumbra-mcp` (MCP server, stdio + networked)
· `antumbra-cli` · `antumbra-tui`.
