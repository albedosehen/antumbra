# Antumbra

> A private, self-improving AI that grows its own team of small specialists on your own hardware -
> and learns the *scope* of what each one is good at.

---

## What this is, in plain terms

Most "AI assistants" are one enormous model in someone else's data center. You rent it, you send it your
data, and it is exactly as good tomorrow as it is today - it never actually *learns your work*.

**Antumbra is the opposite.** It runs on your own computer and grows a **population of small, frozen
specialists** - each good at a narrow, recurring task. When it meets something it can't do well, it spawns a
temporary *apprentice* that practices until it either **graduates** into a new permanent specialist or is
**pruned**. Crucially, it learns *where* each specialist is reliable and where it isn't - so it knows when to
answer locally and when to step back. Over weeks it quietly gets better at *your* work, on *your* machine, with
*your* data never leaving the building.

The name is load-bearing. A shadow cast by an occluder has three regions, and so does this system:

- **Umbra** = the frozen experts - total shadow, full and proven coverage; immutable (immutability is the only
  guarantee a skill is never forgotten) - *ADR-0001*
- **Penumbra** = the partial-shadow region: everything soft, editable, and not-yet-frozen. Two facets, both of
  which deepen into umbra (graduate) or fade (prune): the **shadows-in-training** (trainable adapters, *ADR-0002*)
  and the **memory store** (raw, editable, reinforced experiences — the bootstrap and training unit, *ADR-0012*;
  organized into shareable *compartments*, *ADR-0014*). The hippocampus to the umbra's neocortex.
- **Antumbra** = the keystone - the region beyond the umbra's tip where the geometry *inverts*: the occluder no
  longer covers the light source and a ring of light breaks through. That inversion is the counterfactual
  *scope* - the context where a behavior that was right becomes wrong, where local competence gives out and you
  must escalate - *ADR-0004*

The project is named for that third region: the boundary **is** the thesis. Knowing exactly where your own
shadow stops covering - and which contextual feature governs the edge - is the whole game.

```mermaid
flowchart LR
    PEN["PENUMBRA<br/>shadows-in-training - explore,<br/>then deepen or fade"] -->|"graduate (deepen to full shadow)"| UMB["UMBRA<br/>frozen experts<br/>(adapters over a shared base)"]
    PEN -->|"prune"| X["dissipated"]
    UMB -.->|"cast a new shadow"| PEN
    ANT["ANTUMBRA - the keystone<br/>counterfactual scope:<br/>where coverage inverts,<br/>and when to escalate"] -.->|"gates"| UMB
    X -.->|"log why + where it failed"| ANT
```

---

## Who is it for? (a concrete example)

> **Maya is a marketing specialist.** Half her week is banal: reformatting campaign numbers into the same
> weekly deck, drafting routine emails in her company's voice, triaging the inbox, cleaning spreadsheets. A
> cloud AI could help - but her company's data can't leave the building, and the generic local model she tried
> is mediocre at *her* formats and tone.
>
> Antumbra runs on the workstation under her desk. The first week it is average. But every time it gets one of her
> recurring tasks right - *verified*: the numbers reconcile, the schema matches, the draft survives her edit -
> that competence is **frozen into a specialist**. A month later it has a "weekly-deck" expert, a "brand-voice
> draft" expert, an "inbox-triage" expert: small, private, and genuinely good at the banal 50% of her job.

---

## How it pays for itself

You don't replace the expensive model overnight - you **harvest it, then wean off it.** The paid frontier model
is a *teacher* you consult less and less:

1. **Route through Antumbra.** Keep using the expensive model, but every task + context + result is logged.
2. **Verify & capture.** When a result passes a check (tests, schema, or *you accept the draft*), it becomes a
   training example you already paid for once.
3. **Graduate a local specialist** that reproduces that behavior on *that one narrow recurring task*.
4. **Route local, escalate only out-of-scope.** Once the local expert passes the same verification reliably
   *within its scope*, Antumbra serves that task for **$0** and only calls the expensive model when the
   **boundary** says "out of scope / not confident" - and that call becomes the next training example.

So you **amortize your bill toward zero on the work you repeat**, while keeping the frontier model on retainer
for genuinely novel tasks. The keystone (ADR-0004) is what makes this *safe*: knowing the scope of what you can
do locally is precisely the "answer-locally vs pay-the-expensive-model" decision.

*Honest caveats:* there's an investment period (you pay while teaching); you trade API bills for a GPU + your
time (pays off at volume, not light use); some hard/novel tasks never graduate and stay on the paid model; and
training on a provider's *outputs* can run into their terms - which is why Antumbra grounds truth in **your own
data + real execution**, not the flagship's text (see below).

---

## Why this is deeper than "yet another agent harness"

Most agent frameworks are **orchestration harnesses**: prompt-routing and tool-calling *on top of a frozen
brain*. The brain never changes; only the choreography does.

Antumbra's core thesis is more fundamental: **modeling the *counterfactual boundary of its own competence.*** Most
systems accumulate what *works* (skill libraries). Antumbra's bet is that the more valuable, neglected half is the
**scope** of a rule - learning that a behavior is **right in one context and wrong in a neighbouring one**, and
*which contextual feature governs the switch.* Constraints are scoped, not absolute: *"use `deno task`, not `npm
install`"* is true **in this repo**, not everywhere. Capturing that scope is a form of **world modeling** - and
it is the same thing as knowing when to trust a local expert vs escalate. Everything else (frozen experts,
shadows, critic, gate) is the apparatus that serves this. We will ship a harness too - but here it is the
*surface*, not the substance.

---

## How it learns (and why it's not just distillation)

Antumbra learns from **verifiable outcomes in your environment**, not by imitating a teacher's text:

- The **environment is the truth.** For a coding agent on your repos: the test that passes, the command that
  actually works, the build that goes green. That is the reward.
- A **critic** (which *may* be a flagship model, or your own verifiers) turns a raw failure into a dense,
  *diagnostic* signal - *"`npm install` failed because this is a Deno project; use `deno task`"* - and names the
  **governing feature** of the boundary (Deno-vs-npm). That is counterfactual scope extraction (ADR-0004).
- You **train on the verified outcome, not the critic's words.** That keeps the learning grounded in your data
  (and clear of "trained on a provider's outputs"). The flagship, if used, is an *accelerator for cold-start* -
  optional, because your own repos are a more authoritative teacher about *your* conventions than any model.

This makes **coding-over-your-own-repos the ideal first domain**: maximally verifiable (you can *run* it),
maximally context-scoped (per-repo conventions are textbook boundaries), and the data is yours.

---

## Memory, tenants, and the runtime surface

Antumbra absorbs the memory and runtime role of a separate agent engine, rather than running alongside one.

- **Penumbra memory** (*ADR-0012*) — a first-class, tenant-scoped memory store (`world`/`bank`/`opinion`
  networks, HNSW recall, reinforcement, provenance). It is the **bootstrap** (existing memories seed the
  population without a cold start, via *memory-import*) and the **consolidation source**: an offline "sleep"
  scores memories (recurrence × verifiability × stability), graduates the trusted ones into experts with an
  interleaved **replay** buffer, and **retires** an expert when a consolidated memory is later contradicted. The
  store + population become one circulatory system: memory → weights → forgetting.
- **Engine-enforced multi-tenancy** (*ADR-0013*) — isolation lives in the SurrealDB engine (record access +
  `PERMISSIONS WHERE tenant_id = $auth.tenant`), not in handler code: a forgotten filter cannot leak. `$auth`
  carries `(tenant, user)`; the population (experts/router) is the **shared umbra**, memory is the **private
  penumbra**, and an owner role can read across tenants to profile and train.
- **Compartments** (*ADR-0014*) — named, ownable "latent-spaces" of memory: the unit of organization, deletion,
  sharing (`reference`/`link` grants, user-to-user, engine-enforced), and the natural **training unit** — a
  private compartment consolidates into a *private* expert. The **antumbra** itself can *propose* compartments by
  clustering the penumbra (the same boundary machinery that does routing). Every memory carries who/which-machine
  provenance.
- **MCP server** (*ADR-0015*) — `antumbra-mcp`, a Rust Model Context Protocol server (12 tools: memory, graph,
  compartments, `route`), each engine-isolated to the bound `(tenant, user)`. The surface an agent talks to.

## Status & shape (honest)

- **Research project, not a product.** Every milestone is a falsifiable experiment with a kill criterion.
- **All-Rust, single process.** Data in SurrealDB via `surql-rs`; training (QLoRA adapters **and** the gate)
  via a DIY `candle` path; inference via `llama-cpp-2` / `mistral.rs` (a seam, not yet wired).
- **v0 = shared-base adapters on one RTX 3090 Ti (24 GB):** one frozen, code-capable base + a growing library of
  frozen LoRA experts + a boundary-conditioned **coverage** gate (the learned latent mixer is the north star).

**Validated so far (toy scale, 2026-06-03):**

- The trainer **learns** on a real GPU: RAFT lifts pass-rate to 1.0 under both a convention reward and a verifier
  that *executes* the generated code (`0.38 -> 1.00`).
- A real candle BERT embedder drives a gate that **routes** to the right specialist and **escalates
  out-of-scope** queries - using relative coverage, not an absolute similarity floor.
- Expert capability vectors are **learned from evaluated behavior** (the tasks an expert provably solved), not
  hand-written labels - shown with three GPU-trained specialists in one population.
- The **keystone** (ADR-0004) composes end-to-end: counterfactual search recovers C', the actionable boundary
  persists, and the gate inhibits routing **inside the failure scope only**.

**Not yet:** small corpora / few experts (no generalization or forgetting test); the real `AcceptabilityProbe`
and serving engine; the learned latent gate; GRPO and 4-bit quantized training; heterogeneous composition.

**North star = a heterogeneous composed model** (genuinely separate frozen experts wired by learned
cross-attention bridges) - *ADR-0009*. The v0 gate, boundary engine, loop, and substrate all carry over; only
the composition substrate changes.

---

## Build & run (v0)

The substrate, critic, gate, boundary engine, and the durable generational loop are implemented and tested in
Rust against **surql-rs** on the SurrealDB 3.x driver (builder-only - no hand-written SurrealQL). The real
candle **Qwen2.5-Coder + LoRA trainer** (RAFT over verified outcomes, ADR-0010) is implemented behind a `models`
feature; the default loop runs with a demo trainer so everything is exercisable without a GPU.

```bash
cargo test                                   # whole workspace, green (incl. the keystone test)

# no-GPU demo: drive the durable loop with the fake trainer, then inspect
cargo run -p antumbra-cli -- schema                              # print generated DDL
cargo run -p antumbra-cli -- --url surrealkv://./data/a.skv loop --generations 3
cargo run -p antumbra-cli -- --url surrealkv://./data/a.skv status

# real training + routing (needs a CUDA GPU + ~3 GB Qwen weights, and python for
# exec verifiers) - see the guide below. Train a specialist, then route to it.
cargo run -p antumbra-cli --features models,cuda -- --url surrealkv://./data/a.skv \
  train --corpus corpora/arith.json --run arith --generations 1
cargo run -p antumbra-cli --features models,cuda -- --url surrealkv://./data/a.skv \
  route "add two integers and return the sum"   # routes to the specialist, or escalates if out of scope
cargo run -p antumbra-cli --features models,cuda -- --url surrealkv://./data/a.skv \
  ask "Write a Python function add(a, b) that returns their sum."   # route -> load adapter -> generate
```

See [architecture §7](docs/architecture.md#7-repo-structure-greenfield) for per-crate status and
**[Running the trainer](docs/running-the-trainer.md)** for the GPU run.

---

## Documentation map

- **[Architecture](docs/architecture.md)** - system, substrate, decision chain, training/data flow, schema.
- **[Technical Reference](docs/technical-reference.md)** - crate map, domain model, the port seams, the
  algorithms as coded, the build matrix, and validation results.
- **[Diagram Atlas](docs/diagrams.md)** - every diagram in one place: concept, system, flow, schema, per-ADR.
- **[Glossary](docs/glossary.md)** - every ADR-0010 acronym, with a diagram each, and why it is called candle.
- **[Running the trainer](docs/running-the-trainer.md)** - the CUDA-13 / Windows GPU recipe.
- **[Experiment Ledger](experiments/README.md)** - every falsifiable validation: claim, method, result, kill
  criterion, reproduce command.
- **[Architecture Decision Records](docs/adr/README.md)** - every load-bearing decision, ADR-0001 … ADR-0015
  (incl. Penumbra memory, tenant isolation, compartments, and the MCP runtime surface).
