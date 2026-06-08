# Product surface: how Antumbra is used, and the gap to close

This document answers two questions: **who uses Antumbra and why**, and **what is
still missing** for it to fully supersede a separate agent-memory engine (the
control plane: dashboard, knowledge documents, onboarding, which a hosted product
needs but the engine itself does not).

## Positioning: the memory layer that *learns*

The "memory for AI agents" category (Mem0, Letta (MemGPT), Zep, Cognee) are
**retrieval/context layers**: they store facts and inject them back into a frozen
model's context window. They make a static model *remember*; they do not make it
*better*. Every run re-pays the prompt/loop/lookup cost against the same frozen base.

Antumbra is a different layer. It **metabolizes verified outcomes into weights**: a
recurring task you've completed and checked becomes a frozen LoRA expert in a local
population, routed to automatically next time. The orchestration scaffold (loops,
behavior graphs, prompts, memory lookups) shrinks into the model as competence
accrues. This is the established anti-forgetting pattern: a population of frozen
LoRA experts with routing (Mixture-of-LoRA-Experts), served S-LoRA-style so hundreds
of adapters run on one GPU, combined with trajectory distillation (Structured Agent
Distillation, arXiv:2505.13820) to turn *behavior* into *weights*.

**Where Antumbra is genuinely distinct:**

- **Learns into weights, not context.** Capability compounds instead of re-paying
  scaffold cost every run.
- **The scaffolding shrinks.** Loops/graphs/prompts get absorbed; the opposite of a
  memory layer, where orchestration cost is permanent overhead.
- **A governed population of experts**, not one adapter, and frozen, so a learned skill
  is never silently overwritten.
- **Memory + skill in one private engine**, with **engine-enforced** multi-tenant
  ACL the incumbents have no equivalent of.
- **Privacy is structural, not a setting**: fully offline or hosted-but-private,
  bring-your-own-embedder, data and learned skills never leave your boundary.

**Honest caveats (when *not* to use it):** it wants a local GPU (real CapEx vs a
cloud API call); there is an investment period where experts are immature; and the
bet (small frozen experts matching a frontier model *within scope*) holds for
repetitive, bounded, verifiable work and **breaks on open-ended novelty**. Antumbra
is a compounding specialist engine, not a general-purpose oracle.

## Who it's for, and the three ways to run it

1. **Offline / private (the floor).** A solo developer or an air-gapped, regulated
   box. Embedded `surrealkv://` store, stdio or loopback MCP, one identity. Nothing
   leaves the machine. This is the default and the privacy guarantee.
2. **Hosted, still private (SaaS).** A team or fleet shares one brain without each
   running the infrastructure. Networked HTTP/SSE, JWT-per-request, engine-enforced
   multi-tenancy, device sync, live propagation. Hosted by us; each consumer's data
   and experts are isolated in the engine.
3. **Bespoke contract.** Businesses that want this but want it *implemented and
   operated for them*: the engine plus integration into their repos, verifiers,
   and compliance boundary.

The engine is identical across all three; only transport, identity, and who runs
the box differ.

## Two kinds of parity: **absorb** vs **build**

A predecessor agent engine (Kushtakas) exposed memory, code intelligence, planning,
behavior graphs (a composable "behavior mixer"), an evaluation harness, autonomous
loops, per-workspace embedders, and a web dashboard (2D/3D memory views, stats,
knowledge documents, remote agent control, onboarding). Antumbra does **not** clone
all of it; its thesis is to *absorb* the orchestration and *build* the control plane.

### Absorbed: superseded by metabolizing (not re-implemented as runtime scaffold)

- **Behavior graphs / the behavior mixer become a population of metabolized experts.**
  Where the predecessor *executes* a composed graph every time, Antumbra metabolizes
  a successful, recurrent graph (including its step decomposition) into weights via
  `antumbra metabolize`, and composes experts at serve time (the heterogeneous
  composed model is the learned cross-attention end state; a linear adapter blend is
  the precursor today). The
  composition surface a user wants becomes an **expert mixer**, not a graph mixer.
- **Autonomous loops / planning become the generational training loop + the `answer`
  tool.** The agent's learned competence replaces hand-built iteration where it can;
  out-of-scope tasks escalate.
- **Multi-tenancy becomes engine-enforced ACL**, stronger than app-side
  scoping, validated over the wire.
- **Evaluation harness becomes the experiment ledger + `evaluation_run`**, on the
  SurrealDB substrate.

### To build: the control plane and product surface (the real gap)

These are about the **user's ability to see, steer, and onboard**; they are not
orchestration scaffold, so metabolization does not provide them. They are what a
hosted product needs:

| Capability (predecessor) | Antumbra today | Gap to close |
|---|---|---|
| Web dashboard | CLI + TUI only | A web UI over the existing MCP/store surface |
| 2D/3D memory graph explorer | `memory_edge` graph in the store; no viz | Render the graph (recall, edges, compartments) in the browser |
| Stats / observability | `status` CLI | Web stats: population size, fitness, route hit-rate, escalation rate, cost-avoided |
| Behavior mixer (compose behaviors) | `compose_adapters` precursor | An **expert mixer** UI: pick experts + weights, preview, save a composed serve profile |
| Knowledge documents | memory networks only | A `document` first-class type (ingest → chunk → embed → recall), distinct from episodic memory |
| Remote agent interaction from the dashboard | none | Drive a connected agent (the `answer`/`route` tools) from the web |
| Per-workspace bring-your-own embedder | `Embedder` port (Bert/fake) | Expose an embedder **config per workspace** (set model/endpoint), not a build-time choice |
| Onboarding: signup, docs site, agent setup guide | `docs/` + `scripts/hooks/` templates | A signup/setup flow for the hosted tier; the [`scripts/hooks/`](../scripts/hooks) guide is the offline start |
| Hook / non-interactive auth | per-request JWT, JSON-RPC `/mcp` only | A long-lived **hook token** + a REST `/mcp/call` shim for lifecycle-hook clients |

## Bring your own embedder

The engine never ships an opinionated embedder. Recall quality and the privacy of
the embedding step both belong to the user: the `Embedder` port already abstracts it
(a local candle BERT today, a fake for tests). The parity step is to make it a
**per-workspace runtime config** (model id / local endpoint) instead of a build-time
feature, so each tenant controls (and keeps local) the model that touches their
content. Dimensions stay consistent with the HNSW index the store provisions.

## Phased plan to close the gap

Tracked as roadmap items (see [`roadmap.md`](roadmap.md)); decision recorded in
[the control plane and product surface](adr/0016-control-plane-and-product-surface.md).

1. **Hook auth + REST shim + per-workspace embedder config**: a hook token, a REST
   `/mcp/call` convenience endpoint for shell hooks, and runtime BYO-embedder;
   unblocks real onboarding.
2. **Read-only web dashboard** over the MCP surface: population, experts, fitness,
   memory recall, the compartment/edge graph (2D first, 3D after). Pure observability.
3. **Knowledge documents**: a `document` type and ingest pipeline distinct from
   episodic memory, surfaced in recall and the dashboard.
4. **Interactive control**: the expert mixer (compose + save serve profiles) and
   driving a connected agent's `answer`/`route` from the dashboard.
5. **Hosted onboarding**: signup, tenant provisioning, the setup flow that wraps
   the `scripts/hooks/` templates; billing for the SaaS tier.

The engine is built; this is the surface that makes it *usable by a non-operator and
sellable as a product*, without competing with the predecessor, which winds down as
Antumbra proves superior.
