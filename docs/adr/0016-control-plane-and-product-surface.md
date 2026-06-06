# ADR-0016 - Control plane & product surface

**Status:** Proposed · **Date:** 2026-06-06 · **Related:** 0009 (composition), 0012 (memory), 0013 (identity), 0014 (compartments), 0015 (MCP surface)

## Context

The engine is built and validated: a population of frozen experts, the
boundary-conditioned gate, engine-enforced multi-tenant memory, the MCP runtime
surface (stdio + networked JWT/SSE), sync, and harness metabolization. What it does
*not* have is a **control plane** — a way for a non-operator to see, steer, and
onboard — and a **product packaging** for the three ways Antumbra is meant to ship:
offline-private, hosted-but-private SaaS, and bespoke contract.

A predecessor agent engine reached users through a web dashboard: 2D/3D memory
exploration, population/stats observability, a behavior mixer, knowledge documents,
remote agent interaction, and a signup/onboarding flow. Antumbra's thesis changes
*what* needs cloning. Orchestration scaffold — behavior graphs, autonomous loops,
planning — is **metabolized into weights** (ADR-0001, R-5), so it is superseded, not
re-implemented as runtime features. But observability and control are about the
*user's* relationship to the system; metabolization does not provide them. They are
a real, separate gap.

A second, concrete gap: lifecycle-hook clients (the SessionStart/Stop scripts that
wire a coding agent in — see [`/hooks`](../../hooks) and
[`docs/integration.md`](../integration.md)) are non-interactive and need a
**long-lived credential**, but the networked surface today mints a per-request JWT
(ADR-0015). And the embedder is a build-time feature, not a per-workspace runtime
choice, so "bring your own embedder" is not yet a product control.

## Decision

Treat the **control plane as a thin, separate surface over the existing MCP/store
APIs**, not a second source of truth. Specifically:

1. **Read-only web dashboard first** — population, experts, fitness, route hit-rate,
   escalation/cost-avoided stats, memory recall, and the compartment/`memory_edge`
   graph (2D, then 3D). It calls the same engine-isolated MCP tools an agent does;
   the engine ACL (ADR-0013) governs it, so the dashboard inherits isolation for
   free.
2. **Hook token** — a long-lived, scope-bound credential (API-key-style) for
   non-interactive clients, exchanged for / equivalent to the JWT claims, so hooks
   and the dashboard authenticate without an interactive mint.
3. **Per-workspace embedder config** — promote the `Embedder` port to a runtime,
   per-tenant setting (model id / local endpoint), keeping the embedding step on the
   tenant's side and HNSW dimensions consistent.
4. **Knowledge documents** — a first-class `document` type (ingest → chunk → embed →
   recall) distinct from episodic memory, surfaced in recall and the dashboard.
5. **Interactive control after read-only lands** — an **expert mixer** (compose
   experts + weights into a saved serve profile; the user-facing form of ADR-0009's
   composition) and driving a connected agent's `answer`/`route` from the web.
6. **Hosted onboarding** — signup, tenant provisioning, and the setup flow that wraps
   the `hooks/` templates; billing for the SaaS tier. The offline tier needs none of
   this — `hooks/` + the CLI are its onboarding.

Sequence captured as roadmap items P-1…P-5 (see [`roadmap.md`](../roadmap.md)).

## Consequences

- The dashboard is **additive and ACL-safe**: built on the isolated MCP surface, it
  cannot see across tenants any more than an agent can.
- Antumbra ships in three tiers off one engine; only transport/identity/who-runs-it
  differ.
- The predecessor engine winds down as Antumbra reaches parity on the *control*
  surface while superseding it on the *learning* surface — no parallel enhancement.

## Alternatives

- **Port the predecessor's dashboard wholesale.** Rejected: it assumes orchestration
  scaffold (graphs/loops/plans) as runtime features Antumbra deliberately metabolizes
  away; a 1:1 port would re-introduce the cost the thesis removes.
- **No web surface; CLI/TUI only.** Rejected for the hosted/bespoke tiers: a
  non-operator buyer needs to *see* the population and memory and onboard without a
  terminal. (The offline tier remains fully CLI-usable.)

## Validation

Falsifiable kill criterion: if the read-only dashboard cannot be built purely on the
existing engine-isolated MCP tools — i.e. it needs a privileged, ACL-bypassing path
to render a tenant's own population and memory — then the "thin surface over the same
APIs" decision is wrong and the control plane needs its own access model.
