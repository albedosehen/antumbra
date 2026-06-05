# ADR-0015 - The MCP server: Antumbra's runtime surface

**Status:** Accepted (stdio single-tenant + networked JWT multi-tenant) · **Date:** 2026-06-04 · **Related:** 0005 (gate/route), 0012 (Penumbra), 0013 (identity), 0014 (compartments)

> **Networked multi-tenant surface (2026-06-05).** `--http <addr>` serves the same tools over rmcp's
> streamable-HTTP transport (axum), multi-tenant **per request**: each request carries a signed JWT whose
> `tenant`/`user` claims become `$auth` (the decision was JWT claims over a token→identity lookup — the verified
> token *is* the identity, so a leaked token grants exactly its claimed scope). `auth.rs` verifies the bearer
> token (HS256 secret or RS256 PEM public key; mandatory `exp`; optional `aud`) and `http.rs` resolves identity
> *before* dispatch — rmcp's service factory takes no request context, so the handler verifies the token, binds
> the identity, and delegates.
>
> **Isolation holds on embedded too (corrects an earlier note).** The embedded engine *does* enforce
> record-access PERMISSIONS once a session is signed in — proven in `antumbra-store`'s embedded tests (an `alpha`
> session cannot see `beta`'s rows even on an unfiltered query). The only unenforced mode is the owner/root path,
> used solely for schema + provisioning. The real embedded constraint is **single-writer**: `surrealkv` admits
> one connection (a 2nd is refused — regression-tested), so the server holds **one** shared connection and signs
> it in per request as the JWT identity, **serializing** the authenticated section with a lock. Two tenants over
> that one connection see only their own rows — proven through the real MCP tools
> (`shared_connection_isolates_tenants_under_signin`). The cost is serialization of the authed section (fine for
> an edge device; a high-concurrency deployment points `--url` at a `ws://` server and the model is unchanged).
> The stateless JSON response mode keeps each `handle` bounded so the lock never spans a long-lived stream.
> Verified: JWT core (7), HTTP auth boundary (3, 401 before any store work), embedded single-writer + serialized
> isolation (3), and the **authenticated happy-path** (a valid token → signin → rmcp dispatches `initialize` →
> 200 with `serverInfo`, driven through the router via `oneshot`). Deferred to the [roadmap](../roadmap.md):
> R-1 the collector/sync (local-embedded ↔ remote-authoritative, the multi-device story) and R-2 live
> propagation (server→client SSE push on shared-compartment change; today recall-on-demand).

## Context

For Antumbra to *replace* the runtime role of a separate agent/memory engine, an agent (Claude Code, any MCP
host) must be able to talk to it directly. The harness paradigm ("orchestration on top of a frozen brain") is
what Antumbra metabolizes into weights (ADR-0001/0004); the **transport** to reach the brain + memory is the one
piece of the old engine worth porting as-is.

## Decision

`antumbra-mcp` — a Rust **Model Context Protocol** server (the official `rmcp` SDK, stdio transport) over the
Penumbra and the population. v0 serves a single `(tenant, user)`: it connects as owner, provisions the principal,
signs the session in (ADR-0013) so every tool is engine-isolated, and ensures the session's default compartment.
The tool surface:

| Group | Tools |
|---|---|
| Memory | `store_memory` (compartment + provenance), `recall_memories` (semantic), `reinforce_memory`, `forget_memory`, `list_memories` |
| Graph (ADR-0014) | `relate_memories` (typed edges), `get_neighbors` |
| Compartments (ADR-0014) | `create_compartment`, `list_compartments`, `share_compartment` (reference/link), `revoke_compartment`, `propose_compartments` (cluster the unorganized pool; `apply` to persist as `Origin::Proposed`) |
| Brain (ADR-0005) | `route` (which expert covers a task, ranked, or escalate — pure-arithmetic gate inference) |

The client never passes `tenant`/`user`; the server resolves them from the bound session and applies them. The
real embedder (candle BERT) lands under `--features models`; a byte-histogram fake otherwise.

## Consequences

- **Positive:** Antumbra is usable as an agent's memory + routing engine today; one MCP server, one engine, one
  tenant boundary; consolidate/recall/route all hit a single store.
- **Negative:** the networked surface **serializes** the authenticated section over its single shared connection
  (correct and isolated on embedded *and* networked, but not concurrent — a high-throughput deployment wants a
  `ws://` server, where the same model still holds); **live propagation** (push on shared-compartment change) is
  not yet built (recall-on-demand today); and `ask` (serve *through* the routed expert) plus the
  consolidate/retire owner ops are not yet on the surface (owner ops belong on a future admin surface, not the
  tenant session).
- **Neutral:** `get_neighbors` 1-hop today (native N-hop graph traversal deferred, ADR-0014).

## Alternatives considered

- **A bespoke HTTP/JSON API.** Rejected for v0: MCP is the emerging standard agents already speak; the official
  Rust SDK gives the protocol for free.
- **Re-implement the old engine's full tool surface (loops, behavior graphs, code intel).** Rejected: those are
  the *harness* to metabolize, not to clone (ADR-0001). Only the data/retrieval + routing surface is ported.

## Validation

In-process integration test (store → recall → reinforce → list → forget; compartment create/list/store/share/
revoke) + a real MCP stdio smoke (initialize + `tools/list` returns all 12; store/recall round-trips). *Kill
criterion:* an MCP client cannot drive memory + routing against a real workspace → the runtime surface is not
usable.
