# ADR-0015 - The MCP server: Antumbra's runtime surface

**Status:** Accepted (stdio single-tenant + networked JWT multi-tenant) · **Date:** 2026-06-04 · **Related:** 0005 (gate/route), 0012 (Penumbra), 0013 (identity), 0014 (compartments)

> **Networked multi-tenant surface (2026-06-05).** `--http <addr>` serves the same tools over rmcp's
> streamable-HTTP transport (axum), multi-tenant **per request**: each request carries a signed JWT whose
> `tenant`/`user` claims become `$auth` (the decision was JWT claims over a token→identity lookup — the verified
> token *is* the identity, so a leaked token grants exactly its claimed scope). `auth.rs` verifies the bearer
> token (HS256 secret or RS256 PEM public key; mandatory `exp`; optional `aud`) and `http.rs` resolves identity
> *before* dispatch — rmcp's service factory takes no request context, so the handler verifies the token then
> looks up (or lazily builds) a per-identity session whose `McpServer` is already `signin`'d as that
> `(tenant, user)`, and delegates. Each identity owns its own store connection (signin binds the session).
> **Caveat:** true enforcement of the engine ACL over the network requires pointing `--url` at a real SurrealDB
> server (`ws://`) where record-access PERMISSIONS are live; on the embedded kv engine the in-process session is
> effectively root, so the networked server's isolation guarantee is only as strong as the engine it is given.
> Verified: JWT core (7 tests) and the HTTP auth boundary (rejects missing/garbage/non-bearer with 401 before any
> store work). Still to do: the authenticated happy-path against a live deployment, and "live propagation"
> (server→client SSE notifications when a shared compartment changes) — today it is recall-on-demand.

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
- **Negative:** the networked surface's isolation is only enforced when `--url` points at a real SurrealDB
  server (the embedded kv session is in-process root); **live propagation** (push on shared-compartment change)
  is not yet built (recall-on-demand today); and `ask` (serve *through* the routed expert) plus the
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
