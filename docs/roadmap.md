# Roadmap

Forward-looking items that are **scoped but deliberately deferred** — captured so
they are not lost, ordered roughly by when they unblock. Shipped work lives in
the [ADRs](adr/) and the [experiment ledger](../experiments/README.md); this file
is only the *not-yet-built* queue.

## Deferred — networked / multi-device

### R-1 · Collector / sync: local-embedded penumbra ↔ remote-authoritative store
**Status:** BUILT (crate `antumbra-sync`, CLI `sync`; GPU-free, validated 2026-06-05).
**Shape:** an edge device keeps its **embedded** penumbra (`surrealkv://`, single
writer — see ADR-0015) and a **collector** reconciles it with a **remote
authoritative** SurrealDB (`ws://`), so a fleet shares one source of truth without
each opening the embedded file. Mirrors the supervised reconnect/backoff worker in
`many-tiny-stuff/tinytropolis/sync`. **Conflict policy chosen: bidirectional
last-write-wins** by each row's RFC3339 version field (`memory.updated_at`,
`created_at` for the write-once tables), compared in Rust so no datetime crosses
into a query. Strict-`>` propagation makes the two-way flow converge and self-
terminate (no echo). Replicates `memory`, `memory_edge`, `compartment`, `grant`
(experts/adapters are on-disk safetensors, out of scope). Runs as a root/owner
session spanning tenants; per-tenant isolation is preserved by each row's
`tenant_id`. Generic row access went into `antumbra-store::repo::sync` (surql-rs
builders only — `list_rows`/`put_row`/`row_id`, reusing the record-id target
verbatim to dodge v3 escaping). End-to-end: bidirectional seed (1 push / 1 pull),
convergence (0/0), and LWW propagation (1/0) all verified through the CLI on two
persistent `surrealkv://` stores.
**Known gaps:** delete propagation (needs tombstones); incremental cursors (today
each cycle scans full tables — fine at penumbra scale).
**Unblocks:** multi-device compartment sharing; the fleet (ADR-0006/0009); R-2.
**Was deferred on:** a conflict/ordering policy — now decided (LWW).

### R-2 · Live propagation (real-time awareness)
**Status:** ENGINE BUILT (`antumbra-sync::propagate`, validated 2026-06-05); MCP
SSE delivery remaining.
**Shape:** server→client push when a **shared compartment** changes (a grantee's
agent learns of new/planned memories without polling). SurrealDB `LIVE SELECT`
detects the change; it is delivered as an MCP server notification over the
streamable-HTTP SSE stream.
**Done:** the detection + routing engine. `antumbra-store::repo::sync::watch_table`
wraps surql-rs `LiveQuery` into a tokio change-feed channel — **proven to deliver
on the embedded engine** (the key unknown). `propagate::watch_shared_memories`
parses each memory change, resolves its **audience** (`compartment` owner +
grantees, via new `repo::compartment::get`/`list_grants`), and emits a routed
`MemoryChange { action, tenant, compartment, memory, recipients }`. Tested
end-to-end: a write into a shared compartment reaches the owner and the grantee.
**Remaining (the transport last mile):** deliver a `MemoryChange` to each
recipient's live MCP client as an SSE notification. **Tension to resolve:** the
networked transport runs in stateless-JSON mode (rmcp `StreamableHttpService`,
single shared single-writer connection, serialized authed section — ADR-0015);
live notifications need a persistent per-subscriber SSE stream, so the lock must
wrap only the DB POSTs, never the idle stream. Needs a subscription registry
(identity → active session) and a decision on stateful streaming vs the current
stateless mode.
**Depends on:** R-3 for the networked surface end-to-end.
**Known gaps:** deletes are not routed (the notification payload carries no
compartment on delete) — same tombstone gap as R-1.

## GPU-gated wiring

### R-4 · Wire `MultiAdapterServe` into the MCP `answer` tool
**Status:** seam shipped, GPU wiring deferred.
**Shape:** the `answer` tool (route → serve through the covering expert) takes an
injectable `Serve` engine and is CPU-proven with a fake (`EchoServe`). What's
left is GPU-only: build a resident `MultiAdapterServe` from the population
(register each expert's adapter), wrap it in `Arc`, and pass it to the per-session
`McpServer`s — so a server built `--features models` answers for real. Needs the
3090 Ti, and a decision on the per-tenant private-adapter registry (the engine
sees all adapters; routing already scopes which a session may pick).

### R-5 · Live harness metabolization
**Status:** first increment shipped (normalized-export → capture tasks via
`antumbra metabolize`); live ingestion deferred.
**Shape:** instead of a hand-exported trace file, pull successful orchestration
traces **live** from a running harness — Kushtaka's task-trace / behavior-graph /
loop-run MCP tools — and metabolize on a cadence. Plus behavior-graph-*structure*
aware metabolization (learn the graph's decomposition, not just its collapsed
outcome). The brain absorbs the scaffold continuously, so the harness shrinks.

## Foundational — make the current surface provably work

### R-3 · Networked MCP end-to-end validation
**Status:** done in unit form. The JWT core, auth boundary, per-tenant isolation
(through the real MCP tools), and the authenticated happy-path (a valid token
verifies → signin → rmcp dispatches `initialize` → 200) are all tested. The one
remaining checkpoint is a **live multi-tenant deployment** against a real `ws://`
SurrealDB — a field test, not a unit gap. R-1 and R-2 build on that holding.
