# Roadmap

Forward-looking items that are **scoped but deliberately deferred** — captured so
they are not lost, ordered roughly by when they unblock. Shipped work lives in
the [ADRs](adr/) and the [experiment ledger](../experiments/README.md); this file
is only the *not-yet-built* queue.

## Deferred — networked / multi-device

### R-1 · Collector / sync: local-embedded penumbra ↔ remote-authoritative store
**Status:** deferred until the single-node networked surface is proven end-to-end.
**Shape:** an edge device keeps its **embedded** penumbra (`surrealkv://`, single
writer — see ADR-0015) and a background **collector** syncs it to a **remote
authoritative** SurrealDB (`ws://`), so a fleet of devices shares one source of
truth without each opening the embedded file. Mirrors the proven pattern in
`many-tiny-stuff/tinytropolis/sync` (`connect_local` surrealkv + `connect_authoritative`
ws://, a supervised reconnect/backoff worker). This is the genuine multi-device
story behind "userA's memories become visible to userB's agent across machines."
**Unblocks:** multi-device compartment sharing; the fleet (ADR-0006/0009).
**Depends on:** the networked MCP surface working end-to-end (R-3); a conflict /
ordering policy for two-way sync (last-write-wins vs CRDT-ish per-field).

### R-2 · Live propagation (real-time awareness)
**Status:** deferred.
**Shape:** server→client push when a **shared compartment** changes (a grantee's
agent learns of new/planned memories without polling). SurrealDB `LIVE SELECT`
detects the change; it is delivered as an MCP server notification over the
streamable-HTTP SSE stream. **Tension to resolve:** the networked transport runs
in stateless-JSON mode with a serialized authed section (ADR-0015); live
notifications need a persistent per-subscriber SSE stream, so the lock must wrap
only the DB POSTs, never the idle stream. Today the surface is recall-on-demand.
**Depends on:** R-3; a subscription registry (which identity watches which
compartment).

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
