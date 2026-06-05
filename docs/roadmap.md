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

## Foundational — make the current surface provably work

### R-3 · Networked MCP end-to-end validation
**Status:** in progress. The JWT core, auth boundary, and per-tenant isolation are
unit-tested; the authenticated happy-path (a valid token reaching a real tool
call) is the closing gap, and a live multi-tenant deployment against a `ws://`
SurrealDB is the field test. R-1 and R-2 build on this holding.
