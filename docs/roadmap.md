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
**Delete propagation (done for memory):** `forget` now **soft-deletes** — a memory
becomes a tombstone (`deleted_at` set, `updated_at` bumped) rather than vanishing,
so the deletion is the trace's newest version and propagates under LWW instead of
resurfacing from the other side (validated: a tombstone pushes and the trace does
not resurrect). Read paths hide tombstones; `memory::purge` hard-removes them past
a grace window (run wider than the sync interval, so every replica saw the
tombstone first — resurrection-safe GC, the `gc_grace_seconds` pattern).
**Known gaps:** compartment/grant/expert deletes are still hard deletes (don't
propagate — grant *revoke* not propagating is a security follow-up); incremental
cursors (each cycle scans full tables — fine at penumbra scale).
**Unblocks:** multi-device compartment sharing; the fleet (ADR-0006/0009); R-2.
**Was deferred on:** a conflict/ordering policy — now decided (LWW).

### R-2 · Live propagation (real-time awareness)
**Status:** DONE (engine + MCP SSE delivery, end-to-end validated 2026-06-05).
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
**Delivery (done):** the MCP transport now runs in rmcp **stateful (SSE) mode**
(`http::server_config`), so a client's GET stream carries server-initiated
notifications. The **ADR-0015 tension was a misframing** — the auth lock is held
only while `handle` builds a response; an SSE stream is MCP transport state that
does no DB work and streams after the handler returns, so it never holds the DB
connection (validated: all 20 transport tests still pass under stateful mode).
`http::spawn_live_propagation` runs one owner-mode `LIVE` subscription, resolves
audience under the auth lock in owner mode (validated: an owner-registered watch
still delivers a tenant-authored write — per-request signin does not starve the
feed), and pushes each change to recipients' captured peers. The **subscription
registry** (`notify::PeerRegistry`, identity → live `Peer`s) is populated by
`McpServer::on_initialized` and fanned out as a `notifications/message`
(`type: antumbra/memory_changed`), pruning closed peers.
**End-to-end validated:** `http::tests::live_notification_reaches_a_grantees_stream`
drives the real `/mcp` router through the full stateful handshake (initialize →
initialized → GET SSE) as a grantee, has another user write into the shared
compartment, and asserts the `antumbra/memory_changed` notification arrives on the
grantee's SSE stream — exercising the actual transport, peer capture, watcher, and
push (no real socket / external client needed).
**Deletes now routed:** because a `forget` is a soft-delete (an *update* carrying
`deleted_at`), the change still has the compartment, so `propagate::resolve_change`
resolves its audience and relabels the action `Delete` — grantees are notified of
forgets, not just writes (validated end-to-end).

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
**Status:** DONE — validated live against a real `ws://` SurrealDB v3 (2026-06-05).
On top of the unit coverage (JWT core, auth boundary, per-tenant isolation through
the real tools, the authenticated `initialize` happy-path), the **live
multi-tenant field test** now passes: two JWT tenants drive the real `/mcp` surface
over the network against a root-authenticated SurrealDB; tenant A stores a memory,
A sees it, **B does not see it via `list` or `recall`** — engine-enforced isolation
over the wire. Reproduce with `docs/r3_isolation_probe.py` (recipe in its header).

Two real gaps surfaced and were fixed by doing the live test (not visible on the
embedded engine):
- **DB credentials.** `connect` gained `--db-user`/`--db-pass` (env
  `ANTUMBRA_DB_USER`/`PASS`) so the server can log in to an authenticated remote;
  it then signs in per request as each tenant on top.
- **Owner mode on a remote.** `invalidate` drops to *anonymous*, which equals
  owner only on embedded — on an authenticated `ws://` server it has no
  permissions, so provisioning failed. New `Store::signin_root` re-signs-in as
  root on a credentialed remote (and falls back to `invalidate` on embedded); the
  HTTP layer uses it wherever it needs the cross-tenant owner view (provisioning,
  the R-2 watcher).
