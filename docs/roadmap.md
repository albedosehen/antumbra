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
**Deletes propagate (memory, grant, compartment):** all three replicated tables
now soft-delete via tombstones (`deleted_at` + bumped `updated_at`), so a forget /
revoke / compartment-delete out-versions a stale live row under LWW and cannot
resurrect; each has a grace-windowed `purge`. A deleted **compartment** also fails
closed at the engine — its owner subquery in `MEMORY_SELECT_RULE`/`EDGE_LINK_RULE`
excludes `deleted_at` rows, so its memories go invisible at once. **Expert deletes
stay hard by design:** experts are not replicated (on-disk adapters, out of DB
sync), so a tombstone would serve no propagation purpose — the standard rule that
tombstones are a *replication* concern (Cassandra/Couchbase). If experts ever sync,
they'd need one then.
**Incremental cursors (done):** a long-running collector now tracks a per-table
high-water mark (`reconcile::Cursors`) and each cycle fetches only rows past it
via an engine-side `updated_at > since` (string `>`, no datetime in the query —
chrono's uniform-UTC RFC3339 is lexicographically monotonic with time), instead
of scanning the whole table. A small **lookback window** (`SyncConfig.lookback`,
the CDC "delay") re-includes the boundary so a slightly-stale write is not skipped;
re-reconciling settled rows is a no-op under idempotent LWW. The correctness rests
on an asymmetry: a row in only one side's window is necessarily newer than the
other side's copy (≤ the floor), so it wins without a cross-side compare. The
watermarks reset on reconnect (one full scan = the backstop that re-syncs anything
missed while disconnected); `reconcile_all` (the one-shot `sync --once`) stays a
full scan. Version-field indexes (`*_updated_at_idx`, `memory_edge_created_at_idx`)
back the range filter engine-side. Research-corroborated (Kafka Connect JDBC
timestamp-mode watermark + delay window). Validated over docker `ws://`
(`crates/antumbra-sync/tests/ws_incremental.rs`, gated on `ANTUMBRA_SYNC_WS`): the
real v3 engine's string `>` converges (seed → incremental push → remote-only pull
→ settle).
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

> **Correction + resolution (2026-06-06):** R-3's first pass enforced cross-tenant
> isolation only app-side (the `tenant_id` filter); the **engine** ACL was bypassed
> on `ws://` because requests ran on the root connection. **R-6 (below) fixes this**
> — requests now run on a scoped, non-root per-session connection, so the engine
> enforces both cross-tenant AND intra-tenant compartment isolation on a remote
> (validated: `docs/r3_isolation_probe.py` and `docs/grant_revoke_probe.py` both
> pass over docker `ws://`).

## Security

### Grant-revoke propagation (tombstones) — done on embedded, ws:// pending R-6
A `forget`-style soft delete now covers grants: `compartment::revoke` writes a
**tombstone** (`deleted_at` + bumped `updated_at`) instead of hard-deleting, so the
revocation (a) ends access at once where the engine ACL runs — the grant subqueries
in `MEMORY_SELECT_RULE`/`EDGE_LINK_RULE` now exclude `deleted_at` rows — and (b)
**propagates** under LWW (the bumped `updated_at` out-versions a stale live grant,
so a revoked grantee cannot be kept in by another replica's copy). Research
corroborated the urgency: eventually-consistent systems leave revoked credentials
valid during the propagation window (AWS IAM persistence abuse); a *hard delete that
never propagates* is strictly worse. Validated: the embedded grant-ACL test still
fails closed after revoke, and a reconcile test shows the revocation propagating
without resurrection. `compartment::purge_grants` GCs tombstones past a grace
window. Now enforced on `ws://` too (R-6 landed).

### R-6 · Engine permission enforcement on `ws://` (CRITICAL) — DONE
**Discovered + fixed 2026-06-06 while validating grant-revoke over a real `ws://`
server.** The networked server held **one root-authenticated connection** and
signed in per request as each `(tenant, user)` record — but **a root session
bypasses row-level permissions**, and SurrealDB cannot run a permission-scoped
query from a root/system session ([surrealdb#6259](https://github.com/surrealdb/surrealdb/issues/6259)).
So on `ws://` the engine ACL was effectively **not enforced**: a user saw another
user's private-compartment memories. (It always worked on **embedded**, whose
connection is owner/anonymous, so record signin scopes.) A pre-existing hole the
grant-revoke validation surfaced, not a regression.
**Fix (two parts):**
1. **Non-root serving connection.** `HttpState` now holds a separate `serve_store`
   — on an authenticated remote, a second, credential-less connection
   (`Store::connect_without_schema`; the root `store` already applied the schema)
   that only ever holds record sessions; the root `store` is reserved for
   provisioning + the owner-view R-2 watcher. On embedded it is the same single
   connection (which already scopes).
2. **Per-session signin in `on_initialized`.** rmcp builds one server (and, on a
   remote, one cloned DB connection) per session, and a cloned remote connection
   does **not** share the HTTP layer's signin. So each session binds **its own**
   connection to its identity at `on_initialized` — scoping the engine ACL for every
   tool call in that session.
**Validated** over docker `ws://`: cross-tenant isolation, intra-tenant compartment
privacy, grant visibility, and fail-closed revoke all hold (`r3_isolation_probe.py`,
`grant_revoke_probe.py`). Embedded unchanged (31 suites green).
**Follow-up (perf, not correctness):** on `ws://` each session has its own scoped
connection, so the per-request auth lock + the HTTP-layer signin are now redundant
there (harmless overhead) — they could be skipped on the remote path to lift the
serialized-section bottleneck.
