# Security posture

The trust model and the boundaries that enforce it. Summarizes a 2026-06 audit: no remotely-exploitable issues were found; tenant isolation is engine-enforced; the only code-execution path is operator-supplied config.

## Threat model

- **Adversary:** a malicious or buggy tenant/user on the networked MCP surface, or untrusted data ingested into the store (memories, documents, traces).
- **Out of scope:** the operator running the binaries. CLI inputs (`--corpus`, `--url`, `--embedder-url`, file paths) are operator-trusted; protecting the host from its own operator is not a goal.
- **Crown jewels:** cross-tenant confidentiality (one tenant must never read or write another's data) and the integrity of the frozen population.

## Tenant isolation (the core guarantee)

Isolation is **engine-enforced**, defense-in-depth:

- Every tenant-scoped table carries row-level `PERMISSIONS WHERE tenant_id =
  $auth.tenant` (`antumbra-store/src/schema.rs`), bound at signin via SurrealDB record-access. This holds even for an app query with no `WHERE`, verified by `crates/antumbra-store/tests/penumbra_auth.rs` (a record session sees only its tenant).
- Repos add a second app-layer `WHERE tenant_id = …` as backup.
- **Compartment sharing** is engine-gated: only a compartment's owner may write a `grant`; grantees are intra-tenant; revocation (tombstone) takes effect immediately. There is no cross-tenant self-grant.
- The `*_unscoped` / `all_heads` reads return rows across all tenants **only** under an owner/root session (the operator console); under a tenant session the engine ACL still filters them to the caller's tenant (no leak). Their docstrings state this contract.
- **Owner/root sessions bypass the ACL by design** (`store::signin_root`) and are used only on owner connections (the console, the sync watcher), never to service a tenant request.

### Networked MCP

- Per-request **JWT** verification before any DB access; `exp` is **mandatory** (a non-expiring token is a standing key). HS256 (shared secret, self-hosted) or RS256 (against an auth service). Audience is required so a token minted for another verifier is rejected.
- The verified token **is** the identity; there is no token→identity table, so a leaked token grants exactly its claimed `(tenant, user)` scope and nothing more.
- On an embedded store, where every identity shares one connection, each **tool call** signs the connection in as its identity under an auth lock and holds the lock until the tool returns. The signin travels with the call, not with the HTTP request that carried it, because a call over `/mcp` runs after that request is answered. A remote deployment gives each identity its own signed-in connection and holds no lock on the hot path.

## Trust boundaries / operator-supplied inputs

- **Verifier command execution** (`antumbra-critic` `CommandVerifier`): runs the `{program, args, cwd}` from a corpus task's `verify` spec. This spec comes **only** from an operator-supplied `--corpus` JSON file. It is **not stored in the database, not writable by any tenant or MCP request**, and the trace-ingestion path (`antumbra-train` `harness.rs`) uses in-process marker checks (`contains_all`), never a command. So a corpus file is **executable config**, like a Makefile or CI step, so treat it as trusted. It is `Command::new`
  - explicit args (no shell), so there is no shell-string injection. If you ever run corpora from a shared/community source, sandbox the verifier and/or allowlist the program first.
- **Embeddings endpoint** (`antumbra-embed`): the URL and bearer key are operator-configured (`--embedder-url` / `ANTUMBRA_EMBEDDER_URL` / `ANTUMBRA_EMBEDDER_KEY`), never derived from tenant or stored data, so not an SSRF sink. If a future feature lets a request choose the URL, validate it against localhost/private ranges first.
- **The console's routes** (`/console/snapshot`, `/console/action`): the snapshot's workspace part is read on the caller's scoped connection, so the engine's permissions confine it as they do every tool. Its substrate part and every action are read and written as the owner, and only for a token carrying the `operator` claim, checked on each request; the claim rides beside the identity rather than in it, so the per-identity caches never mix an operator's session with a plain one. Only the self-hosted HS256 mint writes the claim (`--mint-token --operator`); the hosted RS256 mint never does, and a token minted before the claim existed reads as not an operator's. Each action is written to the server's stderr with the identity that took it.
- **The PDF renderer** (`--pdf-renderer`, a Gotenberg service): headless Chromium beside the server, which fetches what a relic's page references while it prints. It must be started with `--chromium-deny-private-ips=true`, as the compose profile starts it, so a relic cannot reach the network the server sits on; and before every print the server replaces a reference to a private, loopback or link-local address, literal or resolved, with `about:blank`, so that holds where the renderer was started otherwise.
- **The expert source** (`--expert-source`) and the route it calls (`GET /expert/adapter`): a server with no copy of an expert's adapter pulls it from the one that trained it. The address is operator-configured, never taken from a request or a stored row. It is trusted for availability only: a pulled file is kept and served only when its sha256 and size are the ones on the expert's row, so the store decides what an expert's weights are, and the body is read no further than one byte past that size. Each pull carries a token the pulling server signs with the shared HS256 secret for the workspace and user it is answering, valid for one minute; no credential is stored for it, and holding that secret already means being able to sign any token. The route answers only for a shared expert or the caller's own standing expert, with one 404 for everything else; it sends only a file whose name is the one the expert's id and sha256 determine and whose bytes hash to that sha256, so a row cannot be used to read another file; and it is budgeted per client, since each answer reads a whole adapter into memory.
- **Artifact paths**: the loop's regression fingerprint reads `adapter_uri` from disk; the URI is trainer-/operator-derived and the run-id-derived filename is sanitised. On a missing file it now falls back to a digest **of the uri** (deterministic, distinct per adapter) rather than echoing the raw path.

## Secrets handling

- The DB root credentials, the JWT signing secret, and the server default embedder's bearer key are **operator-configured** (env/CLI) and never persisted to the store. The one secret the store holds is a workspace's own embedder key (`embedder_config.api_key`, set by the operator's CLI for a hosted workspace that brings its own endpoint): its row is readable and writable by the owner view alone, never by a member's record session.
- No secret-bearing struct derives a leaky `Debug`: `HttpEmbedder` has no `Debug` impl; `EmbedderConfig`'s hand-written `Debug` says whether a key is set and never what it is; the JWT `TokenVerifier` holds an opaque `DecodingKey`; `AuthError` is deliberately coarse so wire responses don't leak detail. The signing secret is used transiently (`&[u8]`).
- Error messages carry only non-secret context (e.g. the embeddings URL, never the key, which rides the `Authorization` header).

## SurrealQL injection

None. Every query goes through the surql-rs builders / `crud` helpers; there is **no hand-authored, string-interpolated SurrealQL** anywhere in antumbra's code (the only exception is the schema `PERMISSIONS` predicates, which are static and contain no user input).

## Operational recommendations

- Mint short-TTL, scope-bound JWTs for sensitive tenants; rotate the RS256 key at the auth service and restart to revoke.
- Only run `--corpus` files you trust (they are executable config); sandbox the verifier if a corpus may come from an untrusted source.
- Keep the embeddings endpoint operator-controlled; if it ever becomes request-selectable, add egress validation.
