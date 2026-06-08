# Security posture

The trust model and the boundaries that enforce it. Summarises a 2026-06 audit:
no remotely-exploitable issues were found; tenant isolation is engine-enforced;
the only code-execution path is operator-supplied config.

## Threat model

- **Adversary:** a malicious or buggy tenant/user on the networked MCP surface,
  or untrusted data ingested into the store (memories, documents, traces).
- **Out of scope:** the operator running the binaries. CLI inputs (`--corpus`,
  `--url`, `--embed-url`, file paths) are operator-trusted; protecting the host
  from its own operator is not a goal.
- **Crown jewels:** cross-tenant confidentiality (one tenant must never read or
  write another's data) and the integrity of the frozen population.

## Tenant isolation (the core guarantee)

Isolation is **engine-enforced**, defence-in-depth:

- Every tenant-scoped table carries row-level `PERMISSIONS WHERE tenant_id =
  $auth.tenant` (`antumbra-store/src/schema.rs`), bound at signin via SurrealDB
  record-access. This holds even for an app query with no `WHERE`, verified by
  `crates/antumbra-store/tests/penumbra_auth.rs` (a record session sees only its
  tenant).
- Repos add a second app-layer `WHERE tenant_id = …` as backup.
- **Compartment sharing** (ADR-0014) is engine-gated: only a compartment's owner
  may write a `grant`; grantees are intra-tenant; revocation (tombstone) takes
  effect immediately. There is no cross-tenant self-grant.
- The `*_unscoped` / `all_heads` reads return rows across all tenants **only**
  under an owner/root session (the operator console); under a tenant session the
  engine ACL still filters them to the caller's tenant (no leak). Their
  docstrings state this contract.
- **Owner/root sessions bypass the ACL by design** (`store::signin_root`) and are
  used only on owner connections (the console, the sync watcher), never to
  service a tenant request.

### Networked MCP (ADR-0015)

- Per-request **JWT** verification before any DB access; `exp` is **mandatory**
  (a non-expiring token is a standing key). HS256 (shared secret, self-hosted)
  or RS256 (against an auth service). Audience is required so a token minted for
  another verifier is rejected.
- The verified token **is** the identity; there is no token→identity table, so a
  leaked token grants exactly its claimed `(tenant, user)` scope and nothing
  more.
- Each request signs the shared connection in as its identity, serialised by an
  auth lock: no session reuse or confused-deputy across tenants.

## Trust boundaries / operator-supplied inputs

- **Verifier command execution** (`antumbra-critic` `CommandVerifier`): runs the
  `{program, args, cwd}` from a corpus task's `verify` spec. This spec comes
  **only** from an operator-supplied `--corpus` JSON file. It is **not stored in
  the database, not writable by any tenant or MCP request**, and the
  trace-ingestion path (`antumbra-train` `harness.rs`) uses in-process marker
  checks (`contains_all`), never a command. So a corpus file is **executable
  config**, like a Makefile or CI step, so treat it as trusted. It is `Command::new`
  + explicit args (no shell), so there is no shell-string injection. If you ever
  run corpora from a shared/community source, sandbox the verifier and/or
  allowlist the program first.
- **Embeddings endpoint** (`antumbra-embed`): the URL and bearer key are
  operator-configured (`--embed-url` / `ANTUMBRA_EMBED_URL` / `ANTUMBRA_EMBED_KEY`),
  never derived from tenant or stored data, so not an SSRF sink. If a future
  feature lets a request choose the URL, validate it against localhost/private
  ranges first.
- **Artifact paths**: the loop's regression fingerprint reads `adapter_uri` from
  disk; the URI is trainer-/operator-derived and the run-id-derived filename is
  sanitised. On a missing file it now falls back to a digest **of the uri**
  (deterministic, distinct per adapter) rather than echoing the raw path.

## Secrets handling

- The DB root credentials, the JWT signing secret, and the embeddings bearer key
  are **operator-configured** (env/CLI) and never persisted to the store.
- No secret-bearing struct derives a leaky `Debug`: `HttpEmbedder` has no `Debug`
  impl; the JWT `TokenVerifier` holds an opaque `DecodingKey`; `AuthError` is
  deliberately coarse so wire responses don't leak detail. The signing secret is
  used transiently (`&[u8]`).
- Error messages carry only non-secret context (e.g. the embeddings URL, never
  the key, which rides the `Authorization` header).

## SurrealQL injection

None. Every query goes through the surql-rs builders / `crud` helpers; there is
**no hand-authored, string-interpolated SurrealQL** anywhere in antumbra's code
(the only exception is the schema `PERMISSIONS` predicates, which are static and
contain no user input).

## Operational recommendations

- Mint short-TTL, scope-bound JWTs for sensitive tenants; rotate the RS256 key at
  the auth service and restart to revoke.
- Only run `--corpus` files you trust (they are executable config); sandbox the
  verifier if a corpus may come from an untrusted source.
- Keep the embeddings endpoint operator-controlled; if it ever becomes
  request-selectable, add egress validation.
