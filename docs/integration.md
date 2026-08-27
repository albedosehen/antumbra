# Using Antumbra: wire it into your coding agent

This is the practical answer to _"how do I actually use this, and what do I get?"_

You don't chat with Antumbra directly. It is the **private, persistent brain** your existing coding agent (Claude Code, Cursor, any MCP client) plugs into: a memory layer that compounds. It is useful from the first session, because your agent boots already knowing your conventions and history, and it **gets better at your work over time** by turning verified outcomes into permanent local skills. You keep your agent; Antumbra gives it private memory, identity, multi-tenant boundaries, and a growing population of specialists, all on your hardware.

The whole integration is three touchpoints on your agent's lifecycle, plus an MCP connection. Once wired, every session **boots smarter and ends by depositing what it learned**; the successful work is metabolized into weights so the scaffold shrinks.

```
                    ┌──────────────────────────── Antumbra ───────────────────────────┐
session starts ───▶ │ bootstrap: identity + conventions + relevant memory             │
                    │   (memory store · compartments · engine-enforced ACL)           │
 agent does work ◀─▶│ tools: recall / store / route / answer / graph / compartment    │
                    │   (the answer tool routes to a frozen expert, or escalates)      │
session stops  ───▶ │ capture: write observations back  ──▶  metabolize ──▶ experts   │
                    └──────────────────────────────────────────────────────────────────┘
```

---

## The three touchpoints

Most agent runtimes (Claude Code's `settings.json` hooks are the reference shape) let you run a command at lifecycle events. Antumbra uses three:

### 1. Bootstrap on session start (the agent boots knowing your world)

A **SessionStart** hook calls Antumbra and injects the result as the session's opening context: the agent's standing conventions, device config, and the memory relevant to this project. No cold start: the agent already knows "this repo uses `deno`, not `npm`," who you are, and what it learned last time.

```jsonc
// settings.json (Claude Code shape; adapt the event names to your runtime)
// Windows: pwsh … .ps1   ·   macOS/Linux: bash … .sh   (both are provided)
"hooks": {
  "SessionStart": [{ "hooks": [{
    "type": "command",
    "command": "pwsh -NonInteractive -File ./scripts/hooks/antumbra-session-start.ps1",
    "timeout": 10
  }]}]
}
```

The script fetches the bootstrap memory and returns it as `additionalContext`. (See the [`scripts/hooks/`](../scripts/hooks/) templates.) _Today:_ the capture + attribution hooks work as-is (they emit hook decisions, no Antumbra call), the long-lived **hook token** is mintable with `antumbra-mcp --mint-token` (P-1a), and the bootstrap fetch's transport (`POST /mcp/call {tool, arguments}`) is now live (P-1b), so the SessionStart script works end-to-end. (Alternatively, have the agent run `recall_memories` at the top of its first turn with no SessionStart script at all.)

### 2. Capture on stop / before compaction (nothing learned is lost)

A **Stop** hook (and a **PreCompact** hook, for when the context window is about to be summarized) nudges the agent to deposit non-obvious observations back into Antumbra before the turn ends. A sentinel file makes it fire once, not in a loop. This is the write half of memory, and the raw successful traces it leaves are what `antumbra metabolize` later turns into a trained expert.

```jsonc
"Stop":       [{ "hooks": [{ "type": "command", "command": "pwsh -NonInteractive -File ./scripts/hooks/antumbra-capture.ps1", "timeout": 5 }]}],
"PreCompact": [{ "hooks": [{ "type": "command", "command": "pwsh -NonInteractive -File ./scripts/hooks/antumbra-capture.ps1", "timeout": 5 }]}]
```

### 3. Behavior overrides (make Antumbra the single source of truth)

Two settings make the agent defer to Antumbra instead of its built-ins:

- **Disable the agent's built-in file memory** so _all_ memory flows through Antumbra (one store, one ACL, one thing to back up). In Claude Code that is `"autoMemoryEnabled": false` plus a `permissions.deny` on the local memory path.
- **Strip vendor attribution** from commits/PRs with a `PreToolUse` deny hook on `git`/`gh`, so the work is attributed to you, not the model vendor.

```jsonc
"autoMemoryEnabled": false,
"permissions": {
  "deny": ["Write(**/.agent/memory/**)", "Edit(**/.agent/memory/**)"]
},
"hooks": { "PreToolUse": [{ "matcher": "Bash", "hooks": [
  { "type": "command", "command": "pwsh -File ./scripts/hooks/strip-attribution.ps1", "if": "Bash(git *)" },
  { "type": "command", "command": "pwsh -File ./scripts/hooks/strip-attribution.ps1", "if": "Bash(gh *)" }
]}]}
```

Template scripts for all of the above live under [`scripts/hooks/`](../scripts/hooks/) for **both platforms**: PowerShell (`.ps1`, Windows) and POSIX `bash` (`.sh`, macOS/Linux; needs `jq` + `curl`). Use the pair for your OS. They are thin: read stdin JSON, call Antumbra's `/mcp/call` (or the stdio server), emit the hook's JSON response. Point them at your endpoint with four env vars:

```
ANTUMBRA_URL=http://127.0.0.1:8081     # the MCP engine (omit for stdio/offline)
ANTUMBRA_WORKSPACE_ID=<your-workspace> # the tenant/workspace scope
ANTUMBRA_API_KEY=<key>                 # for the hosted/networked surface
ANTUMBRA_HOST_ID=<this-device>         # provenance stamp on what it writes
```

---

## Two ways to run it

Antumbra is the **same engine** in both modes; only the transport and identity differ.

|               | **Offline / private**                         | **Hosted (still private to you)**                              |
| ------------- | --------------------------------------------- | -------------------------------------------------------------- |
| Transport     | stdio MCP, single local identity              | networked HTTP/SSE, JWT per request                            |
| Store         | embedded `surrealkv://` on your disk          | authoritative SurrealDB, multi-tenant                          |
| Who sees data | only this machine                             | only your tenant (engine-enforced ACL)                         |
| Embedder      | yours, local                                  | yours, per workspace (bring-your-own)                          |
| Best for      | a solo dev, an air-gapped box, regulated data | a team/fleet sharing one brain; org infra you'd rather not run |

Offline is the default and the privacy floor: nothing leaves the building. The hosted surface adds multi-tenant sharing, device sync, and live propagation. The ACL is enforced **in the database engine**, so a tenant can never see another tenant's rows even if a handler forgets a filter.

---

## Copal as the document of record (knowledge documents)

`ingest_document` chunks, embeds, and stores a knowledge document for recall — and in v0 that is *all* it keeps: the chunks. Recall works, but the original bytes are gone. Point the MCP server at a **Copal** file service (content-addressed, versioned, sealed-at-rest file storage) and Copal becomes the **document of record**: on every ingest the original content is uploaded there first, and each stored chunk carries provenance back to it (`copal_file`, the archived file's id, and `copal_digest`, the content digest of exactly the bytes that were ingested) — so a `recall_documents` answer names not just what it remembers but the original it came from.

Two knobs, on the server's usual clap/env conventions:

```
ANTUMBRA_COPAL_ADDR=127.0.0.1:9010    # --copal-addr: bare host:port (http:// assumed) or a full URL base
ANTUMBRA_COPAL_TENANT=acme            # --copal-tenant: optional; sets SHARED tenancy (see below)
```

The contract:

- **Upload first, fail closed.** The original lands in Copal *before* any chunk is stored, and an unreachable Copal fails the ingest with a clear error. A configured document of record that silently dropped originals would be worse than none.
- **Every workspace is its own Copal tenant** (the default, `--copal-tenant` unset): each workspace presents itself as the `x-copal-tenant`, so quotas, listings, and search scope per workspace and Copal's own tenant boundary isolates them. This rides Copal's *header* auth mode. Set `--copal-tenant` to land every workspace under that one **shared** Copal tenant instead — the shape Copal's deployed *keys* auth mode forces, where a single credential is bound to a single tenant.
- **Re-ingest revisions, never litters.** The create carries an idempotency key derived from (workspace, title) — the *antumbra* workspace tenant, in both tenancy modes — so ingesting the same title again revisions the *same* Copal file, two workspaces sharing a title never revision each other's document (even inside a shared tenant), and moving a deployment between the modes never re-identifies a document. The version history is the document's history; the archived file's metadata names its owning workspace.
- **Absent means exactly today's behavior.** No `--copal-addr`, no archive: ingest keeps only the chunks, nothing new is required, and chunks written either way coexist (the provenance fields are simply absent on archive-less chunks).

---

## Why this beats a plain memory layer

Retrieval-memory tools (give the agent a vector store to recall from) make the agent _remember_. Antumbra makes it **learn**:

1. **Bootstrap**: the agent starts the session already carrying your conventions and history (memory + identity).
2. **Route or answer**: the `answer` tool sends a task to the frozen expert most likely to cover it, or escalates when it is out of scope. A served task costs you nothing; only genuine novelty hits the expensive model.
3. **Capture**: verified outcomes and observations are written back.
4. **Metabolize**: `antumbra metabolize` turns the successful, recurrent traces (and their step-by-step decomposition) into a trained LoRA expert, frozen into the population so it is never forgotten.

Next session, step 1 includes a skill that did not exist before, and the work it covers is now served locally for free. The scaffolding (loops, prompts, lookups) shrinks into weights. A memory layer is static; Antumbra compounds.

See **[Architecture](architecture.md)** for the engine, **[Roadmap](roadmap.md)** for what is built vs queued, and **[Product surface](product.md)** for the control plane (dashboard, knowledge documents, onboarding) and how Antumbra supersedes a separate agent-memory engine.
