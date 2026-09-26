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

Start the server with `--tools agent` (or `ANTUMBRA_TOOLS=agent`) for a coding agent: it advertises and serves only `recall_memories`, `store_memory`, `reinforce_memory`, `penalize_memory`, `recall_documents`, `ingest_document`, `route`, and `answer`. The compartment, graph, and operator tools stay behind the CLI and console, and the agent's context carries eight tool descriptions instead of twenty. `--tools agent,population` extends the profile; `all` is the default; an unknown name refuses at startup.

---

## Memories about code carry their anchor (provenance over extraction)

A memory about code is only as good as its anchor. Static extraction keeps a symbol table fresh by re-extracting and pruning; with many concurrent branches that snapshot goes stale silently. Antumbra keeps the anchor **on the memory** and judges it at recall, where git is ([ADR-0018](adr/0018-provenance-over-extraction.md)):

- **Capture** stamps memories about code with `provenance {repo, commit, branch[, path]}` (the capture hook computes it; `store_memory` stores it as one `git:` evidence entry).
- **Recall** takes the caller's `repo` and `branch` and returns every hit with a `scope` (`in_scope`, `other_branch`, `other_repo`), demoting out-of-scope hits below in-scope ones without hiding them: the branch is a governing feature, exactly as a repo-scoped convention is.
- **Bootstrap** asks git whether each anchor's commit is on HEAD and whether its branch still exists, and tags hits `[live]`, `[not-on-head]`, or `[orphaned]` (`ANTUMBRA_PENALIZE_ORPHANS=1` also penalizes the orphans). Nothing is re-extracted; a stale memory is visible instead of silently wrong.

Inventory questions ("what routes does this service expose?") get a parser-free answer the same way: run the framework's own lister and keep what it printed, stamped with the anchor, and let git state what it already knows.

```bash
antumbra ingest --tenant ws:me --user user:me --title routes -- deno task routes   # or an OpenAPI export, cargo metadata, ...
antumbra git-facts --tenant ws:me --user user:me --compartment comp:repo --days 90  # ownership, hotspots, co-change
```

`recall_documents` then names the commit each chunk describes, and re-ingesting a title replaces its chunks in place. Inventory is never answered from expert weights: provenance-backed memory and documents say *what exists*; experts say *how to do it*.

---

## The GitHub App tells Antumbra when an anchor goes stale

The events that make an anchor stale, a merge and a branch deletion, happen on the hosting platform, so the platform reports them directly instead of a session hook noticing later ([ADR-0019](adr/0019-github-integration-and-evidence-graph.md)). The networked server serves `POST /github/webhook` when given the App's webhook secret and a repository-to-workspace map:

```bash
antumbra-mcp --http 0.0.0.0:8081 --jwt-secret-file /run/secrets/jwt \
  --github-webhook-secret-file /run/secrets/github-webhook \
  --github-repos /etc/antumbra/github-repos.json     # {"github.com/acme/orders": "ws:acme", ...}
  # or --github-tenant ws:acme to land every repository the App sees in one workspace
  --github-app-id 123456 --github-app-key-file /run/secrets/github-app.pem   # lets it read contents
  --github-knowledge-diff        # post the knowledge diff on pull requests (needs Checks: write)
  # --github-api-url https://<host>/api/v3 for Enterprise Server
```

| Delivery | What happens |
| --- | --- |
| `pull_request` merged | Every memory whose anchor sits on the merged branch is re-anchored to the merge commit on the base branch (its path kept, the old anchor kept behind it as history), so a squash merge no longer leaves them `not-on-head` forever. The pull request itself becomes a `bank` memory anchored to the merge commit, with the PR URL as evidence, under a deterministic id (a redelivery revises it). |
| `delete` of a branch | Every memory whose anchor still sits on that branch gets a `git-orphaned:<repo>#<branch>@<when>` evidence entry. Recall judges it `orphaned` (demoted, never hidden) and every view carries `orphaned_at`. A memory the merge already moved to the base branch is untouched, so GitHub's delete-after-merge is safe. |
| `pull_request` merged, with App credentials | The knowledge documents the pull request changed (READMEs, docs, ADRs, OpenAPI and AsyncAPI specs; not source) are read at the merge commit through the contents API and ingested by the same path as `ingest_document` and `antumbra ingest`: original to copal first, the title (`<repository>:<path>`, since one workspace holds many repositories and a bare path would make every repository's `README.md` the same document) replaced in place, every chunk anchored `git:<repo>@<merge>#<base>:<path>`. A document whose file the merge removed, or renamed away from, is dropped with it, so recall does not go on calling it live. Repository documents go to the workspace's shared pool. The response says how many were queued; the ingest itself runs after the response, because GitHub allows a receiver ten seconds. |
| `installation` created, `installation_repositories` added | The cold start: every knowledge document in each mapped repository, at the head of its default branch, ingested and anchored the same way. |
| `pull_request` opened, pushed to, reopened or marked ready, with `--github-knowledge-diff` | The **knowledge diff**: a neutral check run on the head commit, named *Antumbra knowledge diff*. It lists what is anchored to the change:<br>- the knowledge document ingested from each file it changes;<br>- the memories anchored to those paths (a renamed file's old path included);<br>- the memories anchored to its branch, which the merge will re-anchor or an unmerged delete will orphan.<br>It never blocks a merge. It says nothing about what a change contradicts, since that takes judging meaning. It runs after the response, and needs the App's **Checks** permission (write); without it, GitHub refuses the post and the log says so. |
| `ping`, anything else, an unmapped repository, a close without a merge | Acknowledged with the reason, so GitHub does not retry. |

Every delivery is verified against the secret (HMAC-SHA256, constant-time) before anything is read; a delivery that does not verify gets a bare 401. The handlers write as the system user `user:github`, provisioned in the workspace on first contact. Reading contents needs the App's id and private key: the receiver signs a short-lived App token, exchanges it for an installation token (cached until it is about to expire), and reads through that. Without them the receiver still re-anchors, orphans, and remembers pull requests.

A server GitHub cannot reach, such as one on a private network, gets merges reported instead. `antumbra claude reanchor` (in a clone, with `gh` signed in and `ANTUMBRA_URL` / `ANTUMBRA_TOKEN` set) reads the repository's merged pull requests and reports them, in one call, to the `record_merges` tool. That tool applies the webhook's merge rule as the caller, so it moves only memories the caller can write. A memory moves if it was created before the merge, or if it is anchored at one of the branch's own commits, as happens when a session is still on the branch after it merges. Running it again moves nothing new; `--dry-run` lists the merges it would report.

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

## Claude Code with its telemetry off

Turning off Claude Code's telemetry (`DISABLE_TELEMETRY`, `DO_NOT_TRACK`, `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC`) also turns off its feature-flag fetching, and so does running it on a third-party provider. A list of features that have nothing to do with telemetry goes with the flags, and nothing announces it: a repository whose only instruction file is `AGENTS.md` silently stops instructing the agent. [ADR-0021](adr/0021-sovereign-mode.md) calls the state sovereign mode and treats it as the normal case.

```sh
antumbra claude doctor            # judge the current project
antumbra claude doctor --dir ../other-repo
```

It says whether the session is in sovereign mode and which variable, in which file, put it there; lists what that costs; checks the settings that bring some of it back; and names a project's `AGENTS.md` when it is not being read. It exits non-zero when a required setting is missing, which on Windows means the PowerShell tool (the host's dominant shell decides). It reads the agent's settings and never writes them: the file grants the agent its permissions, so the doctor prints the line to add and leaves the edit to you.

When you would rather not paste them by hand:

```sh
antumbra claude apply --dry-run     # say what would be written, and write nothing
antumbra claude apply               # write it, after a backup
```

It adds the `env` names the doctor asks for to your own `~/.claude/settings.json`, and nothing else: only names it already knows, only ones the file does not set, and never anything under `permissions` — not even `permissions.defaultMode`, which the doctor asks for and this reports and leaves to you. Your key order and formatting survive, because the edit is textual rather than a reserialization. It backs the file up first, then reads it back and restores the backup unless the result is exactly what was there plus those names.

When it names an `AGENTS.md` that is not being read, this fixes it:

```sh
antumbra claude bridge            # --dry-run to see what it would do, --remove to take it back out
```

Beside each `AGENTS.md` the agent would have read, it writes a `CLAUDE.local.md` that imports it, and lists that file in the clone's own `.git/info/exclude`. The agent then reads the file natively again: at launch, in subdirectories, after compaction, and in subagents, with no size limit short of its own. Nothing the repository tracks changes, so it is safe in a checkout you do not own. It leaves alone any directory that already has instructions of its own, because the agent was never going to read `AGENTS.md` there, and `--remove` deletes only a file that is still a bridge and nothing else. It is not done through a hook on purpose: a hook's context is capped at 10,000 characters and cut to a preview past that, which a real instruction file exceeds.

Neither command needs a running server or a store, so both work when nothing else does, including when a bad MCP tool schema is failing every request inside the agent. The rules are verified against a named Claude Code release, and the report says so when the installed one differs.

That last case has its own command. With the flags off the agent still checks every MCP tool's input schema, logs the answer, and sends the schema anyway; the API then refuses the whole request with a 400 that names the tool only by its position.

```sh
antumbra claude mcp-lint --server github -- npx -y @some/mcp-server   # ask a stdio server
antumbra claude mcp-lint --server github --from tools-list.json       # or a saved tools/list answer
```

It runs the agent's two checks, prints the deny rule that keeps each offender out of the request (and leaves the edit to you), and also names tools that go missing without a word: one with `anyOf`, `oneOf` or `allOf` at its schema's root is skipped in this state, and one whose root `type` is not `"object"` costs its server every tool, in any state. With tool search on, which is the default, a bad tool breaks nothing until the agent first loads it, so a session that has always worked is not evidence of a clean server. It exits non-zero on a failure, so a server's maintainer can run it in CI.

The agent's classifier decides what counts as leaving your boundary, and `/auto-mode-setup`, which drafts the entries that tell it, is one of the things that goes with the flags:

```sh
antumbra claude auto-mode-env --repos ~/repos     # prints a draft; writes nothing
```

It drafts `Source control` from the remotes of your working trees, proposing an owner only when you push there over ssh and it is plainly yours, and listing every other owner with the reason it was left out. With a surface to ask (`ANTUMBRA_URL`, `ANTUMBRA_TOKEN`) it also offers memories as candidates for the slots only prose can fill, such as which host is production and which is a test node. It reads no transcript. The block goes in your own `~/.claude/settings.json`; the classifier never reads `autoMode` from a project's settings.

`/skill-doctor`, which finds the skills nobody uses, goes as well. `antumbra claude skills` reports the same from counters that two hooks keep ([hooks](../scripts/hooks/README.md)): one for a skill the agent calls and one for a skill you type, since neither hook sees the other's.

The operator console shows both on its Sovereign page (`5`, or `antumbra-tui --page sovereign`): the rules, by how many workspaces hold each current or retired, and skill use across workspaces, stalest first. It is read-only and queries nothing new.

Two more say what the doctor knows to the agent. The session-start hook opens with `antumbra claude brief` ([hooks](../scripts/hooks/README.md)), and `antumbra claude remember` keeps the same rules as `world` memories in a `claude-code` compartment of your own: volatile, so they never train an expert, and safe to run again.

## Who can recall a document

A document is kept the way a memory is: in a compartment, or in the workspace's shared pool. `ingest_document` takes a `compartment` (and `antumbra ingest` a `--compartment`); then only the compartment's owner and the people it is shared with can recall the document, enforced by the engine on every read, and a recalled chunk says which compartment it came from. Leave it out and the document goes to the shared pool, which every member of the workspace can recall. That is the right place for reference material (it is where the GitHub integration puts a repository's documents) and the wrong place for anything private.

A compartment has to be one the caller can write to: their own, or one shared with them with `link`. The check happens before anything is archived or stored, so a refusal leaves nothing behind. A document's identity is its workspace, its compartment and its title, so two members can each keep a private document under one title: neither ingest touches the other's chunks, and each has its own archived original.

Before this, every document in a workspace was recallable by every member. Documents ingested then have no compartment, so they are in the shared pool and exactly as recallable as they were; to make one private, ingest it again into a compartment. An existing database picks the rule up on its next start: the server re-asserts each table's permissions when it connects, since a definition applied `IF NOT EXISTS` would otherwise never change on a table that already exists.

## Copal as the document of record (knowledge documents)

`ingest_document` chunks, embeds, and stores a knowledge document for recall — and in v0 that is *all* it keeps: the chunks. Recall works, but the original bytes are gone. Point the MCP server at a **Copal** file service (content-addressed, versioned, sealed-at-rest file storage) and Copal becomes the **document of record**: on every ingest the original content is uploaded there first, and each stored chunk carries provenance back to it (`copal_file`, the archived file's id, and `copal_digest`, the content digest of exactly the bytes that were ingested) — so a `recall_documents` answer names not just what it remembers but the original it came from.

The address, on the server's usual clap/env conventions:

```
ANTUMBRA_COPAL_ADDR=127.0.0.1:9010    # --copal-addr: bare host:port (http:// assumed) or a full URL base
```

Tenancy is the deployment's second choice: *which Copal tenant does a workspace's document land in, and how does the call prove it?* Copal's `header` auth mode (its default until 1.0) trusts the `x-copal-tenant` header as the identity; its deployed `keys` mode binds the tenant to a `ck1` credential. One knob per shape, at most one of the three (two together refuse at startup):

|                           | Copal `header` auth (dev default)                             | Copal `keys` auth (deployed)                                        |
| ------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------- |
| **Per-workspace tenancy** | *(the default — nothing to set)* each workspace IS its tenant | `--copal-keys <file>`: JSON mapping workspace → `ck1` key           |
| **Shared tenant**         | `--copal-tenant <name>`                                       | `--copal-key <ck1 key>` (the key's tenant is *the* tenant)          |

(`ANTUMBRA_COPAL_TENANT` / `ANTUMBRA_COPAL_KEY` / `ANTUMBRA_COPAL_KEYS` as envs.) Per-workspace tenancy means each workspace gets its own quotas, listings, and search scope, with cross-tenant reads refusing at Copal's own boundary — and the `--copal-keys` column is how that survives a Copal deployment upgrading to `keys` auth: mint a key per workspace on Copal's admin surface, map them in the file, restart to pick up new ones.

The contract:

- **Upload first, fail closed.** The original lands in Copal *before* any chunk is stored, and an unreachable Copal fails the ingest with a clear error. A configured document of record that silently dropped originals would be worse than none. Under `--copal-keys`, a workspace with no mapped key fails the same way — refusing beats archiving into a tenant that is not the workspace's own.
- **Re-ingest revisions, never litters.** The create carries an idempotency key derived from (workspace, title), where the title of a document kept in a compartment carries that compartment, so two members' private documents of one title are two archived files and "the original" of one is never the other's text — the *antumbra* workspace tenant, in every tenancy shape — so ingesting the same title again revisions the *same* Copal file, two workspaces sharing a title never revision each other's document (even inside a shared tenant), and moving a deployment between shapes never re-identifies a document. The version history is the document's history; the archived file's metadata names its owning workspace.
- **Absent means exactly today's behavior.** No `--copal-addr`, no archive: ingest keeps only the chunks, nothing new is required, and chunks written either way coexist (the provenance fields are simply absent on archive-less chunks).
- **Every door archives the same way.** The CLI's `antumbra ingest` takes the same four flags (`--copal-addr`, `--copal-tenant`, `--copal-key`, `--copal-keys`, or the `ANTUMBRA_COPAL_*` envs) and follows the same upload-first, fail-closed contract through the shared `antumbra-copal` client, so a CI step that ingests a framework's lister output or a generated service doc lands a document of record too.

---

## Why this beats a plain memory layer

Retrieval-memory tools (give the agent a vector store to recall from) make the agent _remember_. Antumbra makes it **learn**:

1. **Bootstrap**: the agent starts the session already carrying your conventions and history (memory + identity).
2. **Route or answer**: the `answer` tool sends a task to the frozen expert most likely to cover it, or escalates when it is out of scope. A served task costs you nothing; only genuine novelty hits the expensive model.
3. **Capture**: verified outcomes and observations are written back.
4. **Metabolize**: `antumbra metabolize` turns the successful, recurrent traces (and their step-by-step decomposition) into a trained LoRA expert, frozen into the population so it is never forgotten.

Next session, step 1 includes a skill that did not exist before, and the work it covers is now served locally for free. The scaffolding (loops, prompts, lookups) shrinks into weights. A memory layer is static; Antumbra compounds.

See **[Architecture](architecture.md)** for the engine, **[Roadmap](roadmap.md)** for what is built vs queued, and **[Product surface](product.md)** for the control plane (dashboard, knowledge documents, onboarding) and how Antumbra supersedes a separate agent-memory engine.
