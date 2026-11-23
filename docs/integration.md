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

The script fetches the bootstrap memory and returns it as `additionalContext`. (See the [`scripts/hooks/`](../scripts/hooks/) templates.) _Today:_ the capture + attribution hooks work as-is (they emit hook decisions; the capture hook also reports the end of each turn, see [How the system keeps its memories](#how-the-system-keeps-its-memories)), the long-lived **hook token** is mintable with `antumbra-mcp --mint-token`, and the bootstrap fetch's transport (`POST /mcp/call {tool, arguments}`) is now live, so the SessionStart script works end-to-end. (Alternatively, have the agent run `recall_memories` at the top of its first turn with no SessionStart script at all.)

### 2. Capture on stop / before compaction (nothing learned is lost)

A **Stop** hook reminds the agent to deposit non-obvious observations back into Antumbra before the turn ends. The reminder reaches the agent as feedback that keeps the turn going, not as a block, so nothing is reported to the user as a hook problem; a sentinel file makes it fire once, not in a loop. The same hook on **PreCompact**, for when the context window is about to be summarized, blocks nothing: it tells the summary to keep what has not been stored yet, and the end of the turn after it is reminded. This is the write half of memory, and the raw successful traces it leaves are what `antumbra metabolize` later turns into a trained expert.

```jsonc
"Stop":       [{ "hooks": [{ "type": "command", "command": "pwsh -NonInteractive -File ./scripts/hooks/antumbra-capture.ps1", "timeout": 10 }]}],
"PreCompact": [{ "hooks": [{ "type": "command", "command": "pwsh -NonInteractive -File ./scripts/hooks/antumbra-capture.ps1", "timeout": 10 }]}]
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
"hooks": { "PreToolUse": [
  { "matcher": "Bash", "hooks": [
    { "type": "command", "command": "bash ./scripts/hooks/strip-attribution.sh", "if": "Bash(git *)", "timeout": 15 },
    { "type": "command", "command": "bash ./scripts/hooks/strip-attribution.sh", "if": "Bash(gh *)", "timeout": 15 }
  ]},
  // Windows with the PowerShell tool: its commands never reach a Bash matcher
  { "matcher": "PowerShell", "hooks": [
    { "type": "command", "command": "bash ./scripts/hooks/strip-attribution.sh", "if": "PowerShell(git *)", "timeout": 15 },
    { "type": "command", "command": "bash ./scripts/hooks/strip-attribution.sh", "if": "PowerShell(gh *)", "timeout": 15 }
  ]}
]}
```

The attribution check is the `.sh` on every platform: Claude Code runs hook commands through Git Bash on Windows, and the script uses bash builtins only (no `jq`), so a check is one process of about 45 ms. A PowerShell start per `git`/`gh` command took seconds on a busy machine and timed out, and a timed-out `PreToolUse` hook blocks nothing.

Template scripts for all of the above live under [`scripts/hooks/`](../scripts/hooks/) for **both platforms**: PowerShell (`.ps1`, Windows) and POSIX `bash` (`.sh`, macOS/Linux; needs `jq` + `curl`). Use the pair for your OS. They are thin: read stdin JSON, call Antumbra's `/mcp/call` (or the stdio server), emit the hook's JSON response. Point them at your endpoint with four env vars:

```
ANTUMBRA_URL=http://127.0.0.1:8081     # the MCP engine (omit for stdio/offline)
ANTUMBRA_WORKSPACE_ID=<your-workspace> # the tenant/workspace scope
ANTUMBRA_API_KEY=<key>                 # for the hosted/networked surface
ANTUMBRA_HOST_ID=<this-device>         # provenance stamp on what it writes
```

The machine's name reaches the server as an `X-Antumbra-Host` header on every MCP call, which the `antumbra-mcp-headers` helper adds from `ANTUMBRA_HOST_ID`. The server stamps each write with it per request, so two machines sharing one token still stamp their own names. A client that sends no name gets the server's own name, which on a hosted server is the hub's. Reads return it as `author_host`, and `recall_memories` and `list_memories` filter by it with `host`.

A coding agent's connection asks for the `agent` profile by itself: the headers helper setup writes for Claude Code, and the bridge the other agents use, send `X-Antumbra-Tools: agent`, and the server advertises and serves that connection only what a session needs, 21 of its 42 tools. The dashboard and the CLI send no such header and keep every tool, on the same server with the same token. `ANTUMBRA_AGENT_TOOLS=all` in the agent's environment asks for everything again, and `ANTUMBRA_AGENT_TOOLS=agent,population` for the profile and more; a connection can ask for less than its server serves and never for more. Starting the server itself with `--tools agent` (or `ANTUMBRA_TOOLS=agent`) cuts it down for every caller, the dashboard and the CLI among them, so it is for a server only agents talk to. Those are the memory loop with its feedback (`recall_memories`, `store_memory`, `reinforce_memory`, `penalize_memory`), documents in and out (`recall_documents`, `ingest_document`), `route` and `answer`, handoffs and devices (`leave_handoff`, `handoffs`, `complete_handoff`, `register_device`, `devices`), behaviors (`record_behavior`, `list_behaviors`, `accept_behavior`, `retire_behavior`, `behavior_candidates`), and relics (`publish_relic`, `list_relics`, `get_relic`). The compartment, graph, and operator tools stay behind the CLI and console, and the agent's context carries 21 tool descriptions instead of 42. `--tools agent,population` extends the profile; `all` is the default; an unknown name refuses at startup.

---

## Memories about code carry their anchor (provenance over extraction)

A memory about code is only as good as its anchor. Static extraction keeps a symbol table fresh by re-extracting and pruning; with many concurrent branches that snapshot goes stale silently. Antumbra keeps the anchor **on the memory** and judges it at recall, where git is:

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

The events that make an anchor stale, a merge and a branch deletion, happen on the hosting platform, so the platform reports them directly instead of a session hook noticing later. The networked server serves `POST /github/webhook` when given the App's webhook secret and a repository-to-workspace map:

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
| The cold start, or a merge into the default branch that changes a manifest, with App credentials | The repository's manifests (package.json, Cargo.toml, go.mod, pyproject.toml, requirements files) are read at that commit and kept as its latest reading, and the **declared dependency edges** into and out of it are brought in line with every repository the App has read: recorded or reinforced while a manifest declares them, anchored to the declaring file; retracted once none does; back if one does again. A merge into any other branch reads nothing. |
| `pull_request` opened, pushed to, reopened or marked ready, with `--github-knowledge-diff` | The **knowledge diff**: a neutral check run on the head commit, named *Antumbra knowledge diff*. It lists what is anchored to the change:<br>- the knowledge document ingested from each file it changes;<br>- the memories anchored to those paths (a renamed file's old path included);<br>- the memories anchored to its branch, which the merge will re-anchor or an unmerged delete will orphan.<br>It never blocks a merge. It says nothing about what a change contradicts, since that takes judging meaning. It runs after the response, and needs the App's **Checks** permission (write); without it, GitHub refuses the post and the log says so. |
| `ping`, anything else, an unmapped repository, a close without a merge | Acknowledged with the reason, so GitHub does not retry. |

Every delivery is verified against the secret (HMAC-SHA256, constant-time) before anything is read; a delivery that does not verify gets a bare 401. The handlers write as the system user `user:github`, provisioned in the workspace on first contact. Reading contents needs the App's id and private key: the receiver signs a short-lived App token, exchanges it for an installation token (cached until it is about to expire), and reads through that. Without them the receiver still re-anchors, orphans, and remembers pull requests.

A server GitHub cannot reach, such as one on a private network, gets merges reported instead. `antumbra claude reanchor` (in a clone, with `gh` signed in) reads the repository's merged pull requests and reports them, in one call, to the `record_merges` tool. Like every `antumbra claude` command that calls the server, it takes the address and the token from `--surface` and `--token`, else from `ANTUMBRA_URL` and `ANTUMBRA_TOKEN`, else from what `antumbra setup` recorded and the token file it keeps; the file's token is presented only to the server it was kept for. That tool applies the webhook's merge rule as the caller, so it moves only memories the caller can write. A memory moves if it was created before the merge, or if it is anchored at one of the branch's own commits, as happens when a session is still on the branch after it merges. Running it again moves nothing new; `--dry-run` lists the merges it would report.

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

### The dashboard

The HTTP server also serves a dashboard at `/dashboard` (for example `http://localhost:8081/dashboard`). Sign in with the same kind of bearer token a hook uses (`antumbra-mcp --mint-token`, or the token your sign-in link gave you). It is laid out as pages down a sidebar, with a search of memory in the bar from any of them (`/` goes to it). Overview says what is waiting for you (proposed behaviors, handoffs, a token about to expire) beside the latest memories and your devices. Under You, Account shows who you are signed in as, the name you go by and your image (a PNG, JPEG or WebP, cut square and made 256 pixels across in the browser before it is sent; the workspace's other members see both, and only you change them), your role, when the token was issued and expires, and your preferences (theme, density, opening page), which are kept in that browser and nowhere else. The theme follows the system until you hold it to Light or Dark, on the Account page or in the user menu, and a held theme is on the page before it is first drawn, on the relics pages too. The sidebar collapses to its icons from the button in the bar or with `[`, and stays as you left it, on the relics gallery and a relic's viewer as on the dashboard's own pages; Devices lists the machines your agents run on with when each was last seen, and the handoffs one left for another. Members lists who is in the workspace, each with their name, image and role, to every member. A workspace's first member is its owner; an owner changes who its owners and admins are, an admin or an owner removes a member (whose token is refused from then, and who can be let back in), and a workspace always keeps an owner. A role is the server's record, not a claim in a token, so it changes without a new token; it reaches no further than its workspace, and is not the `operator` a token may carry, which is whoever runs the server. An admin or an owner also lets people in from that page: they make a join code (for how many people, for how many days, and whether whoever holds it is let in without being asked), which is shown once and kept only as its digest; a person asks with it, choosing the handle and name they will go by and writing a line if they like; the request shows on the page and among what is waiting on Overview, to be approved or refused with a word; and an approved person collects their token, once, with the secret they asked with. Policy shows the workspace's settings; with an operator's token it is a form over them, each with its range, saved together, and with any other token it is read only. Across the workspace's pages it shows the workspace counts, the expert population with each expert's status and fitness (and a box to route a task through it, which also says how close the task is to what each of your standing experts was taught and lists the nearest of your behaviors, or to have the covering expert answer it, which writes nothing), your compartments, your knowledge documents, your behaviors by status (each with its scope, how many examples and violations back it, the memories it was drawn from, how its last training went, and what keeps it from being taught), the dependency graph (drawn, a service before what it depends on, with a line's stroke for the source of the claim; under it every pair of services with its evidence, and any service's blast radius either way within one hop, three or six, with what the walk reached marked in the drawing), and memory: recalled by meaning or listed newest first, filtered by network, each with its git anchor and its relations to other memories. Code in a memory, between ``` fences, is drawn as a block in a fixed-width face, coloured by what each part is, without the fence marks. It also does what is yours to do, each through the tool an agent would call: Accept and Retire on a behavior; write a memory, and Reinforce, Penalize or Forget one (Forget asks twice); add a document; make a compartment; and, on Devices, leave a note for the agent on one of your machines, which it is told when it next starts a session, and mark one done. Share, under You, is where you choose which of your own memories go to a workspace you joined; nothing goes by itself. It is built for a store of thousands: narrow by kind, repository, compartment, the machine a memory was written on, the month, the words it holds and whether it has been used, each choice showing how many memories it has; take everything that matches in one action, or decide a memory at a time (Choose, Leave out, Never); and look over what you chose, with Undo. The hooks' own memories and text that looks like it holds a key, a token or a password are held back until you ask to see them. What you decide is kept on your own server, and choosing sends nothing: `antumbra bring`, on a machine that joined the workspace, copies what is chosen (`--dry-run` says how many; `--to` names the workspace when the machine joined several). Each memory is read whole from your server and stored again through the workspace's as you there, with its kind, confidence and git anchor, and is marked brought on your server as it arrives; the originals stay. It lands in a compartment of yours in the workspace named for the machine it came from. A compartment is its owner's until it is shared: `--offer` also offers that compartment to the whole workspace, and so does the Knowledge page of the workspace's dashboard; `share_compartment` shares it with one member. Agent, under You, shows which skills your coding agent's hooks have counted, the longest unused first, and the rules it is given. An operator's token also saves the workspace's policy from it. It calls the same tools through `/mcp/call`, so it sees exactly what the token's identity may see. The token stays in that browser tab and is gone when the tab closes or you sign out. Once you have shared five memories, its Suggested view lists the undecided ones that lie nearer what you shared than what you left out, the nearest first; it suggests and decides nothing, and what is held back is never suggested.

### Relics

A relic is a page an agent wrote, kept so you can find it and open it again: a report, a dashboard, an essay. The agent calls `publish_relic` with a title and a complete HTML document (images, styles and scripts inline). The page is stored in copal, one file per version, and its text is indexed so `recall_documents` and the gallery's search find it by what it says. Publishing again under the same slug adds a version at the same address.

The gallery is at `/relics` (for example `http://localhost:8081/relics`): your relics and those in compartments shared with you, as cards with a live preview of each page, filtered by All, Yours or Shared with you, searchable, in a grid or a list. Each relic opens at `/relic/<slug>`. A relic is private by default, seen by you and whoever its compartment is shared with, and viewing it mints a render link for that view, so nothing you type expires. Make it public from the viewer (or with `set_relic_visibility`) and anyone who can reach the server opens `/relic/<slug>` without signing in.

A relic is never displayed by the dashboard's own origin, where your token lives. It runs in an `<iframe>` from copal's render origin, sandboxed in an origin of its own with no network access. That needs three settings: the server's `--copal-addr` and `--copal-render-origin` (`ANTUMBRA_COPAL_RENDER_ORIGIN`, copal's `COPAL_RENDER_ORIGIN`), and on copal, `COPAL_RENDER_FRAME_ANCESTORS` naming this server's address so copal lets the gallery frame its pages. Without them `publish_relic` refuses and says which is missing.

A relic downloads from its viewer as HTML (the page as it was published), Markdown (its headings, text, lists, code and tables) or PDF, so it can be sent to someone who cannot reach the server. The server reads the current version from copal and answers with a file under a `sandbox` policy, so the dashboard's origin still never runs the page. A script can ask for the same file: `GET /relic/<slug>/export?format=html|md|pdf`, with the token as a bearer for a private relic, or with none for a public one. A private relic and a missing one are the same 404.

PDF takes a browser engine, so the server hands the page to a [Gotenberg](https://gotenberg.dev) service named by `--pdf-renderer` (`ANTUMBRA_PDF_RENDERER`) and returns what it prints, backgrounds included and at the page's own `@page` size. The Compose `pdf` profile runs one with no host port; set `ANTUMBRA_PDF_RENDERER=http://pdf:3000`. It refuses private addresses (`--chromium-deny-private-ips`), so a page can load web fonts and libraries while it prints but cannot reach your network. Without a renderer the PDF button says so, and the browser's Print on **Open alone** saves one instead.

---

## Agents other than Claude Code

### The MCP bridge

Claude Code is given the server's address and a helper that reads the token from its file for each connection. An agent without such a helper would need the token in its own config file, or in the environment of every shell it is started from. `antumbra mcp bridge` is the server's MCP endpoint as a command instead: the agent runs it and talks over its pipes, and the bridge carries each message to the server set up on this machine (`~/.antumbra/setup.json`), with the token read from `~/.antumbra/token.txt` for that request.

```json
{ "command": "antumbra", "args": ["mcp", "bridge", "--agent", "<name>"] }
```

- **No token in the agent's config.** A token replaced on disk is the one the next call presents, with the agent still running.
- **A plain `http://` address works.** Whether the token may travel to an address was setup's decision (`--allow-http`), and it is not made again by each agent.
- **A server that restarted costs the agent nothing.** When the server no longer knows the session, the bridge opens another with the handshake the agent first sent, and asks again.
- **Instructions.** When the server's handshake gives none, the bridge adds a paragraph on when to recall and what to store, which an agent places in its system prompt. It is what keeps an agent recalling and storing where its hooks do not run.
- It names this machine to the server (`X-Antumbra-Host`), so what the agent writes is stamped as written from here.
- **Both ways of speaking MCP are carried.** An agent that opens a session (`initialize`) is given one. An agent on the 2026-07-28 revision opens none: it asks `server/discover`, and each request says which revision it speaks. Over HTTP that revision wants the same said in headers (`MCP-Protocol-Version`, `Mcp-Method`, `Mcp-Name`), which the bridge writes from the request. goose speaks this one.

`ANTUMBRA_URL` (or `--surface`), `ANTUMBRA_TOKEN`, `ANTUMBRA_TOKEN_FILE` (or `--token-file`) and `ANTUMBRA_HOST_ID` win over what setup recorded, as they do in the hooks. `ANTUMBRA_BRIDGE_PATIENCE_SECS` (600) bounds one call. Not carried: what the server sends unasked, which today is a log line when another session changes a memory; and `Mcp-Param-*` headers, which a tool asks for in its schema and none of the server's do.

### One command for every agent's hooks

The three touchpoints are scripts written to Claude Code's hook contract: its field names on standard input, its JSON on standard output, the server's address in its settings. Another agent has the same three moments and a contract of its own for each. Its hooks run `antumbra hook <agent> <event>` (`session-start`, `prompt`, `stop`), which hands the agent's input to the same script in the names the script reads, runs it with the address, workspace and machine name setup recorded, and answers in the shape that agent reads. Every agent gets the bootstrap, the recall for each prompt and the reminder to store that Claude Code gets, from one implementation.

It never fails the agent: whatever goes wrong is said on standard error, and nothing is printed. Run this way, the scripts differ from Claude Code's run of them in two things. The sovereign-mode brief is off (`ANTUMBRA_BRIEF=0`), because it is about Claude Code. And the reminder at a turn's end is the short one (`ANTUMBRA_CAPTURE_REMINDER=brief`), because some agents show it to the person as well as to the model.

### Which agents, and what setup writes for each

`antumbra setup` connects Claude Code and every other agent it finds on the machine: one whose program is on the PATH, or whose config directory holds its own config file (Gemini CLI's `settings.json`, Codex's `config.toml`, Copilot CLI's `settings.json` or `config.json`, goose's `config.yaml`, OpenCode's `opencode.json` or `opencode.jsonc`). A config directory with none of those is what an agent that was removed leaves behind, and is not counted. Setup's last lines say which agents it connected because it found them. `--agent` names them instead (`--agent gemini`, `--agent claude,codex`, `--agent opencode`, `--agent all`; the names are `claude`, `gemini`, `codex`, `copilot`, `goose` and `opencode`), and `--no-claude` leaves Claude Code out.

Setup touches as little of a person's own files as each agent allows. Where an agent reads a directory of Antumbra's own, that is all setup writes: nothing the person wrote is edited, and connecting again replaces it (Gemini CLI, goose's plugin, and OpenCode's). Where an agent reads only its user's files, setup adds to them as it does for Claude Code: with the agent's own command, or as text, adding only, with a backup, and never twice (Codex, Copilot CLI's list of MCP servers, and one line of goose's config). Either way `antumbra setup check` reads back what was written.

| Agent | What setup writes | Opens with memory | Recall for each prompt | Reminder to store |
| --- | --- | --- | --- | --- |
| Claude Code | hooks and `env` in `settings.json`; the MCP server, with a headers helper | yes | yes | yes |
| Gemini CLI | an extension, `~/.gemini/extensions/antumbra`: the bridge and three hooks | yes | yes | yes |
| Codex | the bridge, with `codex mcp add`; three hooks added to `~/.codex/hooks.json` | yes, once you approve the hooks | yes, once you approve the hooks | yes, once you approve the hooks |
| Copilot CLI | a hooks file, `~/.copilot/hooks/antumbra.json`; the bridge added to `~/.copilot/mcp-config.json` | yes | yes | yes |
| goose | a plugin, `~/.agents/plugins/antumbra`: the bridge and three hooks; one line in goose's `config.yaml` | yes, through the file goose keeps in mind | yes, the same way | yes |
| OpenCode (1.x and 2.x) | one plugin file, `~/.config/opencode/plugins/antumbra.js` | yes | yes | yes, while the session stays open |

**Gemini CLI.** The extension holds a manifest with the bridge as its MCP server and a `hooks/hooks.json` with `SessionStart`, `BeforeAgent` and `AfterAgent`. Gemini CLI lists it (`gemini extensions list`) and `gemini extensions uninstall antumbra` removes it. Two things are Gemini CLI's own to decide, and setup leaves them as they are:

- In a folder you have not trusted, Gemini CLI connects no MCP server, your own included.
- It asks before an Antumbra tool runs, and run for one prompt (`gemini -p`) it leaves out tools that would need asking. To allow them, add a policy file of your own:

  ```toml
  # ~/.gemini/policies/antumbra.toml
  [[rule]]
  toolName = "*"
  mcpName = "antumbra"
  decision = "allow"
  priority = 500
  ```

At a turn's end Gemini CLI shows the reminder to you as a warning that the agent was blocked. That is the reminder reaching the agent, which is then asked once more and stops.

**Codex.** The bridge is registered with Codex's own `codex mcp add antumbra -- antumbra mcp bridge --agent codex`, which writes `[mcp_servers.antumbra]` into `~/.codex/config.toml` (or where `CODEX_HOME` says). Without `codex` on the PATH (Codex named with `--agent` on a machine that does not have it), setup prints that command, and `antumbra setup check` says the server waits on Codex being installed without failing. The hooks, on `SessionStart`, `UserPromptSubmit` and `Stop`, go into `hooks.json` beside any of your own. One thing is Codex's own to decide, and setup does not go around it:

- **Codex skips a hook until you approve it.** Start Codex, type `/hooks`, and approve Antumbra's three. Until then a session has Antumbra's tools and the bridge's instructions, and none of the three touchpoints; setup says so when it adds them, and Codex says nothing when it skips them. Approval is asked again when a hook's command changes, which it does when `antumbra` moves on or off the PATH.

Codex moves a hook's context to a file past some 2,500 tokens, so its bootstrap is kept to 7,000 characters (`ANTUMBRA_CONTEXT_LIMIT`) and says how many memories it left out. At a turn's end Codex shows the stop hook as blocked: that is the reminder reaching the agent, which is asked once more and stops.

**Copilot CLI.** The hooks are a file of Antumbra's own in `~/.copilot/hooks` (or under `COPILOT_HOME`), with `sessionStart`, `userPromptSubmitted` and `agentStop`; Copilot CLI runs a user's own hooks without asking. The bridge is one entry, `mcpServers.antumbra`, added to `mcp-config.json` as text, with the rest of the file left byte for byte and a backup beside it. Setup edits that file itself where it uses Codex's own command for Codex's, because the `copilot` on a PATH is not always Copilot CLI: VS Code puts a launcher of that name there.

- Copilot CLI hands its turn's-end hook the path of the session's event log and not the answer, so the answer the server judges a turn's recalls against is read from the end of that log. The log's shape is Copilot CLI's own; when it does not read as expected the turn goes unjudged and nothing else changes.
- It puts an MCP server's instructions in its system prompt only for servers on a list of its own, so the bridge's are not read there (`--allow-all-mcp-server-instructions` reads them for one run). The hooks carry what a session needs.
- It asks before an Antumbra tool runs unless you allow it: `--allow-tool antumbra` for a run, or its permission settings.
- Copilot in VS Code reads the same `~/.copilot` hooks and MCP servers, by VS Code's own account. That was not run here.

**goose.** Antumbra is a plugin, `~/.agents/plugins/antumbra`, with the bridge as its MCP server (`.mcp.json`; goose lists it as `antumbra:antumbra`) and hooks on `SessionStart`, `UserPromptSubmit` and `Stop`. goose's hooks differ from the others' in one thing that shapes the rest: what a hook prints never reaches the model. A prompt hook's output is dropped, and a session-start hook can only show you a line.

What goose does read on every turn is a file its config names, `GOOSE_MOIM_MESSAGE_FILE`, which it puts in front of the model as what is on its mind. So the hooks write that file (`~/.antumbra/goose/top-of-mind.md`): the session's bootstrap, kept to 4,000 characters because goose hands it over again with each turn, and under it what was recalled for the prompt in front of the model. Setup names the file in goose's `config.yaml` by adding one line, with a backup, and only when the config names none.

- **If you already keep a file of your own in mind there, setup leaves it yours** and says so: goose then has Antumbra's tools, the bridge's instructions and the reminder, and nothing recalled is put in its context. Remove your line and run setup again to change that.
- **The file is one for the machine.** Two goose sessions at once each write it, so a turn may read what the other session's prompt recalled. It is your own memory either way, and the next prompt rewrites it.
- **On Windows goose runs its hooks with `sh`**, as it does everywhere, so its hooks need one on the PATH (Git for Windows has one). Setup and `setup check` say when there is none. The tools work either way.
- At a turn's end goose shows "Stop hook `antumbra` blocked ending this turn": that is the reminder reaching the agent, which is asked once more and stops.
- goose's config is `~/.config/goose/config.yaml`, on Windows `%APPDATA%\Block\goose\config\config.yaml`; `GOOSE_PATH_ROOT` moves both it and the plugins.

**OpenCode.** OpenCode has no hooks that run a command. It has plugins, and two release lines, 1.x and 2.x, that share a config directory and a binary name and whose plugins are written differently. Antumbra is one file, `plugins/antumbra.js` in `~/.config/opencode` (on Windows too; `OPENCODE_CONFIG_DIR` and `XDG_CONFIG_HOME` move it), whose default export is both: `server` for 1.x and `setup` for 2.x. Each line takes the half it knows.

- The plugin runs `antumbra hook opencode <event>` at a session's first prompt, at every prompt, and when a turn ends, and puts what the hook answers in that session's system prompt, where the model reads it with every call.
- It registers the bridge as an MCP server from inside the plugin, which both lines allow, so nothing is added to `opencode.json`. A server you already have under the name `antumbra` is left yours.
- The reminder at a turn's end is sent to the session as a prompt, so it needs the session still there: in the terminal interface and against `opencode serve` it is. `opencode run` on 1.x exits as the turn ends, before it can be sent; the turn is still reported.
- 2.x offers MCP tools to the model through its Code Mode by default, so the model calls Antumbra's as `tools.antumbra.recall_memories(...)` inside `execute`. That is OpenCode's own setting (`codemode`), and the tools are the same.
- The plugin also runs for the short request OpenCode makes to title a session, which is told what the session is told.

### Agents started through Unity Gateway

Unity Gateway (`ug`, Databricks' launcher: `ug claude`, `ug gemini`, `ug opencode` and the rest) starts a coding agent against a workspace's gateway. It keeps what it writes in `~/.ucode`, and for some agents that is the config directory itself, which it points the agent at for the session. Such an agent reads nothing setup wrote where the agent keeps its config on its own.

So on a machine `ug` is on (its program on the PATH, as `ug` or `ucode`, or a workspace it saved in `~/.ucode/state.json`), `antumbra setup` also writes for each agent as `ug` starts it, and `antumbra setup check` reads both back. Nothing of Unity Gateway's own is edited: what setup adds is a directory or a file that is Antumbra's whole, beside what `ug` keeps.

| Agent | How `ug` starts it | What setup adds |
| --- | --- | --- |
| Gemini CLI | with its home moved to `~/.ucode/.gemini-home` (`GEMINI_CLI_HOME`) | the same extension, in `~/.ucode/.gemini-home/.gemini/extensions/antumbra` |
| OpenCode | with its config home moved to `~/.ucode/opencode-xdg` (`XDG_CONFIG_HOME`) | the same plugin file, `~/.ucode/opencode-xdg/opencode/plugins/antumbra.js` |
| Claude Code | with `~/.claude/ucode-settings.json` over your own settings, which it still reads | a settings file of Antumbra's own, `~/.antumbra/claude-settings.json`, for the one kind of launch that leaves yours out |
| Codex | from `~/.codex`, with its own settings as overrides | nothing: the hooks and the server setup wrote there are the ones read |
| Copilot CLI | from `~/.copilot` | nothing, likewise. `ug` passes `--allow-all-mcp-server-instructions`, so the bridge's instructions are read there |

**Claude Code and a relayed subscription.** One kind of launch does not read your settings. When a workspace relays a Claude subscription, `ug` starts Claude Code with `--setting-sources project,local`, which leaves your own settings out, and Antumbra's hooks with them. The MCP server is still there, because `ug` hands the servers you registered back to that launch. `ug` merges a `--settings` file its caller passes into its own, keeping the hooks of both, so that launch is started with Antumbra's:

```sh
ug claude --settings ~/.antumbra/claude-settings.json
```

The file holds what setup wires into Claude Code and nothing else: the hooks, `ANTUMBRA_URL`, `ANTUMBRA_WORKSPACE_ID` and `ANTUMBRA_HOST_ID`, and `autoMemoryEnabled: false` where your own settings have it. Setup and the check tell such a launch by what `ug` wrote for it (no `apiKeyHelper`, and a gateway address that is a relay on this machine), and print the command when they see one. Given to a launch that reads your settings too, the file changes nothing: Claude Code runs a handler it finds in two settings files once.

**A setup run from inside one of these sessions.** An agent `ug` started hands the variables `ug` set for it to the commands it runs, `antumbra setup` among them. Setup reads a `GEMINI_CLI_HOME` or an `XDG_CONFIG_HOME` that names a place inside `~/.ucode` as not set, so the agent's own directory is still the one in your home, and goose's config is not looked for under OpenCode's.

**What an organization's policy stops.** A workspace's configuration can reach an agent as managed policy, and a policy can stop what setup wired with nothing said in the session: the hooks do not run, or the server is gone from the agent's list. `antumbra setup check` reads the policy files, says which one stops what, and whose it is to change.

- **Claude Code's managed settings** (`managed-settings.json`, its `managed-settings.d` drop-ins and `managed-mcp.json`, in `/etc/claude-code`, `/Library/Application Support/ClaudeCode` or `C:\Program Files\ClaudeCode`). For the hooks: `allowManagedHooksOnly`, `disableAllHooks`, and `strictPluginOnlyCustomization` over `hooks`. For the server: a `managed-mcp.json` that does not name it, `strictPluginOnlyCustomization` over `mcp`, a `deniedMcpServers` entry that matches it, and an `allowedMcpServers` list that does not admit it. The check judges the server against the lists as Claude Code's own page says it does: by its address once a list holds one (`serverUrl`, with `*`, and a host written without a port matching the scheme's own port alone), and by its name until then. It prints the entry that would admit it.
- **Codex's requirements** (`/etc/codex/requirements.toml`, on Windows `%ProgramData%\OpenAI\Codex\requirements.toml`): `allow_managed_hooks_only = true`, `hooks = false` under `[features]`, and an `mcp_servers` list that does not name `antumbra`.

A list in the settings `ug claude` passes is judged for that launch as well, and said as the workspace's to change, since `ug` writes that file again from the workspace's configuration. With the hooks stopped, the memory tools still work when the agent is asked to use them. With the server kept out, the hooks still put what is recalled in front of the agent, and it cannot search or store. Only files are read: a policy delivered by a vendor's console, an MDM profile or the registry is not seen.

**Not covered.** `ug` also starts Kilo, Pi and Google's Antigravity, which setup does not connect. What is said here of Unity Gateway is what its source and README say (read on 2026-11-20). None of it was run against a workspace: what was run is setup and the check in a home laid out as `ug` leaves one, and Claude Code started the way a relayed launch starts it, where a hook from the `--settings` file ran with that file's `env`.

## Leaving work for another of your machines

A session can leave a note for a session on another of your machines: "rerun this on the GPU box", "check the dashboard after the deploy". The agent calls `leave_handoff` with the text (its first line is the title) and `for_host`, the machine's host name, or nothing for whichever of your machines starts a session next. A session that starts there opens with the count and a line for each handoff waiting for it, ahead of recalled memories, and keeps hearing about one until a session calls `complete_handoff`; after that it is no longer announced but `handoffs` with `include_done` still shows it. The machine's name is `ANTUMBRA_HOST_ID`, the same one the bootstrap prints in its first line, so set it on each machine. Each session start also names the machine to the server (`register_device`), with its operating system, so `devices` lists every one of your machines with when it was last seen. Last seen is the machine's last call, noted at most every five minutes, and not the start of its session; a machine a server runs on is seen for as long as that server is up. On the dashboard's Devices page you can give a machine a name of your own (`POST /devices/name` with `host` and `name`; a blank name takes it away). The name is for people: handoffs are still addressed to the host name. That includes a laptop whose agent talks to a hosted server and runs no server of its own. A machine named that way is a memory node: the server cannot see its hardware, and a machine that trains runs `antumbra-mcp` itself, which registers it with what it can do. Handoffs are yours alone: they live in your own `handoff` compartment, and no other user's session sees them. It is a note, not remote control: nothing runs on the other machine until a session there decides to.

## Behaviors: what your own expert learns

A memory is something to look up. A behavior is how you want an agent to act on a class of tasks: "reach a host by its SSH alias", "name an issue branch `feat/{issue}-{slug}`", "open pull requests against main". When you state or correct one, the agent records it with `record_behavior`:
- `rule`, the rule in one sentence;
- `must` and `must_not` regular expressions that decide whether an answer follows it;
- `examples`, at least four `{task, answer}` pairs whose answers follow it, since training holds one in three out to admit your expert by (ten, varied in phrasing and case, teach a rule about a command or a code form reliably; a prose rule about how to go about a task needs sixteen to twenty, since at ten most such rules were learned on only some of their held-out tasks);
- `violations`, answers that break it, each a plain string (one written as a `{task, answer}` pair is read as its answer).

The check is tested against the examples and the violations before anything is stored, and `problems` says what to fix when it fails. A call that leaves out `rule`, `must`, `examples` or `violations` is answered the same way, with everything it lacks in one `problems` list, so the agent fixes the call once instead of meeting one missing field per try.

A behavior is proposed until you accept it (`accept_behavior`, or Accept in the dashboard's Behaviors section), unless you stated it yourself. One nobody accepts is accepted for you by the server that trains, once it can be taught and has waited proposed for a day: accepting was the one step from a behavior to an expert that waited on a person. The workspace's `accept_after_hours` knob (`K` in the console) sets the wait, and 0 leaves accepting to you. `list_behaviors` and the dashboard say which were accepted that way. The workspace's `standing_training` knob (also under `K`) pauses the training of its standing experts at 0, leaving what is served as it is, and resumes it at 1. One expert is taught at most the workspace's `behaviors_per_expert` behaviors (10 unless changed, also under `K`): past it, an accepted behavior waits for room, and `list_behaviors` (`waiting_for_room`), the dashboard and the console's Training page say how many wait. A behavior the base model already follows is left out of the set and said to be (`followed_by_base`). `retire_behavior` drops one, and recording a new one with `supersedes` replaces an old one. `list_behaviors` shows them by status. They live in your own `behavior` compartment, private to you, and recall finds them like any memory.

The rules you stated before behaviors were records are still in your memories, in session notes and corrections. `behavior_candidates` reads the store in the order it was written and returns the memories that may state one: every opinion memory, any memory opening with a tag such as `[feedback]` or `[convention]`, any memory with a phrase that records a correction or a preference ("standing preference", "pushed back", ...), and any memory with a short sentence that opens like a rule, with a negation or an imperative verb ("never stack a pull request", "use the host alias"). Those sentences come back with the candidate, since a rule is often an aside deep inside a long note. A word like "must" or "never" anywhere is not enough: measured on a store of 7,000 memories, that picked two thirds of it, because technical notes say those words constantly; the sentence shape picks under a third, and most of what it picks states a rule. Behaviors, handoffs, volatile memories and memories a behavior already cites are left out. The agent reads each one and, where it states how to act and a program could check an answer, records it as a proposed behavior citing the memory in `sources`, along with any memories that restate it. `list_behaviors` shows the sources, and a cited memory is not offered again. Each call returns `scanned_through`, which the next call passes as `after`, so a pass over the store converges and a later one starts where the last stopped. `antumbra behave --import --propose` records a file of such behaviors, each row carrying its `sources`.

Accepted behaviors are what your own expert is trained on. It learns each one from tasks the behavior governs and its check, never from the memory's text. A scope's expert is admitted holding the behaviors it learned. A behavior it did not learn stays accepted and is not served, and it does not keep the rest out. The expert is refused when it answers any taught behavior worse than the base model (judged as a probability from the held-out counts, so one lost task of three does not refuse and three lost of three does), when unrelated tasks (the controls) fall, or when nothing it learned rose above the base model. `list_behaviors` shows each behavior's `last_training`: whether it was learned, by its held-out pass rate against the base model's, how likely it was made worse when the base model already followed it, and whether its scope's expert was admitted. An expert is left as it is until a behavior in its scope is added, retired or recorded again. When one is, and the expert serving the scope is on the node, the keeper first measures that expert and the base model on the examples the behaviors gained since it was trained, tasks neither it nor its admission ever saw, and notes the result as each behavior's `last_audit`. A behavior recorded again with more examples is followed: its new examples are measured against what the expert knew of the one it supersedes. The audit changes nothing by itself: it is the evidence the regression confidence is tuned from. On a node told to tune (`--behave-tune-confidence`), the keeper pools each workspace's audits of the admissions that bet a slip was noise, and once two readings in a row agree that the bets held, or failed, moves the confidence a step (0.02, within 0.6 and 0.98). `list_behaviors` returns the value in force as `tuned_confidence`, with the recent readings as `tuning`; absent, the server's recipe applies. A refused set is not trained again, even after a restart, until it changes. A behavior the last training did not learn also shows, as `missed`, one held-out task it was missed on and what the trained expert answered instead. An answer that does the right thing in words the check does not expect says to widen the check; one that does something else says the examples have not taught it yet. To teach a behavior that was missed, record it again with more examples, or a better check, superseding it.

## Claude Code with its telemetry off

Turning off Claude Code's telemetry (`DISABLE_TELEMETRY`, `DO_NOT_TRACK`, `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC`) also turns off its feature-flag fetching, and so does running it on a third-party provider. A list of features that have nothing to do with telemetry goes with the flags, and nothing announces it: a repository whose only instruction file is `AGENTS.md` silently stops instructing the agent. Antumbra calls this state sovereign mode and treats it as the normal case.

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

The operator console shows both on its Agent page (`5` in `antumbra tui`, or `antumbra tui snapshot --page agent` for one frame): the rules, current or retired, and skill use, least recently used first, from the memories of the workspace the console's token reads. It is read-only and queries nothing new.

## How the system keeps its memories

Nobody has to reinforce or penalize a memory by hand. The server watches how each one is used and keeps its standing itself:

- **Shown.** Recall counts every memory it returns. The prompt hook names its session on the recall, so the server knows which memories that session was shown.
- **Used or passed over.** When a turn ends, the capture hook sends the turn's last answer (`observe_turn`, which no agent is offered). The server asks the relevance head, the same one behind recall's floor, whether the answer drew on each memory the session was shown; where no head is configured it compares embeddings. Each one drawn on counts as used, the rest as passed over. What an agent recalled for itself is tied to no one session, so it counts only when the answer drew on it, never against it.
- **Restated.** A new memory is linked (`references`) to the existing ones it comes close to, and one close enough to say the same thing counts as that memory recurring. `store_memory` returns the ids it linked.
- **Judged by an agent.** When an agent's own work shows a memory right or wrong, it may `reinforce_memory` or `penalize_memory` it. That moves the memory's confidence, and the server counts it beside its own signals. The prompt hook shows each memory's id so the agent can.

From those counts the server reads each memory's **strength** whenever it reads the memory: its confidence, halved for every half-life since it was last of use (or written, if never), and lowered the more often it was passed over against how often it was used. Recall lets strength move its order a little after relevance, and returns it as `strength`. Nothing is written back, so a knob change applies to the next read and loses nothing.

The counts live in their own table, apart from the memory, so counting a recall never re-sends a memory to other devices. Each server counts what it sees; counts are not replicated.

**The knobs** belong to the workspace, and an operator turns them from the console (`K`) while the server runs: whether the system governs memories at all (`auto`), the half-life of an unused fact, experience, judgment and volatile memory (180, 90, 180 and 14 days), how sure it must be that an answer drew on a memory (`use_threshold`, 0.55), how close a new memory must come to be linked (0.80) or to count as a restatement (0.92), the most links a new memory gets (3), how much each pass-over weighs (0.1), and how far strength moves recall's order (0.25, where 0 is relevance alone). An agent's token cannot change them.

## The operator console

`antumbra tui` (or `antumbra --tui`) shows the population, the memory, the training loop and its evaluations, read from the Antumbra server your agent uses. It needs no flags after `antumbra setup`, or a Claude Code wired by hand:

- The server is `--surface`, else `ANTUMBRA_URL`, else the one Claude Code's settings name (`env.ANTUMBRA_URL` in `settings.json`, under `CLAUDE_CONFIG_DIR` or `~/.claude`, where `antumbra setup` writes it), else the one `antumbra setup` recorded in `~/.antumbra/setup.json`, else `http://127.0.0.1:8081`.
- The token is `--token`, else whichever token on hand is an operator's, else the first found: `ANTUMBRA_TOKEN`, the file `ANTUMBRA_TOKEN_FILE` names, `~/.antumbra/token.txt`, then those two in Claude Code's settings. A hook token in the environment and an operator's token saved to the file give the console the operator's.

The connect panel (`c`) says where each was found. It reads every few seconds, and the server answers an unchanged view with no body. `--demo` shows a seeded population held in memory, with no server.

The footer lists the keys of the view in front of you, and `?` lists every key of the page you are on; both are drawn from the list the keys are read from. `x` acts on what is selected: it freezes or thaws an expert, prunes a shadow, deletes a boundary, and halts the selected loop run or cancels its halt. `p` graduates a shadow. Each asks first, and only `y` goes ahead. What an action did, a refusal, or the reason nothing happened is said on the line above the footer, and `e` keeps the history. Esc closes a panel and does nothing on the main view; `q` or Ctrl-C quits.

It is drawn in the dashboard's palette. `--theme` (`ANTUMBRA_THEME`) picks which: `night`, the dark one and the default; `mist`, the light one, for a light terminal; or `mono`, greys on the terminal's own background, for a terminal with few colours or one whose background should show through. `t` cycles them while it runs. A state is told by a shape and a word, never by colour alone: a filled dot for what is settled or in force, a ring for what is open or under way, a cross for what ended and a triangle for what failed, which is the one thing drawn in another colour.

It uses the dashboard's words. Its pages are Population (the experts, the candidates in training and the boundaries), Memory, Training, Evaluations, Agent and Services; a candidate is what the rest of this document calls a shadow, and its id still begins `shadow:`. The top bar says which server is read, whether the token is an operator's or read-only, and whether reads are succeeding. Under the tabs, a sentence says what the page in front is, and the help (`?`) ends with what each word means.

`a` asks which expert a task would go to. The answer is the experts it is routed to, each with its share, or that none is routed and the server's reason why. Under it are the standing experts, which are never sent a task and are in every answer in their scope instead: the one for everywhere, and the one for the repository of the directory the console was started in. The console names a standing expert by its scope (the repository's name, or `everywhere`), and an expert's detail says which kind it is. For the task asked, a server that reads them says how close the task is to what each standing expert was taught and what that comes to (it answers the task, it is in scope and too far to answer alone, or it is for another repository), and lists the nearest of the behaviors you have taught with where each stands: learned, not learned at its last training, not trained yet, or proposed and waiting to be accepted. The `route` tool returns the same, as `standing` and `taught`, to any caller; pass it `repo` for the repository the task is in. The dashboard's route form shows both too.

What it shows depends on the token. Any token shows its own workspace: its memories and their links, and the experts it may route to. An operator's token also shows the substrate, which belongs to no workspace: shadows in training, failure boundaries, the loop and its evaluations, and the router. With any other token those lists say an operator's token is needed, not that they are empty. Only an operator's token may act from the console: prune or graduate a shadow, freeze or thaw an expert, delete a boundary, halt the loop or cancel the halt, and turn the workspace's memory knobs (`K`). The Population page opens on a list beside the detail of what is selected: the experts as a table (name, generation, fitness, state, and whether each is shared), and Tab moves along Experts, Candidates and Boundaries, which the list's title names with their counts. `l` changes the layout, to the experts orbiting as a graph among others. A standing expert is trained again each time its behaviors change, and each training is a candidate, so one expert has several: the list says when each was and marks those a later run took the place of as replaced, and the chart beside it is the expert's pass rate from one run to the next, over the base model's on the same tasks. A run kept from before the base model's reading was recorded says so. The Memory page lists every memory, strongest first within each kind, beside the selected one's detail: its strength beside its confidence, how often it was recalled, used and passed over, and when it was last of use. Tab moves along All, World, Bank and Opinion, and `l` swaps the list for the selected memory's links, drawn: the memory, what it links to and what links to it, each by how its content starts, with a line's stroke for the kind of link (solid for a reference, dashed for a supersession, dotted for the rest) and, where the panel is wide enough to keep the names readable, what those are linked to in turn. A store holds thousands of memories and a few links, so there the keys move through the memories that have one, the most linked first, and the title says how many that is. The server's operator mints one with `antumbra-mcp --mint-token --operator`, and `antumbra setup local` mints one for whoever runs the stack. A hosted workspace's tokens are never operators'.

The Training page opens on the standing experts, which the server trains by itself: each scope with how many of its behaviors are trained, accepted and proposed, when it was last trained, and where it stands, which is up to date, trained at the server's next pass, not admitted, about to be dropped, or waiting on a proposed behavior the server accepts in so many hours. It is worked out from the behaviors and experts the console already reads, with the functions the server's keeper decides by, so it says what the server will do next without asking it. Under it is the generational loop, for when one is run. The Evaluations page names a standing expert's training and its audit in words, each with the base model's reading beside the expert's and when it was.

The Services page (`6`) draws the workspace's dependency claims as a graph and says what a change to a service reaches. It reads left to right: a service is drawn before what it depends on, and an arrowhead marks the service a line arrives at. A line's stroke is the source of the strongest claim behind it: solid for one declared in a manifest, dashed for one observed, dotted for one learned or claimed, and the faintest ink for a pair too weak for a walk to follow. Lines into one service join into a trunk, and a dependency that skips a column runs through it in a lane. The selected service is where the walk starts; what it reaches is lit with the lines it took, and listed beside the graph with how sure each path is, how many hops, and what says so: the claim itself one hop away, or the service it was reached through. Tab turns the walk round, from what depends on the service to what it depends on; `h` lets it go one hop, three or six; Enter lists every claim about the service with its source, weight, when it was last seen and the file that declares it. The page opens on the service the most depend on. It is worked out from the memories the console already reads, with the arithmetic `blast_radius` uses, so it asks the server nothing more and reads the same under any tool profile. A graph with more services in a column than the panel has rows is drawn as the selected service and what is within reach of it.

`antumbra tui snapshot` draws one frame with no terminal and writes it as text and as a picture, for a look at a server from a script or a check in CI. `--page`, `--layout` (`list`, `dashboard`, `graph`, `table`), `--focus` (`experts`, `candidates`, `boundaries`) and `--overlay` say what the frame shows, in the words the console draws. The names these had before are still taken (`loop`, `focused`, `shadows`, `gate`), and a name nothing has is refused with the ones that are taken.

The console reads `GET /console/snapshot` and acts with `POST /console/action`, both with the bearer token. Neither is an MCP tool, so no agent is offered them.

Two more say what the doctor knows to the agent. The session-start hook opens with `antumbra claude brief` ([hooks](../scripts/hooks/README.md)), and `antumbra claude remember` keeps the same rules as `world` memories in a `claude-code` compartment of your own: volatile, so they never train an expert, and safe to run again. The session-start hook runs it for you, detached, in the mode the rules are for, and `antumbra setup` wires the two hooks that count which skills are used, so the console's Agent page fills without a command being typed.

## Who can recall a document

A document is kept the way a memory is: in a compartment, or in the workspace's shared pool. `ingest_document` takes a `compartment` (and `antumbra ingest` a `--compartment`); then only the compartment's owner and the people it is shared with can recall the document, enforced by the engine on every read, and a recalled chunk says which compartment it came from. Leave it out and the document goes to the shared pool, which every member of the workspace can recall. That is the right place for reference material (it is where the GitHub integration puts a repository's documents) and the wrong place for anything private.

A compartment has to be one the caller can write to: their own, or one shared with them with `link`. The check happens before anything is archived or stored, so a refusal leaves nothing behind. A document's identity is its workspace, its compartment and its title, so two members can each keep a private document under one title: neither ingest touches the other's chunks, and each has its own archived original.

Before this, every document in a workspace was recallable by every member. Documents ingested then have no compartment, so they are in the shared pool and exactly as recallable as they were; to make one private, ingest it again into a compartment. An existing database picks the rule up on its next start: the server re-asserts each table's permissions when it connects, since a definition applied `IF NOT EXISTS` would otherwise never change on a table that already exists.

## Copal as the document of record (knowledge documents)

`ingest_document` chunks, embeds, and stores a knowledge document for recall — and in v0 that is *all* it keeps: the chunks. Recall works, but the original bytes are gone. Point the MCP server at a **Copal** file service (content-addressed, versioned, sealed-at-rest file storage) and Copal becomes the **document of record**: on every ingest the original content is uploaded there first, and each stored chunk carries provenance back to it (`copal_file`, the archived file's id, and `copal_digest`, the content digest of exactly the bytes that were ingested) — so a `recall_documents` answer names not just what it remembers but the original it came from.

The address, on the server's usual clap/env conventions:

```
ANTUMBRA_COPAL_ADDR=127.0.0.1:9010    # --copal-addr: bare host:port (http:// for this machine, https:// elsewhere) or a full URL base
```

An explicit `http://` address that is not this machine is refused unless `--copal-allow-http` (`ANTUMBRA_COPAL_ALLOW_HTTP`) says the network between the server and the archive is trusted, as the compose stack's own is.

Tenancy is the deployment's second choice: *which Copal tenant does a workspace's document land in, and how does the call prove it?* Copal's `header` auth mode (its default until 1.0) trusts the `x-copal-tenant` header as the identity; its deployed `keys` mode binds the tenant to a `ck1` credential. On the networked server (`--http`), where many identities write through one process, a header-named tenancy leaves isolation at the archive to that network alone, so the server refuses to start with one unless `--copal-trust-headers` (`ANTUMBRA_COPAL_TRUST_HEADERS`) says the network is yours; the compose stack says so for its own network, and a hosted deployment should prefer the keyed modes. One knob per shape, at most one of the three (two together refuse at startup):

|                           | Copal `header` auth (dev default)                             | Copal `keys` auth (deployed)                                        |
| ------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------- |
| **Per-workspace tenancy** | *(the default — nothing to set)* each workspace IS its tenant | `--copal-keys <file>`: JSON mapping workspace → `ck1` key           |
| **Shared tenant**         | `--copal-tenant <name>`                                       | `--copal-key <ck1 key>` (the key's tenant is *the* tenant)          |

(`ANTUMBRA_COPAL_TENANT` / `ANTUMBRA_COPAL_KEY` / `ANTUMBRA_COPAL_KEYS` as envs.) Per-workspace tenancy means each workspace gets its own quotas, listings, and search scope, with cross-tenant reads refusing at Copal's own boundary — and the `--copal-keys` column is how that survives a Copal deployment upgrading to `keys` auth: mint a key per workspace on Copal's admin surface, map them in the file, restart to pick up new ones. Every key needs Copal's `write` scope, since both calls write.

The contract:

- **Upload first, fail closed.** The original lands in Copal *before* any chunk is stored, and an unreachable Copal fails the ingest with a clear error. A configured document of record that silently dropped originals would be worse than none. Under `--copal-keys`, a workspace with no mapped key fails the same way — refusing beats archiving into a tenant that is not the workspace's own.
- **Re-ingest revisions, never litters.** The create carries an idempotency key derived from (workspace, title), where the title of a document kept in a compartment carries that compartment, so two members' private documents of one title are two archived files and "the original" of one is never the other's text — the *antumbra* workspace tenant, in every tenancy shape — so ingesting the same title again revisions the *same* Copal file, two workspaces sharing a title never revision each other's document (even inside a shared tenant), and moving a deployment between shapes never re-identifies a document. The version history is the document's history; the archived file's metadata names its owning workspace.
- **The bytes are verified, and identical bytes are not stored twice.** The upload declares the content's sha256 in `x-copal-digest`, and Copal refuses bytes that do not hash to it, so `copal_digest` is checked end to end rather than taken on trust. A re-ingest whose content the archived file already holds is not uploaded again, so unchanged content adds no version. A refused call fails the ingest with Copal's own reason (its error kind and message), not only a status code.
- **Absent means exactly today's behavior.** No `--copal-addr`, no archive: ingest keeps only the chunks, nothing new is required, and chunks written either way coexist (the provenance fields are simply absent on archive-less chunks).
- **Adapters are archived too.** A server with an archive uploads each adapter it admits (a private file under `adapters/`, `application/octet-stream`, the adapter's sha256 as `x-copal-digest`) and names the file on the expert's card; see [the GPU server](running-the-trainer.md#the-gpu-server-serving--standing-experts). Unlike an ingest, a failed upload does not fail the mint. These transfers have their own budget, `ANTUMBRA_COPAL_TRANSFER_TIMEOUT_SECS` (300 s by default).
- **Every door archives the same way.** The CLI's `antumbra ingest` takes the same four flags (`--copal-addr`, `--copal-tenant`, `--copal-key`, `--copal-keys`, or the `ANTUMBRA_COPAL_*` envs) and follows the same upload-first, fail-closed contract through the shared `antumbra-copal` client, so a CI step that ingests a framework's lister output or a generated service doc lands a document of record too.

---

## What a change to a service reaches

Which services depend on which is kept as claims with evidence, never as a parse. Record one with `record_dependency`: the two services as repository slugs, and the source, `declared` (a manifest or config names it), `observed` (telemetry saw the calls), `learned` (they change or deploy together) or `claimed` (read from the code). `list_dependencies` lists every pair with its evidence, weak ones included. Ask `blast_radius` what depends on a service, or what it depends on, and each service comes back by its strongest path with every hop's evidence, source and current weight, so the answer can be checked rather than trusted. A claim alone counts for little until a declaration or an observation backs it, and an edge nobody records again fades. The edges are the workspace's, readable by every member.

The declared edges can be read from your checkouts instead of recorded by hand:

```bash
antumbra claude dependencies --repos ~/repos --dry-run   # the edges it finds
antumbra claude dependencies --repos ~/repos             # records them
```

Each working tree under `--repos` (and the current one) is read for its package.json, Cargo.toml, go.mod, pyproject.toml and requirements files, and a repository depends on another when one of its manifests names a package the other publishes. Only the repositories read together are matched, so read them together. A fork, a manifest whose `repository` names another project, does not publish its upstream's names; installed and fixture manifests (`node_modules`, `vendor`, `testdata`, `fixtures`) are not read. Run it again when manifests change: an edge seen again is reinforced, one no longer seen fades. With the GitHub App installed this happens by itself for the repositories it is installed on, and an edge between two of them that no manifest makes any more is retracted rather than left to fade.

## Why this beats a plain memory layer

Retrieval-memory tools (give the agent a vector store to recall from) make the agent _remember_. Antumbra makes it **learn**:

1. **Bootstrap**: the agent starts the session already carrying your conventions and history (memory + identity).
2. **Route or answer**: the `answer` tool sends a task to the frozen expert most likely to cover it, or escalates when it is out of scope. A served task costs you nothing; only genuine novelty hits the expensive model.
3. **Capture**: verified outcomes and observations are written back.
4. **Metabolize**: `antumbra metabolize` turns the successful, recurrent traces (and their step-by-step decomposition) into a trained LoRA expert, frozen into the population so it is never forgotten.

Next session, step 1 includes a skill that did not exist before, and the work it covers is now served locally for free. The scaffolding (loops, prompts, lookups) shrinks into weights. A memory layer is static; Antumbra compounds.

See **[Architecture](architecture.md)** for the engine.

### Joining a workspace someone else hosts

A server that holds an HS256 secret can let people into its workspaces without anyone minting a token by hand.

1. An admin or an owner of the workspace makes a **join code** on the dashboard's Members page (or `POST /join/codes` with their token). It lasts 14 days and admits one person unless they say otherwise, and can be revoked. The server keeps its SHA-256, not the code.
2. The person asks with it, with no token: `POST /join/request` with `{code, handle, display, note, host, secret_digest}`, where `secret_digest` is the SHA-256, in hex, of a secret they keep. A handle is 3 to 32 of `a-z 0-9 - _`; `default`, the system's identities and names that read as an office are refused, and so is one somebody has. A code that will not do gets one answer whatever is wrong with it.
3. The workspace's admins see the request (`GET /join/requests`, the Members page) and decide it (`POST /join/decide`). The first decision stands.
4. The person asks how it stands with `POST /join/collect` `{request, secret}`. Approved, the answer carries their token, **once**: they are `user:<handle>` in the workspace, a member, with the name they asked under. A refusal comes with what the admin said. A request nobody decides expires with its code, and an approval nobody collects lapses after seven days.

From a machine, that is two commands. Asking changes nothing about the machine's own setup:

```sh
antumbra join https://memory.example --code aj1_... --handle dana --name "Dana Whitfield" --note "I work on checkout."
antumbra join --status   # how it stands; collects the token once approved
```

The first keeps `~/.antumbra/join.json`, the owner's alone, with a secret made on the machine; the server is sent the secret's digest. `--status` shows the secret to collect, writes the token to `~/.antumbra/workspaces/<server>__<workspace>/token.txt`, and leaves the token the machine already uses where it is: your private memories stay yours and stay in use. `antumbra join --forget` drops a request that is waited on. From then on the machine's prompt hook recalls from both: its own store first, as before, and under it what each workspace it joined holds for the prompt, named as the workspace's and read with the workspace's own token, all asked at once inside the hook's one budget. Nothing is written to a workspace by it, and `ANTUMBRA_RECALL_WORKSPACES=0` asks only your own.

Where a control plane signs a workspace's tokens, its server is started with `--control-plane <address>` (`ANTUMBRA_CONTROL_PLANE`), and joining verifies an address first: `antumbra join <server> --code <code> --handle <handle> --email <address>`. The server says where its control plane is; the request is made there, and a link is sent to the address. The workspace's admins see the request, marked with the verified address, once the link is followed; `antumbra join --status` says whether that has happened yet, whether an admin has decided, and collects the token from the control plane once one approved. An address with no account is given one by joining, and an account can be in several workspaces, as a separate member in each. On a self-hosted server none of this applies: an admin's approval is the verification, and `--email` is not needed.

To sign a machine in again there, for a lost token or a second machine: `antumbra login <server> --email <address>` has the control plane send a link, and `antumbra login --status` collects the token once it is followed. The browser that follows the link is given nothing; the token goes to the machine that asked, once, for a secret it made. An account in several workspaces is asked which (`--workspace`), and a workspace the account was removed from is not offered. The token is kept where `antumbra join` keeps one, and what the machine already recorded of the workspace stays. A self-hosted server keeps no accounts: there a new token is a new join code.

`antumbra join --leave` leaves a workspace: you are removed from it on its server (anyone may leave, except its last owner), your agent's tools for it are taken out, and what the machine kept of it, its token among it, is deleted. What you shared there stays there. `--to` says which workspace when the machine joined several, and `--leave --local` drops only what the machine keeps, for a server that is gone or refuses.

`antumbra export` writes every memory that is yours on your server to a file: JSON Lines, a first line saying what the file is and then a line a memory, whole, with its kind, confidence, compartment, the machine it was written on, when, and its git anchor. `--out` names the file (`-` for standard output) and `--from <workspace>` exports what is yours in a workspace the machine joined. It holds what the server keeps back from sharing too (anything that looks like a secret), so the file is written readable by its owner alone. Embeddings, relations between memories, documents and experts are not in it.

`antumbra join --connect` gives the machine's agent the workspace's tools as well: each joined workspace is registered as an MCP server of its own (`antumbra-<workspace>`) with every agent setup has connected on the machine: Claude Code and Codex by their own commands, Copilot CLI by one entry in its `mcp-config.json`, and Gemini CLI, goose and OpenCode by a bundle of the workspace's own beside the one setup wrote them (an extension, a plugin, a plugin file), which `--disconnect` removes and nothing else. Each is, a bridge that reads the workspace's token from its file and never presents the machine's own; `--disconnect` takes them back out. `antumbra join --write private|workspace|ask` says where an agent's new memories go: `private`, the default, is your own store, and it is kept, not only said: the agent's connection to the workspace refuses a call that would write something new there (`store_memory`, `record_behavior`, `ingest_document`, `publish_relic`) and tells the agent why, while reading and judging go on. The setting is read for each call, so changing it reaches a session that is already open; `workspace` sends what is learned about the team's work to the workspace and keeps what is yours in your own store, and does nothing until the agent has the workspace's tools; `ask` keeps to your own store and has the agent ask when something looks like the team's. `--to` names the workspace when the machine joined several. The session-start hook tells the agent which workspaces the machine is in, whether it has each one's tools and by what name, and which of these rules holds. It also opens the session with a few memories from each joined workspace (at most three workspaces, five memories each), asked with the workspace's own token, placed after your own and marked `[workspace <name>, not your own store]`; `ANTUMBRA_RECALL_WORKSPACES=0` turns that off, as it does for the prompt hook. Like `setup hosted`, it refuses a plain `http://` address that is not this machine unless `--allow-http` says the network is trusted.

The two routes that take no token are counted together, 60 a minute. A server that verifies RS256 holds no key to sign with and takes no requests.

`--members-only` (`ANTUMBRA_MEMBERS_ONLY=true`) makes a server serve only members: a signed token for a pair a workspace does not hold is refused where it was provisioned on its first call. Turn it on once the workspaces a server holds have their owners.

### Sharing with a whole workspace

A compartment is its owner's alone until it is shared. `share_compartment` shares one with a named member. Sharing with everyone in a workspace takes three consents, and the store's own read rule asks for all three:

1. The workspace's **owner opens the pool** (the Knowledge page, or `POST /pool/open {"on": true}`).
2. A member **offers** a compartment of theirs (the Knowledge page, `antumbra bring --offer`, or `POST /pool/offer {"compartment": "<id>"}`). Offering is joining the pool.
3. An **admin or an owner accepts** the offer (`POST /pool/rule {"key": "...", "accept": true}`), having been shown what is in it.

From then every member of the workspace reads that compartment through their own token, by recall as by listing. The member who offered it can take it back, or stop sharing altogether, and it is theirs alone again at once; declining an accepted offer, or closing the pool, does the same from the other side. `GET /pool` is any member's to read. None of these is an MCP tool.
