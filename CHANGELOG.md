# Changelog

## 0.2.3 - 2026-10-29

Relics you can take with you, and a console that keeps reading. A relic downloads as HTML, Markdown or PDF, and `antumbra tui` keeps reading a networked server past the hour its session lasts.

### Relics

- **A relic downloads as HTML, Markdown or PDF** ([#244](https://github.com/albedosehen/antumbra/pull/244)). The viewer gains a Download control, and `GET /relic/<slug>/export?format=html|md|pdf` hands the current version over as a file to whoever may view it: the token's identity, or anyone for a public relic, with a private relic and a missing one the same 404. The server reads the bytes from copal and answers with an attachment under a `sandbox` policy, so the dashboard's origin still never runs a page. Markdown keeps headings, emphasis, links, lists, quotes, code with its language and tables, numeric columns aligned right; scripts, styles, form controls and drawings are left out. PDF is printed by a Gotenberg service named by `--pdf-renderer` (`ANTUMBRA_PDF_RENDERER`), with the page's backgrounds and `@page` size; without one, the viewer says to print from the browser instead.
- **The Compose `pdf` profile runs that renderer** (`gotenberg/gotenberg:8.37.0`) with no host port, refusing private addresses, so a page cannot reach the network while it prints ([#244](https://github.com/albedosehen/antumbra/pull/244)).

### The console

- **The console reads a networked server after an hour** ([#248](https://github.com/albedosehen/antumbra/pull/248)). Where the server signs each identity in to its own database (a Compose or Kubernetes deployment), `/console/snapshot` read through that identity's connection without refreshing its session first, as every tool call does, so a read more than an hour after the last refresh failed with "the session has expired". The snapshot now refreshes it first.

## 0.2.2 - 2026-10-28

The console, in the one command and through the server. `antumbra tui` reads the Antumbra server with the token the hooks use, so a workstation needs no database password, and an operator's token lets it act on the substrate ([#245](https://github.com/albedosehen/antumbra/pull/245)).

### The console

- **`antumbra tui` is the operator console, inside the one `antumbra` command.** `antumbra --tui` is the same. The separate `antumbra-tui` binary and its installers are gone: installing the CLI installs the console.
- **The console reads the Antumbra server**, with the address and token the hooks use (`ANTUMBRA_URL` and `ANTUMBRA_TOKEN`, else `~/.antumbra/token.txt`), instead of opening the database with its root password. It reads every five seconds, and the server answers an unchanged view with no body. `ask` routes through the server, which embeds with its own model. `--url`, `--db-user`, `--db-pass` and `--embed-url` are gone; `--demo` still shows a seeded population with no server.
- **An operator's token.** `antumbra-mcp --mint-token --operator` mints a token with an `operator` claim. Any token's console shows its own workspace: its memories and their links, and the experts it may route to. An operator's also shows the substrate, which belongs to no workspace (shadows, boundaries, the loop, evaluations, the router), and may act on it: prune or graduate a shadow, freeze or thaw an expert, delete a boundary, halt the loop. `antumbra setup local` mints whoever runs the stack an operator's token, replacing a plain one it kept. A token minted before the claim reads as not an operator's, and the hosted mint never writes it.
- **The server's console routes**, `GET /console/snapshot` and `POST /console/action`, take the same bearer token as `/mcp`. They are not MCP tools, so no agent is offered them.

## 0.2.1 - 2026-10-27

Setup without a clone, and behaviors an agent records on the first try. `antumbra setup local` runs the server image published for the CLI's version, so installing the CLI is all a local stack needs, and `record_behavior` tells an agent everything a call lacks in one answer.

### Setup

- **`antumbra setup local` needs no clone** ([#241](https://github.com/albedosehen/antumbra/pull/241)). Outside a clone of the repository, setup writes the compose file the CLI carries to `~/.antumbra/stack/docker`, fills in the `.env` beside it, and pulls the server image published for the CLI's own version, `ghcr.io/albedosehen/antumbra-mcp:<version>`, instead of building it. Running setup again after an upgrade moves the stack to the new version's image, and an image named by hand in that `.env` (`ANTUMBRA_IMAGE`) is kept. A stack an earlier setup ran from a clone carries that clone's `docker/.env` over, so the store's password and the token still work. Inside a clone, setup builds from the source as before, and the GPU build (`--gpu`) still needs one.
- **Release tags publish the server image** for linux/amd64 and linux/arm64, each built on a runner of its own architecture, tagged with the version ([#241](https://github.com/albedosehen/antumbra/pull/241)). Main pushes keep their tags. The compose file's server image is `${ANTUMBRA_IMAGE:-antumbra-mcp:local}`, so a clone that does not set it builds as before.

### Behaviors

- **`record_behavior` says everything a call lacks at once** ([#242](https://github.com/albedosehen/antumbra/pull/242)). A call without `rule`, `must`, `examples` or `violations` failed on the first missing field, so an agent learned the arguments one error per call; it now gets every problem in one `problems` list, and the schema still asks for all four. A violation written as a `{task, answer}` pair, the way an example is, is read as its answer. The capture hooks name the fields and the four-example minimum, where they asked only for "the rule, a check and examples".

## 0.2.0 - 2026-10-26

Relics, and behaviors you can teach. An agent can now keep the pages it writes as relics, found and viewed in the dashboard, and the behavior loop learns a rule from your memories, trains it into a standing expert, and checks that the expert still follows it.

### Relics

- **Relics: pages an agent wrote, kept, found and viewed** ([#236](https://github.com/albedosehen/antumbra/pull/236)). `publish_relic` stores an HTML page in copal (one file per version, at `relics/<slug>/v<N>.html`) and indexes its text, so `recall_documents` finds it. `list_relics` lists or searches the relics you can see, `get_relic` returns one with an address that displays it, and `set_relic_visibility` makes one public or private. The dashboard gains a gallery at `/relics`, with live previews, All / Yours / Shared with you, search and a list view, and a viewer at `/relic/<slug>`. A relic is private to its compartment's audience by default, viewed through a render link minted per view; a public one opens at `/relic/<slug>` for anyone who can reach the server, with no sign-in. Pages are shown only in a sandboxed `<iframe>` from copal's render origin, never by the dashboard's own origin. Needs `--copal-render-origin` (`ANTUMBRA_COPAL_RENDER_ORIGIN`) beside `--copal-addr`, and copal with render links, `/v1/public/{id}` and `COPAL_RENDER_FRAME_ANCESTORS` (Oneiriq/copal#119). A new `relic` table, read through the compartment and written by its owner only, is applied at boot.
- **The Compose `copal` profile runs copal's render listener**, so a Compose host can keep relics: `COPAL_RENDER_ORIGIN`, `COPAL_RENDER_FRAME_ANCESTORS` and `COPAL_RENDER_PUBLISH` in `docker/.env` put it on the LAN (it defaults to this machine only, at `127.0.0.1:8093`); set `ANTUMBRA_COPAL_RENDER_ORIGIN` to the same origin ([#238](https://github.com/albedosehen/antumbra/pull/238)).
- **`/mcp/call` takes a page-sized body.** The REST shim read at most 64 KiB, so a relic or a knowledge document past about 60 KB was refused before it reached the tool; the ceiling is now 4 MiB ([#237](https://github.com/albedosehen/antumbra/pull/237)).

### Behaviors and standing experts (experimental)

- **`behavior_candidates` finds the rules you already stated.** It reads the store in order and offers the memories that may state a behavior (opinions, feedback and convention tags, phrases about how to act), and a behavior records the memories it was drawn from, so a pass converges ([#212](https://github.com/albedosehen/antumbra/pull/212)).
- **The dashboard has a Behaviors section** with Accept and Retire, filtered by status, showing each rule's scope, examples, violations, sources and how its last training went ([#213](https://github.com/albedosehen/antumbra/pull/213)).
- **The training recipe is the node's to set**: `--behave-rank`, `--behave-rounds` and `--behave-lr` (defaults rank 16, 3 rounds, 1.5e-4), refused at start when they could train nothing ([#215](https://github.com/albedosehen/antumbra/pull/215)).
- **One example in three is held out**, so a behavior recorded with ten examples is judged on three and one miss still reads as learned ([#216](https://github.com/albedosehen/antumbra/pull/216)). The docs say how many examples a prose rule needs ([#217](https://github.com/albedosehen/antumbra/pull/217)).
- **A regression is a probability.** A behavior the expert answers worse than the base model refuses the set when that is likely past the regression confidence (`--behave-regression-confidence`, default 0.9), instead of by a fixed distance ([#219](https://github.com/albedosehen/antumbra/pull/219)).
- **A served expert is audited** on the examples its behaviors gained since it was trained, and the audit follows a behavior recorded again with more examples ([#220](https://github.com/albedosehen/antumbra/pull/220), [#221](https://github.com/albedosehen/antumbra/pull/221)). With `--behave-tune-confidence`, the loop tunes the regression confidence per workspace from those audits ([#222](https://github.com/albedosehen/antumbra/pull/222)).
- **A behavior not learned says what the expert answered instead**: its first failed held-out task and the answer, in `list_behaviors` and on the dashboard ([#232](https://github.com/albedosehen/antumbra/pull/232)).
- **One judge for a check.** The trainer scores answers with the same check `record_behavior` uses, so a rule recorded in one regex dialect is never judged in another ([#233](https://github.com/albedosehen/antumbra/pull/233)), and a behavior whose examples hold code fences reads back whole ([#214](https://github.com/albedosehen/antumbra/pull/214)).
- **`answer` can serve through your standing experts alone** (`standing_only`) ([#234](https://github.com/albedosehen/antumbra/pull/234)), and a behavior training is seeded from what it teaches, so the same rules train the same way in any process (`--generation-seed`) ([#235](https://github.com/albedosehen/antumbra/pull/235)).
- **A standing expert is one workspace's**, no longer shared by workspaces whose users have the same id ([#223](https://github.com/albedosehen/antumbra/pull/223)).

### Recall and identity

- **A session that can read little of a workspace is searched exactly**: the chunk leg finds the nearest pieces it may read in milliseconds, where it used to walk the whole index for seconds ([#226](https://github.com/albedosehen/antumbra/pull/226), [#227](https://github.com/albedosehen/antumbra/pull/227)).
- **A user id names one user in one workspace**: hosted signup gives each new workspace's first user an id of its own, so private experts can never be listed or routed across workspaces ([#228](https://github.com/albedosehen/antumbra/pull/228)).

### Server

- **A restart answers at once.** The chunk index's first pass no longer holds the auth lock, and a networked server listens only once its embedder answers, so the first calls after a start no longer wait seconds ([#224](https://github.com/albedosehen/antumbra/pull/224), [#225](https://github.com/albedosehen/antumbra/pull/225)).
- **A first call's trace says what building its session waited on**: the auth lock, provisioning, or the serving connection ([#229](https://github.com/albedosehen/antumbra/pull/229)).

### Docker

- **SurrealDB health checks run over the websocket.** SurrealDB 3.2.4 answers the `http://` form "Method not allowed" while serving normally, which left both databases unhealthy and kept a recreated server from starting; `kusko-deploy.sh` now fails when a service is not running instead of reporting OK ([#239](https://github.com/albedosehen/antumbra/pull/239)).
- **Images built from `main` are also tagged `main-<timestamp>-<short sha>`**, so image automation can order them ([#230](https://github.com/albedosehen/antumbra/pull/230)).

### Security

- **Dependencies moved past the October 2026 advisories**: quinn-proto 0.11.19 and ammonia ([#231](https://github.com/albedosehen/antumbra/pull/231)); rustls 0.23.46, rustls-webpki 0.103.16, webpki-roots and webpki-root-certs 1.0.11, ammonia 4.2.4 and zip 7.4.0 (with #236). The two advisories no release clears, crypto-bigint through SurrealDB's JWT stack and zip through candle, are documented as unreachable in `.cargo/audit.toml`.

## 0.1.1 - 2026-10-09

Fixes. The main one lets `antumbra setup local` finish on Docker Desktop, including a WSL distro that uses Docker Desktop's WSL integration.

### Setup

- **The ollama check passes on any Docker setup the server can reach ollama from.** On Linux, setup probed the Docker bridge gateway from the host. That is right for a native Docker Engine, but under Docker Desktop the bridge lives in Desktop's VM, so the check always failed from WSL and no flag skipped it. The bridge probe still runs first; when it does not answer, a throwaway container asks for ollama at `host.docker.internal`, the address the server uses. When ollama really is out of reach, the message says what to change for native Linux, Docker Desktop under WSL, or Docker Desktop elsewhere ([#209](https://github.com/albedosehen/antumbra/pull/209)).

### Hooks

- **Session start stays within its time budget** (`ANTUMBRA_SESSION_BUDGET_SEC`, default 8 seconds, below the hook's timeout). On Windows PowerShell it no longer garbles non-ASCII text in recalled memories, and no longer reports a single recalled memory as "starting cold" ([#201](https://github.com/albedosehen/antumbra/pull/201)).
- **The attribution check is one bash process on every platform**, using only bash builtins. On a loaded Windows machine it no longer outruns its timeout and lets commands through unchecked, and it checks the PowerShell tool's git and gh commands as well as the Bash tool's ([#203](https://github.com/albedosehen/antumbra/pull/203)).

### Server

- **Tools that take a memory, behavior or handoff back accept `id`**, the field name their results use, as well as `memory_id`, `behaviour_id` and `handoff_id` ([#201](https://github.com/albedosehen/antumbra/pull/201)).
- **Listing a compartment reads an index** instead of the whole workspace: 739 ms to 2.5 ms for 6 rows beside 7,000. A running store gains the index at its next start ([#201](https://github.com/albedosehen/antumbra/pull/201)).
- **An empty endpoint setting counts as unset.** A Compose stack without a reranker no longer builds one at the address `""` and pays a failing rerank call on every recall ([#199](https://github.com/albedosehen/antumbra/pull/199)).

### Behaviors and experts (experimental)

- Behaviors are recorded with at least four examples, so training can hold one in four out and admit a standing expert by it. A scope's standing expert is admitted for the behaviors it learned, instead of being refused when any one is missed. A refused set stays refused across restarts and says why ([#204](https://github.com/albedosehen/antumbra/pull/204), [#205](https://github.com/albedosehen/antumbra/pull/205), [#206](https://github.com/albedosehen/antumbra/pull/206)).

### Docker

- Two opt-in Compose profiles: `copal`, which keeps each ingested document's original, and `telemetry`, an OpenTelemetry Collector that reports the host, containers, SurrealDB, GPU and reranker to the same endpoint as the server's traces ([#199](https://github.com/albedosehen/antumbra/pull/199), [#202](https://github.com/albedosehen/antumbra/pull/202)).
- The control server's `contract` feature takes `oneiriq-kayak` from crates.io, so building it needs no access to a private repository ([#200](https://github.com/albedosehen/antumbra/pull/200)).

## 0.1.0 - 2026-10-07

The first public release. Antumbra plugs into the coding agent you already use and gives it a memory that carries across sessions and machines, along with where each memory came from. Claude Code is the agent it is tested with; any MCP client can use its tools over stdio or HTTP.

### Setting it up

Install the CLI with the installer for your OS (below), then, with Docker and ollama running, set it up from a clone of this repository, which the local stack is built from:

```sh
git clone https://github.com/albedosehen/antumbra.git
cd antumbra
antumbra setup local
```

Connecting to a hosted workspace needs no clone: `antumbra setup hosted <url> --token-file <file>`. Then `antumbra setup check` confirms every piece. [docs/getting-started.md](https://github.com/albedosehen/antumbra/blob/main/docs/getting-started.md) walks through it, and [docs/agent-setup.md](https://github.com/albedosehen/antumbra/blob/main/docs/agent-setup.md) lets your agent do it for you.

### What's in it

- **One-command setup.** `antumbra setup local` checks Docker and ollama (and, on macOS and Linux, `jq` and `curl`), pulls the embedding model, generates the stack's secrets, starts SurrealDB and the server, mints a token, proves recall works with it, and connects Claude Code: the hooks, the MCP server and the settings, backed up and only added to. Running it again keeps what is already in place.
- **Hooks for Claude Code.** Recall at the start of each session and with each prompt, and a nudge to write back what was verified when a session stops.
- **Memory with provenance.** A memory about code carries the repository, commit and branch it was learned at. Recall scopes to where you are and marks each memory as live, not on HEAD, or orphaned, and a merged branch's memories move to its merge commit.
- **Recall** that fuses dense and full-text search by rank, with an optional cross-encoder rerank whose calibrated floor lets it answer "nothing relevant". Its legs rank keys instead of reading rows, and memory's read rule reads the owner's compartment record directly, one key read per row instead of a subquery ([#196](https://github.com/albedosehen/antumbra/pull/196), [#198](https://github.com/albedosehen/antumbra/pull/198)).
- **Handoffs** between your machines, a **dependency graph** with `blast_radius`, **compartments** you share on purpose, **documents** recalled separately from memory, and **behaviors**: rules you record with a check that has to tell examples from violations.
- **The operator console** (`antumbra-tui`) and a read-only web dashboard at `/dashboard`.
- **Experts (experimental).** Training and serving small LoRA experts needs an NVIDIA GPU and a source build with the `models` feature. The prebuilt binaries are the light, no-GPU build.

### Known limits

- Setup does not start the reranker. Recall works without it but cannot say "nothing relevant", and on a CPU the reranker is too slow to sit in front of every prompt.
- The server advertises all 33 MCP tools. The 17-tool `agent` profile is a server setting (`ANTUMBRA_TOOLS=agent`) that setup does not apply yet.
- Claude Code is the only agent tested so far.

### License

Source-available under the [Elastic License 2.0](https://github.com/albedosehen/antumbra/blob/main/LICENSE). You may use, change and run it for yourself or inside your own organization; you may not offer it to others as a hosted or managed service.
