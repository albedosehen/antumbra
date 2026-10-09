# Changelog

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
