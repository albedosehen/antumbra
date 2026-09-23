# Agent hooks: wire your coding agent into Antumbra

These are template lifecycle hooks that turn Antumbra into your agent's persistent, self-improving brain. They are written for **Claude Code's `settings.json` hook shape**; the same three touchpoints exist (under different event names) in Cursor, Gemini CLI, OpenCode, Codex, and Copilot, so adapt the event keys, keep the bodies.

| Script                   | Hook event            | What it does                                                                                                                                                |
| ------------------------ | --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `antumbra-session-start` | `SessionStart`        | **Bootstrap**: pull the agent's standing conventions + relevant memory from Antumbra and inject them as opening context.                                    |
| `antumbra-capture`       | `Stop`, `PreCompact`  | **Capture**: nudge the agent to write non-obvious observations back via `store_memory` before the turn ends or context is compacted (sentinel = fire once). |
| `strip-attribution`      | `PreToolUse` (git/gh) | **Override**: deny commits/PRs that embed model-vendor attribution, so work is attributed to you.                                                           |

**Both platforms are provided.** Each hook ships as a `.ps1` (Windows / PowerShell) and a `.sh` (macOS / Linux, POSIX `bash`) sibling with identical behavior; use the one for your OS. The `.sh` scripts need **`jq`** and **`curl`** (preinstalled on most macOS/Linux dev machines; `brew install jq` / `apt install jq` otherwise). The bodies are ~20 lines: read the hook JSON on stdin, optionally call the MCP surface, emit the hook's JSON response.

## Connect (four env vars)

Hooks talk to a running Antumbra **MCP HTTP surface**, either a local one you start for yourself (`antumbra-mcp --http 127.0.0.1:8081 --url surrealkv://./data.skv`) or your hosted tenant. Either way:

```sh
ANTUMBRA_URL=http://127.0.0.1:8081     # the antumbra-mcp engine
ANTUMBRA_WORKSPACE_ID=<workspace>      # your tenant/workspace scope
ANTUMBRA_TOKEN=<bearer-jwt>            # Authorization: Bearer <token>
ANTUMBRA_HOST_ID=<this-device>         # provenance stamped on what it writes
ANTUMBRA_PENALIZE_ORPHANS=0            # 1: bootstrap also penalizes memories whose branch is gone
ANTUMBRA_TOOLS=agent                   # on the SERVER: advertise only the eight tools a coding agent needs
ANTUMBRA_BIN=antumbra                  # the CLI the bootstrap asks for the sovereign-mode block and runs to report merges (optional)
ANTUMBRA_REANCHOR=1                    # 0: the bootstrap does not report recent merges
ANTUMBRA_REANCHOR_LOG=~/.antumbra/reanchor.log  # where the last merge report's output goes
```

The networked surface authenticates each call with a JWT whose `(tenant, user)` claims become the engine's `$auth`. On the offline / self-hosted tier, mint the long-lived `ANTUMBRA_TOKEN` for a hook with the engine itself:

```sh
antumbra-mcp --mint-token --tenant <workspace> --user <user> \
             --jwt-secret <secret> --token-ttl-days 365
```

It prints a bearer JWT signed with the same HS256 secret the server verifies with (an RS256 deployment mints via its own auth service's private key instead). The token _is_ the identity (it grants exactly `(tenant, user)`) and carries a finite `exp`, so it is long-lived, never an eternal standing key. For a purely **offline**, single identity you can instead run the **stdio** server and have your agent connect directly; then the capture/bootstrap tools are called in-band and the `SessionStart` script is optional.

## Why disable built-in and additional external memory?

Disabling an agent's built-in file memory (`autoMemoryEnabled: false` + the deny rule) makes Antumbra the **single source of truth**: one store, one ACL, one thing to back up. You should disable any other external memory the agent has access to outside of Antumbra as well to reduce side-effects or unintended poisoning/corruption of context.

Pick the block for your OS. Both wire the same three touchpoints; they differ only in how the command is launched (`bash` + the `.sh` script, or `pwsh` + the `.ps1`).

### macOS / Linux (bash) settings.json (claude code)

```json
{
  "autoMemoryEnabled": false,
  "permissions": { "deny": ["Write(**/.agent/memory/**)", "Edit(**/.agent/memory/**)"] },
  "hooks": {
    "SessionStart": [{ "hooks": [{ "type": "command",
      "command": "bash ./scripts/hooks/antumbra-session-start.sh", "timeout": 10 }]}],
    "Stop": [{ "hooks": [{ "type": "command",
      "command": "bash ./scripts/hooks/antumbra-capture.sh", "timeout": 5 }]}],
    "PreCompact": [{ "hooks": [{ "type": "command",
      "command": "bash ./scripts/hooks/antumbra-capture.sh", "timeout": 5 }]}],
    "PreToolUse": [{ "matcher": "Bash", "hooks": [
      { "type": "command", "command": "bash ./scripts/hooks/strip-attribution.sh", "if": "Bash(git *)" },
      { "type": "command", "command": "bash ./scripts/hooks/strip-attribution.sh", "if": "Bash(gh *)" }
    ]}]
  }
}
```

The `.sh` hooks need `jq` and `curl` (preinstalled on most macOS/Linux dev machines; otherwise `brew install jq` on macOS, `apt install jq` on Debian/Ubuntu).

### Windows (PowerShell) settings.json (claude code)

```json
{
  "autoMemoryEnabled": false,
  "permissions": { "deny": ["Write(**/.agent/memory/**)", "Edit(**/.agent/memory/**)"] },
  "hooks": {
    "SessionStart": [{ "hooks": [{ "type": "command",
      "command": "pwsh -NonInteractive -File ./scripts/hooks/antumbra-session-start.ps1", "timeout": 10 }]}],
    "Stop": [{ "hooks": [{ "type": "command",
      "command": "pwsh -NonInteractive -File ./scripts/hooks/antumbra-capture.ps1", "timeout": 5 }]}],
    "PreCompact": [{ "hooks": [{ "type": "command",
      "command": "pwsh -NonInteractive -File ./scripts/hooks/antumbra-capture.ps1", "timeout": 5 }]}],
    "PreToolUse": [{ "matcher": "Bash", "hooks": [
      { "type": "command", "command": "pwsh -File ./scripts/hooks/strip-attribution.ps1", "if": "Bash(git *)" },
      { "type": "command", "command": "pwsh -File ./scripts/hooks/strip-attribution.ps1", "if": "Bash(gh *)" }
    ]}]
  }
}
```

Note: (`pwsh` also runs on macOS/Linux if you install PowerShell, so the `.ps1` form is cross-platform too; the `.sh` siblings are the native, dependency-light option.)

## Sovereign mode: what the bootstrap says first

Turning Claude Code's telemetry off also turns off its feature flags, and the features gated on them, without saying so (ADR-0021). When `antumbra` is on the path (or `ANTUMBRA_BIN` names it), the bootstrap opens with what `antumbra claude brief` prints: whether the project's `AGENTS.md` reached the agent, which shell is the host's, and what the agent must not offer because it is gone. Outside that state it prints nothing, and so does a missing or failing `antumbra`.

If `AGENTS.md` was not loaded, the block tells the agent to read it before anything else. Measured on a 31,026-character file whose last line sets a rule for every reply: without the block the agent answered in one turn and never saw the rule; with it, the agent read the file, followed the rule, and told the user about `antumbra claude bridge`. The block is a fallback for the session in front of you. The bridge is the fix, because a hook reaches neither subagents nor the session after compaction.

`antumbra claude remember` keeps the same rules as `world` memories in a `claude-code` compartment of your own, for recall. It writes through this surface (`ANTUMBRA_URL`, `ANTUMBRA_TOKEN`), marks them volatile so they never train an expert, and is safe to run again.

## Which skills are used

`/skill-doctor` goes with the flags too. Two hooks count instead, because a skill is used two ways and each hook sees one: the agent calls the `Skill` tool (`PostToolUse`), or you type `/name`, which never touches the tool (`UserPromptExpansion`). Both run the same command, which reads the hook's input itself, so there is no script and nothing differs between shells. As a hook it never fails and never speaks, and `async` keeps a slow surface from delaying a skill:

```json
"PostToolUse": [{ "matcher": "Skill", "hooks": [
  { "type": "command", "command": "antumbra claude skill-used", "async": true }]}],
"UserPromptExpansion": [{ "hooks": [
  { "type": "command", "command": "antumbra claude skill-used", "async": true }]}]
```

Each skill gets one volatile memory in your `claude-code` compartment, reinforced on every use. `antumbra claude skills` lists the skills installed for you and the project with their counts, the never-used first, and calls out one not used in `--days` (30).

## The 10,000-character limit

Claude Code caps a hook's context at 10,000 characters. Past the cap the agent is handed a file path and a 2,000-character preview it is never asked to open, so an oversized bootstrap is a truncated one that says nothing about it. The bootstrap stays under 9,500: the session block and the git line go first, memories follow best first while they fit, and the rest are counted in a closing line so the agent knows to `recall_memories` for them.

`scripts/hooks/tests/session-start.sh` (bash, jq) and `scripts/hooks/tests/session-start.ps1` (pwsh) hold both siblings to this with no server and no `antumbra` installed.

## Git provenance: stale memories are visible, not silently wrong

A memory about code is only as good as its anchor. The hooks keep that anchor as **provenance on the memory** and judge it at recall, where git is, instead of re-extracting symbol tables and pruning them:

- **Capture** (`antumbra-capture`) computes the session's git context (origin slug, HEAD commit, branch) and tells the agent to pass it as `store_memory`'s `provenance` `{repo, commit, branch[, path]}`. It lands as one evidence entry, `git:<repo>@<commit>#<branch>[:<path>]`.
- **Bootstrap** (`antumbra-session-start`) passes the same `repo` and `branch` to `recall_memories`, so the server scopes every hit (`in_scope`, `other_branch`, `other_repo`) and demotes out-of-scope ones below in-scope ones without hiding them. Then, with git in hand, it checks each hit's anchor and tags it:

  | Tag             | Meaning                                                                                              |
  | --------------- | ---------------------------------------------------------------------------------------------------- |
  | `[live]`        | the commit it was learned at is an ancestor of HEAD                                                  |
  | `[not-on-head]` | learned on a commit this HEAD does not contain (an unmerged branch, or history this clone lacks)     |
  | `[orphaned]`    | its branch no longer exists locally or on `origin` (as far as this clone knows; fetch to be current), or the server recorded GitHub deleting it (the App's delete event; `orphaned_at` on the hit) |

  Set `ANTUMBRA_PENALIZE_ORPHANS=1` to have the bootstrap also call `penalize_memory` on orphaned hits, so a memory about a branch that is gone loses standing without anyone noticing it first.

- **Merges** move a branch's memories onto the branch it merged into, so recall from there counts them in scope. The GitHub webhook does this when GitHub can reach the server. When it cannot, the bootstrap does it instead. Inside a repository, with `antumbra` on the path, it starts `antumbra claude reanchor --days 3` in the background. That command asks GitHub, through `gh`, which pull requests merged in the last three days and reports each one to the `record_merge` tool. The report runs detached, holding none of the hook's handles, so the session never waits for it. Reporting a merge again moves nothing, so running it every session is safe. The output of the last run is in `~/.antumbra/reanchor.log`. Set `ANTUMBRA_REANCHOR=0` to turn it off.

Outside a repository, or without `git` on the path, both hooks run exactly as before: no anchor, no tags. Nothing here needs a parser, and nothing is garbage-collected; the anchor travels with the memory. For inventory questions ("what routes does this service expose?") the parser-free companion is `antumbra ingest --title routes -- <the framework's own lister>`, which stores what the command printed as a knowledge document stamped with the same anchor. Add `--copal-addr` and the original is archived to copal first, as the server does.
