# Agent hooks: wire your coding agent into Antumbra

These are template lifecycle hooks that turn Antumbra into your agent's persistent,
self-improving brain. They are written for **Claude Code's `settings.json` hook
shape**; the same three touchpoints exist (under different event names) in Cursor,
Gemini CLI, OpenCode, Codex, and Copilot, so adapt the event keys, keep the bodies.

| Script | Hook event | What it does |
|---|---|---|
| `antumbra-session-start` | `SessionStart` | **Bootstrap**: pull the agent's standing conventions + relevant memory from Antumbra and inject them as opening context. |
| `antumbra-capture` | `Stop`, `PreCompact` | **Capture**: nudge the agent to write non-obvious observations back via `store_memory` before the turn ends or context is compacted (sentinel = fire once). |
| `strip-attribution` | `PreToolUse` (git/gh) | **Override**: deny commits/PRs that embed model-vendor attribution, so work is attributed to you. |

**Both platforms are provided.** Each hook ships as a `.ps1` (Windows / PowerShell)
and a `.sh` (macOS / Linux, POSIX `bash`) sibling with identical behavior; use the
one for your OS. The `.sh` scripts need **`jq`** and **`curl`** (preinstalled on most
macOS/Linux dev machines; `brew install jq` / `apt install jq` otherwise). The
bodies are ~20 lines: read the hook JSON on stdin, optionally call the MCP surface,
emit the hook's JSON response.

## Connect (four env vars)

Hooks talk to a running Antumbra **MCP HTTP surface**, either a local one you
start for yourself (`antumbra-mcp --http 127.0.0.1:8081 --url surrealkv://./data.skv`)
or your hosted tenant. Either way:

```sh
ANTUMBRA_URL=http://127.0.0.1:8081     # the antumbra-mcp engine
ANTUMBRA_WORKSPACE_ID=<workspace>      # your tenant/workspace scope
ANTUMBRA_TOKEN=<bearer-jwt>            # Authorization: Bearer <token>
ANTUMBRA_HOST_ID=<this-device>         # provenance stamped on what it writes
```

The networked surface authenticates each call with a JWT whose `(tenant, user)`
claims become the engine's `$auth`. On the offline / self-hosted
tier, mint the long-lived `ANTUMBRA_TOKEN` for a hook with the engine itself:

```sh
antumbra-mcp --mint-token --tenant <workspace> --user <user> \
             --jwt-secret <secret> --token-ttl-days 365
```

It prints a bearer JWT signed with the same HS256 secret the server verifies with
(an RS256 deployment mints via its own auth service's private key instead). The
token *is* the identity (it grants exactly `(tenant, user)`) and carries a finite
`exp`, so it is long-lived, never an eternal standing key. For a purely **offline**,
single identity you can instead run the **stdio** server and have your agent connect
directly; then the capture/bootstrap tools are called in-band and the
`SessionStart` script is optional.

## Wire it (settings.json excerpt)

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

The example uses the Windows form (`pwsh … .ps1`). **On macOS/Linux**, replace each
`pwsh -NonInteractive -File ./scripts/hooks/<name>.ps1` with
`bash ./scripts/hooks/<name>.sh`; same hooks, same behavior. (`pwsh` also runs on
macOS/Linux if you install PowerShell, so the `.ps1` form works cross-platform too;
the `.sh` siblings are the native, dependency-light option.)

Disabling the agent's built-in file memory (`autoMemoryEnabled: false` + the deny
rule) makes Antumbra the **single source of truth**: one store, one ACL, one thing
to back up.

> **What works today.** The **capture** and **strip-attribution** hooks emit hook
> *decisions* only (they make no call to Antumbra), so they work against any agent.
> The long-lived **hook token** is mintable with `--mint-token` (above), and the
> **bootstrap** hook's transport is now live: `POST /mcp/call {tool, arguments}`
> returns the tool's JSON result under your bearer token, no initialize→tools/call
> handshake, so the templates below are drop-in. (The remaining P-1 item is the
> per-workspace embedder config, P-1c, which the hooks don't need.)
