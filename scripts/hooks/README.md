# Agent hooks — wire your coding agent into Antumbra

These are template lifecycle hooks that turn Antumbra into your agent's persistent,
self-improving brain. They are written for **Claude Code's `settings.json` hook
shape**; the same three touchpoints exist (under different event names) in Cursor,
Gemini CLI, OpenCode, Codex, and Copilot — adapt the event keys, keep the bodies.

| Script | Hook event | What it does |
|---|---|---|
| `antumbra-session-start` | `SessionStart` | **Bootstrap**: pull the agent's standing conventions + relevant memory from Antumbra and inject them as opening context. |
| `antumbra-capture` | `Stop`, `PreCompact` | **Capture**: nudge the agent to write non-obvious observations back via `store_memory` before the turn ends or context is compacted (sentinel = fire once). |
| `strip-attribution` | `PreToolUse` (git/gh) | **Override**: deny commits/PRs that embed model-vendor attribution, so work is attributed to you. |

**Both platforms are provided.** Each hook ships as a `.ps1` (Windows / PowerShell)
and a `.sh` (macOS / Linux, POSIX `bash`) sibling with identical behavior — use the
one for your OS. The `.sh` scripts need **`jq`** and **`curl`** (preinstalled on most
macOS/Linux dev machines; `brew install jq` / `apt install jq` otherwise). The
bodies are ~20 lines: read the hook JSON on stdin, optionally call the MCP surface,
emit the hook's JSON response.

## Connect (four env vars)

Hooks talk to a running Antumbra **MCP HTTP surface** — either a local one you
start for yourself (`antumbra-mcp --http 127.0.0.1:8081 --url surrealkv://./data.skv`)
or your hosted tenant. Either way:

```
ANTUMBRA_URL=http://127.0.0.1:8081     # the antumbra-mcp engine
ANTUMBRA_WORKSPACE_ID=<workspace>      # your tenant/workspace scope
ANTUMBRA_TOKEN=<bearer-jwt>            # Authorization: Bearer <token>
ANTUMBRA_HOST_ID=<this-device>         # provenance stamped on what it writes
```

The networked surface authenticates each call with a JWT whose `(tenant, user)`
claims become the engine's `$auth` (ADR-0013/0015). For a purely **offline**, single
identity you can also run the **stdio** server and have your agent connect to it
directly — then the capture/bootstrap tools are called by the agent in-band and the
`SessionStart` script is optional.

## Wire it (settings.json excerpt)

```jsonc
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
`bash ./scripts/hooks/<name>.sh` — same hooks, same behavior. (`pwsh` also runs on
macOS/Linux if you install PowerShell, so the `.ps1` form works cross-platform too;
the `.sh` siblings are the native, dependency-light option.)

Disabling the agent's built-in file memory (`autoMemoryEnabled: false` + the deny
rule) makes Antumbra the **single source of truth** — one store, one ACL, one thing
to back up.

> **What works today vs P-1.** The **capture** and **strip-attribution** hooks emit
> hook *decisions* only — they make no call to Antumbra — so they work now against
> any agent. The **bootstrap** hook fetches context, and Antumbra's networked surface
> today is JSON-RPC at `/mcp` (no `POST /mcp/call {tool, arguments}` REST shape) and
> mints only a per-request JWT. So a long-lived **hook token** and a **REST
> `/mcp/call` convenience endpoint** are tracked as roadmap **P-1** (see
> [`docs/product.md`](../../docs/product.md)). Until P-1 lands, do the bootstrap by
> having the agent run a `recall_memories` call at the top of its first turn (no
> SessionStart script needed), or point the script at a local convenience shim. The
> templates below are written to the target `/mcp/call` shape so they are drop-in
> once P-1 ships.
