#!/usr/bin/env bash
# Antumbra SessionStart hook (macOS / Linux). POSIX mirror of
# antumbra-session-start.ps1. Bootstraps a coding-agent session: pulls the
# agent's standing conventions + relevant memory from a running antumbra-mcp
# surface and returns it as `additionalContext`. NEVER blocks session start --
# every call has a timeout and any failure degrades to an empty bootstrap.
#
# Targets a REST convenience endpoint POST {ANTUMBRA_URL}/mcp/call {tool,arguments}
# (roadmap P-1; see ../../docs/product.md). Antumbra's current networked surface is
# JSON-RPC at /mcp, so until P-1 ships either run a local shim or skip this script
# and have the agent call recall_memories at the top of its first turn.
#
# Deps: curl, jq. set -u but NOT set -e (a curl failure must not crash the hook).
set -u
command -v jq >/dev/null 2>&1 || { printf '%s' '{"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":"[Antumbra bootstrap: jq not installed]"}}'; exit 0; }

URL="${ANTUMBRA_URL:-http://127.0.0.1:8081}"
TOKEN="${ANTUMBRA_TOKEN:-}"
HOST_ID="${ANTUMBRA_HOST_ID:-local}"

auth=(-H 'Content-Type: application/json')
[ -n "$TOKEN" ] && auth+=(-H "Authorization: Bearer $TOKEN")

payload='{"tool":"recall_memories","arguments":{"query":"standing conventions, project context, and active tasks for this agent","limit":12}}'
resp=$(curl -sS --max-time 5 -X POST "$URL/mcp/call" "${auth[@]}" -d "$payload" 2>/dev/null || echo '{}')

# The live /mcp/call answers with the tool's value at the TOP level
# ({memories: [...]}); the .result envelope is tolerated for older shims.
mem_text=$(printf '%s' "$resp" | jq -r '
  (.memories // .result.memories // []) as $m
  | if ($m | length) > 0
    then [$m[].content] | join("\n\n---\n\n")
    else "" end' 2>/dev/null || echo '')

context=$(jq -nc --arg mem "$mem_text" --arg host "$HOST_ID" '
  "# Antumbra session bootstrap (host=" + $host + ")\n\n"
  + ($mem | if . == "" then "[Antumbra bootstrap empty / unreachable -- starting cold.]" else . end)
' 2>/dev/null) || context='"[Antumbra hook serialization error]"'

jq -nc --argjson ctx "$context" '{hookSpecificOutput:{hookEventName:"SessionStart",additionalContext:$ctx}}'
