#!/usr/bin/env bash
# Antumbra capture hook (macOS / Linux). POSIX mirror of antumbra-capture.ps1.
# Wire to BOTH Stop and PreCompact. Nudges the agent to deposit non-obvious
# observations into Antumbra via the `store_memory` MCP tool before the turn ends
# / context is compacted. A sentinel file makes it fire once per turn (no loop).
# Emits a hook decision only -- it makes NO call to Antumbra, so it works today.
#
# Deps: jq (to read session id / event from the hook JSON on stdin).
set -u
command -v jq >/dev/null 2>&1 || exit 0

input=$(cat)
session_id=$(printf '%s' "$input" | jq -r '.session_id // "default"')
event=$(printf '%s' "$input" | jq -r '.hook_event_name // "Stop"')
workspace="${ANTUMBRA_WORKSPACE_ID:-<workspace>}"

SENTINEL="${TMPDIR:-/tmp}/antumbra-capture-${event}-${session_id}"
if [ -f "$SENTINEL" ]; then rm -f "$SENTINEL"; exit 0; fi
touch "$SENTINEL"

reason="Before stopping, deposit any non-obvious observations from this phase into Antumbra so they are not lost. Call the store_memory MCP tool (workspace \"${workspace}\") and pick the network: world (facts), bank (experiences/incidents), opinion (judgments/preferences). Save: project context, conventions, bug/incident history, user feedback/corrections, decisions, deadlines. Skip ephemeral chatter. Where a result was verified (a test passed, a command worked), note it -- those recurrent, verified traces are what metabolizes into a permanent local expert. If nothing is worth saving this turn, just stop again -- the next stop proceeds automatically."

jq -nc --arg reason "$reason" '{decision:"block", reason:$reason}'
