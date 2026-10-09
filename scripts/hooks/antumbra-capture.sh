#!/usr/bin/env bash
# Antumbra capture hook (macOS / Linux). POSIX mirror of antumbra-capture.ps1.
# Wire to BOTH Stop and PreCompact. Nudges the agent to deposit non-obvious
# observations into Antumbra via the `store_memory` MCP tool before the turn ends
# / context is compacted. A sentinel file makes it fire once per turn (no loop).
# Emits a hook decision only -- it makes NO call to Antumbra.
#
# Git-aware: inside a repository the nudge carries the exact provenance to stamp
# on memories about this code (repo slug, commit, branch), so a later session
# can tell whether each one still applies.
#
# Deps: jq (to read session id / event from the hook JSON on stdin); git optional.
set -u
command -v jq >/dev/null 2>&1 || exit 0

input=$(cat)
session_id=$(printf '%s' "$input" | jq -r '.session_id // "default"')
event=$(printf '%s' "$input" | jq -r '.hook_event_name // "Stop"')
workspace="${ANTUMBRA_WORKSPACE_ID:-<workspace>}"

SENTINEL="${TMPDIR:-/tmp}/antumbra-capture-${event}-${session_id}"
if [ -f "$SENTINEL" ]; then rm -f "$SENTINEL"; exit 0; fi
touch "$SENTINEL"

# Where the session is, in git terms (fail-open: no repository, no anchor).
git_note=""
if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  commit=$(git rev-parse HEAD 2>/dev/null || true)
  branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)
  [ "$branch" = "HEAD" ] && branch=""
  remote=$(git remote get-url origin 2>/dev/null || true)
  repo=$(printf '%s' "$remote" \
    | sed -E 's#^[a-z+]+://##; s#^[^/@]*@##; s#^([^/:]+):#\1/#; s#\.git/?$##; s#/+$##' \
    | tr '[:upper:]' '[:lower:]')
  case "$repo" in *\\*|/*) repo="" ;; */*) ;; *) repo="" ;; esac
  if [ -n "$commit" ] && [ -n "$repo" ]; then
    git_note=" For any memory about this code, pass provenance {repo: \"${repo}\", commit: \"${commit}\", branch: \"${branch}\"} (add path when it is about one file): that anchor is how a later session tells whether the memory still applies."
  fi
fi

reason="Before stopping, deposit any non-obvious observations from this phase into Antumbra so they are not lost. Call the store_memory MCP tool (workspace \"${workspace}\") and pick the network: world (facts), bank (experiences/incidents), opinion (judgments/preferences). Save: project context, conventions, bug/incident history, user feedback/corrections, decisions, deadlines. Skip ephemeral chatter. Where a result was verified (a test passed, a command worked), note it. When the user states or corrects how an agent should act (a convention, a preference, a correction), also call record_behavior with the rule, a check and examples: behaviors are what the user's own expert learns.${git_note} If nothing is worth saving this turn, just stop again -- the next stop proceeds automatically."

jq -nc --arg reason "$reason" '{decision:"block", reason:$reason}'
