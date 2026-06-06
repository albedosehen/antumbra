#!/usr/bin/env bash
# Antumbra PreToolUse hook (macOS / Linux). POSIX mirror of strip-attribution.ps1.
# Denies git/gh commands that embed model-vendor attribution, so the work in your
# history is attributed to you, not the agent. Reads the hook JSON from stdin and
# emits a deny decision only on a match. Emits a hook decision only -- no call to
# Antumbra, so it works today.
#
# Deps: jq.
set -euo pipefail
command -v jq >/dev/null 2>&1 || exit 0

cmd=$(jq -r '.tool_input.command // ""')

# Vendor-attribution phrases to keep out of commits/PRs. Extend per your policy.
if printf '%s' "$cmd" | grep -qiE 'Co-Authored-By:.*(Claude|anthropic\.com|GPT|OpenAI|Copilot)|🤖 Generated|Generated with \[|\[Claude Code\]|noreply@anthropic\.com'; then
  cat <<'JSON'
{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"Model-vendor attribution is not allowed in git/gh commands per policy. Remove the Co-Authored-By trailer, any \"Generated with\" line, and the 🤖 emoji before retrying."}}
JSON
fi
