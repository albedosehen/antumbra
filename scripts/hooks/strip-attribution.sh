#!/usr/bin/env bash
# Antumbra PreToolUse hook. Denies git/gh commands that embed model-vendor
# attribution, so the work in your history is attributed to you, not the agent.
# Reads the hook JSON from stdin and emits a deny decision only on a match.
# Emits a hook decision only -- no call to Antumbra, so it works today.
#
# The one to wire on every platform, Windows included: Claude Code runs a hook's
# command through Git Bash there, so this costs one bash process per git or gh
# command. strip-attribution.ps1 is for a Windows without Git Bash.
#
# Bash builtins only: no jq, grep or cat. A PowerShell start per git/gh command,
# measured on a busy Windows machine, outran a 5 s hook timeout 27 times in a
# week, and a PreToolUse hook that times out blocks nothing. This one answers in
# about 45 ms there. Needing jq would also have let everything through on a
# machine without it.

# Hook input is one line of JSON on stdin; read ends at EOF with a non-zero status.
IFS= read -r -d '' input || true

# tool_input.command as the JSON spells it (Bash and PowerShell tools alike):
# escapes stay in place, and none of the phrases below contain one.
re_command='"command"[[:space:]]*:[[:space:]]*"(([^"\\]|\\.)*)"'
[[ $input =~ $re_command ]] || exit 0
cmd=${BASH_REMATCH[1]}

# Vendor-attribution phrases to keep out of commits/PRs, case-insensitive. The
# robot emoji arrives raw or JSON-escaped as a surrogate pair. Extend per your policy.
shopt -s nocasematch
re_attribution='co-authored-by:.*(claude|anthropic\.com|gpt|openai|copilot)|🤖 generated|\\ud83e\\udd16 generated|generated with \[|\[claude code\]|noreply@anthropic\.com'
if [[ $cmd =~ $re_attribution ]]; then
  printf '%s\n' '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"Model-vendor attribution is not allowed in git/gh commands per policy. Remove the Co-Authored-By trailer, any \"Generated with\" line, and the 🤖 emoji before retrying."}}'
fi
exit 0
