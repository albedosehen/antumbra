#!/usr/bin/env bash
# Antumbra capture hook (macOS / Linux). POSIX mirror of antumbra-capture.ps1.
# Wire to BOTH Stop and PreCompact. Nudges the agent to deposit non-obvious
# observations into Antumbra via the `store_memory` MCP tool before the turn ends
# / context is compacted. A sentinel file makes it fire once per turn (no loop).
#
# Git-aware: inside a repository the nudge carries the exact provenance to stamp
# on memories about this code (repo slug, commit, branch), so a later session
# can tell whether each one still applies.
#
# At the end of a turn it also tells the server how the turn ended
# (`observe_turn`, with the turn's last answer), so the system can judge which
# of the memories it showed this session the answer drew on. Nobody has to
# reinforce or penalize by hand. Brief and fail-open: a server that is down or
# slower than two seconds leaves the turn unjudged and the stop unaffected.
#
# Deps: jq (to read the hook JSON on stdin); curl for the report; git optional.
set -u
command -v jq >/dev/null 2>&1 || exit 0

input=$(cat)
session_id=$(printf '%s' "$input" | jq -r '.session_id // "default"')
event=$(printf '%s' "$input" | jq -r '.hook_event_name // "Stop"')
workspace="${ANTUMBRA_WORKSPACE_ID:-<workspace>}"

# The sentinel's name is built from two fields of the hook's input, so both are
# cut to [A-Za-z0-9_-] before they name a file: a value carrying `../` or a
# space is used as given otherwise. It lives under a directory only this user
# writes (the session runtime directory, else one under the home), not a
# world-writable temp directory where another user could plant the name.
name_of() { printf '%s' "$1" | tr -cd 'A-Za-z0-9_-'; }
safe_session="$(name_of "$session_id")"; [ -n "$safe_session" ] || safe_session=default
safe_event="$(name_of "$event")"; [ -n "$safe_event" ] || safe_event=Stop
run_dir="${XDG_RUNTIME_DIR:-$HOME/.antumbra/run}"
mkdir -p "$run_dir" 2>/dev/null && chmod 700 "$run_dir" 2>/dev/null
SENTINEL="$run_dir/antumbra-capture-${safe_event}-${safe_session}"
if [ -f "$SENTINEL" ]; then rm -f "$SENTINEL"; exit 0; fi
# A stop this hook's own nudge caused goes through, whatever became of the
# sentinel.
if [ "$event" = "Stop" ] && [ "$(printf '%s' "$input" | jq -r '.stop_hook_active // false')" = "true" ]; then
  exit 0
fi

if [ "$event" = "Stop" ] && command -v curl >/dev/null 2>&1; then
  url="${ANTUMBRA_URL:-http://127.0.0.1:8081}"
  token="${ANTUMBRA_TOKEN:-}"
  if [ -z "$token" ]; then
    token_file="${ANTUMBRA_TOKEN_FILE:-$HOME/.antumbra/token.txt}"
    [ -r "$token_file" ] && token="$(tr -d '\r\n' < "$token_file" 2>/dev/null || true)"
  fi
  body=$(printf '%s' "$input" | jq -c --arg s "$session_id" '
    (.last_assistant_message // "") as $text
    | if $text == "" then empty else
        { tool: "observe_turn",
          arguments: ({ session: $s, text: $text[0:4000] }
                      + (if .prompt_id then { turn: .prompt_id } else {} end)) }
      end' 2>/dev/null || true)
  # The machine, named as the MCP client names it (antumbra-mcp-headers), so
  # what this machine's agent recalled for itself is judged against this turn.
  host_id="$(printf '%s' "${ANTUMBRA_HOST_ID:-}" | tr -d '\r\n' | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//')"
  if [ -n "$token" ] && [ -n "$body" ]; then
    named=()
    [ -n "$host_id" ] && named=(-H "X-Antumbra-Host: $host_id")
    # The bearer reaches curl as a config line on its stdin, never as an
    # argument: an argument sits in the process list, readable by every other
    # user of the machine, for as long as the request lasts.
    printf 'header = "Authorization: Bearer %s"\n' "$token" | curl -s -o /dev/null -X POST "$url/mcp/call" \
      -H 'Content-Type: application/json' \
      --config - \
      "${named[@]}" \
      --max-time 2 --data-binary "$body" >/dev/null 2>&1 || true
  fi
fi

# Whether this stop is nudged. The end of a turn is not when the agent has
# saved since this hook last looked (the transcript holds a store_memory or
# record_behavior call past the mark kept from then), or when the session was
# nudged, or saved, within the last ANTUMBRA_CAPTURE_EVERY_SECS seconds (600
# unless set; 0 nudges every turn). A long job stops many times while it waits
# on something, and each nudge costs it a turn and says nothing new. A
# compaction is always nudged: what is not saved then is about to be cut.
number() { case "$1" in ''|*[!0-9]*) printf '%s' "$2" ;; *) printf '%s' "$1" ;; esac; }
if [ "$event" = "Stop" ]; then
  every="$(number "${ANTUMBRA_CAPTURE_EVERY_SECS:-}" 600)"
  state="$run_dir/antumbra-capture-state-${safe_session}"
  now="$(date +%s)"
  last=0; mark=0
  if [ -r "$state" ]; then read -r last mark < "$state" || true; fi
  last="$(number "${last:-}" 0)"; mark="$(number "${mark:-}" 0)"
  size=0; saved=0
  transcript="$(printf '%s' "$input" | jq -r '.transcript_path // ""')"
  if [ -n "$transcript" ] && [ -r "$transcript" ]; then
    size="$(number "$(wc -c < "$transcript" | tr -d ' ')" 0)"
    # Another transcript, or this one rewritten: read it from the start.
    [ "$mark" -gt "$size" ] && mark=0
    # No further back than four megabytes, however long the session.
    [ $((size - mark)) -gt 4194304 ] && mark=$((size - 4194304))
    # A tool call's own name, not one quoted inside a result.
    if tail -c "+$((mark + 1))" "$transcript" 2>/dev/null \
      | grep -Eq '"name"[[:space:]]*:[[:space:]]*"[^"]*(store_memory|record_behavior)"'; then
      saved=1
    fi
  fi
  due=1
  if [ "$saved" = 1 ]; then
    due=0
  elif [ "$every" -gt 0 ] && [ $((now - last)) -lt "$every" ]; then
    due=0
  fi
  stamp="$last"
  if [ "$due" = 1 ] || [ "$saved" = 1 ]; then stamp="$now"; fi
  printf '%s %s\n' "$stamp" "$size" > "$state" 2>/dev/null || true
  [ "$due" = 1 ] || exit 0
fi
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
    git_note=" For any memory about this code, pass provenance {repo: \"${repo}\", commit: \"${commit}\", branch: \"${branch}\"} (add path when it is about one file): that anchor is how a later session tells whether the memory still applies. That is the repository this session started in; for work done in another, pass that one's own."
  fi
fi

reason="Before stopping, deposit any non-obvious observations from this phase into Antumbra so they are not lost. Call the store_memory MCP tool (workspace \"${workspace}\") and pick the network: world (facts), bank (experiences/incidents), opinion (judgments/preferences). Save: project context, conventions, bug/incident history, user feedback/corrections, decisions, deadlines. Skip ephemeral chatter. Where a result was verified (a test passed, a command worked), note it. When the user states or corrects how an agent should act (a convention, a preference, a correction), also call record_behavior with rule (one sentence), must (regular expressions every answer that follows it matches), examples (at least four {task, answer} objects that follow it) and violations (plain-string answers that break it): behaviors are what the user's own expert learns.${git_note} If nothing is worth saving this turn, just stop again -- the next stop proceeds automatically."

jq -nc --arg reason "$reason" '{decision:"block", reason:$reason}'
