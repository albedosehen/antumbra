#!/usr/bin/env bash
# Antumbra SessionStart hook (macOS / Linux). POSIX mirror of
# antumbra-session-start.ps1. Bootstraps a coding-agent session: pulls the
# agent's standing conventions + the memory relevant to this project from a
# running antumbra-mcp surface and returns it as `additionalContext`. NEVER
# blocks session start -- every call has a timeout and any failure degrades to
# an empty bootstrap.
#
# Git-aware. Inside a repository it tells Antumbra where the session is (repo
# slug + branch) so recall scopes its hits, then checks each hit's git anchor
# with git itself -- is the commit still on HEAD, does the branch still exist --
# and tags it [live], [not-on-head], or [orphaned]. Nothing is re-extracted and
# nothing is garbage-collected: a stale memory is visible instead of silently
# wrong. Set ANTUMBRA_PENALIZE_ORPHANS=1 to also penalize orphaned memories.
#
# Talks to the REST convenience endpoint POST {ANTUMBRA_URL}/mcp/call
# {tool, arguments}, which dispatches the same tools as the JSON-RPC /mcp
# router under the same auth.
#
# Deps: curl, jq; git is optional (no repository, no anchor). set -u but NOT
# set -e (a curl or git failure must not crash the hook).
#
# One wall-clock budget, ANTUMBRA_SESSION_BUDGET_SEC (default 8), measured from
# the hook's start, bounds its calls and must stay below the hook's "timeout"
# in the agent's settings: past that the agent kills the hook and the session
# starts cold. The calls go out together, each capped at what is left.
set -u
command -v jq >/dev/null 2>&1 || { printf '%s' '{"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":"[Antumbra bootstrap: jq not installed]"}}'; exit 0; }

STARTED=$(date +%s)
BUDGET="${ANTUMBRA_SESSION_BUDGET_SEC:-8}"
case "$BUDGET" in ''|*[!0-9]*) BUDGET=8 ;; esac
# Whole seconds left for a call, keeping one back to judge anchors, render and
# print; never less than one.
remaining() {
  local left=$(( BUDGET - 1 - ($(date +%s) - STARTED) ))
  [ "$left" -lt 1 ] && left=1
  printf '%s' "$left"
}
answers=$(mktemp -d 2>/dev/null || { mkdir -p "/tmp/antumbra-session-$$" && printf '%s' "/tmp/antumbra-session-$$"; })
trap 'rm -rf "$answers"' EXIT

URL="${ANTUMBRA_URL:-http://127.0.0.1:8081}"
# The bearer, from the environment or from a file. The file is the better home
# for it: a hook's environment usually comes from the agent's own settings file,
# which is shared, diffed and backed up, and a long-lived credential does not
# belong there. ANTUMBRA_TOKEN_FILE names it; ~/.antumbra/token.txt is the
# default so the common case needs no configuration at all. A read failure is
# swallowed deliberately -- a bootstrap that cannot authenticate says so by
# starting cold, and must never break the session it is opening.
TOKEN="${ANTUMBRA_TOKEN:-}"
if [ -z "$TOKEN" ]; then
    TOKEN_FILE="${ANTUMBRA_TOKEN_FILE:-$HOME/.antumbra/token.txt}"
    [ -r "$TOKEN_FILE" ] && TOKEN="$(tr -d '\r\n' < "$TOKEN_FILE" 2>/dev/null || true)"
fi
HOST_ID="${ANTUMBRA_HOST_ID:-local}"
PENALIZE_ORPHANS="${ANTUMBRA_PENALIZE_ORPHANS:-0}"

auth=(-H 'Content-Type: application/json')
[ -n "$TOKEN" ] && auth+=(-H "Authorization: Bearer $TOKEN")

# One call to the surface, its answer in a file (empty for no answer). Run in
# the background, so the calls overlap one another and the git work below.
call() { # tool, arguments (JSON), answer file
  curl -sS --max-time "$(remaining)" -X POST "$URL/mcp/call" "${auth[@]}" \
    -d "$(jq -nc --arg tool "$1" --argjson args "$2" '{tool: $tool, arguments: $args}')" \
    >"$3" 2>/dev/null || : >"$3"
}
pending=""

# --- handoffs waiting for this machine ---------------------------------
# What a session on another of the user's machines left for this one, announced
# until a session marks it done. The server writes the lines; this only places
# them. No answer, no line: it fails open like everything else here.
call handoffs "$(jq -nc --arg host "$HOST_ID" '{host: $host}')" "$answers/handoffs.json" &
pending="$pending $!"

# --- this machine, in the user's fabric --------------------------------------
# A server registers the machine it runs on, and a laptop talking to a hosted
# hub runs none, so the session names it: that lists it among the user's devices
# and marks when it was last seen. Skipped for `local`, the name of a machine
# nobody named. The answer is not used, and no answer changes nothing.
if [ "$(printf '%s' "$HOST_ID" | tr '[:upper:]' '[:lower:]')" != "local" ]; then
  call register_device "$(jq -nc --arg host "$HOST_ID" '{host: $host}')" "$answers/register.json" &
  pending="$pending $!"
fi

# --- where the session is, in git terms (fail-open) -------------------------
repo=""; commit=""; branch=""
if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  commit=$(git rev-parse HEAD 2>/dev/null || true)
  branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)
  [ "$branch" = "HEAD" ] && branch=""
  remote=$(git remote get-url origin 2>/dev/null || true)
  # The ssh, scp, and https spellings of one remote become one slug: host/org/name.
  repo=$(printf '%s' "$remote" \
    | sed -E 's#^[a-z+]+://##; s#^[^/@]*@##; s#^([^/:]+):#\1/#; s#\.git/?$##; s#/+$##' \
    | tr '[:upper:]' '[:lower:]')
  case "$repo" in *\\*|/*) repo="" ;; */*) ;; *) repo="" ;; esac
fi

# --- recall, scoped to here when known --------------------------------------
args=$(jq -nc --arg repo "$repo" --arg branch "$branch" '
  {query: "standing conventions, project context, and active tasks for this agent", top_k: 12}
  + (if $repo != "" then {repo: $repo} else {} end)
  + (if $branch != "" then {branch: $branch} else {} end)')
call recall_memories "$args" "$answers/recall.json" &
pending="$pending $!"

# --- report this repository's recent merges (fail-open, detached) ------------
# A server GitHub cannot reach never hears that a branch merged, so what was
# learned on it reads other_branch from the base branch until someone says so.
# `antumbra claude reanchor` asks GitHub (with `gh`) for the last few days'
# merges and reports each one. It runs detached, so it never delays the session,
# and reporting a merge again moves nothing, so every session can run it. Off
# with ANTUMBRA_REANCHOR=0. The last run's output is in ~/.antumbra/reanchor.log,
# or wherever ANTUMBRA_REANCHOR_LOG names.
BIN="${ANTUMBRA_BIN:-antumbra}"
if [ -n "$repo" ] && [ "${ANTUMBRA_REANCHOR:-1}" != "0" ] && command -v "$BIN" >/dev/null 2>&1; then
  log="${ANTUMBRA_REANCHOR_LOG:-$HOME/.antumbra/reanchor.log}"
  mkdir -p "$(dirname "$log")" 2>/dev/null || true
  ( ANTUMBRA_URL="$URL" ANTUMBRA_TOKEN="$TOKEN" nohup "$BIN" claude reanchor --days 3       </dev/null >"$log" 2>&1 & ) 2>/dev/null || true
fi

# --- what is different about this session (fail-open) ---------------------------
# With its telemetry off the agent has also lost its feature flags, and the
# features gated on them, and nothing tells it. `antumbra claude brief`
# prints a few lines when that is so and nothing when it is not. No antumbra on
# the path, or any failure: no lines. It runs while the calls are in flight.
brief=""
if command -v "$BIN" >/dev/null 2>&1; then
  brief=$("$BIN" claude brief 2>/dev/null || true)
fi

# --- the answers -------------------------------------------------------------------
# shellcheck disable=SC2086 # one pid a word
wait $pending 2>/dev/null
handoffs=$(jq -r '.announcement // .result.announcement // empty' "$answers/handoffs.json" 2>/dev/null || true)
resp=$(cat "$answers/recall.json" 2>/dev/null)
[ -n "$resp" ] || resp='{}'

# --- judge each hit's anchor with git in hand --------------------------------
# One status per memory id: live | not-on-head | orphaned. Memories with no
# anchor, or from another repository, get none (the server's `scope` still shows).
# Three git processes and one jq judge every anchor at once, where asking per
# memory took up to three git processes and a jq each.
statuses='{}'
if [ -n "$commit" ]; then
  anchors=$(printf '%s' "$resp" | jq -r --arg repo "$repo" '
    (.memories // .result.memories // [])[]
    | select((.provenance.commit // "") != "" and .provenance.repo == $repo)
    | .provenance.commit' 2>/dev/null | tr -d '\r' | sort -u)
  # Which anchors name a commit this clone has, by its full id: {anchor: full}.
  known='{}'
  if [ -n "$anchors" ]; then
    known=$(paste -d ' ' <(printf '%s\n' "$anchors") <(printf '%s\n' "$anchors" | git cat-file --batch-check 2>/dev/null) \
      | jq -Rn '[inputs | split(" ") | select(.[2] == "commit") | {(.[0]): .[1]}] | add // {}' 2>/dev/null) || known='{}'
  fi
  # Which of those HEAD does not contain: listing what they reach that HEAD
  # does not names each one that is not an ancestor of HEAD.
  off_head=""
  fulls=$(printf '%s' "$known" | jq -r '.[]' 2>/dev/null | tr -d '\r' | sort -u)
  if [ -n "$fulls" ]; then
    # shellcheck disable=SC2086 # one commit id a word
    if ! off_head=$(git rev-list $fulls --not HEAD 2>/dev/null); then
      off_head=""
      for c in $fulls; do git merge-base --is-ancestor "$c" HEAD 2>/dev/null || off_head="$off_head$c"$'\n'; done
    fi
  fi
  # Every branch this clone knows, here and on origin.
  refs=$(git for-each-ref --format='%(refname)' refs/heads refs/remotes/origin 2>/dev/null)
  statuses=$(printf '%s' "$resp" | jq -c --arg repo "$repo" --arg branch "$branch" --argjson known "$known" \
    --arg off "$off_head" --arg refs "$refs" '
    ($off | split("\n") | map(select(. != "") | {(.): true}) | add // {}) as $offset
    | ($refs | split("\n") | map(select(. != "") | {(.): true}) | add // {}) as $refset
    | [ (.memories // .result.memories // [])[]
        | . as $m
        | ( if (($m.provenance.commit // "") != "" and $m.provenance.repo == $repo) then
              ( $known[$m.provenance.commit] as $full
                | (if $full != null and ($offset[$full] | not) then "live" else "not-on-head" end) as $st
                | ($m.provenance.branch // "") as $b
                | if $b != "" and $b != $branch
                     and ($refset["refs/heads/" + $b] | not)
                     and ($refset["refs/remotes/origin/" + $b] | not)
                  then "orphaned" else $st end )
            else "" end ) as $st
        # The server already knows when GitHub deleted the branch (the App delete
        # event marks the memory), even if this clone still has a stale local ref.
        | (if ($m.orphaned_at // "") != "" then "orphaned" else $st end) as $st
        | select($st != "")
        | {($m.id): $st} ]
    | add // {}' 2>/dev/null) || statuses='{}'
  [ -n "$statuses" ] || statuses='{}'
fi

# Optionally push the judgment back: an orphaned memory loses standing now,
# instead of waiting for someone to notice it was about a branch that is gone.
if [ "$PENALIZE_ORPHANS" = "1" ]; then
  penalties=""
  for id in $(printf '%s' "$statuses" | jq -r 'to_entries[] | select(.value == "orphaned") | .key'); do
    call penalize_memory "$(jq -nc --arg id "$id" '{memory_id: $id}')" /dev/null &
    penalties="$penalties $!"
  done
  # shellcheck disable=SC2086 # one pid a word
  wait $penalties 2>/dev/null
fi

# --- render ---------------------------------------------------------------------
# The live /mcp/call answers with the tool's value at the TOP level
# ({memories: [...]}); the .result envelope is tolerated for older shims.
entries=$(printf '%s' "$resp" | jq -c --argjson st "$statuses" '
  [ (.memories // .result.memories // [])[]
    | ( [ ($st[.id] // ""), (.scope // "") ] | map(select(. != ""))
        | if length > 0 then "[" + join(", ") + "] " else "" end )
      + .content
      + ( if .provenance
          then "\n  (learned at " + .provenance.repo + "@" + .provenance.commit
               + (if .provenance.branch then "#" + .provenance.branch else "" end) + ")"
          else "" end )
  ]' 2>/dev/null) || entries='[]'
[ -n "$entries" ] || entries='[]'

git_line=""
if [ -n "$commit" ]; then
  git_line="Git context: repo=${repo:-?} branch=${branch:-(detached)} commit=${commit}. When storing a memory about this code, pass provenance {repo: \"${repo}\", commit: \"${commit}\", branch: \"${branch}\"} to store_memory (add path for a single file) so a later session can tell whether it still applies. Tags: [live] the anchor is on HEAD; [not-on-head] learned on a commit this HEAD does not contain; [orphaned] its branch no longer exists here or on origin, or GitHub reported it deleted -- verify before relying on it, and penalize_memory if it is wrong."
fi

# The agent caps a hook's context at 10,000 characters. Past the cap it is handed
# a file path and a 2,000-character preview it is never asked to open, so an
# oversized bootstrap is a truncated one that says nothing about it. What must
# survive goes first; memories follow, best first, while they fit, and the rest
# are counted so the agent knows to recall them.
LIMIT=9500
context=$(jq -nc --argjson entries "$entries" --arg host "$HOST_ID" --arg git "$git_line" \
  --arg brief "$brief" --arg handoffs "$handoffs" --argjson limit "$LIMIT" '
  ( "# Antumbra session bootstrap (host=" + $host + ")\n\n"
    + (if $handoffs != "" then $handoffs + "\n\n" else "" end)
    + (if $brief != "" then $brief + "\n\n" else "" end)
    + (if $git != "" then $git + "\n\n" else "" end) ) as $head
  | "\n\n---\n\n" as $sep
  | ($limit - 160) as $room
  | ( reduce $entries[] as $e ({kept: [], used: ($head | length), omitted: 0};
        (($e | length) + ($sep | length)) as $cost
        | if .used + $cost <= $room
          then .kept += [$e] | .used += $cost
          else .omitted += 1 end) ) as $fit
  | $head
    + ( if ($entries | length) == 0
        then "[Antumbra bootstrap empty / unreachable -- starting cold.]"
        else ($fit.kept | join($sep)) end )
    + ( if $fit.omitted > 0
        then $sep + "[" + ($fit.omitted | tostring)
             + " more recalled but left out to stay under the 10,000-character limit on hook context; use recall_memories for them.]"
        else "" end )
' 2>/dev/null) || context='"[Antumbra hook serialization error]"'

jq -nc --argjson ctx "$context" '{hookSpecificOutput:{hookEventName:"SessionStart",additionalContext:$ctx}}'
