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
set -u
command -v jq >/dev/null 2>&1 || { printf '%s' '{"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":"[Antumbra bootstrap: jq not installed]"}}'; exit 0; }

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

# --- handoffs waiting for this machine (R-7) ---------------------------------
# What a session on another of the user's machines left for this one, announced
# until a session marks it done. The server writes the lines; this only places
# them. No answer, no line: it fails open like everything else here.
handoffs=$(curl -sS --max-time 5 -X POST "$URL/mcp/call" "${auth[@]}" \
  -d "$(jq -nc --arg host "$HOST_ID" '{tool: "handoffs", arguments: {host: $host}}')" 2>/dev/null \
  | jq -r '.announcement // .result.announcement // empty' 2>/dev/null || true)

# --- recall, scoped to here when known --------------------------------------
args=$(jq -nc --arg repo "$repo" --arg branch "$branch" '
  {query: "standing conventions, project context, and active tasks for this agent", top_k: 12}
  + (if $repo != "" then {repo: $repo} else {} end)
  + (if $branch != "" then {branch: $branch} else {} end)')
payload=$(jq -nc --argjson a "$args" '{tool: "recall_memories", arguments: $a}')
resp=$(curl -sS --max-time 5 -X POST "$URL/mcp/call" "${auth[@]}" -d "$payload" 2>/dev/null || echo '{}')

# --- judge each hit's anchor with git in hand --------------------------------
# One status per memory id: live | not-on-head | orphaned. Memories with no
# anchor, or from another repository, get none (the server's `scope` still shows).
statuses='{}'
if [ -n "$commit" ]; then
  while IFS=$'\t' read -r id m_repo m_commit m_branch m_orphaned; do
    [ -z "$id" ] && continue
    st=""
    if [ -n "$m_commit" ] && [ "$m_repo" = "$repo" ]; then
      st="live"
      git merge-base --is-ancestor "$m_commit" HEAD 2>/dev/null || st="not-on-head"
      if [ -n "$m_branch" ] && [ "$m_branch" != "$branch" ] \
         && ! git show-ref --verify --quiet "refs/heads/$m_branch" \
         && ! git show-ref --verify --quiet "refs/remotes/origin/$m_branch"; then
        st="orphaned"
      fi
    fi
    # The server already knows when GitHub deleted the branch (the App's delete
    # event marks the memory), even if this clone still has a stale local ref.
    [ -n "$m_orphaned" ] && st="orphaned"
    [ -n "$st" ] && statuses=$(printf '%s' "$statuses" | jq -c --arg id "$id" --arg st "$st" '. + {($id): $st}')
  done < <(printf '%s' "$resp" | jq -r '
    (.memories // .result.memories // [])[]
    | [.id, (.provenance.repo // ""), (.provenance.commit // ""), (.provenance.branch // ""), (.orphaned_at // "")]
    | @tsv' 2>/dev/null)
fi

# Optionally push the judgment back: an orphaned memory loses standing now,
# instead of waiting for someone to notice it was about a branch that is gone.
if [ "$PENALIZE_ORPHANS" = "1" ]; then
  for id in $(printf '%s' "$statuses" | jq -r 'to_entries[] | select(.value == "orphaned") | .key'); do
    curl -sS --max-time 3 -X POST "$URL/mcp/call" "${auth[@]}" \
      -d "$(jq -nc --arg id "$id" '{tool: "penalize_memory", arguments: {memory_id: $id}}')" \
      >/dev/null 2>&1 || true
  done
fi

# --- what is different about this session (fail-open) ---------------------------
# With its telemetry off the agent has also lost its feature flags, and the
# features gated on them, and nothing tells it (ADR-0021). `antumbra claude brief`
# prints a few lines when that is so and nothing when it is not. No antumbra on
# the path, or any failure: no lines.
brief=""
if command -v "$BIN" >/dev/null 2>&1; then
  brief=$("$BIN" claude brief 2>/dev/null || true)
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
