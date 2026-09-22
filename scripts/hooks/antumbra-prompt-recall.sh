#!/usr/bin/env bash
# UserPromptSubmit: recall the memories most relevant to what the user just
# typed and inject them as additionalContext, so recall happens on every message
# rather than only at session start. POSIX sibling of
# antumbra-prompt-recall.ps1; identical behaviour. Needs `jq` and `curl`.
#
# The bootstrap (antumbra-session-start) runs once and cannot know what the
# session will turn out to be about. This runs per prompt and does, which is why
# both exist.
#
# Env (see scripts/hooks/README.md):
#   ANTUMBRA_URL         the antumbra-mcp engine (default http://127.0.0.1:8081)
#   ANTUMBRA_TOKEN       bearer JWT, or
#   ANTUMBRA_TOKEN_FILE  a file holding it (default ~/.antumbra/token.txt)
#
# It never fails a prompt. Every failure path exits 0 with no output: a recall
# that cannot answer must not stop the user from talking to the agent.

URL="${ANTUMBRA_URL:-http://127.0.0.1:8081}"
TOKEN="${ANTUMBRA_TOKEN:-}"
if [ -z "$TOKEN" ]; then
    TOKEN_FILE="${ANTUMBRA_TOKEN_FILE:-$HOME/.antumbra/token.txt}"
    [ -r "$TOKEN_FILE" ] && TOKEN="$(tr -d '\r\n' < "$TOKEN_FILE" 2>/dev/null || true)"
fi
[ -z "$TOKEN" ] && exit 0
command -v jq >/dev/null 2>&1 || exit 0
command -v curl >/dev/null 2>&1 || exit 0

PAYLOAD="$(cat)"
PROMPT="$(printf '%s' "$PAYLOAD" | jq -r '.prompt // ""' 2>/dev/null || echo '')"
CWD="$(printf '%s' "$PAYLOAD" | jq -r '.cwd // ""' 2>/dev/null || echo '')"
[ "${#PROMPT}" -lt 2 ] && exit 0

# Bound the query; the embedder has a 512-token window and a novel would be
# truncated into noise anyway.
QUERY="$(printf '%s' "$PROMPT" | cut -c1-500)"

# Where the caller is. The server scopes hits against this and, since scope
# reaches RETRIEVAL rather than only ordering, it also widens the candidate pool
# so a repo-anchored memory can actually compete.
REPO=""
BRANCH=""
if [ -n "$CWD" ] && [ -d "$CWD" ]; then
    ORIGIN="$(git -C "$CWD" config --get remote.origin.url 2>/dev/null || true)"
    if [ -n "$ORIGIN" ]; then
        REPO="$(printf '%s' "$ORIGIN" | sed -E 's#^git@([^:]+):#https://\1/#; s#\.git$##; s#^https?://##')"
    fi
    BRANCH="$(git -C "$CWD" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
    [ "$BRANCH" = "HEAD" ] && BRANCH=""
fi

BODY="$(jq -nc --arg q "$QUERY" --arg repo "$REPO" --arg branch "$BRANCH" '
    { tool: "recall_memories",
      arguments: ({ query: $q, top_k: 3 }
                  + (if $repo   == "" then {} else { repo: $repo }     end)
                  + (if $branch == "" then {} else { branch: $branch } end)) }')"

RESP="$(curl -s -X POST "$URL/mcp/call" \
    -H 'Content-Type: application/json' \
    -H "Authorization: Bearer $TOKEN" \
    --max-time 10 --data-binary "$BODY" 2>/dev/null || true)"
[ -z "$RESP" ] && exit 0

# recall_memories answers {"memories":[...]} at the TOP LEVEL -- there is no
# `result` wrapper. Both shapes are accepted so a future wrapper would not
# silently empty this.
LINES="$(printf '%s' "$RESP" | jq -r '
    (.memories // .result.memories // [])
    | map(select(.content != null))
    | map("- (" + (.network // "unknown") + ")"
          + (if .scope then " [" + .scope + "]" else "" end) + " "
          + (if (.content | length) > 900
             then (.content[0:900] + " [truncated - full entry: " + (.id // "?") + "]")
             else .content end))
    | join("\n\n")' 2>/dev/null || true)"
[ -z "$LINES" ] && exit 0

CONTEXT="## Antumbra recall (top matches for this prompt)

$LINES

These are automatic; call recall_memories for a deeper or differently-phrased search."

jq -nc --arg ctx "$CONTEXT" \
    '{ hookSpecificOutput: { hookEventName: "UserPromptSubmit", additionalContext: $ctx } }'
