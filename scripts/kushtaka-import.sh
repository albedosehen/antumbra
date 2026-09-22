#!/usr/bin/env bash
# Import a kushtaka-export.json into a RUNNING Antumbra MCP server, one memory per
# `store_memory` call (which embeds each through the server's configured embedder,
# so the result is genuinely recallable). The POSIX sibling of kushtaka-import.ps1,
# for hosts without PowerShell; identical behavior, same field mapping.
#
# Requires `jq` and `curl`.
#
# Antumbra MCP must already be serving with a real embedder, and a bearer token
# minted for the target (tenant, user):
#   antumbra-mcp --mint-token --tenant ws:default --user user:default \
#     --jwt-secret "$ANTUMBRA_JWT_SECRET" --token-ttl-days 365
#
# Env:
#   ANTUMBRA_URL         Antumbra MCP base (default http://127.0.0.1:8081)
#   ANTUMBRA_TOKEN       bearer JWT for the target (tenant, user)
#   ANTUMBRA_TOKEN_FILE  read the bearer from this file instead, so the credential
#                        appears in neither the environment nor a process listing
#                        (the same trade `--jwt-secret-file` makes on the server)
#
# Usage:
#   scripts/kushtaka-import.sh -i kushtaka-export.json
#   scripts/kushtaka-import.sh -i export.json --skip 0 --take 20   # canary first
#
# --skip/--take import a contiguous slice, which is what makes a canary possible:
# import a few of the WORST records (longest content), confirm they land and
# recall, then import the remainder. Because store_memory ids come from
# next_id("memory") rather than the content, a re-run duplicates rather than
# reconciles -- so a slice you have already imported must not be imported again.

set -euo pipefail

in_file="kushtaka-export.json"
skip=0
take=-1
throttle_ms=0

while [ $# -gt 0 ]; do
    case "$1" in
        -i|--in)       in_file="$2"; shift 2 ;;
        --skip)        skip="$2"; shift 2 ;;
        --take)        take="$2"; shift 2 ;;
        --throttle-ms) throttle_ms="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

api_url="${ANTUMBRA_URL:-http://127.0.0.1:8081}"
token="${ANTUMBRA_TOKEN:-}"
if [ -z "$token" ] && [ -n "${ANTUMBRA_TOKEN_FILE:-}" ]; then
    token="$(tr -d '\r\n' < "$ANTUMBRA_TOKEN_FILE")"
fi
if [ -z "$token" ]; then
    echo "ANTUMBRA_TOKEN (or ANTUMBRA_TOKEN_FILE) is not set (import needs a bearer token for the target tenant/user)." >&2
    exit 1
fi

total_in_file="$(jq 'length' "$in_file")"
if [ "$take" -lt 0 ]; then
    take=$(( total_in_file - skip ))
fi
echo "importing records [$skip, $(( skip + take ))) of $total_in_file -> $api_url"

# One compact JSON request body per line. `content` may contain newlines; jq -c
# escapes them, so a line is always exactly one record. Fields absent from the
# export are left out rather than sent as null, so the server applies its own
# documented defaults (the same choice the .ps1 makes).
bodies="$(mktemp)"
trap 'rm -f "$bodies"' EXIT
jq -c --argjson skip "$skip" --argjson take "$take" '
    .[$skip:($skip + $take)][]
    | { tool: "store_memory",
        arguments: ({ content: .content, network: .network }
                    + (if .confidence == null then {} else { confidence: .confidence } end)) }
' "$in_file" > "$bodies"

ok=0; fail=0; i=0
while IFS= read -r body; do
    i=$(( i + 1 ))
    code="$(printf '%s' "$body" | curl -s -o /dev/null -w '%{http_code}' \
        -X POST "$api_url/mcp/call" \
        -H 'Content-Type: application/json' \
        -H "Authorization: Bearer $token" \
        --max-time 120 --data-binary @- || echo 000)"
    if [ "$code" = "200" ]; then
        ok=$(( ok + 1 ))
    else
        fail=$(( fail + 1 ))
        if [ "$fail" -le 5 ]; then
            echo "  fail #$fail at record $(( skip + i )): HTTP $code"
        fi
    fi
    if [ $(( i % 200 )) -eq 0 ]; then
        echo "  $i/$take  ($ok ok, $fail fail)"
    fi
    if [ "$throttle_ms" -gt 0 ]; then
        sleep "$(awk -v ms="$throttle_ms" 'BEGIN { print ms / 1000 }')"
    fi
done < "$bodies"

echo "done: $ok stored, $fail failed, of $take"
[ "$fail" -eq 0 ]
