#!/usr/bin/env bash
# Tests for antumbra-mcp-headers.sh: what Claude Code's headersHelper receives.
# Needs bash and jq.
#
#   bash scripts/hooks/tests/mcp-headers.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
helper="$here/../antumbra-mcp-headers.sh"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
failed=0

check() { # description, then a command that must succeed
  local what="$1"; shift
  if "$@" >/dev/null 2>&1; then echo "  ok    $what"; else echo "  FAIL  $what"; failed=1; fi
}
bearer() { jq -r '.Authorization'; }
file="$work/token.txt"

echo "a token file"
printf 'test-token' >"$file"
out=$(ANTUMBRA_TOKEN_FILE="$file" bash "$helper" 2>/dev/null); code=$?
check "exits 0" test "$code" -eq 0
check "prints one JSON object with the bearer" test "$(bearer <<<"$out")" = "Bearer test-token"

echo "a token file with a trailing newline"
printf 'test-token\r\n' >"$file"
out=$(ANTUMBRA_TOKEN_FILE="$file" bash "$helper" 2>/dev/null)
check "sends the token without it" test "$(bearer <<<"$out")" = "Bearer test-token"

echo "a token that JSON must escape"
printf 'a"b\\c' >"$file"
out=$(ANTUMBRA_TOKEN_FILE="$file" bash "$helper" 2>/dev/null)
check "still prints valid JSON carrying it" test "$(bearer <<<"$out")" = 'Bearer a"b\c'

echo "no token file"
out=$(ANTUMBRA_TOKEN_FILE="$work/missing.txt" bash "$helper" 2>/dev/null); code=$?
check "exits non-zero" test "$code" -ne 0
check "prints no header" test -z "$out"

[ "$failed" = 0 ] && echo "all passed" || { echo "FAILED"; exit 1; }
