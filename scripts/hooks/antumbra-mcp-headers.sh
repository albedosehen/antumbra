#!/usr/bin/env bash
# Antumbra MCP auth header for Claude Code's `headersHelper` (macOS / Linux).
# POSIX mirror of antumbra-mcp-headers.ps1. Prints
# {"Authorization": "Bearer <token>"} with the token read from a file, so the
# MCP server entry needs no `${ANTUMBRA_TOKEN}` from the settings file's env
# block: the same file the hooks read, ANTUMBRA_TOKEN_FILE or
# ~/.antumbra/token.txt. A long-lived credential does not belong in a settings
# file that is shared, diffed and backed up.
#
# Claude Code runs the helper when it connects and sends whatever headers it
# prints. A missing or empty token file exits non-zero, which Claude Code reports
# as a failed connection, rather than sending a header that cannot work.
#
# Deps: jq, to write the JSON with the token escaped.
set -u
file="${ANTUMBRA_TOKEN_FILE:-$HOME/.antumbra/token.txt}"
token=""
[ -r "$file" ] && token="$(tr -d '\r\n' < "$file" 2>/dev/null || true)"
if [ -z "$token" ]; then
  echo "antumbra-mcp-headers: no token in $file" >&2
  exit 1
fi
jq -nc --arg t "$token" '{Authorization: ("Bearer " + $t)}'
