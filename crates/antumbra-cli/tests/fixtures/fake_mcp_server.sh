#!/usr/bin/env bash
# A stdio MCP server that knows four things: the handshake, `tools/list`,
# `tools/call` (every tool says "ok"), and `ping`. It lists the tools in the JSON
# file given as its first argument (default: fake_tools.json beside it), which
# must be one line. Needs only bash and sed. PowerShell sibling:
# fake_mcp_server.ps1.
here=$(cd "$(dirname "$0")" && pwd)
file="${1:-$here/fake_tools.json}"
[ -r "$file" ] || { echo "fake_mcp_server: cannot read $file" >&2; exit 1; }
tools=$(tr -d '\r\n' < "$file")
while IFS= read -r line; do
  id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9][0-9]*\).*/\1/p')
  case "$line" in
    *'"method":"initialize"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2025-06-18","capabilities":{"tools":{}},"serverInfo":{"name":"fake","version":"0"}}}\n' "$id" ;;
    *'"method":"tools/list"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":%s}}\n' "$id" "$tools" ;;
    *'"method":"tools/call"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"ok"}]}}\n' "$id" ;;
    *'"method":"ping"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{}}\n' "$id" ;;
  esac
done
