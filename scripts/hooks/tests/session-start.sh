#!/usr/bin/env bash
# Tests for antumbra-session-start.sh. No server and no antumbra needed: `curl`
# and `antumbra` are stubs first on the PATH. Needs bash and jq.
#
#   bash scripts/hooks/tests/session-start.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
hook="$here/../antumbra-session-start.sh"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
failed=0

check() { # description, then a command that must succeed
  local what="$1"; shift
  if "$@" >/dev/null 2>&1; then echo "  ok    $what"; else echo "  FAIL  $what"; failed=1; fi
}

mkdir -p "$work/bin"
cat >"$work/bin/curl" <<'STUB'
#!/usr/bin/env bash
cat "$FAKE_RESPONSE"
STUB
cat >"$work/bin/antumbra" <<'STUB'
#!/usr/bin/env bash
[ "$1 $2" = "claude brief" ] && printf '%s\n' '## Sovereign mode' '' '- a line the agent must see'
STUB
chmod +x "$work/bin/curl" "$work/bin/antumbra"

# Twelve memories of 1,500 characters: 18,000 in all, nearly twice the limit.
jq -nc '{memories: [range(1; 13) | {id: ("memory:" + tostring), content: ("M" + tostring + ":" + ("x" * 1500))}]}' >"$work/large.json"
jq -nc '{memories: [{id: "memory:a", content: "first small"}, {id: "memory:b", content: "second small"}]}' >"$work/small.json"

run() { # response file, then extra environment assignments
  local response="$1"; shift
  (cd "$work" && env PATH="$work/bin:$PATH" FAKE_RESPONSE="$response" "$@" bash "$hook" </dev/null)
}
context() { jq -r '.hookSpecificOutput.additionalContext'; }

echo "an oversized recall"
out=$(run "$work/large.json")
ctx=$(printf '%s' "$out" | context)
check "is a SessionStart answer" jq -e '.hookSpecificOutput.hookEventName == "SessionStart"' <<<"$out"
check "stays under the agent's 10,000-character limit (${#ctx})" test "${#ctx}" -le 10000
check "keeps the best memory" grep -q 'M1:x' <<<"$ctx"
kept=$(grep -o 'M[0-9]*:x' <<<"$ctx" | wc -l | tr -d ' ')
omitted=$(sed -n 's/.*\[\([0-9]*\) more recalled but left out.*/\1/p' <<<"$ctx")
check "says how many it left out, and none go missing (kept $kept, left out ${omitted:-none})" test "$((kept + ${omitted:-0}))" -eq 12
check "leaves some out" test "${omitted:-0}" -gt 0
brief_at=$(grep -n 'Sovereign mode' <<<"$ctx" | head -1 | cut -d: -f1)
memory_at=$(grep -n 'M1:x' <<<"$ctx" | head -1 | cut -d: -f1)
check "says what is different about the session before any memory" test "${brief_at:-999}" -lt "${memory_at:-0}"

echo "a small recall"
ctx=$(run "$work/small.json" | context)
check "keeps everything" grep -q 'second small' <<<"$ctx"
check "mentions no omission" bash -c '! grep -q "left out" <<<"$1"' _ "$ctx"

echo "no antumbra on the path"
ctx=$(run "$work/small.json" ANTUMBRA_BIN=antumbra-is-not-installed | context)
check "still answers" grep -q 'first small' <<<"$ctx"
check "says nothing about the session" bash -c '! grep -q "Sovereign mode" <<<"$1"' _ "$ctx"

echo "no memories at all"
printf '{}' >"$work/empty.json"
ctx=$(run "$work/empty.json" | context)
check "starts cold and says so" grep -q 'starting cold' <<<"$ctx"
check "still says what is different about the session" grep -q 'Sovereign mode' <<<"$ctx"

[ "$failed" = 0 ] && echo "all passed" || { echo "FAILED"; exit 1; }
