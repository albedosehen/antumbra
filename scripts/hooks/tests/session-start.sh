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
# Every call answers with the response file, and its arguments go to CURL_LOG,
# one line a call, so a test can say which tools the hook asked for.
cat >"$work/bin/curl" <<'STUB'
#!/usr/bin/env bash
[ -n "${CURL_LOG:-}" ] && printf '%s\n' "$*" >>"$CURL_LOG"
cat "$FAKE_RESPONSE"
STUB
# `claude reanchor` writes its arguments, and whether it was handed the token,
# to ANTUMBRA_TEST_MARKER, then takes five seconds, so a hook that waited for it
# would be caught.
cat >"$work/bin/antumbra" <<'STUB'
#!/usr/bin/env bash
[ "$1 $2" = "claude brief" ] && printf '%s\n' '## Sovereign mode' '' '- a line the agent must see'
if [ "$1 $2" = "claude reanchor" ]; then
  echo "$*" >"$ANTUMBRA_TEST_MARKER"
  [ -n "${ANTUMBRA_TOKEN:-}" ] && echo "token handed on" >>"$ANTUMBRA_TEST_MARKER"
  sleep 5
fi
exit 0
STUB
chmod +x "$work/bin/curl" "$work/bin/antumbra"

# Twelve memories of 1,500 characters: 18,000 in all, nearly twice the limit.
jq -nc '{memories: [range(1; 13) | {id: ("memory:" + tostring), content: ("M" + tostring + ":" + ("x" * 1500))}]}' >"$work/large.json"
jq -nc '{memories: [{id: "memory:a", content: "first small"}, {id: "memory:b", content: "second small"}]}' >"$work/small.json"

marker="$work/reanchored.txt"
# A clone with a GitHub origin, for the cases that need a repository.
mkdir -p "$work/clone"
git -C "$work/clone" init -q
git -C "$work/clone" remote add origin https://github.com/acme/orders.git

run_in() { # directory, response file, then extra environment assignments
  local dir="$1" response="$2"; shift 2
  (cd "$dir" && env PATH="$work/bin:$PATH" FAKE_RESPONSE="$response" ANTUMBRA_TOKEN=test-token \
    ANTUMBRA_TEST_MARKER="$marker" ANTUMBRA_REANCHOR_LOG="$work/reanchor.log" "$@" bash "$hook" </dev/null)
}
run() { # response file, then extra environment assignments
  local response="$1"; shift
  run_in "$work" "$response" "$@"
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

echo "a handoff waiting for this machine"
jq -nc '{memories: [{id: "memory:a", content: "first small"}],
  announcement: "1 handoff waiting for this machine (windows):\n- Rerun the probe (from kuskokwim, 2h ago; id memory:h)"}' \
  >"$work/handoff.json"
ctx=$(run "$work/handoff.json" | context)
check "announces it" grep -q '1 handoff waiting for this machine (windows):' <<<"$ctx"
handoff_at=$(grep -n 'handoff waiting' <<<"$ctx" | head -1 | cut -d: -f1)
first_at=$(grep -n 'first small' <<<"$ctx" | head -1 | cut -d: -f1)
check "puts it before any memory" test "${handoff_at:-999}" -lt "${first_at:-0}"
ctx=$(run "$work/small.json" | context)
check "says nothing of handoffs when none wait" bash -c '! grep -q "handoff" <<<"$1"' _ "$ctx"

echo "naming this machine"
calls="$work/calls.log"
: >"$calls"
ctx=$(run "$work/small.json" CURL_LOG="$calls" ANTUMBRA_HOST_ID=mac | context)
check "registers it under its name" grep -qF '{"tool":"register_device","arguments":{"host":"mac"}}' "$calls"
check "still answers" grep -q 'first small' <<<"$ctx"
: >"$calls"
run "$work/small.json" CURL_LOG="$calls" ANTUMBRA_HOST_ID= >/dev/null
check "registers nothing when the machine has no name" bash -c '! grep -q register_device "$1"' _ "$calls"
check "still asks for its handoffs" grep -qF '"tool":"handoffs"' "$calls"

echo "no antumbra on the path"
ctx=$(run "$work/small.json" ANTUMBRA_BIN=antumbra-is-not-installed | context)
check "still answers" grep -q 'first small' <<<"$ctx"
check "says nothing about the session" bash -c '! grep -q "Sovereign mode" <<<"$1"' _ "$ctx"

echo "no memories at all"
printf '{}' >"$work/empty.json"
ctx=$(run "$work/empty.json" | context)
check "starts cold and says so" grep -q 'starting cold' <<<"$ctx"
check "still says what is different about the session" grep -q 'Sovereign mode' <<<"$ctx"

echo "outside a repository"
check "reports no merges" test ! -e "$marker"

echo "in a clone"
started=$(date +%s)
ctx=$(run_in "$work/clone" "$work/small.json" | context)
took=$(( $(date +%s) - started ))
check "does not wait for the report (${took} s, the stub takes 5)" test "$took" -lt 4
for _ in $(seq 1 30); do [ -e "$marker" ] && break; sleep 0.2; done
check "reports the last three days of merges" grep -q 'claude reanchor --days 3' "$marker"
check "hands the report the token" grep -q 'token handed on' "$marker"
check "still answers" grep -q 'first small' <<<"$ctx"

echo "in a clone, turned off"
rm -f "$marker"
ctx=$(run_in "$work/clone" "$work/small.json" ANTUMBRA_REANCHOR=0 | context)
sleep 2
check "reports no merges" test ! -e "$marker"

[ "$failed" = 0 ] && echo "all passed" || { echo "FAILED"; exit 1; }
