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
# With CURL_HANG set it is a surface that never answers: it waits out the
# call's own --max-time and fails the way curl does.
cat >"$work/bin/curl" <<'STUB'
#!/usr/bin/env bash
# What `--config -` would read: curl always drains its stdin for it.
cat >>"${CURL_STDIN:-/dev/null}"
[ -n "${CURL_LOG:-}" ] && printf '%s\n' "$*" >>"$CURL_LOG"
if [ -n "${CURL_HANG:-}" ]; then
  limit=30; prev=""
  for a in "$@"; do [ "$prev" = "--max-time" ] && limit="$a"; prev="$a"; done
  sleep "$limit"; exit 28
fi
cat "$FAKE_RESPONSE"
STUB
# `claude reanchor` writes its arguments, and whether it was handed the token,
# to ANTUMBRA_TEST_MARKER, then takes five seconds, so a hook that waited for it
# would be caught. `claude remember` does the same to ANTUMBRA_TEST_REMEMBERED.
cat >"$work/bin/antumbra" <<'STUB'
#!/usr/bin/env bash
[ "$1 $2" = "claude brief" ] && printf '%s\n' '## Sovereign mode' '' '- a line the agent must see'
if [ "$1 $2" = "claude reanchor" ]; then
  echo "$*" >"$ANTUMBRA_TEST_MARKER"
  [ -n "${ANTUMBRA_TOKEN:-}" ] && echo "token handed on" >>"$ANTUMBRA_TEST_MARKER"
  sleep 5
fi
if [ "$1 $2" = "claude remember" ]; then
  echo "$*" >"$ANTUMBRA_TEST_REMEMBERED"
  [ -n "${ANTUMBRA_TOKEN:-}" ] && echo "token handed on" >>"$ANTUMBRA_TEST_REMEMBERED"
  sleep 5
fi
exit 0
STUB
chmod +x "$work/bin/curl" "$work/bin/antumbra"

# Twelve memories of 1,500 characters: 18,000 in all, nearly twice the limit.
jq -nc '{memories: [range(1; 13) | {id: ("memory:" + tostring), content: ("M" + tostring + ":" + ("x" * 1500))}]}' >"$work/large.json"
jq -nc '{memories: [{id: "memory:a", content: "first small"}, {id: "memory:b", content: "second small"}]}' >"$work/small.json"

marker="$work/reanchored.txt"
remembered="$work/remembered.txt"
# A clone with a GitHub origin, for the cases that need a repository.
mkdir -p "$work/clone"
git -C "$work/clone" init -q
git -C "$work/clone" remote add origin https://github.com/acme/orders.git

run_in() { # directory, response file, then extra environment assignments
  local dir="$1" response="$2"; shift 2
  (cd "$dir" && env PATH="$work/bin:$PATH" FAKE_RESPONSE="$response" ANTUMBRA_TOKEN=test-token \
    ANTUMBRA_TEST_MARKER="$marker" ANTUMBRA_REANCHOR_LOG="$work/reanchor.log" \
    ANTUMBRA_TEST_REMEMBERED="$remembered" ANTUMBRA_REMEMBER_LOG="$work/remember.log" "$@" bash "$hook" </dev/null)
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

echo "for an agent that takes less"
ctx=$(run "$work/large.json" ANTUMBRA_CONTEXT_LIMIT=4000 | context)
check "stays under the limit it was given (${#ctx})" test "${#ctx}" -le 4000
check "keeps the best memory" grep -q 'M1:x' <<<"$ctx"
check "says it left some out" grep -q 'more recalled but left out' <<<"$ctx"

echo "a small recall"
ctx=$(run "$work/small.json" | context)
check "keeps everything" grep -q 'second small' <<<"$ctx"
check "mentions no omission" bash -c '! grep -q "left out" <<<"$1"' _ "$ctx"

echo "a machine that joined workspaces"
joined="$work/workspaces"
mkdir -p "$joined/a__ws_crew" "$joined/b__ws_lab" "$joined/c__ws_ask"
printf '%s' '{"url":"https://team.example","workspace":"ws:crew","user":"user:bodil","token_file":"/nowhere","write":"workspace","server":"antumbra-crew"}' > "$joined/a__ws_crew/workspace.json"
printf '%s' '{"url":"https://lab.example","workspace":"ws:lab","user":"user:bodil","token_file":"/nowhere","write":"workspace"}' > "$joined/b__ws_lab/workspace.json"
printf '%s' '{"url":"https://ask.example","workspace":"ws:ask","token_file":"/nowhere","write":"ask"}' > "$joined/c__ws_ask/workspace.json"
ctx=$(run "$work/small.json" ANTUMBRA_WORKSPACES_DIR="$joined" | context)
check "names each" grep -q -- '- ws:crew at https://team.example, as user:bodil' <<<"$ctx"
check "says which has tools, and by what name" grep -q 'Its tools are the MCP server `antumbra-crew`' <<<"$ctx"
check "sends the team's to a workspace it can write to" grep -q "ws:crew.*Store what you learn about this team's work there" <<<"$ctx"
check "and not to one it has no tools for" grep -q 'ws:lab.*You have no tools for it.*New memories go to your own store. Write to the workspace only when the user asks' <<<"$ctx"
check "asks where it was told to ask" grep -q 'ws:ask.*ask the user whether to store it in the workspace too' <<<"$ctx"
check "before any memory" test "$(grep -n 'Workspaces this machine joined' <<<"$ctx" | cut -d: -f1)" -lt "$(grep -n 'first small' <<<"$ctx" | cut -d: -f1)"
ctx=$(run "$work/small.json" ANTUMBRA_WORKSPACES_DIR="$work/none" | context)
check "says nothing of workspaces when it joined none" bash -c '! grep -q "Workspaces this machine joined" <<<"$1"' _ "$ctx"

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

echo "the bearer"
stdin="$work/curl-stdin"
: >"$calls"
: >"$stdin"
ctx=$(run "$work/small.json" CURL_LOG="$calls" CURL_STDIN="$stdin" | context)
check "is not on curl's command line" bash -c '! grep -q Bearer "$1"' _ "$calls"
check "reaches curl through its config on stdin, on every call" test "$(grep -cx 'header = "Authorization: Bearer test-token"' "$stdin")" -ge 2
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
# The stub's brief says the session is in the mode the rules are for.
for _ in $(seq 1 30); do [ -e "$remembered" ] && break; sleep 0.2; done
check "keeps the rules the session runs by" grep -q 'claude remember' "$remembered"
check "hands that the token" grep -q 'token handed on' "$remembered"

echo "the rules, turned off"
rm -f "$remembered"
ctx=$(run_in "$work/clone" "$work/small.json" ANTUMBRA_REANCHOR=0 ANTUMBRA_REMEMBER=0 | context)
sleep 2
check "keeps no rules" test ! -e "$remembered"
check "still answers" grep -q 'first small' <<<"$ctx"

echo "the brief, turned off"
rm -f "$remembered"
ctx=$(run_in "$work/clone" "$work/small.json" ANTUMBRA_REANCHOR=0 ANTUMBRA_BRIEF=0 | context)
sleep 2
check "says nothing of the session's mode" bash -c '! grep -q "Sovereign mode" <<<"$1"' _ "$ctx"
check "keeps no rules" test ! -e "$remembered"
check "still answers" grep -q 'first small' <<<"$ctx"

echo "the rules, outside the mode they are for"
ctx=$(run_in "$work/clone" "$work/small.json" ANTUMBRA_REANCHOR=0 ANTUMBRA_BIN=antumbra-is-not-installed | context)
sleep 2
check "keeps no rules when the brief says nothing" test ! -e "$remembered"

echo "in a clone, turned off"
rm -f "$marker"
ctx=$(run_in "$work/clone" "$work/small.json" ANTUMBRA_REANCHOR=0 | context)
sleep 2
check "reports no merges" test ! -e "$marker"

echo "anchors judged against history"
# `main` is a, then c; `feature` branched at a and holds b, unmerged.
hist="$work/history"
mkdir -p "$hist"
git -C "$hist" init -q
git -C "$hist" checkout -q -b main
git -C "$hist" remote add origin https://github.com/acme/orders.git
commit() { git -C "$hist" -c user.name=t -c user.email=t@example.com commit -q --allow-empty -m "$1"; git -C "$hist" rev-parse HEAD; }
on_main=$(commit a)
git -C "$hist" checkout -q -b feature
on_feature=$(commit b)
git -C "$hist" checkout -q main
commit c >/dev/null
jq -nc --arg a "$on_main" --arg b "$on_feature" '
  def anchored($id; $content; $commit; $branch; $repo):
    {id: $id, content: $content, provenance: {repo: $repo, commit: $commit, branch: $branch}};
  {memories: [
    anchored("memory:1"; "anchor-live"; $a; "main"; "github.com/acme/orders"),
    anchored("memory:2"; "anchor-offhead"; $b; "feature"; "github.com/acme/orders"),
    anchored("memory:3"; "anchor-orphaned"; $a; "gone"; "github.com/acme/orders"),
    anchored("memory:4"; "anchor-unknown"; "0123456789abcdef0123456789abcdef01234567"; "main"; "github.com/acme/orders"),
    anchored("memory:5"; "anchor-short"; $a[0:7]; "main"; "github.com/acme/orders"),
    anchored("memory:6"; "anchor-elsewhere"; $a; "main"; "github.com/acme/other")
  ]}' >"$work/anchors.json"
ctx=$(run_in "$hist" "$work/anchors.json" ANTUMBRA_REANCHOR=0 | context)
check "an anchor on HEAD is live" grep -qF '[live] anchor-live' <<<"$ctx"
check "an unmerged commit is not on HEAD" grep -qF '[not-on-head] anchor-offhead' <<<"$ctx"
check "a deleted branch is orphaned" grep -qF '[orphaned] anchor-orphaned' <<<"$ctx"
check "a commit this clone lacks is not on HEAD" grep -qF '[not-on-head] anchor-unknown' <<<"$ctx"
check "a short commit id is judged too" grep -qF '[live] anchor-short' <<<"$ctx"
check "another repository gets no tag" bash -c '! grep -qF "] anchor-elsewhere" <<<"$1"' _ "$ctx"

echo "a surface that never answers"
started=$(date +%s)
ctx=$(run "$work/small.json" CURL_HANG=1 ANTUMBRA_SESSION_BUDGET_SEC=3 | context)
took=$(( $(date +%s) - started ))
check "answers inside its budget (${took} s, budget 3)" test "$took" -le 4
check "starts cold and says so" grep -q 'starting cold' <<<"$ctx"

[ "$failed" = 0 ] && echo "all passed" || { echo "FAILED"; exit 1; }
