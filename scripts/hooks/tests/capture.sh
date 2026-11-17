#!/usr/bin/env bash
# Tests for antumbra-capture.sh's report of a turn's end. No Antumbra needed:
# `curl` is a stand-in on the PATH that records what it was asked.
#
#   bash scripts/hooks/tests/capture.sh
set -u
HOOK="${ANTUMBRA_TEST_HOOK:-$(dirname "$0")/../antumbra-capture.sh}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
FAILED=0

check() {
    if [ "$2" = 0 ]; then echo "  ok    $1"; else echo "  FAIL  $1"; FAILED=1; fi
}

mkdir -p "$WORK/bin" "$WORK/run"
cat > "$WORK/bin/curl" <<'CURL'
#!/usr/bin/env bash
printf '%s\n' "$@" > "$CURL_ARGS"
# What `--config -` would read: curl always drains its stdin for it.
cat > "${CURL_CONFIG:-/dev/null}"
CURL
chmod +x "$WORK/bin/curl"
export CURL_ARGS="$WORK/args" CURL_CONFIG="$WORK/config" ANTUMBRA_TOKEN=test-token XDG_RUNTIME_DIR="$WORK/run"

run() {
    rm -f "$CURL_ARGS" "$CURL_CONFIG"
    printf '%s' "$1" | PATH="$WORK/bin:$PATH" bash "$HOOK"
}

echo 'the end of a turn'
out="$(run '{"session_id":"s1","prompt_id":"p1","hook_event_name":"Stop","last_assistant_message":"deployed it blue green"}')"
printf '%s' "$out" | jq -e '.decision == "block"' > /dev/null
check 'still asks for memories before stopping' $?
body="$(grep '^{' "$CURL_ARGS" 2>/dev/null)"
printf '%s' "$body" | jq -e '.tool == "observe_turn"' > /dev/null
check 'reports the turn to the server' $?
printf '%s' "$body" | jq -e '.arguments == {session: "s1", text: "deployed it blue green", turn: "p1"}' > /dev/null
check 'with its session, turn and answer' $?
! grep -q 'Bearer' "$CURL_ARGS"
check "keeps the bearer off curl's command line" $?
grep -qx 'header = "Authorization: Bearer test-token"' "$CURL_CONFIG"
check 'and hands it to curl through its config on stdin' $?

echo 'the end of a turn on a named machine'
out="$(ANTUMBRA_HOST_ID=windows run '{"session_id":"s4","prompt_id":"p4","hook_event_name":"Stop","last_assistant_message":"done"}')"
grep -qx 'X-Antumbra-Host: windows' "$CURL_ARGS"
check 'names the machine as the MCP client does' $?

echo 'the stop after the nudge'
out="$(run '{"session_id":"s1","prompt_id":"p1","hook_event_name":"Stop","last_assistant_message":"saved"}')"
test -z "$out" && test ! -e "$CURL_ARGS"
check 'is silent and reports nothing again' $?

echo 'a session nudged a moment ago'
out="$(run '{"session_id":"s1","prompt_id":"p3","hook_event_name":"Stop","last_assistant_message":"waiting on a build"}')"
test -z "$out"
check 'is let stop without another nudge' $?
grep '^{' "$CURL_ARGS" 2>/dev/null | jq -e '.arguments.text == "waiting on a build"' > /dev/null
check 'and the end of its turn is still reported' $?
out="$(ANTUMBRA_CAPTURE_EVERY_SECS=0 run '{"session_id":"s1","prompt_id":"p4","hook_event_name":"Stop","last_assistant_message":"more"}')"
printf '%s' "$out" | jq -e '.decision == "block"' > /dev/null
check 'unless the interval is set to none, which nudges every turn' $?
rm -f "$WORK/run/antumbra-capture-Stop-s1"

echo 'a turn that saved as it went'
TRANSCRIPT="$WORK/transcript.jsonl"
printf '%s\n' '{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t1","name":"mcp__antumbra__store_memory","input":{}}]}}' > "$TRANSCRIPT"
stop() { printf '{"session_id":"%s","hook_event_name":"Stop","last_assistant_message":"%s","transcript_path":"%s"}' "$1" "$2" "$TRANSCRIPT"; }
out="$(run "$(stop s5 'saved it')")"
test -z "$out"
check 'is not stopped to save again' $?
test -e "$CURL_ARGS"
check 'and the end of its turn is still reported' $?
out="$(ANTUMBRA_CAPTURE_EVERY_SECS=0 run "$(stop s5 'nothing new')")"
printf '%s' "$out" | jq -e '.decision == "block"' > /dev/null
check 'the save is not counted a second time' $?
rm -f "$WORK/run/antumbra-capture-Stop-s5"
printf '%s\n' '{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t1","name":"mcp__antumbra__record_behavior","input":{}}]}}' >> "$TRANSCRIPT"
out="$(ANTUMBRA_CAPTURE_EVERY_SECS=0 run "$(stop s5 'recorded one')")"
test -z "$out"
check 'a recorded behavior counts as a save' $?
printf '%s\n' '{"type":"user","message":{"content":[{"type":"tool_result","content":"{\"name\": \"mcp__antumbra__store_memory\"}"}]}}' > "$TRANSCRIPT"
out="$(run "$(stop s6 'looked a tool up')")"
printf '%s' "$out" | jq -e '.decision == "block"' > /dev/null
check 'a tool only named inside a result does not' $?

echo 'a stop the nudge itself caused'
out="$(ANTUMBRA_CAPTURE_EVERY_SECS=0 run '{"session_id":"s7","hook_event_name":"Stop","stop_hook_active":true,"last_assistant_message":"saved"}')"
test -z "$out"
check 'goes through, with no sentinel left to say so' $?

echo 'a compaction'
run '{"session_id":"s2","hook_event_name":"PreCompact"}' > /dev/null
test ! -e "$CURL_ARGS"
check 'reports no turn' $?
# Nudged a moment ago at the end of a turn: a compaction is nudged all the same.
out="$(run '{"session_id":"s1","hook_event_name":"PreCompact"}')"
printf '%s' "$out" | jq -e '.decision == "block"' > /dev/null
check 'is always nudged' $?

echo 'a hostile session id'
rm -f "$WORK/run"/antumbra-capture-*
out="$(run '{"session_id":"../../x y","hook_event_name":"Stop/../z","last_assistant_message":"x"}')"
printf '%s' "$out" | jq -e '.decision == "block"' > /dev/null
check 'is handled like any other' $?
test -f "$WORK/run/antumbra-capture-Stopz-xy"
check 'names the sentinel from the id cut to [A-Za-z0-9_-], under the runtime directory' $?
test "$(ls "$WORK/run" | wc -l | tr -d ' ')" = 1 && test ! -e "$WORK/x y" && test ! -e "$WORK/run/../x y"
check 'and touches nothing else' $?
rm -f "$WORK/run"/antumbra-capture-*

echo 'no token'
rm -f "$WORK/run"/antumbra-capture-*
out="$(printf '%s' '{"session_id":"s3","hook_event_name":"Stop","last_assistant_message":"x"}' |
    ANTUMBRA_TOKEN= ANTUMBRA_TOKEN_FILE="$WORK/none" PATH="$WORK/bin:$PATH" bash "$HOOK")"
printf '%s' "$out" | jq -e '.decision == "block"' > /dev/null && test ! -e "$CURL_ARGS"
check 'still nudges, and reports nothing' $?

exit "$FAILED"
