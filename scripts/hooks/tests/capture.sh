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

mkdir -p "$WORK/bin" "$WORK/tmp"
cat > "$WORK/bin/curl" <<'CURL'
#!/usr/bin/env bash
printf '%s\n' "$@" > "$CURL_ARGS"
CURL
chmod +x "$WORK/bin/curl"
export CURL_ARGS="$WORK/args" ANTUMBRA_TOKEN=test-token TMPDIR="$WORK/tmp"

run() {
    rm -f "$CURL_ARGS"
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

echo 'the end of a turn on a named machine'
out="$(ANTUMBRA_HOST_ID=windows run '{"session_id":"s4","prompt_id":"p4","hook_event_name":"Stop","last_assistant_message":"done"}')"
grep -qx 'X-Antumbra-Host: windows' "$CURL_ARGS"
check 'names the machine as the MCP client does' $?

echo 'the stop after the nudge'
out="$(run '{"session_id":"s1","prompt_id":"p1","hook_event_name":"Stop","last_assistant_message":"saved"}')"
test -z "$out" && test ! -e "$CURL_ARGS"
check 'is silent and reports nothing again' $?

echo 'a compaction'
run '{"session_id":"s2","hook_event_name":"PreCompact"}' > /dev/null
test ! -e "$CURL_ARGS"
check 'reports no turn' $?

echo 'no token'
rm -f "$WORK/tmp"/antumbra-capture-*
out="$(printf '%s' '{"session_id":"s3","hook_event_name":"Stop","last_assistant_message":"x"}' |
    ANTUMBRA_TOKEN= ANTUMBRA_TOKEN_FILE="$WORK/none" PATH="$WORK/bin:$PATH" bash "$HOOK")"
printf '%s' "$out" | jq -e '.decision == "block"' > /dev/null && test ! -e "$CURL_ARGS"
check 'still nudges, and reports nothing' $?

exit "$FAILED"
