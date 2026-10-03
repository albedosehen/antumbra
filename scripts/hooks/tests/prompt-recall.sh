#!/usr/bin/env bash
# Tests for antumbra-prompt-recall.sh. No Antumbra needed: `curl` is a stand-in
# on the PATH that answers from a file and leaves a mark when it is called.
#
#   bash scripts/hooks/tests/prompt-recall.sh
set -u
HOOK="${ANTUMBRA_TEST_HOOK:-$(dirname "$0")/../antumbra-prompt-recall.sh}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
FAILED=0

check() {
    if [ "$2" = 0 ]; then echo "  ok    $1"; else echo "  FAIL  $1"; FAILED=1; fi
}

mkdir -p "$WORK/bin"
cat > "$WORK/bin/curl" <<'CURL'
#!/usr/bin/env bash
touch "$CURL_CALLED"
cat "$CURL_ANSWER"
CURL
chmod +x "$WORK/bin/curl"
printf '%s' '{"memories":[{"id":"memory:a","network":"world","content":"keep it plain"}]}' > "$WORK/answer.json"
export CURL_ANSWER="$WORK/answer.json" CURL_CALLED="$WORK/called" ANTUMBRA_TOKEN=test-token

run() {
    rm -f "$CURL_CALLED"
    jq -nc --arg p "$1" --arg cwd "$WORK" '{ prompt: $p, cwd: $cwd }' |
        PATH="$WORK/bin:$PATH" bash "$HOOK"
}

echo 'a prompt'
out="$(run 'how do we word caveats')"
printf '%s' "$out" | jq -e '.hookSpecificOutput.additionalContext | contains("keep it plain")' > /dev/null
check 'is recalled for' $?

# A background task's notification reaches the hook as a prompt too. A recall
# on its wording finds memories about notifications, every time.
echo "a background task's notification"
out="$(run "$(printf '<task-notification>\n<task-id>b1</task-id>\n<status>completed</status>\n</task-notification>')")"
test -z "$out"
check 'says nothing' $?
test ! -e "$CURL_CALLED"
check 'asks the surface nothing' $?

echo 'a prompt that only mentions one'
out="$(run 'why does a <task-notification> arrive twice')"
test -n "$out"
check 'is recalled for' $?

exit "$FAILED"
