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
# What `--config -` would read: curl always drains its stdin for it.
case "$*" in
    *team.example*) cat > "${CURL_CONFIG:-/dev/null}.team" ;;
    *) printf '%s\n' "$@" > "$CURL_ARGS"; cat > "${CURL_CONFIG:-/dev/null}" ;;
esac
# A workspace's address is answered from the workspace's file.
case "$*" in
    *team.example*) printf '%s\n' "$@" > "$CURL_ARGS.team"; cat "$CURL_ANSWER.team" ;;
    *) cat "$CURL_ANSWER" ;;
esac
CURL
chmod +x "$WORK/bin/curl"
printf '%s' '{"memories":[{"id":"memory:a","network":"world","content":"keep it plain"}]}' > "$WORK/answer.json"
export CURL_ANSWER="$WORK/answer.json" CURL_CALLED="$WORK/called" CURL_ARGS="$WORK/args" CURL_CONFIG="$WORK/config" ANTUMBRA_TOKEN=test-token

run() {
    rm -f "$CURL_CALLED"
    jq -nc --arg p "$1" --arg cwd "$WORK" '{ prompt: $p, cwd: $cwd, session_id: "session-1" }' |
        PATH="$WORK/bin:$PATH" bash "$HOOK"
}

echo 'a prompt'
out="$(run 'how do we word caveats')"
printf '%s' "$out" | jq -e '.hookSpecificOutput.additionalContext | contains("keep it plain")' > /dev/null
check 'is recalled for' $?
printf '%s' "$out" | jq -e '.hookSpecificOutput.additionalContext | contains("memory:a: keep it plain")' > /dev/null
check 'names each memory, so an agent can judge it' $?
grep -q '"session":"session-1"' "$CURL_ARGS"
check 'names its session on the recall' $?
! grep -q 'Bearer' "$CURL_ARGS"
check "keeps the bearer off curl's command line" $?
grep -qx 'header = "Authorization: Bearer test-token"' "$CURL_CONFIG"
check 'and hands it to curl through its config on stdin' $?

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

# A machine that joined a workspace (`antumbra join`) recalls from its own
# store and from the workspace, each said apart, the workspace with its own
# token.
echo 'a machine that joined a workspace'
export HOME="$WORK/home"
mkdir -p "$HOME/.antumbra/workspaces/team.example__ws_crew"
printf 'team-token\n' > "$HOME/.antumbra/workspaces/team.example__ws_crew/token.txt"
jq -nc --arg t "$HOME/.antumbra/workspaces/team.example__ws_crew/token.txt" \
    '{ url: "https://team.example", workspace: "ws:crew", user: "user:bodil", token_file: $t }' \
    > "$HOME/.antumbra/workspaces/team.example__ws_crew/workspace.json"
printf '%s' '{"memories":[{"id":"memory:t","network":"bank","content":"the deploy runs in two steps"}]}' > "$WORK/answer.json.team"
out="$(run 'how do we deploy')"
ctx="$(printf '%s' "$out" | jq -r '.hookSpecificOutput.additionalContext')"
printf '%s' "$ctx" | grep -q 'memory:a: keep it plain'
check 'recalls from its own store' $?
printf '%s' "$ctx" | grep -q 'From the workspace ws:crew'
check 'and from the workspace, named' $?
printf '%s' "$ctx" | grep -q 'memory:t: the deploy runs in two steps'
check "with the workspace's memories under it" $?
own_at="$(printf '%s' "$ctx" | grep -n 'memory:a' | cut -d: -f1)"
team_at="$(printf '%s' "$ctx" | grep -n 'From the workspace' | cut -d: -f1)"
test "$own_at" -lt "$team_at"
check 'your own first' $?
grep -qx 'header = "Authorization: Bearer team-token"' "$CURL_CONFIG.team"
check "asks the workspace with the workspace's token" $?
grep -qx 'header = "Authorization: Bearer test-token"' "$CURL_CONFIG"
check 'and your own store with your own' $?
! grep -q 'Bearer' "$CURL_ARGS.team"
check "keeps that bearer off curl's command line too" $?

echo 'a workspace with nothing to say'
printf '%s' '{"memories":[]}' > "$WORK/answer.json.team"
ctx="$(run 'how do we deploy' | jq -r '.hookSpecificOutput.additionalContext')"
! printf '%s' "$ctx" | grep -q 'From the workspace'
check 'is not mentioned' $?

echo 'your own store with nothing to say'
printf '%s' '{"memories":[]}' > "$WORK/answer.json"
printf '%s' '{"memories":[{"id":"memory:t","network":"bank","content":"the deploy runs in two steps"}]}' > "$WORK/answer.json.team"
ctx="$(run 'how do we deploy' | jq -r '.hookSpecificOutput.additionalContext')"
printf '%s' "$ctx" | grep -q 'memory:t: the deploy runs in two steps'
check "still gives the workspace's" $?

echo 'told to ask only your own'
rm -f "$CURL_ARGS.team"
ANTUMBRA_RECALL_WORKSPACES=0 run 'how do we deploy' > /dev/null
test ! -e "$CURL_ARGS.team"
check 'asks the workspace nothing' $?

exit "$FAILED"
