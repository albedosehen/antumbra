#!/usr/bin/env bash
# Tests for strip-attribution.sh over the cases in strip-attribution.cases,
# the list strip-attribution.ps1 is held to as well.
#
#   bash scripts/hooks/tests/strip-attribution.sh
set -u
HOOK="${ANTUMBRA_TEST_HOOK:-$(dirname "$0")/../strip-attribution.sh}"
CASES="$(dirname "$0")/strip-attribution.cases"
FAILED=0
SEEN=0

check() {
    if [ "$2" = 0 ]; then echo "  ok    $1"; else echo "  FAIL  $1"; FAILED=1; fi
}

tab=$'\t'
while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in '#'*|'') continue ;; esac
    verdict=${line%%"$tab"*}
    input=${line#*"$tab"}
    SEEN=$((SEEN + 1))
    out="$(printf '%s' "$input" | bash "$HOOK")"
    case "$verdict" in
        deny)
            printf '%s' "$out" | jq -e '.hookSpecificOutput.hookEventName == "PreToolUse"
                and .hookSpecificOutput.permissionDecision == "deny"
                and (.hookSpecificOutput.permissionDecisionReason | startswith("Model-vendor attribution is not allowed"))' > /dev/null
            check "denies  $input" $? ;;
        allow)
            test -z "$out"
            check "allows  $input" $? ;;
        *)
            check "a verdict of deny or allow: $line" 1 ;;
    esac
done < "$CASES"

test "$SEEN" -ge 10
check "read the cases ($SEEN)" $?

exit "$FAILED"
