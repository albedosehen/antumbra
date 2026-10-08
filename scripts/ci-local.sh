#!/usr/bin/env bash
# Run what .github/workflows/ci.yml runs, here, since the hosted runners are not
# available to this repository. Every job of the workflow, in its order, with
# its commands; a summary at the end, and a non-zero exit when any job failed.
#
#   bash scripts/ci-local.sh                 # every job
#   bash scripts/ci-local.sh check hooks     # just these
#
# Jobs: check, models-build, hooks, audit, control-server. A job whose
# tool is missing (jq for hooks, cargo-audit for audit) is reported SKIPPED
# with the reason, never passed, and a job run in part is reported PART. The
# hooks job has a POSIX half and a PowerShell half, and each runs where it
# runs for real: the POSIX session-start test needs a POSIX host (it is
# skipped under Git Bash on Windows) and the PowerShell one needs pwsh. The
# control server is its own workspace with a private dependency fetched over
# ssh, so it needs that access.
#
# The check job builds with CARGO_PROFILE_DEV_DEBUG=line-tables-only, as the
# workflow does: panics keep file and line, and target/ stays a fraction of the
# size full debug info makes it.

set -uo pipefail
cd "$(dirname "$0")/.."

# Four compile jobs unless told otherwise: one rustc per core runs a
# workstation out of memory on surrealdb-core ("memory allocation of 2097152
# bytes failed"), which a hosted runner never meets with its two cores.
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-4}"

JOBS=("$@")
if [ ${#JOBS[@]} -eq 0 ]; then
    JOBS=(check models-build hooks audit control-server)
fi
declare -a SUMMARY=()
FAILED=0

record() {
    SUMMARY+=("$1  $2")
    if [ "$1" = "FAIL" ]; then FAILED=1; fi
}

step() {
    echo "== $*"
    "$@"
}

job_check() {
    (
        export CARGO_PROFILE_DEV_DEBUG=line-tables-only
        step cargo fmt --all --check &&
            step cargo clippy --workspace --all-targets -- -D warnings &&
            step cargo test --workspace
    )
}

job_models_build() {
    step cargo clippy -p antumbra-train -p antumbra-serve -p antumbra-mcp -p antumbra-cli \
        --features models --all-targets -- -D warnings
}

job_hooks() {
    if ! command -v jq >/dev/null; then
        echo "SKIPPED: jq is not installed"
        return 2
    fi
    local f out again deny clean ctx
    for f in scripts/hooks/*.sh; do bash -n "$f" || return 1; done
    # Fresh session ids each run: the capture hook remembers the sessions it
    # has fired for, and a hosted runner starts with none.
    local one="ci-$-$RANDOM" two="ci-$-$RANDOM-2"
    out=$(printf '{"session_id":"%s","hook_event_name":"Stop"}' "$one" | bash scripts/hooks/antumbra-capture.sh)
    echo "$out" | jq -e '.decision == "block"' >/dev/null || { echo "capture: no block decision"; return 1; }
    printf '{"session_id":"%s","hook_event_name":"Stop"}' "$two" | bash scripts/hooks/antumbra-capture.sh >/dev/null
    again=$(printf '{"session_id":"%s","hook_event_name":"Stop"}' "$two" | bash scripts/hooks/antumbra-capture.sh)
    test -z "$again" || { echo "capture: fired twice for one turn"; return 1; }
    deny=$(printf '%s' '{"tool_input":{"command":"git commit -m x --trailer Co-Authored-By:Claude"}}' | bash scripts/hooks/strip-attribution.sh)
    echo "$deny" | jq -e '.hookSpecificOutput.permissionDecision == "deny"' >/dev/null || { echo "strip-attribution: no deny"; return 1; }
    clean=$(printf '%s' '{"tool_input":{"command":"git commit -m clean"}}' | bash scripts/hooks/strip-attribution.sh)
    test -z "$clean" || { echo "strip-attribution: blocked a clean command"; return 1; }
    deny=$(printf '%s' '{"tool_name":"PowerShell","tool_input":{"command":"gh pr create --body \"x\n\nco-authored-by: GPT-5\""}}' | bash scripts/hooks/strip-attribution.sh)
    echo "$deny" | jq -e '.hookSpecificOutput.permissionDecision == "deny"' >/dev/null || { echo "strip-attribution: missed a lowercase trailer"; return 1; }
    clean=$(printf '%s' '{"tool_input":{"command":"git commit -m \"plain \\\"quoted\\\" message\"","description":"no Co-Authored-By: Claude here"}}' | bash scripts/hooks/strip-attribution.sh)
    test -z "$clean" || { echo "strip-attribution: judged the description, not the command"; return 1; }
    ctx=$(printf '%s' '{}' | ANTUMBRA_URL=http://127.0.0.1:1 bash scripts/hooks/antumbra-session-start.sh)
    echo "$ctx" | jq -e '.hookSpecificOutput.hookEventName == "SessionStart"' >/dev/null || { echo "session-start: no context"; return 1; }
    local part=0
    # The POSIX hook detaches its report the way a POSIX host does, which Git
    # Bash on Windows does not; there the PowerShell hook is the one that runs.
    case "$(uname -s)" in
        MINGW* | MSYS* | CYGWIN*)
            echo "SKIPPED (the .sh session-start test): run it on a POSIX host"
            part=1
            ;;
        *) step bash scripts/hooks/tests/session-start.sh || return 1 ;;
    esac
    step bash scripts/hooks/tests/prompt-recall.sh || return 1
    if command -v pwsh >/dev/null; then
        step pwsh -NoProfile -File scripts/hooks/tests/session-start.ps1 || return 1
        step pwsh -NoProfile -File scripts/hooks/tests/prompt-recall.ps1 || return 1
    else
        echo "SKIPPED (the .ps1 session-start and prompt-recall tests): pwsh is not installed"
        part=1
    fi
    [ "$part" = 0 ] || return 3
}

job_audit() {
    if ! cargo audit --version >/dev/null 2>&1; then
        echo "SKIPPED: cargo-audit is not installed (cargo install cargo-audit --locked)"
        return 2
    fi
    step cargo audit
}

job_control_server() {
    (
        export CARGO_PROFILE_DEV_DEBUG=line-tables-only
        cd crates/antumbra-control-server &&
            step cargo fmt --all --check &&
            step cargo clippy --all-targets -- -D warnings &&
            step cargo clippy --all-targets --features contract -- -D warnings &&
            step cargo test --features contract
    )
}

for job in "${JOBS[@]}"; do
    echo
    echo "#### $job"
    start=$(date +%s)
    case "$job" in
        check) job_check ;;
        models-build) job_models_build ;;
        hooks) job_hooks ;;
        audit) job_audit ;;
        control-server) job_control_server ;;
        *) echo "unknown job $job"; false ;;
    esac
    status=$?
    took=$(($(date +%s) - start))
    case "$status" in
        0) record PASS "$job (${took}s)" ;;
        2) record SKIP "$job (${took}s)" ;;
        3) record PART "$job (${took}s): part of it skipped, see above" ;;
        *) record FAIL "$job (${took}s)" ;;
    esac
done

echo
echo "#### summary"
printf '%s\n' "${SUMMARY[@]}"
exit "$FAILED"
