#!/usr/bin/env bash
# Train under synthesized verifiers on the GPU host, with the loop rechecking
# them against their anchors every generation. Detached; logs to
# $LOG (/tmp/verifier-train.log by default).
#
# $2 is a directory whose store.skv holds a measured namespace: the one
# verifier-validate.sh or verifier-remeasure.sh left behind, with the
# synthesized checks trusted and each task's authored verifier registered as
# its anchor. The store is copied, never written in place, so the measurement
# it records stays as it was. The skill's corpus is then rewritten so each task
# with a check that grants reward names it (`antumbra verifier name`), and
# `train` runs on that corpus. Every generation prints what the recheck found
# for each check, and the namespace is listed before and after.
#
# Trust lapses seven days after the measurement that granted it, so the
# namespace has to be that recent.
#
# Expects a source tarball at /tmp/antumbra-$SHA.tar.gz, as kusko-deploy.sh does:
#   git archive --format=tar.gz -o /tmp/antumbra-$(git rev-parse --short HEAD).tar.gz HEAD
#   scp /tmp/antumbra-<sha>.tar.gz scripts/verifier-train.sh <host>:/tmp/
#   ssh <host> 'nohup bash /tmp/verifier-train.sh <sha> <directory> >/dev/null 2>&1 &'
#
# The answers verifier-validate.sh built to be wrong (its artifacts.json) come
# along with the store when the directory has them, and every recheck counts
# them as known-bad evidence (`train --recheck-artifacts`).
#
# Each run gets a fresh directory under ~/antumbra-verifier-runs. Nothing here
# touches the deploy checkout, the production database or the running
# services. Knobs: SKILL (strings), GENERATIONS (2), SAMPLES (8), ROUNDS (2),
# ARGS (more train flags) and LOG.

export PATH=$PATH:/run/current-system/sw/bin:/run/wrappers/bin
SHA="$1"
FROM="$2"
SKILL="${SKILL:-strings}"
GENERATIONS="${GENERATIONS:-2}"
SAMPLES="${SAMPLES:-8}"
ROUNDS="${ROUNDS:-2}"
ARGS="${ARGS:-}"
LOG="${LOG:-/tmp/verifier-train.log}"
SRC="$HOME/antumbra-search-src/$SHA"
RUN="$HOME/antumbra-verifier-runs/train-$(date -u +%Y%m%dT%H%M%SZ)-$SHA"
exec >"$LOG" 2>&1
set -euo pipefail
echo "== verifier train $SHA start $(date -u +%FT%TZ): $SKILL from $FROM, $GENERATIONS generation(s)"
echo "== run directory $RUN"
test -d "$FROM/store.skv" || { echo "no store.skv in $FROM"; exit 1; }
mkdir -p "$RUN"
chmod 777 "$RUN"

rm -rf "$SRC"
mkdir -p "$SRC"
tar -xzf "/tmp/antumbra-$SHA.tar.gz" -C "$SRC"
# git archive stamps every file with its commit's time, and the image build's
# target cache is shared across commits, so a commit older than the last build
# would be compiled against that build's artifacts. Stamp the sources now, so
# cargo rebuilds what differs.
find "$SRC" -type f -exec touch {} +
cd "$SRC"
echo "== build antumbra-calibrate:$SHA"
# The whole build goes to a log beside the others, so a failure shows its
# compiler errors here rather than the last lines of a stack of layers.
BUILD_LOG="/tmp/antumbra-build-$SHA.log"
if ! docker build -f docker/Dockerfile.cuda --target calibrate -t "antumbra-calibrate:$SHA" . >"$BUILD_LOG" 2>&1; then
    echo "== build failed; its errors follow, and the whole log is $BUILD_LOG"
    grep -E "error(\[E[0-9]+\])?:" -A8 "$BUILD_LOG" | head -80
    exit 1
fi
tail -5 "$BUILD_LOG"

antumbra() {
    docker run --rm --device nvidia.com/gpu=all \
        -v antumbra-gpu-test-weights:/weights \
        -v "$RUN:/reports" \
        "antumbra-calibrate:$SHA" \
        --url surrealkv:///reports/store.skv "$@"
}

# The container wrote the store as its own user, and part of it is not
# readable from the host, so the copy is made by that user.
docker run --rm -v "$FROM:/from:ro" -v "$RUN:/to" --entrypoint python "antumbra-calibrate:$SHA" \
    -c "import os, shutil; shutil.copytree('/from/store.skv', '/to/store.skv'); \
os.path.exists('/from/artifacts.json') and shutil.copy('/from/artifacts.json', '/to/artifacts.json')"
if [ -e "$RUN/artifacts.json" ]; then
    ARGS="$ARGS --recheck-artifacts /reports/artifacts.json"
fi

antumbra migrate
echo "== the namespace before"
antumbra verifier list --domain "$SKILL"
antumbra verifier name --corpus "/build/corpora/workbench/$SKILL.json" --domain "$SKILL" \
    --out /reports/named.json

echo "== train $(date -u +%FT%TZ)"
# shellcheck disable=SC2086
antumbra train --corpus /reports/named.json --run "verifier:$SHA" \
    --generations "$GENERATIONS" --samples "$SAMPLES" --rounds "$ROUNDS" --holdout $ARGS

echo "== the namespace after"
antumbra verifier list --domain "$SKILL"
echo "== verifier train $SHA end $(date -u +%FT%TZ) OK"
