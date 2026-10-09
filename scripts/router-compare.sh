#!/usr/bin/env bash
# Judge the outcome-trained learned router against the exemplar-only one on
# held-out tasks on the GPU host. Detached; logs to $LOG
# (/tmp/router-compare.log by default).
#
# $2 is a run directory a training run left behind (grow-compare.sh writes
# them under ~/antumbra-search-runs), holding its store and its adapters. It is
# copied to <run>-router-compare first, so the run stays as it ended; the
# winners and both routers are written to the copy.
#   1. gate-outcomes: every routable expert scores the live tasks a grow run's
#      contribution measurement samples, and each task's clear winner is
#      recorded.
#   2. gate-train, then gate-sweep --learned over the withheld
#      tasks: the router as it was.
#   3. gate-train --with-outcomes, then the same sweep: the router that learned
#      the winners.
# Both sweeps score the same withheld tasks under the same seeds, and their
# outcomes land in the copy as sweep-exemplars.json and sweep-outcomes.json.
#
# Expects a source tarball at /tmp/antumbra-$SHA.tar.gz, as kusko-deploy.sh does:
#   git archive --format=tar.gz -o /tmp/antumbra-$(git rev-parse --short HEAD).tar.gz HEAD
#   scp /tmp/antumbra-<sha>.tar.gz scripts/router-compare.sh <host>:/tmp/
#   ssh <host> 'nohup bash /tmp/router-compare.sh <sha> <run directory> >/dev/null 2>&1 &'
#
# Nothing here touches the deploy checkout, the production database or the
# running services. Knobs: CORPUS (the workbench file the run trained on,
# default all), SEEDS (2), SAMPLES (4), MAX_TASKS (the live tasks labeled, 64
# as grow-compare.sh measures), LOG. Expect about an hour for the winners at
# the defaults and three quarters of one for each sweep.

export PATH=$PATH:/run/current-system/sw/bin:/run/wrappers/bin
SHA="$1"
RUN="${2%/}"
CORPUS="${CORPUS:-all}"
SEEDS="${SEEDS:-2}"
SAMPLES="${SAMPLES:-4}"
MAX_TASKS="${MAX_TASKS:-64}"
LOG="${LOG:-/tmp/router-compare.log}"
SRC="$HOME/antumbra-search-src/$SHA"
OUT="$RUN-router-compare"
exec >"$LOG" 2>&1
set -euo pipefail
echo "== router compare $SHA start $(date -u +%FT%TZ): $RUN, $CORPUS, $SEEDS seed(s) of $SAMPLES"
test -d "$RUN/adapters" || { echo "no adapters under $RUN"; exit 1; }
test ! -e "$OUT" || { echo "$OUT exists; move it aside first"; exit 1; }

rm -rf "$SRC"
mkdir -p "$SRC"
tar -xzf "/tmp/antumbra-$SHA.tar.gz" -C "$SRC"
# git archive stamps every file with its commit's time, and the image build's
# target cache is shared across commits, so stamp the sources now and cargo
# rebuilds what differs.
find "$SRC" -type f -exec touch {} +
cd "$SRC"
echo "== build antumbra-calibrate:$SHA"
BUILD_LOG="/tmp/antumbra-build-$SHA.log"
if ! docker build -f docker/Dockerfile.cuda --target calibrate -t "antumbra-calibrate:$SHA" . >"$BUILD_LOG" 2>&1; then
    echo "== build failed; its errors follow, and the whole log is $BUILD_LOG"
    grep -E "error(\[E[0-9]+\])?:" -A8 "$BUILD_LOG" | head -80
    exit 1
fi
tail -5 "$BUILD_LOG"

# The image runs as its own user (uid 65532), which owns the run's store and
# adapters but cannot create a directory beside the run, so the copy lands in
# one made here, open to it as the run directories are.
mkdir -m 777 "$OUT"
docker run --rm --entrypoint sh \
    -v "$RUN:/from:ro" -v "$OUT:/to" \
    "antumbra-calibrate:$SHA" -c 'cp -a /from/* /to/'
antumbra() {
    docker run --rm --device nvidia.com/gpu=all \
        -v antumbra-gpu-test-weights:/weights \
        -v "$OUT:/reports" \
        "antumbra-calibrate:$SHA" \
        --url surrealkv:///reports/store.skv "$@"
}
TASKS="/build/corpora/workbench/$CORPUS.json"
SCORING=(--corpus "$TASKS" --seeds "$SEEDS" --samples "$SAMPLES")

echo "== winners $(date -u +%FT%TZ)"
antumbra gate-outcomes "${SCORING[@]}" --max-tasks "$MAX_TASKS"
echo "== exemplars only $(date -u +%FT%TZ)"
antumbra gate-train
antumbra gate-sweep --learned "${SCORING[@]}" --out /reports/sweep-exemplars.json
echo "== with the winners $(date -u +%FT%TZ)"
antumbra gate-train --with-outcomes
antumbra gate-sweep --learned "${SCORING[@]}" --out /reports/sweep-outcomes.json
echo "== router compare $SHA end $(date -u +%FT%TZ) OK"
