#!/usr/bin/env bash
# Run the recipe search end to end on the GPU host: a searched,
# held-out training run over a workbench corpus, cohort per generation,
# graduation on fresh-seed re-measurement. Detached; logs to $LOG
# (/tmp/search-validate.log by default).
#
# The search is tested on CPU against fakes; this is where the real trainer
# takes the recipes (learning rate, batch size), where seeded sampling draws the
# re-measurements, and where a whole generation's cost shows up.
#
# Expects a source tarball at /tmp/antumbra-$SHA.tar.gz, as kusko-deploy.sh does:
#   git archive --format=tar.gz -o /tmp/antumbra-$(git rev-parse --short HEAD).tar.gz HEAD
#   scp /tmp/antumbra-<sha>.tar.gz scripts/search-validate.sh <host>:/tmp/
#   ssh <host> 'nohup bash /tmp/search-validate.sh <sha> >/dev/null 2>&1 &'
#
# Each run gets a directory of its own under ~/antumbra-search-runs, with its
# own file store, and a trace of the card's memory every second in gpu-mem.log
# beside it. A fresh directory rather than a cleaned one: the
# container writes as its own user, the host cannot delete what it wrote, and a
# cleanup that silently failed once left a stale store behind for the next run
# to resume. Nothing here touches the deploy checkout, the production database
# or the running services. Knobs: CORPUS (a workbench skill, default sequences),
# GENERATIONS (2), COHORT (3), SAMPLES (4), ROUNDS (2), SEARCH (1; 0 trains one
# shadow a generation without the recipe search), ARGS (more train flags,
# for example "--contribution-every 1"), and LOG (where the run logs).

export PATH=$PATH:/run/current-system/sw/bin:/run/wrappers/bin
SHA="$1"
CORPUS="${CORPUS:-sequences}"
GENERATIONS="${GENERATIONS:-2}"
COHORT="${COHORT:-3}"
SAMPLES="${SAMPLES:-4}"
ROUNDS="${ROUNDS:-2}"
SEARCH="${SEARCH:-1}"
ARGS="${ARGS:-}"
LOG="${LOG:-/tmp/search-validate.log}"
SRC="$HOME/antumbra-search-src/$SHA"
RUN="$HOME/antumbra-search-runs/$(date -u +%Y%m%dT%H%M%SZ)-$SHA"
exec >"$LOG" 2>&1
set -euo pipefail
echo "== search $SHA start $(date -u +%FT%TZ): $CORPUS, $GENERATIONS generation(s) of $COHORT, $SAMPLES samples, $ROUNDS rounds, search=$SEARCH, args: $ARGS"
echo "== run directory $RUN"

rm -rf "$SRC"
mkdir -p "$SRC" "$RUN"
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
tail -20 "$BUILD_LOG"
chmod 777 "$RUN"

# The card's memory while the run trains, sampled until it ends. One sampler
# process looping on its own, so the exit trap stops all of it.
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 1000 \
    >"$RUN/gpu-mem.log" 2>&1 &
SAMPLER=$!
trap 'kill "$SAMPLER" 2>/dev/null || true' EXIT

SEARCH_FLAGS=""
if [ "$SEARCH" = "1" ]; then
    SEARCH_FLAGS="--search --cohort $COHORT"
fi
echo "== train $(date -u +%FT%TZ)"
docker run --rm --device nvidia.com/gpu=all \
    -v antumbra-gpu-test-weights:/weights \
    -v "$RUN:/reports" \
    "antumbra-calibrate:$SHA" \
    --url surrealkv:///reports/store.skv \
    train --corpus "/build/corpora/workbench/$CORPUS.json" --run "search:$SHA" \
    --generations "$GENERATIONS" --samples "$SAMPLES" --rounds "$ROUNDS" \
    --holdout $SEARCH_FLAGS $ARGS

echo "== peak card memory $(awk -F', ' '{print $2}' "$RUN/gpu-mem.log" | sort -n | tail -1) MiB"
echo "== search $SHA end $(date -u +%FT%TZ) OK"
