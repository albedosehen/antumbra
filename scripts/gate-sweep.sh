#!/usr/bin/env bash
# Sweep the margin gate's threshold over a trained population (ADR-0024 D-1,
# Validation 1) on the GPU host. Detached; logs to $LOG (/tmp/gate-sweep.log
# by default).
#
# $2 is a run directory a training run left behind (search-validate.sh writes
# them under ~/antumbra-search-runs), holding its store and its adapters. Every
# task the default partition withholds is routed over that population with the
# threshold out of the way, its expert and the base model are both scored under
# the same seeds, and the threshold is swept offline. The curve's best point is
# the bar a typed gate must clear; the shipped threshold is printed beside it.
# The outcomes land in the run directory as gate-sweep.json.
#
# Expects a source tarball at /tmp/antumbra-$SHA.tar.gz, as kusko-deploy.sh does:
#   git archive --format=tar.gz -o /tmp/antumbra-$(git rev-parse --short HEAD).tar.gz HEAD
#   scp /tmp/antumbra-<sha>.tar.gz scripts/gate-sweep.sh <host>:/tmp/
#   ssh <host> 'nohup bash /tmp/gate-sweep.sh <sha> <run directory> >/dev/null 2>&1 &'
#
# It reads the run's store and writes one file beside it; nothing here touches
# the deploy checkout, the production database or the running services.
# Knobs: CORPUS (the workbench file the run trained on, default all), SEEDS (2),
# SAMPLES (4), LOG.

export PATH=$PATH:/run/current-system/sw/bin:/run/wrappers/bin
SHA="$1"
RUN="$2"
CORPUS="${CORPUS:-all}"
SEEDS="${SEEDS:-2}"
SAMPLES="${SAMPLES:-4}"
LOG="${LOG:-/tmp/gate-sweep.log}"
SRC="$HOME/antumbra-search-src/$SHA"
exec >"$LOG" 2>&1
set -euo pipefail
echo "== gate sweep $SHA start $(date -u +%FT%TZ): $RUN, $CORPUS, $SEEDS seed(s) of $SAMPLES"
test -d "$RUN/adapters" || { echo "no adapters under $RUN"; exit 1; }

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
tail -20 "$BUILD_LOG"

docker run --rm --device nvidia.com/gpu=all \
    -v antumbra-gpu-test-weights:/weights \
    -v "$RUN:/reports" \
    "antumbra-calibrate:$SHA" \
    --url surrealkv:///reports/store.skv \
    gate-sweep --corpus "/build/corpora/workbench/$CORPUS.json" \
    --seeds "$SEEDS" --samples "$SAMPLES" --out /reports/gate-sweep.json
echo "== gate sweep $SHA end $(date -u +%FT%TZ) OK"
