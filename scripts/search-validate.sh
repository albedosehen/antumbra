#!/usr/bin/env bash
# Run the recipe search (ADR-0022 S-1) end to end on the GPU host: a searched,
# held-out training run over a workbench corpus, cohort per generation,
# graduation on fresh-seed re-measurement. Detached; logs to
# /tmp/search-validate.log.
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
# It works in its own directory with its own file store, never the deploy
# checkout or the production database, and it does not touch the running
# services. Knobs: CORPUS (a workbench skill, default sequences), GENERATIONS
# (2), COHORT (3), SAMPLES (4), ROUNDS (2).

export PATH=$PATH:/run/current-system/sw/bin:/run/wrappers/bin
SHA="$1"
CORPUS="${CORPUS:-sequences}"
GENERATIONS="${GENERATIONS:-2}"
COHORT="${COHORT:-3}"
SAMPLES="${SAMPLES:-4}"
ROUNDS="${ROUNDS:-2}"
DIR="$HOME/antumbra-search"
exec >/tmp/search-validate.log 2>&1
set -euo pipefail
echo "== search $SHA start $(date -u +%FT%TZ): $CORPUS, $GENERATIONS generation(s) of $COHORT, $SAMPLES samples, $ROUNDS rounds"

rm -rf "$DIR" && mkdir -p "$DIR/run"
tar -xzf "/tmp/antumbra-$SHA.tar.gz" -C "$DIR"
cd "$DIR"

echo "== build antumbra-calibrate:$SHA"
docker build -f docker/Dockerfile.cuda --target calibrate -t "antumbra-calibrate:$SHA" . 2>&1 | tail -20
chmod 777 "$DIR/run"

echo "== train $(date -u +%FT%TZ)"
docker run --rm --device nvidia.com/gpu=all \
    -v antumbra-gpu-test-weights:/weights \
    -v "$DIR/run:/reports" \
    "antumbra-calibrate:$SHA" \
    --url surrealkv:///reports/store.skv \
    train --corpus "/build/corpora/workbench/$CORPUS.json" --run "search:$SHA" \
    --generations "$GENERATIONS" --samples "$SAMPLES" --rounds "$ROUNDS" \
    --holdout --search --cohort "$COHORT"

echo "== search $SHA end $(date -u +%FT%TZ) OK"
