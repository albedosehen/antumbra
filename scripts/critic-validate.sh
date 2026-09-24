#!/usr/bin/env bash
# Train a critic on the verifier's verdicts and read it (ADR-0022 S-2), on the
# GPU host, over one workbench skill. Detached; logs to $LOG
# (/tmp/critic-validate.log by default).
#
# The base model answers every task of the skill SAMPLES times under each of
# three seeds, and the authored judge labels every answer.
# - A critic learns the first set's verdicts on the default partition's
#   visible tasks and is read on its withheld ones, raw and recalibrated.
# - A twin learns the second set the same way.
# - The critic is read on the third set, which neither has seen, per skill:
#   calibration error, agreement, its rank correlation with the verifier
#   (what scales its influence in training), and its agreement with the twin.
#
# Expects a source tarball at /tmp/antumbra-$SHA.tar.gz, as kusko-deploy.sh does:
#   git archive --format=tar.gz -o /tmp/antumbra-$(git rev-parse --short HEAD).tar.gz HEAD
#   scp /tmp/antumbra-<sha>.tar.gz scripts/critic-validate.sh <host>:/tmp/
#   ssh <host> 'nohup bash /tmp/critic-validate.sh <sha> >/dev/null 2>&1 &'
#
# Each run gets a fresh directory under ~/antumbra-critic-runs. Nothing here
# touches the deploy checkout, the production database or the running
# services. Knobs: SKILL (default strings), SAMPLES (16), ROUNDS (2), LOG.

export PATH=$PATH:/run/current-system/sw/bin:/run/wrappers/bin
SHA="$1"
SKILL="${SKILL:-strings}"
SAMPLES="${SAMPLES:-16}"
ROUNDS="${ROUNDS:-2}"
LOG="${LOG:-/tmp/critic-validate.log}"
SRC="$HOME/antumbra-search-src/$SHA"
RUN="$HOME/antumbra-critic-runs/$(date -u +%Y%m%dT%H%M%SZ)-$SHA"
exec >"$LOG" 2>&1
set -euo pipefail
echo "== critic $SHA start $(date -u +%FT%TZ): $SKILL, $SAMPLES samples, $ROUNDS rounds"
echo "== run directory $RUN"

rm -rf "$SRC"
mkdir -p "$SRC" "$RUN"
tar -xzf "/tmp/antumbra-$SHA.tar.gz" -C "$SRC"
cd "$SRC"

echo "== build antumbra-calibrate:$SHA"
docker build -f docker/Dockerfile.cuda --target calibrate -t "antumbra-calibrate:$SHA" . 2>&1 | tail -20
chmod 777 "$RUN"

CORPUS="/build/corpora/workbench/$SKILL.json"
antumbra() {
    docker run --rm --device nvidia.com/gpu=all \
        -v antumbra-gpu-test-weights:/weights \
        -v "$RUN:/reports" \
        "antumbra-calibrate:$SHA" "$@"
}

for n in 1 2 3; do
    echo "== answers, set $n $(date -u +%FT%TZ)"
    antumbra eval --corpus "$CORPUS" --samples "$SAMPLES" --seed "$n" \
        --completions "/reports/answers-$n.json" --report "/reports/eval-$n.json"
done
echo "== train on set 1 $(date -u +%FT%TZ)"
antumbra critic train --corpus "$CORPUS" --completions /reports/answers-1.json \
    --out /reports/critic.safetensors --rounds "$ROUNDS"
echo "== train the twin on set 2 $(date -u +%FT%TZ)"
antumbra critic train --corpus "$CORPUS" --completions /reports/answers-2.json \
    --out /reports/twin.safetensors --rounds "$ROUNDS"
echo "== read on set 3 $(date -u +%FT%TZ)"
antumbra critic measure --adapter /reports/critic.safetensors --corpus "$CORPUS" \
    --completions /reports/answers-3.json --twin /reports/twin.safetensors
echo "== critic $SHA end $(date -u +%FT%TZ) OK"
