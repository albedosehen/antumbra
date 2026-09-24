#!/usr/bin/env bash
# The critic's own test (ADR-0022 S-2's validation): GRPO with a critic
# shaping its advantages against GRPO on verifier-only reward, same corpus,
# same generations, one arm after the other on the GPU host. Detached; the
# summary logs to $LOG (/tmp/critic-compare.log by default), each arm to
# /tmp/critic-compare-<arm>.log.
#
# $2 is a critic adapter (critic-validate.sh leaves one in its run directory).
# Both arms run with --holdout, so each generation reports its held-out pass
# rate beside the fitness it trained for; the record's test is whether the
# critic reaches graduation on fewer samples, and graduation reads verifier
# bits in both arms.
#
# Expects a source tarball at /tmp/antumbra-$SHA.tar.gz, as kusko-deploy.sh does:
#   git archive --format=tar.gz -o /tmp/antumbra-$(git rev-parse --short HEAD).tar.gz HEAD
#   scp /tmp/antumbra-<sha>.tar.gz scripts/critic-compare.sh <host>:/tmp/
#   ssh <host> 'nohup bash /tmp/critic-compare.sh <sha> <critic adapter> >/dev/null 2>&1 &'
#
# Each arm gets a fresh directory under ~/antumbra-critic-runs with its own
# store. Nothing here touches the deploy checkout, the production database or
# the running services. Knobs: SKILL (strings), GENERATIONS (3), SAMPLES (4),
# ROUNDS (2), WEIGHT (the critic's weight, 0.5), LOG.

export PATH=$PATH:/run/current-system/sw/bin:/run/wrappers/bin
SHA="$1"
CRITIC="$2"
SKILL="${SKILL:-strings}"
GENERATIONS="${GENERATIONS:-3}"
SAMPLES="${SAMPLES:-4}"
ROUNDS="${ROUNDS:-2}"
WEIGHT="${WEIGHT:-0.5}"
LOG="${LOG:-/tmp/critic-compare.log}"
SRC="$HOME/antumbra-search-src/$SHA"
exec >"$LOG" 2>&1
set -uo pipefail
echo "== critic comparison $SHA start $(date -u +%FT%TZ): $SKILL, $GENERATIONS generation(s), critic $CRITIC at $WEIGHT"
test -f "$CRITIC" || { echo "no critic adapter at $CRITIC"; exit 1; }

rm -rf "$SRC"
mkdir -p "$SRC"
tar -xzf "/tmp/antumbra-$SHA.tar.gz" -C "$SRC"
cd "$SRC"
echo "== build antumbra-calibrate:$SHA"
docker build -f docker/Dockerfile.cuda --target calibrate -t "antumbra-calibrate:$SHA" . 2>&1 | tail -5

for arm in verifier-only critic; do
    run="$HOME/antumbra-critic-runs/$(date -u +%Y%m%dT%H%M%SZ)-$SHA-$arm"
    mkdir -p "$run" && chmod 777 "$run"
    extra=""
    if [ "$arm" = "critic" ]; then
        cp "$CRITIC" "$run/critic.safetensors"
        extra="--critic /reports/critic.safetensors --critic-weight $WEIGHT"
    fi
    start=$(date -u +%s)
    docker run --rm --device nvidia.com/gpu=all \
        -v antumbra-gpu-test-weights:/weights \
        -v "$run:/reports" \
        "antumbra-calibrate:$SHA" \
        --url surrealkv:///reports/store.skv \
        train --algo grpo --corpus "/build/corpora/workbench/$SKILL.json" \
        --run "critic:$arm" --generations "$GENERATIONS" --samples "$SAMPLES" \
        --rounds "$ROUNDS" --holdout $extra >"/tmp/critic-compare-$arm.log" 2>&1
    status=$?
    minutes=$(( ($(date -u +%s) - start) / 60 ))
    graduated=$(grep -c "graduated=true" "/tmp/critic-compare-$arm.log" || true)
    echo "== $arm: exit $status, $minutes min, $graduated graduation(s), run $run"
    grep -E "^gen |held-out gap" "/tmp/critic-compare-$arm.log" | tail -12
done
echo "== critic comparison $SHA end $(date -u +%FT%TZ)"
