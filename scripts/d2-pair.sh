#!/usr/bin/env bash
# Fine-tune the D-2 pair encoder on the GPU host and score it against the
# control (ADR-0024 D-2): `pair_encoder::tests::d2_pair_encoder_against_the_control`
# in the d2-probe image, over the label file scripts/d2-labels.sh built.
# Detached; logs to $LOG (/tmp/d2-pair.log by default).
#
#   git archive --format=tar.gz -o /tmp/antumbra-<sha>.tar.gz HEAD
#   scp /tmp/antumbra-<sha>.tar.gz scripts/d2-pair.sh <host>:/tmp/
#   ssh <host> 'nohup bash /tmp/d2-pair.sh <sha> >/dev/null 2>&1 &'
#
# Knobs, passed to the test: LABELS (the label file on the host,
# ~/antumbra-d2/labels/d2-labels.json), ENCODER (answerdotai/ModernBERT-base),
# MAX_TOKENS (1024), EPOCHS (3), BATCH (4), LR (3e-5), SEED (0).
set -uo pipefail

SHA="$1"
LOG="${LOG:-/tmp/d2-pair.log}"
LABELS="${LABELS:-$HOME/antumbra-d2/labels/d2-labels.json}"
ENCODER="${ENCODER:-answerdotai/ModernBERT-base}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
EPOCHS="${EPOCHS:-3}"
BATCH="${BATCH:-4}"
LR="${LR:-3e-5}"
SEED="${SEED:-0}"
exec >"$LOG" 2>&1

echo "== d2 pair encoder $SHA start $(date -u +%FT%TZ): $ENCODER, $MAX_TOKENS tokens, $EPOCHS epoch(s), batch $BATCH, lr $LR, seed $SEED"
if [ ! -s "$LABELS" ]; then
    echo "== no label file at $LABELS; build one with scripts/d2-labels.sh"
    exit 1
fi

SRC="$HOME/antumbra-d2-$SHA"
rm -rf "$SRC" && mkdir -p "$SRC"
tar -xzf "/tmp/antumbra-$SHA.tar.gz" -C "$SRC"
cd "$SRC"

echo "== build antumbra-d2:$SHA"
if ! docker build -f docker/Dockerfile.cuda --target d2-probe -t "antumbra-d2:$SHA" . >/tmp/d2-pair-build.log 2>&1; then
    tail -30 /tmp/d2-pair-build.log
    echo "== build failed"
    exit 1
fi

echo "== fine-tune $(date -u +%FT%TZ)"
docker run --rm --device nvidia.com/gpu=all \
    -v antumbra-gpu-test-weights:/weights \
    -v "$LABELS:/labels/d2-labels.json:ro" \
    -e ANTUMBRA_D2_ENCODER="$ENCODER" \
    -e ANTUMBRA_D2_MAX_TOKENS="$MAX_TOKENS" \
    -e ANTUMBRA_D2_EPOCHS="$EPOCHS" \
    -e ANTUMBRA_D2_BATCH="$BATCH" \
    -e ANTUMBRA_D2_LR="$LR" \
    -e ANTUMBRA_D2_SEED="$SEED" \
    --entrypoint /usr/local/bin/antumbra-d2 \
    "antumbra-d2:$SHA" --ignored --nocapture --test-threads=1 d2_pair
status=$?
rm -rf "$SRC"
echo "== d2 pair encoder $SHA end $(date -u +%FT%TZ) exit $status"
exit $status
