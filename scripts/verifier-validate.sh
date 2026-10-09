#!/usr/bin/env bash
# Run verifier synthesis and the trust protocol end to end on
# the GPU host, over one workbench skill. Detached; logs to $LOG
# (/tmp/verifier-validate.log by default).
#
# The model proposes the inputs of a differential check for every task, the
# task's reference computes the outputs, and each check is proposed to a
# namespace of its own. The checks are measured on cases the authored judge
# labeled: the model's own completions, the generator's forgeries, and
# mutants of the reference. Then they are measured again on a second,
# independently seeded set of completions, and challenged on it. A check that
# was trusted must hold its bound on the second set.
#
# Expects a source tarball at /tmp/antumbra-$SHA.tar.gz, as kusko-deploy.sh does:
#   git archive --format=tar.gz -o /tmp/antumbra-$(git rev-parse --short HEAD).tar.gz HEAD
#   scp /tmp/antumbra-<sha>.tar.gz scripts/verifier-validate.sh <host>:/tmp/
#   ssh <host> 'nohup bash /tmp/verifier-validate.sh <sha> >/dev/null 2>&1 &'
#
# Each run gets a fresh directory under ~/antumbra-verifier-runs with its own
# file store. Nothing here touches the deploy checkout, the production
# database or the running services. Knobs: SKILL (a workbench skill, default
# strings), SAMPLES (completions per task per set, 32), INPUTS (inputs asked
# for per answer, 12), DRAWS (answers per task, 2), and LOG.

export PATH=$PATH:/run/current-system/sw/bin:/run/wrappers/bin
SHA="$1"
SKILL="${SKILL:-strings}"
SAMPLES="${SAMPLES:-32}"
INPUTS="${INPUTS:-12}"
DRAWS="${DRAWS:-2}"
LOG="${LOG:-/tmp/verifier-validate.log}"
SRC="$HOME/antumbra-search-src/$SHA"
RUN="$HOME/antumbra-verifier-runs/$(date -u +%Y%m%dT%H%M%SZ)-$SHA"
exec >"$LOG" 2>&1
set -euo pipefail
echo "== verifiers $SHA start $(date -u +%FT%TZ): $SKILL, $SAMPLES samples, $INPUTS inputs x $DRAWS draws"
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

CORPUS="/build/corpora/workbench/$SKILL.json"
antumbra() {
    docker run --rm --device nvidia.com/gpu=all \
        -v antumbra-gpu-test-weights:/weights \
        -v "$RUN:/reports" \
        "antumbra-calibrate:$SHA" \
        --url surrealkv:///reports/store.skv "$@"
}
workbench() {
    docker run --rm -v "$RUN:/reports" --entrypoint python "antumbra-calibrate:$SHA" \
        "/build/corpora/workbench/$1" "${@:2}"
}

antumbra migrate
echo "== synthesize $(date -u +%FT%TZ)"
antumbra verifier synthesize --corpus "$CORPUS" --inputs "$INPUTS" --draws "$DRAWS" \
    --out /reports/proposals.json
workbench synthesize.py --corpus "$CORPUS" --proposals /reports/proposals.json \
    --specs /reports/specs.json --artifacts /reports/artifacts.json
antumbra verifier propose --batch /reports/specs.json

for n in 1 2; do
    echo "== completions, set $n $(date -u +%FT%TZ)"
    antumbra eval --corpus "$CORPUS" --samples "$SAMPLES" --seed "$n" \
        --completions "/reports/samples-$n.json" --report "/reports/eval-$n.json"
    antumbra verifier cases --corpus "$CORPUS" --completions "/reports/samples-$n.json" \
        --completions /reports/artifacts.json --out "/reports/cases-$n.json"
    echo "== measure on set $n $(date -u +%FT%TZ)"
    antumbra verifier measure --domain "$SKILL" --cases "/reports/cases-$n.json"
done
echo "== challenge on set 2 $(date -u +%FT%TZ)"
antumbra verifier challenge --domain "$SKILL" --cases /reports/cases-2.json
antumbra verifier list --domain "$SKILL"
echo "== verifiers $SHA end $(date -u +%FT%TZ) OK"
