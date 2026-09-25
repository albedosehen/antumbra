#!/usr/bin/env bash
# Measure a finished verifier run's synthesized checks again, under this
# commit's trust protocol, without the GPU (ADR-0022 S-4). Detached; logs to
# $LOG (/tmp/verifier-remeasure.log by default). It trains and samples nothing,
# so it can run beside a GPU job.
#
# $2 is a run directory verifier-validate.sh left behind. Its proposals and
# the model's two sets of answers are reused as they are; everything the
# protocol decides is done again in a fresh store beside them: the specs and
# artifacts rebuilt from the proposals, the cases labeled, set 1 measured,
# set 2 re-measured, and the challenge. So a change to the protocol can be
# judged on the same evidence as the run it changes.
#
# Expects a source tarball at /tmp/antumbra-$SHA.tar.gz, as kusko-deploy.sh does:
#   git archive --format=tar.gz -o /tmp/antumbra-$(git rev-parse --short HEAD).tar.gz HEAD
#   scp /tmp/antumbra-<sha>.tar.gz scripts/verifier-remeasure.sh <host>:/tmp/
#   ssh <host> 'nohup bash /tmp/verifier-remeasure.sh <sha> <run directory> >/dev/null 2>&1 &'
#
# Knobs: SKILL (the run's workbench skill, default strings) and LOG.

export PATH=$PATH:/run/current-system/sw/bin:/run/wrappers/bin
SHA="$1"
FROM="$2"
SKILL="${SKILL:-strings}"
LOG="${LOG:-/tmp/verifier-remeasure.log}"
SRC="$HOME/antumbra-search-src/$SHA"
RUN="$FROM/remeasure-$(date -u +%Y%m%dT%H%M%SZ)-$SHA"
exec >"$LOG" 2>&1
set -euo pipefail
echo "== remeasure $SHA start $(date -u +%FT%TZ): $FROM, $SKILL"
for f in proposals.json samples-1.json samples-2.json; do
    test -f "$FROM/$f" || { echo "no $f in $FROM"; exit 1; }
done
mkdir -p "$RUN"
cp "$FROM/proposals.json" "$FROM/samples-1.json" "$FROM/samples-2.json" "$RUN/"
chmod -R 777 "$RUN"

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
docker build -f docker/Dockerfile.cuda --target calibrate -t "antumbra-calibrate:$SHA" . 2>&1 | tail -5

CORPUS="/build/corpora/workbench/$SKILL.json"
# The binary links the CUDA driver, so the device goes in even though
# nothing here uses the card.
antumbra() {
    docker run --rm --device nvidia.com/gpu=all -v "$RUN:/reports" "antumbra-calibrate:$SHA" \
        --url surrealkv:///reports/store.skv "$@"
}
workbench() {
    docker run --rm -v "$RUN:/reports" --entrypoint python "antumbra-calibrate:$SHA" \
        "/build/corpora/workbench/$1" "${@:2}"
}

antumbra migrate
workbench synthesize.py --corpus "$CORPUS" --proposals /reports/proposals.json \
    --specs /reports/specs.json --artifacts /reports/artifacts.json
antumbra verifier propose --batch /reports/specs.json
for n in 1 2; do
    antumbra verifier cases --corpus "$CORPUS" --completions "/reports/samples-$n.json" \
        --completions /reports/artifacts.json --out "/reports/cases-$n.json"
    echo "== measure on set $n $(date -u +%FT%TZ)"
    antumbra verifier measure --domain "$SKILL" --cases "/reports/cases-$n.json"
done
echo "== challenge on set 2 $(date -u +%FT%TZ)"
antumbra verifier challenge --domain "$SKILL" --cases /reports/cases-2.json
antumbra verifier list --domain "$SKILL"
echo "== remeasure $SHA end $(date -u +%FT%TZ) OK"
