#!/usr/bin/env bash
# Score the base model on every workbench skill file, on the GPU host, and keep
# each task's result. Detached; logs to /tmp/workbench-calibrate.log.
#
# The corpus leaves the loop nothing to learn from on tasks the base model always
# passes or never passes, so before it is trained on, it is measured. The
# reports this writes are what corpora/workbench/calibrate.py summarises.
#
# Expects a source tarball at /tmp/antumbra-$SHA.tar.gz, as kusko-deploy.sh does:
#   git archive --format=tar.gz -o /tmp/antumbra-$(git rev-parse --short HEAD).tar.gz HEAD
#   scp /tmp/antumbra-<sha>.tar.gz <host>:/tmp/
#   scp scripts/workbench-calibrate.sh <host>:/tmp/
#   ssh <host> 'nohup bash /tmp/workbench-calibrate.sh <sha> >/dev/null 2>&1 &'
#
# It works in its own directory, never in the deploy checkout, and it does not
# touch the running services. SAMPLES (default 4) sets draws per task.

export PATH=$PATH:/run/current-system/sw/bin:/run/wrappers/bin
SHA="$1"
SAMPLES="${SAMPLES:-4}"
DIR="$HOME/antumbra-calibrate"
exec >/tmp/workbench-calibrate.log 2>&1
set -euo pipefail
echo "== calibrate $SHA start $(date -u +%FT%TZ)"

rm -rf "$DIR" && mkdir -p "$DIR/reports"
tar -xzf "/tmp/antumbra-$SHA.tar.gz" -C "$DIR"
# git archive stamps every file with its commit's time, and the image build's
# target cache is shared across commits, so a commit older than the last build
# would be compiled against that build's artifacts. Stamp the sources now, so
# cargo rebuilds what differs.
find "$DIR" -type f -exec touch {} +
cd "$DIR"

echo "== build antumbra-calibrate:$SHA"
docker build -f docker/Dockerfile.cuda --target calibrate -t "antumbra-calibrate:$SHA" . 2>&1 | tail -20
chmod 777 "$DIR/reports"

for corpus in corpora/workbench/*.json; do
    skill="$(basename "$corpus" .json)"
    [ "$skill" = all ] && continue
    echo "== $skill $(date -u +%FT%TZ)"
    docker run --rm --device nvidia.com/gpu=all \
        -v antumbra-gpu-test-weights:/weights \
        -v "$DIR/reports:/reports" \
        "antumbra-calibrate:$SHA" \
        eval --corpus "/build/$corpus" --samples "$SAMPLES" --report "/reports/$skill.json" 2>&1 | tail -3
done

echo "== calibrate $SHA end $(date -u +%FT%TZ) OK"
