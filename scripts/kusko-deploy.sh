#!/usr/bin/env bash
# Rebuild the CUDA server at $1 beside the managed checkout and recreate the
# compose services. Detached; logs to /tmp/kusko-deploy.log.
#
# Lives in the repo rather than only on the host: it had been sitting in /tmp on
# kuskokwim, which is cleared on reboot and versioned nowhere, so the one script
# that knows how to deploy could vanish without anybody noticing until a deploy
# was needed.
#
# Expects a source tarball at /tmp/antumbra-$SHA.tar.gz:
#   git archive --format=tar.gz -o /tmp/antumbra-$(git rev-parse --short HEAD).tar.gz HEAD
#   scp /tmp/antumbra-<sha>.tar.gz <host>:/tmp/
#   ssh <host> 'nohup bash scripts/kusko-deploy.sh <sha> >/dev/null 2>&1 &'
#
# Only docker/.env survives the rebuild, so anything host-local -- the bind
# address, the rerank endpoint -- belongs there and not in a tracked file.

export PATH=$PATH:/run/current-system/sw/bin:/run/wrappers/bin
SHA="$1"
exec >/tmp/kusko-deploy.log 2>&1
set -e
echo "== deploy $SHA start $(date -u +%FT%TZ)"
docker inspect antumbra-mcp --format 'before: {{.Config.Image}} created {{.Created}}'

rm -rf "$HOME/antumbra-loop.new" && mkdir -p "$HOME/antumbra-loop.new"
tar -xzf "/tmp/antumbra-$SHA.tar.gz" -C "$HOME/antumbra-loop.new"
# git archive stamps every file with its commit's time, and the image build's
# target cache is shared across commits, so a commit older than the last build
# would be compiled against that build's artifacts. Stamp the sources now, so
# cargo rebuilds what differs.
find "$HOME/antumbra-loop.new" -type f -exec touch {} +
cp "$HOME/antumbra-loop/docker/.env" "$HOME/antumbra-loop.new/docker/.env"
rm -rf "$HOME/antumbra-loop.prev"
mv "$HOME/antumbra-loop" "$HOME/antumbra-loop.prev"
mv "$HOME/antumbra-loop.new" "$HOME/antumbra-loop"
cd "$HOME/antumbra-loop"

echo "== build antumbra-mcp-gpu:$SHA"
docker build -f docker/Dockerfile.cuda -t "antumbra-mcp-gpu:$SHA" . 2>&1 | tail -30
docker tag "antumbra-mcp-gpu:$SHA" antumbra-mcp-gpu:local

echo "== recreate"
cd "$HOME/antumbra-loop/docker"
COMPOSE=(-f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.gpu-cdi.yml)
docker compose "${COMPOSE[@]}" up -d --force-recreate antumbra-mcp 2>&1 | tail -15

# Bring the cross-encoder back up if this deployment uses one. Without this a
# deploy could leave it stopped and recall would quietly fall back to the fused
# RRF order -- correct, but a silent loss of the precision stage, and the only
# other signal is a line on antumbra-mcp's stderr. `up -d` without
# --force-recreate is idempotent: a healthy container is left alone.
if grep -q '^ANTUMBRA_RERANK_URL=.\+' .env 2>/dev/null; then
    echo "== ensure rerank (ANTUMBRA_RERANK_URL is set)"
    docker compose "${COMPOSE[@]}" --profile rerank up -d rerank 2>&1 | tail -5
else
    echo "== rerank not configured (ANTUMBRA_RERANK_URL unset), skipping"
fi

# The same for copal, the document of record, when this deployment archives to
# it. Here a stopped service is louder than a lost precision stage: every
# document ingest fails until it is back, since the original is archived before
# any chunk is stored. `up -d copal` starts its database with it.
if grep -q '^ANTUMBRA_COPAL_ADDR=.\+' .env 2>/dev/null; then
    echo "== ensure copal (ANTUMBRA_COPAL_ADDR is set)"
    docker compose "${COMPOSE[@]}" --profile copal up -d copal 2>&1 | tail -5
else
    echo "== copal not configured (ANTUMBRA_COPAL_ADDR unset), skipping"
fi

sleep 8
docker ps --format '{{.Names}}|{{.Image}}|{{.Status}}' | grep antumbra
docker inspect antumbra-mcp --format 'after: {{.Config.Image}} created {{.Created}}'
echo "== deploy $SHA end $(date -u +%FT%TZ) OK"
