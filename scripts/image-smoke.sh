#!/usr/bin/env bash
# Check a built antumbra-mcp image the way `antumbra setup local` runs it: it
# starts, it says it serves experts (`workspace_stats.serving`) on the CPU from
# a quantized base through Antumbra's own kernels, and with no expert to serve
# `answer` escalates cleanly. It needs no database and downloads no model: an
# in-memory store, a fixed embedder, and no expert, so the base is never read.
#
#   scripts/image-smoke.sh ghcr.io/albedosehen/antumbra-mcp:0.5.2
#   SMOKE_PORT=18082 scripts/image-smoke.sh antumbra-mcp:local
#
# Needs docker, curl and jq. The token and the signing secret are made here and
# end with the container.
set -euo pipefail

image="${1:?usage: image-smoke.sh <image>}"
name="antumbra-smoke-$$"
port="${SMOKE_PORT:-18081}"
secret="$(head -c 48 /dev/urandom | base64 | tr -d '\n=+/')"

cleanup() { docker rm -f "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT

fail() {
  echo "image-smoke: $1" >&2
  docker logs "$name" >&2 || true
  exit 1
}

docker run -d --name "$name" -p "127.0.0.1:${port}:8081" \
  -e ANTUMBRA_JWT_SECRET="$secret" \
  "$image" --http 0.0.0.0:8081 --url mem:// --fake-embedder >/dev/null

up=""
for _ in $(seq 1 60); do
  if curl -fs -o /dev/null "http://127.0.0.1:${port}/dashboard"; then
    up=1
    break
  fi
  sleep 1
done
[ -n "$up" ] || fail "the server did not answer on :${port} within a minute"

token="$(docker exec "$name" /usr/local/bin/antumbra-mcp --mint-token | tail -n 1)"
call() {
  curl -fsS -X POST "http://127.0.0.1:${port}/mcp/call" \
    -H "Authorization: Bearer ${token}" -H 'Content-Type: application/json' -d "$1"
}

stats="$(call '{"tool":"workspace_stats","arguments":{}}')" || fail "workspace_stats was refused"
echo "workspace_stats: $stats"
echo "$stats" | jq -e '
  .serving.device == "cpu"
  and .serving.base != "dense"
  and (.serving.kernels | IN("avx-512", "avx2", "neon"))' >/dev/null ||
  fail "the image does not serve experts on the CPU through Antumbra's kernels"

answer="$(call '{"tool":"answer","arguments":{"task":"add two numbers","standing_only":true}}')" ||
  fail "answer was refused"
echo "answer: $answer"
echo "$answer" | jq -e '.escalate == true' >/dev/null ||
  fail "with no expert to serve, answer did not escalate"

echo "image-smoke: ok, $image serves experts on the CPU"
