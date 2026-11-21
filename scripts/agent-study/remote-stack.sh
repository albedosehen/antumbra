#!/usr/bin/env bash
# The agent study's throwaway Antumbra stack on a study host, sent over ssh by
# run-series.ps1 (`ssh <host> bash -s` with this file on stdin, after a preamble
# that sets ACTION and, for `up`, SURREAL_PASS, ANTUMBRA_JWT_SECRET, IMAGE and
# BIND). Every name is agentstudy-*, so nothing a host already runs is touched:
# not its containers, its network, its ports or its volumes. The store is in
# memory and goes with the container; only agentstudy-weights, the public
# embedding model the server downloads, is kept between runs.
set -euo pipefail
: "${ACTION:?ACTION is up or down}"

down() {
  docker rm -f agentstudy-mcp agentstudy-surrealdb >/dev/null 2>&1 || true
  docker network rm agentstudy-net >/dev/null 2>&1 || true
}

case "$ACTION" in
  down)
    down
    echo "agentstudy: down"
    ;;
  up)
    : "${SURREAL_PASS:?}" "${ANTUMBRA_JWT_SECRET:?}" "${IMAGE:?}" "${BIND:?}"
    down
    docker network create agentstudy-net >/dev/null
    docker run -d --name agentstudy-surrealdb --network agentstudy-net \
      --user 65532:65532 --cap-drop ALL --security-opt no-new-privileges:true \
      -e SURREAL_PASS="$SURREAL_PASS" \
      surrealdb/surrealdb:v3.2.4 start --user root --bind 0.0.0.0:8000 memory >/dev/null
    for _ in $(seq 1 30); do
      if docker exec agentstudy-surrealdb /surreal is-ready --endpoint http://localhost:8000 >/dev/null 2>&1; then
        break
      fi
      sleep 1
    done
    # No --embedder-url: the model backend embeds in process, all-MiniLM-L6-v2
    # read into the weights volume the first time.
    docker run -d --name agentstudy-mcp --network agentstudy-net \
      --read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges:true \
      -v agentstudy-weights:/weights \
      -p "$BIND:8081" \
      -e ANTUMBRA_DB_PASS="$SURREAL_PASS" -e ANTUMBRA_JWT_SECRET="$ANTUMBRA_JWT_SECRET" \
      "$IMAGE" --http 0.0.0.0:8081 --url ws://agentstudy-surrealdb:8000/rpc --db-user root >/dev/null
    echo "agentstudy: up on $BIND"
    ;;
  *)
    echo "agentstudy: unknown ACTION $ACTION" >&2
    exit 2
    ;;
esac
