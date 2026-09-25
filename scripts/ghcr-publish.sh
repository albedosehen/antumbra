#!/usr/bin/env bash
# Build the antumbra-mcp image and publish it to GHCR, from the GPU host, as
# .github/workflows/mcp-image.yml does on the hosted runners: docker/Dockerfile,
# tagged ghcr.io/albedosehen/antumbra-mcp:<full commit sha> and :latest. Shaman
# pins the full sha through Flux, so this is how it gets a new image while the
# runners are unavailable. Detached; logs to $LOG (/tmp/ghcr-publish.log by
# default).
#
# $1 is the FULL 40-character commit sha. Expects its source tarball, named by
# the short sha as the other host scripts use:
#   git archive --format=tar.gz -o /tmp/antumbra-$(git rev-parse --short HEAD).tar.gz HEAD
#   scp /tmp/antumbra-<short>.tar.gz scripts/ghcr-publish.sh <host>:/tmp/
#   ssh <host> 'nohup bash /tmp/ghcr-publish.sh <full sha> >/dev/null 2>&1 &'
#
# The token is read from GHCR_TOKEN_FILE, ~/.secrets/GHCR_TOKEN by default (a
# classic personal access token with write:packages and nothing else). The
# login goes to a throwaway docker config, deleted when the script exits, so no
# credential is left in ~/.docker. With DELETE_TOKEN_FILE=1 the token file is
# removed as soon as the login has read it, so a token handed over for one
# publish does not stay on the host:
#   dir=$(gh auth token | ssh <host> 'umask 077; d=$(mktemp -d); cat > "$d/token"; echo "$d"')
#   ssh <host> "GHCR_TOKEN_FILE=$dir/token DELETE_TOKEN_FILE=1 nohup bash /tmp/ghcr-publish.sh <full sha> >/dev/null 2>&1 &"
# DRY_RUN=1 builds and tags without logging in or pushing. Nothing here touches
# the deploy checkout or the running services.

export PATH=$PATH:/run/current-system/sw/bin:/run/wrappers/bin
SHA="$1"
IMAGE="ghcr.io/albedosehen/antumbra-mcp"
USER_NAME="${GHCR_USER:-albedosehen}"
TOKEN_FILE="${GHCR_TOKEN_FILE:-$HOME/.secrets/GHCR_TOKEN}"
DRY_RUN="${DRY_RUN:-0}"
LOG="${LOG:-/tmp/ghcr-publish.log}"
exec >"$LOG" 2>&1
set -euo pipefail
SRC=""
CONFIG=""
# Whatever happens, the working copy and the docker config go, and so does a
# handed-over token, even when the script stops before it logs in.
cleanup() {
    [ -n "$SRC" ] && rm -rf "$SRC"
    [ -n "$CONFIG" ] && rm -rf "$CONFIG"
    if [ "${DELETE_TOKEN_FILE:-0}" = "1" ] && [ -e "$TOKEN_FILE" ]; then
        rm -f "$TOKEN_FILE"
        rmdir "$(dirname "$TOKEN_FILE")" 2>/dev/null || true
    fi
}
trap cleanup EXIT
echo "== publish $SHA start $(date -u +%FT%TZ) (dry run: $DRY_RUN)"

if ! [[ "$SHA" =~ ^[0-9a-f]{40}$ ]]; then
    echo "the full 40-character sha is required: shaman pins it"
    exit 1
fi
SHORT="${SHA:0:7}"
TARBALL="/tmp/antumbra-$SHORT.tar.gz"
test -f "$TARBALL" || { echo "no source tarball at $TARBALL"; exit 1; }
if [ "$DRY_RUN" != "1" ]; then
    test -s "$TOKEN_FILE" || { echo "no token at $TOKEN_FILE"; exit 1; }
fi

SRC="$(mktemp -d)"
CONFIG="$(mktemp -d)"

# Log in before the build, so a bad token fails in seconds rather than after
# it, and let a handed-over token go at once.
if [ "$DRY_RUN" != "1" ]; then
    docker --config "$CONFIG" login ghcr.io -u "$USER_NAME" --password-stdin <"$TOKEN_FILE"
    if [ "${DELETE_TOKEN_FILE:-0}" = "1" ]; then
        rm -f "$TOKEN_FILE"
        rmdir "$(dirname "$TOKEN_FILE")" 2>/dev/null || true
    fi
fi

tar -xzf "$TARBALL" -C "$SRC"
cd "$SRC"

echo "== build $IMAGE:$SHA"
docker build -f docker/Dockerfile -t "$IMAGE:$SHA" -t "$IMAGE:latest" . 2>&1 | tail -20
docker image inspect "$IMAGE:$SHA" --format 'built {{.Id}} {{.Size}} bytes'

if [ "$DRY_RUN" = "1" ]; then
    echo "== dry run: not pushed"
    echo "== publish $SHA end $(date -u +%FT%TZ) OK"
    exit 0
fi

echo "== push"
docker --config "$CONFIG" push "$IMAGE:$SHA"
docker --config "$CONFIG" push "$IMAGE:latest"
docker --config "$CONFIG" manifest inspect "$IMAGE:$SHA" >/dev/null && echo "== $IMAGE:$SHA is in the registry"
docker --config "$CONFIG" logout ghcr.io
echo "== publish $SHA end $(date -u +%FT%TZ) OK"
