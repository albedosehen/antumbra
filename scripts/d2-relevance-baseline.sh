#!/usr/bin/env bash
# The control measurement behind ADR-0024's D-2 bar: how well a threshold on the
# deployed cross-encoder decides "does this memory answer this query".
#
# It exists because the bar it produces (0.782 F1, recorded in ADR-0024) is a
# number a typed head has to beat, and a number nobody can reproduce is not a
# bar. Run it again before trusting that figure, and again against any candidate
# that claims to beat it.
#
# THE LABELS ARE CONSTRUCTED, NOT LOGGED, which is the whole reason this can run
# at all. The generational loop has never run on this deployment, so the store
# holds no evaluation runs, shadows, boundaries or reward signals -- the outcomes
# ADR-0024's other half waits for. D-2 does not need them: a query cut from
# inside a memory has that memory as its answer and does not have any other
# memory as its answer. The verifier is substring provenance, which is checkable,
# reproducible and derived from no model, so it satisfies the typed-decision
# rule's demand that labels come from a verifier.
#
# NEGATIVES ARE HARD ON PURPOSE. Each negative pairs a REAL query with a
# DIFFERENT memory, rather than a nonsense query nothing would match. An earlier
# pass used nonsense negatives on an unbalanced set and reported 0.950 accuracy
# against an 0.833 always-reject baseline, which read as failure and was an
# artefact of both choices. Easy negatives and unbalanced classes make an
# accuracy figure close to meaningless.
#
# Run it ON the host that has the store and the reranker:
#   ssh <host> 'bash -s' < scripts/d2-relevance-baseline.sh
# Needs jq, curl, docker/.env beside the compose file, and the rerank service up.

set -euo pipefail

COMPOSE_DIR="${ANTUMBRA_COMPOSE_DIR:-$HOME/antumbra-loop/docker}"
SURREAL="${ANTUMBRA_SURREAL_URL:-http://127.0.0.1:8000/sql}"
RERANK="${ANTUMBRA_RERANK_ENDPOINT:-http://127.0.0.1:8091/rerank}"
TENANT="${ANTUMBRA_TENANT:-ws:default}"
LIMIT="${ANTUMBRA_SAMPLE:-120}"

cd "$COMPOSE_DIR"
set -a && . ./.env && set +a

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

curl -s -u "root:$SURREAL_PASS" -H "Accept: application/json" \
  -H "surreal-ns: antumbra" -H "surreal-db: main" -X POST "$SURREAL" \
  --data-binary "SELECT content FROM memory WHERE tenant_id=\"$TENANT\" AND string::len(content) > 300 LIMIT $LIMIT;" \
  > "$work/raw.json"
jq -c ".[0].result[].content" "$work/raw.json" > "$work/lines.json"
mapfile -t L < "$work/lines.json"
N=${#L[@]}
echo "memories: $N"

score() { # query, text
  jq -nc --arg q "$1" --arg t "$2" '{query:$q, texts:[$t]}' \
    | curl -s -X POST "$RERANK" -H 'Content-Type: application/json' -d @- --max-time 30 \
    | jq -r '.[0].score'
}

: > "$work/pos"; : > "$work/neg"
for i in $(seq 0 $((N - 1))); do
    text=$(printf '%s' "${L[$i]}" | jq -r .)
    # A twelve-word query from ~60% through, deliberately not the head: the head
    # is what a mean-pooled vector already represents, so a head-answerable query
    # would flatter any retrieval signal being measured.
    q=$(printf '%s' "$text" | tr '\n' ' ' \
        | awk '{n=NF; s=int(n*0.6); for (j=s; j<s+12 && j<=n; j++) printf "%s ", $j}')
    [ "${#q}" -lt 20 ] && continue
    score "$q" "$text" >> "$work/pos"
    # The hard negative: same query, a different memory. 37 is coprime with most
    # sample sizes, so the pairing does not degenerate into neighbours.
    other=$(printf '%s' "${L[$(((i + 37) % N))]}" | jq -r .)
    score "$q" "$other" >> "$work/neg"
done

P=$(wc -l < "$work/pos"); NEG=$(wc -l < "$work/neg")
echo "positives: $P  hard negatives: $NEG"
awk '{print $1" 1"}' "$work/pos"  > "$work/all"
awk '{print $1" 0"}' "$work/neg" >> "$work/all"

sort -g "$work/all" | awk -v P="$P" -v N="$NEG" '
BEGIN { tp=P; fp=N; best=0 }
{
  if ($2 == 1) tp--; else fp--;
  prec = (tp + fp) > 0 ? tp / (tp + fp) : 1;
  rec  = tp / P;
  f1   = (prec + rec) > 0 ? 2 * prec * rec / (prec + rec) : 0;
  acc  = (tp + (N - fp)) / (P + N);
  if (acc > best) { best = acc; thr = $1; bp = prec; br = rec; bf = f1 }
}
END {
  printf "best threshold %s\n  accuracy %.3f  precision %.3f  recall %.3f  F1 %.3f\n", thr, best, bp, br, bf;
  printf "  always-reject baseline %.3f\n", N / (P + N);
}'
