#!/usr/bin/env bash
# Emit the labelled relevance set ADR-0024's D-2 head trains on, as JSON.
#
# Same construction as scripts/d2-relevance-baseline.sh, which measures the
# CONTROL over this data; this one writes the pairs out so a head can be trained
# and scored on exactly what the control was scored on. Keeping the construction
# identical is the point -- a bar measured on one set and beaten on another is
# not beaten.
#
# THE LABELS ARE CONSTRUCTED BY A DETERMINISTIC VERIFIER, not logged. A query cut
# from inside a memory has that memory as its answer and does not have any other
# memory as its answer. Substring provenance is checkable, reproducible and
# derived from no model, which is what ADR-0024's typed-decision rule demands of
# a label. This is why D-2 is measurable on a deployment whose generational loop
# has never produced an evaluation run.
#
# Negatives are HARD: the same real query against a DIFFERENT memory, not a
# nonsense query nothing would match. Easy negatives make an accuracy figure
# close to meaningless, which an earlier pass of this measurement learned.
#
#   ssh <host> 'bash -s' < scripts/d2-labels.sh > d2-labels.json
#
# Output: [{"query": "...", "memory": "...", "relevant": true|false}, ...]

set -euo pipefail

COMPOSE_DIR="${ANTUMBRA_COMPOSE_DIR:-$HOME/antumbra-loop/docker}"
SURREAL="${ANTUMBRA_SURREAL_URL:-http://127.0.0.1:8000/sql}"
TENANT="${ANTUMBRA_TENANT:-ws:default}"
LIMIT="${ANTUMBRA_SAMPLE:-400}"

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
echo "memories: $N" >&2

: > "$work/pairs.jsonl"
for i in $(seq 0 $((N - 1))); do
    text=$(printf '%s' "${L[$i]}" | jq -r .)
    # Twelve words from ~60% through, deliberately not the head: the head is what
    # a mean-pooled vector already represents, so a head-answerable query would
    # flatter whatever is being measured.
    q=$(printf '%s' "$text" | tr '\n' ' ' \
        | awk '{n=NF; s=int(n*0.6); for (j=s; j<s+12 && j<=n; j++) printf "%s ", $j}')
    [ "${#q}" -lt 20 ] && continue
    jq -nc --arg q "$q" --arg m "$text" '{query:$q, memory:$m, relevant:true}' >> "$work/pairs.jsonl"
    # 37 is coprime with most sample sizes, so the pairing cannot degenerate into
    # pairing neighbours.
    other=$(printf '%s' "${L[$(((i + 37) % N))]}" | jq -r .)
    jq -nc --arg q "$q" --arg m "$other" '{query:$q, memory:$m, relevant:false}' >> "$work/pairs.jsonl"
done

jq -s '.' "$work/pairs.jsonl"
echo "pairs: $(wc -l < "$work/pairs.jsonl")" >&2
