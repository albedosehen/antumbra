#!/usr/bin/env bash
# Emit the labeled relevance set ADR-0024's D-2 head trains on, as JSON.
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
# *** THE SPAN IS EXCISED FROM THE MEMORY, AND THAT IS NOT OPTIONAL. ***
#
# The first version of this script left the query verbatim inside its positive
# memory, which made the whole benchmark DEGENERATE: a one-line `contains` check
# with no model scored acc 1.000 / F1 1.000 on it. Every score measured against
# it -- the 0.782 cross-encoder control included -- was measuring substring
# detection rather than relevance, and a chunk-size sweep "beat" the control at
# 0.940 purely because a 150-character chunk IS the query. The tell was the
# monotonic curve: the smaller the chunk, the better the score.
#
# So the twelve words are REMOVED from the memory they were cut from. The pair is
# then "this passage is about what this query asks about", which is the question
# D-2 actually needs answered, and no lexical shortcut survives it. The verifier
# stays deterministic, reproducible and model-free -- the properties that let D-2
# be measured at all -- while no longer being answerable by grep.
#
# Both sides are flattened to single-spaced text so the excision cannot be
# defeated by a line break, and so positives and negatives are shaped alike.
#
# Negatives are HARD: the same real query against a DIFFERENT memory, not a
# nonsense query nothing would match. Easy negatives make an accuracy figure
# close to meaningless, which an earlier pass of this measurement learned.
#
#   ssh <host> 'bash -s' < scripts/d2-labels.sh > d2-labels.json
#
# Set ANTUMBRA_D2_KEEP_SPAN=1 to reproduce the degenerate construction, which is
# useful for exactly one thing: showing that it is degenerate.
#
# Output: [{"query": "...", "memory": "...", "relevant": true|false}, ...]

set -euo pipefail

COMPOSE_DIR="${ANTUMBRA_COMPOSE_DIR:-$HOME/antumbra-loop/docker}"
SURREAL="${ANTUMBRA_SURREAL_URL:-http://127.0.0.1:8000/sql}"
TENANT="${ANTUMBRA_TENANT:-ws:default}"
LIMIT="${ANTUMBRA_SAMPLE:-400}"
KEEP_SPAN="${ANTUMBRA_D2_KEEP_SPAN:-0}"

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
echo "span: $([ "$KEEP_SPAN" = 1 ] && echo "KEPT (degenerate, for demonstration)" || echo excised)" >&2

# First pass: for each memory, the query cut from it and the body it leaves
# behind. Both are needed before pairing, because a negative pairs one memory's
# query with another memory's body and both must be built the same way.
: > "$work/q"; : > "$work/body"
for i in $(seq 0 $((N - 1))); do
    text=$(printf '%s' "${L[$i]}" | jq -r . | tr '\n' ' ')
    # Twelve words from ~60% through, deliberately not the head: the head is what
    # a mean-pooled vector already represents, so a head-answerable query would
    # flatter whatever is being measured.
    printf '%s\n' "$(printf '%s' "$text" \
        | awk '{n=NF; s=int(n*0.6); for (j=s; j<s+12 && j<=n; j++) printf "%s ", $j}')" >> "$work/q"
    if [ "$KEEP_SPAN" = 1 ]; then
        printf '%s\n' "$(printf '%s' "$text" | awk '{$1=$1; print}')" >> "$work/body"
    else
        # The same twelve words, removed. What remains is the passage around
        # them, which is about the query's subject without containing its words.
        printf '%s\n' "$(printf '%s' "$text" \
            | awk '{n=NF; s=int(n*0.6); for (j=1; j<=n; j++) if (j < s || j >= s+12) printf "%s ", $j}')" >> "$work/body"
    fi
done
mapfile -t Q < "$work/q"
mapfile -t BODY < "$work/body"

: > "$work/pairs.jsonl"
for i in $(seq 0 $((N - 1))); do
    q="${Q[$i]}"
    [ "${#q}" -lt 20 ] && continue
    [ "${#BODY[$i]}" -lt 100 ] && continue
    jq -nc --arg q "$q" --arg m "${BODY[$i]}" '{query:$q, memory:$m, relevant:true}' >> "$work/pairs.jsonl"
    # 37 is coprime with most sample sizes, so the pairing cannot degenerate into
    # pairing neighbors.
    jq -nc --arg q "$q" --arg m "${BODY[$(((i + 37) % N))]}" \
        '{query:$q, memory:$m, relevant:false}' >> "$work/pairs.jsonl"
done

jq -s '.' "$work/pairs.jsonl"
echo "pairs: $(wc -l < "$work/pairs.jsonl")" >&2
