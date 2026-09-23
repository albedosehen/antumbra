#!/usr/bin/env bash
# The control measurement behind ADR-0024's D-2 bar: how well a threshold on the
# deployed cross-encoder decides "does this memory answer this query".
#
# It exists because the bar it produces is a number a typed head has to beat, and
# a number nobody can reproduce is not a bar. Run it again before trusting that
# figure, and again against any candidate that claims to beat it.
#
# *** IT READS THE LABEL FILE RATHER THAN REBUILDING IT. ***
#
# This script and scripts/d2-labels.sh used to construct their pairs separately
# and identically, with a comment in each saying that keeping them identical was
# the point. They drifted anyway, in the way that matters: the construction they
# shared turned out to be DEGENERATE -- the query was left verbatim inside its
# positive memory, so a one-line `contains` check with no model scored F1 1.000
# and every score measured against this control was measuring substring
# detection rather than relevance. Fixing one script and not the other would
# have produced a bar and a candidate measured on different tasks, which is the
# exact failure this comment used to warn about.
#
# So there is now one construction, in one file, and this consumes its output:
#
#   ssh <host> 'bash -s' < scripts/d2-labels.sh > d2-labels.json
#   ANTUMBRA_D2_LABELS=d2-labels.json ./scripts/d2-relevance-baseline.sh
#
# Run it ON the host that has the reranker, or point ANTUMBRA_RERANK_ENDPOINT at
# one. Needs jq and curl.
#
# THE NO-MODEL CONTROL IS PRINTED FIRST AND IS THE ONE THAT MATTERS. If a
# `contains` check scores near 1.000, the label set is answerable by grep and no
# figure below it means anything about relevance. That line is here so the
# degeneracy cannot come back silently.

set -euo pipefail

LABELS="${ANTUMBRA_D2_LABELS:?set ANTUMBRA_D2_LABELS to the output of scripts/d2-labels.sh}"
RERANK="${ANTUMBRA_RERANK_ENDPOINT:-http://127.0.0.1:8091/rerank}"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

TOTAL=$(jq 'length' "$LABELS")
POS=$(jq '[.[]|select(.relevant)]|length' "$LABELS")
NEG=$((TOTAL - POS))
echo "pairs: $TOTAL  positives: $POS  hard negatives: $NEG"

# --- the control that decides whether the rest is worth reading --------------
# One jq expression, no model: is the query a substring of the memory?
jq -r '[.[] | select((.memory | ascii_downcase) | contains(.query | ascii_downcase | ltrimstr(" ") | rtrimstr(" "))) as $hit | 1] | length' \
  "$LABELS" > /dev/null 2>&1 || true
GREP_TP=$(jq '[.[] | select(.relevant) | select((.memory) | contains(.query | sub("^ +";"") | sub(" +$";"")))] | length' "$LABELS")
GREP_FP=$(jq '[.[] | select(.relevant|not) | select((.memory) | contains(.query | sub("^ +";"") | sub(" +$";"")))] | length' "$LABELS")
awk -v tp="$GREP_TP" -v fp="$GREP_FP" -v p="$POS" -v n="$NEG" '
BEGIN {
  prec = (tp + fp) > 0 ? tp / (tp + fp) : 0;
  rec  = p > 0 ? tp / p : 0;
  f1   = (prec + rec) > 0 ? 2 * prec * rec / (prec + rec) : 0;
  acc  = (tp + (n - fp)) / (p + n);
  printf "NO MODEL (does the memory contain the query verbatim?)\n";
  printf "  accuracy %.3f  precision %.3f  recall %.3f  F1 %.3f\n", acc, prec, rec, f1;
  if (f1 > 0.9)
    printf "  *** DEGENERATE: this label set is answerable by grep. Nothing below is about relevance. ***\n";
}'

# --- the cross-encoder threshold --------------------------------------------
: > "$work/all"
jq -c '.[]' "$LABELS" | while read -r row; do
    q=$(printf '%s' "$row" | jq -r .query)
    t=$(printf '%s' "$row" | jq -r .memory)
    lbl=$(printf '%s' "$row" | jq -r 'if .relevant then 1 else 0 end')
    s=$(jq -nc --arg q "$q" --arg t "$t" '{query:$q, texts:[$t]}' \
        | curl -s -X POST "$RERANK" -H 'Content-Type: application/json' -d @- --max-time 30 \
        | jq -r '.[0].score')
    printf '%s %s\n' "$s" "$lbl" >> "$work/all"
done

sort -g "$work/all" | awk -v P="$POS" -v N="$NEG" '
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
  printf "CROSS-ENCODER (best single threshold %s)\n", thr;
  printf "  accuracy %.3f  precision %.3f  recall %.3f  F1 %.3f\n", best, bp, br, bf;
  printf "  always-reject baseline %.3f\n", N / (P + N);
}'
