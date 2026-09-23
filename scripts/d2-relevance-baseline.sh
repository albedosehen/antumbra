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
#
# The bindings are explicit because `contains()` evaluates its argument with `.`
# rebound to the string on its left, so the obvious `.memory | contains(.query)`
# indexes a string with "query" and dies. Naming the row first is the fix.
hits() { # $1 = true to count positives, false to count negatives
  jq --argjson want "$1" \
    '[ .[]
       | select(.relevant == $want)
       | . as $r
       | ($r.query | sub("^ +";"") | sub(" +$";"")) as $q
       | select($q != "" and ($r.memory | contains($q)))
     ] | length' "$LABELS"
}
GREP_TP=$(hits true)
GREP_FP=$(hits false)
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

# The raw scores, for calibration rather than thresholding. A threshold asks
# "which side", and a floor that means the same thing for every query needs
# "how likely" -- which is a monotone map from this score, fitted against these
# same verifier labels. ADR-0024's TypedDecider contract requires that fit to
# minimise a strictly proper scoring rule, so the fitting lives in Rust beside
# the Brier implementation rather than in awk here.
if [ -n "${ANTUMBRA_D2_SCORES_OUT:-}" ]; then
    cp "$work/all" "$ANTUMBRA_D2_SCORES_OUT"
    echo "raw scores written to $ANTUMBRA_D2_SCORES_OUT ($(wc -l < "$work/all") rows of 'score label')"
fi

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
