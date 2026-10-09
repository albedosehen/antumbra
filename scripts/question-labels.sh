#!/usr/bin/env bash
# Emit a question-shaped recall benchmark over the store, as the JSON
# antumbra-bench reads (ANTUMBRA_BENCH_LABELS).
#
# The relevance head's label file (scripts/d2-labels.sh) asks with a fragment:
# twelve of the memory's own words, cut from inside it. A person asks with a
# question in their own words, and the prompt hook recalls with whatever they
# typed. A fragment measures how well recall finds a memory by a piece of it;
# this measures how well it finds one by a question it answers.
#
# Each sampled memory gets one question from a chat model, through an
# OpenAI-compatible /v1/chat/completions (the Orin's qwen2.5-1.5b by default),
# asked for a short question the passage answers, in its own words. The model
# reads a window of the memory, not always its head: the windows rotate
# through the head, the middle and the tail, because a question about a detail
# deep inside a long memory is the case a single vector reads worst.
#
# THE LABEL IS THE GENERATOR'S CLAIM, NOT A VERIFIER'S. Unlike the relevance
# head's, these labels are not checkable by construction: they say which memory
# the question was written from. Another memory that also answers it counts as
# a miss, for every system measured alike, so the set compares retrieval
# systems and does not grade any of them absolutely. Use it to compare, never
# to train a head on: a typed decider trains only on labels a verifier made.
#
# Distractors: further memories with no question, so the corpus is closer in
# size to the store than the sample alone would make it.
#
#   ssh <host> 'bash -s' < scripts/question-labels.sh > questions.json
#
# Output: [{"query": "...", "memory": "...", "relevant": true|false}, ...]
# (a distractor is a row with relevant false and an empty query).

set -euo pipefail

COMPOSE_DIR="${ANTUMBRA_COMPOSE_DIR:-$HOME/antumbra-loop/docker}"
SURREAL="${ANTUMBRA_SURREAL_URL:-http://127.0.0.1:8000/sql}"
TENANT="${ANTUMBRA_TENANT:-ws:default}"
SAMPLE="${ANTUMBRA_SAMPLE:-400}"
DISTRACTORS="${ANTUMBRA_DISTRACTORS:-1600}"
GENERATOR="${ANTUMBRA_QGEN_URL:-http://10.0.0.51:8090/v1/chat/completions}"
WINDOW="${ANTUMBRA_QGEN_WINDOW:-1500}"

cd "$COMPOSE_DIR"
set -a && . ./.env && set +a

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

curl -s -u "root:$SURREAL_PASS" -H "Accept: application/json" \
  -H "surreal-ns: antumbra" -H "surreal-db: main" -X POST "$SURREAL" \
  --data-binary "SELECT content FROM memory WHERE tenant_id=\"$TENANT\" AND deleted_at = NONE AND string::len(content) > 200 ORDER BY rand() LIMIT $((SAMPLE + DISTRACTORS));" \
  > "$work/raw.json"
jq -c ".[0].result[].content | gsub(\"\\\\s+\"; \" \")" "$work/raw.json" > "$work/lines.json"
mapfile -t L < "$work/lines.json"
N=${#L[@]}
echo "memories: $N (questions for up to $SAMPLE, the rest distractors)" >&2

: > "$work/pairs.jsonl"
asked=0
for i in $(seq 0 $((N - 1))); do
    memory=$(printf '%s' "${L[$i]}" | jq -r .)
    if [ "$i" -ge "$SAMPLE" ]; then
        jq -nc --arg m "$memory" '{query:"", memory:$m, relevant:false}' >> "$work/pairs.jsonl"
        continue
    fi
    # The window: the head, the middle or the tail, in turn.
    len=${#memory}
    span=$((len > WINDOW ? len - WINDOW : 0))
    start=$(( (i % 3) * span / 2 ))
    window="${memory:$start:$WINDOW}"
    prompt="Here is a note from an engineer's memory store.

$window

Write one short question (under 15 words) that a person might ask later, which this note answers. Use your own words: do not copy names of files, commands or long phrases from the note. Reply with the question only."
    body=$(jq -nc --arg p "$prompt" \
        '{messages:[{role:"user",content:$p}], max_tokens:40, temperature:0.3}')
    question=$(curl -s --max-time 60 -H "Content-Type: application/json" -X POST "$GENERATOR" \
        --data-binary "$body" | jq -r '.choices[0].message.content // empty' \
        | head -n 1 | sed -E 's/^[[:space:]"*]+|[[:space:]"*]+$//g')
    if [ "${#question}" -lt 10 ] || [ "${#question}" -gt 200 ]; then
        echo "skipped $i: no usable question" >&2
        continue
    fi
    jq -nc --arg q "$question" --arg m "$memory" '{query:$q, memory:$m, relevant:true}' >> "$work/pairs.jsonl"
    asked=$((asked + 1))
    [ $((asked % 50)) -eq 0 ] && echo "questions: $asked" >&2
done

jq -s '.' "$work/pairs.jsonl"
echo "questions: $asked, rows: $(wc -l < "$work/pairs.jsonl")" >&2
