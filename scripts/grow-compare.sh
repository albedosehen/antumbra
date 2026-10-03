#!/usr/bin/env bash
# Compare the grow step with its baselines on the GPU host (ADR-0022 S-3's
# validation): the empty status quo, where every generation learns from every
# visible task; uniform sampling of regions; and credit, the grow step proper.
# Each arm runs the same corpus for the same number of generations, one after
# another, through search-validate.sh, with its own store and its own log
# (/tmp/grow-compare-<arm>.log). A summary of what each arm graduated, how long
# it took and where its population ended lands in /tmp/grow-compare.log.
#
# Every arm measures contribution each generation over the same 64 live tasks:
# the grow arms need its census, and the status quo takes it too, so the
# population is read the same way in all three. Admission is on in all three,
# as it is by default. Knobs: CORPUS (all), GENERATIONS (4), ARMS (the arms to
# run, default "status-quo uniform credit"), GROW_FROM (where a grow arm's
# region shadow starts: incumbent, the default, or base) and SEED (the
# generation seed every arm trains under, `train --seed`; 0 by default, which
# is what the arms ran under before the knob existed). A nonzero seed is
# another trajectory of the same comparison, and its logs carry the seed in
# their names so a second run does not overwrite the first.
#
#   git archive --format=tar.gz -o /tmp/antumbra-<sha>.tar.gz HEAD
#   scp /tmp/antumbra-<sha>.tar.gz scripts/search-validate.sh scripts/grow-compare.sh <host>:/tmp/
#   ssh <host> 'nohup bash /tmp/grow-compare.sh <sha> >/dev/null 2>&1 &'

SHA="$1"
CORPUS="${CORPUS:-all}"
GENERATIONS="${GENERATIONS:-4}"
ARMS="${ARMS:-status-quo uniform credit}"
GROW_FROM="${GROW_FROM:-incumbent}"
SEED="${SEED:-0}"
suffix=""
[ "$SEED" = 0 ] || suffix="-seed$SEED"
exec >"/tmp/grow-compare$suffix.log" 2>&1
set -uo pipefail
echo "== grow comparison $SHA start $(date -u +%FT%TZ): $CORPUS, $GENERATIONS generation(s) an arm, arms: $ARMS, grow from $GROW_FROM, seed $SEED"

for arm in $ARMS; do
    case "$arm" in
        status-quo) args="--contribution-every 1 --contribution-tasks 64" ;;
        uniform) args="--grow --grow-by uniform --grow-from $GROW_FROM --contribution-tasks 64" ;;
        credit) args="--grow --grow-by credit --grow-from $GROW_FROM --contribution-tasks 64" ;;
        *) echo "== unknown arm $arm"; continue ;;
    esac
    args="$args --seed $SEED"
    log="/tmp/grow-compare-$arm$suffix.log"
    start=$(date -u +%s)
    CORPUS="$CORPUS" SEARCH=0 GENERATIONS="$GENERATIONS" ARGS="$args" LOG="$log" \
        bash /tmp/search-validate.sh "$SHA"
    status=$?
    minutes=$(( ($(date -u +%s) - start) / 60 ))
    graduated=$(grep -c "graduated=true" "$log" || true)
    population=$(grep -E "^population: " "$log" | tail -1)
    baseline=$(grep -E "population [0-9.]+ over" "$log" | tail -1 | sed 's/^ *//')
    echo "== $arm: exit $status after $minutes min; $graduated generation(s) graduated and admitted; $population"
    echo "   where it ended: $baseline"
done
echo "== grow comparison $SHA end $(date -u +%FT%TZ)"
