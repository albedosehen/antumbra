#!/usr/bin/env bash
# The ADR index must describe the ADRs that exist. This drifted once, silently:
# five statuses disagreed with their own file and four records had no row at
# all, because nothing checked. Now something does.
#
# Three things are checked, and nothing more, so the index stays a summary
# rather than a copy:
#   1. every ADR file has a row
#   2. every row points at a file that exists
#   3. a row's status class matches the file's own Status line
#
# The class is the first word: `Accepted (v0)` and `Accepted - implemented` are
# the same class, so the index may stay short while a record says more. The
# status legend's two shorthands are honoured: the antumbra row and the north
# star row are written as the legend writes them.
#
#   bash scripts/adr-index-check.sh
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
adr="$here/docs/adr"
index="$adr/README.md"
failed=0

fail() { echo "  FAIL  $1"; failed=1; }

# The status cell of a row, and the first word of it, markdown stripped.
row_status() { sed -n "s/^| \[$1\]([^)]*) *| *[^|]*| *\([^|]*[^| ]\) *|.*/\1/p" "$index" | head -1; }
first_word() { printf '%s' "$1" | tr -d '*' | awk '{print tolower($1)}'; }

for file in "$adr"/0*.md; do
  n=$(basename "$file" | cut -d- -f1)
  if ! grep -q "^| \[$n\](" "$index"; then
    fail "ADR $n has no row in the index"
    continue
  fi
  file_status=$(sed -n 's/^\*\*Status:\*\* *//p' "$file" | head -1 | sed 's/ · \*\*Date.*//')
  if [ -z "$file_status" ]; then
    fail "ADR $n has no **Status:** line"
    continue
  fi
  want=$(first_word "$file_status")
  got=$(first_word "$(row_status "$n")")
  [ -n "$got" ] || { fail "ADR $n: the index row has no status cell"; continue; }
  if [ "$want" != "$got" ]; then
    # The legend writes the thesis row as `**antumbra**`, whose file says
    # `Accepted - **antumbra / central thesis**`. That is the legend, not drift.
    case "$got" in
      antumbra) printf '%s' "$file_status" | grep -qi antumbra \
        || fail "ADR $n: the index says '$got', the file says '$want'" ;;
      *) fail "ADR $n: the index says '$got', the file says '$want'" ;;
    esac
  fi
done

# Every row must point at a file that is really there.
grep -oE '^\| \[[0-9]{4}\]\([^)]+\)' "$index" | sed 's/.*(//; s/)$//' | while read -r path; do
  [ -f "$adr/$path" ] || echo "  FAIL  the index links $path, which does not exist"
done | tee /tmp/adr-index-dangling.$$
[ ! -s /tmp/adr-index-dangling.$$ ] || failed=1
rm -f /tmp/adr-index-dangling.$$

if [ "$failed" = 0 ]; then
  echo "the ADR index agrees with $(ls "$adr"/0*.md | wc -l | tr -d ' ') records"
else
  echo "the ADR index and the records disagree; fix whichever is behind"
  exit 1
fi
